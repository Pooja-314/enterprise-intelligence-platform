"""Agent Orchestrator for Multi-Step AI Agent Workflows.
Implements ReAct reasoning loop, Pydantic argument validation, privilege boundary defense,
and comprehensive execution tracing.
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.agent.base import BaseTool, ToolExecutionError
from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.agent.tools.doc_analysis import DocumentAnalysisTool
from app.agent.tools.kb_search import KnowledgeBaseSearchTool
from app.agent.tools.report_gen import StructuredReportGenerationTool
from app.core.security import PromptInjectionDefender
from app.llm.fallback_router import LLMRouter, llm_router

logger = logging.getLogger(__name__)


# Token & Context Budget Limits to guarantee compliance with Groq 8,000 TPM limit
MAX_OBSERVATION_CHARS = 1200       # Compact observation limit per tool execution (~300 tokens)
MAX_SCRATCHPAD_CHARS = 4500        # Maximum cumulative scratchpad context (~1,125 tokens)
MAX_CONVERSATION_CONTEXT_CHARS = 1200  # Maximum conversational history context (~300 tokens)
MAX_TOTAL_PROMPT_CHARS = 9000      # Safety ceiling on total prompt characters (~2,250 tokens)
STEP_MAX_TOKENS = 768              # Max output tokens per agent step (reduced from 1024)


@dataclass
class AgentTraceStep:
    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    latency_ms: float
    status: str = "SUCCESS"  # SUCCESS, FAILED, RETRY
    error: Optional[str] = None


@dataclass
class AgentExecutionResult:
    query: str
    final_answer: str
    steps: List[AgentTraceStep]
    tools_used: List[str]
    citations: List[str]
    total_latency_ms: float
    tokens: Dict[str, int] = field(default_factory=dict)
    status: str = "COMPLETED"  # COMPLETED, MAX_STEPS_REACHED, FAILED
    error: Optional[str] = None


class AgentOrchestrator:
    """Multi-step ReAct agent orchestrator with strict privilege isolation and tool validation."""

    def __init__(
        self,
        router: Optional[LLMRouter] = None,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 5,
    ):
        self.router = router or llm_router
        self.max_iterations = max_iterations

        # Register default tools if none provided
        default_tools = [
            KnowledgeBaseSearchTool(),
            DocumentAnalysisTool(router=self.router),
            StructuredReportGenerationTool(router=self.router),
        ]
        registered_list = tools if tools is not None else default_tools
        self.tools: Dict[str, BaseTool] = {t.name: t for t in registered_list}

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns JSON schema definitions of all registered tools."""
        return [tool.get_schema() for tool in self.tools.values()]

    def run(
        self,
        db: Session,
        query: str,
        user_id: Optional[int] = None,
        max_steps: Optional[int] = None,
        conversation_id: Optional[str] = None,
        conversation_context: Optional[str] = None,
    ) -> AgentExecutionResult:
        """Executes the multi-step agent reasoning loop."""
        start_time = time.perf_counter()
        limit = max_steps or self.max_iterations

        # 1. Inspect input query for direct prompt injection
        sanitized_query = PromptInjectionDefender.validate_or_raise(query)

        # 2. Retrieve conversational context if conversation_id provided
        if conversation_id and not conversation_context:
            try:
                from app.memory.manager import conversation_memory
                ctx = conversation_memory.get_relevant_context(
                    db=db,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                if ctx and ctx.formatted_history:
                    conversation_context = ctx.formatted_history
            except Exception as e:
                logger.warning(f"Failed to load conversational context for {conversation_id}: {e}")

        steps: List[AgentTraceStep] = []
        tools_called: List[str] = []
        total_tokens = {"prompt": 0, "completion": 0, "total": 0}

        # 3. Build tool descriptions for system prompt
        tools_desc = "\n\n".join([
            f"Tool: {t.name}\nDescription: {t.description}\nParameters: {json.dumps(t.args_schema.model_json_schema()['properties'], indent=2)}"
            for t in self.tools.values()
        ])
        system_prompt = AGENT_SYSTEM_PROMPT.format(tools_description=tools_desc)

        trace_scratchpad = ""

        for step_idx in range(1, limit + 1):
            step_start = time.perf_counter()
            context_block = ""
            if conversation_context:
                trimmed_history = conversation_context[:MAX_CONVERSATION_CONTEXT_CHARS]
                if len(conversation_context) > MAX_CONVERSATION_CONTEXT_CHARS:
                    trimmed_history += "\n... [Earlier conversational history truncated for token budget]"
                context_block = f"\nRelevant Conversational History:\n{trimmed_history}\n"

            # Enforce sliding-window compaction on scratchpad to guarantee staying under TPM limit
            budgeted_scratchpad = self._prune_scratchpad(trace_scratchpad, max_chars=MAX_SCRATCHPAD_CHARS)

            current_prompt = f"""User Request:
{sanitized_query}
{context_block}
Current Progress / Scratchpad:
{budgeted_scratchpad if budgeted_scratchpad else '(No tools executed yet. Decide on the first step.)'}

What is your next Thought and Action? Respond strictly with the required JSON structure:"""

            # 3. LLM Step Inference (conservative max_tokens to preserve 8,000 TPM limit)
            try:
                llm_response = self.router.generate(
                    prompt=current_prompt,
                    system_prompt=system_prompt,
                    temperature=0.1,
                    max_tokens=STEP_MAX_TOKENS,
                )
                total_tokens["prompt"] += llm_response.prompt_tokens
                total_tokens["completion"] += llm_response.completion_tokens
                total_tokens["total"] += llm_response.total_tokens

                parsed_action = self._parse_json_action(llm_response.content)
            except Exception as e:
                logger.error(f"Agent step {step_idx} failed during LLM inference: {e}")
                latency = (time.perf_counter() - step_start) * 1000
                steps.append(
                    AgentTraceStep(
                        step_number=step_idx,
                        thought="LLM inference error occurred.",
                        action="error",
                        action_input={},
                        observation=f"Inference error: {str(e)}",
                        latency_ms=round(latency, 2),
                        status="FAILED",
                        error=str(e),
                    )
                )
                break

            thought = parsed_action.get("thought", "")
            action = parsed_action.get("action", "")
            action_input = parsed_action.get("action_input", {})

            # 4. Check for Final Answer
            if action == "final_answer":
                final_answer = (
                    parsed_action.get("final_answer")
                    or action_input.get("final_answer")
                    or action_input.get("answer")
                    or ""
                )
                citations = (
                    parsed_action.get("citations")
                    or action_input.get("citations")
                    or []
                )
                latency = (time.perf_counter() - step_start) * 1000

                steps.append(
                    AgentTraceStep(
                        step_number=step_idx,
                        thought=thought,
                        action="final_answer",
                        action_input={"citations_count": len(citations)},
                        observation="Final answer synthesized successfully.",
                        latency_ms=round(latency, 2),
                        status="SUCCESS",
                    )
                )

                total_latency = (time.perf_counter() - start_time) * 1000
                return AgentExecutionResult(
                    query=sanitized_query,
                    final_answer=final_answer,
                    steps=steps,
                    tools_used=list(dict.fromkeys(tools_called)),
                    citations=citations,
                    total_latency_ms=round(total_latency, 2),
                    tokens=total_tokens,
                    status="COMPLETED",
                )

            # 5. Check Tool Allowlist
            if action not in self.tools:
                obs = (
                    f"Error: Tool '{action}' is not in the allowlist. "
                    f"Registered tools are: {list(self.tools.keys())}. "
                    f"Please select a valid tool or respond with 'final_answer'."
                )
                latency = (time.perf_counter() - step_start) * 1000
                steps.append(
                    AgentTraceStep(
                        step_number=step_idx,
                        thought=thought,
                        action=action,
                        action_input=action_input,
                        observation=obs,
                        latency_ms=round(latency, 2),
                        status="RETRY",
                        error="Tool not in allowlist",
                    )
                )
                trace_scratchpad += (
                    f"\nStep {step_idx}:\nThought: {thought}\nAction: {action}\n"
                    f"Observation: <tool_observation name=\"{action}\">\n{obs}\n</tool_observation>\n"
                )
                continue

            # 6. Execute Selected Tool with Argument Validation
            tool = self.tools[action]
            tools_called.append(action)

            try:
                tool_output = tool.run(db=db, user_id=user_id, **action_input)
                # Format observation compactly while preserving citations
                obs = self._compact_tool_observation(action=action, tool_output=tool_output)
                step_status = "SUCCESS"
                step_error = None
            except ToolExecutionError as tee:
                obs = f"Validation Error on '{action}': {tee.message}. Please adjust the input parameters and retry."
                step_status = "RETRY"
                step_error = tee.message
            except Exception as exc:
                obs = f"Tool Execution Failure on '{action}': {str(exc)}."
                step_status = "FAILED"
                step_error = str(exc)

            latency = (time.perf_counter() - step_start) * 1000

            # Guarded observation format (passive untrusted data encapsulation)
            guarded_obs = f"<tool_observation name=\"{action}\">\n{obs}\n</tool_observation>"

            steps.append(
                AgentTraceStep(
                    step_number=step_idx,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=obs[:500] + ("..." if len(obs) > 500 else ""),
                    latency_ms=round(latency, 2),
                    status=step_status,
                    error=step_error,
                )
            )

            trace_scratchpad += (
                f"\nStep {step_idx}:\nThought: {thought}\nAction: {action}\n"
                f"Action Input: {json.dumps(action_input)}\nObservation: {guarded_obs}\n"
            )

        # 7. Check if execution failed on a terminal step error
        total_latency = (time.perf_counter() - start_time) * 1000
        if steps and steps[-1].status == "FAILED":
            err_msg = steps[-1].error or "An error occurred during agent execution."
            actionable_err = self._format_actionable_error(err_msg)

            # Preserve citations from any successful prior tool steps
            recovered_citations: List[str] = []
            for st in steps:
                if st.status == "SUCCESS" and st.observation:
                    cits = re.findall(r"\[Doc:[^\]]+\]", st.observation)
                    for c in cits:
                        if c not in recovered_citations:
                            recovered_citations.append(c)

            fallback_answer = f"Agent execution encountered an error: {actionable_err}"
            if recovered_citations:
                fallback_answer += (
                    f"\n\nPrior tool executions collected evidence and verified citations before the error. "
                    f"Retained citations: {', '.join(recovered_citations)}"
                )

            return AgentExecutionResult(
                query=sanitized_query,
                final_answer=fallback_answer,
                steps=steps,
                tools_used=list(dict.fromkeys(tools_called)),
                citations=recovered_citations,
                total_latency_ms=round(total_latency, 2),
                tokens=total_tokens,
                status="FAILED",
                error=actionable_err,
            )

        # 8. Genuine Max Steps Boundary Reached (no failure, but budget exhausted)
        fallback_answer = (
            f"The agent reached the maximum reasoning iterations ({limit}) while analyzing the request. "
            f"Evidence collected across {len(tools_called)} tool executions is recorded in the trace."
        )

        return AgentExecutionResult(
            query=sanitized_query,
            final_answer=fallback_answer,
            steps=steps,
            tools_used=list(dict.fromkeys(tools_called)),
            citations=[],
            total_latency_ms=round(total_latency, 2),
            tokens=total_tokens,
            status="MAX_STEPS_REACHED",
        )

    def _parse_json_action(self, text_content: str) -> Dict[str, Any]:
        """Robustly parses agent JSON actions, handling markdown code fences."""
        cleaned = text_content.strip()

        # Extract content between ```json ... ``` if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if json_match:
            cleaned = json_match.group(1).strip()

        # Fallback to looking for the outermost { ... }
        if not (cleaned.startswith("{") and cleaned.endswith("}")):
            brace_match = re.search(r"(\{[\s\S]*\})", cleaned)
            if brace_match:
                cleaned = brace_match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                # Normalize OpenAI/Groq tool-call format {"name": ..., "arguments": ...}
                if "name" in parsed and "action" not in parsed:
                    args = parsed.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    parsed["action"] = parsed["name"]
                    parsed["action_input"] = args if isinstance(args, dict) else {"query": str(args)}
                    if "thought" not in parsed:
                        parsed["thought"] = f"Decided to invoke tool: {parsed['name']}"

                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: if model produced unstructured answer, treat as final answer
        return {
            "thought": "Synthesizing direct response",
            "action": "final_answer",
            "final_answer": text_content.strip(),
            "citations": [],
        }

    def _compact_tool_observation(self, action: str, tool_output: Any, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
        """Compacts intermediate tool outputs into token-efficient observations.
        Preserves all citation identifiers (document_id, title, page_number, chunk_id)
        while pruning massive full-text dumps and redundant scoring metrics to respect 8,000 TPM limits.
        """
        if not isinstance(tool_output, dict):
            text = str(tool_output)
            return text[:max_chars] + ("... [truncated]" if len(text) > max_chars else "")

        if action == "knowledge_base_search":
            results = tool_output.get("results", [])
            total_retrieved = tool_output.get("total_retrieved", len(results))
            compact_passages = []

            # Retain up to 3 most relevant passages to prevent prompt bloat
            for item in results[:3]:
                if not isinstance(item, dict):
                    continue
                citation = item.get("citation", "")
                chunk_id = item.get("chunk_id", "")
                doc_title = item.get("document_title", "")
                doc_id = item.get("document_id", "")
                meta = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
                page = meta.get("page_number")
                row = meta.get("row_number")

                # Keep concise content snippet (up to 200 characters)
                raw_content = str(item.get("content", "")).strip()
                clean_snippet = re.sub(r"\s+", " ", raw_content)[:200]
                if len(raw_content) > 200:
                    clean_snippet += "..."

                passage_entry = {
                    "citation": citation,
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "title": doc_title,
                    "snippet": clean_snippet,
                }
                if page is not None:
                    passage_entry["page"] = page
                if row is not None:
                    passage_entry["row"] = row
                compact_passages.append(passage_entry)

            compact_dict = {
                "total_retrieved": total_retrieved,
                "showing": len(compact_passages),
                "passages": compact_passages,
            }
            return json.dumps(compact_dict, separators=(",", ":"))

        elif action == "document_analysis":
            doc_id = tool_output.get("document_id", "")
            title = tool_output.get("title", "")
            total_chunks = tool_output.get("total_chunks", 0)
            focus = tool_output.get("focus_area")
            analysis_text = str(tool_output.get("analysis", "")).strip()
            summary = tool_output.get("summary")

            # Extract key findings / summary text up to 700 characters
            finding_text = summary or analysis_text
            clean_finding = re.sub(r"\s+", " ", finding_text)[:700]
            if len(finding_text) > 700:
                clean_finding += "..."

            # Retain top chunk citations
            citations = []
            for ref in tool_output.get("available_chunk_references", [])[:5]:
                if isinstance(ref, dict):
                    c_id = ref.get("chunk_id", "")
                    p_num = ref.get("page_number", 1)
                    citations.append(f"[Doc: {title}, Page: {p_num}, Chunk: {c_id}]")
                elif isinstance(ref, str):
                    citations.append(ref)

            compact_dict = {
                "document_id": doc_id,
                "title": title,
                "total_chunks": total_chunks,
                "summary": clean_finding,
                "citations": citations,
            }
            if focus:
                compact_dict["focus"] = focus
            return json.dumps(compact_dict, separators=(",", ":"))

        elif action == "generate_report":
            topic = tool_output.get("topic", "")
            fmt = tool_output.get("report_format", "")
            report_md = str(tool_output.get("report_markdown", "")).strip()
            citations = tool_output.get("citations", [])

            clean_excerpt = re.sub(r"\s+", " ", report_md)[:600]
            if len(report_md) > 600:
                clean_excerpt += "..."

            compact_dict = {
                "topic": topic,
                "report_format": fmt,
                "report_excerpt": clean_excerpt,
                "citations": citations[:6],
            }
            return json.dumps(compact_dict, separators=(",", ":"))

        else:
            clean_str = json.dumps(tool_output, separators=(",", ":"), default=str)
            if len(clean_str) > max_chars:
                return clean_str[:max_chars] + "... [truncated]"
            return clean_str

    def _prune_scratchpad(self, scratchpad: str, max_chars: int = MAX_SCRATCHPAD_CHARS) -> str:
        """Applies a sliding context window to scratchpad: compresses older observations
        while keeping recent steps and all citations intact to prevent prompt token explosions.
        """
        if len(scratchpad) <= max_chars:
            return scratchpad

        step_blocks = re.split(r"(?=\nStep \d+:)", scratchpad)
        if len(step_blocks) <= 2:
            return scratchpad[-max_chars:]

        pruned_blocks = []
        for block in step_blocks[:-1]:
            obs_match = re.search(r"<tool_observation name=\"([^\"]+)\">([\s\S]*?)</tool_observation>", block)
            if obs_match:
                tool_name = obs_match.group(1)
                inner = obs_match.group(2)
                citations_found = re.findall(r"\[Doc:[^\]]+\]", inner)
                cit_summary = f" (Citations: {', '.join(citations_found[:3])})" if citations_found else ""
                short_obs = f"<tool_observation name=\"{tool_name}\">\n[Prior observation summarized: evidence captured{cit_summary}]\n</tool_observation>"
                block = block[:obs_match.start()] + short_obs + block[obs_match.end():]
            pruned_blocks.append(block)

        pruned_blocks.append(step_blocks[-1])
        pruned = "".join(pruned_blocks)
        if len(pruned) > max_chars:
            pruned = pruned[-max_chars:]
        return pruned

    @staticmethod
    def _format_actionable_error(error_str: str) -> str:
        """Translates technical provider errors (429, 503, TPM) into actionable guidance."""
        err_lower = error_str.lower()
        if "tpm" in err_lower or "tokens per minute" in err_lower or "rate limit" in err_lower or "429" in err_lower:
            if "503" in err_lower or "unavailable" in err_lower:
                return "Groq rate limit reached (8,000 TPM limit) and Gemini fallback is temporarily unavailable (HTTP 503). Please retry in a few moments."
            return "Groq rate limit reached (8,000 TPM limit). Please wait a moment before sending additional queries."
        if "503" in err_lower or "unavailable" in err_lower:
            return "LLM service is temporarily unavailable (HTTP 503). Please retry your request shortly."
        return error_str


# Global orchestrator singleton instance
agent_orchestrator = AgentOrchestrator()
