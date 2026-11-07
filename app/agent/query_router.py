"""Query Router and Intent Classifier for AI Conversational Intelligence.
Provides reliable multi-tier routing:
1. CONVERSATIONAL: greetings, pleasantries, assistant identity, capabilities.
2. DOCUMENT_ANALYSIS: in-depth analysis of a specific document by ID.
3. MULTI_STEP_AGENT: autonomous multi-step reasoning workflows and structured report generation.
4. KNOWLEDGE_BASE_RAG: factual document-grounded question answering with citations.
"""
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.agent.orchestrator import AgentExecutionResult, AgentOrchestrator, agent_orchestrator
from app.agent.tools.doc_analysis import DocumentAnalysisTool
from app.config import settings
from app.core.security import PromptInjectionDefender
from app.llm.fallback_router import LLMRouter, llm_router
from app.rag.generator import RAGGenerator, RAGResponse, rag_generator

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    CONVERSATIONAL = "conversational"
    KNOWLEDGE_BASE_RAG = "knowledge_base_rag"
    DOCUMENT_ANALYSIS = "document_analysis"
    MULTI_STEP_AGENT = "multi_step_agent"


@dataclass
class RoutedQueryResult:
    answer: str
    citations: List[str]
    tools_used: List[str]
    intent: QueryIntent
    status: str  # COMPLETED, FAILED, MAX_STEPS_REACHED
    total_latency_ms: float
    tokens: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    verification: Optional[Dict[str, Any]] = None
    citation_details: List[Dict[str, Any]] = field(default_factory=list)


# Pattern definitions for fast, deterministic intent classification
_CONVERSATIONAL_EXACT_PATTERNS = [
    r"^(hi|hello|hey(\s+there)?|greetings|howdy|good\s+(morning|afternoon|evening|day))([\s!.,?]+(how\s+are\s+you(\s+doing)?|how\'?s\s+it\s+going|what\'?s\s+up|what\s+can\s+you\s+do|who\s+are\s+you|help(\s+me)?))?[\s!.,?]*$",
    r"^(how\s+are\s+you(\s+doing)?|how\'?s\s+it\s+going|what\'?s\s+up)[\s!.,?]*$",
    r"^(who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+your\s+name|what\s+are\s+your\s+capabilities|help(\s+me)?)[\s!.,?]*$",
    r"^(thank\s+you(\s+very\s+much)?|thanks|thx|much\s+appreciated|appreciate\s+it|good\s+job)[\s!.,?]*$",
    r"^(bye|goodbye|see\s+you|cya|have\s+a\s+good\s+day)[\s!.,?]*$",
    r"^(ok|okay|cool|awesome|great|got\s+it|understood|nice)[\s!.,?]*$",
]

_REPORT_AGENT_PATTERNS = [
    r"\b(generate|create|build|write|draft|produce)\s+(a\s+)?(structured\s+|comprehensive\s+)?report\b",
    r"\b(multi[- ]step|autonomous\s+agent|orchestrate\s+tools|investigate\s+and\s+report)\b",
]

_DOC_ANALYSIS_PATTERNS = [
    r"\b(analyze|analyse|summarize|inspect|deep\s+analysis\s+of)\s+document\s+([0-9a-f-]{36})\b",
    r"\bdocument\s+([0-9a-f-]{36})\s+(analysis|summary|inspection)\b",
]

_UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


class ConversationQueryRouter:
    """Intelligent query intent classifier and execution router for conversational AI."""

    def __init__(
        self,
        rag: Optional[RAGGenerator] = None,
        orchestrator: Optional[AgentOrchestrator] = None,
        router: Optional[LLMRouter] = None,
    ):
        self._rag = rag
        self._orchestrator = orchestrator
        self._router = router

    @property
    def rag(self) -> RAGGenerator:
        if self._rag is not None:
            return self._rag
        from app.rag.generator import rag_generator
        return rag_generator

    @rag.setter
    def rag(self, val: RAGGenerator):
        self._rag = val

    @property
    def orchestrator(self) -> AgentOrchestrator:
        if self._orchestrator is not None:
            return self._orchestrator
        from app.agent.orchestrator import agent_orchestrator
        return agent_orchestrator

    @orchestrator.setter
    def orchestrator(self, val: AgentOrchestrator):
        self._orchestrator = val

    @property
    def router(self) -> LLMRouter:
        if self._router is not None:
            return self._router
        from app.llm.fallback_router import llm_router
        return llm_router

    @router.setter
    def router(self, val: LLMRouter):
        self._router = val

    def classify_intent(self, query: str) -> QueryIntent:
        """Determines the appropriate execution pipeline for a user message."""
        cleaned = query.strip().lower()

        # 1. Conversational greetings and small talk (must match entire pattern)
        for pat in _CONVERSATIONAL_EXACT_PATTERNS:
            if re.match(pat, cleaned, re.IGNORECASE):
                return QueryIntent.CONVERSATIONAL

        # 2. Explicit multi-step agent report or orchestration
        for pat in _REPORT_AGENT_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                return QueryIntent.MULTI_STEP_AGENT

        # 3. Explicit document analysis requests with document UUID
        for pat in _DOC_ANALYSIS_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                return QueryIntent.DOCUMENT_ANALYSIS

        # Check if UUID exists along with general analysis keywords
        if re.search(_UUID_PATTERN, cleaned, re.IGNORECASE) and any(
            k in cleaned for k in ["analyze", "analyse", "summary", "summarize", "key points", "findings"]
        ):
            return QueryIntent.DOCUMENT_ANALYSIS

        # 4. Default: Knowledge Base RAG for document-grounded questions
        return QueryIntent.KNOWLEDGE_BASE_RAG

    def route_and_execute(
        self,
        db: Session,
        query: str,
        user_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        conversation_context: Optional[str] = None,
        max_steps: Optional[int] = 5,
    ) -> RoutedQueryResult:
        """Classifies user intent and executes the optimal pipeline."""
        start_time = time.perf_counter()
        sanitized_query = PromptInjectionDefender.validate_or_raise(query)
        intent = self.classify_intent(sanitized_query)

        logger.info(f"Classified conversation query intent as '{intent.value}' for query: {sanitized_query[:60]}...")

        # --- ROUTE 1: CONVERSATIONAL ---
        if intent == QueryIntent.CONVERSATIONAL:
            return self._handle_conversational(
                query=sanitized_query,
                conversation_context=conversation_context,
                start_time=start_time,
            )

        # --- ROUTE 2: DOCUMENT ANALYSIS ---
        if intent == QueryIntent.DOCUMENT_ANALYSIS:
            return self._handle_document_analysis(
                db=db,
                query=sanitized_query,
                user_id=user_id,
                start_time=start_time,
            )

        # --- ROUTE 3: MULTI-STEP AGENT ---
        if intent == QueryIntent.MULTI_STEP_AGENT:
            return self._handle_multi_step_agent(
                db=db,
                query=sanitized_query,
                user_id=user_id,
                conversation_id=conversation_id,
                conversation_context=conversation_context,
                max_steps=max_steps,
                start_time=start_time,
            )

        # --- ROUTE 4: KNOWLEDGE BASE RAG (Default) ---
        return self._handle_knowledge_base_rag(
            db=db,
            query=sanitized_query,
            start_time=start_time,
        )

    def _handle_conversational(
        self,
        query: str,
        conversation_context: Optional[str],
        start_time: float,
    ) -> RoutedQueryResult:
        """Fast direct conversational response without triggering tool loops."""
        system_prompt = (
            "You are the Enterprise Intelligence Platform AI Assistant. "
            "Respond in a helpful, professional, and friendly manner. "
            "You can answer questions grounded in uploaded documents (PDFs, CSVs, TXT, Markdown), "
            "search the vector knowledge base, analyze documents, and generate structured reports. "
            "Keep conversational responses concise (2-4 sentences)."
        )
        context_block = f"\nRelevant Conversational History:\n{conversation_context}\n" if conversation_context else ""
        prompt = f"{context_block}User message: {query}"

        try:
            res = self.router.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=256,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RoutedQueryResult(
                answer=res.content.strip(),
                citations=[],
                tools_used=[],
                intent=QueryIntent.CONVERSATIONAL,
                status="COMPLETED",
                total_latency_ms=round(latency_ms, 2),
                tokens={"prompt": res.prompt_tokens, "completion": res.completion_tokens, "total": res.total_tokens},
            )
        except Exception as e:
            logger.warning(f"Conversational generation fallback: {e}")
            latency_ms = (time.perf_counter() - start_time) * 1000
            # Friendly fallback if LLM is unreachable
            return RoutedQueryResult(
                answer="Hello! I am your Enterprise Intelligence Assistant. You can ask me questions about your uploaded documents, search the knowledge base, or request document analysis.",
                citations=[],
                tools_used=[],
                intent=QueryIntent.CONVERSATIONAL,
                status="COMPLETED",
                total_latency_ms=round(latency_ms, 2),
                tokens={},
            )

    def _handle_document_analysis(
        self,
        db: Session,
        query: str,
        user_id: Optional[int],
        start_time: float,
    ) -> RoutedQueryResult:
        """Executes Document Analysis tool for specific document IDs."""
        match = re.search(_UUID_PATTERN, query, re.IGNORECASE)
        if not match:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RoutedQueryResult(
                answer="Please specify a valid document UUID to analyze (e.g., 'Analyze document 12345678-abcd-...').",
                citations=[],
                tools_used=["document_analysis"],
                intent=QueryIntent.DOCUMENT_ANALYSIS,
                status="FAILED",
                total_latency_ms=round(latency_ms, 2),
                error="No document UUID specified in query",
            )

        doc_id = match.group(0)
        tool = DocumentAnalysisTool(router=self.router)
        try:
            tool_output = tool.run(db=db, user_id=user_id, document_id=doc_id)
            latency_ms = (time.perf_counter() - start_time) * 1000

            citations = [
                f"[Doc: {c.get('document_title', 'Doc')}, Page: {c.get('page_number', 1)}, Chunk: {c.get('chunk_id')}]"
                for c in tool_output.get("citations", [])
            ]
            citation_details = [
                {
                    "document_id": c.get("document_id"),
                    "document_title": c.get("document_title", "Doc"),
                    "chunk_id": c.get("chunk_id", "N/A"),
                    "page_number": c.get("page_number", 1),
                    "row_number": c.get("row_number"),
                    "snippet": c.get("snippet", ""),
                    "content": c.get("content") or c.get("snippet", ""),
                }
                for c in tool_output.get("citations", [])
            ]
            answer = tool_output.get("summary", "")
            key_findings = tool_output.get("key_findings", [])
            if key_findings:
                answer += "\n\n**Key Findings:**\n" + "\n".join(f"- {f}" for f in key_findings)

            return RoutedQueryResult(
                answer=answer,
                citations=citations,
                tools_used=["document_analysis"],
                intent=QueryIntent.DOCUMENT_ANALYSIS,
                status="COMPLETED",
                total_latency_ms=round(latency_ms, 2),
                citation_details=citation_details,
            )
        except Exception as e:
            logger.error(f"Document analysis tool error: {e}")
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RoutedQueryResult(
                answer=f"Document analysis encountered an error: {str(e)}",
                citations=[],
                tools_used=["document_analysis"],
                intent=QueryIntent.DOCUMENT_ANALYSIS,
                status="FAILED",
                total_latency_ms=round(latency_ms, 2),
                error=str(e),
            )

    def _handle_multi_step_agent(
        self,
        db: Session,
        query: str,
        user_id: Optional[int],
        conversation_id: Optional[str],
        conversation_context: Optional[str],
        max_steps: Optional[int],
        start_time: float,
    ) -> RoutedQueryResult:
        """Executes the Autonomous Multi-Step Agent Orchestrator."""
        try:
            agent_res: AgentExecutionResult = self.orchestrator.run(
                db=db,
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                conversation_context=conversation_context,
                max_steps=max_steps,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RoutedQueryResult(
                answer=agent_res.final_answer,
                citations=agent_res.citations,
                tools_used=agent_res.tools_used,
                intent=QueryIntent.MULTI_STEP_AGENT,
                status=agent_res.status,
                total_latency_ms=round(latency_ms, 2),
                tokens=agent_res.tokens,
                error=agent_res.error,
            )
        except Exception as e:
            logger.error(f"Agent orchestrator error: {e}")
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RoutedQueryResult(
                answer=f"Agent execution encountered an error: {str(e)}",
                citations=[],
                tools_used=[],
                intent=QueryIntent.MULTI_STEP_AGENT,
                status="FAILED",
                total_latency_ms=round(latency_ms, 2),
                error=str(e),
            )

    def _handle_knowledge_base_rag(
        self,
        db: Session,
        query: str,
        start_time: float,
    ) -> RoutedQueryResult:
        """Executes Hybrid RAG search (pgvector + FTS + RRF) with verified citations."""
        try:
            target_top_k = max(8, getattr(settings, "TOP_K_FINAL", 8))
            rag_res: RAGResponse = self.rag.query(
                db=db,
                query_text=query,
                top_k=target_top_k,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Format citations cleanly
            citations: List[str] = []
            citation_details: List[Dict[str, Any]] = []
            if not rag_res.is_insufficient_evidence:
                for c in rag_res.citations:
                    page = c.page_number
                    row = c.row_number
                    loc = f"Page: {page}" if page else (f"Row: {row}" if row else "General Section")
                    citations.append(f"[Doc: {c.document_title}, {loc}, Chunk: {c.chunk_id}]")
                    citation_details.append({
                        "document_id": c.document_id,
                        "document_title": c.document_title,
                        "chunk_id": c.chunk_id,
                        "page_number": c.page_number,
                        "row_number": c.row_number,
                        "section_header": c.section_header,
                        "snippet": c.snippet,
                        "content": getattr(c, "content", "") or c.snippet,
                    })

            verification_dict = None
            if rag_res.verification:
                verification_dict = {
                    "is_verified": rag_res.verification.is_verified,
                    "confidence_score": rag_res.verification.confidence_score,
                    "supported_claims": rag_res.verification.supported_claims,
                    "total_claims": rag_res.verification.total_claims,
                    "hallucination_detected": rag_res.verification.hallucination_detected,
                }

            return RoutedQueryResult(
                answer=rag_res.answer,
                citations=citations,
                tools_used=["knowledge_base_search"],
                intent=QueryIntent.KNOWLEDGE_BASE_RAG,
                status="COMPLETED",
                total_latency_ms=round(latency_ms, 2),
                tokens=rag_res.tokens,
                verification=verification_dict,
                citation_details=citation_details,
            )
        except Exception as e:
            logger.error(f"Knowledge base RAG query error: {e}")
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RoutedQueryResult(
                answer=f"Knowledge base search encountered an error: {str(e)}",
                citations=[],
                tools_used=["knowledge_base_search"],
                intent=QueryIntent.KNOWLEDGE_BASE_RAG,
                status="FAILED",
                total_latency_ms=round(latency_ms, 2),
                error=str(e),
            )


# Global singleton instance
conversation_router = ConversationQueryRouter()
