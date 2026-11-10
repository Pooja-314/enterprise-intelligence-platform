"""REST API endpoints for Multi-Tool AI Agent and individual tool executions."""
import logging
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.orchestrator import AgentExecutionResult, agent_orchestrator
from app.agent.tools.doc_analysis import DocAnalysisArgs, DocumentAnalysisTool
from app.agent.tools.kb_search import KBSearchArgs, KnowledgeBaseSearchTool
from app.agent.tools.report_gen import ReportGenArgs, StructuredReportGenerationTool
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["AI Agents"])


class AgentRunRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Goal, inquiry, or instruction for the AI Agent.")
    max_steps: Optional[int] = Field(default=5, ge=1, le=10, description="Maximum reasoning and action iterations.")


class AgentStepResponse(BaseModel):
    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation_summary: str
    latency_ms: float
    status: str
    error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AgentRunResponse(BaseModel):
    query: str
    final_answer: str
    status: str
    tools_used: List[str]
    citations: List[str]
    steps: List[AgentStepResponse]
    total_latency_ms: float
    tokens: Dict[str, int]

    model_config = ConfigDict(from_attributes=True)


class ToolDefinitionResponse(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


@router.post("/run", response_model=AgentRunResponse)
def run_agent_workflow(
    request: AgentRunRequest,
    db: Session = Depends(get_db),
):
    """Executes an autonomous multi-step reasoning workflow using registered tools."""
    start_time = time.perf_counter()
    logger.info(
        f"Agent run request received: max_steps={request.max_steps}, query_length={len(request.query)}"
    )
    try:
        result: AgentExecutionResult = agent_orchestrator.run(
            db=db,
            query=request.query,
            max_steps=request.max_steps,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Agent run completed successfully: status={result.status}, "
            f"duration_ms={duration_ms:.2f}, tools_used={len(result.tools_used)}"
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            f"Agent run failed after {duration_ms:.2f}ms: {exc}"
        )
        raise

    return AgentRunResponse(
        query=result.query,
        final_answer=result.final_answer,
        status=result.status,
        tools_used=result.tools_used,
        citations=result.citations,
        steps=[
            AgentStepResponse(
                step_number=s.step_number,
                thought=s.thought,
                action=s.action,
                action_input=s.action_input,
                observation_summary=s.observation[:300] + ("..." if len(s.observation) > 300 else ""),
                latency_ms=s.latency_ms,
                status=s.status,
                error=s.error,
            )
            for s in result.steps
        ],
        total_latency_ms=result.total_latency_ms,
        tokens=result.tokens,
    )


@router.get("/tools", response_model=List[ToolDefinitionResponse])
def list_registered_tools():
    """Lists all registered tools, descriptions, and JSON parameter schemas."""
    return [
        ToolDefinitionResponse(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
        )
        for t in agent_orchestrator.get_tool_definitions()
    ]


@router.post("/tools/knowledge_base_search")
def execute_knowledge_base_search_tool(
    request: KBSearchArgs,
    db: Session = Depends(get_db),
):
    """Direct execution of the Knowledge Base Search Tool."""
    tool = KnowledgeBaseSearchTool()
    return tool.run(db=db, **request.model_dump())


@router.post("/tools/document_analysis")
def execute_document_analysis_tool(
    request: DocAnalysisArgs,
    db: Session = Depends(get_db),
):
    """Direct execution of the Document Analysis Tool."""
    tool = DocumentAnalysisTool(router=agent_orchestrator.router)
    return tool.run(db=db, **request.model_dump())


@router.post("/tools/generate_report")
def execute_report_generation_tool(
    request: ReportGenArgs,
    db: Session = Depends(get_db),
):
    """Direct execution of the Structured Report Generation Tool."""
    tool = StructuredReportGenerationTool(router=agent_orchestrator.router)
    return tool.run(db=db, **request.model_dump())
