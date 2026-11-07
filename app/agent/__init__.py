"""AI Agent architecture package."""
from app.agent.base import BaseTool, ToolExecutionError
from app.agent.orchestrator import AgentExecutionResult, AgentOrchestrator, AgentTraceStep, agent_orchestrator
from app.agent.query_router import ConversationQueryRouter, QueryIntent, RoutedQueryResult, conversation_router
from app.agent.tools.kb_search import KnowledgeBaseSearchTool
from app.agent.tools.doc_analysis import DocumentAnalysisTool
from app.agent.tools.report_gen import StructuredReportGenerationTool

__all__ = [
    "BaseTool",
    "ToolExecutionError",
    "AgentOrchestrator",
    "AgentExecutionResult",
    "AgentTraceStep",
    "agent_orchestrator",
    "ConversationQueryRouter",
    "QueryIntent",
    "RoutedQueryResult",
    "conversation_router",
    "KnowledgeBaseSearchTool",
    "DocumentAnalysisTool",
    "StructuredReportGenerationTool",
]
