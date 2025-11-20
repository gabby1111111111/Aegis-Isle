"""
Enhanced Researcher Agent with web search integration and fallback mechanisms.
Combines knowledge base retrieval with real-time web search capabilities.
"""

import json
import time
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentRole
from ...tools import SearchTool, ToolConfig, get_tool_registry
from ...rag import get_retriever, EnhancedQueryResult
from ...core.config import settings
from ...core.logging import logger


class EnhancedResearcherAgent(BaseAgent):
    """Enhanced Researcher Agent with integrated web search and RAG capabilities.

    This agent provides:
    - Knowledge base retrieval using the RAG pipeline
    - Real-time web search as fallback for recent information
    - Combined results synthesis for comprehensive answers
    - Source attribution and relevance scoring
    - Intelligent routing between local and web search
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        enable_web_search: bool = True,
        enable_rag_retrieval: bool = True,
        search_providers_config: Optional[Dict[str, Dict[str, str]]] = None,
        fallback_strategy: str = "web_on_empty"  # always_web, web_on_empty, rag_only
    ):
        """Initialize enhanced researcher agent.

        Args:
            config: Agent configuration (uses defaults if None)
            enable_web_search: Whether to enable web search capabilities
            enable_rag_retrieval: Whether to enable RAG knowledge base retrieval
            search_providers_config: Configuration for web search providers
            fallback_strategy: Strategy for combining RAG and web search
        """
        if config is None:
            config = AgentConfig(
                name="enhanced_researcher_agent",
                role=AgentRole.RESEARCHER,
                description="Enhanced research agent with web search and RAG integration",
                max_tokens=2000,
                temperature=0.7,
                tools=["web_search", "rag_retrieval"] if enable_web_search else ["rag_retrieval"]
            )

        super().__init__(config)

        self.enable_web_search = enable_web_search
        self.enable_rag_retrieval = enable_rag_retrieval
        self.fallback_strategy = fallback_strategy
        self.search_tool: Optional[SearchTool] = None
        self.rag_retriever = None
        self.llm = None

        # Initialize tools and components
        if enable_web_search:
            self._initialize_search_tool(search_providers_config or {})
        if enable_rag_retrieval:
            self._initialize_rag_retriever()
        self._initialize_llm()

        logger.info(
            f"Initialized Enhanced Researcher Agent - "
            f"Web: {enable_web_search}, RAG: {enable_rag_retrieval}, "
            f"Strategy: {fallback_strategy}"
        )

    def _initialize_search_tool(self, providers_config: Dict[str, Dict[str, str]]) -> None:
        """Initialize web search tool.

        Args:
            providers_config: Configuration for search providers
        """
        try:
            # Check if tool is already registered
            tool_registry = get_tool_registry()
            existing_tool = tool_registry.get_tool("web_search")

            if existing_tool:
                self.search_tool = existing_tool
                logger.debug("Using existing web search tool")
            else:
                # Create new search tool
                tool_config = ToolConfig(
                    name="researcher_web_search",
                    description="Web search for research and real-time information",
                    timeout=30,
                    max_retries=3
                )

                self.search_tool = SearchTool(
                    config=tool_config,
                    primary_provider="duckduckgo",  # Default to free provider
                    providers_config=providers_config,
                    enable_fallback=True,
                    max_results=10
                )

                # Register the tool
                tool_registry.register_tool(self.search_tool)
                logger.info("Created and registered researcher web search tool")

        except Exception as e:
            logger.error(f"Failed to initialize web search tool: {e}")
            self.search_tool = None

    def _initialize_rag_retriever(self) -> None:
        """Initialize RAG retriever for knowledge base search."""
        try:
            # Get enhanced retriever with reranking
            self.rag_retriever = get_retriever(
                retriever_type="enhanced_multimodal",
                vector_db_type=getattr(settings, 'vector_db_type', 'qdrant'),
                enable_query_expansion=True,
                enable_reranking=True
            )

            logger.info("Initialized RAG retriever for knowledge base search")

        except Exception as e:
            logger.error(f"Failed to initialize RAG retriever: {e}")
            self.rag_retriever = None

    def _initialize_llm(self) -> None:
        """Initialize the language model for result synthesis."""
        try:
            model_name = self.config.model_name or getattr(settings, 'default_model', 'gpt-3.5-turbo')

            self.llm = ChatOpenAI(
                model=model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=getattr(settings, 'openai_api_key', None)
            )

            logger.debug(f"Initialized LLM: {model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None

    async def process(self, message: Union[str, AgentMessage]) -> AgentResponse:
        """Process a research request with combined RAG and web search.

        Args:
            message: Research query from user

        Returns:
            AgentResponse with research results and sources
        """
        start_time = time.time()

        try:
            # Extract message content
            if isinstance(message, AgentMessage):
                content = message.content
                sender_id = message.sender_id
            else:
                content = str(message)
                sender_id = "user"

            logger.info(f"Researcher agent processing query: {content[:100]}...")

            # Add to memory
            if isinstance(message, AgentMessage):
                self.add_to_memory(message)

            # Perform research using combined strategy
            research_result = await self._conduct_research(content)

            execution_time = time.time() - start_time

            # Create response
            response = AgentResponse(
                agent_id=self.id,
                content=research_result.get("content", "Research completed"),
                success=research_result.get("success", True),
                execution_time=execution_time,
                metadata={
                    "sources": research_result.get("sources", []),
                    "rag_results": research_result.get("rag_results", 0),
                    "web_results": research_result.get("web_results", 0),
                    "strategy_used": research_result.get("strategy_used"),
                    "total_sources": research_result.get("total_sources", 0)
                }
            )

            if not research_result.get("success", True):
                response.error = research_result.get("error", "Research failed")

            logger.info(
                f"Research completed in {execution_time:.2f}s - "
                f"RAG: {research_result.get('rag_results', 0)}, "
                f"Web: {research_result.get('web_results', 0)} results"
            )

            return response

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Research agent failed: {e}")

            return AgentResponse(
                agent_id=self.id,
                content="Research failed",
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    async def _conduct_research(self, query: str) -> Dict[str, Any]:
        """Conduct research using the configured strategy.

        Args:
            query: Research query

        Returns:
            Dictionary with research results
        """
        try:
            rag_results = []
            web_results = []
            sources = []

            # Determine research strategy
            if self.fallback_strategy == "always_web":
                # Always use web search first
                if self.enable_web_search:
                    web_results = await self._search_web(query)
                if self.enable_rag_retrieval and not web_results:
                    rag_results = await self._search_rag(query)

            elif self.fallback_strategy == "rag_only":
                # Only use RAG
                if self.enable_rag_retrieval:
                    rag_results = await self._search_rag(query)

            else:  # web_on_empty (default)
                # Try RAG first, fallback to web if no results
                if self.enable_rag_retrieval:
                    rag_results = await self._search_rag(query)

                if self.enable_web_search and not rag_results:
                    logger.info("No RAG results found, falling back to web search")
                    web_results = await self._search_web(query)

            # Combine and process results
            all_results = rag_results + web_results
            if not all_results:
                return {
                    "success": False,
                    "error": "No research results found",
                    "content": "Unable to find relevant information for your query.",
                    "sources": [],
                    "rag_results": 0,
                    "web_results": 0,
                    "strategy_used": self.fallback_strategy
                }

            # Synthesize results into coherent response
            synthesized_content = await self._synthesize_results(query, all_results)

            # Extract sources
            for result in all_results:
                source_info = {
                    "title": result.get("title", "Unknown"),
                    "url": result.get("url", result.get("source", "")),
                    "snippet": result.get("snippet", result.get("content", ""))[:200],
                    "type": result.get("type", "unknown"),
                    "score": result.get("score", 0.0)
                }
                sources.append(source_info)

            return {
                "success": True,
                "content": synthesized_content,
                "sources": sources,
                "rag_results": len(rag_results),
                "web_results": len(web_results),
                "total_sources": len(sources),
                "strategy_used": self.fallback_strategy
            }

        except Exception as e:
            logger.error(f"Research conduct failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": "An error occurred during research"
            }

    async def _search_rag(self, query: str) -> List[Dict[str, Any]]:
        """Search the knowledge base using RAG.

        Args:
            query: Search query

        Returns:
            List of RAG search results
        """
        if not self.rag_retriever:
            logger.warning("RAG retriever not available")
            return []

        try:
            logger.debug(f"Searching knowledge base: {query}")

            # Perform enhanced search with reranking
            search_result: EnhancedQueryResult = await self.rag_retriever.search(
                query=query,
                limit=5
            )

            results = []
            for retrieval_result in search_result.results:
                chunk = retrieval_result.chunk

                result = {
                    "title": f"KB Document: {chunk.document_id}",
                    "content": chunk.content,
                    "source": f"knowledge_base:{chunk.document_id}",
                    "score": retrieval_result.score,
                    "type": "rag",
                    "metadata": {
                        "chunk_index": chunk.chunk_index,
                        "chunk_type": chunk.chunk_type,
                        "document_id": chunk.document_id
                    }
                }
                results.append(result)

            logger.debug(f"Found {len(results)} RAG results")
            return results

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """Search the web using the search tool.

        Args:
            query: Search query

        Returns:
            List of web search results
        """
        if not self.search_tool:
            logger.warning("Web search tool not available")
            return []

        try:
            logger.debug(f"Searching web: {query}")

            # Perform web search
            search_result = await self.search_tool.run({
                "query": query,
                "num_results": 5
            })

            if search_result.status.value != "success":
                logger.warning(f"Web search failed: {search_result.error}")
                return []

            search_data = search_result.content
            results = []

            for web_result in search_data.get("results", []):
                result = {
                    "title": web_result.get("title", ""),
                    "url": web_result.get("url", ""),
                    "snippet": web_result.get("snippet", ""),
                    "content": web_result.get("snippet", ""),  # Use snippet as content
                    "source": web_result.get("url", ""),
                    "score": 1.0 - (web_result.get("rank", 1) * 0.1),  # Simple ranking score
                    "type": "web",
                    "metadata": {
                        "rank": web_result.get("rank"),
                        "search_provider": search_data.get("provider_used")
                    }
                }
                results.append(result)

            logger.debug(f"Found {len(results)} web results")
            return results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    async def _synthesize_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Synthesize search results into a coherent response.

        Args:
            query: Original query
            results: Combined search results

        Returns:
            Synthesized response text
        """
        if not self.llm or not results:
            # Fallback: simple concatenation
            return self._simple_synthesis(query, results)

        try:
            # Prepare context from results
            context_parts = []
            for i, result in enumerate(results[:8]):  # Limit to 8 results
                source_type = result.get("type", "unknown").upper()
                content = result.get("content", result.get("snippet", ""))[:500]
                title = result.get("title", f"Source {i+1}")

                context_parts.append(f"[{source_type}] {title}:\n{content}")

            context = "\n\n".join(context_parts)

            # Create synthesis prompt
            synthesis_prompt = f"""
            Based on the following sources, provide a comprehensive answer to the user's question.
            Synthesize the information to give a complete, accurate response.

            User Question: "{query}"

            Sources:
            {context}

            Instructions:
            1. Provide a comprehensive answer using the source information
            2. Cite sources when mentioning specific facts
            3. If sources conflict, mention the different perspectives
            4. If information is insufficient, state what additional research might be needed
            5. Maintain a helpful, informative tone

            Response:
            """

            messages = [HumanMessage(content=synthesis_prompt)]
            response = await self.llm.ainvoke(messages)

            synthesized_text = response.content

            # Add source list at the end
            source_list = "\n\nSources:"
            for i, result in enumerate(results, 1):
                title = result.get("title", f"Source {i}")
                url = result.get("url", result.get("source", ""))
                if url:
                    source_list += f"\n{i}. {title} - {url}"
                else:
                    source_list += f"\n{i}. {title}"

            return synthesized_text + source_list

        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}, using simple synthesis")
            return self._simple_synthesis(query, results)

    def _simple_synthesis(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Simple fallback synthesis without LLM.

        Args:
            query: Original query
            results: Search results

        Returns:
            Simple synthesized response
        """
        if not results:
            return "No relevant information found for your query."

        response_parts = [
            f"Here's what I found regarding '{query}':",
            ""
        ]

        for i, result in enumerate(results[:5], 1):
            title = result.get("title", f"Result {i}")
            content = result.get("content", result.get("snippet", ""))
            source_type = result.get("type", "unknown").upper()

            response_parts.append(f"{i}. [{source_type}] {title}")
            response_parts.append(f"   {content[:300]}...")
            response_parts.append("")

        # Add source URLs if available
        web_sources = [r for r in results if r.get("type") == "web" and r.get("url")]
        if web_sources:
            response_parts.extend([
                "Web Sources:",
                ""
            ])
            for i, result in enumerate(web_sources, 1):
                response_parts.append(f"{i}. {result['title']} - {result['url']}")

        return "\n".join(response_parts)

    async def initialize(self) -> bool:
        """Initialize the agent resources."""
        try:
            # Initialize search tool
            if self.search_tool:
                success = await self.search_tool.initialize()
                if not success:
                    logger.warning("Search tool initialization failed")

            self.update_status("initialized")
            return True

        except Exception as e:
            logger.error(f"Researcher agent initialization failed: {e}")
            return False

    async def cleanup(self) -> bool:
        """Clean up agent resources."""
        try:
            if self.search_tool:
                await self.search_tool.cleanup()

            self.update_status("cleaned_up")
            return True

        except Exception as e:
            logger.error(f"Researcher agent cleanup failed: {e}")
            return False


# Legacy compatibility class
class ResearcherAgent(EnhancedResearcherAgent):
    """Legacy wrapper for EnhancedResearcherAgent to maintain compatibility."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize with default enhanced configuration."""
        if config is None:
            config = AgentConfig(
                name="researcher_agent",
                role=AgentRole.RESEARCHER,
                description="Research agent (legacy wrapper)",
                tools=["web_search", "rag_retrieval"]
            )

        super().__init__(
            config=config,
            enable_web_search=True,
            enable_rag_retrieval=True,
            fallback_strategy="web_on_empty"
        )
        logger.info("Initialized legacy ResearcherAgent wrapper")