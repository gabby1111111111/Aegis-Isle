"""
Search Tool for external web search integration.
Supports multiple search providers with fallback mechanisms.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import aiohttp

from .base import BaseTool, ToolConfig, ToolError, ToolResult, ToolStatus
from ..core.config import settings
from ..core.logging import logger


class SearchResult(Dict[str, Any]):
    """Search result with standardized structure."""

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source: str = "unknown",
        **kwargs
    ):
        super().__init__()
        self.update({
            "title": title,
            "url": url,
            "snippet": snippet,
            "source": source,
            **kwargs
        })

    @property
    def title(self) -> str:
        return self.get("title", "")

    @property
    def url(self) -> str:
        return self.get("url", "")

    @property
    def snippet(self) -> str:
        return self.get("snippet", "")

    @property
    def source(self) -> str:
        return self.get("source", "unknown")


class SearchProvider:
    """Base class for search providers."""

    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key

    async def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """Perform search and return results.

        Args:
            query: Search query
            num_results: Maximum number of results
            **kwargs: Additional search parameters

        Returns:
            List of search results

        Raises:
            ToolError: If search fails
        """
        raise NotImplementedError


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider (no API key required)."""

    def __init__(self):
        super().__init__("duckduckgo")
        self.base_url = "https://api.duckduckgo.com/"

    async def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """Search using DuckDuckGo API."""
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_redirect': '1',
                'no_html': '1',
                'skip_disambig': '1'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        raise ToolError(f"DuckDuckGo API returned {response.status}")

                    data = await response.json()

            results = []

            # Process instant answer
            if data.get('AbstractText'):
                results.append(SearchResult(
                    title="DuckDuckGo Instant Answer",
                    url=data.get('AbstractURL', ''),
                    snippet=data['AbstractText'],
                    source="duckduckgo_instant"
                ))

            # Process related topics
            for topic in data.get('RelatedTopics', [])[:num_results]:
                if isinstance(topic, dict) and 'Text' in topic:
                    # Extract URL from FirstURL if available
                    url = topic.get('FirstURL', '')
                    text = topic['Text']

                    # Try to extract title from text (usually before " - ")
                    title_match = text.split(' - ', 1)
                    title = title_match[0] if len(title_match) > 1 else text[:100]
                    snippet = title_match[1] if len(title_match) > 1 else text

                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="duckduckgo"
                    ))

            return results[:num_results]

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            raise ToolError(f"DuckDuckGo search failed: {str(e)}", "search")


class GoogleCustomSearchProvider(SearchProvider):
    """Google Custom Search API provider."""

    def __init__(self, api_key: str, search_engine_id: str):
        super().__init__("google", api_key)
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    async def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """Search using Google Custom Search API."""
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10),  # Google CSE max is 10
                'safe': 'medium'
            }

            # Add additional parameters
            if 'site' in kwargs:
                params['siteSearch'] = kwargs['site']

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        error_data = await response.text()
                        raise ToolError(f"Google API returned {response.status}: {error_data}")

                    data = await response.json()

            if 'error' in data:
                error_msg = data['error'].get('message', 'Unknown error')
                raise ToolError(f"Google API error: {error_msg}")

            results = []
            for item in data.get('items', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    source="google",
                    display_link=item.get('displayLink', ''),
                    formatted_url=item.get('formattedUrl', '')
                ))

            return results

        except Exception as e:
            logger.error(f"Google search failed: {e}")
            raise ToolError(f"Google search failed: {str(e)}", "search")


class BingSearchProvider(SearchProvider):
    """Bing Search API provider."""

    def __init__(self, api_key: str):
        super().__init__("bing", api_key)
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"

    async def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """Search using Bing Search API."""
        try:
            params = {
                'q': query,
                'count': min(num_results, 50),  # Bing max is 50
                'offset': 0,
                'mkt': 'en-US',
                'safesearch': 'Moderate'
            }

            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    params=params,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_data = await response.text()
                        raise ToolError(f"Bing API returned {response.status}: {error_data}")

                    data = await response.json()

            if 'error' in data:
                error_msg = data['error'].get('message', 'Unknown error')
                raise ToolError(f"Bing API error: {error_msg}")

            results = []
            for item in data.get('webPages', {}).get('value', []):
                results.append(SearchResult(
                    title=item.get('name', ''),
                    url=item.get('url', ''),
                    snippet=item.get('snippet', ''),
                    source="bing",
                    display_url=item.get('displayUrl', ''),
                    date_last_crawled=item.get('dateLastCrawled', '')
                ))

            return results

        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            raise ToolError(f"Bing search failed: {str(e)}", "search")


class SerperProvider(SearchProvider):
    """Serper.dev search API provider."""

    def __init__(self, api_key: str):
        super().__init__("serper", api_key)
        self.base_url = "https://google.serper.dev/search"

    async def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """Search using Serper API."""
        try:
            payload = {
                'q': query,
                'num': min(num_results, 100)  # Serper max is 100
            }

            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_data = await response.text()
                        raise ToolError(f"Serper API returned {response.status}: {error_data}")

                    data = await response.json()

            results = []

            # Process organic results
            for item in data.get('organic', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    source="serper",
                    position=item.get('position'),
                    domain=item.get('domain', '')
                ))

            return results

        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            raise ToolError(f"Serper search failed: {str(e)}", "search")


class SearchTool(BaseTool):
    """Web search tool with multiple provider support and fallback mechanisms."""

    def __init__(
        self,
        config: Optional[ToolConfig] = None,
        primary_provider: str = "duckduckgo",
        providers_config: Optional[Dict[str, Dict[str, str]]] = None,
        enable_fallback: bool = True,
        max_results: int = 10
    ):
        """Initialize search tool.

        Args:
            config: Tool configuration
            primary_provider: Primary search provider to use
            providers_config: Configuration for search providers
            enable_fallback: Whether to enable fallback to other providers
            max_results: Maximum number of search results
        """
        if config is None:
            config = ToolConfig(
                name="web_search",
                description="Search the web using multiple search providers",
                timeout=30,
                max_retries=2
            )

        super().__init__(config)

        self.max_results = max_results
        self.enable_fallback = enable_fallback
        self.providers: Dict[str, SearchProvider] = {}
        self.provider_order = []

        # Initialize providers
        self._initialize_providers(primary_provider, providers_config or {})

        logger.info(
            f"Initialized search tool with providers: {list(self.providers.keys())}, "
            f"primary: {primary_provider}"
        )

    def _initialize_providers(
        self,
        primary_provider: str,
        providers_config: Dict[str, Dict[str, str]]
    ) -> None:
        """Initialize search providers.

        Args:
            primary_provider: Primary provider name
            providers_config: Provider configurations
        """
        # Always include DuckDuckGo as it doesn't require API key
        self.providers["duckduckgo"] = DuckDuckGoProvider()

        # Initialize other providers based on configuration
        if "google" in providers_config:
            config = providers_config["google"]
            if "api_key" in config and "search_engine_id" in config:
                self.providers["google"] = GoogleCustomSearchProvider(
                    config["api_key"],
                    config["search_engine_id"]
                )

        if "bing" in providers_config:
            config = providers_config["bing"]
            if "api_key" in config:
                self.providers["bing"] = BingSearchProvider(config["api_key"])

        if "serper" in providers_config:
            config = providers_config["serper"]
            if "api_key" in config:
                self.providers["serper"] = SerperProvider(config["api_key"])

        # Set provider order (primary first)
        if primary_provider in self.providers:
            self.provider_order = [primary_provider]
            # Add other providers for fallback
            other_providers = [p for p in self.providers.keys() if p != primary_provider]
            self.provider_order.extend(other_providers)
        else:
            self.provider_order = list(self.providers.keys())

    async def _execute(self, validated_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Execute web search.

        Args:
            validated_input: Search query string or dict with search parameters

        Returns:
            Dictionary with search results and metadata

        Raises:
            ToolError: If all search providers fail
        """
        # Parse input
        if isinstance(validated_input, str):
            query = validated_input.strip()
            num_results = self.max_results
            provider = None
        elif isinstance(validated_input, dict):
            query = validated_input.get("query", "").strip()
            num_results = validated_input.get("num_results", self.max_results)
            provider = validated_input.get("provider")
        else:
            raise ToolError(f"Invalid input type: {type(validated_input)}", self.name)

        if not query:
            raise ToolError("Search query cannot be empty", self.name)

        num_results = min(num_results, self.max_results)

        logger.debug(f"Searching for: '{query}' with {num_results} results")

        # Determine provider order
        if provider and provider in self.providers:
            search_order = [provider]
            if self.enable_fallback:
                search_order.extend([p for p in self.provider_order if p != provider])
        else:
            search_order = self.provider_order.copy()

        results = []
        errors = []
        successful_provider = None

        # Try providers in order
        for provider_name in search_order:
            if provider_name not in self.providers:
                continue

            search_provider = self.providers[provider_name]

            try:
                logger.debug(f"Attempting search with provider: {provider_name}")

                provider_results = await search_provider.search(
                    query=query,
                    num_results=num_results
                )

                if provider_results:
                    results = provider_results
                    successful_provider = provider_name
                    logger.info(
                        f"Search successful with {provider_name}: {len(results)} results"
                    )
                    break
                else:
                    logger.debug(f"No results from provider: {provider_name}")

            except Exception as e:
                error_msg = f"{provider_name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Search failed with {provider_name}: {e}")

                if not self.enable_fallback:
                    break

        # Check if any search succeeded
        if not results and not successful_provider:
            all_errors = "; ".join(errors)
            raise ToolError(
                f"All search providers failed. Errors: {all_errors}",
                self.name,
                {"errors": errors, "query": query}
            )

        # Format results for output
        formatted_results = []
        for i, result in enumerate(results):
            formatted_result = {
                "rank": i + 1,
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "source": result.source,
            }

            # Add provider-specific fields
            for key, value in result.items():
                if key not in formatted_result:
                    formatted_result[key] = value

            formatted_results.append(formatted_result)

        return {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "provider_used": successful_provider,
            "providers_tried": search_order[:len(errors) + (1 if successful_provider else 0)],
            "errors": errors if errors else None
        }

    async def _validate_input(
        self,
        tool_input: Union[str, Dict[str, Any]]
    ) -> Union[str, Dict[str, Any]]:
        """Validate search input.

        Args:
            tool_input: Input to validate

        Returns:
            Validated input

        Raises:
            ToolError: If input is invalid
        """
        if tool_input is None:
            raise ToolError("Search input cannot be None", self.name)

        if isinstance(tool_input, str):
            query = tool_input.strip()
            if not query:
                raise ToolError("Search query cannot be empty", self.name)
            if len(query) > 1000:
                raise ToolError("Search query too long (max 1000 characters)", self.name)
            return query

        if isinstance(tool_input, dict):
            query = tool_input.get("query", "").strip()
            if not query:
                raise ToolError("Search query cannot be empty", self.name)
            if len(query) > 1000:
                raise ToolError("Search query too long (max 1000 characters)", self.name)

            # Validate num_results
            num_results = tool_input.get("num_results")
            if num_results is not None:
                if not isinstance(num_results, int) or num_results < 1:
                    raise ToolError("num_results must be a positive integer", self.name)

            # Validate provider
            provider = tool_input.get("provider")
            if provider is not None:
                if provider not in self.providers:
                    available = list(self.providers.keys())
                    raise ToolError(
                        f"Provider '{provider}' not available. Available: {available}",
                        self.name
                    )

            return tool_input

        raise ToolError(f"Invalid input type: {type(tool_input)}", self.name)

    def get_available_providers(self) -> List[str]:
        """Get list of available search providers.

        Returns:
            List of provider names
        """
        return list(self.providers.keys())

    async def test_connection(self) -> ToolResult:
        """Test search functionality.

        Returns:
            ToolResult indicating test status
        """
        test_query = "test search query"

        try:
            result = await self.run({"query": test_query, "num_results": 1})

            if result.status == ToolStatus.SUCCESS:
                content = result.content
                if content.get("total_results", 0) >= 0:  # Allow 0 results as success
                    return ToolResult(
                        tool_name=self.name,
                        status=ToolStatus.SUCCESS,
                        content="Search tool test passed",
                        metadata={
                            "test_query": test_query,
                            "provider_used": content.get("provider_used"),
                            "results_count": content.get("total_results", 0)
                        }
                    )

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                content="Search tool test failed",
                error="Test search did not return expected results",
                metadata={"test_result": result.content}
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                content="Search tool test failed",
                error=str(e),
                metadata={"test_query": test_query}
            )