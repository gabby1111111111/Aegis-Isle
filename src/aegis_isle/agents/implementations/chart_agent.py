"""
Enhanced Chart Agent with tool integration for data visualization and analysis.
Supports advanced chart generation, data processing, and analysis using Python tools.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentRole
from ...tools import PythonREPLTool, ToolConfig, get_tool_registry
from ...core.config import settings
from ...core.logging import logger


class EnhancedChartAgent(BaseAgent):
    """Enhanced Chart Agent with integrated Python REPL tool for data visualization.

    This agent can:
    - Analyze data and generate appropriate visualizations
    - Execute Python code for chart generation using matplotlib, seaborn, plotly
    - Process CSV data and create insights
    - Generate interactive charts and dashboards
    - Provide data analysis recommendations
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        enable_python_tools: bool = True,
        chart_libraries: Optional[List[str]] = None
    ):
        """Initialize enhanced chart agent.

        Args:
            config: Agent configuration (uses defaults if None)
            enable_python_tools: Whether to enable Python REPL tool
            chart_libraries: List of preferred chart libraries (matplotlib, seaborn, plotly)
        """
        if config is None:
            config = AgentConfig(
                name="enhanced_chart_agent",
                role=AgentRole.CHART_GENERATOR,
                description="Enhanced chart generation agent with Python tool integration",
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more consistent code generation
                tools=["python_repl"] if enable_python_tools else []
            )

        super().__init__(config)

        self.chart_libraries = chart_libraries or ["matplotlib", "seaborn", "plotly"]
        self.python_tool: Optional[PythonREPLTool] = None
        self.llm = None

        # Initialize tools and LLM
        if enable_python_tools:
            self._initialize_python_tool()
        self._initialize_llm()

        logger.info(
            f"Initialized Enhanced Chart Agent with libraries: {self.chart_libraries}"
        )

    def _initialize_python_tool(self) -> None:
        """Initialize Python REPL tool for code execution."""
        try:
            # Check if tool is already registered
            tool_registry = get_tool_registry()
            existing_tool = tool_registry.get_tool("python_repl")

            if existing_tool:
                self.python_tool = existing_tool
                logger.debug("Using existing Python REPL tool")
            else:
                # Create new tool with chart-specific configuration
                chart_allowed_imports = {
                    # Core libraries
                    'math', 'statistics', 'random', 'datetime', 'json', 're',
                    'collections', 'itertools', 'functools', 'operator',
                    # Data processing
                    'numpy', 'pandas', 'scipy',
                    # Visualization
                    'matplotlib', 'seaborn', 'plotly',
                    # Utility
                    'io', 'base64', 'urllib', 'textwrap'
                }

                tool_config = ToolConfig(
                    name="chart_python_repl",
                    description="Python REPL for chart generation and data analysis",
                    timeout=60,  # Longer timeout for chart generation
                    max_retries=2
                )

                self.python_tool = PythonREPLTool(
                    config=tool_config,
                    max_output_length=20000,  # Larger output for charts
                    allowed_imports=chart_allowed_imports,
                    enable_matplotlib=True
                )

                # Register the tool
                tool_registry.register_tool(self.python_tool)
                logger.info("Created and registered chart-specific Python REPL tool")

        except Exception as e:
            logger.error(f"Failed to initialize Python tool: {e}")
            self.python_tool = None

    def _initialize_llm(self) -> None:
        """Initialize the language model for chart generation logic."""
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
        """Process a chart generation request.

        Args:
            message: User request for chart generation

        Returns:
            AgentResponse with chart generation results
        """
        start_time = 0.0
        import time
        start_time = time.time()

        try:
            # Extract message content
            if isinstance(message, AgentMessage):
                content = message.content
                sender_id = message.sender_id
            else:
                content = str(message)
                sender_id = "user"

            logger.info(f"Chart agent processing request: {content[:100]}...")

            # Add to memory
            if isinstance(message, AgentMessage):
                self.add_to_memory(message)

            # Generate chart using intelligent analysis
            result = await self._generate_chart(content)

            execution_time = time.time() - start_time

            # Create response
            response = AgentResponse(
                agent_id=self.id,
                content=result.get("content", "Chart generation completed"),
                success=result.get("success", True),
                execution_time=execution_time,
                metadata={
                    "chart_type": result.get("chart_type"),
                    "libraries_used": result.get("libraries_used", []),
                    "data_points": result.get("data_points"),
                    "execution_details": result.get("execution_details", {})
                }
            )

            if not result.get("success", True):
                response.error = result.get("error", "Unknown error in chart generation")

            logger.info(f"Chart agent completed in {execution_time:.2f}s")
            return response

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Chart agent failed: {e}")

            return AgentResponse(
                agent_id=self.id,
                content="Chart generation failed",
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    async def _generate_chart(self, request: str) -> Dict[str, Any]:
        """Generate a chart based on the user request.

        Args:
            request: User's chart generation request

        Returns:
            Dictionary with generation results
        """
        try:
            # Step 1: Analyze the request to understand chart requirements
            chart_analysis = await self._analyze_chart_request(request)

            if not chart_analysis.get("success", False):
                return {
                    "success": False,
                    "error": "Failed to analyze chart request",
                    "content": "Unable to understand chart requirements"
                }

            # Step 2: Generate Python code for the chart
            python_code = self._generate_chart_code(chart_analysis)

            # Step 3: Execute the code using Python REPL tool
            if self.python_tool:
                execution_result = await self.python_tool.run(python_code)

                if execution_result.status.value == "success":
                    return {
                        "success": True,
                        "content": f"Chart generated successfully:\n\n{execution_result.content.get('output', '')}",
                        "chart_type": chart_analysis.get("chart_type"),
                        "libraries_used": chart_analysis.get("libraries", []),
                        "execution_details": {
                            "code_executed": python_code,
                            "execution_time": execution_result.execution_time,
                            "variables_created": execution_result.content.get("variables_created", [])
                        }
                    }
                else:
                    return {
                        "success": False,
                        "error": execution_result.error,
                        "content": "Chart generation failed during code execution",
                        "execution_details": {
                            "code_attempted": python_code,
                            "error_details": execution_result.content
                        }
                    }
            else:
                # Fallback: return the generated code
                return {
                    "success": True,
                    "content": f"Python code for chart generation:\n\n```python\n{python_code}\n```",
                    "chart_type": chart_analysis.get("chart_type"),
                    "execution_details": {
                        "code_generated": python_code,
                        "note": "Python tool not available - showing code only"
                    }
                }

        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": "An error occurred during chart generation"
            }

    async def _analyze_chart_request(self, request: str) -> Dict[str, Any]:
        """Analyze user request to determine chart requirements.

        Args:
            request: User's chart request

        Returns:
            Dictionary with analysis results
        """
        if not self.llm:
            # Fallback analysis without LLM
            return self._simple_chart_analysis(request)

        analysis_prompt = f"""
        Analyze this chart generation request and provide a structured response:

        Request: "{request}"

        Determine:
        1. Chart type (bar, line, scatter, pie, histogram, box, heatmap, etc.)
        2. Data source (if specified: CSV file, sample data, custom data)
        3. Required libraries (matplotlib, seaborn, plotly)
        4. Key variables and axes
        5. Any specific styling or formatting requirements

        Respond in JSON format with:
        {{
            "chart_type": "detected chart type",
            "data_source": "description of data source",
            "libraries": ["list", "of", "required", "libraries"],
            "x_axis": "x-axis variable name",
            "y_axis": "y-axis variable name",
            "title": "suggested chart title",
            "styling": "any specific styling requirements",
            "complexity": "simple|medium|complex"
        }}
        """

        try:
            messages = [HumanMessage(content=analysis_prompt)]
            response = await self.llm.ainvoke(messages)

            # Parse JSON response
            analysis_text = response.content
            # Extract JSON from response (handle markdown code blocks)
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = analysis_text[json_start:json_end]
                analysis = json.loads(json_str)
                analysis["success"] = True
                return analysis
            else:
                logger.warning("Failed to parse JSON from LLM response")
                return self._simple_chart_analysis(request)

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, falling back to simple analysis")
            return self._simple_chart_analysis(request)

    def _simple_chart_analysis(self, request: str) -> Dict[str, Any]:
        """Simple fallback chart analysis without LLM.

        Args:
            request: User request

        Returns:
            Basic analysis results
        """
        request_lower = request.lower()

        # Determine chart type based on keywords
        chart_type = "bar"  # default
        if any(word in request_lower for word in ["line", "trend", "time", "series"]):
            chart_type = "line"
        elif any(word in request_lower for word in ["scatter", "correlation", "relationship"]):
            chart_type = "scatter"
        elif any(word in request_lower for word in ["pie", "proportion", "percentage"]):
            chart_type = "pie"
        elif any(word in request_lower for word in ["histogram", "distribution", "frequency"]):
            chart_type = "histogram"
        elif any(word in request_lower for word in ["box", "boxplot", "quartile"]):
            chart_type = "box"
        elif any(word in request_lower for word in ["heatmap", "correlation matrix", "matrix"]):
            chart_type = "heatmap"

        # Determine libraries
        libraries = ["matplotlib", "numpy"]
        if chart_type in ["heatmap", "box", "violin"] or "seaborn" in request_lower:
            libraries.append("seaborn")
        if "interactive" in request_lower or "plotly" in request_lower:
            libraries = ["plotly", "numpy"]

        return {
            "success": True,
            "chart_type": chart_type,
            "data_source": "sample data (will be generated)",
            "libraries": libraries,
            "x_axis": "x",
            "y_axis": "y",
            "title": f"{chart_type.title()} Chart",
            "styling": "default",
            "complexity": "simple"
        }

    def _generate_chart_code(self, analysis: Dict[str, Any]) -> str:
        """Generate Python code for chart creation based on analysis.

        Args:
            analysis: Chart analysis results

        Returns:
            Python code string for chart generation
        """
        chart_type = analysis.get("chart_type", "bar")
        libraries = analysis.get("libraries", ["matplotlib"])
        title = analysis.get("title", "Chart")

        # Base imports
        code_parts = [
            "import numpy as np",
            "import pandas as pd"
        ]

        # Add library-specific imports
        if "matplotlib" in libraries:
            code_parts.extend([
                "import matplotlib.pyplot as plt",
                "import matplotlib.style as style",
                "plt.style.use('default')"
            ])

        if "seaborn" in libraries:
            code_parts.append("import seaborn as sns")

        if "plotly" in libraries:
            code_parts.extend([
                "import plotly.graph_objects as go",
                "import plotly.express as px"
            ])

        # Generate sample data
        code_parts.extend([
            "",
            "# Generate sample data",
            "np.random.seed(42)"
        ])

        # Generate chart-specific code
        if chart_type == "line":
            code_parts.extend(self._generate_line_chart_code(analysis))
        elif chart_type == "scatter":
            code_parts.extend(self._generate_scatter_chart_code(analysis))
        elif chart_type == "pie":
            code_parts.extend(self._generate_pie_chart_code(analysis))
        elif chart_type == "histogram":
            code_parts.extend(self._generate_histogram_code(analysis))
        elif chart_type == "box":
            code_parts.extend(self._generate_box_chart_code(analysis))
        elif chart_type == "heatmap":
            code_parts.extend(self._generate_heatmap_code(analysis))
        else:  # default to bar chart
            code_parts.extend(self._generate_bar_chart_code(analysis))

        return "\n".join(code_parts)

    def _generate_bar_chart_code(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate code for bar chart."""
        return [
            "categories = ['A', 'B', 'C', 'D', 'E']",
            "values = np.random.randint(10, 100, 5)",
            "",
            "plt.figure(figsize=(10, 6))",
            "bars = plt.bar(categories, values, color='skyblue', alpha=0.7)",
            f"plt.title('{analysis.get('title', 'Bar Chart')}')",
            f"plt.xlabel('{analysis.get('x_axis', 'Categories')}')",
            f"plt.ylabel('{analysis.get('y_axis', 'Values')}')",
            "plt.grid(axis='y', alpha=0.3)",
            "",
            "# Add value labels on bars",
            "for bar, value in zip(bars, values):",
            "    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,",
            "             f'{value}', ha='center', va='bottom')",
            "",
            "plt.tight_layout()",
            "plt.show()",
            "print(f'Generated bar chart with {len(categories)} categories')"
        ]

    def _generate_line_chart_code(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate code for line chart."""
        return [
            "x = np.linspace(0, 10, 50)",
            "y1 = np.sin(x) + np.random.normal(0, 0.1, 50)",
            "y2 = np.cos(x) + np.random.normal(0, 0.1, 50)",
            "",
            "plt.figure(figsize=(12, 6))",
            "plt.plot(x, y1, label='Series 1', marker='o', markersize=3)",
            "plt.plot(x, y2, label='Series 2', marker='s', markersize=3)",
            f"plt.title('{analysis.get('title', 'Line Chart')}')",
            f"plt.xlabel('{analysis.get('x_axis', 'X Values')}')",
            f"plt.ylabel('{analysis.get('y_axis', 'Y Values')}')",
            "plt.legend()",
            "plt.grid(True, alpha=0.3)",
            "plt.tight_layout()",
            "plt.show()",
            "print(f'Generated line chart with {len(x)} data points')"
        ]

    def _generate_scatter_chart_code(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate code for scatter chart."""
        return [
            "n_points = 100",
            "x = np.random.normal(50, 15, n_points)",
            "y = 2 * x + np.random.normal(0, 10, n_points)",
            "colors = np.random.rand(n_points)",
            "",
            "plt.figure(figsize=(10, 8))",
            "scatter = plt.scatter(x, y, c=colors, alpha=0.6, s=60, cmap='viridis')",
            f"plt.title('{analysis.get('title', 'Scatter Plot')}')",
            f"plt.xlabel('{analysis.get('x_axis', 'X Values')}')",
            f"plt.ylabel('{analysis.get('y_axis', 'Y Values')}')",
            "plt.colorbar(scatter, label='Color Scale')",
            "plt.grid(True, alpha=0.3)",
            "plt.tight_layout()",
            "plt.show()",
            "print(f'Generated scatter plot with {n_points} points')"
        ]

    def _generate_pie_chart_code(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate code for pie chart."""
        return [
            "labels = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']",
            "sizes = np.random.randint(10, 30, 5)",
            "colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'plum']",
            "explode = (0.1, 0, 0, 0, 0)  # explode first slice",
            "",
            "plt.figure(figsize=(10, 8))",
            "wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors,",
            "                                    autopct='%1.1f%%', explode=explode,",
            "                                    shadow=True, startangle=90)",
            f"plt.title('{analysis.get('title', 'Pie Chart')}')",
            "",
            "# Enhance text appearance",
            "for autotext in autotexts:",
            "    autotext.set_color('white')",
            "    autotext.set_fontweight('bold')",
            "",
            "plt.axis('equal')",
            "plt.tight_layout()",
            "plt.show()",
            "print(f'Generated pie chart with {len(labels)} categories')"
        ]

    def _generate_histogram_code(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate code for histogram."""
        return [
            "data = np.random.normal(100, 15, 1000)",
            "",
            "plt.figure(figsize=(10, 6))",
            "n, bins, patches = plt.hist(data, bins=30, color='skyblue', alpha=0.7, edgecolor='black')",
            f"plt.title('{analysis.get('title', 'Histogram')}')",
            f"plt.xlabel('{analysis.get('x_axis', 'Values')}')",
            "plt.ylabel('Frequency')",
            "plt.grid(axis='y', alpha=0.3)",
            "",
            "# Add statistics text",
            "mean_val = np.mean(data)",
            "std_val = np.std(data)",
            "plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.1f}')",
            "plt.legend()",
            "",
            "plt.tight_layout()",
            "plt.show()",
            "print(f'Generated histogram with {len(data)} data points')",
            "print(f'Mean: {mean_val:.2f}, Std: {std_val:.2f}')"
        ]

    def _generate_box_chart_code(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate code for box plot."""
        return [
            "# Generate sample data for different groups",
            "group1 = np.random.normal(20, 5, 100)",
            "group2 = np.random.normal(25, 7, 100)",
            "group3 = np.random.normal(15, 3, 100)",
            "group4 = np.random.normal(30, 8, 100)",
            "data_groups = [group1, group2, group3, group4]",
            "labels = ['Group A', 'Group B', 'Group C', 'Group D']",
            "",
            "plt.figure(figsize=(10, 6))",
            "box_plot = plt.boxplot(data_groups, labels=labels, patch_artist=True)",
            "",
            "# Color the boxes",
            "colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']",
            "for patch, color in zip(box_plot['boxes'], colors):",
            "    patch.set_facecolor(color)",
            "",
            f"plt.title('{analysis.get('title', 'Box Plot')}')",
            f"plt.xlabel('{analysis.get('x_axis', 'Groups')}')",
            f"plt.ylabel('{analysis.get('y_axis', 'Values')}')",
            "plt.grid(axis='y', alpha=0.3)",
            "plt.tight_layout()",
            "plt.show()",
            "print(f'Generated box plot for {len(labels)} groups')"
        ]

    def _generate_heatmap_code(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate code for heatmap."""
        return [
            "# Generate correlation matrix data",
            "data = np.random.randn(10, 12)",
            "correlation_matrix = np.corrcoef(data)",
            "",
            "plt.figure(figsize=(10, 8))",
            "if 'seaborn' in analysis.get('libraries', []):",
            "    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',",
            "                center=0, square=True, linewidths=0.5)",
            "else:",
            "    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')",
            "    plt.colorbar(im)",
            "",
            f"plt.title('{analysis.get('title', 'Heatmap')}')",
            "plt.tight_layout()",
            "plt.show()",
            "print(f'Generated heatmap with {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]} matrix')"
        ]

    async def initialize(self) -> bool:
        """Initialize the agent resources."""
        try:
            # Initialize Python tool if available
            if self.python_tool:
                success = await self.python_tool.initialize()
                if not success:
                    logger.warning("Python tool initialization failed")

            self.update_status("initialized")
            return True

        except Exception as e:
            logger.error(f"Chart agent initialization failed: {e}")
            return False

    async def cleanup(self) -> bool:
        """Clean up agent resources."""
        try:
            if self.python_tool:
                await self.python_tool.cleanup()

            self.update_status("cleaned_up")
            return True

        except Exception as e:
            logger.error(f"Chart agent cleanup failed: {e}")
            return False


# Legacy compatibility class
class ChartAgent(EnhancedChartAgent):
    """Legacy wrapper for EnhancedChartAgent to maintain compatibility."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize with default enhanced configuration."""
        if config is None:
            config = AgentConfig(
                name="chart_agent",
                role=AgentRole.CHART_GENERATOR,
                description="Chart generation agent (legacy wrapper)",
                tools=["python_repl"]
            )

        super().__init__(config)
        logger.info("Initialized legacy ChartAgent wrapper")