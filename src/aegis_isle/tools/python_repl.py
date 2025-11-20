"""
Python REPL Tool for safe code execution.
Provides sandboxed Python code execution with security restrictions.
"""

import ast
import builtins
import io
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseTool, ToolConfig, ToolError, ToolResult, ToolStatus
from ..core.logging import logger


class RestrictedPython:
    """Restricted Python execution environment."""

    # Allowed built-in functions for safe execution
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'callable', 'chr', 'classmethod', 'complex', 'dict', 'dir', 'divmod',
        'enumerate', 'filter', 'float', 'format', 'frozenset', 'getattr',
        'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass',
        'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'oct', 'ord',
        'pow', 'print', 'range', 'repr', 'reversed', 'round', 'set',
        'setattr', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'vars',
        'zip', 'help'
    }

    # Dangerous modules and attributes to block
    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'glob', 'tempfile', 'pickle',
        'marshal', 'shelve', 'dbm', 'sqlite3', 'socket', 'urllib', 'http',
        'ftplib', 'smtplib', 'imaplib', 'poplib', 'telnetlib', 'webbrowser',
        'threading', 'multiprocessing', '_thread', 'thread', 'asyncio',
        'ctypes', 'gc', 'weakref', '__builtin__', '__builtins__', 'builtins',
        'importlib', 'imp', 'code', 'codeop', 'compile', 'eval', 'exec',
        'memoryview', 'open', 'file', 'input', 'raw_input'
    }

    BLOCKED_ATTRIBUTES = {
        '__import__', '__loader__', '__package__', '__spec__', '__file__',
        '__cached__', '__doc__', '__name__', '__dict__', '__class__',
        '__bases__', '__mro__', '__subclasses__', '__module__', '__globals__',
        '__locals__', '__code__', '__closure__', '__defaults__', '__kwdefaults__'
    }

    def __init__(self, allowed_imports: Optional[Set[str]] = None):
        """Initialize restricted Python environment.

        Args:
            allowed_imports: Set of allowed module names for import
        """
        self.allowed_imports = allowed_imports or {
            'math', 'statistics', 'random', 'datetime', 'time', 'json', 're',
            'collections', 'itertools', 'functools', 'operator', 'string',
            'decimal', 'fractions', 'uuid', 'hashlib', 'base64', 'binascii',
            'calendar', 'heapq', 'bisect', 'array', 'copy', 'pprint',
            # Data science libraries (if available)
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'sklearn'
        }

    def is_safe_node(self, node: ast.AST) -> bool:
        """Check if an AST node is safe to execute.

        Args:
            node: AST node to check

        Returns:
            True if node is safe, False otherwise
        """
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] not in self.allowed_imports:
                    return False

        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] not in self.allowed_imports:
                return False

        elif isinstance(node, ast.Attribute):
            if node.attr in self.BLOCKED_ATTRIBUTES:
                return False

        elif isinstance(node, ast.Name):
            if node.id in self.BLOCKED_MODULES:
                return False

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Allow function definitions but check their contents
            for child in ast.walk(node):
                if not self.is_safe_node(child):
                    return False

        elif isinstance(node, ast.ClassDef):
            # Allow class definitions but check their contents
            for child in ast.walk(node):
                if not self.is_safe_node(child):
                    return False

        return True

    def validate_code(self, code: str) -> None:
        """Validate Python code for safety.

        Args:
            code: Python code to validate

        Raises:
            ToolError: If code contains unsafe constructs
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ToolError(f"Syntax error in code: {e}", "python_repl")

        for node in ast.walk(tree):
            if not self.is_safe_node(node):
                node_type = type(node).__name__
                if hasattr(node, 'name'):
                    detail = f"node '{node.name}'"
                elif hasattr(node, 'id'):
                    detail = f"identifier '{node.id}'"
                elif hasattr(node, 'attr'):
                    detail = f"attribute '{node.attr}'"
                else:
                    detail = f"node type {node_type}"

                raise ToolError(
                    f"Unsafe code detected: {detail}",
                    "python_repl",
                    {"node_type": node_type, "line": getattr(node, 'lineno', None)}
                )

    def create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe global namespace for code execution.

        Returns:
            Dictionary with safe global variables
        """
        safe_globals = {
            '__builtins__': {
                name: getattr(builtins, name)
                for name in self.SAFE_BUILTINS
                if hasattr(builtins, name)
            }
        }

        # Add commonly used modules if they're in allowed imports
        try:
            if 'math' in self.allowed_imports:
                import math
                safe_globals['math'] = math

            if 'random' in self.allowed_imports:
                import random
                safe_globals['random'] = random

            if 'datetime' in self.allowed_imports:
                import datetime
                safe_globals['datetime'] = datetime

            if 'json' in self.allowed_imports:
                import json
                safe_globals['json'] = json

            if 're' in self.allowed_imports:
                import re
                safe_globals['re'] = re

            # Data science libraries (optional)
            if 'numpy' in self.allowed_imports:
                try:
                    import numpy as np
                    safe_globals['np'] = np
                    safe_globals['numpy'] = np
                except ImportError:
                    pass

            if 'pandas' in self.allowed_imports:
                try:
                    import pandas as pd
                    safe_globals['pd'] = pd
                    safe_globals['pandas'] = pd
                except ImportError:
                    pass

        except Exception as e:
            logger.warning(f"Failed to import some allowed modules: {e}")

        return safe_globals


class PythonREPLTool(BaseTool):
    """Python REPL tool for safe code execution.

    Provides sandboxed execution of Python code with security restrictions
    to prevent harmful operations while allowing data analysis and computation.
    """

    def __init__(
        self,
        config: Optional[ToolConfig] = None,
        max_output_length: int = 10000,
        allowed_imports: Optional[Set[str]] = None,
        enable_matplotlib: bool = True
    ):
        """Initialize Python REPL tool.

        Args:
            config: Tool configuration (uses defaults if None)
            max_output_length: Maximum length of execution output
            allowed_imports: Set of allowed module names for import
            enable_matplotlib: Whether to enable matplotlib plotting
        """
        if config is None:
            config = ToolConfig(
                name="python_repl",
                description="Execute Python code in a secure sandbox environment",
                timeout=30,
                max_retries=1
            )

        super().__init__(config)

        self.max_output_length = max_output_length
        self.enable_matplotlib = enable_matplotlib
        self.restricted_python = RestrictedPython(allowed_imports)
        self.execution_context = {}  # Persistent variables between executions

        logger.info(f"Initialized Python REPL tool with max output: {max_output_length} chars")

    async def _execute(self, validated_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Python code safely.

        Args:
            validated_input: Python code string or dict with 'code' key

        Returns:
            Dictionary with execution results

        Raises:
            ToolError: If code execution fails or is unsafe
        """
        # Extract code from input
        if isinstance(validated_input, str):
            code = validated_input
        elif isinstance(validated_input, dict):
            code = validated_input.get('code', '')
            if not code:
                raise ToolError("No 'code' field found in input", self.name)
        else:
            raise ToolError(f"Invalid input type: {type(validated_input)}", self.name)

        code = code.strip()
        if not code:
            raise ToolError("Empty code provided", self.name)

        logger.debug(f"Executing Python code: {code[:100]}...")

        # Validate code safety
        self.restricted_python.validate_code(code)

        # Prepare execution environment
        safe_globals = self.restricted_python.create_safe_globals()

        # Merge with persistent context
        safe_globals.update(self.execution_context)

        # Capture output
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        execution_result = {
            "success": True,
            "output": "",
            "error": None,
            "variables_created": [],
            "result": None,
            "code": code
        }

        try:
            # Execute code with captured output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Split code into statements for better error tracking
                statements = self._split_code_statements(code)
                last_result = None

                for i, statement in enumerate(statements):
                    if not statement.strip():
                        continue

                    try:
                        # Check if this is an expression that should return a value
                        try:
                            # Try to compile as an expression first
                            compiled = compile(statement, f"<statement_{i}>", "eval")
                            last_result = eval(compiled, safe_globals)

                            # Print the result if it's not None (like in REPL)
                            if last_result is not None:
                                print(repr(last_result))

                        except SyntaxError:
                            # Not an expression, try as a statement
                            compiled = compile(statement, f"<statement_{i}>", "exec")
                            exec(compiled, safe_globals)

                    except Exception as e:
                        # Add statement context to error
                        error_msg = f"Error in statement {i+1}: {str(e)}"
                        raise RuntimeError(error_msg) from e

            # Collect outputs
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()

            # Combine output
            output_parts = []
            if stdout_content:
                output_parts.append(stdout_content.rstrip())
            if stderr_content:
                output_parts.append(f"STDERR: {stderr_content.rstrip()}")

            full_output = "\n".join(output_parts)

            # Limit output length
            if len(full_output) > self.max_output_length:
                full_output = full_output[:self.max_output_length] + "... (output truncated)"

            execution_result["output"] = full_output
            execution_result["result"] = last_result

            # Track new variables
            new_vars = []
            for key, value in safe_globals.items():
                if (key not in self.restricted_python.create_safe_globals() and
                    not key.startswith('_')):
                    new_vars.append(key)
                    # Update persistent context
                    self.execution_context[key] = value

            execution_result["variables_created"] = new_vars

            logger.debug(
                f"Python execution completed: {len(full_output)} chars output, "
                f"{len(new_vars)} new variables"
            )

            return execution_result

        except Exception as e:
            stderr_content = stderr_buffer.getvalue()
            error_traceback = traceback.format_exc()

            error_msg = str(e)
            if stderr_content:
                error_msg += f"\nSTDERR: {stderr_content}"

            logger.warning(f"Python execution failed: {error_msg}")

            execution_result.update({
                "success": False,
                "error": error_msg,
                "traceback": error_traceback,
                "output": stdout_buffer.getvalue()
            })

            raise ToolError(
                f"Code execution failed: {error_msg}",
                self.name,
                {"traceback": error_traceback, "code": code}
            )

        finally:
            stdout_buffer.close()
            stderr_buffer.close()

    def _split_code_statements(self, code: str) -> List[str]:
        """Split code into individual statements for execution.

        Args:
            code: Python code string

        Returns:
            List of individual statements
        """
        try:
            tree = ast.parse(code)
            statements = []

            for node in tree.body:
                # Get the source code for each top-level statement
                start_line = node.lineno - 1
                end_line = getattr(node, 'end_lineno', node.lineno) - 1

                lines = code.split('\n')
                if end_line < len(lines):
                    statement = '\n'.join(lines[start_line:end_line + 1])
                    statements.append(statement)

            return statements if statements else [code]

        except:
            # Fallback: return original code as single statement
            return [code]

    async def _validate_input(self, tool_input: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """Validate Python REPL input.

        Args:
            tool_input: Input to validate

        Returns:
            Validated input

        Raises:
            ToolError: If input is invalid
        """
        if tool_input is None:
            raise ToolError("Input cannot be None", self.name)

        if isinstance(tool_input, str):
            if not tool_input.strip():
                raise ToolError("Code cannot be empty", self.name)
            return tool_input

        if isinstance(tool_input, dict):
            if 'code' not in tool_input:
                raise ToolError("Input dictionary must contain 'code' key", self.name)

            code = tool_input.get('code', '')
            if not isinstance(code, str):
                raise ToolError("Code must be a string", self.name)

            if not code.strip():
                raise ToolError("Code cannot be empty", self.name)

            return tool_input

        raise ToolError(f"Invalid input type: {type(tool_input)}", self.name)

    def clear_context(self) -> None:
        """Clear the persistent execution context."""
        self.execution_context.clear()
        logger.debug("Cleared Python REPL execution context")

    def get_context_variables(self) -> Dict[str, str]:
        """Get current context variables with their types.

        Returns:
            Dictionary mapping variable names to their type strings
        """
        return {
            name: type(value).__name__
            for name, value in self.execution_context.items()
            if not name.startswith('_')
        }

    async def test_connection(self) -> ToolResult:
        """Test Python REPL functionality.

        Returns:
            ToolResult indicating test status
        """
        test_code = "result = 2 + 2\nprint(f'Test calculation: {result}')"

        try:
            result = await self.run(test_code)

            if result.status == ToolStatus.SUCCESS:
                output = result.content.get("output", "")
                if "Test calculation: 4" in output:
                    return ToolResult(
                        tool_name=self.name,
                        status=ToolStatus.SUCCESS,
                        content="Python REPL test passed",
                        metadata={"test": "basic_calculation"}
                    )

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                content="Python REPL test failed",
                error="Test calculation did not produce expected output",
                metadata={"test_result": result.content}
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                content="Python REPL test failed",
                error=str(e),
                metadata={"test": "basic_calculation"}
            )