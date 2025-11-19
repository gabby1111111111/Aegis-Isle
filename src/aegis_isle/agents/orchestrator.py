"""
Agent Orchestrator - Coordinates complex workflows between multiple agents.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

from .base import AgentMessage, AgentResponse, BaseAgent
from .router import AgentRouter
from ..core.logging import logger


class WorkflowStep:
    """Represents a single step in a workflow."""

    def __init__(
        self,
        name: str,
        agent_roles: List[str],
        input_message: str,
        dependencies: Optional[List[str]] = None,
        timeout: int = 60,
        parallel: bool = False
    ):
        self.name = name
        self.agent_roles = agent_roles
        self.input_message = input_message
        self.dependencies = dependencies or []
        self.timeout = timeout
        self.parallel = parallel
        self.results: Dict[str, Any] = {}
        self.completed = False
        self.error: Optional[str] = None


class Workflow:
    """Defines a workflow with multiple steps."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_order: List[str] = []

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        self.steps[step.name] = step

    def get_executable_steps(self) -> List[WorkflowStep]:
        """Get steps that can be executed (dependencies met)."""
        executable = []

        for step_name, step in self.steps.items():
            if step.completed:
                continue

            # Check if all dependencies are completed
            dependencies_met = all(
                self.steps[dep_name].completed
                for dep_name in step.dependencies
                if dep_name in self.steps
            )

            if dependencies_met:
                executable.append(step)

        return executable

    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(step.completed for step in self.steps.values())

    def has_errors(self) -> bool:
        """Check if any step has errors."""
        return any(step.error is not None for step in self.steps.values())


class AgentOrchestrator:
    """Orchestrates complex workflows between multiple agents."""

    def __init__(self, router: AgentRouter):
        self.router = router
        self.active_workflows: Dict[str, Workflow] = {}
        self.workflow_templates: Dict[str, Workflow] = {}

    def register_workflow_template(self, workflow: Workflow) -> None:
        """Register a workflow template."""
        self.workflow_templates[workflow.name] = workflow
        logger.info(f"Registered workflow template: {workflow.name}")

    async def execute_workflow(
        self,
        workflow_name: str,
        initial_input: Union[str, Dict[str, Any]],
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a workflow."""
        if workflow_name not in self.workflow_templates:
            raise ValueError(f"Workflow template '{workflow_name}' not found")

        # Create a copy of the template for execution
        template = self.workflow_templates[workflow_name]
        workflow = self._create_workflow_instance(template, initial_input)

        workflow_id = workflow_id or f"{workflow_name}_{int(time.time())}"
        self.active_workflows[workflow_id] = workflow

        logger.info(f"Starting workflow execution: {workflow_id}")

        try:
            results = await self._execute_workflow_steps(workflow)
            logger.info(f"Workflow {workflow_id} completed successfully")
            return {
                "workflow_id": workflow_id,
                "success": True,
                "results": results,
                "execution_time": time.time()
            }

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            return {
                "workflow_id": workflow_id,
                "success": False,
                "error": str(e),
                "partial_results": {
                    step_name: step.results
                    for step_name, step in workflow.steps.items()
                    if step.completed
                }
            }

        finally:
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

    async def _execute_workflow_steps(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute all steps in a workflow."""
        results = {}
        max_iterations = 100  # Prevent infinite loops

        iteration = 0
        while not workflow.is_complete() and iteration < max_iterations:
            iteration += 1

            executable_steps = workflow.get_executable_steps()
            if not executable_steps:
                if workflow.has_errors():
                    break
                # No executable steps but workflow not complete - possible deadlock
                raise RuntimeError("Workflow execution deadlock detected")

            # Group steps by parallelization
            parallel_steps = [step for step in executable_steps if step.parallel]
            sequential_steps = [step for step in executable_steps if not step.parallel]

            # Execute parallel steps concurrently
            if parallel_steps:
                await self._execute_parallel_steps(parallel_steps)

            # Execute sequential steps one by one
            for step in sequential_steps:
                await self._execute_single_step(step)

        if iteration >= max_iterations:
            raise RuntimeError("Workflow execution exceeded maximum iterations")

        # Collect all results
        for step_name, step in workflow.steps.items():
            if step.completed:
                results[step_name] = step.results

        return results

    async def _execute_parallel_steps(self, steps: List[WorkflowStep]) -> None:
        """Execute multiple steps in parallel."""
        tasks = [self._execute_single_step(step) for step in steps]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_single_step(self, step: WorkflowStep) -> None:
        """Execute a single workflow step."""
        logger.info(f"Executing workflow step: {step.name}")
        start_time = time.time()

        try:
            # Find agents with the required roles
            target_agents = []
            for agent_id, agent in self.router.agents.items():
                if agent.role.value in step.agent_roles and agent.config.enabled:
                    target_agents.append(agent_id)

            if not target_agents:
                raise RuntimeError(f"No available agents found for roles: {step.agent_roles}")

            # Send message to agents
            message = AgentMessage(
                sender_id="orchestrator",
                content=step.input_message
            )

            # Execute with timeout
            results = await asyncio.wait_for(
                self.router.send_to_agents(message, target_agents),
                timeout=step.timeout
            )

            # Process results
            step.results = {
                "agent_results": results,
                "execution_time": time.time() - start_time,
                "target_agents": target_agents
            }

            step.completed = True
            logger.info(f"Step {step.name} completed successfully")

        except asyncio.TimeoutError:
            error_msg = f"Step {step.name} timed out after {step.timeout} seconds"
            step.error = error_msg
            logger.error(error_msg)

        except Exception as e:
            error_msg = f"Step {step.name} failed: {e}"
            step.error = error_msg
            logger.error(error_msg)

    def _create_workflow_instance(
        self,
        template: Workflow,
        initial_input: Union[str, Dict[str, Any]]
    ) -> Workflow:
        """Create a workflow instance from a template."""
        instance = Workflow(template.name, template.description)

        for step_name, template_step in template.steps.items():
            # Create a copy of the step
            step = WorkflowStep(
                name=template_step.name,
                agent_roles=template_step.agent_roles.copy(),
                input_message=self._process_input_template(
                    template_step.input_message,
                    initial_input
                ),
                dependencies=template_step.dependencies.copy(),
                timeout=template_step.timeout,
                parallel=template_step.parallel
            )
            instance.add_step(step)

        return instance

    def _process_input_template(
        self,
        template: str,
        input_data: Union[str, Dict[str, Any]]
    ) -> str:
        """Process input template with actual data."""
        if isinstance(input_data, str):
            return template.replace("{input}", input_data)

        # Handle dictionary input
        if isinstance(input_data, dict):
            result = template
            for key, value in input_data.items():
                placeholder = "{" + key + "}"
                result = result.replace(placeholder, str(value))
            return result

        return template

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an active workflow."""
        if workflow_id not in self.active_workflows:
            return None

        workflow = self.active_workflows[workflow_id]
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "total_steps": len(workflow.steps),
            "completed_steps": sum(1 for step in workflow.steps.values() if step.completed),
            "failed_steps": sum(1 for step in workflow.steps.values() if step.error),
            "is_complete": workflow.is_complete(),
            "has_errors": workflow.has_errors(),
            "steps": {
                step_name: {
                    "completed": step.completed,
                    "error": step.error,
                    "dependencies": step.dependencies,
                    "parallel": step.parallel
                }
                for step_name, step in workflow.steps.items()
            }
        }

    def create_rag_workflow(self) -> Workflow:
        """Create a standard RAG workflow template."""
        workflow = Workflow("rag_query", "Standard RAG query processing workflow")

        # Step 1: Retrieve relevant documents
        retrieve_step = WorkflowStep(
            name="retrieve_documents",
            agent_roles=["retriever"],
            input_message="Retrieve relevant documents for query: {input}",
            timeout=30
        )
        workflow.add_step(retrieve_step)

        # Step 2: Research and analyze (parallel with retrieval)
        research_step = WorkflowStep(
            name="research_context",
            agent_roles=["researcher"],
            input_message="Research additional context for query: {input}",
            timeout=45,
            parallel=True
        )
        workflow.add_step(research_step)

        # Step 3: Summarize and generate response
        summarize_step = WorkflowStep(
            name="generate_response",
            agent_roles=["summarizer", "coordinator"],
            input_message="Generate comprehensive response based on retrieved documents and research",
            dependencies=["retrieve_documents", "research_context"],
            timeout=60
        )
        workflow.add_step(summarize_step)

        return workflow