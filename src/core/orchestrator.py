"""
Orchestrator for MedVision-AI
===============================

Coordinates all pipeline components and manages the full asynchronous
workflow from data ingestion to output delivery.

Typical usage::

    orchestrator = Orchestrator()
    orchestrator.register_component("inference", inference_pipeline)
    await orchestrator.initialize()
    result = await orchestrator.run(input_data)
    await orchestrator.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types & configuration
# ---------------------------------------------------------------------------

class ComponentState(Enum):
    """Lifecycle state of a registered component."""

    UNINITIALISED = "uninitialised"
    INITIALISING = "initialising"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class OrchestratorState(Enum):
    """Lifecycle state of the orchestrator itself."""

    IDLE = "idle"
    INITIALISING = "initialising"
    READY = "ready"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"
    ERROR = "error"


@dataclass
class ComponentInfo:
    """Metadata about a registered component.

    Attributes
    ----------
    name : str
        Unique component identifier.
    component : Any
        The component instance.
    state : ComponentState
        Current lifecycle state.
    dependencies : Set[str]
        Names of components that must be initialised before this one.
    init_order : int
        Order of initialisation (lower = earlier).
    """

    name: str
    component: Any
    state: ComponentState = ComponentState.UNINITIALISED
    dependencies: Set[str] = field(default_factory=set)
    init_order: int = 0


@dataclass
class WorkflowStep:
    """A single step in the orchestrated workflow.

    Attributes
    ----------
    name : str
        Step identifier.
    component_name : str
        Name of the component that handles this step.
    method : str
        Method name to invoke on the component.
    input_key : Optional[str]
        Key in the shared context from which to read input.
    output_key : Optional[str]
        Key in the shared context to which the result is written.
    is_async : bool
        Whether the method is a coroutine.
    """

    name: str
    component_name: str
    method: str
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    is_async: bool = True


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator.

    Parameters
    ----------
    max_concurrent_tasks : int
        Maximum number of concurrently running workflow tasks.
    task_timeout_seconds : float
        Per-task timeout.
    retry_attempts : int
        Number of retry attempts for failed tasks.
    retry_delay_seconds : float
        Delay between retries.
    enable_health_checks : bool
        Whether to periodically check component health.
    health_check_interval_seconds : float
        Interval between health checks.
    """

    max_concurrent_tasks: int = 10
    task_timeout_seconds: float = 300.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 5.0
    enable_health_checks: bool = True
    health_check_interval_seconds: float = 60.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Asynchronous orchestrator that coordinates pipeline components.

    The orchestrator is responsible for:

    * **Component registration** – components are registered with names
      and dependency declarations.
    * **Initialisation** – components are initialised in topological order
      respecting declared dependencies.
    * **Workflow execution** – a sequence of :class:`WorkflowStep` objects
      defines the data flow from ingestion to output.
    * **Health monitoring** – optional periodic health checks on all
      components.
    * **Graceful shutdown** – components are torn down in reverse
      initialisation order.

    Parameters
    ----------
    config : OrchestratorConfig, optional
        Orchestrator configuration.

    Examples
    --------
    >>> orch = Orchestrator()
    >>> orch.register_component("loader", data_loader, dependencies=set())
    >>> orch.register_component("preprocessor", preprocessor, dependencies={"loader"})
    >>> orch.register_component("inference", inferencer, dependencies={"preprocessor"})
    >>> await orch.initialize()
    >>> result = await orch.run({"raw_data": image_bytes})
    >>> await orch.shutdown()
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self._config = config or OrchestratorConfig()
        self._components: Dict[str, ComponentInfo] = {}
        self._workflow: List[WorkflowStep] = []
        self._state = OrchestratorState.IDLE
        self._context: Dict[str, Any] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._health_task: Optional[asyncio.Task] = None
        self._init_order: List[str] = []
        self._run_count: int = 0
        self._error_count: int = 0
        self._last_run_time: Optional[float] = None
        logger.info("Orchestrator created (max_concurrent=%d)", self._config.max_concurrent_tasks)

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def register_component(
        self,
        name: str,
        component: Any,
        dependencies: Optional[Set[str]] = None,
        init_order: int = 0,
    ) -> None:
        """Register a pipeline component with the orchestrator.

        Parameters
        ----------
        name : str
            Unique component name.
        component : Any
            The component instance.
        dependencies : Set[str], optional
            Names of components that must be initialised first.
        init_order : int
            Initialisation priority (lower = initialised earlier).

        Raises
        ------
        ValueError
            If a component with the same name is already registered or
            the orchestrator is not in an IDLE/READY state.
        """
        if self._state not in (OrchestratorState.IDLE, OrchestratorState.READY):
            raise ValueError(
                f"Cannot register components when orchestrator is in {self._state.value} state."
            )
        if name in self._components:
            raise ValueError(f"Component '{name}' is already registered.")
        deps = dependencies or set()
        for dep in deps:
            if dep not in self._components and dep != name:
                logger.warning(
                    "Dependency '%s' for component '%s' is not yet registered.", dep, name
                )

        info = ComponentInfo(
            name=name,
            component=component,
            dependencies=deps,
            init_order=init_order,
        )
        self._components[name] = info
        logger.info("Registered component '%s' (dependencies=%s, order=%d)", name, deps, init_order)

    # ------------------------------------------------------------------
    # Workflow definition
    # ------------------------------------------------------------------

    def add_workflow_step(
        self,
        name: str,
        component_name: str,
        method: str,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        is_async: bool = True,
    ) -> None:
        """Append a step to the orchestrated workflow.

        Parameters
        ----------
        name : str
            Step name.
        component_name : str
            Name of the registered component.
        method : str
            Method name to call on the component.
        input_key : str, optional
            Key in the shared context for input data.
        output_key : str, optional
            Key in the shared context for the result.
        is_async : bool
            Whether the method is an async coroutine.

        Raises
        ------
        ValueError
            If the specified component is not registered.
        """
        if component_name not in self._components:
            raise ValueError(f"Component '{component_name}' is not registered.")
        step = WorkflowStep(
            name=name,
            component_name=component_name,
            method=method,
            input_key=input_key,
            output_key=output_key,
            is_async=is_async,
        )
        self._workflow.append(step)
        logger.info("Workflow step added: %s → %s.%s", name, component_name, method)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialise all registered components in dependency order.

        Raises
        ------
        RuntimeError
            If initialisation fails for any component.
        """
        if self._state not in (OrchestratorState.IDLE, OrchestratorState.READY):
            raise RuntimeError(f"Cannot initialise from state {self._state.value}.")

        self._state = OrchestratorState.INITIALISING
        logger.info("Initialising orchestrator ...")

        # Create semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_tasks)

        # Topological sort by dependencies and init_order
        self._init_order = self._topological_sort()
        logger.info("Initialisation order: %s", self._init_order)

        for name in self._init_order:
            info = self._components[name]
            info.state = ComponentState.INITIALISING
            try:
                component = info.component
                if hasattr(component, "initialize"):
                    if asyncio.iscoroutinefunction(component.initialize):
                        await component.initialize()
                    else:
                        component.initialize()
                info.state = ComponentState.READY
                logger.info("Component '%s' initialised.", name)
            except Exception as exc:
                info.state = ComponentState.ERROR
                self._state = OrchestratorState.ERROR
                logger.error("Failed to initialise component '%s': %s", name, exc)
                raise RuntimeError(f"Initialisation failed for '{name}': {exc}") from exc

        self._state = OrchestratorState.READY

        # Start health check loop
        if self._config.enable_health_checks:
            self._health_task = asyncio.create_task(self._health_check_loop())

        logger.info("Orchestrator initialisation complete.")

    async def run(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the full workflow on the given input data.

        The *input_data* dictionary is merged into the shared context
        before the workflow starts.  Each step reads from and writes to
        this shared context according to its ``input_key`` /
        ``output_key`` settings.

        Parameters
        ----------
        input_data : Dict[str, Any], optional
            Initial context data.

        Returns
        -------
        Dict[str, Any]
            The shared context after all workflow steps complete.

        Raises
        ------
        RuntimeError
            If the orchestrator is not in a READY or RUNNING state,
            or if any workflow step fails after all retry attempts.
        """
        if self._state not in (OrchestratorState.READY, OrchestratorState.RUNNING):
            raise RuntimeError(f"Cannot run workflow from state {self._state.value}.")

        self._state = OrchestratorState.RUNNING
        self._run_count += 1

        # Merge input data into context
        if input_data is not None:
            self._context.update(input_data)

        run_id = self._run_count
        logger.info("Workflow run #%d started (%d steps).", run_id, len(self._workflow))

        run_start = time.perf_counter()

        for step_idx, step in enumerate(self._workflow):
            step_start = time.perf_counter()
            component_info = self._components.get(step.component_name)
            if component_info is None:
                raise RuntimeError(f"Component '{step.component_name}' not found for step '{step.name}'.")

            component = component_info.component
            method = getattr(component, step.method, None)
            if method is None:
                raise RuntimeError(
                    f"Method '{step.method}' not found on component '{step.component_name}'."
                )

            # Update component state
            component_info.state = ComponentState.RUNNING

            # Resolve input
            step_input = self._context.get(step.input_key) if step.input_key else self._context

            # Execute step with timeout and retries
            result = await self._execute_step_with_retries(
                step, method, step_input, run_id, step_idx
            )

            # Write output to context
            if step.output_key is not None:
                self._context[step.output_key] = result

            # Update component state back to READY
            component_info.state = ComponentState.READY

            elapsed = time.perf_counter() - step_start
            logger.info(
                "Run #%d | Step %d/%d '%s' completed in %.2fs",
                run_id,
                step_idx + 1,
                len(self._workflow),
                step.name,
                elapsed,
            )

        self._state = OrchestratorState.READY
        self._last_run_time = time.perf_counter() - run_start
        logger.info("Workflow run #%d complete (%.2fs).", run_id, self._last_run_time)
        return dict(self._context)

    async def shutdown(self) -> None:
        """Gracefully shut down all components in reverse initialisation order.

        Cancels the health check loop and then invokes each component's
        ``shutdown()`` method (if it exists).
        """
        if self._state == OrchestratorState.SHUTDOWN:
            logger.warning("Orchestrator already shut down.")
            return

        self._state = OrchestratorState.SHUTTING_DOWN
        logger.info("Shutting down orchestrator ...")

        # Cancel health check
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

        # Shutdown in reverse order
        for name in reversed(self._init_order):
            info = self._components[name]
            try:
                component = info.component
                if hasattr(component, "shutdown"):
                    if asyncio.iscoroutinefunction(component.shutdown):
                        await component.shutdown()
                    else:
                        component.shutdown()
                info.state = ComponentState.SHUTDOWN
                logger.info("Component '%s' shut down.", name)
            except Exception as exc:
                logger.error("Error shutting down component '%s': %s", name, exc)
                info.state = ComponentState.ERROR

        self._state = OrchestratorState.SHUTDOWN
        logger.info("Orchestrator shut down complete.")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return the current status of the orchestrator and its components.

        Returns
        -------
        Dict[str, Any]
            Status dictionary with orchestrator state, component states,
            and run statistics.
        """
        component_states = {
            name: {
                "state": info.state.value,
                "dependencies": list(info.dependencies),
                "init_order": info.init_order,
            }
            for name, info in self._components.items()
        }
        return {
            "orchestrator_state": self._state.value,
            "components": component_states,
            "workflow_steps": len(self._workflow),
            "total_runs": self._run_count,
            "total_errors": self._error_count,
            "last_run_time_seconds": self._last_run_time,
            "context_keys": list(self._context.keys()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_step_with_retries(
        self,
        step: WorkflowStep,
        method: Callable,
        step_input: Any,
        run_id: int,
        step_idx: int,
    ) -> Any:
        """Execute a workflow step with retry logic and timeout.

        Parameters
        ----------
        step : WorkflowStep
        method : Callable
        step_input : Any
        run_id : int
        step_idx : int

        Returns
        -------
        Any
            The step result.

        Raises
        ------
        RuntimeError
            If all retry attempts fail.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._config.retry_attempts + 1):
            try:
                async with asyncio.timeout(self._config.task_timeout_seconds):
                    if step.is_async and asyncio.iscoroutinefunction(method):
                        result = await method(step_input)
                    else:
                        result = method(step_input)
                    return result
            except asyncio.TimeoutError:
                logger.warning(
                    "Run #%d | Step '%s' timed out on attempt %d/%d",
                    run_id, step.name, attempt, self._config.retry_attempts,
                )
                last_exc = TimeoutError(f"Step '{step.name}' timed out after {self._config.task_timeout_seconds}s")
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Run #%d | Step '%s' failed on attempt %d/%d: %s",
                    run_id, step.name, attempt, self._config.retry_attempts, exc,
                )

            if attempt < self._config.retry_attempts:
                await asyncio.sleep(self._config.retry_delay_seconds)

        self._error_count += 1
        self._components[step.component_name].state = ComponentState.ERROR
        raise RuntimeError(
            f"Step '{step.name}' failed after {self._config.retry_attempts} attempts: {last_exc}"
        ) from last_exc

    async def _health_check_loop(self) -> None:
        """Periodically check the health of all components."""
        logger.info("Health check loop started (interval=%.0fs)", self._config.health_check_interval_seconds)
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)
                for name, info in self._components.items():
                    if info.state in (ComponentState.READY, ComponentState.RUNNING):
                        component = info.component
                        if hasattr(component, "health_check"):
                            try:
                                if asyncio.iscoroutinefunction(component.health_check):
                                    healthy = await component.health_check()
                                else:
                                    healthy = component.health_check()
                                if not healthy:
                                    logger.warning("Health check FAILED for component '%s'.", name)
                                    info.state = ComponentState.ERROR
                            except Exception as exc:
                                logger.warning("Health check error for '%s': %s", name, exc)
                                info.state = ComponentState.ERROR
            except asyncio.CancelledError:
                logger.info("Health check loop cancelled.")
                break
            except Exception as exc:
                logger.error("Unexpected error in health check loop: %s", exc)

    def _topological_sort(self) -> List[str]:
        """Sort components by dependencies (Kahn's algorithm).

        Returns
        -------
        List[str]
            Component names in initialisation order.

        Raises
        ------
        RuntimeError
            If circular dependencies are detected.
        """
        in_degree: Dict[str, int] = {name: 0 for name in self._components}
        graph: Dict[str, List[str]] = {name: [] for name in self._components}

        for name, info in self._components.items():
            for dep in info.dependencies:
                if dep in self._components:
                    graph[dep].append(name)
                    in_degree[name] += 1

        # Start with nodes that have no incoming edges
        queue = sorted(
            [n for n, deg in in_degree.items() if deg == 0],
            key=lambda n: self._components[n].init_order,
        )
        order: List[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbour in sorted(graph[node], key=lambda n: self._components[n].init_order):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(order) != len(self._components):
            raise RuntimeError("Circular dependency detected among components.")

        return order
