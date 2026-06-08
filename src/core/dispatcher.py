"""
Dispatcher for MedVision-AI
=============================

Routes incoming requests to the appropriate handler based on request
type, priority, and current load.  Supports priority-based dispatching
and simple round-robin load balancing across handler instances.

Typical usage::

    dispatcher = Dispatcher()
    dispatcher.register_handler("image_analysis", image_handler, priority=10)
    dispatcher.register_handler("text_analysis", text_handler, priority=5)
    result = dispatcher.dispatch("image_analysis", input_data)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types & configuration
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    """Request priority levels (higher value = higher priority)."""

    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class HandlerInfo:
    """Metadata about a registered handler.

    Attributes
    ----------
    name : str
        Handler identifier.
    handler : Callable
        The handler callable.
    priority : int
        Default priority for requests routed to this handler.
    max_concurrency : int
        Maximum concurrent invocations allowed.
    current_load : int
        Current number of in-flight requests.
    total_requests : int
        Total number of requests dispatched to this handler.
    total_errors : int
        Total number of errors encountered.
    avg_latency_ms : float
        Running average latency in milliseconds.
    is_async : bool
        Whether the handler is a coroutine function.
    """

    name: str
    handler: Callable
    priority: int = Priority.NORMAL
    max_concurrency: int = 100
    current_load: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    is_async: bool = False


@dataclass
class DispatchRequest:
    """Encapsulates a single dispatchable request.

    Attributes
    ----------
    task_type : str
        The request category (maps to a handler name).
    payload : Any
        The request data.
    priority : int
        Override priority for this specific request.
    metadata : Dict[str, Any]
        Additional request metadata.
    request_id : Optional[str]
        Optional unique request identifier.
    timestamp : float
        Creation timestamp (epoch seconds).
    """

    task_type: str
    payload: Any = None
    priority: int = Priority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class DispatchResult:
    """Result of a dispatched request.

    Attributes
    ----------
    success : bool
        Whether the request was handled successfully.
    result : Any
        The handler's return value.
    handler_name : str
        Name of the handler that processed the request.
    latency_ms : float
        End-to-end latency in milliseconds.
    error : Optional[str]
        Error message if the request failed.
    """

    success: bool = True
    result: Any = None
    handler_name: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class DispatcherConfig:
    """Configuration for the dispatcher.

    Parameters
    ----------
    default_priority : int
        Priority used when none is specified.
    max_queue_size : int
        Maximum pending requests in the priority queue.
    request_timeout_seconds : float
        Timeout per request.
    load_balance_strategy : str
        Strategy for selecting among multiple handlers of the same type
        (``"round_robin"`` or ``"least_loaded"``).
    """

    default_priority: int = Priority.NORMAL
    max_queue_size: int = 10000
    request_timeout_seconds: float = 120.0
    load_balance_strategy: str = "round_robin"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class Dispatcher:
    """Request dispatcher with priority-based routing and load balancing.

    The dispatcher maintains a registry of named handlers.  Incoming
    requests are routed to the appropriate handler based on the
    ``task_type`` field.  When multiple handler instances share the
    same type, a configurable load-balancing strategy distributes
    requests among them.

    Parameters
    ----------
    config : DispatcherConfig, optional
        Dispatcher configuration.

    Examples
    --------
    >>> dispatcher = Dispatcher()
    >>> dispatcher.register_handler("image", process_image, priority=10)
    >>> dispatcher.register_handler("image", process_image_gpu, priority=10)
    >>> result = dispatcher.dispatch("image", img_bytes)
    """

    def __init__(self, config: Optional[DispatcherConfig] = None) -> None:
        self._config = config or DispatcherConfig()
        # Map from task_type → list of HandlerInfo
        self._handlers: Dict[str, List[HandlerInfo]] = defaultdict(list)
        self._round_robin_idx: Dict[str, int] = defaultdict(int)
        self._pending: List[DispatchRequest] = []
        self._total_dispatched: int = 0
        self._total_failed: int = 0
        logger.info(
            "Dispatcher created (strategy=%s, timeout=%.0fs)",
            self._config.load_balance_strategy,
            self._config.request_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Handler management
    # ------------------------------------------------------------------

    def register_handler(
        self,
        task_type: str,
        handler: Callable,
        priority: int = Priority.NORMAL,
        max_concurrency: int = 100,
    ) -> None:
        """Register a handler for a given task type.

        Multiple handlers can be registered under the same task type; the
        dispatcher will load-balance among them.

        Parameters
        ----------
        task_type : str
            Request type / handler group name.
        handler : Callable
            The handler function or coroutine.
        priority : int
            Default priority for this handler.
        max_concurrency : int
            Maximum concurrent requests this handler can process.

        Raises
        ------
        ValueError
            If the handler is not callable.
        """
        if not callable(handler):
            raise ValueError(f"Handler for '{task_type}' must be callable, got {type(handler)}.")

        is_async = asyncio.iscoroutinefunction(handler)
        info = HandlerInfo(
            name=task_type,
            handler=handler,
            priority=priority,
            max_concurrency=max_concurrency,
            is_async=is_async,
        )
        self._handlers[task_type].append(info)
        logger.info(
            "Handler registered: '%s' (priority=%d, async=%s, concurrency=%d)",
            task_type, priority, is_async, max_concurrency,
        )

    def get_handler(self, task_type: str, strategy: Optional[str] = None) -> Optional[HandlerInfo]:
        """Select a handler for the given task type.

        Parameters
        ----------
        task_type : str
            Request type.
        strategy : str, optional
            Override load-balancing strategy.

        Returns
        -------
        HandlerInfo or None
            The selected handler, or *None* if no handler is registered
            for the given task type.
        """
        handlers = self._handlers.get(task_type, [])
        if not handlers:
            return None
        if len(handlers) == 1:
            return handlers[0]
        return self._select_handler(task_type, handlers, strategy or self._config.load_balance_strategy)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        task_type: str,
        data: Any = None,
        priority: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Any:
        """Synchronously dispatch a request to the appropriate handler.

        Parameters
        ----------
        task_type : str
            The request category that determines the handler.
        data : Any
            Request data / payload.
        priority : int, optional
            Override priority for this request.
        metadata : Dict[str, Any], optional
            Additional metadata.
        request_id : str, optional
            Unique request identifier.

        Returns
        -------
        Any
            The result from the handler if successful, or a
            :class:`DispatchResult` on failure.
        """
        request = DispatchRequest(
            task_type=task_type,
            payload=data,
            priority=priority or self._config.default_priority,
            metadata=metadata or {},
            request_id=request_id,
        )
        dispatch_result = self._dispatch_sync(request)
        if dispatch_result.success:
            return dispatch_result.result
        return dispatch_result

    async def dispatch_async(
        self,
        task_type: str,
        data: Any = None,
        priority: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> DispatchResult:
        """Asynchronously dispatch a request to the appropriate handler.

        Parameters
        ----------
        task_type : str
        data : Any
        priority : int, optional
        metadata : Dict[str, Any], optional
        request_id : str, optional

        Returns
        -------
        DispatchResult
        """
        request = DispatchRequest(
            task_type=task_type,
            payload=data,
            priority=priority or self._config.default_priority,
            metadata=metadata or {},
            request_id=request_id,
        )
        return await self._dispatch_async(request)

    def dispatch_batch(
        self,
        requests: Sequence[DispatchRequest],
    ) -> List[DispatchResult]:
        """Dispatch a batch of requests in priority order.

        Higher-priority requests are dispatched first.

        Parameters
        ----------
        requests : Sequence[DispatchRequest]

        Returns
        -------
        List[DispatchResult]
        """
        sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)
        results: List[DispatchResult] = []
        for req in sorted_requests:
            results.append(self._dispatch_sync(req))
        return results

    # ------------------------------------------------------------------
    # Status & metrics
    # ------------------------------------------------------------------

    def get_handler_stats(self, task_type: str) -> List[Dict[str, Any]]:
        """Return statistics for all handlers registered under *task_type*.

        Parameters
        ----------
        task_type : str
            Request type.

        Returns
        -------
        List[Dict[str, Any]]
        """
        handlers = self._handlers.get(task_type, [])
        return [
            {
                "name": h.name,
                "priority": h.priority,
                "current_load": h.current_load,
                "max_concurrency": h.max_concurrency,
                "total_requests": h.total_requests,
                "total_errors": h.total_errors,
                "avg_latency_ms": round(h.avg_latency_ms, 2),
                "is_async": h.is_async,
                "utilization": round(h.current_load / max(h.max_concurrency, 1) * 100, 1),
            }
            for h in handlers
        ]

    def get_status(self) -> Dict[str, Any]:
        """Return overall dispatcher status.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "total_dispatched": self._total_dispatched,
            "total_failed": self._total_failed,
            "registered_types": list(self._handlers.keys()),
            "handlers_per_type": {k: len(v) for k, v in self._handlers.items()},
            "pending_requests": len(self._pending),
            "load_balance_strategy": self._config.load_balance_strategy,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch_sync(self, request: DispatchRequest) -> DispatchResult:
        """Execute a synchronous dispatch.

        Parameters
        ----------
        request : DispatchRequest

        Returns
        -------
        DispatchResult
        """
        handler_info = self.get_handler(request.task_type)
        if handler_info is None:
            self._total_failed += 1
            return DispatchResult(
                success=False,
                handler_name=request.task_type,
                error=f"No handler registered for '{request.task_type}'.",
            )

        if handler_info.current_load >= handler_info.max_concurrency:
            self._total_failed += 1
            return DispatchResult(
                success=False,
                handler_name=handler_info.name,
                error=f"Handler '{handler_info.name}' is at max concurrency ({handler_info.max_concurrency}).",
            )

        handler_info.current_load += 1
        handler_info.total_requests += 1
        self._total_dispatched += 1

        start = time.perf_counter()
        try:
            result = handler_info.handler(request.payload)
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._update_latency(handler_info, latency_ms)
            return DispatchResult(
                success=True,
                result=result,
                handler_name=handler_info.name,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            handler_info.total_errors += 1
            self._total_failed += 1
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._update_latency(handler_info, latency_ms)
            logger.error("Handler '%s' failed: %s", handler_info.name, exc)
            return DispatchResult(
                success=False,
                handler_name=handler_info.name,
                latency_ms=latency_ms,
                error=str(exc),
            )
        finally:
            handler_info.current_load -= 1

    async def _dispatch_async(self, request: DispatchRequest) -> DispatchResult:
        """Execute an asynchronous dispatch.

        Parameters
        ----------
        request : DispatchRequest

        Returns
        -------
        DispatchResult
        """
        handler_info = self.get_handler(request.task_type)
        if handler_info is None:
            self._total_failed += 1
            return DispatchResult(
                success=False,
                handler_name=request.task_type,
                error=f"No handler registered for '{request.task_type}'.",
            )

        if handler_info.current_load >= handler_info.max_concurrency:
            self._total_failed += 1
            return DispatchResult(
                success=False,
                handler_name=handler_info.name,
                error=f"Handler '{handler_info.name}' is at max concurrency ({handler_info.max_concurrency}).",
            )

        handler_info.current_load += 1
        handler_info.total_requests += 1
        self._total_dispatched += 1

        start = time.perf_counter()
        try:
            if handler_info.is_async:
                result = await asyncio.wait_for(
                    handler_info.handler(request.payload),
                    timeout=self._config.request_timeout_seconds,
                )
            else:
                result = handler_info.handler(request.payload)
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._update_latency(handler_info, latency_ms)
            return DispatchResult(
                success=True,
                result=result,
                handler_name=handler_info.name,
                latency_ms=latency_ms,
            )
        except asyncio.TimeoutError:
            handler_info.total_errors += 1
            self._total_failed += 1
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._update_latency(handler_info, latency_ms)
            return DispatchResult(
                success=False,
                handler_name=handler_info.name,
                latency_ms=latency_ms,
                error=f"Handler timed out after {self._config.request_timeout_seconds}s.",
            )
        except Exception as exc:
            handler_info.total_errors += 1
            self._total_failed += 1
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._update_latency(handler_info, latency_ms)
            logger.error("Async handler '%s' failed: %s", handler_info.name, exc)
            return DispatchResult(
                success=False,
                handler_name=handler_info.name,
                latency_ms=latency_ms,
                error=str(exc),
            )
        finally:
            handler_info.current_load -= 1

    def _select_handler(
        self,
        name: str,
        handlers: List[HandlerInfo],
        strategy: str,
    ) -> HandlerInfo:
        """Select a handler from multiple candidates using the given strategy.

        Parameters
        ----------
        name : str
        handlers : List[HandlerInfo]
        strategy : str

        Returns
        -------
        HandlerInfo
        """
        if strategy == "round_robin":
            idx = self._round_robin_idx[name] % len(handlers)
            self._round_robin_idx[name] = idx + 1
            return handlers[idx]
        elif strategy == "least_loaded":
            return min(handlers, key=lambda h: h.current_load)
        else:
            logger.warning("Unknown strategy '%s' – falling back to round_robin.", strategy)
            return handlers[0]

    @staticmethod
    def _update_latency(info: HandlerInfo, latency_ms: float) -> None:
        """Update the running average latency for a handler.

        Uses an exponential moving average with α = 0.1.

        Parameters
        ----------
        info : HandlerInfo
        latency_ms : float
        """
        if info.total_requests <= 1:
            info.avg_latency_ms = latency_ms
        else:
            alpha = 0.1
            info.avg_latency_ms = alpha * latency_ms + (1 - alpha) * info.avg_latency_ms
