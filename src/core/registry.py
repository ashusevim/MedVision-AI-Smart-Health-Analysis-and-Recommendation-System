"""
Registry for MedVision-AI
===========================

Implements the registry pattern for model and component registration,
supporting factory-based creation and decorator-based registration.

Typical usage::

    registry = Registry()

    # Decorator-based registration
    @Registry.register("chest_xray_model")
    class ChestXRayModel:
        def __init__(self, **kwargs):
            ...

    # Factory creation
    model = Registry.get("chest_xray_model", pretrained=True)

    # Or using an instance
    registry = Registry(name="models")
    @registry.register("resnet50")
    class ResNet50:
        pass
    model = registry.get("resnet50")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class RegistryEntry:
    """Metadata about a registered component.

    Attributes
    ----------
    name : str
        Unique registration key.
    cls : Type
        The registered class or factory callable.
    factory : Optional[Callable]
        Optional factory function that overrides ``cls`` for instantiation.
    metadata : Dict[str, Any]
        Arbitrary metadata (version, description, tags, etc.).
    tags : Set[str]
        Searchable tags for filtering.
    is_singleton : bool
        If *True*, only one instance is created and reused.
    _instance : Optional[Any]
        Cached singleton instance (internal).
    """

    name: str
    cls: Type
    factory: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    is_singleton: bool = False
    _instance: Optional[Any] = field(default=None, repr=False)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """Instantiate the registered component.

        If a factory function was provided, it is used instead of the
        class constructor.  For singletons, the cached instance is
        returned on subsequent calls.

        Parameters
        ----------
        *args, **kwargs
            Arguments forwarded to the factory / constructor.

        Returns
        -------
        Any
            A new (or cached singleton) instance.
        """
        if self.is_singleton and self._instance is not None:
            return self._instance

        if self.factory is not None:
            instance = self.factory(*args, **kwargs)
        else:
            instance = self.cls(*args, **kwargs)

        if self.is_singleton:
            self._instance = instance
        return instance


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class Registry:
    """A general-purpose registry for classes, factories, and components.

    Supports:

    * **Explicit registration** – call :meth:`register` with a name and
      class / factory.
    * **Decorator registration** – use ``@Registry.register("name")`` as
      a class decorator (class-level) or ``@registry.register("name")``
      on an instance.
    * **Factory pattern** – register a factory function that produces
      instances with custom logic.
    * **Singletons** – mark a registration as a singleton so that
      :meth:`get` always returns the same instance.
    * **Tag-based filtering** – attach tags and query by tag.

    Examples
    --------
    >>> # Class-level decorator registration (uses global registry)
    >>> @Registry.register("resnet50")
    ... class ResNet50:
    ...     def __init__(self, pretrained: bool = False):
    ...         self.pretrained = pretrained
    ...
    >>> model = Registry.get("resnet50", pretrained=True)
    >>>
    >>> # Instance-based registration
    >>> registry = Registry(name="models")
    >>> registry.register("optimizer", factory=lambda lr=1e-3: {"type": "adam", "lr": lr})
    >>> opt = registry.get("optimizer", lr=3e-4)
    """

    # Class-level (global) registry for decorator usage
    _global_registry: Optional[Registry] = None

    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._entries: Dict[str, RegistryEntry] = {}
        logger.info("Registry '%s' created.", self._name)

    # ------------------------------------------------------------------
    # Class-level (static) registration for @Registry.register("name")
    # ------------------------------------------------------------------

    @classmethod
    def _get_global_registry(cls) -> Registry:
        """Return (and lazily create) the global class-level registry.

        Returns
        -------
        Registry
        """
        if cls._global_registry is None:
            cls._global_registry = Registry(name="global")
        return cls._global_registry

    @classmethod
    def register(
        cls,
        name: str,
        klass: Optional[Type] = None,
        factory: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        is_singleton: bool = False,
    ) -> Callable:
        """Register a class or factory under the given name.

        Can be used as a **class decorator** or called directly.  When
        used as ``@Registry.register("name")``, the class is registered
        in the global registry.  When called on an instance, it
        registers in that instance.

        Parameters
        ----------
        name : str
            Unique registration key.
        klass : Type, optional
            The class to register.  If *None*, the method returns a
            decorator.
        factory : Callable, optional
            A factory function.  If provided, ``factory`` is used for
            instance creation instead of ``klass``.
        metadata : Dict[str, Any], optional
            Arbitrary metadata.
        tags : Set[str], optional
            Searchable tags.
        is_singleton : bool
            If *True*, only one instance is created and cached.

        Returns
        -------
        Callable
            When used as a decorator, returns the class unchanged.

        Examples
        --------
        >>> # Decorator registration on the class
        >>> @Registry.register("model_a")
        ... class MyModel:
        ...     pass
        ...
        >>> # Direct registration on an instance
        >>> registry = Registry(name="local")
        >>> registry.register("model_b", klass=MyOtherModel)
        """
        def _decorator(kls: Type) -> Type:
            # Determine target registry: if called on instance, use self;
            # if called on class, use global registry.
            target = cls._get_global_registry() if isinstance(cls, type) else self  # type: ignore[name-error]
            target._add_entry(
                name=name,
                cls=kls,
                factory=factory,
                metadata=metadata,
                tags=tags,
                is_singleton=is_singleton,
            )
            return kls

        # When called on an instance (self.register(...)), we need
        # instance-level dispatch.  This is handled below.
        if not isinstance(cls, type):
            # Called on an instance: registry.register("name", cls=MyClass)
            self = cls  # noqa: F841 – captured by _decorator closure
            if klass is not None:
                _decorator(klass)
                return klass
            return _decorator

        # Called on the class: Registry.register("name") as decorator
        if klass is not None:
            _decorator(klass)
            return klass

        return _decorator

    def _add_entry(
        self,
        name: str,
        cls: Type,
        factory: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        is_singleton: bool = False,
    ) -> None:
        """Internal helper to add or update an entry.

        Parameters
        ----------
        name : str
        cls : Type
        factory : Callable, optional
        metadata : Dict[str, Any], optional
        tags : Set[str], optional
        is_singleton : bool
        """
        if name in self._entries:
            logger.warning("Overwriting existing registration for '%s'.", name)

        entry = RegistryEntry(
            name=name,
            cls=cls,
            factory=factory,
            metadata=metadata or {},
            tags=tags or set(),
            is_singleton=is_singleton,
        )
        self._entries[name] = entry
        logger.info(
            "Registered '%s' (cls=%s, singleton=%s, tags=%s)",
            name,
            cls.__name__ if hasattr(cls, "__name__") else str(cls),
            is_singleton,
            tags or set(),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        """Retrieve (and instantiate) a registered component.

        When called as ``Registry.get("name")``, looks up the global
        registry.  When called on an instance, looks up that instance's
        entries.

        Parameters
        ----------
        name : str
            Registration key.
        *args, **kwargs
            Arguments forwarded to the factory / constructor.

        Returns
        -------
        Any
            A new or cached singleton instance.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if isinstance(cls, type):
            # Called on the class → use global registry
            registry = cls._get_global_registry()
        else:
            # Called on an instance
            registry = cls  # type: ignore[assignment]

        entry = registry._entries.get(name)
        if entry is None:
            raise KeyError(
                f"'{name}' is not registered in registry '{registry._name}'. "
                f"Available: {list(registry._entries.keys())}"
            )
        return entry.create(*args, **kwargs)

    def get_entry(self, name: str) -> RegistryEntry:
        """Return the raw :class:`RegistryEntry` for *name*.

        Parameters
        ----------
        name : str

        Returns
        -------
        RegistryEntry

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in self._entries:
            raise KeyError(f"'{name}' is not registered.")
        return self._entries[name]

    # ------------------------------------------------------------------
    # Listing & search
    # ------------------------------------------------------------------

    def list_registered(self, tag: Optional[str] = None) -> List[str]:
        """List all registered component names, optionally filtered by tag.

        Parameters
        ----------
        tag : str, optional
            If provided, only components with this tag are returned.

        Returns
        -------
        List[str]
        """
        if tag is None:
            return list(self._entries.keys())
        return [
            name for name, entry in self._entries.items()
            if tag in entry.tags
        ]

    def list_entries(self, tag: Optional[str] = None) -> Dict[str, RegistryEntry]:
        """Return a mapping of name → RegistryEntry, optionally filtered by tag.

        Parameters
        ----------
        tag : str, optional

        Returns
        -------
        Dict[str, RegistryEntry]
        """
        if tag is None:
            return dict(self._entries)
        return {
            name: entry for name, entry in self._entries.items()
            if tag in entry.tags
        }

    def has(self, name: str) -> bool:
        """Check whether a component with *name* is registered.

        Parameters
        ----------
        name : str

        Returns
        -------
        bool
        """
        return name in self._entries

    # ------------------------------------------------------------------
    # Un-registration
    # ------------------------------------------------------------------

    def unregister(self, name: str) -> RegistryEntry:
        """Remove a component from the registry.

        Parameters
        ----------
        name : str
            Registration key to remove.

        Returns
        -------
        RegistryEntry
            The removed entry.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in self._entries:
            raise KeyError(f"'{name}' is not registered in registry '{self._name}'.")
        entry = self._entries.pop(name)
        logger.info("Unregistered '%s' from registry '%s'.", name, self._name)
        return entry

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all registered components."""
        self._entries.clear()
        logger.info("Registry '%s' cleared.", self._name)

    def __len__(self) -> int:
        """Return the number of registered components."""
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        """Check whether *name* is registered."""
        return self.has(name)

    def __repr__(self) -> str:
        return f"Registry(name='{self._name}', entries={len(self._entries)})"


# ---------------------------------------------------------------------------
# Global registry singleton accessor
# ---------------------------------------------------------------------------

def get_global_registry() -> Registry:
    """Return the module-level global registry singleton.

    Returns
    -------
    Registry
    """
    return Registry._get_global_registry()
