# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""Helpers for reporting missing per-application optional dependencies.

Each djura application declares its third-party requirements as an extra
(see ``[project.optional-dependencies]`` in ``pyproject.toml``), so a bare
``pip install djura`` provides only the shared core. Importing an
application whose extra is not installed would otherwise fail with a bare
``ModuleNotFoundError`` naming a package the user never heard of; this
module turns that into a message naming the install command to run.
"""
from types import TracebackType
from typing import Optional, Type


class require_extra:
    """Context manager re-raising import failures with install guidance.

    Wraps the top-level imports of an application's ``__init__`` module.
    An :class:`ImportError` for a missing third-party package is re-raised
    with the ``pip install`` command that would provide it. Import errors
    originating inside djura itself are left untouched, since those signal
    a bug rather than an incomplete installation.

    Parameters
    ----------
    extra : str
        Name of the extra providing this application's dependencies, which
        by convention matches the submodule name (e.g. ``"slf"``).

    Examples
    --------
    >>> with require_extra("slf"):          # doctest: +SKIP
    ...     from .slf import SLF
    """

    def __init__(self, extra: str) -> None:
        self.extra = extra

    def __enter__(self) -> "require_extra":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        if exc_type is None or not issubclass(exc_type, ImportError):
            return False

        missing = getattr(exc, "name", None)
        # A missing djura module is an internal error, not a missing extra.
        if missing is not None and missing.split(".")[0] == "djura":
            return False

        detail = f" (missing dependency: {missing!r})" if missing else ""
        raise ImportError(
            f"djura.{self.extra} requires the '{self.extra}' extra, which "
            f"is not installed{detail}.\n\n"
            f"Install it with:\n\n"
            f'    pip install "djura[{self.extra}]"\n\n'
            f"Or install every djura application at once:\n\n"
            f'    pip install "djura[all]"\n'
        ) from exc
