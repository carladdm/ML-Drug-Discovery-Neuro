"""Centralised logging setup.

Every module obtains its logger via :func:`get_logger`; the root logging
configuration is installed exactly once instead of calling
``logging.basicConfig`` in every module.
"""

from __future__ import annotations

import logging
import sys

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_CONFIGURED = False


def configure_logging(level: int = logging.INFO, *, force: bool = False) -> None:
    """Configure the root logger exactly once (idempotent unless ``force=True``).

    Args:
        level: Root logging level (default ``INFO``).
        force: Re-apply configuration even if it was set before.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    logging.basicConfig(level=level, format=DEFAULT_FORMAT, stream=sys.stdout, force=force)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, ensuring the root logger is configured.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    configure_logging()
    return logging.getLogger(name)
