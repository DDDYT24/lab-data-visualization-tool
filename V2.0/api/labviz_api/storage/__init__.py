"""Provider-neutral object storage interfaces and local development backend."""

from .base import ObjectInfo, ObjectStorage
from .local import LocalObjectStorage

__all__ = ["LocalObjectStorage", "ObjectInfo", "ObjectStorage"]
