"""Define the custom handler."""

from datetime import datetime
from typing import Any, ClassVar

import onnx
from attrs import define
from lazyscribe._utils import utcnow
from lazyscribe.artifacts.base import Artifact
from slugify import slugify

__all__: list[str] = ["ONNXArtifact"]


@define(auto_attribs=True)
class ONNXArtifact(Artifact):
    """Handler for ONNX model objects."""

    alias: ClassVar[str] = "onnx"
    suffix: ClassVar[str] = "onnx"
    binary: ClassVar[bool] = True
    output_only: ClassVar[bool] = False

    @classmethod
    def construct(
        cls,
        name: str,
        value: Any | None = None,
        fname: str | None = None,
        created_at: datetime | None = None,
        writer_kwargs: dict | None = None,
        version: int = 0,
        dirty: bool = True,
        **kwargs,
    ):
        """Construct the handler class."""
        created_at = created_at or utcnow()
        return cls(  # type: ignore[call-arg]
            name=name,
            value=value,
            fname=fname
            or f"{slugify(name)}-{slugify(created_at.strftime('%Y%m%d%H%M%S'))}.{cls.suffix}",
            writer_kwargs=writer_kwargs or {},
            version=version,
            created_at=created_at,
            dirty=dirty,
        )

    @classmethod
    def read(cls, buf, **kwargs):
        """Read in the artifact.

        Parameters
        ----------
        buf : file-like object
            The buffer from a ``fsspec`` filesystem.
        **kwargs
            Keyword arguments for the read method.

        Returns
        -------
        Any
            The artifact.
        """
        return onnx.load(buf, **kwargs)

    @classmethod
    def write(cls, obj, buf, **kwargs):
        """Write the content to an ONNX file.

        Parameters
        ----------
        obj : object
            The ONNX-serializable object.
        buf : file-like object
            The buffer from a ``fsspec`` filesystem.
        **kwargs
            Keyword arguments for :py:meth:`onnx.save`.
        """
        onnx.save(obj, buf, **kwargs)
