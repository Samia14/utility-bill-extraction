"""Open a bill image. JPG and PNG only. No company-specific image templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bill_extract.errors import DependencyError, DocumentLoadError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class LoadedPage:
    """One bill image, validated and encoded as PNG bytes for the vision model."""

    bill_id: str
    source: str
    png_bytes: bytes


def is_supported(path: Path) -> bool:
    return path.suffix.casefold() in IMAGE_SUFFIXES


def ensure_pillow() -> None:
    """Fail clearly when image support is not installed."""
    try:
        import PIL  # noqa: F401
    except ImportError as exc:
        raise DependencyError(
            "The pillow package is not installed. Install the packages in requirements.txt and try again."
        ) from exc


def load_pages(path: Path) -> list[LoadedPage]:
    """Load one bill image. A bad file raises DocumentLoadError."""
    if not path.exists() or not path.is_file():
        raise DocumentLoadError(f"The file '{path.name}' was not found.")
    if path.suffix.casefold() not in IMAGE_SUFFIXES:
        raise DocumentLoadError(
            f"The file '{path.name}' is not a supported bill image. Use a JPG or PNG file."
        )
    return [_image_page(path)]


def _image_page(path: Path) -> LoadedPage:
    image = _open_image(path)
    try:
        png_bytes = _png_bytes(image)
    finally:
        image.close()
    return LoadedPage(bill_id=path.stem, source=str(path), png_bytes=png_bytes)


def _open_image(path: Path):
    ensure_pillow()
    from PIL import Image, UnidentifiedImageError

    bomb_error = getattr(Image, "DecompressionBombError", OSError)
    try:
        with Image.open(path) as probe:
            probe.verify()
        image = Image.open(path)
        image.load()
    except (DocumentLoadError, DependencyError):
        raise
    except (OSError, UnidentifiedImageError, ValueError, bomb_error) as exc:
        raise DocumentLoadError(
            f"The image '{path.name}' could not be opened. It may be damaged."
        ) from exc
    except Exception as exc:
        raise DocumentLoadError(
            f"The image '{path.name}' could not be opened. It may be damaged."
        ) from exc
    return image


def _png_bytes(image) -> bytes:
    from io import BytesIO

    from PIL import ImageOps

    oriented = ImageOps.exif_transpose(image)
    if oriented is None:
        oriented = image
    if oriented.mode != "RGB":
        oriented = oriented.convert("RGB")
    buffer = BytesIO()
    oriented.save(buffer, format="PNG")
    return buffer.getvalue()
