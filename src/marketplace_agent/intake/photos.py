"""Photo ingest: normalize to a metadata-free JPEG, dedupe by content hash, store under data_dir.

Stripping is mandatory (CLAUDE.md rule 8): phone photos carry GPS coordinates of the seller's home.
"""
import hashlib
import io
import warnings
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageCms, ImageOps
from sqlmodel import Session, func, select

from marketplace_agent.config import Settings
from marketplace_agent.core.items import get_item
from marketplace_agent.core.models import Photo

MAX_EDGE = 2048
JPEG_QUALITY = 88

# EPS, PDF and friends shell out or parse far more than a phone photo needs. JPEG covers MPO.
_ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF")
_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
# Mode I and F carry no bit depth, so full scale is the smallest of these holding the brightest
# pixel, else the brightest pixel.
_FULL_SCALES = {"I": (255.0, 65535.0), "F": (1.0, 255.0, 65535.0)}


class UnsupportedPhoto(ValueError):
    pass


def normalize_photo(data: bytes) -> tuple[bytes, int, int]:
    """Return (jpeg_bytes, width, height): upright, sRGB, long edge <= MAX_EDGE, no metadata."""
    image = _flatten(_to_srgb(_to_8bit(_open_upright(data))))
    image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
    return _encode(image), image.width, image.height


def ingest_photo(
    session: Session, settings: Settings, item_id: int, data: bytes, position: int | None = None
) -> Photo:
    """Normalize and store one photo. Flushes; the caller commits."""
    get_item(session, item_id)
    return _store(session, settings, item_id, normalize_photo(data), position)


def ingest_photos(
    session: Session, settings: Settings, item_id: int, sources: Iterable[bytes | Path]
) -> list[Photo]:
    """Normalize every source before writing any, so one bad file stores nothing."""
    get_item(session, item_id)
    normalized = [_normalize_source(i, source) for i, source in enumerate(sources)]
    return [_store(session, settings, item_id, n, None) for n in normalized]


def photo_file(settings: Settings, photo: Photo) -> Path:
    return settings.data_dir / photo.path


def load_photo_bytes(settings: Settings, photo: Photo) -> bytes:
    return photo_file(settings, photo).read_bytes()


def _open_upright(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data), formats=_ALLOWED_FORMATS)
            image.load()
            return ImageOps.exif_transpose(image)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise UnsupportedPhoto(f"image too large: {exc}") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise UnsupportedPhoto("not a supported image (JPEG, PNG, WebP, GIF, BMP, TIFF)") from exc


def _to_8bit(image: Image.Image) -> Image.Image:
    """Scale 16 and 32-bit grayscale (PNG, TIFF, scanners) to 8-bit. convert("RGB") clips
    instead of scaling, which turns all but the darkest pixels white."""
    if image.mode.startswith("I;16"):
        full = 65535.0
        image = image.convert("I")
    elif image.mode in _FULL_SCALES:
        top = image.getextrema()[1]
        full = next((s for s in _FULL_SCALES[image.mode] if top <= s), float(top))
    else:
        return image
    scaled = image.convert("F").point(lambda v: v * (255.0 / full) + 0.5).convert("L")
    scaled.info.pop("transparency", None)  # a 16-bit key color matches no 8-bit pixel
    return scaled


def _to_srgb(image: Image.Image) -> Image.Image:
    """Bake an embedded color profile into the pixels, since the profile itself is dropped."""
    icc = image.info.get("icc_profile")
    if not icc:
        return image
    mode = "RGBA" if image.mode == "RGBA" else "RGB"
    try:
        converted = ImageCms.profileToProfile(image, io.BytesIO(icc), _SRGB, outputMode=mode)
    except (ImageCms.PyCMSError, OSError, ValueError):
        return image  # unusable profile: keep pixels as they are
    return converted or image


def _flatten(image: Image.Image) -> Image.Image:
    """RGB with any transparency composited onto white."""
    if not image.has_transparency_data:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    flat = Image.new("RGB", rgba.size, "white")
    flat.paste(rgba, mask=rgba.getchannel("A"))
    return flat


def _encode(image: Image.Image) -> bytes:
    image.info.clear()  # Pillow's JPEG writer falls back to image.info for some markers
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def _normalize_source(index: int, source: bytes | Path) -> tuple[bytes, int, int]:
    data = source.read_bytes() if isinstance(source, Path) else source
    try:
        return normalize_photo(data)
    except UnsupportedPhoto as exc:
        label = source.name if isinstance(source, Path) else f"photo {index + 1}"
        raise UnsupportedPhoto(f"{label}: {exc}") from exc


def _store(
    session: Session,
    settings: Settings,
    item_id: int,
    normalized: tuple[bytes, int, int],
    position: int | None,
) -> Photo:
    jpeg, width, height = normalized
    sha = hashlib.sha256(jpeg).hexdigest()
    existing = session.exec(
        select(Photo).where(Photo.item_id == item_id, Photo.sha256 == sha)
    ).first()
    if existing is not None:
        return existing
    target = settings.photos_dir / str(item_id) / f"{sha[:16]}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(jpeg)
    photo = Photo(
        item_id=item_id,
        path=target.relative_to(settings.data_dir).as_posix(),
        position=_next_position(session, item_id) if position is None else position,
        width=width,
        height=height,
        sha256=sha,
    )
    session.add(photo)
    session.flush()
    return photo


def _next_position(session: Session, item_id: int) -> int:
    highest = session.exec(select(func.max(Photo.position)).where(Photo.item_id == item_id)).one()
    return 0 if highest is None else highest + 1
