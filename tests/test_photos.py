import hashlib
import io
import struct

import pytest
from PIL import ExifTags, Image, ImageCms
from sqlmodel import select

from marketplace_agent.core.items import ItemNotFound, create_item
from marketplace_agent.core.models import Photo
from marketplace_agent.intake.photos import (
    MAX_EDGE,
    UnsupportedPhoto,
    ingest_photo,
    ingest_photos,
    load_photo_bytes,
    normalize_photo,
    photo_file,
)

XMP = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF/></x:xmpmeta>'
SECRET_COMMENT = b"home: 43.6532N 79.3832W"
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _jpeg(size=(400, 300), color=(120, 80, 40), **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "JPEG", **save_kwargs)
    return buffer.getvalue()


def _exif(orientation: int | None = None, gps: bool = False) -> Image.Exif:
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "Phone"
    if orientation is not None:
        exif[ExifTags.Base.Orientation] = orientation
    if gps:
        exif[ExifTags.Base.GPSInfo] = {
            ExifTags.GPS.GPSLatitudeRef: "N",
            ExifTags.GPS.GPSLatitude: (43.0, 39.0, 11.5),
            ExifTags.GPS.GPSLongitudeRef: "W",
            ExifTags.GPS.GPSLongitude: (79.0, 22.0, 59.5),
        }
    return exif


def _jpeg_markers(data: bytes) -> set[int]:
    """Marker bytes of every JPEG segment before the image data."""
    markers, pos = set(), 2
    while data[pos + 1] != 0xDA:  # SOS: compressed scan follows
        marker, length = data[pos + 1], struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        markers.add(marker)
        pos += 2 + length
    return markers


@pytest.fixture
def item_id(session, clock) -> int:
    item_id = create_item(session, clock, "IKEA desk $120").id
    assert item_id is not None
    return item_id


@pytest.mark.parametrize("with_icc", [True, False])  # the color conversion path rebuilds the image
def test_strips_gps_exif_xmp_icc_and_comments(with_icc):
    icc = {"icc_profile": SRGB_PROFILE} if with_icc else {}
    source = _jpeg(exif=_exif(gps=True), xmp=XMP, comment=SECRET_COMMENT, **icc)
    original = Image.open(io.BytesIO(source))
    assert original.getexif().get_ifd(ExifTags.IFD.GPSInfo), "fixture must carry GPS"
    assert {"exif", "xmp", "comment", *icc} <= original.info.keys()

    jpeg, _, _ = normalize_photo(source)

    stored = Image.open(io.BytesIO(jpeg))
    assert not stored.getexif()
    assert not stored.getexif().get_ifd(ExifTags.IFD.GPSInfo)
    assert not {"exif", "xmp", "icc_profile", "comment"} & stored.info.keys()
    assert SECRET_COMMENT not in jpeg and b"Exif\x00\x00" not in jpeg
    # Only APP0 (JFIF header) may appear: APP1 holds EXIF, its thumbnail and XMP; APP2 holds ICC.
    app_markers = {m for m in _jpeg_markers(jpeg) if 0xE0 <= m <= 0xEF}
    assert app_markers <= {0xE0}
    assert 0xFE not in _jpeg_markers(jpeg)  # COM


def test_orientation_6_rotates_upright():
    source = io.BytesIO()
    image = Image.new("RGB", (400, 300), "blue")
    image.paste((255, 0, 0), (0, 0, 200, 300))  # left half red
    image.save(source, "JPEG", exif=_exif(orientation=6))

    jpeg, width, height = normalize_photo(source.getvalue())

    upright = Image.open(io.BytesIO(jpeg))
    assert (width, height) == upright.size == (300, 400)
    red, _, blue = upright.getpixel((150, 50))  # stored left edge now on top
    assert red > 200 and blue < 60


def test_long_edge_is_downscaled():
    jpeg, width, height = normalize_photo(_jpeg(size=(4000, 1000)))
    assert (width, height) == Image.open(io.BytesIO(jpeg)).size == (MAX_EDGE, MAX_EDGE // 4)


def test_small_photo_keeps_size():
    assert normalize_photo(_jpeg(size=(640, 480)))[1:] == (640, 480)


def test_png_alpha_flattens_onto_white_rgb_jpeg():
    source = io.BytesIO()
    Image.new("RGBA", (50, 40), (0, 0, 0, 0)).save(source, "PNG")

    jpeg, _, _ = normalize_photo(source.getvalue())

    stored = Image.open(io.BytesIO(jpeg))
    assert (stored.format, stored.mode) == ("JPEG", "RGB")
    assert all(channel > 245 for channel in stored.getpixel((25, 20)))


def _encode(image: Image.Image, fmt: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def _stored_gray(data: bytes) -> int:
    jpeg, _, _ = normalize_photo(data)
    stored = Image.open(io.BytesIO(jpeg))
    assert stored.mode == "RGB"
    red, green, blue = stored.getpixel((4, 4))
    assert red == green == blue
    return red


@pytest.mark.parametrize(
    ("mode", "value", "fmt", "gray"),
    [
        ("I;16", 32768, "PNG", 128),  # mid gray, from a scanner or an edited 16-bit PNG
        ("I;16", 2000, "PNG", 8),  # near black
        ("I;16", 1000, "TIFF", 4),
        ("I", 200, "TIFF", 200),  # 32-bit container, 8-bit values
        ("I", 40000, "TIFF", 156),  # 32-bit container, 16-bit values
        ("F", 0.5, "TIFF", 128),  # float, 0 to 1
    ],
)
def test_high_bit_depth_grayscale_is_scaled_not_clipped(mode, value, fmt, gray):
    assert _stored_gray(_encode(Image.new(mode, (8, 8), value), fmt)) == gray


def test_16_bit_gradient_keeps_its_shades():
    gradient = Image.new("I;16", (256, 1))
    gradient.putdata([i * 257 for i in range(256)])

    jpeg, _, _ = normalize_photo(_encode(gradient, "PNG"))

    stored = Image.open(io.BytesIO(jpeg)).convert("L")
    shades = [stored.getpixel((x, 0)) for x in range(256)]
    assert shades[0] <= 2 and shades[-1] >= 253
    assert shades[128] == pytest.approx(128, abs=3)
    assert len(set(shades)) > 200


@pytest.mark.parametrize("data", [b"", b"definitely not a photo", _jpeg()[:200]])
def test_rejects_non_images(data):
    with pytest.raises(UnsupportedPhoto):
        normalize_photo(data)


@pytest.mark.parametrize("side", [120, 300])  # Pillow warns above the limit, errors above 2x
def test_rejects_decompression_bombs(monkeypatch, side):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100 * 100)
    with pytest.raises(UnsupportedPhoto, match="too large"):
        normalize_photo(_jpeg(size=(side, side)))


def test_ingest_stores_relative_path_and_content(session, settings, item_id):
    photo = ingest_photo(session, settings, item_id, _jpeg())

    jpeg, width, height = normalize_photo(_jpeg())
    sha = hashlib.sha256(jpeg).hexdigest()
    assert photo.path == f"photos/{item_id}/{sha[:16]}.jpg"
    assert (photo.sha256, photo.width, photo.height, photo.position) == (sha, width, height, 0)
    assert photo_file(settings, photo) == settings.data_dir / photo.path
    assert load_photo_bytes(settings, photo) == jpeg


def test_duplicate_ingest_returns_existing_row(session, settings, item_id):
    first = ingest_photo(session, settings, item_id, _jpeg())
    second = ingest_photo(session, settings, item_id, _jpeg())

    assert second.id == first.id
    assert len(session.exec(select(Photo)).all()) == 1
    assert len(list((settings.photos_dir / str(item_id)).iterdir())) == 1


def test_positions_default_to_next_index(session, settings, item_id):
    first = ingest_photo(session, settings, item_id, _jpeg(color=(1, 2, 3)))
    second = ingest_photo(session, settings, item_id, _jpeg(color=(200, 100, 50)))
    pinned = ingest_photo(session, settings, item_id, _jpeg(color=(9, 9, 9)), position=7)
    after = ingest_photo(session, settings, item_id, _jpeg(color=(90, 200, 90)))

    assert [first.position, second.position, pinned.position, after.position] == [0, 1, 7, 8]


def test_ingest_photos_accepts_paths_and_bytes(session, settings, item_id, tmp_path):
    path = tmp_path / "desk.jpg"
    path.write_bytes(_jpeg(color=(10, 20, 30)))

    photos = ingest_photos(session, settings, item_id, [path, _jpeg(color=(200, 10, 10))])

    assert [p.position for p in photos] == [0, 1]
    assert all(photo_file(settings, p).is_file() for p in photos)


def test_ingest_photos_stores_nothing_when_one_is_bad(session, settings, item_id, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_bytes(b"not a photo")

    with pytest.raises(UnsupportedPhoto, match="notes.txt"):
        ingest_photos(session, settings, item_id, [_jpeg(), bad])

    assert not session.exec(select(Photo)).all()
    assert not settings.photos_dir.exists()


def test_ingest_for_missing_item_raises(session, settings):
    with pytest.raises(ItemNotFound):
        ingest_photo(session, settings, 999, _jpeg())
    assert not settings.photos_dir.exists()
