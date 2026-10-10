import io

import pytest
from PIL import Image
from sqlmodel import Session, select
from typer.testing import CliRunner

from marketplace_agent.cli import app
from marketplace_agent.core.db import make_engine
from marketplace_agent.core.models import Goal, Item, Photo

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep a developer's .env and toml out of the run
    monkeypatch.setenv("MA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("MA_DB_URL", raising=False)
    return tmp_path / "data"


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "desk.jpg"
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), (150, 100, 60)).save(buffer, "JPEG")
    path.write_bytes(buffer.getvalue())
    return path


@pytest.fixture
def item(data_dir, photo):
    result = runner.invoke(app, ["add-item", "--hint", "IKEA desk $120", "--photo", str(photo)])
    assert result.exit_code == 0, result.output
    return 1


def _rows(data_dir, model):
    engine = make_engine(f"sqlite:///{data_dir / 'agent.db'}")
    with Session(engine) as session:
        rows = session.exec(select(model)).all()
    engine.dispose()
    return rows


def test_init_creates_data_dir_and_db(data_dir):
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0, result.output
    assert (data_dir / "agent.db").is_file()
    assert str(data_dir.resolve()) in result.output
    assert "agent.db" in result.output


def test_add_item_stores_item_and_photos(data_dir, photo):
    result = runner.invoke(
        app, ["add-item", "--hint", "IKEA desk $120", "--photo", str(photo), "--photo", str(photo)]
    )

    assert result.exit_code == 0, result.output
    assert "Created item 1 with 1 photo(s)" in result.output  # same file twice dedupes
    assert "Hint price: $120.00" in result.output
    [item] = _rows(data_dir, Item)
    [stored] = _rows(data_dir, Photo)
    assert item.hint == "IKEA desk $120"
    assert (data_dir / stored.path).is_file()


def test_add_item_without_price_in_hint(data_dir, photo):
    result = runner.invoke(app, ["add-item", "--hint", "IKEA desk 2 drawers", "--photo", str(photo)])

    assert result.exit_code == 0, result.output
    assert "Hint price" not in result.output


def test_add_item_requires_a_photo(data_dir):
    result = runner.invoke(app, ["add-item", "--hint", "desk"])

    assert result.exit_code == 2
    assert "--photo" in result.output


def test_add_item_with_bad_photo_creates_nothing(data_dir, tmp_path):
    bad = tmp_path / "notes.jpg"
    bad.write_bytes(b"not a photo")

    result = runner.invoke(app, ["add-item", "--hint", "desk", "--photo", str(bad)])

    assert result.exit_code == 1
    assert "notes.jpg" in result.output
    assert _rows(data_dir, Item) == []


def test_items_lists_and_filters(item):
    result = runner.invoke(app, ["items"])

    assert result.exit_code == 0, result.output
    header, row = result.output.splitlines()
    assert header.split() == ["ID", "STATUS", "ITEM", "LIST", "PRICE"]
    assert row.split() == ["1", "draft", "IKEA", "desk", "$120", "-"]
    assert runner.invoke(app, ["items", "--status", "listed"]).output.strip() == "No items."


def test_items_shows_list_price_once_goal_is_set(item):
    runner.invoke(app, ["goal", "1", "--price", "110", "--floor", "90"])

    assert runner.invoke(app, ["items"]).output.splitlines()[1].endswith("$110.00")


def test_show_prints_item_goal_and_photos(item):
    result = runner.invoke(app, ["show", "1"])

    assert result.exit_code == 0, result.output
    assert "Hint:      IKEA desk $120" in result.output
    assert "Goal:      none" in result.output
    assert "Photos:    1" in result.output
    assert "photos/1/" in result.output and "(640x480)" in result.output


def test_show_missing_item(data_dir):
    result = runner.invoke(app, ["show", "9"])

    assert result.exit_code == 1
    assert "Item 9 not found" in result.output


def test_goal_converts_dollars_to_exact_cents(item, data_dir):
    result = runner.invoke(
        app, ["goal", "1", "--price", "120.10", "--floor", "95", "--firmness", "FIRM", "--days", "7"]
    )

    assert result.exit_code == 0, result.output
    assert "list $120.10, target $120.10, floor $95.00" in result.output
    [goal] = _rows(data_dir, Goal)
    assert (goal.list_price_cents, goal.target_price_cents, goal.floor_price_cents) == (
        12010,
        12010,
        9500,
    )
    assert goal.firmness == "firm"
    assert "list $120.10" in runner.invoke(app, ["show", "1"]).output


def test_goal_rejects_floor_above_price(item, data_dir):
    result = runner.invoke(app, ["goal", "1", "--price", "100", "--floor", "150"])

    assert result.exit_code == 1
    assert "floor price cannot be above the list price" in result.output
    assert _rows(data_dir, Goal) == []


@pytest.mark.parametrize("price", ["abc", "12.345", "NaN", "1e3", "-5"])
def test_goal_rejects_malformed_dollars(item, price):
    result = runner.invoke(app, ["goal", "1", "--price", price, "--floor", "1"])

    assert result.exit_code == 2
    assert "expected dollars" in result.output


def test_goal_for_missing_item(data_dir):
    result = runner.invoke(app, ["goal", "5", "--price", "10", "--floor", "5"])

    assert result.exit_code == 1
    assert "Item 5 not found" in result.output


TOO_BIG = "99999999999999999999"  # above SQLite's 2**63 - 1


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["show", TOO_BIG], "not in the range"),
        (["draft", TOO_BIG], "not in the range"),
        (["goal", TOO_BIG, "--price", "10", "--floor", "5"], "not in the range"),
        (["goal", "1", "--price", TOO_BIG, "--floor", "1"], "at most $10,000,000.00"),
        (["goal", "1", "--price", "100", "--floor", "1", "--days", "100000000"], "1<=x<=3650"),
    ],
)
def test_out_of_range_numbers_are_usage_errors(item, data_dir, args, error):
    result = runner.invoke(app, args)

    assert result.exit_code == 2, result.output
    assert isinstance(result.exception, SystemExit)
    assert error in " ".join(result.output.replace("│", " ").split())
    assert _rows(data_dir, Goal) == []


def test_add_item_with_absurd_hint_price_prints_no_price(data_dir, photo):
    hint = "desk $" + "9" * 4400
    result = runner.invoke(app, ["add-item", "--hint", hint, "--photo", str(photo)])

    assert result.exit_code == 0, result.output
    assert "Created item 1 with 1 photo(s)" in result.output
    assert "Hint price" not in result.output
