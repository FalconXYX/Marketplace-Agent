"""web.auth: the access token and its cookie, the Host check, and cross-site posts."""

import pytest

from marketplace_agent.web.auth import COOKIE, session_value
from test_web_support import BASE, build_world

TOKEN = "s3cret-token"


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path, web_token=TOKEN)


@pytest.fixture
def open_world(tmp_path):
    return build_world(tmp_path)


def _login(world) -> None:
    response = world.client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert response.status_code == 303


# ---------------------------------------------------------------- token and cookie


def test_without_the_token_every_page_is_refused(world):
    for path in ("/", "/items/new", "/items/1", "/photos/1"):
        response = world.client.get(path)
        assert response.status_code == 401
        assert "?token=" in response.text and TOKEN not in response.text
    assert world.client.post("/items/1/pause").status_code == 401


def test_a_wrong_token_is_refused(world):
    response = world.client.get("/?token=guess", follow_redirects=False)

    assert response.status_code == 401 and "set-cookie" not in response.headers


def test_the_token_once_sets_a_strict_http_only_cookie_and_leaves_the_url(world):
    response = world.client.get(f"/items/new?a=1&token={TOKEN}", follow_redirects=False)

    assert response.status_code == 303 and response.headers["location"] == "/items/new?a=1"
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE}={session_value(TOKEN)};")
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and "Path=/" in cookie
    assert "Secure" not in cookie  # plain http on loopback
    assert TOKEN not in cookie
    assert world.client.get("/items/new").status_code == 200  # the client kept the cookie


def test_over_https_the_cookie_is_secure(tmp_path):
    world = build_world(tmp_path, web_token=TOKEN, base_url="https://127.0.0.1")

    response = world.client.get(f"/?token={TOKEN}", follow_redirects=False)

    assert "Secure" in response.headers["set-cookie"]


def test_a_forged_cookie_is_refused(world):
    world.client.cookies.set(COOKIE, TOKEN)

    assert world.client.get("/").status_code == 401


def test_the_token_is_exchanged_only_on_a_get(world):
    response = world.client.post(f"/items/1/pause?token={TOKEN}")

    assert response.status_code == 401


def test_with_the_cookie_posts_work(world):
    _login(world)
    item_id = world.listed()

    response = world.htmx(f"/items/{item_id}/pause")

    assert response.status_code == 200 and "Paused." in response.text


def test_static_files_need_no_token(world):
    assert world.client.get("/static/htmx.min.js").status_code == 200
    assert world.client.get("/static/app.css").status_code == 200


def test_the_refusal_page_offers_to_try_again_for_links_opened_from_another_app(world):
    """A SameSite=Strict cookie is not sent on a navigation from another site; a tap on a link
    of this site sends it."""
    response = world.client.get("/items/3")

    assert response.status_code == 401 and 'href="">Try again</a>' in response.text


# ---------------------------------------------------------------- host names


@pytest.mark.parametrize(
    "host", ["127.0.0.1:8765", "localhost:8765", "[::1]:8765", "box.tail1234.ts.net"]
)
def test_without_a_token_loopback_and_tailscale_names_are_served(open_world, host):
    assert open_world.client.get("/", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "192.168.1.20:8765", "ts.net.evil.example"])
def test_without_a_token_other_host_names_are_refused(open_world, host):
    """A DNS rebinding page reaches the server under its own host name."""
    response = open_world.client.get("/", headers={"Host": host})

    assert response.status_code == 403 and "MA_WEB_TOKEN" in response.text


def test_with_a_token_any_host_name_works(world):
    _login(world)

    assert world.client.get("/", headers={"Host": "192.168.1.20:8765"}).status_code == 200


# ---------------------------------------------------------------- cross-site posts


def _post(world, **headers):
    item_id = world.listed()
    return world.client.post(f"/items/{item_id}/pause", headers={"Origin": "", **headers})


def test_a_post_from_another_site_is_refused(open_world):
    response = _post(open_world, Origin="https://evil.example")

    assert response.status_code == 403 and "did not come from this web UI" in response.text


def test_a_post_with_neither_origin_nor_referer_is_refused(open_world):
    item_id = open_world.listed()
    client = open_world.client
    client.headers.pop("Origin")

    assert client.post(f"/items/{item_id}/pause").status_code == 403


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1.evil.example"},
        {"Origin": "http://127.0.0.1:9999"},
        {"Referer": "https://evil.example/items/1"},
    ],
)
def test_lookalike_origins_are_refused(open_world, headers):
    assert _post(open_world, **headers).status_code == 403


def test_a_same_site_referer_is_enough(open_world):
    assert _post(open_world, Referer=f"{BASE}/items/1").status_code == 200


def test_behind_a_proxy_the_forwarded_host_counts(open_world):
    """`tailscale serve` may pass the phone's host name as X-Forwarded-Host."""
    response = _post(
        open_world,
        Origin="https://box.tail1234.ts.net",
        **{"X-Forwarded-Host": "box.tail1234.ts.net"},
    )

    assert response.status_code == 200


def test_default_ports_are_the_same_origin(tmp_path):
    world = build_world(tmp_path, base_url="https://box.tail1234.ts.net")

    assert _post(world, Origin="https://box.tail1234.ts.net:443").status_code == 200


def test_a_cross_site_post_is_refused_even_with_the_cookie(world):
    _login(world)

    assert _post(world, Origin="https://evil.example").status_code == 403
