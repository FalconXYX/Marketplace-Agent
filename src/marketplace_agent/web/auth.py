"""Who may use the web UI, and from where.

| Check | Rule |
|---|---|
| Token | With web_token set, every page needs the session cookie |
| Cookie | Set by opening any page once with ?token=<web_token>, which then redirects without it |
| Host | Without web_token, only loopback and Tailscale (*.ts.net) names: no DNS rebinding |
| Origin | A POST needs an Origin, else a Referer, naming this host; anything else is refused |

The cookie is HttpOnly and SameSite=Strict, and holds an HMAC of the token, never the token.
/static/ is public: it holds only htmx, CSS, and JS.
"""

import hashlib
import hmac
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse

from marketplace_agent.config import Settings

COOKIE = "ma_session"
COOKIE_MAX_AGE = 180 * 86_400
TOKEN_PARAM = "token"
STATIC_PREFIX = "/static/"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
TAILSCALE_SUFFIX = ".ts.net"
_DEFAULT_PORTS = (":80", ":443")
_SESSION_LABEL = b"marketplace-agent web session"

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; "
        "form-action 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "Vary": "HX-Request, Cookie",
}

Deny = Callable[[Request, int, str], Response]
Next = Callable[[Request], Awaitable[Response]]

NEED_TOKEN = (
    "This page needs your access link. Open the web UI once with the link that ends in "
    "?token=..., and this browser remembers it."
)
BAD_TOKEN = "That access token is not right. Check MA_WEB_TOKEN and open the link again."
BAD_HOST = (
    "This address is not allowed without a token. Use http://127.0.0.1 or your Tailscale "
    "name, or set MA_WEB_TOKEN to open the web UI from elsewhere."
)
CROSS_SITE = "This request did not come from this web UI, so it was refused. Reload and try again."


def install_guard(app: FastAPI, settings: Settings, deny: Deny) -> None:
    secret = settings.web_token.get_secret_value() if settings.web_token else ""

    @app.middleware("http")
    async def guard(request: Request, call_next: Next) -> Response:
        response = await _check(request, call_next, secret, deny)
        response.headers.update(SECURITY_HEADERS)
        return response


def session_value(secret: str) -> str:
    return hmac.new(secret.encode(), _SESSION_LABEL, hashlib.sha256).hexdigest()


async def _check(request: Request, call_next: Next, secret: str, deny: Deny) -> Response:
    if request.url.path.startswith(STATIC_PREFIX):
        return await call_next(request)
    if not secret and not _allowed_host(request.url.hostname or ""):
        return deny(request, 403, BAD_HOST)
    if request.method not in SAFE_METHODS and not _same_origin(request):
        return deny(request, 403, CROSS_SITE)
    if not secret:
        return await call_next(request)
    offered = request.query_params.get(TOKEN_PARAM)
    if offered is not None and request.method == "GET":
        if not hmac.compare_digest(offered.encode(), secret.encode()):
            return deny(request, 401, BAD_TOKEN)
        return _remember(request, secret)
    cookie = request.cookies.get(COOKIE, "")
    if not hmac.compare_digest(cookie.encode(), session_value(secret).encode()):
        return deny(request, 401, NEED_TOKEN)
    return await call_next(request)


def _remember(request: Request, secret: str) -> Response:
    """Set the session cookie and drop the token from the address bar and history."""
    clean = request.url.remove_query_params(TOKEN_PARAM)
    target = clean.path + (f"?{clean.query}" if clean.query else "")
    response = RedirectResponse(target, status_code=303)
    response.set_cookie(
        COOKIE,
        session_value(secret),
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
    )
    return response


def _allowed_host(hostname: str) -> bool:
    host = hostname.lower().strip("[]")
    return host in LOOPBACK_HOSTS or host.endswith(TAILSCALE_SUFFIX)


def _same_origin(request: Request) -> bool:
    """Origin, else Referer, must name the host this request was sent to. A proxy such as
    `tailscale serve` may pass the browser's host as X-Forwarded-Host; a cross-site page cannot
    set that header on a form post."""
    source = request.headers.get("origin") or request.headers.get("referer")
    if not source:
        return False
    claimed = _netloc(urlsplit(source).netloc)
    if not claimed:
        return False
    hosts = [request.headers.get("host", "")]
    hosts += request.headers.get("x-forwarded-host", "").split(",")
    return claimed in {_netloc(host) for host in hosts if host.strip()}


def _netloc(value: str) -> str:
    netloc = value.strip().lower()
    for port in _DEFAULT_PORTS:
        netloc = netloc.removesuffix(port)
    return netloc
