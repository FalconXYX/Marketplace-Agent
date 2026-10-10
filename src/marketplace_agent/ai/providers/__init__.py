"""Model providers. Shared helpers only; each provider module imports its own SDK."""


def is_header_safe(api_key: str) -> bool:
    """Printable ASCII with no spaces. Anything else fails inside the HTTP client, whose error
    message shows the header escaped ("SECRET\\r"), where redaction cannot find the key."""
    return api_key != "" and api_key.isascii() and api_key.isprintable() and " " not in api_key
