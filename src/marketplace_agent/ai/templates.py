"""Canned replies sent by the pattern cache with no model call."""


def availability_reply(*, committed: bool, pickup_area: str) -> str:
    """Never mentions price or an exact address; pickup_area must be a cross street."""
    if committed:
        return (
            "Hi! It's pending with someone else right now. "
            "Want me to message you if it falls through?"
        )
    area = pickup_area.strip()
    pickup = f" Pickup is near {area}." if area else ""
    return f"Hi! Yes, it's still available.{pickup}"
