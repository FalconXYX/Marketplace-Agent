"""ListingPayload to a Kijiji ad, and Kijiji conversations and search hits back to platform types.

Category: attributes["kijiji_category_id"] when set (any Kijiji category id), else the first
CATEGORY_RULES keyword found in the payload's category and title, else Buy & Sell > Other.
"""
import re
from collections.abc import Sequence

from marketplace_agent.platforms.base import (
    ListingPayload,
    RemoteMessage,
    RemoteThread,
    SearchResult,
)
from marketplace_agent.platforms.kijiji import api_notes as api
from marketplace_agent.platforms.kijiji.client import (
    KijijiAd,
    KijijiConversation,
    KijijiMessage,
    KijijiSearchHit,
)

CATEGORY_OVERRIDE = "kijiji_category_id"

# First match wins, so specific words come before general ones ("tv stand" before "tv").
CATEGORY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("tv stand", "tv unit", "entertainment unit", "media console"), "furniture_tv_units"),
    (("coffee table", "side table", "end table"), "furniture_coffee_tables"),
    (("dining table", "dining set", "kitchen table"), "furniture_dining"),
    (("desk",), "furniture_desks"),
    (("sofa", "couch", "futon", "sectional", "loveseat"), "furniture_couches"),
    (("chair", "recliner", "stool"), "furniture_chairs"),
    (("mattress", "bed frame", "bunk bed", "bed"), "furniture_beds"),
    (("dresser", "wardrobe", "armoire", "nightstand"), "furniture_dressers"),
    (("bookcase", "bookshelf", "shelving", "shelf", "shelves"), "furniture_shelving"),
    (("table",), "furniture_other_tables"),
    (("furniture",), "furniture_other"),
    (("ps5", "playstation 5"), "games_ps5"),
    (("ps4", "playstation 4"), "games_ps4"),
    (("xbox",), "games_xbox"),
    (("nintendo switch",), "games_switch"),
    (("video game", "console"), "games_other"),
    (("laptop", "macbook", "chromebook"), "computers_laptop"),
    (("ipad", "tablet"), "computers_tablet"),
    (("desktop", "gaming pc"), "computers_desktop"),
    (("monitor",), "computer_monitors"),
    (("mouse", "webcam"), "computer_input"),
    (("printer", "scanner"), "computer_printers"),
    (("computer",), "computers_other"),
    (("iphone", "smartphone", "cell phone", "phone"), "phones"),
    (("tv", "television"), "tvs"),
    (("camera", "camcorder", "lens"), "cameras"),
    (("headphone", "earbud", "headset"), "audio_headphones"),
    (("speaker",), "audio_speakers"),
    (("stereo", "receiver", "turntable", "home theatre", "home theater"), "audio_stereo"),
    (("audio",), "audio_other"),
    (("microwave",), "appliances_microwaves"),
    (("fridge", "refrigerator"), "appliances_fridges"),
    (("washer", "dryer"), "appliances_laundry"),
    (("vacuum",), "appliances_vacuums"),
    (("coffee maker", "espresso", "keurig"), "appliances_coffee"),
    (("stove", "oven"), "appliances_stoves"),
    (("appliance",), "appliances_other"),
    (("electronic",), "electronics_general"),
    (("lamp", "lighting", "chandelier"), "home_lighting"),
    (("rug", "carpet"), "home_rugs"),
    (("bedding", "duvet", "comforter"), "home_bedding"),
    (("cookware", "dishes", "cutlery", "kitchen"), "home_kitchen"),
    (("mirror", "decor"), "home_decor"),
    (("bbq", "barbecue", "grill"), "outdoor_bbq"),
    (("lawn mower", "lawnmower", "leaf blower"), "outdoor_mowers"),
    (("patio",), "outdoor_patio"),
    (("garden", "outdoor"), "outdoor_other"),
    (("drill", "saw", "sander", "grinder", "power tool"), "tools_power"),
    (("ladder",), "tools_ladders"),
    (("wrench", "hammer", "screwdriver", "hand tool"), "tools_hand"),
    (("tool",), "tools_other"),
    (("guitar",), "music_guitars"),
    (("piano", "synthesizer"), "music_pianos"),
    (("amp", "amplifier", "pedal"), "music_amps"),
    (("drum",), "music_drums"),
    (("instrument", "violin", "ukulele", "trumpet", "saxophone"), "music_other"),
    (("ebike", "e-bike", "electric bike"), "bikes_ebike"),
    (("bike", "bicycle"), "bikes_other"),
    (("treadmill", "dumbbell", "kettlebell", "weights", "exercise", "elliptical"),
     "sports_exercise"),
    (("golf",), "sports_golf"),
    (("hockey",), "sports_hockey"),
    (("ski",), "sports_ski"),
    (("tent", "camping", "fishing", "kayak"), "sports_camping"),
    (("sport",), "sports_other"),
    (("stroller", "car seat"), "baby_strollers"),
    (("crib", "bassinet"), "baby_cribs"),
    (("baby",), "baby_other"),
    (("toy", "lego", "board game", "puzzle"), "toys_games"),
    (("book", "novel", "textbook"), "books_other"),
    (("cd", "dvd", "blu-ray", "vinyl"), "cds_dvds"),
    (("jacket", "coat", "shirt", "dress", "jeans", "shoe", "boot", "sneaker", "clothing",
      "clothes"), "clothing_other"),
    (("jewellery", "jewelry", "necklace", "bracelet", "watch"), "jewellery"),
    (("luggage", "suitcase", "backpack", "bag"), "bags_luggage"),
    (("painting", "collectible", "antique", "art"), "arts_collectibles"),
)


def to_ad(listing: ListingPayload, *, location_id: int | None, postal_code: str | None) -> KijijiAd:
    return KijijiAd(
        title=_title(listing.title),
        description=listing.description,
        price=dollars(listing.price_cents),
        category_id=category_id(listing),
        condition=api.CONDITIONS.get(listing.condition or ""),
        postal_code=postal_code,
        location_id=location_id,
    )


def category_id(listing: ListingPayload) -> int:
    override = listing.attributes.get(CATEGORY_OVERRIDE, "").strip()
    if override.isdigit():
        return int(override)
    text = f"{listing.category or ''} {listing.title}".lower()
    for words, key in CATEGORY_RULES:
        if any(re.search(rf"\b{re.escape(word)}(?:e?s)?\b", text) for word in words):
            return api.CATEGORY_IDS[key]
    return api.DEFAULT_CATEGORY_ID


def dollars(cents: int) -> str:
    """Whole dollars stay whole ("120"); anything else keeps its cents ("120.50")."""
    whole, part = divmod(cents, 100)
    return str(whole) if part == 0 else f"{whole}.{part:02d}"


def to_thread(conversation: KijijiConversation, messages: Sequence[KijijiMessage]) -> RemoteThread:
    return RemoteThread(
        remote_id=conversation.id,
        listing_remote_id=conversation.listing_id,
        buyer_handle=conversation.counterparty_id or f"conversation-{conversation.id}",
        buyer_name=conversation.counterparty_name,
        messages=[
            RemoteMessage(m.id, m.body, m.sent_at, from_buyer=m.inbound) for m in messages
        ],
    )


def to_search_result(hit: KijijiSearchHit) -> SearchResult | None:
    """None for ads without a fixed price (free, swap, "please contact"): they are not comps.
    The link is the view URL publish returns, not the hit's SEO one, so comps can spot our own ads
    by Listing.url."""
    if hit.price_cents is None:
        return None
    url = api.VIEW_URL.format(ad_id=hit.id)
    return SearchResult(hit.title, hit.price_cents, url, hit.posted_at)


def search_url(query: str, location_id: int | None) -> str | None:
    """Buy & Sell search URL for the query near location_id (all of Canada when None).
    None when the query has no letters or digits."""
    keywords = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    if not keywords:
        return None
    return api.SEARCH_URL.format(keywords=keywords, location_id=location_id or 0)


def _title(title: str) -> str:
    title = " ".join(title.split())
    if len(title) <= api.TITLE_MAX_CHARS:
        return title
    head, space, _ = title[: api.TITLE_MAX_CHARS + 1].rpartition(" ")
    return head if space and head else title[: api.TITLE_MAX_CHARS]
