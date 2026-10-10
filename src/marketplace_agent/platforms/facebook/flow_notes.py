"""Facebook Marketplace flows as of October 2026. Constants only; the adapter holds the logic.

There is no API. Everything runs in the owner's own logged in browser profile, which breaches
Meta's terms (see DESIGN.md, Constraints). Locators live in selectors.json, never here.

## Sources (read October 2026)

| Source | Last change | What it showed |
|---|---|---|
| github.com/BoPeng/ai-marketplace-monitor, src/ai_marketplace_monitor/facebook.py | 2026 | Signed in = `c_user` cookie and a path outside /login, /checkpoint, /two_step_verification, /recover; cookie banner "Allow all cookies" |
| github.com/HabibUrRehmanBhattii/fb-automation, main.js | 2026-09 | Create form: labels Title, Price, Description, Category; hidden input[type=file]; combobox "Condition" with role=option entries; buttons Next then Publish; skip buttons whose aria-label mentions image, photo, carousel; aria-disabled on Next |
| github.com/earlonrails/fb-marketplace-skills, src/scraper/listings.ts, src/editor/listingUpdater.ts | 2026-09 | Your listings at /marketplace/you/selling, cards hold "Mark as sold", "N clicks on listing", "Active · Listed on"; edit form at /marketplace/edit/?listing_id=ID with inputs aria-label Title, Price, textarea Description, button Update or Save |
| github.com/matthewmiglio/openclaw-facebook-marketplace, src/browser.py | 2026-10-07 | Messenger at /messages/t/, grid aria-label "Chats", one "Marketplace" row that opens the Marketplace folder, links a[href*="/messages/t/"]; bubbles labeled "Enter, Message sent 4:01 PM by Lucas: text" ("by You" for the owner); heading "Conversation titled Buyer · Listing"; a[aria-label="See details"] links the listing; "You've reached the messaging limit" popup; headless needs channel="chromium" to read a persistent profile |
| github.com/rjgustilo87/messenger-marketplace-autoresponder, DOM_STRUCTURE.md | 2026-02 | Input is div[role=textbox][contenteditable=true]; Enter sends; unread rows carry aria-label "... unread" |
| about.fb.com, "Introducing Seller" | 2026-07 | Full insights (views, clicks, saves) moved to the iOS Seller app; the web keeps clicks per listing |

## Flows

| Action | URL | Steps |
|---|---|---|
| Create | CREATE_URL | upload photos to the hidden file input, Title, Price (whole dollars), Category (virtualized list: open it, page down until the option renders), Condition (combobox), Description, Next, Publish on "List in more places", dismiss a boost prompt with "Not now" |
| Find the new id | SELLING_URL | card ids read before publishing; the new card is one not among them with the title as a line; id from a[href*="/marketplace/item/ID"] |
| Edit | EDIT_URL | same field labels as create, then Update (older builds: Save) |
| Mark sold | ITEM_URL | "Mark as sold"; a dialog may ask who bought it: Skip / Someone else / Mark as sold |
| Delete | ITEM_URL | More options, "Delete listing", confirm Delete in the dialog |
| Stats | SELLING_URL | "N clicks on listing" per card; saves and views only in the Seller app |
| Inbox | INBOX_URL | click the "Marketplace" row; one link per buyer and listing, newest first |
| Read | THREAD_URL | bubbles by aria-label, heading for buyer and listing, See details for the listing id |
| Send | THREAD_URL | focus the textbox, type (Shift+Enter between lines, Enter alone sends), Enter or "Press enter to send" |

## Gotchas

| Gotcha | Handling |
|---|---|
| Class names are generated (x1yc5d2u ...) and change weekly | Locators use roles, labels, and visible text; css only for attributes |
| UI language follows the account, not the browser | Labels assume English; other languages need overrides or healing |
| Bubble times move: "4:01 PM", then "Mon 4:01 PM", then "Oct 3, 2026, 4:01 PM" | Message ids hash sender, text, and repeat count, never the time label |
| "Enter your PIN to restore your chats" (end-to-end encrypted history) | Needs the owner: treated as signed out |
| Messenger rate limit on new chats | PlatformError, nothing retried |
| Headless profile reads | channel="chromium" (full build), never the headless shell |
| One browser per profile directory | A second process gets "profile in use"; run browser work in one process |
| Location field is prefilled from the profile | Left as is; ListingPayload.location is not typed in |
| Selling about 10 or more listings a day is the reported spam threshold | daily_publish_cap.facebook defaults to 5 |
"""

BASE_URL = "https://www.facebook.com"
HOME_URL = f"{BASE_URL}/"
LOGIN_URL = f"{BASE_URL}/login"
CREATE_URL = f"{BASE_URL}/marketplace/create/item"
EDIT_URL = f"{BASE_URL}/marketplace/edit/?listing_id={{listing_id}}"
ITEM_URL = f"{BASE_URL}/marketplace/item/{{listing_id}}/"
SELLING_URL = f"{BASE_URL}/marketplace/you/selling"
INBOX_URL = f"{BASE_URL}/messages/t/"
THREAD_URL = f"{BASE_URL}/messages/t/{{thread_id}}/"

SESSION_COOKIE = "c_user"
LOGIN_PATHS = ("/login", "/checkpoint", "/two_step_verification", "/recover")

LISTING_ID_PATTERN = r"/marketplace/item/(\d+)"
THREAD_ID_PATTERN = r"/messages/t/(\d+)"
PROFILE_ID_PATTERNS = (r"/marketplace/profile/(\d+)", r"profile\.php\?id=(\d+)")
BUBBLE_PATTERN = r"Message sent (?P<when>.+?) by (?P<sender>.+?): (?P<text>.*)"
OWN_SENDER = "You"
HEADING_PREFIX = "Conversation titled "
TITLE_SEPARATOR = " · "
CLICKS_PATTERN = r"([\d,]+)\s+clicks?\s+on\s+listing"
SAVES_PATTERN = r"([\d,]+)\s+saves?\b"
SOLD_CARD_PATTERN = r"Mark as available|Mark as in stock|\bSold\s·"

MAX_PHOTOS = 10

# Condition values from drafting -> option text on the form (substring, except "New")
CONDITION_OPTIONS: dict[str, tuple[str, bool]] = {
    "new": ("New", True),
    "like_new": ("Like New", False),
    "good": ("Good", False),
    "fair": ("Fair", False),
    "poor": ("Fair", False),
}

# Keywords (whole words, from the item's category, then its title) -> category option text
CATEGORY_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("video game", "video games", "nintendo", "playstation", "xbox"), "Video Games"),
    (("phone", "phones", "iphone", "smartphone"), "Mobile Phones"),
    (
        (
            "laptop",
            "computer",
            "monitor",
            "tv",
            "television",
            "camera",
            "electronics",
            "headphones",
            "speaker",
            "tablet",
        ),
        "Electronics",
    ),
    (
        (
            "sofa",
            "couch",
            "desk",
            "chair",
            "table",
            "dresser",
            "bed",
            "shelf",
            "bookcase",
            "cabinet",
            "furniture",
            "wardrobe",
            "nightstand",
        ),
        "Furniture",
    ),
    (("drill", "saw", "tool", "tools", "wrench", "sander"), "Tools"),
    (
        (
            "fridge",
            "washer",
            "dryer",
            "microwave",
            "dishwasher",
            "appliance",
            "appliances",
            "stove",
            "oven",
            "vacuum",
        ),
        "Appliances",
    ),
    (("book", "books", "dvd", "vinyl", "movie", "movies", "music"), "Books"),
    (("bike", "bicycle", "bicycles"), "Bicycles"),
    (("guitar", "piano", "drum", "drums", "instrument", "instruments"), "Musical Instruments"),
    (("toy", "toys", "lego", "puzzle", "board game"), "Toys"),
    (("garden", "plant", "patio", "lawn", "mower", "bbq", "grill"), "Garden"),
    (("baby", "stroller", "crib", "kids"), "Baby"),
    (("dress", "shirt", "jacket", "shoes", "clothing", "coat", "jeans"), "Clothing"),
    (("ring", "necklace", "watch", "jewelry", "jewellery"), "Jewelry"),
    (("sport", "sports", "fitness", "weights", "treadmill", "camping", "ski", "golf"), "Sports"),
    (("tire", "tires", "auto parts", "car parts"), "Auto Parts"),
    (("art", "craft", "crafts", "painting"), "Arts & Crafts"),
    (("antique", "antiques", "collectible", "collectibles", "vintage"), "Antiques"),
    (("pet", "pets", "dog", "cat", "aquarium"), "Pet Supplies"),
    (("bag", "bags", "luggage", "suitcase", "backpack"), "Luggage"),
    (("kitchen", "household", "home", "decor", "lamp", "rug", "mirror"), "Household"),
)
FALLBACK_CATEGORY = "Miscellaneous"
