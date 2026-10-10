"""Kijiji endpoints as of 2026-10-10. Research notes and constants only; no logic.

## What changed

| When | Change | Source |
|---|---|---|
| 2024-04/05 | The mobile "mingle" XML API (mingle.kijiji.ca/api, X-ECG-Authorization-User) answers 502. The app moved to Flutter with certificate pinning, host api.ca-kijiji-production.classifiedscloud.io, undocumented | [kijiji-manager #67](https://github.com/jackm/kijiji-manager/issues/67), [kijiji-scraper #73](https://github.com/mwpenny/kijiji-scraper/issues/73) |
| 2024-2025 | Web login moved to id.kijiji.ca (Apereo CAS acting as OIDC provider "CIS"). Cookie-replay tools broke: the `kj-at` cookie rotates about every 10 minutes | [Kijiji-Repost-Headless #228, #229](https://github.com/ArthurG/Kijiji-Repost-Headless/issues/228) |
| 2025-2026 | Every working poster found is browser automation (Playwright, Selenium). Sign-in often asks for a code emailed to the account | [KaranChawlaD/bots](https://github.com/KaranChawlaD/bots) (2026-09), [IanDarwin/eSelling](https://github.com/IanDarwin/eSelling) (2026-05), [Programmer484/kijiji-automation](https://github.com/Programmer484/kijiji-automation) (2026-01) |
| now | www.kijiji.ca is a Next.js app (v1.94.7) on NextAuth with one OAuth provider, `cis`. Its Apollo client posts to /anvil/api with `credentials: "include"` and `Apollo-Require-Preflight: true`. Search, listing detail (with view counts), conversations, and replies go through that GraphQL endpoint | live bundle and anonymous calls on 2026-10-10; [fetchaller-mcp kijiji/api.py](https://github.com/Averyy/fetchaller-mcp/blob/main/src/fetchaller/kijiji/api.py) (2026) |
| now | Posting, editing, deleting, and My Ads are still the legacy server rendered app (p-post-ad.html, m-my-ads), outside the Next.js build. Its form field names (postAdForm.title, postAdForm.description, postAdForm.priceAmount) still match in 2026 | eSelling KijijiMarket.java (2026-05), [OakandRocks/kijiji-bot](https://github.com/OakandRocks/kijiji-bot) (2020) |

Confidence: high = seen working in code updated in 2026 or verified live on 2026-10-10;
medium = from the live web bundle or a standard (NextAuth, CAS) but not exercised with an account;
low = from code older than 2023 or inferred. KijijiClient's public reads (search, listing views,
a missing ad) passed live tests on 2026-10-10. `agent kijiji probe` checks every read path.

## Capabilities

| Capability | Endpoint | Method, payload | Response | Confidence |
|---|---|---|---|---|
| Login: CSRF | GET /api/auth/csrf | | `{"csrfToken"}` | high (live) |
| Login: start | POST /api/auth/signin/cis | form csrfToken, callbackUrl, json=true | `{"url": "https://id.kijiji.ca/oidc/authorize?..."}` | high (live) |
| Login: form | GET that url, redirects to id.kijiji.ca/login?service=... | | HTML `<form id="login" action="login">`: hidden execution, service, state, nonce, locale, kc_locale, scope, tmSessionId, _eventId=submit, geolocation; username, password, trustThisDevice | high (live, not submitted) |
| Login: submit | POST id.kijiji.ca/login | the form above, urlencoded | 302 chain to www.kijiji.ca/api/auth/callback/cis, which sets the session cookies. Wrong password: the form again. Risky sign-in: an extra step, usually an emailed code | medium |
| Session | GET /api/auth/session | | `{}` signed out; `{"user": {"sub", "name", "email", "error"?}, "expires"}` signed in. `user.sub` is the user id; `error == "RefreshAccessTokenError"` means sign in again | medium (bundle) |
| GraphQL transport | POST /anvil/api | JSON `{operationName, query, variables}`, session cookies, headers below | HTTP 200 even for errors: `errors[].extensions.code` UNAUTHENTICATED (statusCode 401) or NOT_FOUND_ERROR (404). A query the schema rejects: HTTP 400, GRAPHQL_VALIDATION_FAILED naming the field | high (live) |
| Listing, view count | query `listing(id, trackView: false)` | LISTING_QUERY | id, title, url, status, activationDate, `price {type amount}` in cents, `metrics {views}`. Missing ad: NOT_FOUND_ERROR and `listing: null` | high (live) |
| List my ads | query `profile(id: user.sub) { listings }` | PROFILE_LISTINGS_QUERY, sort DATE DESC | active public ads, no view counts (call listing per ad) | high for the query (live, anonymous); medium that user.sub is the profile id |
| Conversations | query `conversations(limit, offset, role: SELLER)` | CONVERSATIONS_QUERY | totalCount, items: id, lastMessageAt, counterparty {userId name}, listing {id title} | medium: passes live schema validation (anonymous call stops at UNAUTHENTICATED); never run signed in |
| Messages | query `conversation(id, markAsRead, limit, cursor)` | CONVERSATION_QUERY | `messages {nextCursor items {id body sentAt direction}}`, direction INBOUND or OUTBOUND relative to the viewer | medium: as above |
| Reply | mutation `sendMessage(conversationId, message)` | SEND_MESSAGE_MUTATION | `{messageId conversationId}`; the web app caches messageId as the ConversationMessage id, so ingestion dedupes our own replies | medium: as above |
| Search (comps) | query `searchResultsPageByUrl(input: {url, pagination})` | SEARCH_QUERY, url like /b-buy-sell/canada/ikea-desk/k0c10l1700273 | `results.mainListings[]`: id, title, url, activationDate, price {type amount}; type FIXED has amount in cents; CONTACT, GIVE_AWAY, SWAP_TRADE have none | high (live) |
| Categories | query `category(id)` | CATEGORY_QUERY | localizedName, categoryPaths, attributes with canonical values | high (live) |
| Locations | query `location(id)` | LOCATION_QUERY | `{id localizedName}`, null when unknown | high (live) |
| XSRF token | mutation `generateXsrf` | GENERATE_XSRF_MUTATION | `{xsrf}`; the web app sends it to logoutUser. Assumed to be the legacy `ca.kijiji.xsrf.token` | medium for the call (live), low for the legacy use |
| Post ad | GET /p-post-ad.html?categoryId=N, then POST the form to its action | field names in POST_FIELDS; replay every other field the form carries | redirect to the new ad; id in the URL, an adId parameter, or the kjrva cookie | low |
| Image upload | POST /p-upload-image.html | multipart `file`, XSRF token header | JSON with thumbnailUrl; the urls go comma separated into `images` | low (2020 flow; photos now live on media.kijiji.ca) |
| Edit ad | GET /p-edit-ad.html?adId=N, POST its form | same fields as posting | | low |
| Delete ad | POST /j-delete-ad.json | Action=DELETE_ADS, Mode=ACTIVE, needsRedirect=false, ads=[{"adId","reason":"PREFER_NOT_TO_SAY","otherReason":""}], ca.kijiji.xsrf.token | body contains OK | low (2020 flow; the 2025 UI still asks for a reason with "Prefer not to say") |

## Risks

| Risk | Effect |
|---|---|
| Akamai bot management on www.kijiji.ca | Datacenter IPs get 403. A home IP passes plain requests |
| ThreatMetrix (`tmSessionId`) and emailed sign-in codes | Login from a new HTTP client may need a code; the adapter cannot enter it |
| Legacy posting form drift | Posting fails loudly (missing field) rather than posting garbage |
"""

BASE_URL = "https://www.kijiji.ca"
GRAPHQL_PATH = "/anvil/api"
AUTH_CSRF_PATH = "/api/auth/csrf"
AUTH_SIGNIN_PATH = "/api/auth/signin/cis"
AUTH_SESSION_PATH = "/api/auth/session"
LOGIN_HOST = "id.kijiji.ca"
LOGIN_FORM_ID = "login"
LOGIN_USERNAME_FIELD = "username"
LOGIN_PASSWORD_FIELD = "password"
LOGIN_TRUST_FIELD = "trustThisDevice"
SESSION_REFRESH_ERROR = "RefreshAccessTokenError"
CONSUMER_LOGIN_PATH = "/consumer/login"  # where the legacy pages send a signed-out visitor

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
COMMON_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-CA,en;q=0.9"}
GRAPHQL_HEADERS = {
    "Accept": "application/json",
    "Origin": BASE_URL,
    "Referer": f"{BASE_URL}/",
    "Apollo-Require-Preflight": "true",
}

UNAUTHENTICATED = "UNAUTHENTICATED"
NOT_FOUND = "NOT_FOUND_ERROR"
GRAPHQL_ERROR_STATUS = 400  # schema validation failures; the body names the field

# Listing.status values that mean the ad is no longer up (enum from the v1.94.7 bundle).
GONE_STATUSES = frozenset({
    "ACTIVE_EXPIRED", "ADMIN_DELETED", "BLOCKED", "DELETED", "DELETED_EXPIRED", "EXPIRED",
    "SOLD", "SYSTEM_DELETED", "USER_DELETED",
})
FIXED_PRICE = "FIXED"
SELLER_ROLE = "SELLER"
INBOUND = "INBOUND"

LISTING_QUERY = """
query GetListing($listingId: ID!) {
  listing(id: $listingId, trackView: false) {
    id
    title
    url
    status
    activationDate
    price { type ... on StandardAmountPrice { amount } }
    metrics { views }
  }
}
"""

PROFILE_LISTINGS_QUERY = """
query GetPosterOtherAds($sort: ProfileSortInput!, $pagination: PaginationInputV2!, $profileId: ID!) {
  profile(id: $profileId) {
    id
    listings(sort: $sort, pagination: $pagination) {
      id
      title
      url
      price { type ... on StandardAmountPrice { amount } }
    }
  }
}
"""
PROFILE_SORT = {"by": "DATE", "direction": "DESC"}

CONVERSATIONS_QUERY = """
query GetConversations($limit: Int, $offset: Int, $role: ConversationRole) {
  conversations(limit: $limit, offset: $offset, role: $role) {
    totalCount
    items {
      id
      lastMessageAt
      counterparty { userId name }
      listing { id title }
    }
  }
}
"""

CONVERSATION_QUERY = """
query GetConversation($id: ID!, $markAsRead: Boolean, $cursor: String, $limit: Int) {
  conversation(id: $id, markAsRead: $markAsRead, cursor: $cursor, limit: $limit) {
    id
    messages {
      nextCursor
      items { id body sentAt direction }
    }
  }
}
"""

SEND_MESSAGE_MUTATION = """
mutation SendMessage($conversationId: ID!, $message: String!) {
  sendMessage(conversationId: $conversationId, message: $message) {
    messageId
    conversationId
  }
}
"""

SEARCH_QUERY = """
query GetSearchResultsPageByUrl($searchResultsByUrlInput: SearchResultsByUrlInput!, $pagination: PaginationInputV2!) {
  searchResultsPageByUrl(input: $searchResultsByUrlInput) {
    results {
      mainListings(pagination: $pagination) {
        id
        title
        url
        activationDate
        price { type ... on AmountPrice { amount } }
      }
    }
  }
}
"""
SEARCH_URL = BASE_URL + "/b-buy-sell/canada/{keywords}/k0c10l{location_id}"

CATEGORY_QUERY = """
query GetCategories($categoryId: Int!) {
  category(id: $categoryId) {
    id
    categoryPaths { id localizedName }
  }
}
"""

LOCATION_QUERY = """
query getLocationFromId($locationId: Int!) {
  location(id: $locationId) { id localizedName }
}
"""

GENERATE_XSRF_MUTATION = """
mutation generateXsrf {
  generateXsrf { xsrf }
}
"""

VIEW_URL = BASE_URL + "/v-view-details.html?adId={ad_id}"
POST_AD_PATH = "/p-post-ad.html"
EDIT_AD_PATH = "/p-edit-ad.html"
UPLOAD_IMAGE_PATH = "/p-upload-image.html"
DELETE_AD_PATH = "/j-delete-ad.json"
XSRF_FIELD = "ca.kijiji.xsrf.token"
XSRF_HEADER = "X-Ebay-Box-Token"
UPLOAD_URL_KEYS = ("thumbnailUrl", "url", "imageUrl")
DELETE_REASON = "PREFER_NOT_TO_SAY"
DELETE_OK_MARKER = "OK"
NEW_AD_COOKIE = "kjrva"

# Legacy post and edit form: field -> names to try, first present wins.
POST_FIELDS: dict[str, tuple[str, ...]] = {
    "title": ("postAdForm.title",),
    "description": ("postAdForm.description",),
    "price": ("postAdForm.priceAmount",),
    "price_type": ("postAdForm.priceType",),
    "postal_code": ("postAdForm.postalCode",),
    "location_id": ("postAdForm.locationId", "locationId"),
    "images": ("images", "postAdForm.images"),
    "condition": ("postAdForm.attributeMap[condition_s]", "condition_s"),
    "for_sale_by": ("postAdForm.attributeMap[forsaleby_s]", "forsaleby_s"),
    "terms": ("postAdForm.confirmedTerms",),
    "package": ("featuresForm.featurePackage",),
}
REQUIRED_POST_FIELDS = ("title", "description", "price")
FOR_SALE_BY_OWNER = "ownr"
BASIC_PACKAGE = "PKG_BASIC"
TITLE_MAX_CHARS = 64

# Leaf category ids under Buy & Sell (10), read from menuPrefetch on 2026-10-10.
DEFAULT_CATEGORY_ID = 26  # Buy & Sell > Other
CATEGORY_IDS = {
    "arts_collectibles": 12, "audio_headphones": 770, "audio_speakers": 14922002,
    "audio_stereo": 14922001, "audio_other": 771, "baby_cribs": 269, "baby_strollers": 270,
    "baby_toys": 256, "baby_other": 254, "bags_luggage": 931, "bikes_ebike": 14654001,
    "bikes_other": 651, "books_other": 14970007, "cameras": 103, "cds_dvds": 104,
    "clothing_other": 275, "computers_desktop": 772, "computers_laptop": 773,
    "computers_tablet": 776, "computers_other": 29324001, "computer_monitors": 782,
    "computer_input": 781, "computer_printers": 784, "electronics_general": 15,
    "free_stuff": 17220001, "furniture_beds": 246, "furniture_shelving": 249,
    "furniture_chairs": 245, "furniture_coffee_tables": 241, "furniture_couches": 238,
    "furniture_desks": 239, "furniture_dining": 243, "furniture_dressers": 247,
    "furniture_other_tables": 244, "furniture_tv_units": 242, "furniture_other": 236,
    "appliances_coffee": 689, "appliances_fridges": 696, "appliances_microwaves": 694,
    "appliances_stoves": 697, "appliances_vacuums": 699, "appliances_laundry": 700,
    "appliances_other": 701, "home_bedding": 719, "home_decor": 720, "home_lighting": 721,
    "home_kitchen": 722, "home_rugs": 723, "home_other": 725, "outdoor_bbq": 678,
    "outdoor_mowers": 682, "outdoor_patio": 686, "outdoor_other": 726, "jewellery": 133,
    "music_amps": 610, "music_drums": 612, "music_guitars": 613, "music_pianos": 614,
    "music_other": 618, "phones": 760, "sports_exercise": 655, "sports_camping": 656,
    "sports_golf": 658, "sports_hockey": 659, "sports_ski": 664, "sports_other": 669,
    "tools_hand": 702, "tools_power": 703, "tools_ladders": 705, "tools_other": 715,
    "toys_games": 108, "tvs": 15093002, "games_ps5": 39730002, "games_ps4": 792,
    "games_xbox": 39730001, "games_switch": 33035001, "games_other": 625,
}

# Condition attribute (canonical values, live on category 239). "poor" has no match there.
CONDITIONS = {
    "new": "new", "like_new": "usedlikenew", "good": "usedgood", "fair": "usedfair",
    "poor": "usedfair",
}
