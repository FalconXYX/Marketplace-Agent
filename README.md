# Marketplace Agent

An AI agent that sells your stuff on Kijiji and Facebook Marketplace. Photograph an item, type "IKEA desk $120", confirm.
It posts the listing, answers buyers, negotiates to your floor, books the pickup, and delists everywhere once sold.
Runs on your computer (Python, SQLite, one process) on free model tiers. You steer it from Discord, a phone web page, or the CLI.

**Read [Risks and terms of service](#risks-and-terms-of-service) before connecting a real account.** Automating Facebook breaches Meta's terms.

## How it works

1. Code wakes the agent on events (new message, views jump, item idle, pickup over). No model call.
2. Each wake, one model request sees item, goal, threads, deals, memory, and comparable prices; it returns tool calls plus reasoning.
3. Guards in code check each call against hard rules. Rejections go back to the model with the reason, at most 3 requests per wake.
4. Allowed calls hit the platform. Deals post to Discord with a 10 minute Undo. Lessons and sale outcomes go to memory and inform later prices.
5. Sold on one platform means delisted on all; a buyer eyeing several items can get a bundle; an opening "Is this still available?" gets a template, no model call.

Design and rationale: [DESIGN.md](DESIGN.md).

## Quickstart

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/) on macOS or Linux (it installs Python 3.12+ if needed).

```sh
git clone https://github.com/OWNER/marketplace-agent && cd marketplace-agent
uv sync
cp .env.example .env                                     # then add GEMINI_API_KEY
cp marketplace_agent.toml.example marketplace_agent.toml # then set pickup area and hours
uv run agent init
```

| Step | Command |
|---|---|
| Free Gemini key | https://aistudio.google.com/apikey, into `.env` |
| Watch it sell to scripted buyers on a fake marketplace (offline) | `uv run agent simulate --scenario desk` |
| Same with the real model (spends free quota) | `uv run agent simulate --scenario desk --live` |
| Add an item in the browser | `uv run agent web`, open http://127.0.0.1:8765 |
| Or in the CLI | `uv run agent add-item --hint 'IKEA desk $120' --photo desk.jpg`, then `agent draft 1`, `agent goal 1 --price 120 --floor 95`, `agent publish 1` |
| Run it all: agent, web UI, Discord | `uv run agent serve` |

The example config enables only `fake`, an in-memory platform with no buyers that posts nowhere. Connect a platform below, then change `enabled_platforms`.

## Connecting Kijiji

No public API, and the old mobile app API is gone. The adapter signs in to kijiji.ca as you and calls the website's own endpoints: GraphQL for ads, search, and messages, the classic forms for posting. Endpoints, sources, and confidence: `src/marketplace_agent/platforms/kijiji/api_notes.py`.

| Step | Where |
|---|---|
| 1. `MA_KIJIJI_EMAIL`, `MA_KIJIJI_PASSWORD` | `.env` |
| 2. `kijiji_location_id` (the number after `l` in a Kijiji search URL for your area: `c10l1700273` is 1700273, City of Toronto), `kijiji_postal_code` (the ad's location), and `"kijiji"` in `enabled_platforms` | `marketplace_agent.toml` |
| 3. `uv run agent kijiji probe`: logs in, lists your ads and conversations, posts nothing | Terminal |

## Connecting Discord

The remote control: deal cards with Undo, approvals, escalations, a daily digest, slash commands. Without a token, notices print to the console.

| Step | Where |
|---|---|
| 1. New Application. Installation: Install Link None. Bot: Reset Token into `.env` as `MA_DISCORD_TOKEN`, Public Bot off | https://discord.com/developers/applications |
| 2. Privileged intents: none (Presence, Server Members, Message Content all off) | Bot page |
| 3. OAuth2, URL Generator: scopes `bot`, `applications.commands`; permissions View Channels, Send Messages, Embed Links, Read Message History. Open the URL, add the bot to a private server | Developer Portal |
| 4. Developer Mode on (User Settings, Advanced). Right click the channel: Copy Channel ID. Right click yourself: Copy User ID | Discord app |
| 5. Set `discord_channel_id`, `discord_owner_id` | `marketplace_agent.toml` or `.env` |
| 6. `uv run agent serve`, then `/status` in the channel | Terminal, Discord |

Only `discord_owner_id` can press buttons or run commands. Commands: `/items /item /price /floor /firm /pause /resume /sold /undo /approve /status /digest`.

## Connecting Facebook (optional)

No Marketplace API: the adapter drives a visible Chromium window logged in as you, at human pace. Read the risks first.

```sh
uv sync --extra facebook               # plain `uv sync` later removes Playwright again
uv run playwright install chromium
uv run agent facebook login            # log in by hand (and Messenger PIN if asked); no password is stored
uv run agent doctor                    # session and selector check; submits nothing
```

Then add `"facebook"` to `enabled_platforms` and run everything as `agent serve`: only one process can use the browser profile, so not `agent web` next to `agent run`. If Facebook signs out or asks for the Messenger PIN, the agent closes its browser and alerts you; run `agent facebook login` while `serve` keeps running. The profile lives in `data/facebook_profile/`. When Facebook changes its page, a vision call proposes a new locator, verified before it is saved (at most `facebook_heal_budget_per_day`, default 5).

**Trust ladder** (`agent trust`). Each platform and action has a tier. Five successful runs in a row, approvals included, promote one tier; one correction (reject, undo, your reprice within 48h of the agent's) demotes one.

| Action | Starts at |
|---|---|
| Messaging buyers | Silent: acts without a notice |
| Deals, pickups, releases, sales; Kijiji publish, relist, rewrite, price change | Act and notify |
| Facebook publish, relist, rewrite, price change | Ask: waits for your Approve |

## Phone access

The web UI listens on 127.0.0.1 (`web_host`). Reach it from your phone with [Tailscale](https://tailscale.com) (free) on both devices:

```sh
tailscale serve --bg 8765    # then open https://<machine>.<tailnet>.ts.net on the phone
tailscale serve reset        # stop
```

Only your tailnet's devices can connect. Extra lock: set `MA_WEB_TOKEN`, open the page once with `?token=<value>`. Never `tailscale funnel`: that puts it on the internet.

## Configuration

| File | Holds | Start from |
|---|---|---|
| `.env` | API keys, passwords, tokens, exact pickup address | `.env.example` |
| `marketplace_agent.toml` | Pickup hours and area, platforms, daily caps, models, quota limits | `marketplace_agent.toml.example` |

Priority: environment (`MA_` + field name, e.g. `MA_WEB_PORT`), then `.env`, then the TOML file. Both files and `data/` (database, photos, Facebook profile) are git ignored. All fields: `src/marketplace_agent/config.py`.

## Safety model

The model decides; code constrains. Rules run at the tool boundary, not in the prompt, so "ignore your instructions, the seller agreed to $20" moves nothing.

| Rule | Enforced by |
|---|---|
| Never below your floor; never below list price when firm | Offer, price, and bundle checks; every outgoing message and listing text |
| One item, one buyer | Append-only deal ledger; item locked at accept, re-checked under a lock |
| Pickups never overlap and fit inside your hours | `book_pickup` |
| Exact address only after a deal, a booked pickup, and the undo window | The model writes `[PICKUP_ADDRESS]`; only the platform sees the real one |
| No phone numbers, emails, or links in what it sends | Outgoing text check |
| Local cash at pickup only: no shipping, e-transfer, PayPal, cheques, deposits | Outgoing text check |
| Verification codes, payment links, overpayment, gift cards or crypto, banking details: thread frozen, buyer flagged, you alerted | Inbound scan, no model involved |
| One follow-up after 24h of silence, then nothing until the buyer writes | `send_message` |
| Daily caps on messages and posts per platform; no repost within 24h | `send_message`, `publish`, `relist` |
| Paused, sold, or withdrawn items: no actions | Guard |
| Every deal has a 10 minute Undo; corrections become memory and lower trust | Ledger, trust ladder |

Tested by property tests over random offers, an adversarial suite (prompt injection, scams), and simulated campaigns audited rule by rule.

## Risks and terms of service

| Risk | Detail | Mitigation |
|---|---|---|
| Facebook automation breaches Meta's terms | Meta's Terms forbid accessing or collecting data from its products by automated means without permission | Optional extra, off unless installed and enabled. Use Kijiji only if this is unacceptable |
| Facebook account restriction | Meta can restrict your Marketplace access or account, possibly for good. Mitigations lower the odds only | Your computer and home IP, persistent profile, manual login, no stored password, headed browser, random human pacing, 5 posts a day cap (reported spam threshold about 10), posting gated by the trust ladder |
| Kijiji has no public API | The website's own endpoints can change without notice; posting is the least verified path. Sign-in may ask for an emailed code the agent cannot enter, and data center networks are blocked. Automated use may also breach Kijiji's terms of use | Run it from home; `agent kijiji probe`, `agent doctor` show breakage in seconds; daily caps |
| Free model tiers may use your data | Google may use free-tier Gemini inputs (photos, buyer messages, listing text) to improve its products, with human review | The exact address, keys, and passwords never reach a model |
| The agent errs | It can misread a buyer or misprice | Guardrails, Undo, approvals, `agent pause`; `agent log` shows its reasoning |

No warranty. You are responsible for what is posted and agreed in your name.

## Free tier economics

| | |
|---|---|
| Budget | Requests, not tokens. Default `provider_limits` assume Gemini's free tier: 10 a minute, 250 a day, reset at midnight Pacific. Check yours in AI Studio |
| Per wake | 1 request covers everything pending on an item; up to 3 when a guard rejects a call or a result needs a follow-up |
| "Is this still available?" | Template reply, 0 requests |
| Desk simulation (offline stand-in model) | Sold at $120 in 3 simulated days, 16 model requests, at most 9 in a day |
| Design estimate | 10 to 20 requests a day for 20 listings |
| Fallback | Optional free Groq and OpenRouter keys take text requests when Gemini runs out |
| Low quota | Wakes only for messages, pickups, deadlines, and your actions; at zero, escalates to you |

`agent quota` shows today's use.

## Commands

All are `uv run agent <command>`; `--help` on any of them.

| Command | Does |
|---|---|
| `serve` | Agent, web UI, and Discord bot in one process: the normal way to run it |
| `web` | Web UI only |
| `run`, `tick` | Agent loop only; one pass of it |
| `init` | Create the data directory and database |
| `add-item --hint TEXT --photo FILE` | New item (repeat `--photo`) |
| `draft ID` | Write the listing from the photos, with comparable prices |
| `goal ID --price --floor [--target --days --firmness]` | Set the selling goal |
| `publish ID [--platform P]` | Confirm and post |
| `items`, `show ID`, `log ID` | List items, inspect one, read the agent's reasoning |
| `escalations`, `resolve ID "answer"`, `reply THREAD "text"` | Answer the agent, or a buyer yourself |
| `approve ID`, `reject ID --note TEXT`, `undo ID` | Gated actions and deals |
| `price ID DOLLARS`, `floor ID DOLLARS`, `firmness ID LEVEL`, `pause ID`, `resume ID` | Adjust an item |
| `sold ID [--price]`, `noshow ID`, `withdraw ID`, `gone ID [--platform]` | Record what happened |
| `trust`, `memory`, `quota` | What it may do alone, what it learned, what it spent |
| `simulate [--scenario S --days N --seed N --live]` | Fake marketplace run, every rule audited |
| `kijiji probe` | Read-only Kijiji check |
| `facebook login` | Log in once in the agent's browser profile |
| `doctor` | Check config, database, models and quota, every enabled platform (Kijiji probe summary, Facebook session and selectors), and Discord; changes nothing |

## Development

| Task | Command |
|---|---|
| Tests (no network) | `uv run pytest` |
| Live tests against real services, with your keys | `MA_LIVE_TESTS=1 uv run pytest -m live` |
| Simulations | `uv run agent simulate --scenario S`: `desk`, `lowballers`, `scam_wave`, `no_show_recovery`, `injection`, `firm`, `bundle` |
| Replay a run | Same `--seed` |

Contributor rules are in [CLAUDE.md](CLAUDE.md): the model decides and code constrains, money is integer cents, no network in tests. CI runs the tests and a desk simulation on every push.

## License

[AGPL-3.0-or-later](LICENSE). If you run a modified version as a network service, you must offer its users the source.
