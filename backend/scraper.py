"""
Public data fetcher — four sources:
  1. Reddit        (posts + top comments via JSON API)
  2. Apple App Store RSS
  3. Google Play Store  (google-play-scraper)
  4. Hacker News        (Algolia API — no auth, no rate limits)

Fully dynamic — works for any company via CompanyContext.
Falls back to embedded Duolingo seed data when analysing Duolingo.
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Tuple

import aiohttp

from backend.models import CompanyContext, FeedbackBundle, FeedbackItem

logger = logging.getLogger(__name__)

REDDIT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

_executor = ThreadPoolExecutor(max_workers=2)


# ===========================================================================
# Public entry point
# ===========================================================================

async def fetch_bundle(
    company: CompanyContext,
    progress_cb=None,
) -> FeedbackBundle:
    """Fetch feedback from all available sources for any company."""

    async def cb(msg: str):
        if progress_cb:
            await progress_cb(msg)

    subreddit    = company.subreddit    or _guess_subreddit(company.company_name)
    app_store_id = company.app_store_id or await _find_app_store_id(company.company_name)
    play_id      = _guess_play_id(company.company_name)

    timeout   = aiohttp.ClientTimeout(total=25, connect=8)
    connector = aiohttp.TCPConnector(limit=10, ssl=False)

    reddit_items:      List[FeedbackItem] = []
    app_store_items:   List[FeedbackItem] = []
    google_play_items: List[FeedbackItem] = []
    hn_items:          List[FeedbackItem] = []

    async with aiohttp.ClientSession(
        headers=REDDIT_HEADERS, connector=connector, timeout=timeout
    ) as session:

        # ── 1. Reddit ──────────────────────────────────────────────────────
        await cb(f"Fetching Reddit posts from r/{subreddit}...")
        try:
            reddit_items = await _fetch_reddit(session, subreddit, cb)
        except Exception as e:
            logger.warning(f"Reddit fetch failed: {e}")

        # ── 2. Apple App Store ─────────────────────────────────────────────
        if app_store_id:
            await cb("Fetching App Store reviews...")
            try:
                app_store_items = await _fetch_app_store(session, app_store_id)
            except Exception as e:
                logger.warning(f"App Store fetch failed: {e}")

        # ── 3. Google Play Store ───────────────────────────────────────────
        if play_id:
            await cb("Fetching Google Play reviews...")
            try:
                google_play_items = await _fetch_google_play(play_id)
            except Exception as e:
                logger.warning(f"Google Play fetch failed: {e}")

        # ── 4. Hacker News ─────────────────────────────────────────────────
        await cb("Fetching Hacker News discussions...")
        try:
            hn_items = await _fetch_hacker_news(session, company.company_name)
        except Exception as e:
            logger.warning(f"HN fetch failed: {e}")

    # ── Seed data (Duolingo only) ──────────────────────────────────────────
    seed_items: List[FeedbackItem] = []
    if company.company_name.strip().lower() in ("duolingo", "duolingo inc"):
        await cb("Loading seed feedback dataset...")
        seed_items = _get_duolingo_seed_data()

    all_items = reddit_items + app_store_items + google_play_items + hn_items + seed_items

    await cb(
        f"Data ready: {len(reddit_items)} Reddit · "
        f"{len(app_store_items)} App Store · "
        f"{len(google_play_items)} Google Play · "
        f"{len(hn_items)} Hacker News · "
        f"{len(seed_items)} seed"
    )

    return FeedbackBundle(
        items=all_items,
        reddit_count=len(reddit_items),
        app_store_count=len(app_store_items),
        google_play_count=len(google_play_items),
        hacker_news_count=len(hn_items),
        seed_count=len(seed_items),
        company=company,
    )


# ===========================================================================
# Source 1 — Reddit (posts + comments)
# ===========================================================================

async def _fetch_reddit(
    session: aiohttp.ClientSession,
    subreddit: str,
    progress_cb=None,
) -> List[FeedbackItem]:
    items: List[FeedbackItem] = []
    permalinks: List[Tuple[int, str]] = []

    urls = [
        f"https://www.reddit.com/r/{subreddit}/hot.json?limit=100",
        f"https://www.reddit.com/r/{subreddit}/top.json?limit=100&t=month",
    ]
    for url in urls:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    continue
                data = await resp.json(content_type=None)
                for post in data.get("data", {}).get("children", []):
                    p     = post.get("data", {})
                    title = p.get("title", "").strip()
                    body  = p.get("selftext", "").strip()
                    text  = f"{title}. {body}".strip(". ") if body else title
                    if len(text) < 20:
                        continue
                    items.append(FeedbackItem(
                        source="reddit",
                        text=text[:600],
                        upvotes=p.get("score", 0),
                        date=str(datetime.utcfromtimestamp(p.get("created_utc", 0)).date()),
                        url=f"https://reddit.com{p.get('permalink', '')}",
                    ))
                    permalink = p.get("permalink", "")
                    if permalink and p.get("num_comments", 0) > 2:
                        permalinks.append((p.get("score", 0), permalink))
            await asyncio.sleep(1)
        except Exception as e:
            logger.debug(f"Reddit fetch error {url}: {e}")

    # Comments from the top 40 most-upvoted posts
    top_permalinks = [pl for _, pl in sorted(permalinks, reverse=True)[:40]]
    if top_permalinks and progress_cb:
        await progress_cb(f"Fetching comments from top {len(top_permalinks)} posts...")
    for permalink in top_permalinks:
        items.extend(await _fetch_post_comments(session, permalink))
        await asyncio.sleep(0.5)

    return items


async def _fetch_post_comments(
    session: aiohttp.ClientSession,
    permalink: str,
    max_comments: int = 5,
) -> List[FeedbackItem]:
    items: List[FeedbackItem] = []
    url = f"https://www.reddit.com{permalink}.json?limit={max_comments}&sort=top"
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return items
            data = await resp.json(content_type=None)
            if not isinstance(data, list) or len(data) < 2:
                return items
            for c in data[1].get("data", {}).get("children", [])[:max_comments]:
                body = c.get("data", {}).get("body", "").strip()
                if not body or len(body) < 20 or body in ("[deleted]", "[removed]"):
                    continue
                items.append(FeedbackItem(
                    source="reddit",
                    text=body[:600],
                    upvotes=c.get("data", {}).get("score", 0),
                    url=f"https://reddit.com{permalink}",
                ))
    except Exception:
        pass
    return items


# ===========================================================================
# Source 2 — Apple App Store
# ===========================================================================

async def _fetch_app_store(session: aiohttp.ClientSession, app_id: str) -> List[FeedbackItem]:
    items: List[FeedbackItem] = []
    url = f"https://itunes.apple.com/us/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return items
            data = await resp.json(content_type=None)
            for entry in data.get("feed", {}).get("entry", []):
                if not isinstance(entry, dict):
                    continue
                content = entry.get("content", {})
                text    = content.get("label", "") if isinstance(content, dict) else ""
                rating  = entry.get("im:rating", {})
                rating  = rating.get("label", "") if isinstance(rating, dict) else ""
                if len(text) < 20:
                    continue
                prefix = f"[{rating}★] " if rating else ""
                items.append(FeedbackItem(
                    source="app_store",
                    text=(prefix + text)[:600],
                    upvotes=0,
                ))
    except Exception as e:
        logger.debug(f"App Store fetch error: {e}")
    return items


# ===========================================================================
# Source 3 — Google Play Store
# ===========================================================================

async def _fetch_google_play(play_id: str, count: int = 200) -> List[FeedbackItem]:
    """Run google-play-scraper (sync) in a thread pool."""
    loop = asyncio.get_event_loop()

    def _sync_fetch():
        try:
            from google_play_scraper import Sort, reviews as gp_reviews
            result, _ = gp_reviews(
                play_id,
                lang="en",
                country="us",
                sort=Sort.MOST_RELEVANT,
                count=count,
            )
            items = []
            for r in result:
                content = (r.get("content") or "").strip()
                if len(content) < 20:
                    continue
                score  = r.get("score", 0)
                prefix = f"[{score}★] " if score else ""
                items.append(FeedbackItem(
                    source="google_play",
                    text=(prefix + content)[:600],
                    upvotes=r.get("thumbsUpCount", 0),
                    date=str(r.get("at", ""))[:10],
                ))
            return items
        except Exception as e:
            logger.warning(f"google-play-scraper error for {play_id}: {e}")
            return []

    return await loop.run_in_executor(_executor, _sync_fetch)


# ===========================================================================
# Source 4 — Hacker News (Algolia API)
# ===========================================================================

_HN_STRIP_TAGS = re.compile(r"<[^>]+>")
_HN_COLLAPSE   = re.compile(r"\s+")


async def _fetch_hacker_news(
    session: aiohttp.ClientSession,
    company_name: str,
) -> List[FeedbackItem]:
    """
    Search HN comments + Ask HN stories via Algolia.
    Runs multiple queries to maximise coverage:
      - exact company name
      - company name + 'review'
      - company name + 'alternative'
    Fetches parent story title for context, deduplicates by objectID.
    """
    name = company_name.strip()

    # Multiple targeted queries — more coverage, less noise
    queries = [
        name,
        f"{name} review",
        f"{name} alternative",
        f"{name} feature",
    ]

    seen:  set  = set()
    items: List[FeedbackItem] = []

    # Cache parent story titles to avoid redundant fetches
    story_cache: dict = {}

    for query in queries:
        url = (
            f"https://hn.algolia.com/api/v1/search_by_date"
            f"?query={query}&tags=comment&hitsPerPage=100"
        )
        try:
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                if resp.status != 200:
                    continue
                data = await resp.json(content_type=None)

                for hit in data.get("hits", []):
                    oid = hit.get("objectID", "")
                    if oid in seen:
                        continue

                    # Clean HTML
                    raw  = hit.get("comment_text") or ""
                    text = _HN_COLLAPSE.sub(" ", _HN_STRIP_TAGS.sub(" ", raw)).strip()

                    if len(text) < 40:
                        continue

                    points = hit.get("points") or 0

                    # Enrich with parent story title for context
                    story_id    = hit.get("story_id") or hit.get("parent_id")
                    story_title = ""
                    if story_id:
                        if story_id not in story_cache:
                            story_cache[story_id] = await _fetch_hn_story_title(session, story_id)
                        story_title = story_cache[story_id]

                    enriched = f"[{story_title}] {text}" if story_title else text

                    seen.add(oid)
                    hn_url = f"https://news.ycombinator.com/item?id={oid}"
                    items.append(FeedbackItem(
                        source="hacker_news",
                        text=enriched[:600],
                        upvotes=points,
                        date=(hit.get("created_at") or "")[:10],
                        url=hn_url,
                    ))

        except Exception as e:
            logger.debug(f"HN fetch error for query '{query}': {e}")
        await asyncio.sleep(0.3)

    return items


async def _fetch_hn_story_title(session: aiohttp.ClientSession, story_id: int) -> str:
    """Fetch the title of an HN story by ID (used to give comments context)."""
    try:
        url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                return (data or {}).get("title", "")
    except Exception:
        pass
    return ""


# ===========================================================================
# Lookup helpers
# ===========================================================================

def _guess_subreddit(company_name: str) -> str:
    known = {
        "duolingo": "duolingo", "spotify": "spotify", "notion": "Notion",
        "slack": "Slack", "airbnb": "airbnb", "uber": "uber", "lyft": "lyft",
        "robinhood": "RobinHood", "coinbase": "CoinBase", "discord": "discordapp",
        "netflix": "netflix", "instagram": "Instagram", "twitter": "twitter",
        "x": "twitter", "tiktok": "Tiktokhelp", "reddit": "ideasfortheadmins",
        "youtube": "youtube", "snapchat": "snapchat", "pinterest": "pinterest",
        "linkedin": "linkedin", "zoom": "Zoom", "figma": "figma", "canva": "canva",
        "adobe": "adobe", "shopify": "shopify", "squarespace": "squarespace",
        "wix": "wix", "headspace": "headspace", "calm": "Calm", "strava": "Strava",
        "peloton": "pelotoncycle", "grammarly": "grammarly", "todoist": "todoist",
        "asana": "asana", "trello": "trello", "jira": "jira", "github": "github",
        "gitlab": "gitlab", "vscode": "vscode", "obsidian": "ObsidianMD",
        "anki": "Anki", "babbel": "babbel", "rosettastone": "languagelearning",
    }
    key = company_name.strip().lower().replace(" ", "")
    return known.get(key, company_name.strip().lower())


async def _find_app_store_id(company_name: str) -> str:
    known_ids = {
        "duolingo": "570060128", "spotify": "324684580", "notion": "1232780281",
        "slack": "618783545", "discord": "985746746", "airbnb": "401626263",
        "uber": "368677368", "lyft": "529379082", "robinhood": "938003185",
        "coinbase": "886427730", "netflix": "363590051", "instagram": "389801252",
        "tiktok": "835599320", "youtube": "544007664", "snapchat": "447188370",
        "pinterest": "429047995", "zoom": "546505307", "figma": "1152099778",
        "canva": "897446215", "shopify": "1288003325", "headspace": "493145008",
        "calm": "571800810", "strava": "426826309", "grammarly": "1158835574",
        "todoist": "572688855", "asana": "489969512", "trello": "461504587",
        "github": "1477376905", "obsidian": "1557175442", "anki": "373493387",
        "babbel": "829587901",
    }
    key = company_name.strip().lower().replace(" ", "")
    if key in known_ids:
        return known_ids[key]
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            url = (
                f"https://itunes.apple.com/search"
                f"?term={company_name.replace(' ', '+')}&entity=software&limit=1&country=us"
            )
            async with s.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    results = data.get("results", [])
                    if results:
                        return str(results[0].get("trackId", ""))
    except Exception as e:
        logger.debug(f"App Store ID search failed: {e}")
    return ""


def _guess_play_id(company_name: str) -> str:
    known_ids = {
        "duolingo":   "com.duolingo",
        "spotify":    "com.spotify.music",
        "notion":     "notion.id",
        "slack":      "com.Slack",
        "discord":    "com.discord",
        "airbnb":     "com.airbnb.android",
        "uber":       "com.ubercab",
        "lyft":       "me.lyft.android",
        "robinhood":  "com.robinhood.android",
        "coinbase":   "com.coinbase.android",
        "netflix":    "com.netflix.mediaclient",
        "instagram":  "com.instagram.android",
        "tiktok":     "com.zhiliaoapp.musically",
        "youtube":    "com.google.android.youtube",
        "snapchat":   "com.snapchat.android",
        "pinterest":  "com.pinterest",
        "zoom":       "us.zoom.videomeetings",
        "canva":      "com.canva.editor",
        "shopify":    "com.shopify.mobile",
        "headspace":  "com.getsomeheadspace.android",
        "calm":       "com.calm.android",
        "strava":     "com.strava",
        "grammarly":  "com.grammarly.android.keyboard",
        "todoist":    "com.todoist.android.Todoist",
        "trello":     "com.trello",
        "github":     "com.github.android",
        "babbel":     "com.babbel.mobile.android.en",
        "anki":       "com.ichi2.anki",
    }
    key = company_name.strip().lower().replace(" ", "")
    return known_ids.get(key, "")


# ===========================================================================
# Duolingo seed data (guaranteed baseline for the default demo)
# ===========================================================================

def _get_duolingo_seed_data() -> List[FeedbackItem]:
    raw = [
        # Speaking / conversation practice
        ("reddit", "I've been using Duolingo for 2 years and I still can't hold a basic conversation. There's almost zero speaking practice and it shows.", 2847),
        ("reddit", "The speaking exercises are a joke. It accepts anything you say. I said gibberish and it marked it correct.", 1923),
        ("reddit", "What I actually need is to practice having a real conversation, not translating isolated sentences. The Roleplay feature in Max is too scripted.", 1654),
        ("reddit", "Duolingo should have an AI you can just... talk to. About anything. In the language you're learning. That would actually be useful.", 3102),
        ("reddit", "I paid for Duolingo Max specifically for the AI roleplay and it's so limited. The scenarios are like 5 and they feel canned.", 987),
        ("app_store", "The speaking practice doesn't work. I speak clearly and it says wrong every time, then I mumble and it accepts. Useless feature.", 0),
        ("app_store", "Would love a free conversation mode where I can practice speaking without fear of judgment. That would be worth paying for.", 0),
        ("reddit", "Babbel lets you practice pronunciation with actual feedback on which sounds you got wrong. Why can't Duolingo do this?", 1245),
        ("reddit", "I want the AI to correct my pronunciation specifically, not just say 'try again'. What sound did I mess up?", 876),
        ("seed", "After 500 days of Duolingo I froze when a Spanish speaker asked me a simple question. The app doesn't prepare you for real speech.", 0),
        # Grammar
        ("reddit", "The grammar tips are so shallow. 'In Spanish, nouns have gender.' Ok... but WHY and how do I know which is which?", 2341),
        ("reddit", "Explain My Answer is the best feature they've added in years. But it only works when you get something wrong. I want explanations for the things I get right too.", 1567),
        ("reddit", "I don't understand why I'm wrong, I just know I am. The app needs to actually teach grammar, not just expose you to it.", 3456),
        ("reddit", "Can we get a grammar mode where you can ask follow-up questions? Like 'but what about when...' I want to ask the AI questions.", 2109),
        ("app_store", "No grammar teaching at all. Just trial and error. I've been doing this for months and still don't know the rules.", 0),
        ("reddit", "I genuinely don't understand why ser and estar are different and after hundreds of exercises I'm still guessing.", 1789),
        ("seed", "Duolingo teaches you to pattern match, not to understand. That works until you hit a sentence you've never seen before.", 0),
        # Personalization
        ("reddit", "Duolingo has no idea that I already know French. The Spanish course keeps explaining basic stuff I could skip.", 1234),
        ("reddit", "I want to learn business Spanish but the course treats everyone the same. Medical Spanish, travel Spanish — where's the specialization?", 2678),
        ("reddit", "The AI should know by now that I always get verb conjugations wrong. Why is it still giving me the same mix?", 1890),
        ("reddit", "There's no way to tell Duolingo what you actually want to learn. I'm preparing for a job interview in German.", 3234),
        ("app_store", "Would be great if the app learned from my mistakes and focused more on my weak areas automatically.", 0),
        ("reddit", "After 1000 days, Duolingo still gives me the same exercises as a beginner. There's no real progression for advanced learners.", 2456),
        ("seed", "The app feels like a one-size-fits-all product in 2024 when AI could make it completely personal.", 0),
        ("reddit", "I'd pay double for a version that adapts to what I actually struggle with.", 1567),
        # Streak / gamification
        ("reddit", "The streak mechanic is psychologically damaging. I feel actual anxiety missing a day.", 5678),
        ("reddit", "I took a mental health day and lost my 400 day streak. Cried. That's not healthy.", 4321),
        ("reddit", "The streak should be weekly not daily. Real life happens. Missing ONE day shouldn't erase months of work.", 6789),
        ("app_store", "Streak anxiety ruined language learning for me. I quit because the pressure was making me hate the app.", 0),
        ("reddit", "I spend 10 minutes doing the minimum required to keep my streak, not actually learning. The incentive structure is broken.", 3456),
        ("seed", "The gamification works to build habits but actively works against deep learning. You optimize for streaks not comprehension.", 0),
        # Content quality
        ("reddit", "The listening exercises use robotic TTS voices. In 2024 this is embarrassing.", 3456),
        ("reddit", "Why are all the Duolingo sentences so weird? 'The penguin drinks the purple milk.' I need sentences I'll actually use.", 2789),
        ("reddit", "The vocabulary is bizarrely skewed toward animals and children's food. I'm a 35 year old professional.", 1987),
        ("app_store", "Stories are amazing and I wish there were hundreds more. They actually teach me real language in context.", 0),
        ("reddit", "DuoRadio and podcasts are underrated. Real spoken content at different speeds is exactly what I need.", 1567),
        ("seed", "The gap between Duolingo intermediate and actual fluency is a chasm. The app doesn't help you cross it.", 0),
        # AI requests
        ("reddit", "I want an AI tutor I can have a full conversation with about literally anything. In the target language. With corrections in real time.", 4567),
        ("reddit", "GPT-4 can already do what I'm describing. Why isn't Duolingo doing this? I'd pay $50/month for a real AI conversation partner.", 3456),
        ("reddit", "Imagine if the AI could roleplay as a shopkeeper, a friend, a job interviewer. That would be transformative.", 2789),
        ("reddit", "The AI should be able to write with me — I write a paragraph in Spanish, it corrects and explains every correction.", 1987),
        ("app_store", "The AI features in Max are promising but too limited. More scenarios, more flexibility.", 0),
        ("seed", "AI tutoring that knows your history, your weak points, your goals, and adapts every session. That's the product I'd pay for.", 0),
        # Competitor comparisons
        ("reddit", "ChatGPT is now my main language tutor. I have full conversations, get detailed corrections, can ask about anything. Duolingo feels outdated.", 4567),
        ("reddit", "Pimsleur taught me more conversational Portuguese in 30 lessons than Duolingo did in 300.", 2345),
        ("reddit", "I switched to Anki + iTalki + podcasts. Duolingo is a supplement at best.", 1789),
        ("seed", "The fact that people are replacing Duolingo with ChatGPT should be an existential concern for the product team.", 0),
        # Max feedback
        ("reddit", "Explain My Answer changed my life. I actually understand grammar now. I need this for EVERY exercise.", 3456),
        ("reddit", "The Roleplay AI in Max is great but it only has like 5 scenarios. I've done all of them.", 1987),
        ("reddit", "I want the AI to proactively point out patterns in my mistakes. 'You consistently confuse ser and estar' — that insight would be gold.", 2345),
        ("app_store", "Max subscription feels incomplete. The two AI features are good but it doesn't feel like a complete premium product yet.", 0),
        ("reddit", "Duolingo Max should have: unlimited AI conversation, personalized grammar explanations, writing practice, pronunciation coaching.", 3456),
        ("seed", "Duolingo Max is the right idea but needs 10x more AI features to justify the price vs just using ChatGPT.", 0),
        # Positive
        ("reddit", "The gamification is genuinely genius for building habits. I've done Spanish every day for 2 years because of the streak.", 2345),
        ("reddit", "Duolingo is the best tool for absolute beginners. Nothing else makes starting a language feel so achievable.", 3456),
        ("app_store", "Best language app I've used. The interface is clean and lessons are just the right length.", 0),
        ("seed", "Duolingo nails onboarding and habit formation. Where it fails is depth and output skills.", 0),
        # Advanced learners
        ("reddit", "There's nothing for B2+ learners. Once you finish the course you're just... done.", 2789),
        ("reddit", "I finished the Japanese course and I still can't read a newspaper.", 3456),
        ("seed", "The ceiling on Duolingo is B1. That's fine if that's the goal, but they should be honest about it.", 0),
    ]
    return [
        FeedbackItem(source=source, text=text, upvotes=upvotes)
        for source, text, upvotes in raw
    ]
