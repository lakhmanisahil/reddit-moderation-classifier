"""
Create a realistic *synthetic* Reddit dataset so the whole project runs out of the box.

Why synthetic?  Real moderation data (which posts moderators removed, how often they
were reported) is not available through Reddit's public API any more, and the public
datasets that have it can't be redistributed in a repo. This script makes data with the
same shape and the same *kinds* of patterns you see in real Reddit data:

* five kinds of post — SAFE, SPAM, TOXIC, LOW_EFFORT, MISINFORMATION
* spam comes from new, low-karma accounts; toxic posts get reported a lot
* deleted posts get fewer upvotes (they were visible for less time)
* and plenty of NOISE: good posts removed for "wrong flair", spam that slipped through,
  normal posts that happen to say "free" or "link" ... so the models can't be perfect.

Use a real dataset for the numbers you quote on your resume (see scripts/prepare_real_data.py).

Run:  python scripts/make_synthetic_data.py            (8,000 posts)
      python scripts/make_synthetic_data.py --n 2000   (smaller, faster)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RANDOM_STATE, SYNTHETIC_DATA, ensure_dirs  # noqa: E402

# ── Vocabulary ───────────────────────────────────────────────────────────
TOPICS = {
    "technology": ["my new laptop", "Linux on an old PC", "mechanical keyboards", "the latest phone update",
                   "my home server", "VPNs", "password managers", "a budget monitor", "open-source tools"],
    "personalfinance": ["an emergency fund", "my first SIP", "credit card debt", "index funds",
                        "my student loan", "a car loan", "budgeting apps", "saving for a house", "tax filing"],
    "fitness": ["deadlifts", "a beginner program", "my squat form", "protein intake", "running a 5k",
                "knee pain after squats", "cutting vs bulking", "home workouts", "rest days"],
    "gaming": ["the new open-world RPG", "my first gaming PC", "speedrunning", "controller vs keyboard",
               "the latest patch", "co-op games", "an indie platformer", "frame drops", "retro consoles"],
    "movies": ["the director's new film", "practical effects", "90s thrillers", "film scores",
               "the sequel everyone hates", "foreign films", "plot holes", "cinematography", "animated films"],
    "science": ["black holes", "CRISPR", "vaccines", "climate data", "the Mars rover",
                "quantum computing", "microplastics", "sleep research", "the James Webb telescope"],
}
CATEGORY_WEIGHTS = {"SAFE": 0.62, "SPAM": 0.10, "TOXIC": 0.09, "LOW_EFFORT": 0.12, "MISINFORMATION": 0.07}

SAFE_TITLES = [
    "Question about {t}", "Finally hit a milestone with {t}", "Looking for advice on {t}",
    "What's your experience with {t}?", "I wrote a detailed guide on {t} after two years",
    "Comparison: {t} vs {t2}", "Discussion: is {t} overrated?", "Beginner here, need help understanding {t}",
    "Lessons I learned from {t}", "Weekly thread: share your progress with {t}",
    "Is it worth getting into {t} in 2025?", "Free resources for learning about {t}?",
]
SAFE_SENTENCES = [
    "I've been looking into {t} for a few weeks and I'm still not sure where to start.",
    "Here is what I tried so far: {d}.", "Any recommendations would be really appreciated.",
    "Thanks in advance, this community has been super helpful.",
    "I read the wiki and the FAQ but couldn't find an answer to this.", "For context, I'm {c}.",
    "Would love to hear what worked for you.", "Edit: thanks everyone for the thoughtful replies!",
    "I compared it with {t2} and honestly both have pros and cons.",
    "The biggest surprise for me was how much consistency matters.",
    "I'll link the source in the comments if anyone wants it.",
    "Not sure if this is the right sub, happy to move it.",
]
DETAILS = ["reading the official docs", "watching a couple of tutorials", "asking a friend who knows more",
           "comparing a few options in a spreadsheet", "keeping notes for a month", "following the sidebar guide"]
CONTEXTS = ["a complete beginner", "a student on a tight budget", "about a year into this",
            "coming back after a long break", "doing this as a weekend hobby"]

SPAM_TITLES = [
    "FREE {p} giveaway - limited time!!", "Make $500/day from home with this one trick",
    "Best {p} deals, 90% off, click now", "I found a site that gives free {p}", "DM me for cheap {p}",
    "Earn passive income fast - guaranteed", "I made a tool for {t}, would love feedback (link in comments)",
    "Get {p} for free before they patch it", "Huge {p} sale today only",
]
SPAM_SENTENCES = [
    "Check out my website www.{s}.com for the best deals.", "Use my referral code {code} to get a bonus!",
    "Limited offer, only for the first 100 people.", "Click the link in my profile now!!!",
    "100% legit, no scam, guaranteed results.", "Subscribe to my channel for more giveaways.",
    "Buy now before the price goes up.", "Message me on telegram for the discount.",
    "It is totally free, you just need to sign up.",
]
PRODUCTS = ["gift cards", "crypto", "followers", "game skins", "VPN accounts", "phones", "courses"]
SITES = ["dealz4u", "freegift-now", "cashfast-pro", "topdeals-hub", "promo-king"]

TOXIC_TITLES = [
    "Anyone who likes {t} is an idiot", "This sub is full of morons", "Mods here are pathetic losers",
    "Stop posting garbage about {t}, clowns", "You people are the worst", "Unpopular opinion: {t} fans are braindead",
    "Why is everyone here so stupid about {t}?",
]
TOXIC_SENTENCES = [
    "Honestly you are all braindead.", "Get a life, nobody cares about your stupid opinion.",
    "Shut up and leave, you clown.", "Only an idiot would believe this trash.",
    "I hope the mods ban every one of you losers.", "This is the dumbest community on the internet.",
    "Go cry about it, snowflake.",
]
LOW_TITLES = ["help", "?", "{t}", "title", "thoughts?", "lol", "this", "pls answer", "question", "anyone??"]
LOW_BODIES = ["", "", "same", "see title", "pls", "anyone?", "idk", "lol", "^", "help me"]

MISINFO_TITLES = [
    "Doctors don't want you to know this about {t}", "{T} is a hoax and here's proof",
    "The truth about {t} they are hiding", "Wake up: {t} is controlled by the government",
    "Scientists admit {t} was faked", "Leaked: the real reason behind {t}",
]
MISINFO_SENTENCES = [
    "Do your own research, the mainstream media won't tell you.",
    "My cousin's friend saw it with their own eyes.", "They delete this every time it gets posted.",
    "It's all a cover-up, follow the money.", "Big companies are paying to hide the facts.",
    "Share this before it gets taken down!", "Nobody is talking about this and that's not a coincidence.",
]


def _fill(template: str, rng, sub: str) -> str:
    t, t2 = rng.choice(TOPICS[sub], size=2, replace=False)
    return template.format(
        t=t, t2=t2, T=t[0].upper() + t[1:], d=rng.choice(DETAILS), c=rng.choice(CONTEXTS),
        p=rng.choice(PRODUCTS), s=rng.choice(SITES), code=f"X{rng.integers(1000, 9999)}",
    )


def _body(pool, rng, sub, lo, hi):
    k = int(rng.integers(lo, hi + 1))
    return " ".join(_fill(s, rng, sub) for s in rng.choice(pool, size=k, replace=False))


def make_post(category: str, sub: str, rng) -> tuple[str, str]:
    """Return (title, body) for one post of the given category."""
    if category == "SAFE":
        title = _fill(rng.choice(SAFE_TITLES), rng, sub)
        body = _body(SAFE_SENTENCES, rng, sub, 1, 5) if rng.random() > 0.08 else ""
    elif category == "SPAM":
        title = _fill(rng.choice(SPAM_TITLES), rng, sub)
        body = _body(SPAM_SENTENCES, rng, sub, 1, 4)
        if rng.random() < 0.25:            # disguised spam: starts like a normal post
            body = _body(SAFE_SENTENCES, rng, sub, 1, 2) + " " + body
    elif category == "TOXIC":
        title = _fill(rng.choice(TOXIC_TITLES), rng, sub)
        body = _body(TOXIC_SENTENCES, rng, sub, 1, 3)
        if rng.random() < 0.35:            # a normal post with an insult at the end
            title = _fill(rng.choice(SAFE_TITLES), rng, sub)
            body = _body(SAFE_SENTENCES, rng, sub, 1, 3) + " " + _body(TOXIC_SENTENCES, rng, sub, 1, 1)
    elif category == "LOW_EFFORT":
        title = _fill(rng.choice(LOW_TITLES), rng, sub)
        body = _fill(rng.choice(LOW_BODIES), rng, sub)
    else:  # MISINFORMATION
        title = _fill(rng.choice(MISINFO_TITLES), rng, sub)
        body = _body(MISINFO_SENTENCES, rng, sub, 1, 4)
    return title, body


def generate(n: int = 8000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = rng.choice(list(CATEGORY_WEIGHTS), size=n, p=list(CATEGORY_WEIGHTS.values()))
    subs = rng.choice(list(TOPICS), size=n)

    rows = []
    for i, (cat, sub) in enumerate(zip(cats, subs)):
        title, body = make_post(cat, sub, rng)
        bad = cat != "SAFE"

        # Author: spammers use new, low-karma accounts
        age_scale = {"SPAM": 20, "TOXIC": 200, "MISINFORMATION": 250}.get(cat, 600)
        account_age = int(rng.exponential(age_scale)) + 1
        karma = int(rng.lognormal(3 if cat == "SPAM" else 6, 1.5))

        # Reports (= "flags") from other users
        lam = {"SAFE": 0.12, "SPAM": 2.0, "TOXIC": 2.8, "LOW_EFFORT": 0.6, "MISINFORMATION": 1.8}[cat]
        reports = int(rng.poisson(lam))
        flagged = int(reports > 0 or (cat == "SPAM" and rng.random() < 0.3))  # AutoModerator flags some spam

        # Deletion = removed by moderators. Mostly depends on category, plus noise.
        p = {"SAFE": 0.07, "SPAM": 0.75, "TOXIC": 0.62, "LOW_EFFORT": 0.40, "MISINFORMATION": 0.50}[cat]
        p += 0.10 * (account_age < 30) + 0.12 * flagged
        deleted = int(rng.random() < min(p, 0.95))

        # Engagement: deleted posts were visible for less time → fewer votes
        base = rng.lognormal(3.2 if not bad else 1.5, 1.1)
        score = int(base * (0.3 if deleted else 1.0)) - (int(rng.integers(0, 15)) if cat == "TOXIC" else 0)
        ratio_mean = {"SAFE": 0.88, "SPAM": 0.35, "TOXIC": 0.42, "LOW_EFFORT": 0.6, "MISINFORMATION": 0.45}[cat]
        upvote_ratio = round(float(np.clip(rng.normal(ratio_mean, 0.1), 0.05, 1.0)), 2)
        comments = int(rng.poisson(max(score, 0) * 0.3 + (12 if cat in ("TOXIC", "MISINFORMATION") else 2)))

        rows.append(dict(
            post_id=f"t3_{i:06d}", subreddit=sub, title=title, selftext=body,
            author=f"user_{rng.integers(0, n // 2):05d}", author_karma=karma, account_age_days=account_age,
            created_utc=int(1735689600 + rng.integers(0, 365 * 24 * 3600)),   # during 2025
            score=score, upvote_ratio=upvote_ratio, num_comments=comments,
            num_reports=reports, is_flagged=flagged, is_deleted=deleted, category=cat,
        ))

    df = pd.DataFrame(rows)
    # Real data is messy: a few duplicates and missing bodies, so cleaning has work to do
    dupes = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    df.loc[df.sample(frac=0.02, random_state=seed + 1).index, "selftext"] = np.nan
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    args = ap.parse_args()
    ensure_dirs()
    df = generate(args.n)
    df.to_csv(SYNTHETIC_DATA, index=False)
    print(f"Saved {len(df):,} posts to {SYNTHETIC_DATA}")
    print(f"deleted: {df.is_deleted.mean():.1%} | flagged: {df.is_flagged.mean():.1%}")
    print(df.category.value_counts().to_string())
