import calendar
import os
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import feedparser
import httpx
import psycopg
import trafilatura
from dotenv import load_dotenv


FEED_URL = "https://feeds.bbci.co.uk/news/business/rss.xml"
USER_AGENT = "discover-agent/0.1 (personal learning project)"
SKIPPED_PATHS = ("/videos/", "/live/", "/sounds/", "/programmes/")
MIN_BODY_LENGTH = 500
FOOTER_MARKERS = ("Get in touch", "Related topics")


def clean_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def clean_body(body: str) -> str:
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip() in FOOTER_MARKERS and i > len(lines) * 0.6:
            lines = lines[:i]
            break
    return "\n".join(lines).strip()


def published_at(entry: feedparser.FeedParserDict) -> datetime | None:
    parsed = entry.get("published_parsed")
    if parsed is None:
        return None
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)


def print_result(status: str, title: str, detail: str | None = None) -> None:
    line = f"{status}: {title}"
    if detail:
        line += f" ({detail})"
    print(line, flush=True)


def main() -> None:
    load_dotenv()

    counts = {"OK": 0, "SKIPPED": 0, "TOO SHORT": 0, "FAILED": 0}
    headers = {"User-Agent": USER_AGENT}

    with (
        psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn,
        httpx.Client(headers=headers, follow_redirects=True, timeout=30.0) as client,
    ):
        feed_response = client.get(FEED_URL)
        feed_response.raise_for_status()
        feed = feedparser.parse(feed_response.content)
        last_request_at = time.monotonic()

        for entry in feed.entries:
            title = entry.get("title", "Untitled article")

            try:
                raw_url = entry.get("link")
                if not raw_url:
                    raise ValueError("feed entry has no URL")

                url = clean_url(raw_url)
                path = urlsplit(url).path.lower()

                if any(skipped_path in path for skipped_path in SKIPPED_PATHS):
                    counts["SKIPPED"] += 1
                    print_result("SKIPPED", title, "unsupported URL")
                    continue

                exists = conn.execute(
                    "SELECT 1 FROM articles WHERE url = %s", (url,)
                ).fetchone()
                if exists:
                    counts["SKIPPED"] += 1
                    print_result("SKIPPED", title, "already in database")
                    continue

                wait_seconds = 1.0 - (time.monotonic() - last_request_at)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)

                response = client.get(url)
                last_request_at = time.monotonic()
                response.raise_for_status()

                body = trafilatura.extract(response.text)
                if body is not None:
                    body = clean_body(body)
                if body is None or len(body) < MIN_BODY_LENGTH:
                    counts["TOO SHORT"] += 1
                    body_length = len(body) if body else 0
                    print_result("TOO SHORT", title, f"{body_length} characters")
                    continue

                metadata = trafilatura.extract_metadata(response.text)
                author = metadata.author if metadata is not None else None

                conn.execute(
                    """
                    INSERT INTO articles
                        (url, title, section, author, published_at, body)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                    """,
                    (
                        url,
                        title,
                        "Business",
                        author,
                        published_at(entry),
                        body,
                    ),
                )

                counts["OK"] += 1
                print_result("OK", title)
            except Exception as exc:
                counts["FAILED"] += 1
                print_result("FAILED", title, f"{type(exc).__name__}: {exc}")

    print(
        "Summary: " + ", ".join(f"{status}={count}" for status, count in counts.items())
    )


def remove_callouts(lines: list[str]) -> list[str]:
    cleaned = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        if line.strip() == "Get in touch":
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if next_line.endswith("?"):
                skip_next = True
            continue
        cleaned.append(line)
    return cleaned


def clean_body(body: str) -> str:
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip() in FOOTER_MARKERS and i > len(lines) * 0.6:
            lines = lines[:i]
            break
    lines = remove_callouts(lines)
    return "\n".join(lines).strip()

if __name__ == "__main__":
    main()