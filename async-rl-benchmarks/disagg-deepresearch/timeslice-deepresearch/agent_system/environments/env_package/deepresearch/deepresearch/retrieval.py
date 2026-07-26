# Vendored from cxcscmu/verl-agent-deepresearch (commit 9c311053).
# Changes for the disaggregated benchmark:
#   - query_serper points at the local pyserini Wikipedia search server
#     (Serper-compatible API) via SEARCH_URL (default http://localhost:8877/search)
#     instead of google.serper.dev — no API key header needed.
#   - clueweb path, dotenv/diskcache/aiohttp imports dropped (unused).

import requests
import json
import time
import sys
import os
import random
from collections import deque
import threading

MAX_RETRIES = 20
RETRY_DELAY = 2

serper_time_log = "/tmp/serper_time_log.txt"
serper_error_log = "/tmp/serper_error_log.txt"


class RateLimiter:
    """
    Simple in-process sliding window rate limiting: at most max_calls calls in
    any 1-second window. Thread-safe.
    """
    def __init__(self, max_calls: int, per_seconds: float = 1.0):
        self.max_calls = max_calls
        self.per = per_seconds
        self.calls = deque()  # store timestamps
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                # clean up calls outside the window
                while self.calls and now - self.calls[0] >= self.per:
                    self.calls.popleft()

                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return

                # otherwise, wait for the oldest call to expire
                sleep_for = self.per - (now - self.calls[0])

            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                time.sleep(0.001)


# Global rate limiter: 100 QPS
SERPER_RATE_LIMITER = RateLimiter(max_calls=100, per_seconds=1.0)


def query_clueweb(query, num_docs=10):
    raise NotImplementedError(
        "clueweb retrieval was stripped from this vendored copy — "
        "use search_engine='serper' (local Wikipedia BM25 server)")


def query_serper(query: str):
    # Local Serper-format search server (pyserini BM25 over wikipedia-dpr-100w)
    url = os.environ.get("SEARCH_URL", "http://localhost:8877/search")
    headers = {
        'Content-Type': 'application/json',
    }
    q = (query or "").strip()
    data = {
        "q": q,
        "num": 10,
        "extendParams": {
            "country": "en",
            "page": 1,
        },
    }

    if not q:
        print(f"'{query}' is a blank query.", file=sys.stderr)
        return [f"'{query}' is a blank query."]

    response = None
    max_attempts = 10
    for i in range(max_attempts):
        try:
            SERPER_RATE_LIMITER.acquire()
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
            break
        except Exception as e:
            if i == max_attempts - 1:
                with open(serper_error_log, "a") as f:
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    f.write(f"{time_str} Search attempt failed after {i + 1} attempts, "
                            f"query: {q}, error: {repr(e)}\n")
                print(f"Search attempt failed after {i + 1} attempts, query: {q}, "
                      f"error: {repr(e)}", file=sys.stderr)
                return ["Search timeout, return None, Please try again later."]
            else:
                time.sleep(random.uniform(0.5, 2))

    if response is None or response.status_code != 200:
        status = None if response is None else response.status_code
        body = ""
        try:
            body = (response.text or "")[:200].replace("\n", " ") if response is not None else "<no response>"
        except Exception:
            body = "<unavailable>"
        print(f"Search HTTP error, status={status}, query: {q}, body: {body}", file=sys.stderr)
        with open(serper_error_log, "a") as f:
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"{time_str} Search HTTP error, status={status}, query: {q}, body: {body}\n")
        return ["Search timeout, return None, Please try again later."]

    try:
        results = response.json()
    except Exception as e:
        print(f"Search JSON decode error, query: {q}, error: {repr(e)}", file=sys.stderr)
        with open(serper_error_log, "a") as f:
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"{time_str} Search JSON decode error, query: {q}, error: {repr(e)}\n")
        return ["Search timeout, return None, Please try again later."]

    if not isinstance(results, dict) or "organic" not in results:
        print(f"Search parse issue: 'organic' not in results, query: {q}", file=sys.stderr)
        return [f"No results found for '{query}'. Try with a more general query."]

    try:
        web_snippets = list()
        idx = 0
        for page in results["organic"]:
            idx += 1
            date_published = ""
            if "date" in page:
                date_published = "\nDate published: " + page["date"]

            source = ""
            if "source" in page:
                source = "\nSource: " + page["source"]

            snippet = ""
            if "snippet" in page:
                snippet = "\n" + page["snippet"]

            redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)

        if not web_snippets:
            return [f"No results found for '{query}'. Try with a more general query."]

        content = (f"A search for '{query}' found {len(web_snippets)} results:\n\n"
                   f"## Web Results\n" + "\n\n".join(web_snippets))
        return [content]
    except Exception:
        return [f"No results found for '{query}'. Try with a more general query."]
