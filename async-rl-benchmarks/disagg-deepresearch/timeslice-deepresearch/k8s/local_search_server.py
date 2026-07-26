#!/usr/bin/env python3
"""Local Wikipedia BM25 search server matching Serper API format.

Identical to the one used by the colocated benchmark
(GPU-CR/benchmark-deepresearch/k8s-job.yaml) — runs in the
deepresearch-benchmark image (has pyserini + JDK), downloads the ~12GB
wikipedia-dpr-100w prebuilt index on first start.
"""
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

SEARCHER = None


def init_searcher():
    global SEARCHER
    print("Loading pyserini Wikipedia index (this may take a few minutes)...")
    from pyserini.search.lucene import LuceneSearcher
    SEARCHER = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
    if SEARCHER is None:
        raise RuntimeError("Failed to load pyserini prebuilt index wikipedia-dpr-100w")
    print("Search index loaded. Ready to serve.")


class SearchHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
            query = data.get('q', '')
            num_results = data.get('num', 10)
        except Exception:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "invalid json"}')
            return

        results = []
        if SEARCHER and query:
            hits = SEARCHER.search(query, k=num_results)
            for hit in hits:
                doc = SEARCHER.doc(hit.docid)
                raw = json.loads(doc.raw()) if doc.raw() else {}
                contents = raw.get('contents', doc.raw() or '')
                if not isinstance(contents, str):
                    contents = str(contents)
                title = raw.get('title', '')
                # DPR-100w format: contents = "Title"\npassage text
                if not title and contents.startswith('"'):
                    first, _, rest = contents.partition('\n')
                    title = first.strip('"')
                    contents = rest or contents
                if not title:
                    title = str(hit.docid)
                snippet = contents[:500]
                results.append({
                    'title': title,
                    'link': f'https://en.wikipedia.org/wiki/{title.replace(" ", "_")}',
                    'snippet': snippet,
                    'date': '',
                    'source': 'wikipedia'
                })

        response = json.dumps({
            'searchParameters': {'q': query},
            'organic': results
        })
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(response.encode())

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    port = int(os.environ.get('SEARCH_PORT', '8877'))
    init_searcher()
    server = HTTPServer(('0.0.0.0', port), SearchHandler)
    print(f"Search server running on port {port}")
    server.serve_forever()
