#!/usr/bin/env python3
"""Convert RESULTS.md to a self-contained HTML file with base64-embedded images."""

import base64
import html
import re
from pathlib import Path

BASE = Path("/Users/aishuk/workspaces/GPU-CR/async-rl-timeslicing")
SRC = BASE / "RESULTS.md"
OUT = BASE / "RESULTS.html"


def inline_fmt(text):
    """Bold, inline code, and image handling for a single line's text."""
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def embed_image(alt, src):
    p = BASE / src
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode()
        return (f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">'
                f"<figcaption>{html.escape(alt)}</figcaption></figure>")
    return f"<p><em>[missing image: {html.escape(src)}]</em></p>"


lines = SRC.read_text().splitlines()
out = []
i = 0
in_code = False
code_buf = []
in_list = False

while i < len(lines):
    line = lines[i]

    if line.startswith("```"):
        if in_code:
            out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
            code_buf, in_code = [], False
        else:
            in_code = True
        i += 1
        continue
    if in_code:
        code_buf.append(line)
        i += 1
        continue

    m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line)
    if m:
        if in_list:
            out.append("</ul>"); in_list = False
        out.append(embed_image(m.group(1), m.group(2)))
        i += 1
        continue

    hm = re.match(r"^(#{1,4})\s+(.*)$", line)
    if hm:
        if in_list:
            out.append("</ul>"); in_list = False
        level = len(hm.group(1))
        out.append(f"<h{level}>{inline_fmt(hm.group(2))}</h{level}>")
        i += 1
        continue

    # Table: current line starts with | and next is a separator row
    if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|?\s*$", lines[i + 1]):
        if in_list:
            out.append("</ul>"); in_list = False
        headers = [c.strip() for c in line.strip("|").split("|")]
        rows = []
        j = i + 2
        while j < len(lines) and lines[j].startswith("|"):
            rows.append([c.strip() for c in lines[j].strip("|").split("|")])
            j += 1
        t = ["<table><thead><tr>"]
        t += [f"<th>{inline_fmt(h)}</th>" for h in headers]
        t.append("</tr></thead><tbody>")
        for r in rows:
            t.append("<tr>" + "".join(f"<td>{inline_fmt(c)}</td>" for c in r) + "</tr>")
        t.append("</tbody></table>")
        out.append("".join(t))
        i = j
        continue

    lm = re.match(r"^(\d+\.|-)\s+(.*)$", line)
    if lm:
        if not in_list:
            out.append("<ul>"); in_list = True
        # Gather continuation lines (indented)
        item = lm.group(2)
        while i + 1 < len(lines) and re.match(r"^\s{2,}\S", lines[i + 1]) and not re.match(r"^\s*(\d+\.|-)\s", lines[i + 1]):
            i += 1
            item += " " + lines[i].strip()
        out.append(f"<li>{inline_fmt(item)}</li>")
        i += 1
        continue

    if not line.strip():
        if in_list:
            out.append("</ul>"); in_list = False
        i += 1
        continue

    # Paragraph: merge consecutive non-empty plain lines
    para = [line]
    while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||```|!\[|\d+\.\s|-\s)", lines[i + 1]):
        i += 1
        para.append(lines[i])
    if in_list:
        out.append("</ul>"); in_list = False
    out.append(f"<p>{inline_fmt(' '.join(para))}</p>")
    i += 1

if in_list:
    out.append("</ul>")

body = "\n".join(out)
html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Async RL Time-Slicing Feasibility — Results</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; line-height: 1.55;
         color: #1a1a1a; max-width: 980px; margin: 0 auto; padding: 2rem 1.5rem 6rem; }}
  h1 {{ border-bottom: 3px solid #E53935; padding-bottom: .4rem; }}
  h2 {{ margin-top: 2.2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
  h3 {{ margin-top: 1.6rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; font-size: .92rem; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: .45rem .6rem; text-align: left; }}
  th {{ background: #f5f5f5; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  code {{ background: #f2f2f2; border-radius: 3px; padding: .1rem .3rem; font-size: .88em; }}
  pre {{ background: #f6f8fa; border: 1px solid #e0e0e0; border-radius: 6px;
        padding: .9rem 1rem; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
  figure {{ margin: 1.4rem 0; text-align: center; }}
  figure img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
  figcaption {{ font-size: .85rem; color: #666; margin-top: .4rem; }}
  strong {{ color: #000; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
OUT.write_text(html_doc)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")
