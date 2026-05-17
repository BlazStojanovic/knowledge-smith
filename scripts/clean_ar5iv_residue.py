#!/usr/bin/env python3
"""Audit and clean ar5iv-conversion residue in raw paper markdown.

Two-phase strategy:

  1. **Header chrome strip** — for ar5iv papers with body content, drop everything
     between end-of-frontmatter and the first ATX heading (`# Title`). That block
     contains all the leaked JS (color-scheme, mathjax loader, citation preview),
     ar5iv navigation (prev/next/feeling-lucky), the LaTeXML mascot, and the
     Copyright / Privacy / Conversion-report footer-of-the-header. For papers
     where ar5iv only embedded a PDF placeholder (no `# ` heading found), the
     file is left alone — there is no content to protect, and the chrome IS
     the file.

  2. **Footer chrome strip** — at end of file, drop trailing site chrome
     bounded by markers like "Generated on ... by [LaTeXML...]" or the
     standalone Copyright / Privacy line.

  3. **Atom strip** — within body content, remove residue that survived:
     `[Refer to caption]` accessibility alt text, stray theme/mascot lines.

Leaves alone:
  - `[[N](#bib.bibN)]` citation refs (functional in-document cross-references)
  - `[N.M](#S...)` section refs
  - LaTeX math residue `[k2]`, `[rgb]`, `[0,1]` (context-sensitive)

Usage:
  uv run python clean_ar5iv_residue.py --audit            # dry-run, print stats
  uv run python clean_ar5iv_residue.py --apply            # commit changes in place
  uv run python clean_ar5iv_residue.py --audit --file X   # one file
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VAULT = Path(__file__).resolve().parents[1]
RAW_PAPERS = VAULT / "raw" / "papers" / "md"

# Atom-level residue patterns for in-body cleanup.
ATOM_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Accessibility alt text on figures (most common in-body residue).
    (re.compile(r"\[Refer to caption\]"), "", "refer-to-caption"),
    # ar5iv homepage logo wrapped in image link wrapped in link (in case it survives header strip).
    (re.compile(r"\[!\[ar5iv homepage\]\([^)]+\)\]\([^)]+\)"), "", "ar5iv-logo"),
    # LaTeXML mascot watermark (in case footer survives).
    (re.compile(r"\[LaTeXML!\[[^\]]+\]\([^)]+\)\]\([^)]+\)"), "", "latexml-mascot"),
    # `![[Uncaptioned image]](path)` — broken in Obsidian (parser sees `![[` as
    # image-embed wikilink, then `Uncaptioned image]` as a non-existent target).
    # ar5iv emits these for figures with no `\caption{}`. Image src is always an
    # ar5iv asset URL that doesn't resolve in Obsidian anyway. Drop the whole image.
    (re.compile(r"!\[\[Uncaptioned image\]\]\([^)]+\)"), "", "uncaptioned-image"),
    # Same shape but wrapped in an outer link `[![[Uncaptioned image]](src)](href)`
    # — drop the entire wrapper too. Order matters: this must come BEFORE the
    # bare image strip would over-match. (PATTERNS run in order; this is fine.)
    (re.compile(r"\[!\[\[Uncaptioned image\]\]\([^)]+\)\]\([^)]+\)"), "", "uncaptioned-image-linked"),
    # `[<arxiv-id>] Untitled Document` tagline (placeholder-file-only).
    (re.compile(r"^\[\d{4}\.\d{4,5}\] Untitled Document\s*$", re.MULTILINE), "", "untitled-doc-tagline"),
    # Bare `[Uncaptioned image]` text (no `(src)` after) — leftover label in
    # table cells where the original was a missing image. Not a link, just noise.
    (re.compile(r"\[Uncaptioned image\](?!\()"), "", "uncaptioned-image-bare"),
]

# JS function blocks that leaked from ar5iv <script>/<noscript> into placeholder
# files (where my header-chrome-block strip can't fire because there's no `# `
# heading to anchor the strip to). Applied AFTER the header/footer phases so
# the in-body content of real papers doesn't get touched (their JS lives in
# the chrome block we already removed).
JS_BLOCK_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # `function name(...) { ... }` — multi-line, balanced-ish (ar5iv JS is small).
    # We match across lines but stop at a closing `} }` followed by blank/EOF.
    (re.compile(r"^function \w+\([^)]*\)\s*\{.*?^\}\s*\}\s*$", re.MULTILINE | re.DOTALL), "", "js-function-block"),
    # Standalone JS function call lines like `detectColorScheme();`.
    (re.compile(r"^\w+\(\);\s*$", re.MULTILINE), "", "js-call-line"),
    # MathJax loader scaffold lines that survive (`var canMathML = ...`).
    (re.compile(r"^var \w+\s*=.*?(?:;|\})\s*$", re.MULTILINE), "", "js-var-decl"),
]

# Footer chrome boundary — strip from the FIRST occurrence of any of these
# ar5iv site-chrome markers to end of file. Listed in approximate document
# order; the `\[◄\]\(/html/` nav arrow is usually first.
FOOTER_BOUNDARY_RE = re.compile(
    r"\n+(?:"
    r"\[◄\]\(/html/"
    r"|\[!\[ar5iv homepage\]\("
    r"|\[Feeling\s*\n?\s*lucky\?\]"
    r"|\[Conversion\s*\n?\s*report\]"
    r"|\[Report\s*\n?\s*an issue\]"
    r"|\[View original\s*\n?\s*on arXiv\]"
    r"|\[Copyright\]\(https://arxiv\.org/help/license\)"
    r"|Generated on .+? by \[LaTeXML"
    r"|var canMathML"
    r")"
)

FRONTMATTER_RE = re.compile(r"\A(---\n.*?\n---\n)", re.DOTALL)
FIRST_HEADING_RE = re.compile(r"^# [^\n]+", re.MULTILINE)
# ar5iv emits this when it could only embed the PDF (no LaTeX source available):
#   `See pages 1-last of <0_paper-slug.pdf>` (or `1-NN`, etc.)
PDF_EMBED_RE = re.compile(r"^See pages \d+-(?:last|\d+) of <[^>]+>", re.MULTILINE)


def clean_text(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    out = text

    # Phase 1 — header chrome strip.
    fm_match = FRONTMATTER_RE.match(out)
    fm_end = fm_match.end() if fm_match else 0
    heading_match = FIRST_HEADING_RE.search(out, fm_end)
    if heading_match:
        # There is a body heading — strip everything between frontmatter end and heading start.
        chrome_chars = heading_match.start() - fm_end
        if chrome_chars > 0:
            out = out[:fm_end] + "\n" + out[heading_match.start():]
            counts["header-chrome-block"] = chrome_chars

    # Phase 2 — footer chrome strip.
    footer_match = FOOTER_BOUNDARY_RE.search(out)
    if footer_match:
        footer_chars = len(out) - footer_match.start()
        out = out[:footer_match.start()].rstrip() + "\n"
        counts["footer-chrome-block"] = footer_chars

    # Phase 3 — atom-level in-body residue.
    for pat, repl, label in ATOM_PATTERNS:
        out, n = pat.subn(repl, out)
        if n:
            counts[label] = counts.get(label, 0) + n

    # Phase 4 — JS-block fallback for placeholder files (no body heading).
    # If header-chrome-block didn't fire (no `# ` heading), the file is one of:
    #  (a) a placeholder where ar5iv only embedded the PDF (`See pages X-last of <Y.pdf>`)
    #      — these still have all the JS chrome between frontmatter and the marker.
    #  (b) a non-ar5iv source (PDF-extract, .notes.md sidecar) — no JS chrome at all.
    # For (a), drop everything between frontmatter end and the `See pages` marker.
    # For (b), the patterns below no-op.
    if "header-chrome-block" not in counts:
        pdf_embed_match = PDF_EMBED_RE.search(out, fm_end)
        if pdf_embed_match:
            chrome_chars = pdf_embed_match.start() - fm_end
            if chrome_chars > 1:  # ignore single-newline gap
                out = out[:fm_end] + "\n" + out[pdf_embed_match.start():]
                counts["placeholder-chrome-block"] = chrome_chars
        # Even after that, sweep stray JS calls / vars across the whole file.
        for pat, repl, label in JS_BLOCK_PATTERNS:
            out, n = pat.subn(repl, out)
            if n:
                counts[label] = counts.get(label, 0) + n

    # Collapse runs of >2 blank lines that the strips create.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out, counts


def process(files: list[Path], *, apply: bool) -> None:
    total_counts: dict[str, int] = {}
    files_changed = 0
    for fp in files:
        try:
            original = fp.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"!! {fp.name}: read failed: {exc}", file=sys.stderr)
            continue
        cleaned, counts = clean_text(original)
        if cleaned == original:
            continue
        files_changed += 1
        for label, n in counts.items():
            total_counts[label] = total_counts.get(label, 0) + n
        if apply:
            fp.write_text(cleaned, encoding="utf-8")
            tag = "[written]"
        else:
            tag = "[would-strip]"
        per_file = ", ".join(f"{label}={n}" for label, n in sorted(counts.items()))
        try:
            rel = fp.resolve().relative_to(VAULT)
        except ValueError:
            rel = fp
        print(f"{tag} {rel}: {per_file}")

    print()
    print("=== Summary ===")
    print(f"Files {'changed' if apply else 'would-change'}: {files_changed} of {len(files)}")
    if total_counts:
        for label, n in sorted(total_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {label:<28} {n:>6}")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit", action="store_true", help="dry-run; print what would change")
    g.add_argument("--apply", action="store_true", help="commit changes in place")
    ap.add_argument("--file", type=Path, default=None, help="restrict to one file (path)")
    args = ap.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = sorted(RAW_PAPERS.glob("*.md"))
    if not files:
        print(f"no files matched under {RAW_PAPERS}", file=sys.stderr)
        return 1

    process(files, apply=args.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
