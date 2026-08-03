"""Scan submission assets for strings that would de-anonymise a blind submission.

Detects leaks by PATTERN (emails, absolute paths, code-host URLs) rather than by
hardcoding the authors' names -- a scanner that contains the author list is
itself a leak if it ships in the artifact.

Usage:
    python scripts/check_anon.py <paper-dir> [extra-needle ...]
    python scripts/check_anon.py ../paper Smith acme.edu

Extra needles are optional literal strings to also search for. Pass them on the
command line; do not commit them.
"""

import re
import sys
from pathlib import Path

# Patterns that identify a person or machine regardless of who they are.
PATTERNS = {
    "email address": re.compile(rb"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
    "windows path": re.compile(rb"[A-Za-z]:\\\\?Users\\\\?[^\\\\\s\"')]+"),
    "unix home path": re.compile(rb"/(?:home|Users)/[^/\s\"')]+"),
    "code host URL": re.compile(rb"https?://(?:www\.)?(?:github|gitlab|bitbucket)\.com/[^\s\"')]+"),
    "cloud sync dir": re.compile(rb"(?:Dropbox|OneDrive|Google Drive)"),
    "PDF author tag": re.compile(rb"/Author\s*\(([^)]{1,120})\)"),
}

SKIP_DIRS = {"data", "papers", "__pycache__", ".git", "figures_raw"}


def scan(path: Path, extra: list[str]) -> list[str]:
    raw = path.read_bytes()
    found = []
    for name, pat in PATTERNS.items():
        for m in pat.finditer(raw):
            hit = m.group(0).decode("latin1", "replace")
            found.append(f"{name}: {hit[:100]}")
    for needle in extra:
        if needle.encode() in raw:
            found.append(f"literal: {needle}")
    return sorted(set(found))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1]).expanduser().resolve()
    extra = sys.argv[2:]
    if not target.exists():
        sys.exit(f"no such path: {target}")

    # Skip this file: it contains the detector patterns themselves, which would
    # otherwise self-report as leaks.
    me = Path(__file__).resolve()
    files = [target] if target.is_file() else [
        p for p in sorted(target.rglob("*"))
        if p.is_file()
        and p.resolve() != me
        and not any(d in p.parts for d in SKIP_DIRS)
        and p.suffix.lower() in {".tex", ".bib", ".pdf", ".py", ".md", ".txt", ".json", ".cls", ".sty"}
    ]

    total = 0
    for f in files:
        hits = scan(f, extra)
        rel = f.relative_to(target) if target.is_dir() else f.name
        if hits:
            total += len(hits)
            print(f"\n{rel}")
            for h in hits:
                print(f"    {h}")
        else:
            print(f"{rel}: clean")

    print(f"\n{total} potential leak(s) across {len(files)} file(s)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
