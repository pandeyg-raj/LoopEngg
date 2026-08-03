"""Which downloaded PDFs are actually readable, and which are encrypted?"""

from pathlib import Path

PAPERS = Path("papers")

for f in sorted(PAPERS.glob("*.pdf")):
    raw = f.read_bytes()
    encrypted = b"/Encrypt" in raw
    # Rough page count from the page-tree objects.
    pages = raw.count(b"/Type /Page") + raw.count(b"/Type/Page")
    print(f"{f.name:30} {'ENCRYPTED' if encrypted else 'clean':10} "
          f"{len(raw) / 1e6:5.2f} MB  ~{pages} page objs")
