"""Extract PDF text to .txt so it can be read without the PDF parser.

Usage:  python scripts/extract_pdfs.py [name-substring ...]
Writes papers/txt/<name>.txt
"""

import sys
from pathlib import Path

from pypdf import PdfReader

SRC = Path("papers")
DST = SRC / "txt"
DST.mkdir(exist_ok=True)

wanted = [a.lower() for a in sys.argv[1:]]

for f in sorted(SRC.glob("*.pdf")):
    if wanted and not any(w in f.name.lower() for w in wanted):
        continue
    out = DST / (f.stem + ".txt")
    try:
        reader = PdfReader(f)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                print(f"{f.name}: ENCRYPTED, cannot open")
                continue
        chunks = []
        for i, page in enumerate(reader.pages, 1):
            try:
                chunks.append(f"\n\n===== PAGE {i} =====\n" + (page.extract_text() or ""))
            except Exception as e:
                chunks.append(f"\n\n===== PAGE {i} (extract failed: {e}) =====\n")
        text = "".join(chunks)
        out.write_text(text, encoding="utf-8")
        print(f"{f.name:30} -> {out.name:34} {len(reader.pages):3} pages, "
              f"{len(text):,} chars")
    except Exception as e:
        print(f"{f.name}: FAILED {type(e).__name__}: {e}")
