"""One-shot migration: replace a hand-written thebibliography block with
BibTeX commands (\\bibliographystyle + \\bibliography).

Already applied to the paper; kept for provenance and re-use.

Usage:
    python scripts/swap_bib.py <paper.tex> [bibname]
"""

import sys
from pathlib import Path

START_MARKER = r"\begin{thebibliography}"
END_MARKER = r"\end{thebibliography}"

TEMPLATE = r"""%% Bibliography: BibTeX + the official ACM style, as ASE requires.
%% ACM-Reference-Format.bst ships with acmart, so no extra files are needed.
%% Compile order: pdflatex, bibtex, pdflatex, pdflatex (Overleaf automates this).
\bibliographystyle{ACM-Reference-Format}
\bibliography{%s}"""


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    paper = Path(sys.argv[1])
    bibname = sys.argv[2] if len(sys.argv) > 2 else "refs"

    text = paper.read_text(encoding="utf-8")
    if START_MARKER not in text:
        sys.exit("no thebibliography block found -- already migrated?")

    i = text.index(START_MARKER)
    j = text.index(END_MARKER) + len(END_MARKER)
    removed = text[i:j]

    paper.write_text(text[:i] + (TEMPLATE % bibname) + text[j:], encoding="utf-8")
    print(f"removed {removed.count(chr(10)) + 1} lines "
          f"({removed.count(chr(92) + 'bibitem')} bibitems)")
    print(f"replaced with \\bibliography{{{bibname}}}")


if __name__ == "__main__":
    main()
