"""Cross-check every \\cite key in a .tex file against a .bib file.

Usage:
    python scripts/check_cites.py <paper.tex> <refs.bib>
    python scripts/check_cites.py ../paper/paper.tex ../paper/refs.bib
"""

import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    tex_path, bib_path = Path(sys.argv[1]), Path(sys.argv[2])
    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")

    # \cite, \citep, \citet, \Cite ... with optional [..] arguments
    keys = set()
    for m in re.finditer(r"\\[Cc]ite[a-zA-Z]*(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        for k in m.group(1).split(","):
            if k.strip():
                keys.add(k.strip())

    defined = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib))

    print(f"cited in paper : {len(keys)}")
    print(f"defined in bib : {len(defined)}")

    missing = keys - defined
    unused = defined - keys
    print("MISSING from bib:", sorted(missing) if missing else "none")
    print("unused in bib   :", sorted(unused) if unused else "none")

    todos = re.findall(r"\\todo\{([^}]*)\}", tex, flags=re.S)
    print(f"\n\\todo markers remaining: {len(todos)}")
    for t in todos:
        print(f"  - {' '.join(t.split())}")

    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
