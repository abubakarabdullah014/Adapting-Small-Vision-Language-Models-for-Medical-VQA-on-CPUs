"""Lightweight paper quality checks: spelling heuristics, figure/table refs, bib keys."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def collect_tex() -> str:
    parts = []
    main = PAPER / "main.tex"
    parts.append(main.read_text(encoding="utf-8"))
    for p in sorted((PAPER / "sections").glob("*.tex")):
        parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


def main():
    text = collect_tex()
    issues = []

    # Unresolved placeholders that look unfinished
    for m in re.finditer(r"\bTBD\b|\bTODO\b|\?\?\?", text):
        issues.append(f"Placeholder near: ...{text[max(0,m.start()-30):m.end()+30]}...")

    # Cite keys
    cites = set(re.findall(r"\\cite\{([^}]+)\}", text))
    keys = set()
    for group in cites:
        for k in group.split(","):
            keys.add(k.strip())
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = sorted(keys - bib_keys)
    if missing:
        issues.append(f"Missing bib keys: {missing}")

    # Figure/table references
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    # Also labels in inputted tables
    for p in (PAPER / "tables").glob("*.tex"):
        labels |= set(re.findall(r"\\label\{([^}]+)\}", p.read_text(encoding="utf-8")))
    refs = set()
    for group in re.findall(r"\\(?:C)?ref\{([^}]+)\}", text):
        for k in group.split(","):
            refs.add(k.strip())
    dangling = sorted(refs - labels)
    if dangling:
        issues.append(f"Dangling refs (may resolve after table gen): {dangling}")

    # Common typos
    typos = {
        "teh ": "the",
        "seperate": "separate",
        "occured": "occurred",
        "performace": "performance",
        "quantisation": "quantization (US) or keep BE consistent",
        "adapation": "adaptation",
    }
    low = text.lower()
    for bad, fix in typos.items():
        if bad in low:
            issues.append(f"Possible typo '{bad.strip()}' -> {fix}")

    out = PAPER / "proofread_report.txt"
    if not issues:
        msg = "No issues found by automated checks.\n"
    else:
        msg = "Issues:\n" + "\n".join(f"- {i}" for i in issues) + "\n"
    out.write_text(msg, encoding="utf-8")
    print(msg)


if __name__ == "__main__":
    main()
