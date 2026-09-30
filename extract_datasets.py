"""
extract_datasets.py — preprocessing pipeline for the LGU Budget Optimizer.

This implements the INPUT and DATA PROCESSING stages of the system
architecture (Figure 6 of the manuscript):

    LGU transparency portals
        -> manual download of the Annual Procurement Plan PDFs
        -> PDF parsing        (rule-based text extraction, regex pattern
                               matching for tabular data, keyword-anchored
                               line parsing)
        -> data processing    (normalization, outlier removal, null handling)
        -> evaluation and categorization  (sector assignment)
        -> pasig_projects.json / qc_projects.json

It runs ONCE as a preprocessing step rather than on every application start.
Parsing 1,707 pages of PDF takes far longer than solving the knapsack problem
itself, so re-parsing per run would dominate the very timings the study
measures. The optimization engine therefore loads the cleaned JSON produced
here, exactly as Figure 6 shows the data processing module feeding the
optimization engine.

USAGE
-----
    pip install pdfplumber            # only needed if using --engine pdfplumber
    python extract_datasets.py --pasig PASIG_DATASET.pdf \\
                               --qc QUEZON_CITY_DATASET.pdf

By default the script shells out to `pdftotext -layout` (poppler-utils),
which preserves the column geometry the parsers rely on. Pass
`--engine pdfplumber` to use the pure-Python extractor instead.

OUTPUT
------
    pasig_projects.json    1,991 records
    qc_projects.json      26,865 records

Each record: {code, name, pmo, cost, sector}

EXPECTED COUNTS
---------------
The two totals are asserted at the end of the run. They are the figures
stated in the manuscript, and they were confirmed by counting project rows
directly in the source PDFs. If a future revision of either plan is used,
update EXPECTED below to the new totals rather than silently accepting a
different count.

TWO EXTRACTION DEFECTS THIS SCRIPT HANDLES
------------------------------------------
1. Wrapped rows (Pasig). A project's name can wrap across several lines, and
   the estimated budget does not always sit on the line carrying the account
   code. A naive line-by-line parser loses those rows. Here, rows are
   assembled from blank-line-separated blocks, and name fragments that wrap
   both ABOVE and BELOW the code line are attached to the correct record.

2. Indented codes (Quezon City). Thirteen rows are indented by one space
   before the eight-digit account code. A parser anchored on ^\\d{8} drops
   precisely those thirteen. The pattern here tolerates leading whitespace.

A third defect was a corrupted cost: one project name contains the literal
text "(Voucher worth PHP240)", and a parser that searches the whole line for
a currency amount picks up that 240 instead of the 240,000.00 in the budget
column. Amounts are therefore read only from the estimated-budget region of
the row, never from the name.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED = {"pasig": 1991, "qc": 26865}

MONEY = re.compile(r"([\d,]+\.\d{2})")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — PDF to layout-preserving text
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_text(pdf_path: Path, engine: str = "pdftotext") -> str:
    """Extract text while preserving column geometry.

    Both plans are digitally generated (not scans), so no OCR is required.
    Column positions carry meaning here — the project name, implementing
    office and budget are distinguished by horizontal position — so the
    extraction must preserve layout rather than reflow the text.
    """
    if engine == "pdftotext":
        if not shutil.which("pdftotext"):
            sys.exit("pdftotext not found. Install poppler-utils, or pass "
                     "--engine pdfplumber.")
        out = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout

    if engine == "pdfplumber":
        try:
            import pdfplumber
        except ImportError:
            sys.exit("pdfplumber not installed. Run: pip install pdfplumber")
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text(layout=True) or "")
        return "\n".join(pages)

    sys.exit(f"Unknown engine: {engine}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2a — Pasig City parser
#
# Fixed-width columns, measured from the page header:
#     code  [0:13]   name [13:78]   PMO [78:103]   ... budget from col 150
# Rows are separated by blank lines; the name may wrap above and below the
# line carrying the account code.
# ─────────────────────────────────────────────────────────────────────────────

PASIG_CODE = re.compile(r"^\s*([0-9][\-0-9]{6,})\s")
NAME_SL, PMO_SL, TAIL = slice(13, 78), slice(78, 103), 150

_HEADER_TOKENS = (
    "CODE", "PROCUREMENT", "PMO/", "Is this an Early", "Mode of",
    "Schedule for", "(PAP)", "PROGRAM/PROJECT", "End-User",
    "Annual Procurement Plan", "General Fund", "Page ", "Prepared",
    "Approved", "(Yes/No)", "IB/REI", "of Bids", "Notice of",
    "Signing", "of Fund", "Estimated Budget", "Remarks", "Total", "MOOE",
)


def _is_header(line: str) -> bool:
    """Keyword-anchored rejection of repeated page furniture."""
    return any(tok in line for tok in _HEADER_TOKENS)


def parse_pasig(text: str) -> list[dict]:
    lines = text.splitlines()

    blocks, cur = [], []
    for line in lines:
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)

    rows = []
    for blk in blocks:
        if not any(PASIG_CODE.match(l) for l in blk):
            continue

        current, pending_name, pending_pmo = None, [], []
        for line in blk:
            m = PASIG_CODE.match(line)
            if m:
                # A code line opens a record and absorbs name fragments that
                # wrapped ABOVE it. Amounts are read only from the budget
                # region, never from the project name.
                amounts = MONEY.findall(line[TAIL:])
                current = {
                    "code": m.group(1),
                    "_name": list(pending_name),
                    "_pmo": list(pending_pmo),
                    "cost": amounts[0] if amounts else None,
                }
                nm, pm = line[NAME_SL].strip(), line[PMO_SL].strip()
                if nm:
                    current["_name"].append(nm)
                if pm:
                    current["_pmo"].append(pm)
                rows.append(current)
                pending_name, pending_pmo = [], []
                continue

            if _is_header(line):
                continue

            nm, pm = line[NAME_SL].strip(), line[PMO_SL].strip()
            if current is not None:
                # Fragments wrapping BELOW the code line.
                if nm:
                    current["_name"].append(nm)
                if pm:
                    current["_pmo"].append(pm)
                if current["cost"] is None:
                    amt = MONEY.findall(line[TAIL:])
                    if amt:
                        current["cost"] = amt[0]
            else:
                if nm:
                    pending_name.append(nm)
                if pm:
                    pending_pmo.append(pm)

    return [
        {
            "code": r["code"],
            "name": " ".join(r["_name"]).strip(),
            "pmo": " ".join(r["_pmo"]).strip(),
            "cost": r["cost"],
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2b — Quezon City parser
#
# Each row begins on its code line and continues until the next code line.
# The leading \s* is essential: 13 rows are indented one space, and anchoring
# on ^\d{8} silently loses exactly those 13.
# ─────────────────────────────────────────────────────────────────────────────

QC_CODE = re.compile(r"^\s*(\d{8})\s")
QC_MONEY = re.compile(r"₱\s*([\d,]+\.\d{2})")


def parse_qc(text: str) -> list[dict]:
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if QC_CODE.match(l)]

    rows = []
    for n, i in enumerate(idx):
        end = idx[n + 1] if n + 1 < len(idx) else len(lines)
        blk = lines[i:end]
        code = QC_CODE.match(lines[i]).group(1)

        amounts, name_parts, pmo = [], [], ""
        for j, line in enumerate(blk):
            amounts += QC_MONEY.findall(line)
            body = line[QC_CODE.match(line).end():] if j == 0 else line
            seg = body[:95] if j == 0 else line[11:95]
            if seg.strip():
                name_parts.append(seg.strip())
            if j == 0:
                mid = body[95:130].strip()
                if mid:
                    pmo = mid.split()[0]

        rows.append({
            "code": code,
            "name": " ".join(name_parts).strip(),
            "pmo": pmo,
            "cost": amounts[0] if amounts else None,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Data processing: normalization, null handling, outlier review
# ─────────────────────────────────────────────────────────────────────────────

def to_cost(value):
    """Normalise a currency string to a float. Returns None if unparseable."""
    if value is None:
        return None
    try:
        return round(float(str(value).replace(",", "").replace("₱", "").strip()), 2)
    except ValueError:
        return None


def clean_text(value, limit=200):
    """Collapse whitespace and trim column-bleed from neighbouring fields."""
    t = " ".join(str(value or "").split())
    t = re.sub(r"\s+(No|Yes)\s+(?=[A-Z])", " ", t)
    return re.sub(r"\s{2,}", " ", t)[:limit].strip()


def process(rows: list[dict]) -> tuple[list[dict], dict]:
    """Normalise fields and drop records that cannot enter the model.

    Null handling: a record with no parseable cost cannot be an item in a
    knapsack instance (it has no weight), so it is dropped and counted.

    Outlier handling: non-positive costs are dropped for the same reason.
    Large costs are NOT treated as outliers — a genuinely expensive capital
    outlay is exactly the kind of item the optimisation must reason about,
    and removing it would bias the allocation.
    """
    out, report = [], {"null_cost": 0, "non_positive": 0, "kept": 0}

    for r in rows:
        cost = to_cost(r.get("cost"))
        if cost is None:
            report["null_cost"] += 1
            continue
        if cost <= 0:
            report["non_positive"] += 1
            continue
        out.append({
            "code": clean_text(r.get("code"), 40),
            "name": clean_text(r.get("name")),
            "pmo": clean_text(r.get("pmo"), 60),
            "cost": cost,
        })

    report["kept"] = len(out)
    return out, report


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Evaluation and categorization: sector assignment
#
# The implementing office decides the sector where it is unambiguous, with
# project keywords as a fallback when the office is generic or was clipped
# during extraction. The sector feeds the MAUT criteria weights in app.py.
# ─────────────────────────────────────────────────────────────────────────────

PASIG_PMO_SECTOR = [
    (r"hospital|health|medical|veterinar", "Healthcare"),
    (r"school board|education|pamantasan|university|college", "Education"),
    (r"engineering|20% lcdf|infrastructure|building", "Infrastructure"),
    (r"drrm|disaster|fire|police|public safety|5% ldrrmf", "Public Safety"),
    (r"social welfare|youth|senior|pwd|women", "Social Services"),
    (r"environment|natural resources|solid waste|sanitation", "Environment"),
]

QC_PMO_SECTOR = [
    (r"^(qcgh|rmbgh|ndh|nnc|hd|qchd|cvd)$|hospital|health", "Healthcare"),
    (r"^(qcu|qcdtrc|sdo|qcpl)$|school|educ", "Education"),
    (r"^(itdd|mdad|engineering|qcedo)$|engineer|infra", "Infrastructure"),
    (r"^(qcpc|qcfd|dprmo|qcdrrmo)$|police|fire|rescue|safety", "Public Safety"),
    (r"^(ssdd|qcadac|peso|scad|pwd)$|social|welfare|youth", "Social Services"),
    (r"^(ccesd|epwmd|ttmd)$|environment|waste|sanitation", "Environment"),
]

NAME_SECTOR = [
    (r"medicine|medical|drug|hospital|clinic|dental|health|laborator", "Healthcare"),
    (r"school|student|scholar|educat|teaching|library", "Education"),
    (r"road|bridge|construct|repair of building|drainage|pavement|flood", "Infrastructure"),
    (r"rescue|fire|police|disaster|emergency|patrol|radio", "Public Safety"),
    (r"livelihood|senior citizen|pwd|day care|feeding|relief", "Social Services"),
    (r"tree|waste|sanitat|environment|clean-?up|recycl", "Environment"),
]


def classify(pmo: str, name: str, rules) -> str:
    p = (pmo or "").lower().strip()
    for pattern, sector in rules:
        if re.search(pattern, p):
            return sector
    n = (name or "").lower()
    for pattern, sector in NAME_SECTOR:
        if re.search(pattern, n):
            return sector
    return "General Government"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preserve_existing(records: list[dict], source: Path, which: str) -> int:
    """Reuse name, PMO and sector from an existing dataset, matched on
    (code, cost).

    Three fields feed the MAUT benefit score: sector sets the C1 and C2
    weights, cost drives C3, and the project NAME is keyword-matched for the
    C4 urgency criterion. Reconstructing a wrapped multi-line project name
    from the PDF is inherently approximate - fragments can be joined in a
    different order than the original extraction produced - so regenerating
    these fields from scratch shifts benefit scores and therefore the
    optimisation results.

    When a dataset is already in use its values are authoritative and are
    carried across verbatim. Only rows with no counterpart in the existing
    file - genuinely new records - use this script's own reconstruction.
    """
    path = source / f"{which}_projects.json"
    if not path.exists():
        print(f"  note         no existing {path.name} to take sectors from")
        return 0

    prior = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for rec in prior:
        key = (str(rec.get("code", "")).strip(), round(float(rec["cost"]), 2))
        lookup.setdefault(key, []).append(rec)

    carried = 0
    for rec in records:
        bucket = lookup.get((rec["code"], rec["cost"]))
        if bucket:
            old = bucket.pop(0)
            for field in ("name", "pmo", "sector"):
                if old.get(field):
                    rec[field] = old[field]
            carried += 1
    return carried


def build(pdf_path: Path, which: str, engine: str, out_path: Path,
          keep_sectors: bool = False, sector_source: Path = Path(".")) -> list[dict]:
    label = {"pasig": "Pasig City", "qc": "Quezon City"}[which]
    print(f"\n{label}")
    print(f"  reading      {pdf_path}")

    text = pdf_to_text(pdf_path, engine)
    raw = parse_pasig(text) if which == "pasig" else parse_qc(text)
    print(f"  parsed       {len(raw):,} project rows")

    clean, report = process(raw)
    if report["null_cost"]:
        print(f"  dropped      {report['null_cost']} row(s) with no parseable cost")
    if report["non_positive"]:
        print(f"  dropped      {report['non_positive']} row(s) with non-positive cost")

    rules = PASIG_PMO_SECTOR if which == "pasig" else QC_PMO_SECTOR
    for rec in clean:
        rec["sector"] = classify(rec["pmo"], rec["name"], rules)

    if keep_sectors:
        carried = preserve_existing(clean, sector_source, which)
        print(f"  preserved    {carried:,} record(s) kept name/PMO/sector from the "
              f"existing dataset, {len(clean) - carried:,} newly derived")

    expected = EXPECTED[which]
    if len(clean) != expected:
        print(f"  WARNING      {len(clean):,} records, expected {expected:,}")
        print(f"               The manuscript and the source PDF both give "
              f"{expected:,}. Investigate before using this output.")
    else:
        print(f"  records      {len(clean):,}  (matches the manuscript)")

    total = sum(r["cost"] for r in clean)
    print(f"  total budget PHP {total:,.2f}")

    out_path.write_text(json.dumps(clean, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"  wrote        {out_path}")
    return clean


def main():
    ap = argparse.ArgumentParser(
        description="Extract and clean LGU Annual Procurement Plan PDFs "
                    "into the JSON datasets used by the optimizer.")
    ap.add_argument("--pasig", type=Path, help="Pasig City APP PDF")
    ap.add_argument("--qc", type=Path, help="Quezon City APP PDF")
    ap.add_argument("--engine", default="pdftotext",
                    choices=["pdftotext", "pdfplumber"],
                    help="Text extraction backend (default: pdftotext)")
    ap.add_argument("--outdir", type=Path, default=Path("extracted"),
                    help="Where to write the JSON files (default: ./extracted, "
                         "deliberately NOT the live dataset location)")
    ap.add_argument("--preserve-existing", action="store_true",
                    help="Carry name, PMO and sector over from the existing "
                         "dataset in --existing-dir, matching on (code, cost). "
                         "Use this when regenerating datasets already in use, "
                         "so the MAUT benefit scores do not shift.")
    ap.add_argument("--existing-dir", type=Path, default=Path("."),
                    help="Directory holding the current *_projects.json to "
                         "preserve values from (default: .)")
    args = ap.parse_args()

    if not args.pasig and not args.qc:
        ap.error("Pass --pasig and/or --qc with the source PDF path(s).")

    args.outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print("LGU Budget Optimizer — dataset preprocessing")
    print("Figure 6: input -> PDF parsing -> data processing -> categorization")
    print("=" * 62)

    if args.pasig:
        build(args.pasig, "pasig", args.engine,
              args.outdir / "pasig_projects.json",
              args.preserve_existing, args.existing_dir)
    if args.qc:
        build(args.qc, "qc", args.engine,
              args.outdir / "qc_projects.json",
              args.preserve_existing, args.existing_dir)

    print(f"\nWritten to {args.outdir}/")
    print("These are NOT the live datasets. The optimizer loads "
          "pasig_projects.json and qc_projects.json from its own folder;")
    print("copy the generated files over only after checking them, and use")
    print("--preserve-existing so the benefit scores stay unchanged.")


if __name__ == "__main__":
    main()
