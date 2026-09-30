# LGU Budget Optimizer
## Hybrid 0/1 Knapsack with Branch-and-Bound and Genetic Algorithm
### BSCS 3-1N · Thesis Group 2 · Polytechnic University of the Philippines

---

## Requirements

```
pip install flask pdfplumber openpyxl
```

> `openpyxl` is required for the **Export to Excel** feature.

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Flask application + all 3 algorithms |
| `pasig_projects.json` | Pasig City APP FY 2025 — 1,991 projects (matches manuscript) |
| `qc_projects.json` | Quezon City APP FY 2025 — 26,865 projects (matches manuscript) |

## Running

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## Datasets

| Dataset | City | Projects | Source |
|---------|------|----------|--------|
| Small | Pasig City | 1,979 | Annual Procurement Plan FY 2025 (General Fund) |
| Large | Quezon City | 26,192 | Annual Procurement Plan FY 2025 (4th Quarter) |

Both datasets were extracted directly from the official LGU transparency portal PDFs using pdfplumber.

## Algorithms

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Knapsack (DP) | O(n·W) | O(n·W) |
| Knapsack + B&B | O(2n) worst / O(n log n) avg | O(n) |
| Knapsack + B&B + GA | O(P*G*n + n log n) avg | O(P*n) |

## Interface

The tool is laid out as a six-step protocol, with a status bar docked at the
bottom that always shows the current dataset, budget, project selection,
algorithms and run count.

1. **Choose a dataset** — Pasig (small) or Quezon City (large)
2. **Set the budget ceiling** — the knapsack capacity
3. **Select candidate projects** — search, filter by sector, paginate
4. **Configure the run** — algorithms and independent runs per algorithm
5. **Compare the results** — optimality metrics per algorithm
6. **Review and export run history** — per-run log and Excel export

The Run button stays disabled until a project and an algorithm are selected,
and states the reason inline rather than failing after the click.

Fonts are loaded from Google Fonts; without internet the tool falls back to
system typefaces and remains fully usable.

## Run History & Excel Export

Every algorithm run is automatically recorded. The log keeps the **last 30 runs
for each dataset + algorithm combination** — matching the four experiment tables
in the manuscript:

| Combination | Manuscript table |
|---|---|
| KB (Pure B&B) · Pasig | Experiment Paper — B&B, Small |
| KB (Pure B&B) · Quezon City | Experiment Paper — B&B, Large |
| KBG (B&B + GA) · Pasig | Experiment Paper — Hybrid, Small |
| KBG (B&B + GA) · Quezon City | Experiment Paper — Hybrid, Large |

**Running 30 trials:** set *Independent runs per algorithm* to `30` (or click the
`30` preset) before pressing Run. Ch.3 steps 5–6 require exactly 30 independent
runs per algorithm per dataset.

**Exporting:** click **Export to Excel** in the run history panel (step 6). The
workbook contains:

- One sheet per algorithm, laid out like the manuscript's blank experiment
  tables — Trials 1–30 down the rows; Runtime, Execution Time and Pruning Rate
  for both datasets across the columns.
- **Mean, Variance and Standard Deviation** rows written as live Excel formulas
  (`=AVERAGE`, `=VAR`, `=STDEV`), so they recalculate if trials are edited.
- An **SPSS Data** sheet in wide format — one row per trial, one column per
  dataset/algorithm/metric (`KB_Runtime_Pasig`, `KBG_Runtime_Pasig`, …). Values
  are literal numbers, not formulas, so SPSS reads them directly.
- An **SPSS Codebook** sheet defining every variable, plus step-by-step notes
  for running the paired- and independent-samples t-tests.
- A **Run Log** sheet with every recorded trial and all captured fields
  (nodes explored/pruned, total benefit, projects funded, GA parameters, etc.).

## Statistical Report

The tool computes the tests specified in the Statistical Treatment section:

| Test | Purpose | Equations |
|---|---|---|
| Paired t-test | B&B vs. B&B + GA on identical instances | 4-7 |
| Independent t-test | Small vs. large dataset (efficiency) | 1-3 |

Efficiency is the ratio of each optimality parameter on the Pasig dataset
relative to Quezon City. p-values use the exact Student's t distribution via
the regularised incomplete beta function (no SciPy required), validated against
SciPy to machine precision.

A metric constant across runs (typically pruning rate, since branch-and-bound
is deterministic) has zero standard deviation, so t is undefined. The tool
reports this as **not computable** rather than hiding it.

The report appears as step 7 in the interface and as a **Statistical Report**
sheet in the Excel export.

### Analysing in SPSS

`File > Open > Data`, set *Files of type* to Excel, choose the **SPSS Data**
worksheet, and tick *Read variable names from the first row of data*.

- **Paired-samples t-test** (KB vs KBG): `Analyze > Compare Means >
  Paired-Samples T Test`, pairing e.g. `KB_Runtime_Pasig` with
  `KBG_Runtime_Pasig`. Pairing is by problem instance — both algorithms solve
  the identical project set under the identical budget.
- **Independent-samples t-test** (small vs large dataset): restructure the two
  dataset columns into one variable with a grouping variable via
  `Data > Restructure`.

> Pruning-rate columns can have zero variance, because branch-and-bound is
> deterministic and explores the same tree every run. SPSS cannot compute *t*
> for a constant; this is a property of the algorithm, not a data error.

> ⚠ The log is held **in memory only** — restarting `app.py` clears it.
> Export before shutting the tool down.

### Metric definitions (Ch.1, Definition of Terms)

- **Runtime** — active algorithm execution only (excludes sorting / setup).
- **Execution Time** — total duration including setup and final output.

## API Endpoints

- GET  /                   — Main UI
- GET  /api/meta           — Dataset counts
- GET  /api/projects       — Full project list (?ds=pasig or ?ds=qc)
- POST /api/run            — Run algorithms (each run is logged)
- GET  /api/history        — Current contents of the 30-run log
- POST /api/history/clear  — Clear the log (optionally scoped to ds + algo)
- GET  /api/stats          — Statistical report (t-tests, efficiency ratios)
- GET  /api/export         — Download the experiment workbook (.xlsx)

POST /api/run body:
{
  "ds":       "pasig",
  "budget":   5000000000,
  "selected": [0, 1, 2, ...],
  "algos":    ["dp", "bnb", "ga"],
  "trials":   30,
  "pop_size": 60,
  "gens":     100,
  "mut_rate": 0.03
}


## Dataset Reconciliation

Both datasets were verified against the source APP PDFs and now match the
counts stated in the manuscript exactly:

| Dataset | Manuscript | PDF (verified) | Tool |
|---|---|---|---|
| Pasig City | 1,991 | 1,991 | 1,991 |
| Quezon City | 26,865 | 26,865 | 26,865 |

Three defects in the earlier extraction were corrected:

1. **Pasig — 7 rows missing.** Rows whose project name wrapped across several
   lines were lost when the amount did not sit on the same line as the account
   code.
2. **Quezon City — 13 rows missing.** Thirteen rows are indented by one space
   before the account code; the original parser anchored on `^\d{8}` and
   dropped precisely those.
3. **Quezon City — one corrupted cost.** "Lunch for Participants at the
   Heritage & Food Bike Tour (Voucher worth PHP240)" was stored at ₱240.00
   because the parser matched the amount *inside the project name* instead of
   the estimated-budget column. Its true cost is ₱240,000.00.

Existing correct records were left untouched; only the missing rows were
appended and the single bad cost corrected. Sectors for the added rows follow
the classification already present in the data (e.g. TTMD → Environment,
matching 87 of the 91 existing TTMD rows).


## Interface notes

**Sequential phases.** Each phase ends with a button to the next. The Setup
phase cannot be left until at least one project is selected.

**Live progress.** Runs stream over Server-Sent Events (`/api/run/stream`).
Each algorithm has its own lane showing the current trial, last runtime and
running means, and reports its result the moment it finishes its own trials.

Trials are **interleaved** (trial 1 of each algorithm, then trial 2, and so
on) rather than run in parallel threads. CPython's global interpreter lock
means two CPU-bound solvers in threads would compete for one core and inflate
each other's measured runtimes - the very numbers the paired t-test compares.
Interleaving keeps every measurement contention-free while still letting both
progress bars advance together.

**Dynamic programming is a baseline, off by default.** DP is a third exact
method used to confirm that branch-and-bound and the hybrid reach the same
optimum. It is not one of the two variations compared in SOP 3, so it starts
unchecked and is opted into deliberately.

## Verification performed

- All three algorithms were checked against exhaustive brute-force
  enumeration on small instances: every result matched the true optimum.
- The two exact methods were compared on larger random instances (n up to
  300) with no disagreements.
- No solution exceeded the budget constraint across repeated runs.
- t-statistics and p-values were validated against SciPy to ~1e-14.
- Dataset counts were reconciled against the source APP PDFs
  (Pasig 1,991; Quezon City 26,865).


## Preprocessing (`extract_datasets.py`)

Implements the INPUT and DATA PROCESSING stages of Figure 6 — manual
download, PDF parsing (rule-based extraction, regex table matching,
keyword-anchored line parsing), normalization, null handling, and sector
categorization — producing the JSON the optimizer loads.

It runs once as a preprocessing step, not on every application start:
parsing 1,707 pages takes far longer than solving the knapsack problem, so
re-parsing per run would dominate the very timings the study measures.

```
python extract_datasets.py --pasig PASIG_DATASET.pdf --qc QUEZON_CITY_DATASET.pdf
```

Output goes to `./extracted/` — deliberately NOT over the live datasets.

### Regenerating datasets already in use

```
python extract_datasets.py --pasig ... --qc ... --preserve-existing --existing-dir .
```

Sector, project name and cost all feed the MAUT benefit score, and
reconstructing a wrapped multi-line project name from a PDF is inherently
approximate. `--preserve-existing` carries name, PMO and sector across from
the current dataset by matching on (code, cost), so regeneration cannot shift
any benefit score. Verified: benefit totals come out identical to the live
data for both cities.

### Extraction defects this script handles

1. **Wrapped rows (Pasig).** A project name can wrap above and below the line
   carrying the account code, and the budget is not always on that line. A
   naive line parser loses 7 rows.
2. **Indented codes (Quezon City).** 13 rows are indented one space before the
   account code; anchoring on `^\d{8}` drops exactly those 13.
3. **Amounts inside project names.** One project reads "(Voucher worth
   PHP240)"; searching the whole line for a currency amount captures 240
   instead of the real 240,000.00. Amounts are read only from the
   estimated-budget column.


## Verification & Review phase

Implements step 7 of the Data Generation/Gathering Procedure (post-allocation
verification and comparative review). The manuscript specifies a MANUAL
verification phase; the tool establishes the mechanical facts, and the manual
judgement is recorded in the exported workbook.

**Automated integrity checks** run over every funded project: each appears in
the source Annual Procurement Plan, each has a positive estimated budget, each
benefit value falls within the 0-10 scale, the total cost is within the budget
ceiling, and no project is funded twice.

**Sector alignment** compares each sector's share of the candidate plan with
its share of the funded set, which is the evidence for whether the allocation
reflects the local government's own priorities.

**Comparative review** reports candidates, projects funded, budget spent, total
benefit and benefit per PhP 1 million for each dataset, for the scaling
question. It appears once an allocation has been run on both cities.

**Funded projects** lists the complete allocation for each city in the tool.
Pick a dataset and algorithm, then search, filter by sector, sort by benefit,
cost, benefit per PhP 1 million or name, and page through the results. Paging
happens on the server, because a Quezon City allocation can exceed 25,000
projects. It is the same list the export contains.

**The manual review is done in Excel.** Each funded-project sheet carries empty
**Reviewer verdict** and **Reviewer note** columns, the verdict column having a
dropdown of aligned / not aligned / flagged. The reviewer therefore works over
the complete allocation with Excel's filtering and sorting, rather than over a
sample in the browser, and the completed sheet is the evidence for the manual
phase.

> The budget matters here. At a generous ceiling nearly every project is funded
> and the review is uninformative; at a selective budget real trade-offs appear
> and the sector differences become meaningful. State the budget used.

### Funded project sheets

The export includes one sheet per city - **Funded - Pasig** and
**Funded - Quezon City** - each listing every project funded by the most
recent run, with an Algorithm column so both variations can be compared within
the same city. Columns: account code, project name, implementing office,
sector, cost, benefit, and benefit per PhP 1 million.

Each sheet carries the budget ceiling at the top, an autofilter and a frozen
header, and per-algorithm totals at the bottom written as live SUMIF formulas
so they follow any filtering or editing.

A third sheet, **Funded - All**, repeats the same rows with a Dataset column
and its header on row 1, so both cities can be compared in a single
PivotTable - something the split sheets cannot do. Because it duplicates every
row it roughly doubles the workbook: about 9 MB and 20 seconds, against 4.5 MB
and 10 seconds without it. The checkbox beside the Export button controls
whether it is included.

This is the complete allocation, as distinct from the purposive sample used in
the manual review. At a selective budget the Quezon City sheet runs to roughly
52,000 rows.
