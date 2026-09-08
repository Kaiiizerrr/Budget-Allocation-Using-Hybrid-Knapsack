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
| `pasig_projects.json` | Pasig City APP FY 2025 — 1,979 projects (extracted from PDF) |
| `qc_projects.json` | Quezon City APP FY 2025 — 26,192 projects (extracted from PDF) |

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
