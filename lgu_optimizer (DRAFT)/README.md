# LGU Budget Optimizer
## Hybrid 0/1 Knapsack with Branch-and-Bound and Genetic Algorithm
### BSCS 3-1N · Thesis Group 2 · Polytechnic University of the Philippines

---

## Requirements

```
pip install flask pdfplumber
```

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

## API Endpoints

- GET /              — Main UI
- GET /api/meta      — Dataset counts
- GET /api/projects  — Full project list (?ds=pasig or ?ds=qc)
- POST /api/run      — Run algorithms

POST /api/run body:
{
  "ds":       "pasig",
  "budget":   5000000000,
  "selected": [0, 1, 2, ...],
  "algos":    ["dp", "bnb", "ga"],
  "pop_size": 60,
  "gens":     100,
  "mut_rate": 0.03
}
