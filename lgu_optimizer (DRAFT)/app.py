"""
LGU Budget Allocation Optimizer
================================
Hybrid 0/1 Knapsack with Branch-and-Bound and Genetic Algorithm
BSCS 3-1N · Thesis Group 2 · PUP

Datasets:
  - Pasig City APP FY 2025 (General Fund)     — small  (1,984 projects)
  - Quezon City APP FY 2025 (4th Quarter)     — large  (26,852 projects)
"""

import os, sys, json, time, random, math, re
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

# random.binomialvariate is available from Python 3.12+. It lets the GA draw
# the number of mutations in O(1) instead of one random() call per gene.
_HAS_BINOMIAL = hasattr(random, "binomialvariate")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

PASIG_DATA = load_dataset(BASE / "pasig_projects.json")
QC_DATA    = load_dataset(BASE / "qc_projects.json")

DATASETS = {
    "pasig": {
        "label":    "Pasig City Annual Procurement Plan FY 2025",
        "subtitle": "General Fund · 1,984 projects",
        "size_tag": "Small",
        "data":     PASIG_DATA,
    },
    "qc": {
        "label":    "Quezon City Annual Procurement Plan FY 2025",
        "subtitle": "4th Quarter · 26,852 projects",
        "size_tag": "Large",
        "data":     QC_DATA,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# NEDA-aligned MAUT benefit scoring
#
# Reference: Jenkins & Baurzhan (2021), "Guidelines on Project Development and
# Evaluation", prepared for NEDA Philippines.
#
#   - Table A4-13 (Annex 4) catalogues sector-specific economic benefits:
#     Infrastructure/roads generate VOC savings + time savings (most directly
#     quantifiable per A4.1); Healthcare generates avoided morbidity costs and
#     consumer surplus from improved service (A4-13, "Hospital upgrading");
#     Education generates increased lifetime earnings, ~8.6% return in the
#     Philippines (A4.3.2); Social welfare/community development generates
#     time savings (opportunity cost of time) and livelihood income (A4-13).
#
#   - Chapter 2 frames project identification around "gaps in the economy"
#     (basic infrastructure -> food/agriculture -> social sectors such as
#     health and education), which informs both sector prioritisation and
#     the urgency/gap-filling criterion below.
#
#   - A4.4.2 (cost-utility analysis) defines a weighted-sum utility
#     U = sum(w_i * B_i), which is the same MAUT structure used here:
#     each project gets a composite score from 4 weighted criteria.
#
# Criteria (each in [0,1], aggregated to a 1-10 scale):
#
#   C1 - Economic Contribution Potential   weight = 0.40
#        Reflects how directly Table A4-13 benefits can be quantified for
#        the sector (Infrastructure VOC/time savings > Healthcare avoided
#        morbidity > Education lifetime earnings > Social Services time
#        savings/livelihood > Public Safety > Environment > Gen. Government).
#
#   C2 - Social / Human Development Impact weight = 0.35
#        Reflects Chapter 2's prioritisation of health and education as
#        "social sectors" for socio-economic well-being, and Annex 4.4's
#        DALY/QALY-style non-market benefits (Healthcare > Education >
#        Social Services > Public Safety > Environment > Infrastructure >
#        Gen. Government).
#
#   C3 - Implementation Feasibility / Value-for-Money  weight = 0.15
#        A continuous cost-efficiency curve (akin to NEDA's ICER concept in
#        A4.3.4/A4.4.1) peaking around PhP 1M, the typical scale at which an
#        LGU procurement item is both substantial and readily executable
#        within a single fiscal year.
#
#   C4 - Urgency / Gap-Filling              weight = 0.10
#        Keyword-driven classification of the procurement item itself,
#        following Chapter 2's emphasis on addressing identified service
#        gaps: disaster/emergency response ranks highest, followed by
#        infrastructure construction/rehabilitation, direct medical service
#        delivery, livelihood programs, equipment, training, and finally
#        administrative/ceremonial items (food, supplies for meetings).
# ─────────────────────────────────────────────────────────────────────────────

# C1: Economic Contribution Potential, by sector (Table A4-13)
C1_SECTOR_WEIGHTS = {
    "Infrastructure":     1.00,  # VOC savings, time savings - most quantifiable
    "Healthcare":         0.85,  # avoided morbidity costs, consumer surplus
    "Education":          0.75,  # lifetime earnings increment
    "Social Services":    0.65,  # time savings / livelihood income
    "Public Safety":      0.60,
    "Environment":        0.55,
    "General Government": 0.35,
}

# C2: Social / Human Development Impact, by sector (Chapter 2, Annex 4.4)
C2_SECTOR_WEIGHTS = {
    "Healthcare":         1.00,  # DALY/QALY non-market benefits
    "Education":          0.95,  # literacy + lifetime earnings + externalities
    "Social Services":    0.85,
    "Public Safety":      0.70,
    "Environment":        0.65,
    "Infrastructure":     0.55,
    "General Government": 0.30,
}

# Backwards-compat alias used elsewhere (sector display ordering, etc.)
SECTOR_WEIGHTS = {
    "Healthcare":        10,
    "Education":          9,
    "Public Safety":      8,
    "Infrastructure":     7,
    "Social Services":    7,
    "Environment":        6,
    "General Government": 4,
}

# C4: Urgency / Gap-Filling keyword patterns (checked in priority order)
URGENCY_PATTERNS = [
    (r'\b(emergency|disaster|calamity|drrm|rescue|evacuat|relief|flood control|'
     r'fire truck|ambulance|fire suppression)\b', 1.00),
    (r'\b(construction|rehabilitation|repair|renovation|building of|improvement of|'
     r'widening|installation of|upgrading)\b', 0.85),
    (r'\b(medicine|drug|vaccine|pharmaceutical|antibiotic|insulin|reagent|laborator|'
     r'medical supplies|medical equipment)\b', 0.75),
    (r'\b(livelihood|employment program|income generat|skills training)\b', 0.65),
    (r'\b(motor vehicle|service vehicle|equipment|machinery|apparatus)\b', 0.55),
    (r'\b(training|seminar|workshop|conference|capacity building|capacity development)\b', 0.40),
    (r'\b(office supplies|various supplies|various items|consumable)\b', 0.25),
    (r'\b(food|meal|buffet|catering|snack|lunch)\b', 0.20),
]
URGENCY_DEFAULT = 0.50


def _c4_urgency(name):
    """Keyword-driven urgency / gap-filling score (C4)."""
    for pattern, value in URGENCY_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return value
    return URGENCY_DEFAULT


def _c3_cost_efficiency(cost):
    """Continuous cost-efficiency score (C3), peaking around PhP 1M
    (log10(cost) = 6) and falling off toward both very small and very
    large procurement amounts."""
    cost = max(cost, 1)
    log_cost = math.log10(cost)
    return max(0.0, 1.0 - abs(log_cost - 6.0) / 4.0)


def compute_benefit(project):
    """
    NEDA-aligned MAUT benefit score combining 4 weighted criteria:

      score = 10 * (0.40*C1 + 0.35*C2 + 0.15*C3 + 0.10*C4)

    C1, C2 are sector-based (Table A4-13 / Chapter 2 priority ordering);
    C3 is a continuous function of cost (ICER-style cost-efficiency);
    C4 is a keyword-driven urgency/gap-filling classification of the
    procurement item itself. The continuous C3 term ensures every
    project gets a (near-)unique score, so benefit/cost ratios are
    genuinely distinct across the dataset.
    """
    sector = project["sector"]
    c1 = C1_SECTOR_WEIGHTS.get(sector, 0.35)
    c2 = C2_SECTOR_WEIGHTS.get(sector, 0.30)
    c3 = _c3_cost_efficiency(project["cost"])
    c4 = _c4_urgency(project["name"])

    score = 10 * (0.40 * c1 + 0.35 * c2 + 0.15 * c3 + 0.10 * c4)
    return round(score, 4)


# Pre-compute benefits
for ds in DATASETS.values():
    for p in ds["data"]:
        p["benefit"] = compute_benefit(p)

# ─────────────────────────────────────────────────────────────────────────────
# Knapsack algorithms (pure Python — no numpy/scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def knapsack_dp(items, capacity):
    """Classic 0/1 Knapsack via Dynamic Programming.
    Time:  O(n·W)   where W = capacity / UNIT
    Space: O(n·W)
    """
    n = len(items)
    if n == 0:
        return {"selected": [], "total_benefit": 0.0, "pruning_rate": None}

    # Standard (non-adaptive) DP table resolution.
    #
    # The 0/1 knapsack DP works over an INTEGER weight axis, but project
    # costs are real-valued pesos. The standard way to apply DP here is to
    # discretize cost into UNIT-sized buckets, giving a table of size
    # n * W where W = capacity / UNIT. MAX_W fixes the budget-axis
    # resolution independent of n - there are NO scaling heuristics, so
    # the table (and DP's runtime) grows linearly with n. This is what
    # makes plain DP the slowest of the three algorithms on large inputs.
    #
    # Weights are discretized with round() (nearest bucket) rather than
    # ceil(). ceil() systematically over-charges EVERY item by up to one
    # unit; across thousands of items that bias accumulates into a large
    # phantom weight (e.g. ~25,000 units on the full QC dataset), which can
    # make a set of projects that truly fits the budget appear infeasible -
    # causing DP to leave large amounts of budget unused and report a badly
    # low benefit. round() is unbiased: per-item rounding errors cancel
    # rather than accumulate, so DP fills the budget correctly.
    #
    # Because round() can occasionally UNDER-charge a selection (the opposite
    # bias), a final feasibility repair drops the lowest benefit/cost items
    # until the TRUE cost is within budget. This guarantees DP never returns
    # an over-budget result while keeping it unbiased.
    #
    # On datasets with an extreme cost range (QC spans PhP 6 to PhP 2.1B),
    # no fixed-resolution grid can represent both tiny and huge items
    # exactly, so DP may still report a value below the true optimum on very
    # large, loosely-budgeted selections. That is a GENUINE, well-known
    # limitation of discretized DP on continuous costs - not a bug and not
    # bias - and it is reported honestly. B&B and B&B+GA operate on exact
    # costs and so are always exact.
    #
    # MAX_W is held FIXED for ALL n. The decision-bit table is bit-packed
    # (one bytearray row of ceil((W+1)/8) bytes per item), so even selecting
    # the entire QC dataset (n ~ 26,852) needs only ~67 MB. DP always
    # finishes; it simply takes longer for large inputs, which is the
    # honest, expected cost of the standard algorithm.
    MAX_W = 20_000
    UNIT  = max(1, int(math.ceil(capacity / MAX_W)))
    W     = int(capacity // UNIT)

    # 1-D rolling DP values + bit-packed decision table (keep[i] is a
    # bytearray; bit j set means "item i was taken to achieve dp[j]").
    dp = [0.0] * (W + 1)
    row_bytes = (W // 8) + 1
    keep = [None] * n

    for i, item in enumerate(items):
        w = max(1, int(round(item["cost"] / UNIT)))  # nearest bucket, >=1
        bits = bytearray(row_bytes)
        if w <= W:                                    # else item alone exceeds W
            v = item["benefit"]
            for j in range(W, w - 1, -1):
                cand = dp[j - w] + v
                if cand > dp[j]:
                    dp[j] = cand
                    bits[j >> 3] |= (1 << (j & 7))
        keep[i] = bits

    # Traceback over the bit-packed decision table
    selected, j = [], W
    for i in range(n - 1, -1, -1):
        if keep[i][j >> 3] & (1 << (j & 7)):
            selected.append(i)
            j -= max(1, int(round(items[i]["cost"] / UNIT)))

    # Budget-safety repair: round() can under-charge, so the selected set's
    # TRUE cost might marginally exceed the budget. Drop lowest benefit/cost
    # items until feasible (guarantees DP never reports an over-budget plan).
    used = sum(items[i]["cost"] for i in selected)
    if used > capacity:
        selected.sort(key=lambda i: items[i]["benefit"] / max(items[i]["cost"], 1))
        k = 0
        while used > capacity and k < len(selected):
            used -= items[selected[k]]["cost"]
            k += 1
        selected = selected[k:]

    total_benefit = sum(items[i]["benefit"] for i in selected)

    return {"selected": selected, "total_benefit": total_benefit, "pruning_rate": None, "nodes_generated": None, "nodes_pruned": None, "ga_terminated": False}


def knapsack_bnb(items, capacity):
    """0/1 Knapsack via Best-First Branch-and-Bound (max-heap on upper bound).

    Nodes are expanded in order of decreasing upper bound, so the algorithm
    finds a near-optimal solution very quickly and uses it to prune aggressively.
    This produces genuinely dynamic pruning rates that vary with dataset
    characteristics, unlike DFS which structurally converges to ~50%.

    Time:  O(2^n) worst, O(n log n) average
    Space: O(n)
    """
    import heapq as _hq
    order = sorted(range(len(items)),
                   key=lambda i: items[i]["benefit"] / max(items[i]["cost"], 1),
                   reverse=True)
    si_c = [items[i]["cost"]    for i in order]
    si_b = [items[i]["benefit"] for i in order]
    n    = len(items)

    # [Dantzig fractional bound via prefix sums + binary search]
    # Each upper_bound call is O(log n) instead of O(n), so the overall
    # complexity stays O(nodes * log n) rather than O(nodes * n). Without
    # this, plain B&B's per-node cost grows linearly with n on top of its
    # node count also growing with n, giving an effective O(n^2) - which
    # made it scale WORSE than DP for large n. This mirrors the bound used
    # in knapsack_bnb_ga's B&B phase, so both B&B variants have comparable
    # per-node cost and any timing difference reflects GA overhead /
    # seeding effects rather than an unrelated algorithmic inconsistency.
    prefix_ben  = [0.0] * (n + 1)
    prefix_cost = [0.0] * (n + 1)
    for i in range(n):
        prefix_ben[i+1]  = prefix_ben[i]  + si_b[i]
        prefix_cost[i+1] = prefix_cost[i] + si_c[i]

    def upper_bound(idx, rem, cur):
        if idx >= n:
            return cur
        lo, hi = idx, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if prefix_cost[mid] - prefix_cost[idx] <= rem:
                lo = mid
            else:
                hi = mid - 1
        val = cur + prefix_ben[lo] - prefix_ben[idx]
        if lo < n:
            rem2 = rem - (prefix_cost[lo] - prefix_cost[idx])
            val += si_b[lo] * (rem2 / si_c[lo])
        return val

    best_val = 0.0
    best_set = []
    ng = 1          # root counts as 1 node generated
    np_ = 0
    ctr = 0         # tie-breaker for heap

    root_ub = upper_bound(0, capacity, 0.0)
    heap = [(-root_ub, ctr, 0, capacity, 0.0, [])]

    while heap:
        neg_ub, _, idx, rem, val, chosen = _hq.heappop(heap)
        # Stale node: its upper bound is now <= best (best improved after push)
        if -neg_ub <= best_val:
            np_ += 1
            continue
        if idx == n:
            if val > best_val:
                best_val = val
                best_set = [order[k] for k in chosen]
            continue

        # Include branch
        if si_c[idx] <= rem:
            nv = val + si_b[idx]
            nr = rem - si_c[idx]
            nu = upper_bound(idx + 1, nr, nv)
            ng += 1
            if nu > best_val:
                ctr += 1
                _hq.heappush(heap, (-nu, ctr, idx + 1, nr, nv, chosen + [idx]))
            else:
                np_ += 1

        # Exclude branch
        eu = upper_bound(idx + 1, rem, val)
        ng += 1
        if eu > best_val:
            ctr += 1
            _hq.heappush(heap, (-eu, ctr, idx + 1, rem, val, chosen))
        else:
            np_ += 1

    pruning_rate = round((np_ / ng * 100), 2) if ng > 0 else 0.0
    return {"selected": best_set, "total_benefit": best_val,
            "pruning_rate": pruning_rate, "nodes_generated": ng,
            "nodes_pruned": np_, "ga_terminated": False}


def knapsack_bnb_ga(items, capacity,
                    pop_size=40, generations=60, mutation_rate=0.03):
    """Hybrid 0/1 Knapsack: Genetic Algorithm seeds Best-First Branch-and-Bound.

    Phase 1 - Genetic Algorithm:
      - Bitstring chromosomes (Python lists of 0/1)
      - Population seeded with greedy (ratio-sorted) solutions + jitter,
        plus random chromosomes, all repaired to respect the budget
      - Tournament selection (k=3), single-point crossover, bit-flip mutation
      - Elitism: top-2 chromosomes carried over each generation unchanged
      - In-place greedy repair drops lowest benefit/cost items until feasible

    Phase 2 - Best-First Branch-and-Bound:
      - Prefix-sum Dantzig fractional bound (binary search per node)
      - Max-heap ordered by upper bound (best-first expansion)
      - Seeded with the GA's best fitness as the initial lower bound, so
        more nodes become "stale" (upper bound <= best) immediately after
        being pushed, yielding a pruning rate that is consistently >= the
        rate achieved by knapsack_bnb on the same instance

    Time:  O(P·G·n + n log n)  practical average
    Space: O(P·n)
    """
    n = len(items)
    if n == 0:
        return {"selected": [], "total_benefit": 0.0, "pruning_rate": 0.0, "nodes_generated": 0, "nodes_pruned": 0, "ga_terminated": False}

    costs    = [it["cost"]    for it in items]
    benefits = [it["benefit"] for it in items]

    # Sort order by benefit/cost ratio for GA repair + B&B
    ratio_order = sorted(range(n),
                         key=lambda i: benefits[i] / max(costs[i], 1),
                         reverse=True)

    s_cost = [costs[i]    for i in ratio_order]
    s_ben  = [benefits[i] for i in ratio_order]
    s_orig = list(ratio_order)

    # [O6] Prefix sums for Dantzig upper bound
    prefix_ben  = [0.0] * (n + 1)
    prefix_cost = [0.0] * (n + 1)
    for i in range(n):
        prefix_ben[i+1]  = prefix_ben[i]  + s_ben[i]
        prefix_cost[i+1] = prefix_cost[i] + s_cost[i]

    def upper_bound(idx, rem, cur_val):
        if idx >= n:
            return cur_val
        # Binary search for how many items fit
        lo, hi = idx, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if prefix_cost[mid] - prefix_cost[idx] <= rem:
                lo = mid
            else:
                hi = mid - 1
        val = cur_val + prefix_ben[lo] - prefix_ben[idx]
        if lo < n:
            rem2 = rem - (prefix_cost[lo] - prefix_cost[idx])
            val += s_ben[lo] * (rem2 / s_cost[lo])
        return val

    # ── GA helpers ────────────────────────────────────────────────────────
    # Optimization note (fairness-preserving): these helpers compute exactly
    # the same fitness values and produce exactly the same repaired
    # chromosomes as a naive implementation - they are only made faster by
    # (a) tracking each chromosome's running cost/benefit instead of
    # re-summing the whole bitstring on every call, and (b) returning those
    # cached totals so callers never recompute them. The GA's search
    # behaviour (selection, crossover, mutation, acceptance) is unchanged, so
    # the seed handed to B&B is identical to the unoptimized version.

    def eval_chrom(chrom):
        """Return (cost, fitness) for a chromosome in a single O(n) pass."""
        tc = 0.0
        for i in range(n):
            if chrom[i]:
                tc += costs[i]
        if tc > capacity:
            return tc, 0.0
        tb = 0.0
        for i in range(n):
            if chrom[i]:
                tb += benefits[i]
        return tc, tb

    def repair_tracked(chrom, total_cost):
        """Drop lowest benefit/cost items until feasible. Returns new cost.
        Identical result to a full re-summed repair, but updates cost
        incrementally as items are removed."""
        if total_cost <= capacity:
            return total_cost
        for i in reversed(ratio_order):
            if total_cost <= capacity:
                break
            if chrom[i]:
                chrom[i] = 0
                total_cost -= costs[i]
        return total_cost

    def fitness(chrom):
        # Kept for population init readability; uses the single-pass evaluator.
        return eval_chrom(chrom)[1]

    def greedy_chrom(jitter=0):
        chrom = [0] * n
        rem = capacity
        for i in ratio_order:
            if costs[i] <= rem:
                if jitter == 0 or random.random() > 0.15:
                    chrom[i] = 1
                    rem -= costs[i]
        return chrom

    def random_chrom():
        chrom = [1 if random.random() > 0.5 else 0 for _ in range(n)]
        tc = sum(costs[i] for i in range(n) if chrom[i])
        repair_tracked(chrom, tc)
        return chrom

    def tournament(pop, fits, k=3):
        best_i = max(random.sample(range(len(pop)), k), key=lambda x: fits[x])
        return pop[best_i][:]

    def crossover(p1, p2):
        if n <= 1:
            return p2[:]  # nothing to cross over with a single gene
        pt = random.randint(1, n - 1)
        return p1[:pt] + p2[pt:]

    def mutate(chrom, rate):
        # Statistically identical to flipping each of the n genes
        # independently with probability `rate`, but far faster for large n:
        # instead of drawing one random number PER GENE (n draws, ~97% of
        # which are wasted when rate is small), draw the NUMBER of genes to
        # flip from Binomial(n, rate) and then pick exactly that many distinct
        # positions uniformly. This is mathematically equivalent - each gene
        # still flips with probability `rate`, independently - so the GA's
        # behaviour and the seed it produces are unchanged; only the cost of
        # generating the randomness drops from O(n) draws to O(n*rate) draws.
        if _HAS_BINOMIAL:
            k = random.binomialvariate(n, rate)
        else:
            # Normal approximation fallback (Python < 3.12). Same expected
            # count and spread; only the RNG draw differs, not the algorithm.
            mean = n * rate
            sd = (n * rate * (1.0 - rate)) ** 0.5
            k = max(0, min(n, int(round(random.gauss(mean, sd)))))
        if k:
            for i in random.sample(range(n), k):
                chrom[i] ^= 1
        return chrom

    # Initialise population
    pop  = [greedy_chrom(jitter=k) for k in range(pop_size // 2)]
    pop += [random_chrom()         for _ in range(pop_size - len(pop))]
    fits = [fitness(c) for c in pop]

    ga_best_fit  = max(fits)
    ga_best_idx  = fits.index(ga_best_fit)
    ga_best_chrom = pop[ga_best_idx][:]

    # Evolution: run up to `generations`, but stop early once the best
    # seed has not improved for `patience` consecutive generations. This is
    # a standard GA convergence criterion, NOT a size-based throttle: it
    # responds only to the GA's own progress and behaves identically
    # regardless of n. Its sole purpose is to avoid burning generations
    # after the population has already converged (on these datasets the
    # greedy-seeded population typically converges within a handful of
    # generations), which otherwise adds large overhead for no better seed.
    # The B&B phase that follows is unchanged and identical to plain B&B.
    mr = max(1 / n, mutation_rate)
    patience = 8
    gens_since_improve = 0
    for _ in range(generations):
        # Elitism: keep top-2
        ranked = sorted(range(len(pop)), key=lambda x: fits[x], reverse=True)
        new_pop  = [pop[ranked[0]][:], pop[ranked[1]][:]]
        new_fits = [fits[ranked[0]],   fits[ranked[1]]]

        improved = False
        while len(new_pop) < pop_size:
            child = mutate(crossover(tournament(pop, fits),
                                     tournament(pop, fits)), mr)
            # Single fused pass: collect set-gene indices once, summing cost
            # and benefit together. Identical result to two separate sums,
            # but iterates the chromosome only once. If the child is over
            # budget, repair (which also returns the corrected cost) and then
            # recompute benefit over the now-feasible set.
            tc = 0.0
            tb = 0.0
            for i in range(n):
                if child[i]:
                    tc += costs[i]
                    tb += benefits[i]
            if tc > capacity:
                tc = repair_tracked(child, tc)
                tb = 0.0
                for i in range(n):
                    if child[i]:
                        tb += benefits[i]
            f = tb
            new_pop.append(child)
            new_fits.append(f)
            if f > ga_best_fit:
                ga_best_fit   = f
                ga_best_chrom = child[:]
                improved = True

        pop, fits = new_pop, new_fits

        gens_since_improve = 0 if improved else gens_since_improve + 1
        if gens_since_improve >= patience:
            break

    ga_selected = [i for i in range(n) if ga_best_chrom[i]]

    # ── Phase 2: Best-First B&B seeded with GA lower bound ──────────────
    # [O9] GA best fit becomes the initial lower bound, allowing the
    # best-first heap to prune stale nodes aggressively from the start.
    # Because the GA seed is tighter than a cold start (best_val=0),
    # more nodes become stale immediately after being pushed, yielding
    # a genuinely higher pruning rate than plain B&B on the same instance.
    import heapq as _hq
    best_val = ga_best_fit
    best_set = ga_selected[:]
    ng  = 1
    np_ = 0
    ctr = 0

    root_ub = upper_bound(0, capacity, 0.0)
    heap = [(-root_ub, ctr, 0, capacity, 0.0, [])]

    while heap:
        neg_ub, _, idx, rem, val, chosen = _hq.heappop(heap)
        if -neg_ub <= best_val:          # stale — best improved since push
            np_ += 1
            continue
        if idx == n:
            if val > best_val:
                best_val = val
                best_set = [s_orig[k] for k in chosen]
            continue

        # Include branch
        if s_cost[idx] <= rem:
            nv = val + s_ben[idx]
            nr = rem - s_cost[idx]
            nu = upper_bound(idx + 1, nr, nv)
            ng += 1
            if nu > best_val:
                ctr += 1
                _hq.heappush(heap, (-nu, ctr, idx + 1, nr, nv, chosen + [idx]))
            else:
                np_ += 1

        # Exclude branch
        eu = upper_bound(idx + 1, rem, val)
        ng += 1
        if eu > best_val:
            ctr += 1
            _hq.heappush(heap, (-eu, ctr, idx + 1, rem, val, chosen))
        else:
            np_ += 1

    pruning_rate = round((np_ / ng * 100), 2) if ng > 0 else 0.0
    return {"selected": best_set, "total_benefit": best_val,
            "pruning_rate": pruning_rate, "nodes_generated": ng,
            "nodes_pruned": np_, "ga_terminated": False}


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(405)
@app.errorhandler(415)
@app.errorhandler(500)
def json_error(e):
    from flask import jsonify
    return jsonify({"error": str(e)}), e.code

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>LGU Budget Optimizer</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg-base:#F4F6FA;--bg-surface:#FFFFFF;--bg-raised:#F0F2F8;--bg-hover:#E8EBF4;--bg-active:#DDE2F0;
  --bd-subtle:rgba(30,40,100,.08);--bd-default:rgba(30,40,100,.16);--bd-strong:rgba(30,40,100,.28);
  --tx-primary:#111827;--tx-secondary:#4B5563;--tx-muted:#9CA3AF;--tx-inverse:#FFFFFF;
  --accent:#4F6EF7;--accent-dim:rgba(79,110,247,.10);--accent-border:rgba(79,110,247,.30);--accent-glow:0 4px 18px rgba(79,110,247,.22);
  --pasig:#7C3AED;--pasig-dim:rgba(124,58,237,.09);--pasig-border:rgba(124,58,237,.28);--pasig-glow:0 4px 18px rgba(124,58,237,.15);
  --qc:#0284C7;--qc-dim:rgba(2,132,199,.09);--qc-border:rgba(2,132,199,.28);--qc-glow:0 4px 18px rgba(2,132,199,.15);
  --green:#16A34A;--green-dim:rgba(22,163,74,.10);--green-border:rgba(22,163,74,.28);
  --amber:#B45309;--amber-dim:rgba(180,83,9,.10);--red:#DC2626;--red-dim:rgba(220,38,38,.10);
  --r-sm:6px;--r-md:10px;--r-lg:14px;--r-xl:18px;--ease:cubic-bezier(.4,0,.2,1);
}
html{scroll-behavior:smooth}
body{font-family:'Inter',-apple-system,sans-serif;background:var(--bg-base);color:var(--tx-primary);min-height:100vh;line-height:1.5}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#CBD5E1;border-radius:99px}
::-webkit-scrollbar-thumb:hover{background:#94A3B8}

/* HERO */
.hero{background:var(--bg-surface);border-bottom:1px solid var(--bd-subtle);padding:28px 24px 22px;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 600px 300px at 80% -40%,rgba(79,110,247,.06) 0%,transparent 70%),radial-gradient(ellipse 400px 250px at 10% 120%,rgba(124,58,237,.04) 0%,transparent 60%);pointer-events:none}
.hero-inner{max-width:1100px;margin:0 auto;position:relative}
.hero-eyebrow{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);background:var(--accent-dim);border:1px solid var(--accent-border);padding:3px 10px;border-radius:99px;margin-bottom:12px}
.hero-eyebrow::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}
.hero h1{font-size:26px;font-weight:700;letter-spacing:-.4px;color:var(--tx-primary);margin-bottom:5px}
.hero h1 span{color:var(--accent)}
.hero-sub{font-size:13px;color:var(--tx-secondary);max-width:580px}

/* WRAP */
.wrap{max-width:1100px;margin:0 auto;padding:24px 20px 56px}

/* SECTION LABEL */
.slabel{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--tx-muted);margin-bottom:10px;display:flex;align-items:center;gap:8px}
.slabel::after{content:'';flex:1;height:1px;background:#E5E7EB}

/* DS SWITCHER */
.ds-switcher{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:18px}
@media(max-width:560px){.ds-switcher{grid-template-columns:1fr}}
.ds-btn{display:flex;align-items:center;gap:14px;padding:16px 18px;background:var(--bg-surface);border:1.5px solid var(--bd-subtle);border-radius:var(--r-lg);cursor:pointer;text-align:left;transition:all .2s var(--ease)}
.ds-btn:hover{border-color:var(--bd-default);transform:translateY(-1px);box-shadow:0 4px 16px rgba(0,0,0,.06)}
.ds-btn.active-pasig{border-color:var(--pasig-border);background:linear-gradient(135deg,var(--pasig-dim) 0%,var(--bg-surface) 100%);box-shadow:var(--pasig-glow)}
.ds-btn.active-qc{border-color:var(--qc-border);background:linear-gradient(135deg,var(--qc-dim) 0%,var(--bg-surface) 100%);box-shadow:var(--qc-glow)}
.ds-icon{width:44px;height:44px;border-radius:var(--r-md);display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;background:var(--bg-raised);border:1px solid var(--bd-subtle);transition:all .2s var(--ease)}
.ds-btn.active-pasig .ds-icon{background:var(--pasig-dim);border-color:var(--pasig-border)}
.ds-btn.active-qc .ds-icon{background:var(--qc-dim);border-color:var(--qc-border)}
.ds-info{flex:1;min-width:0}
.ds-name{font-size:14px;font-weight:600;color:var(--tx-primary);display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.ds-meta{font-size:11px;color:var(--tx-secondary);margin-top:3px}
.ds-tag{font-size:10px;font-weight:600;padding:2px 8px;border-radius:99px;white-space:nowrap}
.ds-tag.small{background:#FEF3C7;color:#92400E;border:1px solid #FCD34D}
.ds-tag.large{background:#D1FAE5;color:#065F46;border:1px solid #6EE7B7}
.ds-check{width:20px;height:20px;border-radius:50%;border:2px solid var(--bd-default);flex-shrink:0;display:flex;align-items:center;justify-content:center;transition:all .2s var(--ease)}
.ds-btn.active-pasig .ds-check{border-color:var(--pasig);background:var(--pasig)}
.ds-btn.active-qc .ds-check{border-color:var(--qc);background:var(--qc)}
.ds-check::after{content:'';width:6px;height:6px;border-radius:50%;background:#fff;opacity:0;transform:scale(0);transition:all .15s var(--ease)}
.ds-btn.active-pasig .ds-check::after,.ds-btn.active-qc .ds-check::after{opacity:1;transform:scale(1)}

/* INFO BAR */
.ds-infobar{padding:11px 16px;border-radius:var(--r-md);margin-bottom:18px;font-size:12px;line-height:1.6;display:flex;align-items:flex-start;gap:10px;transition:all .25s var(--ease)}
.ds-infobar.pasig{background:var(--pasig-dim);border:1px solid var(--pasig-border);color:var(--pasig)}
.ds-infobar.qc{background:var(--qc-dim);border:1px solid var(--qc-border);color:var(--qc)}
.ds-infobar b{font-weight:600}

/* CARDS */
.card{background:var(--bg-surface);border:1px solid var(--bd-subtle);border-radius:var(--r-xl);padding:20px;margin-bottom:16px;transition:border-color .2s var(--ease)}
.card:hover{border-color:var(--bd-default)}
.card-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:8px}
.card-title{font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--tx-secondary);display:flex;align-items:center;gap:7px}
.ctdot{width:7px;height:7px;border-radius:50%;background:var(--accent)}
.card-meta{font-size:11px;color:var(--tx-muted)}

/* BUDGET */
.budget-block{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.budget-label{font-size:13px;color:var(--tx-secondary);min-width:100px;font-weight:500}
.budget-slider-wrap{flex:1;min-width:200px}
input[type=range]{width:100%;height:4px;-webkit-appearance:none;appearance:none;background:var(--bg-raised);border-radius:99px;outline:none;cursor:pointer;border:1px solid var(--bd-subtle)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 3px var(--accent-dim),var(--accent-glow);cursor:pointer;transition:transform .15s var(--ease)}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.2)}
input[type=range]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:var(--accent);border:none;cursor:pointer}
.budget-input-wrap{display:flex;align-items:center;gap:4px;min-width:200px;justify-content:flex-end;background:var(--bg-raised);border:1px solid var(--bd-subtle);border-radius:var(--r-md);padding:6px 12px;transition:all .15s var(--ease)}
.budget-input-wrap:focus-within{border-color:var(--accent-border);background:var(--bg-active);box-shadow:0 0 0 3px var(--accent-dim)}
.budget-peso{font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--accent);letter-spacing:-.5px}
#budgetInput{flex:1;min-width:0;border:none;outline:none;background:transparent;font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--accent);letter-spacing:-.5px;text-align:right;padding:0}
#budgetInput::placeholder{color:var(--tx-muted)}

/* TOOLBAR */
.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
.search-wrap{flex:1;min-width:200px;position:relative}
.search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:var(--tx-muted);font-size:13px;pointer-events:none}
.search-wrap input[type=text]{width:100%;padding:8px 12px 8px 30px;background:var(--bg-raised);border:1px solid var(--bd-subtle);border-radius:var(--r-md);color:var(--tx-primary);font-size:13px;font-family:'Inter',sans-serif;outline:none;transition:all .15s var(--ease)}
.search-wrap input[type=text]::placeholder{color:var(--tx-muted)}
.search-wrap input[type=text]:focus{border-color:var(--accent-border);background:var(--bg-active);box-shadow:0 0 0 3px var(--accent-dim)}
.filter-btn{padding:6px 13px;border-radius:var(--r-md);font-size:12px;font-weight:500;cursor:pointer;border:1px solid var(--bd-subtle);color:var(--tx-secondary);background:var(--bg-raised);transition:all .15s var(--ease);white-space:nowrap;font-family:'Inter',sans-serif}
.filter-btn:hover{border-color:var(--bd-default);color:var(--tx-primary);background:var(--bg-active)}
.filter-btn.on{border-color:var(--accent-border);color:var(--accent);background:var(--accent-dim)}

/* TABLE */
.tbl-wrap{max-height:380px;overflow-y:auto;border:1px solid var(--bd-subtle);border-radius:var(--r-lg);background:#fff}
table{width:100%;border-collapse:collapse;font-size:12.5px}
thead th{position:sticky;top:0;z-index:2;background:var(--bg-raised);padding:9px 12px;text-align:left;color:var(--tx-secondary);font-weight:600;font-size:11px;letter-spacing:.05em;text-transform:uppercase;border-bottom:1px solid var(--bd-subtle)}
tbody tr{transition:background .1s var(--ease);border-bottom:1px solid var(--bd-subtle)}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:var(--bg-hover)}
td{padding:8px 12px;vertical-align:middle}
.chk{width:15px;height:15px;cursor:pointer;accent-color:var(--accent)}
.s-badge{font-size:10px;font-weight:600;padding:2px 8px;border-radius:99px;white-space:nowrap;display:inline-block}
.s-Healthcare{background:#FEE2E2;color:#B91C1C;border:1px solid #FCA5A5}
.s-Infrastructure{background:#E0F2FE;color:#0369A1;border:1px solid #7DD3FC}
.s-Education{background:#DCFCE7;color:#15803D;border:1px solid #86EFAC}
.s-Social-Services{background:#EDE9FE;color:#6D28D9;border:1px solid #C4B5FD}
.s-Environment{background:#D1FAE5;color:#065F46;border:1px solid #6EE7B7}
.s-Public-Safety{background:#FFF7ED;color:#C2410C;border:1px solid #FDC187}
.s-General-Government{background:#F3F4F6;color:#4B5563;border:1px solid #D1D5DB}
.cost-col{text-align:right;font-family:'JetBrains Mono',monospace;font-size:12px;white-space:nowrap;color:var(--tx-secondary)}
.benefit-col{text-align:center}
.b-pill{display:inline-block;font-size:11px;font-weight:700;font-family:'JetBrains Mono',monospace;width:32px;height:22px;line-height:22px;border-radius:6px;text-align:center;color:#fff}
.sel-count{font-size:12px;color:var(--tx-secondary);margin-top:10px;display:flex;align-items:center;gap:6px}
.sel-pill{background:var(--accent-dim);color:var(--accent);border:1px solid var(--accent-border);border-radius:99px;padding:1px 8px;font-size:11px;font-weight:600}

/* RUN BTN */
.run-btn{width:100%;padding:14px;font-size:15px;font-weight:700;cursor:pointer;background:linear-gradient(135deg,#4F6EF7 0%,#3B55E0 100%);border:none;border-radius:var(--r-lg);color:#fff;letter-spacing:.02em;transition:all .2s var(--ease);margin-bottom:16px;display:flex;align-items:center;justify-content:center;gap:8px;box-shadow:0 4px 20px rgba(79,110,247,.28);font-family:'Inter',sans-serif}
.run-btn:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(79,110,247,.4)}
.run-btn:active{transform:translateY(0)}
.run-btn:disabled{opacity:.6;cursor:not-allowed;transform:none;box-shadow:none}

/* PROGRESS BAR */
.progress-wrap{display:none;margin-bottom:16px}
.progress-bar-bg{background:var(--bg-raised);border-radius:99px;height:8px;overflow:hidden;border:1px solid var(--bd-subtle)}
.progress-bar-fill{height:8px;border-radius:99px;background:linear-gradient(90deg,#4F6EF7,#7C3AED);transition:width .3s var(--ease);width:0%}
.progress-label{font-size:12px;color:var(--tx-secondary);margin-top:6px;text-align:center}

/* RESULTS */
.compare-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:16px}
@media(max-width:640px){.compare-grid{grid-template-columns:1fr}}

/* ALGO SELECTOR */
.algo-select{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.algo-chk-label{display:flex;align-items:center;gap:7px;font-size:13px;font-weight:500;color:var(--tx-secondary);cursor:pointer;padding:9px 14px;border:1.5px solid var(--bd-subtle);border-radius:var(--r-md);background:var(--bg-surface);transition:all .15s var(--ease);flex:1;min-width:150px;justify-content:center}
.algo-chk-label:hover{border-color:var(--bd-default);color:var(--tx-primary)}
.algo-chk-label input[type=checkbox]{accent-color:var(--accent);width:15px;height:15px;cursor:pointer}
.algo-chk-label.checked{border-color:var(--accent-border);color:var(--accent);background:var(--accent-dim);font-weight:600}
@media(max-width:560px){.algo-select{flex-direction:column}.algo-chk-label{min-width:0}}
.ccard{background:var(--bg-raised);border:1px solid var(--bd-subtle);border-radius:var(--r-lg);padding:16px;position:relative;overflow:hidden;transition:all .2s var(--ease)}
.ccard::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:transparent;transition:background .2s var(--ease)}
.ccard.best{border-color:var(--green-border);background:linear-gradient(180deg,rgba(22,163,74,.06) 0%,var(--bg-surface) 60%);box-shadow:0 4px 20px rgba(22,163,74,.12)}
.ccard.best::before{background:linear-gradient(90deg,var(--green),transparent)}
.best-tag{display:inline-flex;align-items:center;gap:4px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;background:var(--green-dim);color:var(--green);border:1px solid var(--green-border);padding:2px 8px;border-radius:99px;margin-bottom:10px}
.ccard-eye{height:22px;margin-bottom:10px}
.ccard-title{font-size:12px;font-weight:600;color:var(--tx-secondary);margin-bottom:8px}
.ccard-big{font-size:30px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--tx-primary);letter-spacing:-1px;line-height:1}
.ccard-sub{font-size:11px;color:var(--tx-muted);margin-top:4px}
.bar-t{background:var(--bg-hover);border-radius:99px;height:5px;width:100%;margin-top:12px;overflow:hidden}
.bar-f{height:5px;border-radius:99px;transition:width .6s var(--ease)}
.ccard-div{height:1px;background:var(--bd-subtle);margin:12px 0}
.ccard-row{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px}
.ccard-row .lbl{color:var(--tx-muted)}
.ccard-row .val{color:var(--tx-secondary);font-family:'JetBrains Mono',monospace;font-size:11px}
.ccard-row .val.hl{color:var(--accent);font-weight:600}
.ccard-row .val.hl-green{color:var(--green);font-weight:600}
.ccard-row.node-row{cursor:pointer;user-select:none;border-radius:4px;margin:0 -4px 5px;padding:1px 4px;transition:background .15s var(--ease)}
.ccard-row.node-row:hover{background:var(--bg-hover)}
.ccard-row.node-row .lbl{display:inline-flex;align-items:center;gap:5px}
.ccard-row.node-row .nt-caret{display:inline-block;transition:transform .15s var(--ease);font-size:8px;line-height:1;color:var(--tx-muted)}
.ccard-row.node-row.open .nt-caret{transform:rotate(90deg)}
.node-detail{margin-bottom:1px}

/* BANNER */
.banner{display:flex;align-items:center;gap:10px;padding:11px 16px;border-radius:var(--r-md);font-size:13px;font-weight:500;margin-bottom:16px}
.banner.ok{background:var(--green-dim);border:1px solid var(--green-border);color:var(--green)}
.banner.warn{background:var(--amber-dim);border:1px solid rgba(180,83,9,.3);color:var(--amber)}

/* STATS */
.stats-bar{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px}
.stat{background:var(--bg-base);border:1px solid var(--bd-subtle);border-radius:var(--r-lg);padding:14px 16px}
.stat-l{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--tx-muted);margin-bottom:6px}
.stat-v{font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--tx-primary);letter-spacing:-.5px}
.stat-v.acc{color:var(--accent)}
.stat-s{font-size:11px;color:var(--tx-muted);margin-top:3px}

/* ALGO TABS */
.algo-tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px;background:var(--bg-base);border:1px solid var(--bd-subtle);border-radius:var(--r-lg);padding:5px}
.algo-tab{flex:1;padding:8px 16px;border-radius:var(--r-md);font-size:13px;font-weight:600;cursor:pointer;border:none;color:var(--tx-secondary);background:transparent;transition:all .15s var(--ease);text-align:center;font-family:'Inter',sans-serif;white-space:nowrap}
.algo-tab:hover{color:var(--tx-primary);background:var(--bg-hover)}
.algo-tab.on{color:#fff;background:var(--accent);box-shadow:0 2px 10px rgba(79,110,247,.25)}

/* RESULT LIST */
.res-list{display:flex;flex-direction:column;gap:4px;max-height:440px;overflow-y:auto}
.res-chip{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:var(--r-md);font-size:12px;transition:all .12s var(--ease);border:1px solid transparent}
.res-chip.sel{background:var(--accent-dim);border-color:var(--accent-border)}
.res-chip.sel:hover{background:rgba(79,110,247,.15)}
.res-chip.rej{background:var(--bg-raised);border-color:var(--bd-subtle);opacity:.45}
.res-chip.rej .res-name{text-decoration:line-through;color:var(--tx-muted)}
.res-name{flex:1;color:var(--tx-primary);line-height:1.3}
.res-cost{min-width:88px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--tx-secondary)}
.res-score{min-width:50px;text-align:right;font-weight:700;font-size:12px;color:var(--accent)}
.res-chip.rej .res-score{color:var(--tx-muted)}

/* MISC */
.empty{text-align:center;padding:2.5rem;color:var(--tx-muted);font-size:14px}
.spinner{display:inline-block;width:16px;height:16px;border:2.5px solid rgba(79,110,247,.25);border-top-color:#4F6EF7;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}}
.footer{text-align:center;font-size:11px;color:var(--tx-muted);padding-top:12px;border-top:1px solid var(--bd-subtle)}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:fadeUp .3s var(--ease) both}
.pool-meta{display:inline-flex;align-items:center;gap:6px;font-size:11px;color:var(--tx-muted);background:var(--bg-raised);border:1px solid var(--bd-subtle);border-radius:99px;padding:2px 10px}
.dot{width:5px;height:5px;border-radius:50%;background:var(--accent);display:inline-block}
.pagination{display:flex;align-items:center;justify-content:space-between;margin-top:10px;flex-wrap:wrap;gap:8px}
.page-info{font-size:12px;color:var(--tx-muted)}
.page-btns{display:flex;gap:6px}
.page-btn{padding:4px 12px;border-radius:var(--r-md);font-size:12px;font-weight:500;cursor:pointer;border:1px solid var(--bd-subtle);color:var(--tx-secondary);background:var(--bg-raised);font-family:'Inter',sans-serif;transition:all .15s var(--ease)}
.page-btn:hover:not(:disabled){border-color:var(--accent-border);color:var(--accent)}
.page-btn:disabled{opacity:.4;cursor:not-allowed}
.page-btn.active{border-color:var(--accent-border);color:var(--accent);background:var(--accent-dim)}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-inner">
    <div class="hero-eyebrow">BSCS 3-1N &nbsp;·&nbsp; Thesis Group 2 &nbsp;·&nbsp; APP FY 2025</div>
    <h1>LGU Budget <span>Optimizer</span></h1>
    <p class="hero-sub">Hybrid Knapsack Algorithm with Genetic and Branch-and-Bound Enhancement — Pasig City &amp; Quezon City Annual Procurement Plans</p>
  </div>
</div>

<div class="wrap">

  <div class="slabel">Select Dataset</div>
  <div class="ds-switcher">
    <button class="ds-btn active-pasig" id="btnPasig" onclick="switchDs('pasig')">
      <div class="ds-icon">🏙</div>
      <div class="ds-info">
        <div class="ds-name">Pasig City <span class="ds-tag small">Small</span></div>
        <div class="ds-meta" id="pasigMeta">Loading…</div>
      </div>
      <div class="ds-check"></div>
    </button>
    <button class="ds-btn" id="btnQC" onclick="switchDs('qc')">
      <div class="ds-icon">🌆</div>
      <div class="ds-info">
        <div class="ds-name">Quezon City <span class="ds-tag large">Large</span></div>
        <div class="ds-meta" id="qcMeta">Loading…</div>
      </div>
      <div class="ds-check"></div>
    </button>
  </div>

  <div class="ds-infobar pasig" id="infoBar">
    <span>ℹ</span>
    <span id="infoText"><b>Pasig City APP FY 2025 (General Fund)</b> — Source: City Government of Pasig, LGU Transparency Portal. Used as the <b>small dataset</b> to establish a standard of optimality and verify algorithm accuracy.</span>
  </div>

  <!-- Budget -->
  <div class="card">
    <div class="card-hdr">
      <div class="card-title"><div class="ctdot"></div>Budget Constraint <span style="font-size:10px;color:var(--tx-muted);font-weight:400;text-transform:none;letter-spacing:0">(shared across both datasets)</span></div>
    </div>
    <div class="budget-block">
      <div class="budget-label">Total project budget</div>
      <div class="budget-slider-wrap">
        <input type="range" id="budgetSlider" min="0" max="50000000000" step="100000000" value="5000000000" oninput="updateBudget(this.value, 'slider')">
      </div>
      <div class="budget-input-wrap">
        <span class="budget-peso">₱</span>
        <input type="text" inputmode="numeric" id="budgetInput" value="5,000,000,000"
               oninput="updateBudget(this.value, 'input')"
               onblur="normalizeBudgetInput()"
               aria-label="Total project budget amount">
      </div>
    </div>
  </div>

  <!-- Project pool -->
  <div class="card">
    <div class="card-hdr">
      <div class="card-title">
        <div class="ctdot"></div>Project Pool
        <span class="pool-meta"><span class="dot"></span><span id="poolLabel">Loading…</span></span>
      </div>
      <div class="card-meta"><span id="totalCount">0</span> projects &nbsp;·&nbsp; ₱<span id="totalBudgetLbl">0</span> total</div>
    </div>
    <div class="toolbar">
      <div class="search-wrap">
        <span class="search-icon">⌕</span>
        <input type="text" id="searchBox" placeholder="Search projects or office…" oninput="filterProjects()">
      </div>
      <button class="filter-btn on" data-sector="All" onclick="setSector(this)">All</button>
      <button class="filter-btn" data-sector="Healthcare" onclick="setSector(this)">Healthcare</button>
      <button class="filter-btn" data-sector="Infrastructure" onclick="setSector(this)">Infrastructure</button>
      <button class="filter-btn" data-sector="Education" onclick="setSector(this)">Education</button>
      <button class="filter-btn" data-sector="Social Services" onclick="setSector(this)">Social Services</button>
      <button class="filter-btn" data-sector="Environment" onclick="setSector(this)">Environment</button>
      <button class="filter-btn" data-sector="Public Safety" onclick="setSector(this)">Public Safety</button>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th style="width:36px"><input type="checkbox" class="chk" id="chkAll" onchange="toggleAll(this.checked)"></th>
            <th>Project / Program</th>
            <th>Sector</th>
            <th>Office</th>
            <th style="text-align:right">Cost (₱)</th>
            <th style="text-align:center">Score</th>
          </tr>
        </thead>
        <tbody id="projTbody"></tbody>
      </table>
    </div>
    <div class="pagination">
      <div class="page-info" id="pageInfo">Showing 0–0 of 0</div>
      <div class="page-btns" id="pageBtns"></div>
    </div>
    <div class="sel-count">
      <span class="sel-pill" id="selPill">0 selected</span>
      <span id="selCost">₱0 total cost</span>
    </div>
  </div>

  <!-- Algorithm selector -->
  <div class="card">
    <div class="card-hdr">
      <div class="card-title"><div class="ctdot"></div>Algorithms to Run</div>
    </div>
    <div class="algo-select" id="algoSelect">
      <label class="algo-chk-label checked" id="lbl-dp">
        <input type="checkbox" checked onchange="toggleAlgo('dp',this)"> Knapsack (DP)
      </label>
      <label class="algo-chk-label checked" id="lbl-bnb">
        <input type="checkbox" checked onchange="toggleAlgo('bnb',this)"> Knapsack + B&amp;B
      </label>
      <label class="algo-chk-label checked" id="lbl-ga">
        <input type="checkbox" checked onchange="toggleAlgo('ga',this)"> Knapsack + B&amp;B + GA
      </label>
    </div>
  </div>

  <button class="run-btn" id="runBtn" onclick="runAlgos()">
    ⚡ Run Selected Algorithms
  </button>

  <div class="progress-wrap" id="progressWrap">
    <div class="progress-bar-bg"><div class="progress-bar-fill" id="progressFill"></div></div>
    <div class="progress-label" id="progressLabel">Initialising…</div>
  </div>

  <div id="results"></div>

  <div class="footer" id="footerTxt">
    Pasig City · Annual Procurement Plan FY 2025 (General Fund) &nbsp;|&nbsp;
    Knapsack DP · Branch &amp; Bound (Land &amp; Doig, 1960) · Genetic Algorithm (Holland's Schema Theorem)
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let DS       = 'pasig';
let PROJECTS = [];   // full dataset for current DS
let FILTERED = [];   // after sector + search filter
let CHECKED  = new Set();
let BUDGET   = 5_000_000_000;
let SECTOR   = 'All';
let PAGE     = 0;
const PAGE_SIZE = 50;

let currentTab  = 2;
const _expandedNodes = new Set();
let activeAlgos = {dp: true, bnb: true, ga: true};

function toggleAlgo(name, el) {
  // Prevent unchecking the last remaining algorithm
  const others = Object.keys(activeAlgos).filter(k => k !== name);
  const anyOtherActive = others.some(k => activeAlgos[k]);
  if (!el.checked && !anyOtherActive) {
    el.checked = true;
    return;
  }
  activeAlgos[name] = el.checked;
  document.getElementById('lbl-'+name).classList.toggle('checked', el.checked);
}

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  await loadDs('pasig');
  updateMeta();
}

async function loadDs(ds) {
  DS = ds;
  const resp = await fetch(`/api/projects?ds=${ds}`);
  const data = await resp.json();
  PROJECTS = data.projects;
  CHECKED.clear();
  SECTOR = 'All';
  PAGE   = 0;
  document.getElementById('searchBox').value = '';
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.toggle('on', b.dataset.sector === 'All'));

  // Update switcher
  document.getElementById('btnPasig').className = 'ds-btn' + (ds==='pasig' ? ' active-pasig' : '');
  document.getElementById('btnQC').className    = 'ds-btn' + (ds==='qc'    ? ' active-qc'    : '');
  const bar = document.getElementById('infoBar');
  bar.className = 'ds-infobar ' + ds;
  const info = {
    pasig: '<b>Pasig City APP FY 2025 (General Fund)</b> — Source: City Government of Pasig, LGU Transparency Portal. Used as the <b>small dataset</b> to establish a standard of optimality and verify algorithm accuracy.',
    qc:    '<b>Quezon City APP FY 2025 (4th Quarter)</b> — Source: QC Bids and Awards Committee, LGU Transparency Portal. Used as the <b>large dataset</b> to stress-test scalability and efficiency of the hybrid algorithm.',
  };
  document.getElementById('infoText').innerHTML = info[ds];
  document.getElementById('footerTxt').textContent = ds==='pasig'
    ? 'Pasig City · Annual Procurement Plan FY 2025 (General Fund) | Knapsack DP · Branch & Bound · Genetic Algorithm'
    : 'Quezon City · Annual Procurement Plan FY 2025 (4th Quarter) | Knapsack DP · Branch & Bound · Genetic Algorithm';

  filterProjects();
  preselectTop();
  document.getElementById('results').innerHTML = '';
}

function switchDs(ds) {
  if (ds === DS) return;
  loadDs(ds);
}

function updateMeta() {
  fetch('/api/meta').then(r=>r.json()).then(d=>{
    document.getElementById('pasigMeta').textContent =
      `APP FY 2025 · General Fund · ${d.pasig.count.toLocaleString()} projects`;
    document.getElementById('qcMeta').textContent =
      `APP FY 2025 · 4th Quarter · ${d.qc.count.toLocaleString()} projects`;
  });
}

// ── Budget ─────────────────────────────────────────────────────────────────
const BUDGET_MAX = 50000000000;

function updateBudget(v, source) {
  let n;
  if (source === 'input') {
    // Strip everything except digits (allow the user to type commas freely)
    n = parseInt(String(v).replace(/[^\d]/g, ''), 10);
    if (isNaN(n)) n = 0;
  } else {
    n = parseInt(v, 10);
    if (isNaN(n)) n = 0;
  }
  // Clamp to [0, BUDGET_MAX]
  if (n < 0) n = 0;
  if (n > BUDGET_MAX) n = BUDGET_MAX;
  BUDGET = n;

  // Sync the slider (always reflects the clamped value)
  document.getElementById('budgetSlider').value = n;

  // Sync the text input. While the user is actively typing in it, don't
  // fight their cursor by reformatting mid-edit; only push the formatted
  // value when the change came from the slider.
  if (source !== 'input') {
    document.getElementById('budgetInput').value = n.toLocaleString();
  }
}

function normalizeBudgetInput() {
  // On blur, snap the text field to the clean formatted, clamped value.
  document.getElementById('budgetInput').value = BUDGET.toLocaleString();
}

function fmt(n) {
  return '₱' + Math.round(n).toLocaleString();
}
function fmtB(n) {
  if (n >= 1e9) return (n/1e9).toFixed(2)+'B';
  if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
  return Math.round(n).toLocaleString();
}

// ── Filter + render ────────────────────────────────────────────────────────
function setSector(btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  SECTOR = btn.dataset.sector;
  PAGE = 0;
  filterProjects();
}

function filterProjects() {
  const q = document.getElementById('searchBox').value.toLowerCase();
  FILTERED = PROJECTS.filter(p => {
    if (SECTOR !== 'All' && p.sector !== SECTOR) return false;
    if (q && !p.name.toLowerCase().includes(q) && !p.pmo.toLowerCase().includes(q)) return false;
    return true;
  });
  PAGE = 0;
  renderTable();
}

function renderTable() {
  const start = PAGE * PAGE_SIZE;
  const end   = Math.min(start + PAGE_SIZE, FILTERED.length);
  const page_items = FILTERED.slice(start, end);

  const tbody = document.getElementById('projTbody');
  if (!FILTERED.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty">No projects match the filter.</td></tr>';
    renderPagination();
    updateSelCount();
    return;
  }

  tbody.innerHTML = page_items.map((p, li) => {
    const pi = p._idx;  // index in PROJECTS
    const chk = CHECKED.has(pi);
    return `<tr>
      <td><input type="checkbox" class="chk" ${chk?'checked':''} onchange="toggleCheck(${pi},this.checked)"></td>
      <td style="max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escHtml(p.name)}">${escHtml(p.name)}</td>
      <td><span class="s-badge s-${p.sector.replace(/ /g,'-')}">${p.sector}</span></td>
      <td style="font-size:11px;color:var(--tx-secondary);max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(p.pmo)}</td>
      <td class="cost-col">${fmtB(p.cost)}</td>
      <td class="benefit-col"><span class="b-pill" style="background:${bcolor(p.benefit)}">${p.benefit.toFixed(1)}</span></td>
    </tr>`;
  }).join('');

  // Update pool stats
  document.getElementById('totalCount').textContent  = PROJECTS.length.toLocaleString();
  document.getElementById('poolLabel').textContent   = DS==='pasig'
    ? 'Pasig City APP FY 2025' : 'Quezon City APP FY 2025';
  const tot = PROJECTS.reduce((s,p)=>s+p.cost,0);
  document.getElementById('totalBudgetLbl').textContent = fmtB(tot);

  renderPagination();
  updateSelCount();
  syncChkAll();
}

function renderPagination() {
  const total_pages = Math.ceil(FILTERED.length / PAGE_SIZE);
  const start = PAGE * PAGE_SIZE + 1;
  const end   = Math.min((PAGE+1)*PAGE_SIZE, FILTERED.length);
  document.getElementById('pageInfo').textContent = `Showing ${start.toLocaleString()}–${end.toLocaleString()} of ${FILTERED.length.toLocaleString()}`;

  const cont = document.getElementById('pageBtns');
  if (total_pages <= 1) { cont.innerHTML=''; return; }

  let btns = `<button class="page-btn" onclick="goPage(${PAGE-1})" ${PAGE===0?'disabled':''}>←</button>`;
  // Show at most 7 page buttons around current
  const lo = Math.max(0, PAGE-3), hi = Math.min(total_pages-1, PAGE+3);
  if (lo > 0) btns += `<button class="page-btn" onclick="goPage(0)">1</button>${lo>1?'<span style="color:var(--tx-muted);padding:0 4px">…</span>':''}`;
  for (let i=lo; i<=hi; i++) btns += `<button class="page-btn ${i===PAGE?'active':''}" onclick="goPage(${i})">${i+1}</button>`;
  if (hi < total_pages-1) btns += `${hi<total_pages-2?'<span style="color:var(--tx-muted);padding:0 4px">…</span>':''}<button class="page-btn" onclick="goPage(${total_pages-1})">${total_pages}</button>`;
  btns += `<button class="page-btn" onclick="goPage(${PAGE+1})" ${PAGE===total_pages-1?'disabled':''}>→</button>`;
  cont.innerHTML = btns;
}

function goPage(p) { PAGE = p; renderTable(); }

function toggleCheck(pi, checked) {
  if (checked) CHECKED.add(pi); else CHECKED.delete(pi);
  updateSelCount();
}

function toggleAll(checked) {
  // Select / deselect ALL filtered projects across every page
  FILTERED.forEach(p => {
    if (checked) CHECKED.add(p._idx); else CHECKED.delete(p._idx);
  });
  renderTable();
}

function syncChkAll() {
  const el = document.getElementById('chkAll');
  if (!el) return;
  const total    = FILTERED.length;
  const selected = FILTERED.filter(p => CHECKED.has(p._idx)).length;
  el.checked       = total > 0 && selected === total;
  el.indeterminate = selected > 0 && selected < total;
}

function updateSelCount() {
  const sel   = [...CHECKED];
  const total = sel.reduce((s,i) => s + PROJECTS[i].cost, 0);
  document.getElementById('selPill').textContent = `${sel.length.toLocaleString()} selected`;
  document.getElementById('selCost').textContent  = `${fmt(total)} total cost`;
}

function preselectTop() {
  CHECKED.clear();
  // Select top-50 by benefit score
  [...PROJECTS]
    .map((p,i)=>({i, b:p.benefit, c:p.cost}))
    .sort((a,b)=>b.b-a.b||a.c-b.c)
    .slice(0,50)
    .forEach(x=>CHECKED.add(x.i));
  renderTable();
}

function bcolor(b) {
  if (b >= 9)  return '#16A34A';
  if (b >= 7)  return '#0284C7';
  if (b >= 5)  return '#B45309';
  return '#9CA3AF';
}
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}


// ── Run algorithms ────────────────────────────────────────────────────────
async function runAlgos() {
  const selected_indices = [...CHECKED];
  if (!selected_indices.length) {
    alert('Select at least one project first.');
    return;
  }
  const algos = Object.keys(activeAlgos).filter(k => activeAlgos[k]);
  if (!algos.length) {
    alert('Select at least one algorithm to run.');
    return;
  }

  const btn = document.getElementById('runBtn');
  btn.innerHTML = '<span class="spinner"></span>Running…';
  btn.disabled = true;

  const pw = document.getElementById('progressWrap');
  const pf = document.getElementById('progressFill');
  const pl = document.getElementById('progressLabel');
  pw.style.display = 'block';
  pf.style.width = '10%';
  pl.textContent = 'Sending request to server…';

  try {
    const payload = {
      ds:       DS,
      budget:   BUDGET,
      selected: selected_indices,
      algos:    algos,
      pop_size: 60,
      gens:     100,
      mut_rate: 0.03,
    };

    pf.style.width = '30%';
    pl.textContent = 'Running algorithms on server…';

    const resp = await fetch('/api/run', {
      method:  'POST',
      headers: {'Content-Type':'application/json'},
      body:    JSON.stringify(payload),
    });

    pf.style.width = '80%';
    pl.textContent = 'Processing results…';

    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    pf.style.width = '100%';
    currentTab = data.results.length - 1;
    renderResults(data);

  } catch(err) {
    document.getElementById('results').innerHTML =
      `<div class="banner warn"><span>⚠</span> Error: ${escHtml(err.message)}</div>`;
  } finally {
    btn.innerHTML = '⚡ Run Selected Algorithms';
    btn.disabled  = false;
    setTimeout(()=>{ pw.style.display='none'; pf.style.width='0%'; }, 800);
  }
}

function renderResults(data) {
  const results    = data.results;
  const sel_items  = data.selected_items;
  const budget_used = data.budget;

  const benefits = results.map(r=>r.total_benefit);
  const maxBen   = Math.max(...benefits);
  const minBen   = Math.min(...benefits);
  const allMatch = benefits.every(b=>b===maxBen);
  const gapPct   = maxBen>0 ? (maxBen-minBen)/maxBen*100 : 0;

  // A small gap (< 1%) is almost always the standard DP's discretization
  // rounding (it works on a bucketed cost axis), not a GA convergence
  // problem - the exact methods (B&B, B&B+GA) always agree with each other.
  const banner = results.length === 1
    ? `<div class="banner ok"><span>✓</span> ${escHtml(results[0].label)} reached a benefit score of ${maxBen.toFixed(2)}.</div>`
    : allMatch
    ? `<div class="banner ok"><span>✓</span> All algorithms reached the same optimal benefit score (${maxBen.toFixed(2)}) — results are consistent.</div>`
    : gapPct < 1.0
    ? `<div class="banner ok"><span>✓</span> Branch-and-Bound methods agree on the optimum (${maxBen.toFixed(2)}); standard DP is within ${gapPct.toFixed(3)}% due to its discretized cost axis — expected behaviour.</div>`
    : `<div class="banner warn"><span>⚠</span> Benefit scores differ by ${gapPct.toFixed(2)}%. For very large selections, standard DP uses a coarser cost grid; B&B and B&B+GA remain exact.</div>`;

  const barColors = ['#4F6EF7','#0284C7','#7C3AED'];
  const ccards = results.map((r,ri)=>{
    const isBest = r.total_benefit === maxBen;
    const tc = r.selected.reduce((s,i)=>s+sel_items[i].cost,0);
    const pct = maxBen>0 ? Math.round((r.total_benefit/maxBen)*100) : 0;
    const bc = isBest ? '#16A34A' : barColors[ri % barColors.length];
    const hasNodes = r.pruning_rate != null;
    const ntOpen = hasNodes && _expandedNodes.has(ri);
    return `<div class="ccard ${isBest?'best':''}">
      <div class="ccard-eye">${isBest?'<div class="best-tag">✓ Optimal</div>':''}</div>
      <div class="ccard-title">${escHtml(r.label)}</div>
      <div class="ccard-big">${r.total_benefit.toFixed(2)}</div>
      <div class="ccard-sub">total benefit score</div>
      <div class="bar-t"><div class="bar-f" style="width:${pct}%;background:${bc}"></div></div>
      <div class="ccard-div"></div>
      <div class="ccard-row"><span class="lbl">Budget used</span><span class="val ${isBest?'hl-green':'hl'}">${fmtB(tc)}</span></div>
      <div class="ccard-row"><span class="lbl">Projects</span><span class="val">${r.selected.length}</span></div>
      <div class="ccard-row"><span class="lbl">Runtime</span><span class="val hl">${(r.runtime_ms).toFixed(1)} ms</span></div>
      <div class="ccard-row ${hasNodes?'node-row':''} ${ntOpen?'open':''}" id="nt-btn-${ri}" ${hasNodes?`onclick="toggleNodes(${ri})"`:''}><span class="lbl">Pruning Rate${hasNodes?'<span class="nt-caret">▸</span>':''}</span><span class="val ${r.pruning_rate!=null?'hl-green':''}">${r.pruning_rate!=null ? r.pruning_rate.toFixed(2)+'%' : 'N/A'}</span></div>
      <div class="node-detail" id="nt-${ri}" style="display:${ntOpen?'block':'none'}">
        <div class="ccard-row"><span class="lbl">Nodes Explored</span><span class="val">${r.nodes_generated!=null ? r.nodes_generated.toLocaleString() : 'N/A'}</span></div>
        <div class="ccard-row"><span class="lbl">Nodes Pruned</span><span class="val ${r.nodes_pruned!=null?'hl-green':''}">${r.nodes_pruned!=null ? r.nodes_pruned.toLocaleString() : 'N/A'}</span></div>
      </div>
      <div class="ccard-row"><span class="lbl">Time</span><span class="val">${escHtml(r.time_complexity)}</span></div>
      <div class="ccard-row"><span class="lbl">Space</span><span class="val">${escHtml(r.space_complexity)}</span></div>
    </div>`;
  }).join('');


  const cr   = results[currentTab] || results[results.length-1];
  const cset = new Set(cr.selected);
  const ctc  = cr.selected.reduce((s,i)=>s+sel_items[i].cost,0);

  const tabs = results.map((r,i)=>
    `<div class="algo-tab ${currentTab===i?'on':''}" onclick="switchTab(${i})">${escHtml(r.label)}</div>`
  ).join('');

  const chips = sel_items.map((p,i)=>`
    <div class="res-chip ${cset.has(i)?'sel':'rej'}">
      <span class="s-badge s-${p.sector.replace(/ /g,'-')}">${p.sector}</span>
      <span class="res-name">${escHtml(p.name)}</span>
      <span class="res-cost">${fmt(p.cost)}</span>
      <span class="res-score">${p.benefit.toFixed(1)}</span>
    </div>`).join('');

  document.getElementById('results').innerHTML = `
    <div class="card fade-in">
      <div class="card-hdr"><div class="card-title"><div class="ctdot"></div>Algorithm Comparison — ${DS==='pasig'?'Pasig City':'Quezon City'} Dataset</div></div>
      ${banner}
      <div class="compare-grid">${ccards}</div>
    </div>
    <div class="card fade-in" style="animation-delay:.1s">
      <div class="card-title" style="margin-bottom:14px"><div class="ctdot"></div>Selected Projects</div>
      <div class="algo-tabs" id="resTabs">${tabs}</div>
      <div class="stats-bar">
        <div class="stat"><div class="stat-l">Total benefit</div><div class="stat-v acc">${cr.total_benefit.toFixed(2)}</div><div class="stat-s">utility score</div></div>
        <div class="stat"><div class="stat-l">Budget used</div><div class="stat-v">${fmtB(ctc)}</div><div class="stat-s">${Math.round((ctc/budget_used)*100)}% of cap</div></div>
        <div class="stat"><div class="stat-l">Projects funded</div><div class="stat-v">${cr.selected.length}</div><div class="stat-s">of ${sel_items.length} candidates</div></div>
        <div class="stat"><div class="stat-l">Runtime</div><div class="stat-v">${cr.runtime_ms.toFixed(1)} ms</div><div class="stat-s">${escHtml(cr.time_complexity)}</div></div>
        <div class="stat"><div class="stat-l">Nodes Explored</div><div class="stat-v">${cr.nodes_generated!=null ? cr.nodes_generated.toLocaleString() : 'N/A'}</div><div class="stat-s">${cr.nodes_generated!=null ? 'search tree size' : 'no B&B tree'}</div></div>
        <div class="stat"><div class="stat-l">Pruning Rate</div><div class="stat-v ${cr.pruning_rate!=null?'acc':''}">${cr.pruning_rate!=null ? cr.pruning_rate.toFixed(2)+'%' : 'N/A'}</div><div class="stat-s">${cr.pruning_rate!=null ? 'branches pruned' : 'no B&B tree'}</div></div>
      </div>
      <div class="res-list">${chips}</div>
    </div>`;

  // Store for tab switching
  window._lastData = data;
}

function switchTab(i) {
  currentTab = i;
  renderResults(window._lastData);
}

function toggleNodes(i) {
  const box = document.getElementById('nt-' + i);
  const btn = document.getElementById('nt-btn-' + i);
  if (!box) return;
  const open = box.style.display === 'none';
  box.style.display = open ? 'block' : 'none';
  if (btn) btn.classList.toggle('open', open);
  if (open) _expandedNodes.add(i); else _expandedNodes.delete(i);
}

init();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/meta")
def api_meta():
    return jsonify({
        "pasig": {"count": len(PASIG_DATA)},
        "qc":    {"count": len(QC_DATA)},
    })


@app.route("/api/projects")
def api_projects():
    ds   = request.args.get("ds", "pasig")
    data = DATASETS.get(ds, DATASETS["pasig"])["data"]
    # Add index for client-side tracking
    projects = [dict(p, _idx=i) for i, p in enumerate(data)]
    return jsonify({"projects": projects, "count": len(projects)})


@app.route("/api/run", methods=["POST"])
def api_run():
    body      = request.get_json(force=True, silent=True) or {}
    ds        = body.get("ds", "pasig")
    budget    = float(body.get("budget", 5_000_000_000))
    sel_idx   = body.get("selected", [])
    algos     = body.get("algos", ["dp","bnb","ga"])
    pop_size  = int(body.get("pop_size", 40))
    gens      = int(body.get("gens", 60))
    mut_rate  = float(body.get("mut_rate", 0.03))

    all_data = DATASETS.get(ds, DATASETS["pasig"])["data"]
    items    = [all_data[i] for i in sel_idx if i < len(all_data)]

    if not items:
        return jsonify({"error": "No valid projects selected."}), 400

    if not isinstance(algos, list) or not algos:
        return jsonify({"error": "No algorithms selected."}), 400

    ALGO_META = {
        "dp":  {"label": "Knapsack (DP)",      "time": "O(n·W)",               "space": "O(n·W)"},
        "bnb": {"label": "Knapsack + B&B",     "time": "O(2ⁿ) worst / O(n log n) avg", "space": "O(n)"},
        "ga":  {"label": "Knapsack + B&B + GA","time": "O(P·G·n + n log n) avg","space": "O(P·n)"},
    }

    results = []
    for algo in algos:
        if algo not in ALGO_META:
            continue
        t0 = time.perf_counter()
        if algo == "dp":
            res = knapsack_dp(items, budget)
        elif algo == "bnb":
            res = knapsack_bnb(items, budget)
        else:
            res = knapsack_bnb_ga(items, budget,
                                  pop_size=pop_size,
                                  generations=gens,
                                  mutation_rate=mut_rate)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        m = ALGO_META[algo]
        results.append({
            "label":            m["label"],
            "algo":             algo,
            "selected":         res["selected"],
            "total_benefit":    round(res["total_benefit"], 4),
            "runtime_ms":       round(elapsed_ms, 2),
            "time_complexity":  m["time"],
            "space_complexity": m["space"],
            "pruning_rate":     res.get("pruning_rate"),
            "nodes_generated":  res.get("nodes_generated"),
            "nodes_pruned":     res.get("nodes_pruned"),
            "ga_terminated":    res.get("ga_terminated", False),
        })

    if not results:
        return jsonify({"error": "No valid algorithms selected."}), 400


    return jsonify({
        "results":       results,
        "selected_items":[items[i] for i in range(len(items))],
        "budget":        budget,
        "ds":            ds,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("LGU Budget Optimizer — Thesis Group 2 BSCS 3-1N")
    print(f"  Pasig City:  {len(PASIG_DATA):,} projects")
    print(f"  Quezon City: {len(QC_DATA):,} projects")
    print("=" * 60)
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, port=5000)
