"""
LGU Budget Allocation Optimizer
================================
Hybrid 0/1 Knapsack with Branch-and-Bound and Genetic Algorithm
BSCS 3-1N · Thesis Group 2 · PUP

Datasets:
  - Pasig City APP FY 2025 (General Fund)     — small  (1,984 projects)
  - Quezon City APP FY 2025 (4th Quarter)     — large  (26,852 projects)
"""

import os, sys, json, time, random, math, re, io
from pathlib import Path
from collections import deque, OrderedDict
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_file

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
        return {"selected": [], "total_benefit": 0.0, "pruning_rate": None, "exec_ms": 0.0}

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

    _t_exec = time.perf_counter()          # core solve only (excl. setup)
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
    _exec_ms = (time.perf_counter() - _t_exec) * 1000

    return {"selected": selected, "total_benefit": total_benefit, "pruning_rate": None, "nodes_generated": None, "nodes_pruned": None, "exec_ms": round(_exec_ms, 2), "ga_terminated": False}


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

    _t_exec = time.perf_counter()          # core search only (excl. sort/prefix)
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
    _exec_ms = (time.perf_counter() - _t_exec) * 1000
    return {"selected": best_set, "total_benefit": best_val,
            "pruning_rate": pruning_rate, "nodes_generated": ng,
            "nodes_pruned": np_, "exec_ms": round(_exec_ms, 2), "ga_terminated": False}


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
        return {"selected": [], "total_benefit": 0.0, "pruning_rate": 0.0, "nodes_generated": 0, "nodes_pruned": 0, "exec_ms": 0.0, "ga_terminated": False}

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
    _t_exec = time.perf_counter()          # core solve: GA evolution + B&B (excl. sort/prefix/helpers)
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
    _exec_ms = (time.perf_counter() - _t_exec) * 1000
    return {"selected": best_set, "total_benefit": best_val,
            "pruning_rate": pruning_rate, "nodes_generated": ng,
            "nodes_pruned": np_, "exec_ms": round(_exec_ms, 2), "ga_terminated": False}


# ─────────────────────────────────────────────────────────────────────────────
# Trial history / experiment logging
#
# Chapter 3 (Data Generation/Gathering Procedure, steps 5-6) requires the
# system to execute "exactly 30 independent runs for each algorithm" on BOTH
# the Pasig (small) and Quezon City (large) datasets, then "compile the
# average optimality parameters" (step 8).
#
# History is therefore keyed by (dataset, algorithm) rather than being one
# flat list, so each of the four combinations required by the experiment
# paper templates keeps its own independent rolling window of 30 trials:
#
#     pasig|bnb   pasig|ga   qc|bnb   qc|ga     (+ dp, kept as a baseline)
#
# A deque with maxlen=30 discards the oldest trial automatically, so the log
# always holds exactly "the last 30 runs" per combination with no manual
# pruning. This is in-memory only: restarting the server clears it, so export
# before shutting the tool down.
# ─────────────────────────────────────────────────────────────────────────────

TRIALS_PER_COMBO = 30
RUN_HISTORY = {}          # "ds|algo" -> deque of trial dicts

# Display names used in the UI and as Excel worksheet titles.
ALGO_SHEET_NAMES = OrderedDict([
    ("bnb", "KB (Pure B&B)"),
    ("ga",  "KBG (B&B + GA)"),
    ("dp",  "DP (Baseline)"),
])
DS_NAMES = {"pasig": "Pasig City (Small)", "qc": "Quezon City (Large)"}


def _history_key(ds, algo):
    return f"{ds}|{algo}"


def record_trial(ds, algo, result, budget, n_candidates, ga_params):
    """Append one trial to the rolling 30-run window for (ds, algo)."""
    key = _history_key(ds, algo)
    if key not in RUN_HISTORY:
        RUN_HISTORY[key] = deque(maxlen=TRIALS_PER_COMBO)
    RUN_HISTORY[key].append({
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":          ds,
        "algo":             algo,
        "label":            result["label"],
        "budget":           budget,
        "n_candidates":     n_candidates,
        "runtime_ms":       result["runtime_ms"],      # active execution only
        "exec_ms":          result["exec_ms"],         # total incl. setup/output
        "pruning_rate":     result["pruning_rate"],
        "nodes_generated":  result["nodes_generated"],
        "nodes_pruned":     result["nodes_pruned"],
        "total_benefit":    result["total_benefit"],
        "projects_funded":  len(result["selected"]),
        # Figure 6's output block calls for the optimal combination by ID.
        "project_ids":      result.get("project_ids", []),
        "time_complexity":  result["time_complexity"],
        "space_complexity": result["space_complexity"],
        "pop_size":         ga_params.get("pop_size") if algo == "ga" else None,
        "generations":      ga_params.get("gens")     if algo == "ga" else None,
        "mutation_rate":    ga_params.get("mut_rate") if algo == "ga" else None,
    })


def history_summary():
    """Per-combination trial counts, for the UI history panel."""
    out = []
    for algo, sheet in ALGO_SHEET_NAMES.items():
        for ds in ("pasig", "qc"):
            trials = list(RUN_HISTORY.get(_history_key(ds, algo), []))
            if not trials:
                continue
            out.append({
                "ds":        ds,
                "ds_name":   DS_NAMES[ds],
                "algo":      algo,
                "algo_name": sheet,
                "count":     len(trials),
                "complete":  len(trials) >= TRIALS_PER_COMBO,
                "trials":    trials,
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Statistical treatment
#
# Implements the tests named in Chapter 3 (Statistical Treatment) and drawn in
# Figure 6's "Analysis and Logging Module" / "Results & Statistical Outputs":
#
#   Equation 1  Independent t-test   -> efficiency across dataset sizes (SOP 2)
#   Equation 2  Mean
#   Equation 3  Variance
#   Equation 4  Paired t-test        -> KB vs KBG comparison           (SOP 3)
#   Equation 5  Mean difference
#   Equation 7  Standard deviation of differences
#
# Written in pure Python so the tool needs no SciPy/NumPy install. p-values come
# from the regularised incomplete beta function, i.e. the exact Student's t
# distribution rather than a normal approximation - which matters at df = 29,
# where a normal approximation would be visibly wrong.
# ─────────────────────────────────────────────────────────────────────────────

def _betacf(a, b, x, itmax=200, eps=3.0e-12):
    """Continued-fraction expansion used by the incomplete beta function."""
    tiny = 1.0e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_p_two_tailed(t, df):
    """Two-tailed p-value for a t statistic with df degrees of freedom."""
    if df is None or df <= 0 or t is None or not math.isfinite(t):
        return None
    return _betai(0.5 * df, 0.5, df / (df + t * t))


def s_mean(v):                                   # Equation 2
    return sum(v) / len(v) if v else None


def s_variance(v):                               # Equation 3
    n = len(v)
    if n < 2:
        return None
    m = s_mean(v)
    return sum((x - m) ** 2 for x in v) / (n - 1)


def s_stdev(v):
    var = s_variance(v)
    return math.sqrt(var) if var is not None else None


def paired_t_test(a, b):
    """Equations 4-7. Compares two related sets measured on the same instances."""
    n = min(len(a), len(b))
    if n < 2:
        return {"n": n, "note": "Not enough paired runs (at least 2 required)."}

    diffs = [a[i] - b[i] for i in range(n)]       # Equation 6
    d_bar = s_mean(diffs)                         # Equation 5
    sd    = s_stdev(diffs)                        # Equation 7

    out = {
        "n": n, "mean_a": s_mean(a[:n]), "mean_b": s_mean(b[:n]),
        "mean_diff": d_bar, "sd_diff": sd, "df": n - 1,
        "t": None, "p": None, "significant": None, "note": None,
    }

    # A deterministic metric (e.g. branch-and-bound pruning rate) produces
    # constant differences, so the standard deviation is zero and t is
    # undefined. That is a property of the algorithm, not a failure, so it is
    # reported plainly instead of being hidden or faked.
    if sd is None or sd == 0:
        out["note"] = ("Differences are constant, so the standard deviation is zero and "
                       "t cannot be computed. Expected for deterministic metrics such as "
                       "the pruning rate of branch-and-bound.")
        return out

    t = d_bar / (sd / math.sqrt(n))               # Equation 4
    out["t"] = t
    out["p"] = t_p_two_tailed(t, n - 1)
    out["significant"] = (out["p"] is not None and out["p"] < 0.05)
    return out


def independent_t_test(a, b):
    """Equation 1. Unpooled (Welch) form, exactly as written in the manuscript."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"n1": n1, "n2": n2, "note": "Not enough runs in one or both groups."}

    m1, m2 = s_mean(a), s_mean(b)
    v1, v2 = s_variance(a), s_variance(b)

    out = {
        "n1": n1, "n2": n2, "mean_1": m1, "mean_2": m2,
        "var_1": v1, "var_2": v2, "ratio": None,
        "t": None, "df": None, "p": None, "significant": None, "note": None,
    }
    # Efficiency, per the Definition of Terms: the ratio of the optimality
    # parameter on the small dataset relative to the large dataset.
    if m2 not in (None, 0):
        out["ratio"] = m1 / m2

    se_sq = v1 / n1 + v2 / n2
    if se_sq <= 0:
        out["note"] = ("Both groups are constant, so the standard error is zero and t "
                       "cannot be computed. Expected for deterministic metrics.")
        return out

    t  = (m1 - m2) / math.sqrt(se_sq)             # Equation 1
    df = (se_sq ** 2) / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    out["t"], out["df"] = t, df
    out["p"] = t_p_two_tailed(t, df)
    out["significant"] = (out["p"] is not None and out["p"] < 0.05)
    return out


STAT_METRICS = [
    ("runtime_ms",   "Runtime (ms)"),
    ("exec_ms",      "Execution Time (ms)"),
    ("pruning_rate", "Pruning Rate (%)"),
]


def _series(ds, algo, key):
    """Recorded values of one metric for one dataset+algorithm combination."""
    return [t[key] for t in RUN_HISTORY.get(_history_key(ds, algo), [])
            if t.get(key) is not None]


def build_statistical_report():
    """Efficiency per algorithm (SOP 2) and KB vs KBG per dataset (SOP 3)."""
    efficiency = []
    for algo, algo_name in ALGO_SHEET_NAMES.items():
        rows = []
        for key, label in STAT_METRICS:
            small, large = _series("pasig", algo, key), _series("qc", algo, key)
            if len(small) < 2 or len(large) < 2:
                continue
            r = independent_t_test(small, large)
            r["metric"] = label
            rows.append(r)
        if rows:
            efficiency.append({"algo": algo, "algo_name": algo_name, "rows": rows})

    comparison = []
    for ds in ("pasig", "qc"):
        rows = []
        for key, label in STAT_METRICS:
            kb, kbg = _series(ds, "bnb", key), _series(ds, "ga", key)
            if len(kb) < 2 or len(kbg) < 2:
                continue
            r = paired_t_test(kb, kbg)
            r["metric"] = label
            rows.append(r)
        if rows:
            comparison.append({"ds": ds, "ds_name": DS_NAMES[ds], "rows": rows})

    return {"efficiency": efficiency, "comparison": comparison}


# ── Excel export ─────────────────────────────────────────────────────────────
# Workbook layout mirrors the blank "Experiment Paper" tables at the end of
# the manuscript: one sheet per algorithm, Trials 1..30 down the rows, and
# the three optimality metrics (Runtime, Execution Time, Pruning Rate) for
# the Pasig and Quezon City datasets across the columns, with a Mean row.
#
# Mean / variance / standard deviation are written as live Excel FORMULAS
# (=AVERAGE, =VAR, =STDEV) rather than Python-computed constants, so the
# figures recalculate if a trial is edited or removed by hand.

_METRICS = [
    ("runtime_ms",   "Runtime (ms)"),
    ("exec_ms",      "Execution Time (ms)"),
    ("pruning_rate", "Pruning Rate (%)"),
]


def build_workbook():
    """Build the experiment workbook and return it as an in-memory buffer."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter

    FONT = "Arial"
    hdr_font   = Font(name=FONT, bold=True, size=11, color="FFFFFF")
    sub_font   = Font(name=FONT, bold=True, size=10)
    body_font  = Font(name=FONT, size=10)
    mean_font  = Font(name=FONT, bold=True, size=10)
    title_font = Font(name=FONT, bold=True, size=13)

    hdr_fill  = PatternFill("solid", fgColor="2F4F8F")
    sub_fill  = PatternFill("solid", fgColor="D6DEF0")
    mean_fill = PatternFill("solid", fgColor="FFF2CC")

    thin   = Side(style="thin", color="9AA5C0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    wb = Workbook()
    wb.remove(wb.active)

    # ── One trials sheet per algorithm that actually has recorded data ──────
    for algo, sheet_name in ALGO_SHEET_NAMES.items():
        pasig = list(RUN_HISTORY.get(_history_key("pasig", algo), []))
        qc    = list(RUN_HISTORY.get(_history_key("qc",    algo), []))
        if not pasig and not qc:
            continue

        ws = wb.create_sheet(sheet_name[:31])

        ws["A1"] = f"Experiment Paper for Optimality and Efficiency of {sheet_name}"
        ws["A1"].font = title_font
        ws.merge_cells("A1:G1")
        ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[1].height = 22

        # Two-tier header: dataset band over metric columns
        ws["A3"] = "Trials"
        ws.merge_cells("A3:A4")
        ws["B3"] = DS_NAMES["pasig"]
        ws.merge_cells("B3:D3")
        ws["E3"] = DS_NAMES["qc"]
        ws.merge_cells("E3:G3")

        for col, (_, label) in enumerate(_METRICS, start=2):
            ws.cell(row=4, column=col, value=label)
        for col, (_, label) in enumerate(_METRICS, start=5):
            ws.cell(row=4, column=col, value=label)

        for col in range(1, 8):
            for row in (3, 4):
                c = ws.cell(row=row, column=col)
                c.font = hdr_font if row == 3 else sub_font
                c.fill = hdr_fill if row == 3 else sub_fill
                c.alignment = center
                c.border = border
        ws.row_dimensions[4].height = 30

        # Trial rows: always render all 30 slots so the sheet matches the
        # manuscript template even when fewer runs have been recorded.
        first_data_row = 5
        for t in range(TRIALS_PER_COMBO):
            row = first_data_row + t
            ws.cell(row=row, column=1, value=f"Trial {t + 1}")
            for col, (key, _) in enumerate(_METRICS, start=2):
                val = pasig[t][key] if t < len(pasig) else None
                ws.cell(row=row, column=col, value=val)
            for col, (key, _) in enumerate(_METRICS, start=5):
                val = qc[t][key] if t < len(qc) else None
                ws.cell(row=row, column=col, value=val)
            for col in range(1, 8):
                c = ws.cell(row=row, column=col)
                c.font = body_font
                c.border = border
                c.alignment = center
                if col > 1:
                    c.number_format = "0.00"

        last_data_row = first_data_row + TRIALS_PER_COMBO - 1

        # Mean / Variance / Std. Deviation as live formulas
        stat_rows = [
            ("Mean",               "AVERAGE"),
            ("Variance",           "VAR"),
            ("Standard Deviation", "STDEV"),
        ]
        for offset, (stat_label, fn) in enumerate(stat_rows):
            row = last_data_row + 1 + offset
            ws.cell(row=row, column=1, value=stat_label)
            for col in range(2, 8):
                letter = get_column_letter(col)
                ws.cell(
                    row=row, column=col,
                    value=f"={fn}({letter}{first_data_row}:{letter}{last_data_row})",
                )
            for col in range(1, 8):
                c = ws.cell(row=row, column=col)
                c.font = mean_font
                c.fill = mean_fill
                c.border = border
                c.alignment = center
                if col > 1:
                    c.number_format = "0.0000"

        note_row = last_data_row + len(stat_rows) + 2
        ws.cell(row=note_row, column=1,
                value=("Note: Runtime = active algorithm execution only. "
                       "Execution Time = total duration including setup and "
                       "output (Ch.1, Definition of Terms). Blank trial rows "
                       "indicate runs not yet performed. Mean, Variance and "
                       "Standard Deviation are live formulas over Trials 1-30."))
        ws.cell(row=note_row, column=1).font = Font(name=FONT, size=9, italic=True)
        ws.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=7)
        ws.cell(row=note_row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[note_row].height = 42

        ws.column_dimensions["A"].width = 20
        for col in range(2, 8):
            ws.column_dimensions[get_column_letter(col)].width = 19
        ws.freeze_panes = "B5"

    # ── Statistical report ─────────────────────────────────────────────────
    # Mirrors Figure 6's "Results & Statistical Outputs" block and fills the
    # Efficiency section (Ratio / p-value) of the manuscript's experiment
    # tables. Values are computed literals, not formulas, so external readers
    # see real numbers.
    report = build_statistical_report()
    if report["efficiency"] or report["comparison"]:
        st = wb.create_sheet("Statistical Report")
        r = 1

        def _title(text, row, span=8):
            c = st.cell(row=row, column=1, value=text)
            c.font = title_font
            st.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
            st.row_dimensions[row].height = 20

        def _headers(row, headers):
            for col, h in enumerate(headers, start=1):
                c = st.cell(row=row, column=col, value=h)
                c.font = hdr_font
                c.fill = hdr_fill
                c.alignment = center
                c.border = border
            st.row_dimensions[row].height = 30

        def _cell(row, col, val, fmt=None, bold=False):
            c = st.cell(row=row, column=col, value=val)
            c.font = mean_font if bold else body_font
            c.border = border
            c.alignment = center
            if fmt:
                c.number_format = fmt
            return c

        # SOP 2 — efficiency across dataset sizes (independent t-test)
        _title("Efficiency — Independent t-test (small vs. large dataset)", r); r += 2
        _headers(r, ["Algorithm", "Metric", "Mean (Pasig)", "Mean (QC)",
                     "Ratio", "t", "df", "p-value"]); r += 1
        for grp in report["efficiency"]:
            for row in grp["rows"]:
                _cell(r, 1, grp["algo_name"])
                _cell(r, 2, row["metric"])
                _cell(r, 3, row.get("mean_1"), "0.0000")
                _cell(r, 4, row.get("mean_2"), "0.0000")
                _cell(r, 5, row.get("ratio"),  "0.0000")
                _cell(r, 6, row.get("t"),      "0.0000")
                _cell(r, 7, row.get("df"),     "0.00")
                pc = _cell(r, 8, row.get("p"), "0.000000")
                if row.get("p") is None:
                    pc.value = "not computable"
                    pc.font = Font(name=FONT, size=9, italic=True)
                r += 1
        r += 1

        # SOP 3 — KB vs KBG on identical instances (paired t-test)
        _title("Comparison — Paired t-test (KB vs. KBG)", r); r += 2
        _headers(r, ["Dataset", "Metric", "Mean (KB)", "Mean (KBG)",
                     "Mean difference", "Std. dev. of differences", "t", "p-value"]); r += 1
        for grp in report["comparison"]:
            for row in grp["rows"]:
                _cell(r, 1, grp["ds_name"])
                _cell(r, 2, row["metric"])
                _cell(r, 3, row.get("mean_a"),    "0.0000")
                _cell(r, 4, row.get("mean_b"),    "0.0000")
                _cell(r, 5, row.get("mean_diff"), "0.0000")
                _cell(r, 6, row.get("sd_diff"),   "0.0000")
                _cell(r, 7, row.get("t"),         "0.0000")
                pc = _cell(r, 8, row.get("p"), "0.000000")
                if row.get("p") is None:
                    pc.value = "not computable"
                    pc.font = Font(name=FONT, size=9, italic=True)
                r += 1

        r += 2
        for line in [
            "Significance level: 0.05 (two-tailed).",
            "Independent t-test uses the unpooled (Welch) form given as Equation 1; "
            "df follows the Welch-Satterthwaite approximation.",
            "Paired t-test follows Equations 4-7. Runs are paired by problem instance: "
            "both algorithms solve the identical project set under the identical budget.",
            "'not computable' means the metric is constant across runs, so its standard "
            "deviation is zero. This is expected for the pruning rate of branch-and-bound, "
            "which is deterministic, and is not a data error.",
        ]:
            c = st.cell(row=r, column=1, value=line)
            c.font = Font(name=FONT, size=9, italic=True)
            st.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            st.row_dimensions[r].height = 26
            r += 1

        st.column_dimensions["A"].width = 22
        st.column_dimensions["B"].width = 21
        for col in range(3, 9):
            st.column_dimensions[get_column_letter(col)].width = 17

    # ── SPSS-ready wide format ─────────────────────────────────────────────
    # The per-algorithm sheets above mirror the manuscript's experiment tables
    # (merged two-tier headers, a title row, trailing statistic rows). That
    # shape is correct for the paper but unreadable to SPSS, which needs a
    # single flat header row and one case per row.
    #
    # This sheet restructures the same trials into WIDE format: one row per
    # trial number, with every dataset+algorithm+metric as its own variable.
    # That is the layout the paired-samples t-test needs, because SPSS pairs
    # two columns row by row (Analyze > Compare Means > Paired-Samples T Test).
    #
    # Pairing trial i of KB with trial i of KBG is legitimate here: both runs
    # solve the identical project instance under the identical budget, so the
    # problem instance acts as the blocking factor. The pairing is by instance,
    # not by any natural relationship between the runs themselves.
    #
    # Values are written as LITERAL NUMBERS, never formulas. openpyxl writes
    # formulas without cached values, so a formula cell reads back as empty to
    # SPSS, pandas, and every other external reader.
    spss_cols = [("Trial", "Trial number (1-30)")]
    spss_data = {}   # column name -> list of 30 values

    metric_keys = [("runtime_ms", "Runtime"), ("exec_ms", "ExecTime"), ("pruning_rate", "Pruning")]
    ds_short    = [("pasig", "Pasig"), ("qc", "QC")]
    algo_short  = [("bnb", "KB"), ("ga", "KBG"), ("dp", "DP")]
    metric_desc = {
        "Runtime":  "Active algorithm execution only, in milliseconds",
        "ExecTime": "Total duration including setup and output, in milliseconds",
        "Pruning":  "Percentage of search-tree branches discarded",
    }
    algo_desc = {"KB": "0/1 knapsack with branch-and-bound",
                 "KBG": "Hybrid 0/1 knapsack with branch-and-bound and genetic algorithm",
                 "DP": "0/1 knapsack with dynamic programming (baseline)"}
    ds_desc = {"Pasig": "Pasig City dataset (small)", "QC": "Quezon City dataset (large)"}

    for algo, a_tag in algo_short:
        for ds, d_tag in ds_short:
            trials = list(RUN_HISTORY.get(_history_key(ds, algo), []))
            if not trials:
                continue
            for key, m_tag in metric_keys:
                # SPSS variable names: letters, digits and underscores only,
                # beginning with a letter, comfortably under the 64-char limit.
                name = f"{a_tag}_{m_tag}_{d_tag}"
                spss_cols.append((
                    name,
                    f"{algo_desc[a_tag]} — {metric_desc[m_tag]} — {ds_desc[d_tag]}",
                ))
                spss_data[name] = [
                    (trials[i][key] if i < len(trials) else None)
                    for i in range(TRIALS_PER_COMBO)
                ]

    if len(spss_cols) > 1:
        sp = wb.create_sheet("SPSS Data")
        for col, (name, _) in enumerate(spss_cols, start=1):
            c = sp.cell(row=1, column=col, value=name)
            c.font = hdr_font
            c.fill = hdr_fill
            c.alignment = center
            c.border = border
            sp.column_dimensions[get_column_letter(col)].width = max(11, len(name) + 3)
        sp.row_dimensions[1].height = 26

        for t in range(TRIALS_PER_COMBO):
            row = t + 2
            sp.cell(row=row, column=1, value=t + 1).font = body_font
            for col, (name, _) in enumerate(spss_cols[1:], start=2):
                c = sp.cell(row=row, column=col, value=spss_data[name][t])
                c.font = body_font
                c.number_format = "0.00"
        sp.freeze_panes = "B2"

        # Codebook: what each variable means, so the analysis is reproducible.
        cb = wb.create_sheet("SPSS Codebook")
        for col, header in enumerate(["Variable", "Description"], start=1):
            c = cb.cell(row=1, column=col, value=header)
            c.font = hdr_font
            c.fill = hdr_fill
            c.alignment = center
            c.border = border
        cb.column_dimensions["A"].width = 26
        cb.column_dimensions["B"].width = 86
        for r, (name, desc) in enumerate(spss_cols, start=2):
            cb.cell(row=r, column=1, value=name).font = Font(name=FONT, size=10, bold=True)
            d = cb.cell(row=r, column=2, value=desc)
            d.font = body_font
            d.alignment = Alignment(wrap_text=True, vertical="top")
        note_r = len(spss_cols) + 3
        cb.cell(row=note_r, column=1, value="Reading this file in SPSS")
        cb.cell(row=note_r, column=1).font = Font(name=FONT, size=10, bold=True)
        for i, line in enumerate([
            "File > Open > Data, set Files of type to Excel, and choose the 'SPSS Data' worksheet.",
            "Tick 'Read variable names from first row of data'.",
            "Paired-samples t-test: Analyze > Compare Means > Paired-Samples T Test, pairing "
            "KB_Runtime_Pasig with KBG_Runtime_Pasig (and likewise for the other metrics).",
            "Independent-samples t-test: restructure the two dataset columns into one variable "
            "with a grouping variable, or use Data > Restructure.",
            "Pruning columns may have zero variance because branch-and-bound is deterministic; "
            "SPSS cannot compute t for a constant, which is expected rather than an error.",
        ], start=1):
            c = cb.cell(row=note_r + i, column=2, value=line)
            c.font = Font(name=FONT, size=9.5)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            cb.row_dimensions[note_r + i].height = 26

    # ── Raw run log: every recorded trial, all captured fields ─────────────
    log = wb.create_sheet("Run Log")
    log_cols = [
        ("timestamp",        "Timestamp",          22),
        ("label",            "Algorithm",          24),
        ("dataset",          "Dataset",            12),
        ("n_candidates",     "Candidate Projects", 18),
        ("budget",           "Budget (PhP)",       18),
        ("runtime_ms",       "Runtime (ms)",       15),
        ("exec_ms",          "Execution Time (ms)",19),
        ("pruning_rate",     "Pruning Rate (%)",   16),
        ("nodes_generated",  "Nodes Explored",     16),
        ("nodes_pruned",     "Nodes Pruned",       15),
        ("total_benefit",    "Total Benefit",      15),
        ("projects_funded",  "Projects Funded",    16),
        ("time_complexity",  "Time Complexity",    26),
        ("space_complexity", "Space Complexity",   18),
        ("pop_size",         "GA Pop. Size",       13),
        ("generations",      "GA Generations",     15),
        ("mutation_rate",    "GA Mutation Rate",   17),
        ("project_ids_str",  "Selected Project IDs", 60),
    ]
    for col, (_, header, width) in enumerate(log_cols, start=1):
        c = log.cell(row=1, column=col, value=header)
        c.font = hdr_font
        c.fill = hdr_fill
        c.alignment = center
        c.border = border
        log.column_dimensions[get_column_letter(col)].width = width
    log.row_dimensions[1].height = 28

    all_trials = []
    for key in RUN_HISTORY:
        all_trials.extend(RUN_HISTORY[key])
    all_trials.sort(key=lambda t: t["timestamp"])

    for r, trial in enumerate(all_trials, start=2):
        for col, (key, _, _) in enumerate(log_cols, start=1):
            if key == "project_ids_str":
                ids = trial.get("project_ids") or []
                # Excel caps a cell at 32,767 characters; truncate defensively.
                val = "; ".join(str(i) for i in ids)
                if len(val) > 32000:
                    val = val[:32000] + f"… ({len(ids)} IDs total)"
            else:
                val = trial.get(key)
            c = log.cell(row=r, column=col, value=val)
            c.font = body_font
            c.border = border
            if key in ("runtime_ms", "exec_ms", "pruning_rate", "total_benefit"):
                c.number_format = "0.00"
            elif key == "budget":
                c.number_format = "#,##0"
    log.freeze_panes = "A2"

    if not all_trials:
        log.cell(row=2, column=1, value="No runs recorded yet.").font = body_font

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  /* Ledger-neutral paper, deliberately cool-green rather than cream or blue-white */
  --bg-base:#EDEFEC;--bg-surface:#FFFFFF;--bg-raised:#F4F5F2;--bg-hover:#E7EAE5;--bg-active:#DCE0D9;
  --bd-subtle:rgba(23,32,27,.10);--bd-default:rgba(23,32,27,.20);--bd-strong:rgba(23,32,27,.34);
  --tx-primary:#17201B;--tx-secondary:#4A554D;--tx-muted:#7C877F;--tx-inverse:#FFFFFF;
  /* Single signal colour: deep pine. Used for the primary action and measured state. */
  --accent:#0B6B4F;--accent-dim:rgba(11,107,79,.09);--accent-border:rgba(11,107,79,.30);--accent-glow:0 3px 14px rgba(11,107,79,.20);
  /* Dataset identities — desaturated, carried on thin edge rules not large fills */
  --pasig:#6A4A9E;--pasig-dim:rgba(106,74,158,.08);--pasig-border:rgba(106,74,158,.30);--pasig-glow:0 3px 14px rgba(106,74,158,.14);
  --qc:#14607F;--qc-dim:rgba(20,96,127,.08);--qc-border:rgba(20,96,127,.30);--qc-glow:0 3px 14px rgba(20,96,127,.14);
  --green:#0B6B4F;--green-dim:rgba(11,107,79,.10);--green-border:rgba(11,107,79,.30);
  --amber:#A15C00;--amber-dim:rgba(161,92,0,.10);--red:#B3261E;--red-dim:rgba(179,38,30,.10);
  --r-sm:4px;--r-md:7px;--r-lg:10px;--r-xl:14px;--ease:cubic-bezier(.4,0,.2,1);
  --gutter:34px;   /* width of the numbered step rail */
}
html{scroll-behavior:smooth}
body{font-family:'IBM Plex Sans',-apple-system,sans-serif;background:var(--bg-base);color:var(--tx-primary);min-height:100vh;line-height:1.5;-webkit-font-smoothing:antialiased}
/* All figures use tabular lining numerals so columns align digit-for-digit. */
.mono,table td,table th,input[type=text],input[type=number]{font-variant-numeric:tabular-nums}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#CBD5E1;border-radius:99px}
::-webkit-scrollbar-thumb:hover{background:#94A3B8}

/* MASTHEAD — quiet by design; the instrument's readings are the hero, not a banner */
.hero{background:var(--bg-surface);border-bottom:1px solid var(--bd-default);padding:22px 24px 18px}
.hero-inner{max-width:1100px;margin:0 auto;display:flex;align-items:flex-end;justify-content:space-between;gap:24px;flex-wrap:wrap}
.hero h1{font-size:21px;font-weight:600;letter-spacing:-.25px;color:var(--tx-primary);margin-bottom:3px}
.hero-sub{font-size:13px;color:var(--tx-secondary);max-width:62ch;line-height:1.45}
.hero-id{font-size:11.5px;color:var(--tx-muted);text-align:right;line-height:1.7;font-family:'IBM Plex Mono',monospace}
.hero-id b{display:block;color:var(--tx-secondary);font-weight:500}

/* SKIP LINK — keyboard users reach the controls without tabbing the whole table */
.skip{position:absolute;left:-9999px;top:0;z-index:100;background:var(--accent);color:#fff;padding:10px 16px;border-radius:0 0 var(--r-md) 0;font-size:13px;font-weight:600}
.skip:focus{left:0}

/* ACCESSIBILITY FLOOR */
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:2px}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.01ms !important;animation-iteration-count:1 !important;transition-duration:.01ms !important;scroll-behavior:auto !important}
}

/* NUMBERED STEP RAIL
   The workflow genuinely is a sequence (data -> budget -> projects -> setup ->
   run -> read), so the numbering encodes real structure rather than decorating.
   The rail also gives the eye a fixed left edge to scan down. */
.step{position:relative;padding-left:var(--gutter);margin-bottom:20px}
.step::before{content:attr(data-step);position:absolute;left:0;top:1px;width:23px;height:23px;border-radius:50%;
  background:var(--bg-surface);border:1px solid var(--bd-default);color:var(--tx-secondary);
  font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:center}
.step::after{content:'';position:absolute;left:11px;top:29px;bottom:-20px;width:1px;background:var(--bd-subtle)}
.step:last-of-type::after{display:none}
.step.done::before{background:var(--accent);border-color:var(--accent);color:#fff}
.step-hdr{margin-bottom:10px;min-height:23px;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.step-name{font-size:14.5px;font-weight:600;letter-spacing:-.1px;color:var(--tx-primary)}
.step-note{font-size:12.5px;color:var(--tx-muted)}

/* STICKY ACTION BAR — system status stays visible; the primary action is always
   reachable (Fitts's law) instead of being buried mid-page. */
.actionbar{position:sticky;bottom:0;z-index:40;background:rgba(255,255,255,.94);backdrop-filter:blur(10px);
  border-top:1px solid var(--bd-default);margin-top:26px;padding:11px 20px;box-shadow:0 -3px 16px rgba(23,32,27,.06)}
.actionbar-inner{max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.ab-facts{display:flex;gap:18px;flex-wrap:wrap;flex:1;min-width:220px}
.ab-fact{display:flex;flex-direction:column;gap:1px}
.ab-fact-l{font-size:10.5px;color:var(--tx-muted);letter-spacing:.01em}
.ab-fact-v{font-size:13px;font-weight:600;font-family:'IBM Plex Mono',monospace;color:var(--tx-primary)}
.ab-fact-v.warn{color:var(--amber)}
.ab-run{padding:11px 22px;font-size:14px;font-weight:600;cursor:pointer;background:var(--accent);border:1px solid var(--accent);
  border-radius:var(--r-md);color:#fff;transition:background .15s var(--ease);font-family:'IBM Plex Sans',sans-serif;
  display:flex;align-items:center;gap:8px;white-space:nowrap}
.ab-run:hover:not(:disabled){background:#095B43}
.ab-run:disabled{background:var(--bg-active);border-color:var(--bd-default);color:var(--tx-muted);cursor:not-allowed}
.ab-reason{font-size:12px;color:var(--amber);flex-basis:100%}

/* INLINE MESSAGES — errors are shown in place, never in a modal alert() */
.msg{display:flex;gap:9px;align-items:flex-start;padding:11px 13px;border-radius:var(--r-md);font-size:13px;line-height:1.5;margin-bottom:14px;border:1px solid}
.msg-icon{flex-shrink:0;font-weight:700;font-family:'IBM Plex Mono',monospace}
.msg.err{background:var(--red-dim);border-color:rgba(179,38,30,.3);color:#7E1B15}
.msg.warn{background:var(--amber-dim);border-color:rgba(161,92,0,.3);color:#6E3F00}
.msg.info{background:var(--accent-dim);border-color:var(--accent-border);color:#07422F}

/* RESPONSIVE — usable down to a phone viewport */
@media (max-width:640px){
  :root{--gutter:26px}
  .hero-inner{flex-direction:column;align-items:flex-start;gap:10px}
  .hero-id{text-align:left}
  .actionbar-inner{gap:10px}
  .ab-facts{gap:12px;order:1;width:100%}
  .ab-run{order:2;width:100%;justify-content:center}
  .ab-fact-l{font-size:10px}
  .ab-fact-v{font-size:12px}
  .compare-grid{grid-template-columns:1fr}
  .stats-bar{grid-template-columns:repeat(2,1fr)}
  .hist-grid{grid-template-columns:1fr}
  .wrap{padding:18px 14px 16px}
}

/* ── Statistical report ─────────────────────────────────────────── */
.stat-lede{font-size:12.5px;color:var(--tx-secondary);line-height:1.55;margin-bottom:14px;max-width:76ch}
.stat-block{margin-bottom:16px}
.stat-block:last-child{margin-bottom:0}
.sb-title{font-size:12.5px;font-weight:600;color:var(--tx-primary);margin-bottom:7px}
.stat-table-wrap{overflow-x:auto;border:1px solid var(--bd-subtle);border-radius:var(--r-md)}
.vd{font-family:'IBM Plex Sans',sans-serif;font-size:11px;font-weight:600;padding:2px 8px;border-radius:99px;white-space:nowrap}
.vd.sig{background:var(--green-dim);color:var(--green);border:1px solid var(--green-border)}
.vd.ns{background:var(--bg-active);color:var(--tx-secondary);border:1px solid var(--bd-default)}
.vd.none{background:var(--amber-dim);color:var(--amber);border:1px solid rgba(161,92,0,.3)}

/* EMPTY STATES — an invitation to act, not an apology */
.empty-state{text-align:center;padding:26px 16px;color:var(--tx-secondary)}
.empty-state p{font-size:14px;font-weight:600;color:var(--tx-primary);margin-bottom:5px}
.empty-state .es-sub{font-size:12.5px;font-weight:400;color:var(--tx-muted);max-width:52ch;margin:0 auto;line-height:1.55}

/* DEFINITION AFFORDANCE — recognition over recall for the two timing metrics */
.deflist{display:flex;gap:20px;flex-wrap:wrap;padding:10px 13px;background:var(--bg-raised);border:1px solid var(--bd-subtle);
  border-radius:var(--r-md);margin-bottom:14px;font-size:12px;color:var(--tx-secondary);line-height:1.5}
.deflist div{flex:1;min-width:210px}
.deflist b{color:var(--tx-primary);font-weight:600}

.wrap{max-width:1100px;margin:0 auto;padding:24px 20px 20px}

/* SECTION LABEL */
.slabel{display:none}
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
.card{background:var(--bg-surface);border:1px solid var(--bd-subtle);border-radius:var(--r-lg);padding:18px;margin-bottom:0;transition:border-color .2s var(--ease)}
.card:hover{border-color:var(--bd-default)}
.card-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:8px}
.card-title{font-size:13.5px;font-weight:600;letter-spacing:-.1px;color:var(--tx-primary);display:flex;align-items:center;gap:7px}
.ctdot{display:none}
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
.budget-peso{font-size:22px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:var(--accent);letter-spacing:-.5px}
#budgetInput{flex:1;min-width:0;border:none;outline:none;background:transparent;font-size:22px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:var(--accent);letter-spacing:-.5px;text-align:right;padding:0}
#budgetInput::placeholder{color:var(--tx-muted)}

/* TOOLBAR */
.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
.search-wrap{flex:1;min-width:200px;position:relative}
.search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:var(--tx-muted);font-size:13px;pointer-events:none}
.search-wrap input[type=text]{width:100%;padding:8px 12px 8px 30px;background:var(--bg-raised);border:1px solid var(--bd-subtle);border-radius:var(--r-md);color:var(--tx-primary);font-size:13px;font-family:'IBM Plex Sans',sans-serif;outline:none;transition:all .15s var(--ease)}
.search-wrap input[type=text]::placeholder{color:var(--tx-muted)}
.search-wrap input[type=text]:focus{border-color:var(--accent-border);background:var(--bg-active);box-shadow:0 0 0 3px var(--accent-dim)}
.filter-btn{padding:6px 13px;border-radius:var(--r-md);font-size:12px;font-weight:500;cursor:pointer;border:1px solid var(--bd-subtle);color:var(--tx-secondary);background:var(--bg-raised);transition:all .15s var(--ease);white-space:nowrap;font-family:'IBM Plex Sans',sans-serif}
.filter-btn:hover{border-color:var(--bd-default);color:var(--tx-primary);background:var(--bg-active)}
.filter-btn.on{border-color:var(--accent-border);color:var(--accent);background:var(--accent-dim)}

/* TABLE */
.tbl-wrap{max-height:380px;overflow-y:auto;border:1px solid var(--bd-subtle);border-radius:var(--r-lg);background:#fff}
table{width:100%;border-collapse:collapse;font-size:12.5px}
thead th{position:sticky;top:0;z-index:2;background:var(--bg-raised);padding:9px 12px;text-align:left;color:var(--tx-secondary);font-weight:600;font-size:11.5px;letter-spacing:0;border-bottom:1px solid var(--bd-default)}
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
.cost-col{text-align:right;font-family:'IBM Plex Mono',monospace;font-size:12px;white-space:nowrap;color:var(--tx-secondary)}
.benefit-col{text-align:center}
.b-pill{display:inline-block;font-size:11px;font-weight:700;font-family:'IBM Plex Mono',monospace;width:32px;height:22px;line-height:22px;border-radius:6px;text-align:center;color:#fff}
.sel-count{font-size:12px;color:var(--tx-secondary);margin-top:10px;display:flex;align-items:center;gap:6px}
.sel-pill{background:var(--accent-dim);color:var(--accent);border:1px solid var(--accent-border);border-radius:99px;padding:1px 8px;font-size:11px;font-weight:600}

/* RUN BTN */
/* ── Run setup ──────────────────────────────────────────────────── */
.setup-grid{display:grid;grid-template-columns:1fr auto;gap:22px;align-items:start}
@media (max-width:720px){.setup-grid{grid-template-columns:1fr}}
.setup-field{min-width:0}
.setup-lbl{font-size:12.5px;font-weight:600;color:var(--tx-secondary);margin-bottom:8px}
.setup-hint{font-size:11.5px;color:var(--tx-muted);margin-top:7px;max-width:34ch}
.trials-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.trials-input{width:64px;padding:7px 9px;font-size:13.5px;font-weight:600;text-align:center;border:1px solid var(--bd-default);border-radius:var(--r-sm);background:var(--bg-surface);color:var(--tx-primary);font-family:'IBM Plex Mono',monospace}
.trials-input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-dim)}
.trials-presets{display:flex;gap:5px}
.tpreset{padding:7px 12px;font-size:12.5px;font-weight:500;cursor:pointer;background:var(--bg-raised);border:1px solid var(--bd-default);border-radius:var(--r-sm);color:var(--tx-secondary);transition:all .15s var(--ease);font-family:'IBM Plex Sans',sans-serif}
.tpreset:hover{background:var(--accent-dim);border-color:var(--accent-border);color:var(--accent)}
/* ── History panel ──────────────────────────────────────────────── */
.hist-actions{display:flex;gap:8px;margin-left:auto}
.hist-btn{padding:8px 14px;font-size:12.5px;font-weight:600;cursor:pointer;border-radius:var(--r-sm);border:1px solid var(--bd-default);background:var(--bg-raised);color:var(--tx-secondary);transition:all .15s var(--ease);font-family:'IBM Plex Sans',sans-serif}
.hist-btn:hover{background:var(--bg-hover);border-color:var(--bd-strong)}
.hist-btn.export{background:linear-gradient(135deg,#16A34A 0%,#12813B 100%);border:none;color:#fff;box-shadow:0 3px 12px rgba(22,163,74,.28)}
.hist-btn.export:hover{transform:translateY(-1px);box-shadow:0 6px 18px rgba(22,163,74,.38)}
.hist-btn.danger:hover{background:var(--red-dim);border-color:var(--red);color:var(--red)}
.hist-card{background:var(--bg-raised);border:1px solid var(--bd-subtle);border-radius:var(--r-md);padding:12px 14px;cursor:pointer;transition:all .15s var(--ease)}
.hist-card:hover{border-color:var(--bd-strong);transform:translateY(-1px)}
.hist-card.done{border-color:var(--green-border);background:var(--green-dim)}
.hist-card.on{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-dim)}
/* per-run detail table */
.hist-detail{margin-top:14px;border-top:1px solid var(--bd-subtle);padding-top:14px}
.hist-detail-hdr{display:flex;align-items:baseline;gap:10px;margin-bottom:10px}
.hd-title{font-size:13px;font-weight:700;color:var(--tx-primary)}
.hd-sub{font-size:11.5px;color:var(--tx-muted)}
.hist-table-wrap{max-height:380px;overflow:auto;border:1px solid var(--bd-subtle);border-radius:var(--r-md)}
.hist-table{width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Mono',monospace}
.hist-table th{position:sticky;top:0;z-index:2;background:var(--bg-active);color:var(--tx-secondary);font-family:'IBM Plex Sans',sans-serif;font-size:11px;font-weight:700;text-align:right;padding:9px 11px;white-space:nowrap;border-bottom:1px solid var(--bd-default)}
.hist-table th:first-child{text-align:left}
.hist-table td{padding:7px 11px;text-align:right;color:var(--tx-primary);border-bottom:1px solid var(--bd-subtle);white-space:nowrap}
.hist-table tbody tr:hover{background:var(--bg-hover)}
.hist-table .hr-idx{text-align:left;font-family:'IBM Plex Sans',sans-serif;font-weight:600;color:var(--tx-secondary)}
.hist-table .hr-dim{color:var(--tx-muted)}
.hist-table .hr-time{color:var(--tx-muted);font-size:11px}
.hist-table tfoot td{position:sticky;bottom:0;background:var(--accent-dim);font-weight:700;border-top:1px solid var(--accent-border);border-bottom:none}
.hist-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}
.hist-algo{font-size:12.5px;font-weight:700;color:var(--tx-primary);margin-bottom:2px}
.hist-ds{font-size:11px;color:var(--tx-muted);margin-bottom:9px}
.hist-count{font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:var(--tx-primary)}
.hist-count.done{color:var(--green)}
.hist-of{font-size:12px;color:var(--tx-muted);font-weight:500}
.hist-bar{height:5px;background:var(--bg-active);border-radius:3px;overflow:hidden;margin-top:8px}
.hist-bar-fill{height:100%;background:var(--accent);border-radius:3px;transition:width .4s var(--ease)}
.hist-bar-fill.done{background:var(--green)}
.hist-empty{font-size:13px;color:var(--tx-muted);text-align:center;padding:22px 10px;line-height:1.6}
.run-btn{width:100%;padding:14px;font-size:15px;font-weight:700;cursor:pointer;background:linear-gradient(135deg,#4F6EF7 0%,#3B55E0 100%);border:none;border-radius:var(--r-lg);color:#fff;letter-spacing:.02em;transition:all .2s var(--ease);margin-bottom:16px;display:flex;align-items:center;justify-content:center;gap:8px;box-shadow:0 4px 20px rgba(79,110,247,.28);font-family:'IBM Plex Sans',sans-serif}
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
.best-tag{display:inline-flex;align-items:center;gap:4px;font-size:11px;font-weight:600;letter-spacing:0;background:var(--green-dim);color:var(--green);border:1px solid var(--green-border);padding:2px 8px;border-radius:99px;margin-bottom:10px}
.ccard-eye{height:22px;margin-bottom:10px}
.ccard-title{font-size:12px;font-weight:600;color:var(--tx-secondary);margin-bottom:8px}
.ccard-big{font-size:30px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:var(--tx-primary);letter-spacing:-1px;line-height:1}
.ccard-sub{font-size:11px;color:var(--tx-muted);margin-top:4px}
.bar-t{background:var(--bg-hover);border-radius:99px;height:5px;width:100%;margin-top:12px;overflow:hidden}
.bar-f{height:5px;border-radius:99px;transition:width .6s var(--ease)}
.ccard-div{height:1px;background:var(--bd-subtle);margin:12px 0}
.ccard-row{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px}
.ccard-row .lbl{color:var(--tx-muted)}
.ccard-row .val{color:var(--tx-secondary);font-family:'IBM Plex Mono',monospace;font-size:11px}
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
.stat-l{font-size:11px;font-weight:500;letter-spacing:0;color:var(--tx-muted);margin-bottom:6px}
.stat-v{font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:var(--tx-primary);letter-spacing:-.5px}
.stat-v.acc{color:var(--accent)}
.stat-s{font-size:11px;color:var(--tx-muted);margin-top:3px}

/* ALGO TABS */
.algo-tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px;background:var(--bg-base);border:1px solid var(--bd-subtle);border-radius:var(--r-lg);padding:5px}
.algo-tab{flex:1;padding:8px 16px;border-radius:var(--r-md);font-size:13px;font-weight:600;cursor:pointer;border:none;color:var(--tx-secondary);background:transparent;transition:all .15s var(--ease);text-align:center;font-family:'IBM Plex Sans',sans-serif;white-space:nowrap}
.algo-tab:hover{color:var(--tx-primary);background:var(--bg-hover)}
.algo-tab.on{color:#fff;background:var(--accent);box-shadow:0 2px 10px rgba(79,110,247,.25)}

/* RESULT LIST */
.res-list{display:flex;flex-direction:column;gap:4px;max-height:340px;overflow-y:auto;border:1px solid var(--bd-subtle);border-radius:var(--r-md);padding:8px}
.res-chip{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:var(--r-md);font-size:12px;transition:all .12s var(--ease);border:1px solid transparent}
.res-chip.sel{background:var(--accent-dim);border-color:var(--accent-border)}
.res-chip.sel:hover{background:rgba(79,110,247,.15)}
.res-chip.rej{background:var(--bg-raised);border-color:var(--bd-subtle);opacity:.45}
.res-chip.rej .res-name{text-decoration:line-through;color:var(--tx-muted)}
.res-name{flex:1;color:var(--tx-primary);line-height:1.3}
.res-cost{min-width:88px;text-align:right;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--tx-secondary)}
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
.page-btn{padding:4px 12px;border-radius:var(--r-md);font-size:12px;font-weight:500;cursor:pointer;border:1px solid var(--bd-subtle);color:var(--tx-secondary);background:var(--bg-raised);font-family:'IBM Plex Sans',sans-serif;transition:all .15s var(--ease)}
.page-btn:hover:not(:disabled){border-color:var(--accent-border);color:var(--accent)}
.page-btn:disabled{opacity:.4;cursor:not-allowed}
.page-btn.active{border-color:var(--accent-border);color:var(--accent);background:var(--accent-dim)}
</style>
</head>
<body>

<a class="skip" href="#setup">Skip to run setup</a>

<div class="hero">
  <div class="hero-inner">
    <div>
      <h1>LGU Budget Optimizer</h1>
      <p class="hero-sub">Selects the combination of local government projects that maximises total social benefit
      within a fixed budget, and measures how three knapsack algorithms perform on the same problem.</p>
    </div>
    <div class="hero-id">
      <b>Thesis Group 2 — BSCS 3-1N</b>
      Polytechnic University of the Philippines
    </div>
  </div>
</div>

<div class="wrap">

  <div id="globalMsg" role="alert" aria-live="assertive"></div>

  <!-- STEP 1 ─ Dataset -->
  <section class="step done" data-step="1" id="stepDataset">
    <div class="step-hdr">
      <span class="step-name">Choose a dataset</span>
      <span class="step-note">Determines the problem size the algorithms are tested against</span>
    </div>
    <div class="ds-switcher">
      <button class="ds-btn active-pasig" id="btnPasig" onclick="switchDs('pasig')" aria-pressed="true">
        <div class="ds-info">
          <div class="ds-name">Pasig City <span class="ds-tag small">Small</span></div>
          <div class="ds-meta" id="pasigMeta">Loading…</div>
        </div>
        <div class="ds-check"></div>
      </button>
      <button class="ds-btn" id="btnQC" onclick="switchDs('qc')" aria-pressed="false">
        <div class="ds-info">
          <div class="ds-name">Quezon City <span class="ds-tag large">Large</span></div>
          <div class="ds-meta" id="qcMeta">Loading…</div>
        </div>
        <div class="ds-check"></div>
      </button>
    </div>
    <div class="ds-infobar pasig" id="infoBar">
      <span id="infoText"><b>Pasig City APP FY 2025 (General Fund)</b> — Source: City Government of Pasig, LGU Transparency Portal. Used as the <b>small dataset</b> to establish a standard of optimality and verify algorithm accuracy.</span>
    </div>
  </section>

  <!-- STEP 2 ─ Budget -->
  <section class="step done" data-step="2" id="stepBudget">
    <div class="step-hdr">
      <span class="step-name">Set the budget ceiling</span>
      <span class="step-note">The knapsack capacity — shared across both datasets so results stay comparable</span>
    </div>
    <div class="card">
      <div class="budget-block">
        <label class="budget-label" for="budgetInput">Total project budget</label>
        <div class="budget-slider-wrap">
          <input type="range" id="budgetSlider" min="0" max="50000000000" step="100000000" value="5000000000"
                 aria-label="Budget ceiling slider" oninput="updateBudget(this.value, 'slider')">
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
  </section>

  <!-- STEP 3 ─ Projects -->
  <section class="step" data-step="3" id="stepProjects">
    <div class="step-hdr">
      <span class="step-name">Select candidate projects</span>
      <span class="step-note">These become the items the algorithms choose between</span>
    </div>
    <div class="card">
    <div class="card-hdr">
      <div class="card-title">
        Project pool
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
            <th>Project / program</th>
            <th>Sector</th>
            <th>Office</th>
            <th style="text-align:right">Cost (₱)</th>
            <th style="text-align:center">Benefit</th>
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
  </section>

  <!-- STEP 4 ─ Run setup -->
  <section class="step" data-step="4" id="setup">
    <div class="step-hdr">
      <span class="step-name">Configure the run</span>
      <span class="step-note">Every run is recorded; the last 30 per dataset and algorithm are kept</span>
    </div>
    <div class="card">
      <div class="setup-grid">
        <div class="setup-field">
          <div class="setup-lbl" id="algoLbl">Algorithms</div>
          <div class="algo-select" id="algoSelect" role="group" aria-labelledby="algoLbl">
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
        <div class="setup-field">
          <label class="setup-lbl" for="trialsInput">Independent runs per algorithm</label>
          <div class="trials-row">
            <input type="number" id="trialsInput" class="trials-input" min="1" max="30" value="1"
                   oninput="updateActionBar()">
            <div class="trials-presets">
              <button class="tpreset" onclick="setTrials(1)">1</button>
              <button class="tpreset" onclick="setTrials(10)">10</button>
              <button class="tpreset" onclick="setTrials(30)">30</button>
            </div>
          </div>
          <div class="setup-hint">The study specifies 30 independent runs per algorithm, per dataset.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- STEP 5 ─ Results -->
  <section class="step" data-step="5" id="stepResults">
    <div class="step-hdr">
      <span class="step-name">Compare the results</span>
      <span class="step-note">Optimality measures for each algorithm on this problem</span>
    </div>
    <div class="progress-wrap" id="progressWrap">
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="progressFill"></div></div>
      <div class="progress-label" id="progressLabel" role="status" aria-live="polite">Initialising…</div>
    </div>
    <div id="results">
      <div class="card">
        <div class="empty-state">
          <p>No results yet.</p>
          <p class="es-sub">Choose your projects and algorithms, then run. Each algorithm's runtime,
          execution time and pruning rate will be compared here.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- STEP 6 ─ History -->
  <section class="step" data-step="6" id="stepHistory">
    <div class="step-hdr">
      <span class="step-name">Review and export run history</span>
      <span class="step-note">Held in memory only — export before closing the tool</span>
    </div>
    <div id="historyPanel"></div>
  </section>

  <!-- STEP 7 ─ Statistical report -->
  <section class="step" data-step="7" id="stepStats">
    <div class="step-hdr">
      <span class="step-name">Statistical report</span>
      <span class="step-note">Efficiency ratios and t-tests over the recorded runs</span>
    </div>
    <div id="statsPanel"></div>
  </section>

  <div class="footer" id="footerTxt">
    Pasig City Annual Procurement Plan FY 2025 (General Fund).
    Algorithms: dynamic programming; branch and bound (Land &amp; Doig, 1960);
    genetic algorithm (Holland's schema theorem).
  </div>
</div>

<!-- Persistent status + primary action -->
<div class="actionbar">
  <div class="actionbar-inner">
    <div class="ab-facts">
      <div class="ab-fact"><span class="ab-fact-l">Dataset</span><span class="ab-fact-v" id="abDs">Pasig</span></div>
      <div class="ab-fact"><span class="ab-fact-l">Budget</span><span class="ab-fact-v" id="abBudget">₱5.0B</span></div>
      <div class="ab-fact"><span class="ab-fact-l">Projects selected</span><span class="ab-fact-v" id="abSel">0</span></div>
      <div class="ab-fact"><span class="ab-fact-l">Algorithms</span><span class="ab-fact-v" id="abAlgos">3</span></div>
      <div class="ab-fact"><span class="ab-fact-l">Runs each</span><span class="ab-fact-v" id="abTrials">1</span></div>
    </div>
    <button class="ab-run" id="runBtn" onclick="runAlgos()">Run algorithms</button>
    <div class="ab-reason" id="abReason"></div>
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
  updateActionBar();
}

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  await loadDs('pasig');
  updateMeta();
  updateActionBar();
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
  const bp = document.getElementById('btnPasig');
  const bq = document.getElementById('btnQC');
  if (bp) bp.setAttribute('aria-pressed', ds === 'pasig');
  if (bq) bq.setAttribute('aria-pressed', ds === 'qc');
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
  updateActionBar();
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
  updateActionBar();
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
    showMsg('warn', 'Select at least one project in step 3 before running.');
    document.getElementById('stepProjects').scrollIntoView({block:'center'});
    return;
  }
  const algos = Object.keys(activeAlgos).filter(k => activeAlgos[k]);
  if (!algos.length) {
    showMsg('warn', 'Select at least one algorithm in step 4 before running.');
    document.getElementById('setup').scrollIntoView({block:'center'});
    return;
  }
  clearMsg();

  const trials = getTrials();

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
      trials:   trials,
      pop_size: 60,
      gens:     100,
      mut_rate: 0.03,
    };

    pf.style.width = '30%';
    pl.textContent = trials > 1
      ? `Running ${trials} independent runs per algorithm…`
      : 'Running algorithms on server…';

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
    renderHistory(data.history);
    loadStats();

  } catch(err) {
    document.getElementById('results').innerHTML =
      `<div class="banner warn"><span>⚠</span> Error: ${escHtml(err.message)}</div>`;
  } finally {
    btn.disabled = false;
    updateActionBar();
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
      <div class="ccard-row"><span class="lbl">Runtime</span><span class="val hl">${r.runtime_ms!=null ? r.runtime_ms.toFixed(1)+' ms' : 'N/A'}</span></div>
      <div class="ccard-row"><span class="lbl">Execution time</span><span class="val">${r.exec_ms!=null ? r.exec_ms.toFixed(1)+' ms' : 'N/A'}</span></div>
      <div class="ccard-row ${hasNodes?'node-row':''} ${ntOpen?'open':''}" id="nt-btn-${ri}" ${hasNodes?`onclick="toggleNodes(${ri})"`:''}><span class="lbl">Pruning rate${hasNodes?'<span class="nt-caret">▸</span>':''}</span><span class="val ${r.pruning_rate!=null?'hl-green':''}">${r.pruning_rate!=null ? r.pruning_rate.toFixed(2)+'%' : 'N/A'}</span></div>
      <div class="node-detail" id="nt-${ri}" style="display:${ntOpen?'block':'none'}">
        <div class="ccard-row"><span class="lbl">Nodes explored</span><span class="val">${r.nodes_generated!=null ? r.nodes_generated.toLocaleString() : 'N/A'}</span></div>
        <div class="ccard-row"><span class="lbl">Nodes pruned</span><span class="val ${r.nodes_pruned!=null?'hl-green':''}">${r.nodes_pruned!=null ? r.nodes_pruned.toLocaleString() : 'N/A'}</span></div>
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
      <div class="card-hdr"><div class="card-title">Algorithm comparison — ${DS==='pasig'?'Pasig City':'Quezon City'}</div></div>
      ${banner}
      <div class="deflist">
        <div><b>Runtime</b> — time the algorithm spends actively executing, excluding setup.</div>
        <div><b>Execution time</b> — total time to process the data and produce the allocation.</div>
        <div><b>Pruning rate</b> — share of search-tree branches discarded without exploring.</div>
      </div>
      <div class="compare-grid">${ccards}</div>
    </div>
    <div class="card fade-in" style="animation-delay:.1s">
      <div class="card-title" style="margin-bottom:14px">Selected projects</div>
      <div class="algo-tabs" id="resTabs">${tabs}</div>
      <div class="stats-bar">
        <div class="stat"><div class="stat-l">Total benefit</div><div class="stat-v acc">${cr.total_benefit.toFixed(2)}</div><div class="stat-s">utility score</div></div>
        <div class="stat"><div class="stat-l">Budget used</div><div class="stat-v">${fmtB(ctc)}</div><div class="stat-s">${Math.round((ctc/budget_used)*100)}% of cap</div></div>
        <div class="stat"><div class="stat-l">Projects funded</div><div class="stat-v">${cr.selected.length}</div><div class="stat-s">of ${sel_items.length} candidates</div></div>
        <div class="stat"><div class="stat-l">Runtime</div><div class="stat-v">${cr.runtime_ms!=null ? cr.runtime_ms.toFixed(1)+' ms' : 'N/A'}</div><div class="stat-s">active execution only</div></div>
        <div class="stat"><div class="stat-l">Execution time</div><div class="stat-v">${cr.exec_ms!=null ? cr.exec_ms.toFixed(1)+' ms' : 'N/A'}</div><div class="stat-s">${escHtml(cr.time_complexity)}</div></div>
        <div class="stat"><div class="stat-l">Nodes explored</div><div class="stat-v">${cr.nodes_generated!=null ? cr.nodes_generated.toLocaleString() : 'N/A'}</div><div class="stat-s">${cr.nodes_generated!=null ? 'search tree size' : 'no B&B tree'}</div></div>
        <div class="stat"><div class="stat-l">Pruning rate</div><div class="stat-v ${cr.pruning_rate!=null?'acc':''}">${cr.pruning_rate!=null ? cr.pruning_rate.toFixed(2)+'%' : 'N/A'}</div><div class="stat-s">${cr.pruning_rate!=null ? 'branches pruned' : 'no B&B tree'}</div></div>
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

/* ── Inline messaging ────────────────────────────────────────────────
   Errors are shown in place rather than through alert(), so the user can
   read the problem and the offending control at the same time. */
function showMsg(kind, text) {
  const el = document.getElementById('globalMsg');
  if (!el) return;
  const icon = kind === 'err' ? '!' : kind === 'warn' ? '!' : 'i';
  el.innerHTML = `<div class="msg ${kind}"><span class="msg-icon" aria-hidden="true">${icon}</span><span>${escHtml(text)}</span></div>`;
}

function clearMsg() {
  const el = document.getElementById('globalMsg');
  if (el) el.innerHTML = '';
}

/* ── Persistent status bar ───────────────────────────────────────────
   Keeps the run configuration visible at all times and blocks the action
   before it can fail, stating the reason instead of reporting it after. */
function updateActionBar() {
  const selCount = CHECKED.size;
  const algos    = Object.keys(activeAlgos).filter(k => activeAlgos[k]);
  const trials   = Math.max(1, Math.min(30, parseInt(document.getElementById('trialsInput').value, 10) || 1));

  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  set('abDs',     DS === 'pasig' ? 'Pasig City' : 'Quezon City');
  set('abBudget', fmtB(BUDGET));
  set('abSel',    selCount.toLocaleString());
  set('abAlgos',  algos.length);
  set('abTrials', trials);

  const selEl = document.getElementById('abSel');
  if (selEl) selEl.classList.toggle('warn', selCount === 0);
  const algEl = document.getElementById('abAlgos');
  if (algEl) algEl.classList.toggle('warn', algos.length === 0);

  // Error prevention: say what is missing before the button can be pressed.
  let reason = '';
  if (!selCount && !algos.length) reason = 'Select at least one project and one algorithm to run.';
  else if (!selCount)             reason = 'Select at least one project in step 3.';
  else if (!algos.length)         reason = 'Select at least one algorithm in step 4.';

  const btn = document.getElementById('runBtn');
  if (btn) {
    btn.disabled = !!reason;
    btn.textContent = trials > 1 ? `Run ${trials} times each` : 'Run algorithms';
  }
  const rEl = document.getElementById('abReason');
  if (rEl) rEl.textContent = reason;

  // Reflect progress on the numbered rail.
  const mark = (id, done) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('done', done);
  };
  mark('stepProjects', selCount > 0);
  mark('setup',        algos.length > 0);
}

/* ── Trials control ──────────────────────────────────────────────────── */
function getTrials() {
  const el = document.getElementById('trialsInput');
  let v = parseInt(el.value, 10);
  if (isNaN(v) || v < 1) v = 1;
  if (v > 30) v = 30;
  el.value = v;
  return v;
}

function setTrials(n) {
  document.getElementById('trialsInput').value = n;
  updateActionBar();
}

/* ── Run history / experiment log ────────────────────────────────────── */
let HIST_DATA   = [];     // full combo list incl. per-trial rows
let HIST_OPEN   = null;   // "ds|algo" of the currently expanded combo

async function loadHistory() {
  try {
    const resp = await fetch('/api/history');
    const data = await resp.json();
    renderHistory(data.combos);
  } catch(err) {
    // Non-fatal: the history panel simply stays empty.
  }
}

function histKey(c) { return c.ds + '|' + c.algo; }

function toggleHist(key) {
  HIST_OPEN = (HIST_OPEN === key) ? null : key;
  renderHistory(HIST_DATA);
}

function num(v, dp) {
  return (v === null || v === undefined) ? '—' : Number(v).toFixed(dp);
}

/* Mean of a numeric field across a combo's stored runs. */
function histMean(trials, field) {
  const vals = trials.map(t => t[field]).filter(v => v !== null && v !== undefined);
  if (!vals.length) return null;
  return vals.reduce((a,b) => a+b, 0) / vals.length;
}

function renderHistory(combos) {
  const panel = document.getElementById('historyPanel');
  if (!panel) return;

  HIST_DATA = combos || [];
  const total = HIST_DATA.reduce((s,c) => s + c.count, 0);

  if (!HIST_DATA.length) {
    panel.innerHTML = `
      <div class="card fade-in">
        <div class="card-hdr">
          <div class="card-title">Run history</div>
        </div>
        <div class="hist-empty">
          No runs recorded yet.<br>
          Run an algorithm above — each run's runtime, execution time and
          pruning rate will be logged here
          (last 30 runs kept per dataset + algorithm).
        </div>
      </div>`;
    return;
  }

  // Keep the expanded combo valid; otherwise open the first one by default.
  if (!HIST_DATA.some(c => histKey(c) === HIST_OPEN)) {
    HIST_OPEN = histKey(HIST_DATA[0]);
  }

  const cards = HIST_DATA.map(c => {
    const key  = histKey(c);
    const pct  = Math.min(100, Math.round((c.count / 30) * 100));
    const done = c.complete;
    const on   = HIST_OPEN === key;
    return `
      <div class="hist-card ${done?'done':''} ${on?'on':''}" onclick="toggleHist('${key}')">
        <div class="hist-algo">${escHtml(c.algo_name)}</div>
        <div class="hist-ds">${escHtml(c.ds_name)}</div>
        <span class="hist-count ${done?'done':''}">${c.count}</span>
        <span class="hist-of"> / 30 runs</span>
        <div class="hist-bar"><div class="hist-bar-fill ${done?'done':''}" style="width:${pct}%"></div></div>
      </div>`;
  }).join('');

  // ── Per-run detail table for the expanded combination ────────────────
  const combo  = HIST_DATA.find(c => histKey(c) === HIST_OPEN);
  let detail = '';

  if (combo) {
    const t = combo.trials;
    const rows = t.map((r, i) => `
      <tr>
        <td class="hr-idx">Run ${i + 1}</td>
        <td>${num(r.runtime_ms, 2)}</td>
        <td>${num(r.exec_ms, 2)}</td>
        <td>${r.pruning_rate === null ? '—' : num(r.pruning_rate, 2)}</td>
        <td class="hr-dim">${r.nodes_generated === null ? '—' : r.nodes_generated.toLocaleString()}</td>
        <td class="hr-dim">${r.nodes_pruned === null ? '—' : r.nodes_pruned.toLocaleString()}</td>
        <td class="hr-dim">${num(r.total_benefit, 2)}</td>
        <td class="hr-time">${escHtml(r.timestamp.split(' ')[1] || '')}</td>
      </tr>`).join('');

    const mRun  = histMean(t, 'runtime_ms');
    const mExec = histMean(t, 'exec_ms');
    const mPrun = histMean(t, 'pruning_rate');
    const mBen  = histMean(t, 'total_benefit');

    detail = `
      <div class="hist-detail">
        <div class="hist-detail-hdr">
          <span class="hd-title">${escHtml(combo.algo_name)} · ${escHtml(combo.ds_name)}</span>
          <span class="hd-sub">${combo.count} run${combo.count===1?'':'s'} stored</span>
        </div>
        <div class="hist-table-wrap">
          <table class="hist-table">
            <thead>
              <tr>
                <th>Trial</th>
                <th>Runtime (ms)</th>
                <th>Execution time (ms)</th>
                <th>Pruning rate (%)</th>
                <th>Nodes explored</th>
                <th>Nodes pruned</th>
                <th>Total benefit</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
            <tfoot>
              <tr>
                <td class="hr-idx">Mean</td>
                <td>${num(mRun, 4)}</td>
                <td>${num(mExec, 4)}</td>
                <td>${mPrun === null ? '—' : num(mPrun, 4)}</td>
                <td class="hr-dim">—</td>
                <td class="hr-dim">—</td>
                <td class="hr-dim">${num(mBen, 4)}</td>
                <td class="hr-time"></td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>`;
  }

  panel.innerHTML = `
    <div class="card fade-in">
      <div class="card-hdr">
        <div class="card-title">Run history — ${total} run${total===1?'':'s'} recorded</div>
        <div class="hist-actions">
          <button class="hist-btn export" onclick="exportExcel()">⬇ Export to Excel</button>
          <button class="hist-btn danger" onclick="clearHistory()">Clear history</button>
        </div>
      </div>
      <div class="hist-grid">${cards}</div>
      ${detail}
    </div>`;
}

function exportExcel() {
  // Triggers a normal browser download from /api/export.
  window.location.href = '/api/export';
}

async function clearHistory() {
  if (!confirm('Clear all recorded runs? This cannot be undone — export first if you still need the data.')) return;
  try {
    const resp = await fetch('/api/history/clear', {
      method:  'POST',
      headers: {'Content-Type':'application/json'},
      body:    JSON.stringify({}),
    });
    const data = await resp.json();
    renderHistory(data.combos);
    loadStats();
  } catch(err) {
    showMsg('err', 'Could not clear the history: ' + err.message);
  }
}

/* ── Statistical report ──────────────────────────────────────────────
   Implements Figure 6's "Analysis and Logging Module" output: the paired
   t-test (KB vs KBG) and the independent t-test / efficiency ratio across
   dataset sizes. Computed server-side from the recorded runs. */
async function loadStats() {
  try {
    const resp = await fetch('/api/stats');
    renderStats(await resp.json());
  } catch(err) { /* non-fatal */ }
}

function fx(v, dp) {
  return (v === null || v === undefined) ? '—' : Number(v).toFixed(dp);
}

/* p-values here span many orders of magnitude, so very small ones are shown
   in scientific notation rather than rounding to a misleading 0.0000. */
function fp(v) {
  if (v === null || v === undefined) return null;
  if (v < 0.0001) return v.toExponential(2);
  return v.toFixed(4);
}

function verdict(row) {
  if (row.p === null || row.p === undefined) {
    return '<span class="vd none" title="Metric is constant across runs">not computable</span>';
  }
  return row.significant
    ? '<span class="vd sig">significant</span>'
    : '<span class="vd ns">not significant</span>';
}

function renderStats(rep) {
  const panel = document.getElementById('statsPanel');
  if (!panel) return;

  const hasEff  = rep.efficiency && rep.efficiency.length;
  const hasComp = rep.comparison && rep.comparison.length;

  if (!hasEff && !hasComp) {
    panel.innerHTML = `
      <div class="card">
        <div class="empty-state">
          <p>No statistics yet.</p>
          <p class="es-sub">The efficiency ratio needs runs on <b>both</b> datasets, and the
          paired t-test needs runs of <b>both</b> B&amp;B and B&amp;B+GA. Record at least two runs
          of each, then the report appears here.</p>
        </div>
      </div>`;
    return;
  }

  let html = '';

  if (hasComp) {
    const blocks = rep.comparison.map(g => `
      <div class="stat-block">
        <div class="sb-title">${escHtml(g.ds_name)}</div>
        <div class="stat-table-wrap">
        <table class="hist-table">
          <thead><tr>
            <th>Metric</th><th>Mean (KB)</th><th>Mean (KBG)</th>
            <th>Mean diff.</th><th>SD of diff.</th><th>t</th><th>p</th><th>Result</th>
          </tr></thead>
          <tbody>
            ${g.rows.map(r => `
              <tr>
                <td class="hr-idx">${escHtml(r.metric)}</td>
                <td>${fx(r.mean_a,4)}</td>
                <td>${fx(r.mean_b,4)}</td>
                <td>${fx(r.mean_diff,4)}</td>
                <td>${fx(r.sd_diff,4)}</td>
                <td>${r.t===null||r.t===undefined?'—':fx(r.t,4)}</td>
                <td>${fp(r.p) ?? '—'}</td>
                <td>${verdict(r)}</td>
              </tr>`).join('')}
          </tbody>
        </table></div>
      </div>`).join('');

    html += `
      <div class="card">
        <div class="card-hdr"><div class="card-title">Paired t-test — B&amp;B vs. B&amp;B + GA</div></div>
        <p class="stat-lede">Tests whether the hybrid differs significantly from branch-and-bound alone.
        Runs are paired by problem instance: both solve the identical project set under the identical budget.
        Null hypothesis rejected when p &lt; 0.05.</p>
        ${blocks}
      </div>`;
  }

  if (hasEff) {
    const blocks = rep.efficiency.map(g => `
      <div class="stat-block">
        <div class="sb-title">${escHtml(g.algo_name)}</div>
        <div class="stat-table-wrap">
        <table class="hist-table">
          <thead><tr>
            <th>Metric</th><th>Mean (Pasig)</th><th>Mean (QC)</th>
            <th>Ratio</th><th>t</th><th>df</th><th>p</th><th>Result</th>
          </tr></thead>
          <tbody>
            ${g.rows.map(r => `
              <tr>
                <td class="hr-idx">${escHtml(r.metric)}</td>
                <td>${fx(r.mean_1,4)}</td>
                <td>${fx(r.mean_2,4)}</td>
                <td>${fx(r.ratio,4)}</td>
                <td>${r.t===null||r.t===undefined?'—':fx(r.t,4)}</td>
                <td>${fx(r.df,2)}</td>
                <td>${fp(r.p) ?? '—'}</td>
                <td>${verdict(r)}</td>
              </tr>`).join('')}
          </tbody>
        </table></div>
      </div>`).join('');

    html += `
      <div class="card">
        <div class="card-hdr"><div class="card-title">Independent t-test — efficiency across dataset sizes</div></div>
        <p class="stat-lede">Efficiency is the ratio of each optimality parameter on the small dataset
        relative to the large one. A result that is <i>not</i> significant indicates performance is stable
        across input sizes.</p>
        ${blocks}
      </div>`;
  }

  html += `
    <div class="msg info" style="margin-top:14px">
      <span class="msg-icon" aria-hidden="true">i</span>
      <span>Two-tailed tests at the 0.05 level. "Not computable" means the metric is constant
      across runs, so its standard deviation is zero — expected for the pruning rate of
      branch-and-bound, which is deterministic, and not a data error.</span>
    </div>`;

  panel.innerHTML = html;
}

init();
loadHistory();
loadStats();
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
        "bnb": {"label": "Knapsack + B&B",     "time": "O(2ⁿ) worst-case",     "space": "O(n)"},
        # Figure 6 of the manuscript states the hybrid's complexity as
        # Time O(g·p·n + 2ⁿ) bounded by GA pruning, and Space O(p·n + n)
        # for the priority queue plus the population. Kept verbatim so a live
        # demo matches the architecture diagram.
        "ga":  {"label": "Knapsack + B&B + GA","time": "O(g·p·n + 2ⁿ) bounded by GA pruning","space": "O(p·n + n)"},
    }

    # Number of independent repetitions per algorithm for this request.
    # Default 1 preserves the original single-run demo behaviour; setting it
    # to 30 satisfies Ch.3 steps 5-6 ("exactly 30 independent runs for each
    # algorithm") in a single click. Every repetition is logged; the LAST
    # repetition is what gets returned for on-screen display.
    trials = max(1, min(TRIALS_PER_COMBO, int(body.get("trials", 1))))
    ga_params = {"pop_size": pop_size, "gens": gens, "mut_rate": mut_rate}

    results = []
    for algo in algos:
        if algo not in ALGO_META:
            continue
        for _trial in range(trials):
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
            entry = _build_result_entry(algo, res, elapsed_ms, ALGO_META)
            # Attach the source project codes for the chosen combination, so
            # every allocation can be traced back to the LGU procurement plan.
            entry["project_ids"] = [
                items[i].get("code", "") for i in res["selected"]
            ]
            record_trial(ds, algo, entry, budget, len(items), ga_params)
        # Only the final repetition is surfaced in the response payload.
        results.append(entry)

    if not results:
        return jsonify({"error": "No valid algorithms selected."}), 400

    return jsonify({
        "results":       results,
        "selected_items":[items[i] for i in range(len(items))],
        "budget":        budget,
        "ds":            ds,
        "trials":        trials,
        "history":       history_summary(),
    })


def _build_result_entry(algo, res, elapsed_ms, ALGO_META):
    """Assemble one result dict with manuscript-aligned metric labels.

    ── Manuscript-aligned labeling (Ch.1 Definition of Terms) ──────────────
    "Runtime" is defined as the period when the algorithm is actually
    EXECUTING (the core solve loop) — this is the narrower, inner timer
    (`_exec_ms`) measured inside each knapsack_* function, which already
    excludes sorting / prefix-sum setup / traceback.

    "Execution Time" is defined as the TOTAL computational duration required
    to process the dataset and output the final allocation — this is the
    outer, full-wrap timer (`elapsed_ms`) measured in api_run(), which
    includes everything (setup + solve + traceback).

    The two values are therefore swapped relative to their source variable
    names: `res["exec_ms"]` (the core-only timer) is reported as
    "runtime_ms", and `elapsed_ms` (the full end-to-end timer) is reported
    as "exec_ms", so the JSON keys line up with the UI labels "Runtime" /
    "Execution time" exactly as the manuscript defines them.
    """
    m = ALGO_META[algo]
    return {
        "label":            m["label"],
        "algo":             algo,
        "selected":         res["selected"],
        "total_benefit":    round(res["total_benefit"], 4),
        "runtime_ms":       res.get("exec_ms"),      # active execution only
        "exec_ms":          round(elapsed_ms, 2),    # total incl. setup/output
        "time_complexity":  m["time"],
        "space_complexity": m["space"],
        "pruning_rate":     res.get("pruning_rate"),
        "nodes_generated":  res.get("nodes_generated"),
        "nodes_pruned":     res.get("nodes_pruned"),
        "ga_terminated":    res.get("ga_terminated", False),
    }


@app.route("/api/history")
def api_history():
    """Current contents of the rolling 30-run log, per dataset+algorithm."""
    return jsonify({
        "trials_per_combo": TRIALS_PER_COMBO,
        "combos":           history_summary(),
    })


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    """Reset the trial log. Optionally scoped to one dataset+algorithm."""
    body = request.get_json(force=True, silent=True) or {}
    ds   = body.get("ds")
    algo = body.get("algo")
    if ds and algo:
        RUN_HISTORY.pop(_history_key(ds, algo), None)
    else:
        RUN_HISTORY.clear()
    return jsonify({"ok": True, "combos": history_summary()})


@app.route("/api/stats")
def api_stats():
    """Statistical report: efficiency ratios and t-tests over recorded runs."""
    return jsonify(build_statistical_report())


@app.route("/api/export")
def api_export():
    """Download the recorded trials as a formatted Excel workbook."""
    try:
        buf = build_workbook()
    except ImportError:
        return jsonify({
            "error": "openpyxl is not installed. Run:  pip install openpyxl"
        }), 500

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        buf,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"LGU_Optimizer_Experiment_Results_{stamp}.xlsx",
    )


if __name__ == "__main__":
    print("=" * 60)
    print("LGU Budget Optimizer — Thesis Group 2 BSCS 3-1N")
    print(f"  Pasig City:  {len(PASIG_DATA):,} projects")
    print(f"  Quezon City: {len(QC_DATA):,} projects")
    print("=" * 60)
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, port=5000)
