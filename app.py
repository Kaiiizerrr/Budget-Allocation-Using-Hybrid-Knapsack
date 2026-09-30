"""
LGU Budget Allocation Optimizer
================================
Hybrid 0/1 Knapsack with Branch-and-Bound and Genetic Algorithm
BSCS 4-1N · Thesis Group 2 · PUP

Datasets:
  - Pasig City APP FY 2025 (General Fund)     — small  (1,991 projects)
  - Quezon City APP FY 2025 (4th Quarter)     — large  (26,865 projects)
"""

import os, sys, json, time, random, math, re, io
from pathlib import Path
from collections import deque, OrderedDict, Counter as collections_Counter
from datetime import datetime
from flask import (Flask, render_template, request, jsonify, send_file,
                   Response, stream_with_context)

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
                    pop_size=40, generations=60, mutation_rate=None):
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
        """Chu & Beasley (1998) repair operator: DROP then ADD.

        DROP removes the lowest benefit/cost items until the chromosome is
        within budget; ADD then spends any leftover budget on the highest
        benefit/cost items that still fit. Without the ADD phase a mutation
        that added a project left the freed budget unspent, so offspring were
        almost always worse than their parents and the GA never produced a
        seed better than the greedy solution.
        """
        if total_cost > capacity:
            for i in reversed(ratio_order):
                if total_cost <= capacity:
                    break
                if chrom[i]:
                    chrom[i] = 0
                    total_cost -= costs[i]
        for i in ratio_order:
            if not chrom[i] and costs[i] <= capacity - total_cost:
                chrom[i] = 1
                total_cost += costs[i]
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
    # Mutation defaults to 1/n - one expected flip per chromosome (Back, 1993).
    # A fixed 3% flipped ~60 genes on Pasig and ~800 on Quezon City, which
    # destroys a near-optimal solution rather than refining it.
    mr = (1.0 / n) if mutation_rate is None else max(1.0 / n, mutation_rate)
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
            # Always repair: DROP restores feasibility, ADD spends leftover budget.
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



# Funded projects of the most recent run for each dataset+algorithm, kept so
# the export can list every funded project for review. Holds references to the
# existing project records rather than copies.
FUNDED_LATEST = {}   # "ds|algo" -> {"budget":…, "items":[…], "timestamp":…}


def record_funded(ds, algo, items, selected_idx, budget):
    FUNDED_LATEST[f"{ds}|{algo}"] = {
        "ds": ds, "algo": algo, "budget": budget,
        "items": [items[i] for i in selected_idx if 0 <= i < len(items)],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


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


def build_workbook(include_combined=True):
    """Build the experiment workbook and return it as an in-memory buffer.

    include_combined adds a 'Funded - All' sheet that repeats every funded
    row with a Dataset column, for cross-city PivotTables. It roughly
    doubles the file size and build time, so it can be switched off.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

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
                    # Wrapped in IFERROR so a sheet exported before both
                    # datasets have been run shows blanks rather than
                    # #DIV/0! across the empty columns.
                    value=f'=IFERROR({fn}({letter}{first_data_row}:{letter}{last_data_row}),"")',
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

    # ── Funded projects, one sheet per city ────────────────────────────────
    # Split by dataset so each city can be analysed on its own, rather than
    # scrolling one 55,000-row table. Each sheet still carries an Algorithm
    # column, because both variations are listed together for comparison, and
    # an autofilter so a single algorithm or sector can be isolated.
    if FUNDED_LATEST:
        for ds, ds_label, sheet_name in (("pasig", DS_NAMES["pasig"], "Funded - Pasig"),
                                         ("qc",    DS_NAMES["qc"],    "Funded - Quezon City")):
            blocks = [(ALGO_SHEET_NAMES[a], FUNDED_LATEST[f"{ds}|{a}"])
                      for a in ALGO_SHEET_NAMES if f"{ds}|{a}" in FUNDED_LATEST]
            if not blocks:
                continue

            fp = wb.create_sheet(sheet_name)
            fp["A1"] = f"Funded Projects \u2014 {ds_label}"
            fp["A1"].font = title_font
            fp.merge_cells("A1:J1")
            fp.row_dimensions[1].height = 22

            budget = blocks[0][1]["budget"]
            fp["A2"] = f"Budget ceiling: PHP {budget:,.2f}"
            fp["A2"].font = Font(name=FONT, size=10, italic=True)
            fp.merge_cells("A2:J2")

            # The last two columns are left empty for the manual verification
            # phase: the reviewer records the judgement here, over the complete
            # allocation, with Excel's filtering and sorting available.
            fp_cols = [("Algorithm", 22), ("Account Code", 15),
                       ("Project / Program", 62), ("Implementing Office", 26),
                       ("Sector", 19), ("Cost (PhP)", 16), ("Benefit", 10),
                       ("Benefit per PhP 1M", 18),
                       ("Reviewer verdict", 18), ("Reviewer note", 46)]
            hdr_row = 4
            for col, (head, width) in enumerate(fp_cols, start=1):
                c = fp.cell(row=hdr_row, column=col, value=head)
                c.font = hdr_font
                c.fill = hdr_fill
                c.alignment = center
                c.border = border
                fp.column_dimensions[get_column_letter(col)].width = width
            fp.row_dimensions[hdr_row].height = 26

            # The same Font instance is reused across rows: a city allocation
            # can run to tens of thousands of rows.
            r = hdr_row + 1
            for label, rec in blocks:
                for it in rec["items"]:
                    cost = float(it["cost"])
                    ben = compute_benefit(it)
                    fp.cell(row=r, column=1, value=label).font = body_font
                    fp.cell(row=r, column=2, value=str(it.get("code", ""))).font = body_font
                    fp.cell(row=r, column=3, value=it.get("name", "")).font = body_font
                    fp.cell(row=r, column=4, value=it.get("pmo", "")).font = body_font
                    fp.cell(row=r, column=5, value=it.get("sector", "")).font = body_font
                    c6 = fp.cell(row=r, column=6, value=round(cost, 2))
                    c6.font = body_font; c6.number_format = "#,##0.00"
                    c7 = fp.cell(row=r, column=7, value=round(ben, 3))
                    c7.font = body_font; c7.number_format = "0.000"
                    c8 = fp.cell(row=r, column=8,
                                 value=round(ben / (cost / 1e6), 4) if cost else None)
                    c8.font = body_font; c8.number_format = "0.0000"
                    r += 1

            last = r - 1
            if last > hdr_row:
                fp.auto_filter.ref = f"A{hdr_row}:J{last}"
                # Dropdown on the verdict column so the entries stay consistent
                # and can be filtered or counted afterwards.
                dv = DataValidation(
                    type="list",
                    formula1='"aligned,not aligned,flagged"',
                    allow_blank=True, showDropDown=False,
                )
                dv.prompt = ("Does this allocation align with the local government's "
                             "development priorities?")
                dv.promptTitle = "Manual verification"
                fp.add_data_validation(dv)
                dv.add(f"I{hdr_row + 1}:I{last}")
            fp.freeze_panes = f"A{hdr_row + 1}"

            # Per-algorithm totals, as live formulas over this sheet only.
            tot = last + 2
            fp.cell(row=tot, column=1, value="Totals").font = mean_font
            for label, rec in blocks:
                tot += 1
                fp.cell(row=tot, column=1, value=label).font = body_font
                fp.cell(row=tot, column=2, value=f"{len(rec['items']):,} projects").font = body_font
                cf = fp.cell(row=tot, column=6,
                             value=f'=SUMIF(A{hdr_row + 1}:A{last},A{tot},F{hdr_row + 1}:F{last})')
                cf.font = mean_font; cf.number_format = "#,##0.00"
                bf = fp.cell(row=tot, column=7,
                             value=f'=SUMIF(A{hdr_row + 1}:A{last},A{tot},G{hdr_row + 1}:G{last})')
                bf.font = mean_font; bf.number_format = "0.000"

    # ── Funded projects, both cities combined (optional) ───────────────────
    # The per-city sheets above are for reading; this one is for pivoting. It
    # repeats the same rows with a Dataset column so a PivotTable can compare
    # the two cities in a single field - which the split sheets cannot do.
    if FUNDED_LATEST and include_combined:
        allfp = wb.create_sheet("Funded - All")
        all_cols = [("Dataset", 20), ("Algorithm", 22), ("Account Code", 15),
                    ("Project / Program", 62), ("Implementing Office", 26),
                    ("Sector", 19), ("Cost (PhP)", 16), ("Benefit", 10),
                    ("Benefit per PhP 1M", 18)]
        for col, (head, width) in enumerate(all_cols, start=1):
            c = allfp.cell(row=1, column=col, value=head)
            c.font = hdr_font
            c.fill = hdr_fill
            c.alignment = center
            c.border = border
            allfp.column_dimensions[get_column_letter(col)].width = width
        allfp.row_dimensions[1].height = 26

        # Header on row 1 with no title above it, so the range can be selected
        # directly as a PivotTable source.
        r = 2
        for algo in ALGO_SHEET_NAMES:
            for ds in ("pasig", "qc"):
                rec = FUNDED_LATEST.get(f"{ds}|{algo}")
                if not rec:
                    continue
                label, dsname = ALGO_SHEET_NAMES[algo], DS_NAMES[ds]
                for it in rec["items"]:
                    cost = float(it["cost"])
                    ben = compute_benefit(it)
                    allfp.cell(row=r, column=1, value=dsname).font = body_font
                    allfp.cell(row=r, column=2, value=label).font = body_font
                    allfp.cell(row=r, column=3, value=str(it.get("code", ""))).font = body_font
                    allfp.cell(row=r, column=4, value=it.get("name", "")).font = body_font
                    allfp.cell(row=r, column=5, value=it.get("pmo", "")).font = body_font
                    allfp.cell(row=r, column=6, value=it.get("sector", "")).font = body_font
                    c7 = allfp.cell(row=r, column=7, value=round(cost, 2))
                    c7.font = body_font; c7.number_format = "#,##0.00"
                    c8 = allfp.cell(row=r, column=8, value=round(ben, 3))
                    c8.font = body_font; c8.number_format = "0.000"
                    c9 = allfp.cell(row=r, column=9,
                                    value=round(ben / (cost / 1e6), 4) if cost else None)
                    c9.font = body_font; c9.number_format = "0.0000"
                    r += 1

        last = r - 1
        if last >= 2:
            allfp.auto_filter.ref = f"A1:I{last}"
        allfp.freeze_panes = "A2"

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


# Display label and theoretical complexity per algorithm. Defined at module
# level so both the plain and the streaming run endpoints share one source of
# truth for these strings.
ALGO_META = {
    "dp":  {"label": "Knapsack (DP)",      "time": "O(n·W)",           "space": "O(n·W)"},
    # Best-first search stores its open subproblems in a priority queue, which
    # can grow exponentially in the worst case - hence O(2ⁿ) space, not O(n).
    # (O(n) would be correct for a depth-first traversal, which keeps only the
    # current path.) These strings match the amended Figure 6.
    "bnb": {"label": "Knapsack + B&B",     "time": "O(2ⁿ) worst case", "space": "O(2ⁿ) worst case"},
    # Figure 6 of the manuscript states the hybrid's complexity as
    # Time O(g·p·n + 2ⁿ) bounded by GA pruning, and Space O(p·n + n) for the
    # priority queue plus the population. Kept verbatim so a live demo matches
    # the architecture diagram.
    # Time O(g·p·n + 2ⁿ): the GA phase plus the branch-and-bound phase. The GA
    # term is polynomial, so the worst case remains exponential - the GA
    # improves practical performance, not the asymptotic bound, which is why
    # "bounded by GA pruning" was dropped.
    # Space O(p·n + 2ⁿ): the GA population plus the priority queue.
    "ga":  {"label": "Knapsack + B&B + GA","time": "O(g·p·n + 2ⁿ) worst case","space": "O(p·n + 2ⁿ) worst case"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(405)
@app.errorhandler(415)
@app.errorhandler(500)
def json_error(e):
    from flask import jsonify
    return jsonify({"error": str(e)}), e.code




@app.route("/")
def index():
    return render_template('index.html')


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
    mut_rate  = body.get("mut_rate")
    mut_rate  = None if mut_rate in (None, "") else float(mut_rate)

    all_data = DATASETS.get(ds, DATASETS["pasig"])["data"]
    items    = [all_data[i] for i in sel_idx if i < len(all_data)]

    if not items:
        return jsonify({"error": "No valid projects selected."}), 400

    if not isinstance(algos, list) or not algos:
        return jsonify({"error": "No algorithms selected."}), 400


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
            entry = _build_result_entry(algo, res, elapsed_ms)
            # Attach the source project codes for the chosen combination, so
            # every allocation can be traced back to the LGU procurement plan.
            entry["project_ids"] = [
                items[i].get("code", "") for i in res["selected"]
            ]
            record_trial(ds, algo, entry, budget, len(items), ga_params)
            record_funded(ds, algo, items, res["selected"], budget)
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


def _build_result_entry(algo, res, elapsed_ms, _meta=None):
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


@app.route("/api/run/stream", methods=["POST"])
def api_run_stream():
    """Run the trials and stream live progress as Server-Sent Events.

    Why trials are INTERLEAVED rather than run in parallel
    ------------------------------------------------------
    The obvious way to "run both at the same time" is a thread per algorithm.
    That would be wrong here. CPython holds a global interpreter lock, so two
    CPU-bound knapsack solvers in threads do not run simultaneously - they
    take turns, competing for the same core, and each one's measured runtime
    is inflated by however long the other held the lock. Runtime and execution
    time are precisely what the study's paired t-test compares, so contaminated
    timings would invalidate the results.

    Instead the loop interleaves by trial: trial 1 of every algorithm, then
    trial 2, and so on. Only one algorithm computes at any instant, so each
    measurement is as clean as the sequential path, while every algorithm's
    progress advances together in real time. Each algorithm's result is
    emitted the moment it finishes its own final trial, so a faster algorithm
    reports without waiting for a slower one.
    """
    body = request.get_json(force=True, silent=True) or {}
    ds       = body.get("ds", "pasig")
    data     = PASIG_DATA if ds == "pasig" else QC_DATA
    budget   = float(body.get("budget", 0) or 0)
    sel      = body.get("selected") or []
    algos    = body.get("algos") or ["dp", "bnb", "ga"]
    pop_size = int(body.get("pop_size", 60))
    gens     = int(body.get("gens", 100))
    mut_rate = body.get("mut_rate")
    mut_rate = None if mut_rate in (None, "") else float(mut_rate)
    trials   = max(1, min(TRIALS_PER_COMBO, int(body.get("trials", 1))))

    items = [data[i] for i in sel if 0 <= i < len(data)]
    algos = [a for a in algos if a in ALGO_META]

    if not items or not algos:
        return jsonify({"error": "Select at least one project and one algorithm."}), 400

    ga_params = {"pop_size": pop_size, "gens": gens, "mut_rate": mut_rate}

    def sse(payload):
        return f"data: {json.dumps(payload)}\n\n"

    def generate():
        yield sse({"type": "start", "algos": algos, "trials": trials,
                   "labels": {a: ALGO_META[a]["label"] for a in algos},
                   "n_items": len(items)})

        # Running tallies so the UI can show a live mean per algorithm.
        acc  = {a: {"runtime": [], "exec": [], "pruning": []} for a in algos}
        last = {}

        for t in range(trials):
            for algo in algos:
                t0 = time.perf_counter()
                if algo == "dp":
                    res = knapsack_dp(items, budget)
                elif algo == "bnb":
                    res = knapsack_bnb(items, budget)
                else:
                    res = knapsack_bnb_ga(items, budget, pop_size=pop_size,
                                          generations=gens, mutation_rate=mut_rate)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                entry = _build_result_entry(algo, res, elapsed_ms)
                entry["project_ids"] = [items[i].get("code", "") for i in res["selected"]]
                record_trial(ds, algo, entry, budget, len(items), ga_params)
                record_funded(ds, algo, items, res["selected"], budget)
                last[algo] = entry

                acc[algo]["runtime"].append(entry["runtime_ms"])
                acc[algo]["exec"].append(entry["exec_ms"])
                if entry.get("pruning_rate") is not None:
                    acc[algo]["pruning"].append(entry["pruning_rate"])

                mean = lambda v: (sum(v) / len(v)) if v else None
                yield sse({
                    "type": "progress", "algo": algo,
                    "trial": t + 1, "trials": trials,
                    "runtime_ms": entry["runtime_ms"],
                    "exec_ms": entry["exec_ms"],
                    "pruning_rate": entry.get("pruning_rate"),
                    "mean_runtime": mean(acc[algo]["runtime"]),
                    "mean_exec": mean(acc[algo]["exec"]),
                    "mean_pruning": mean(acc[algo]["pruning"]),
                    "total_benefit": entry["total_benefit"],
                })

                # This algorithm has finished all of its trials - report now.
                if t + 1 == trials:
                    yield sse({"type": "result", "algo": algo, "result": entry})

        yield sse({
            "type": "done",
            "ds": ds, "budget": budget, "trials": trials,
            "selected_items": items,
            "results": [last[a] for a in algos if a in last],
            "history": history_summary(),
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx-style proxies buffering the stream into one chunk.
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Post-allocation verification and comparative review
#
# Implements step 7 of the Data Generation/Gathering Procedure. The manuscript
# specifies a MANUAL verification phase, so this module does not replace that
# judgement - it supplies the material the reviewer works from:
#
#   1. Automated integrity checks over EVERY selected project (traceability to
#      the source plan, budget compliance, arithmetic consistency). These are
#      mechanical facts a script can establish.
#   2. Sector alignment: each sector's share of the published plan against its
#      share of the funded set, which is the evidence for whether the
#      allocation reflects the LGU's own priorities.
#   3. Comparative review of the two cities, for the scaling question.
#   4. The manual judgement itself is recorded in the exported workbook: the
#      funded-project sheets carry Reviewer verdict and Reviewer note columns,
#      so the reviewer works over the complete allocation in Excel with
#      filtering and sorting rather than over a sample in the browser.
#
# The distinction matters: a script can verify that a project exists in the
# source plan; only a reviewer can judge whether the allocation "aligns with
# realistic public policy demands."
# ─────────────────────────────────────────────────────────────────────────────

def _selection_for(ds, budget, selected_idx, algo_selected):
    data = PASIG_DATA if ds == "pasig" else QC_DATA
    items = [data[i] for i in selected_idx if 0 <= i < len(data)]
    return items, [items[i] for i in algo_selected if 0 <= i < len(items)]


def verification_checks(ds, candidates, funded, budget):
    """Mechanical checks over every funded project. Returns pass/fail rows."""
    data = PASIG_DATA if ds == "pasig" else QC_DATA
    source_codes = collections_Counter(str(p.get("code", "")).strip() for p in data)

    total_cost = sum(float(p["cost"]) for p in funded)
    total_benefit = sum(compute_benefit(p) for p in funded)

    traceable = sum(1 for p in funded
                    if source_codes.get(str(p.get("code", "")).strip(), 0) > 0)
    positive_cost = sum(1 for p in funded if float(p["cost"]) > 0)
    scored = sum(1 for p in funded if 0 <= compute_benefit(p) <= 10)

    n = len(funded)
    rows = [
        {"check": "Every funded project appears in the source Annual Procurement Plan",
         "result": f"{traceable:,} of {n:,}", "pass": traceable == n},
        {"check": "Every funded project has a positive estimated budget",
         "result": f"{positive_cost:,} of {n:,}", "pass": positive_cost == n},
        {"check": "Every social benefit value falls within the 0-10 scale",
         "result": f"{scored:,} of {n:,}", "pass": scored == n},
        {"check": "Total cost of the funded set is within the budget ceiling",
         "result": f"PHP {total_cost:,.2f} of PHP {budget:,.2f}",
         "pass": total_cost <= budget + 1e-6},
        {"check": "No project is funded more than once",
         "result": f"{len({id(p) for p in funded}):,} distinct entries",
         "pass": len({id(p) for p in funded}) == n},
    ]
    return rows, total_cost, total_benefit


def sector_alignment(ds, candidates, funded):
    """Share of each sector in the candidate plan vs in the funded set."""
    allc = collections_Counter(p["sector"] for p in candidates)
    selc = collections_Counter(p["sector"] for p in funded)
    out = []
    for sector, in_plan in allc.most_common():
        got = selc.get(sector, 0)
        out.append({
            "sector":        sector,
            "in_plan":       in_plan,
            "funded":        got,
            "funded_pct":    (got / in_plan * 100) if in_plan else 0.0,
            "share_plan":    (in_plan / len(candidates) * 100) if candidates else 0.0,
            "share_funded":  (got / len(funded) * 100) if funded else 0.0,
        })
    return out


@app.route("/api/review", methods=["POST"])
def api_review():
    """Verification material for one allocation."""
    body = request.get_json(force=True, silent=True) or {}
    ds       = body.get("ds", "pasig")
    budget   = float(body.get("budget", 0) or 0)
    sel      = body.get("selected") or []          # candidate pool indices
    funded_i = body.get("funded") or []            # indices into the candidate pool

    candidates, funded = _selection_for(ds, budget, sel, funded_i)
    if not candidates:
        return jsonify({"error": "No candidate projects to review."}), 400

    checks, total_cost, total_benefit = verification_checks(ds, candidates, funded, budget)
    sectors = sector_alignment(ds, candidates, funded)

    return jsonify({
        "ds": ds, "budget": budget,
        "candidates": len(candidates), "funded": len(funded),
        "total_cost": total_cost, "total_benefit": total_benefit,
        "benefit_per_million": (total_benefit / (total_cost / 1e6)) if total_cost else 0.0,
        "checks": checks,
        "sectors": sectors,
    })


@app.route("/api/funded")
def api_funded():
    """Browse the funded projects of the most recent run, page by page.

    Paged on the server because a Quezon City allocation can exceed 25,000
    projects, which is far more than is sensible to ship to the browser in
    one response.
    """
    ds     = request.args.get("ds", "pasig")
    algo   = request.args.get("algo", "ga")
    q      = request.args.get("q", "").strip().lower()
    sector = request.args.get("sector", "All")
    sort   = request.args.get("sort", "benefit")
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    per = 50

    available = [
        {"ds": d, "algo": a, "ds_name": DS_NAMES[d], "algo_name": ALGO_SHEET_NAMES[a],
         "count": len(FUNDED_LATEST[f"{d}|{a}"]["items"])}
        for a in ALGO_SHEET_NAMES for d in ("pasig", "qc")
        if f"{d}|{a}" in FUNDED_LATEST
    ]

    rec = FUNDED_LATEST.get(f"{ds}|{algo}")
    if not rec:
        return jsonify({"rows": [], "total": 0, "page": 1, "pages": 0,
                        "available": available, "sectors": [],
                        "sum_cost": 0, "sum_benefit": 0, "budget": 0})

    items = rec["items"]
    sectors = sorted({p.get("sector", "") for p in items})

    rows = items
    if sector and sector != "All":
        rows = [p for p in rows if p.get("sector") == sector]
    if q:
        rows = [p for p in rows
                if q in str(p.get("name", "")).lower()
                or q in str(p.get("pmo", "")).lower()
                or q in str(p.get("code", "")).lower()]

    keyed = [(p, compute_benefit(p), float(p["cost"])) for p in rows]
    if sort == "cost":
        keyed.sort(key=lambda t: t[2], reverse=True)
    elif sort == "ratio":
        keyed.sort(key=lambda t: (t[1] / (t[2] / 1e6)) if t[2] else 0, reverse=True)
    elif sort == "name":
        keyed.sort(key=lambda t: str(t[0].get("name", "")))
    else:
        keyed.sort(key=lambda t: t[1], reverse=True)

    total = len(keyed)
    pages = max(1, -(-total // per))
    page = min(page, pages)
    window = keyed[(page - 1) * per: page * per]

    return jsonify({
        "available": available, "sectors": sectors,
        "ds": ds, "algo": algo, "budget": rec["budget"],
        "total": total, "page": page, "pages": pages,
        "sum_cost": sum(t[2] for t in keyed),
        "sum_benefit": sum(t[1] for t in keyed),
        "rows": [{
            "code": str(p.get("code", "")), "name": p.get("name", ""),
            "pmo": p.get("pmo", ""), "sector": p.get("sector", ""),
            "cost": cost, "benefit": round(ben, 3),
            "ratio": round(ben / (cost / 1e6), 3) if cost else None,
        } for p, ben, cost in window],
    })


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
    combined = request.args.get("combined", "1").lower() not in ("0", "false", "no")
    try:
        buf = build_workbook(include_combined=combined)
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
    print("LGU Budget Optimizer — Thesis Group 2 BSCS 4-1N")
    print(f"  Pasig City:  {len(PASIG_DATA):,} projects")
    print(f"  Quezon City: {len(QC_DATA):,} projects")
    print("=" * 60)
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, port=5000, threaded=True)
