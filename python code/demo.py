"""
Planar linkage coupler-curve synthesis for Inter-IIT PS01:
four-bar, six-bar and eight-bar, one engine, same scoring for all three.

Purpose
-------
Search a WIDE, generic parameter space for a one-DOF linkage, driven by a
fully rotating crank, whose traced point follows the target foot-tip path -
without seeding, biasing, or hard-coding ANY published linkage's dimensions
(Jansen, Klann, etc.). Every run starts from a random point in the box.

How the chains are built (why six/eight-bar are "the same process")
-------------------------------------------------------------------
Base       : Grashof crank-rocker four-bar (ground, crank, coupler, rocker).
Each dyad  : two extra links (an RR Assur group) joined at a new pin C.
             Their far ends pin onto two DIFFERENT existing bodies (or
             ground) at free points, each given as (distance, angle) in
             that body's own frame. Adding one dyad adds 2 links and 3
             joints, so DOF stays 1:   3*(n-1) - 2*j = 1  for every chain.
Traced pt  : a point on one link of the LAST dyad (four-bar: on the coupler),
             given as (distance, angle) in that link's frame.

  4-bar = base only            (6 parameters - identical to the old script)
  6-bar = base + 1 dyad        (13 parameters, 10 topologies: Watt + Stephenson)
  8-bar = base + 2 dyads       (20 parameters, 90 topologies, a random subset is searched)

Rule that keeps chains non-degenerate: every dyad must attach at least one end
to a body created by the previous stage, so the traced point really depends
on every loop (no decorative loops), and the two ends are never on the same
body (that would just freeze into a rigid triangle).

Scoring (unchanged in spirit from the four-bar script)
-----------------------------------------------------
Uniform scale to the target's lift, centre in x, best cyclic phase, best
mirror/direction variant. NO independent x/y stretching anywhere.

Fix vs. the previous four-bar script: the phase scan used to roll x only and
compare against an un-rolled y, which paired x(t+s) with y(t) and silently
scored a curve the linkage does not trace. Both coordinates are now rolled
together, the scan is exact over every shift (FFT cross-correlation), and
every reported RMSE is re-computed directly from the returned (x, y) and
asserted to match.

Needs target_foot_tip_path.csv (header row; columns: index, x, y) next to
this file.
"""

import csv
import itertools
import multiprocessing as mp
import random
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 0. Settings
# --------------------------------------------------------------------------
TARGET_CSV = "target_foot_tip_path.csv"

PRESET = "report"        # "smoke" = ~10 s, just to check nothing is broken
                         # "report" = the numbers you put in the write-up (roughly 10-15 min on one core)
BARS_TO_RUN = (4, 6, 8)
WORKERS = -1             # -1 = all cores, 1 = serial (use 1 if multiprocessing
                         # misbehaves in your IDE / notebook)
RUN_SELFTESTS = True
SAMPLE_SEED = 7          # which random subset of eight-bar topologies is searched

# runs = independent restarts per topology; popsize is multiplied by the
# number of parameters inside scipy (4-bar: 6, 6-bar: 13, 8-bar: 20).
SETTINGS = {
    "smoke": {
        4: dict(runs=2,  maxiter=10,  popsize=8,  max_topos=None),
        6: dict(runs=1,  maxiter=15,  popsize=6,  max_topos=2),
        8: dict(runs=1,  maxiter=15,  popsize=5,  max_topos=2),
    },
    "report": {
        4: dict(runs=10, maxiter=60,  popsize=15, max_topos=None),
        6: dict(runs=4,  maxiter=200, popsize=12, max_topos=None),
        8: dict(runs=3,  maxiter=300, popsize=10, max_topos=12),
    },
}

TWO_PI = 2 * np.pi
PENALTY = 1e4            # objective value for anything that cannot be built
MARGIN = 2e-3            # circles must intersect with this relative margin,
                         # so a pin never sits on a dead-centre where the
                         # assembly branch would be ambiguous

# --------------------------------------------------------------------------
# 1. Target
# --------------------------------------------------------------------------
data = np.loadtxt(TARGET_CSV, delimiter=",", skiprows=1)
XT, YT = data[:, 1], data[:, 2]
YT = YT - YT.min()
XT_centered = XT - XT.mean()
N_PTS = len(XT)
LIFT = YT.max()          # was hard-coded 200; identical if your lift is 200 mm

THETA = np.linspace(0, TWO_PI, N_PTS, endpoint=False)

_FT_X = np.fft.rfft(XT_centered)
_FT_Y = np.fft.rfft(YT)
_T2 = float(np.sum(XT_centered ** 2 + YT ** 2))

# --------------------------------------------------------------------------
# 2. Topologies
# --------------------------------------------------------------------------
BASE_BODIES = ["G", "crank", "coupler", "rocker"]
_ADJACENT = {frozenset(p) for p in [("G", "crank"), ("crank", "coupler"),
                                    ("coupler", "rocker"), ("rocker", "G")]}


@dataclass(frozen=True)
class Topology:
    n_dyads: int
    dyads: tuple        # ((bodyP, bodyQ), ...) - where each dyad's ends attach
    tracked: str        # body carrying the traced point
    name: str

    @property
    def bars(self):
        return 4 + 2 * self.n_dyads

    @property
    def n_joints(self):
        return 4 + 3 * self.n_dyads

    @property
    def dof(self):      # Grubler: 3(n-1) - 2j
        return 3 * (self.bars - 1) - 2 * self.n_joints


def _chains(n_dyads):
    """All valid dyad sequences (see module docstring for the two rules)."""
    def rec(chain, bodies, newest):
        if len(chain) == n_dyads:
            yield tuple(chain)
            return
        k = len(chain) + 1
        for pair in itertools.combinations(bodies, 2):
            if not (set(pair) & newest):
                continue
            yield from rec(chain + [pair],
                           bodies + [f"D{k}a", f"D{k}b"],
                           {f"D{k}a", f"D{k}b"})
    yield from rec([], list(BASE_BODIES), {"coupler", "rocker"})


def enumerate_topologies(n_dyads):
    if n_dyads == 0:
        return [Topology(0, (), "coupler", "4-bar")]
    out = []
    for chain in _chains(n_dyads):
        for side in "ab":
            tracked = f"D{n_dyads}{side}"
            if n_dyads == 1:
                kind = "Watt" if frozenset(chain[0]) in _ADJACENT else "Stephenson"
                name = f"6-bar {kind}: dyad({chain[0][0]},{chain[0][1]}) trace {tracked}"
            else:
                name = ("8-bar: " + " ".join(
                    f"dyad{k}({p},{q})" for k, (p, q) in enumerate(chain, 1))
                    + f" trace {tracked}")
            out.append(Topology(n_dyads, chain, tracked, name))
    return out


def bounds_for(topo):
    b = [(30, 400),      # ground link
         (10, 150),      # crank (short-ish so it can be the shortest link)
         (30, 500),      # coupler
         (30, 500)]      # rocker
    for _ in topo.dyads:
        b += [(0, 500), (0, TWO_PI),    # end P: distance, angle in its body frame
              (0, 500), (0, TWO_PI),    # end Q
              (30, 600), (30, 600),     # link lengths P->C, Q->C
              (0, 1)]                   # assembly branch (<0.5 -> +1, else -1)
    b += [(10, 600), (0, TWO_PI)]       # traced point: distance, angle
    return b


# --------------------------------------------------------------------------
# 3. Kinematics (vector loops / circle intersections, vectorised over the
#    whole crank revolution)
# --------------------------------------------------------------------------
def circle_intersection(c1, r1, c2, r2, branch):
    """Intersection of circle(c1, r1) and circle(c2, r2); branch = +1/-1 picks
    the assembly mode. Returns (points, fraction_of_samples_that_cannot_assemble)."""
    dxy = c2 - c1
    d = np.hypot(dxy[0], dxy[1])
    eps = MARGIN * (r1 + r2)
    bad = (d < eps) | (d < abs(r1 - r2) + eps) | (d > r1 + r2 - eps)
    ds = np.maximum(d, 1e-9)
    a = (r1 ** 2 - r2 ** 2 + ds ** 2) / (2 * ds)
    h = np.sqrt(np.maximum(r1 ** 2 - a ** 2, 0.0))
    mid = c1 + (a / ds) * dxy
    perp = np.array([-dxy[1], dxy[0]]) / ds
    return mid + branch * h * perp, float(bad.mean())


def _point(body, dist, ang):
    """Point at (dist, ang) in a body's frame; body = (origin (2,N), heading (N,))."""
    origin, heading = body
    return origin + dist * np.array([np.cos(heading + ang), np.sin(heading + ang)])


def solve(params, topo):
    """Returns (traced point (2,N), infeasibility score, joint checks).
    infeasibility = 0 means every pin assembles at every crank angle;
    otherwise it is the summed fraction of failing samples (a graded
    penalty gives the optimiser something to climb toward feasibility)."""
    ground, crank, coupler, rocker = params[:4]
    zeros = np.zeros(N_PTS)
    O2 = np.zeros((2, N_PTS))
    O4 = np.vstack([np.full(N_PTS, ground), zeros])
    A = np.vstack([crank * np.cos(THETA), crank * np.sin(THETA)])

    B, infeasible = circle_intersection(A, coupler, O4, rocker, +1.0)
    bodies = {
        "G": (O2, zeros),
        "crank": (O2, THETA),
        "coupler": (A, np.arctan2(B[1] - A[1], B[0] - A[0])),
        "rocker": (O4, np.arctan2(B[1] - O4[1], B[0] - O4[0])),
    }
    checks = [(B, A, coupler, O4, rocker)]

    i = 4
    for k, (bp, bq) in enumerate(topo.dyads, start=1):
        dP, aP, dQ, aQ, L1, L2, br = params[i:i + 7]
        i += 7
        P = _point(bodies[bp], dP, aP)
        Q = _point(bodies[bq], dQ, aQ)
        C, bad = circle_intersection(P, L1, Q, L2, 1.0 if br < 0.5 else -1.0)
        infeasible += bad
        bodies[f"D{k}a"] = (P, np.arctan2(C[1] - P[1], C[0] - P[0]))
        bodies[f"D{k}b"] = (Q, np.arctan2(C[1] - Q[1], C[0] - Q[0]))
        checks.append((C, P, L1, Q, L2))

    tip = _point(bodies[topo.tracked], params[i], params[i + 1])
    return tip, infeasible, checks


def grashof_violation(g, c, cp, r):
    """0 if the crank is the shortest link and s + l <= p + q (so it can turn
    a full revolution); otherwise a positive, graded amount."""
    lens = sorted([g, c, cp, r])
    total = sum(lens)
    v = 0.0
    if c > lens[0]:
        v += (c - lens[0]) / total
    v += max(0.0, lens[0] + lens[3] - lens[1] - lens[2]) / total
    return v


# --------------------------------------------------------------------------
# 4. Scoring
# --------------------------------------------------------------------------
def _variants(x, y):
    """All 8 mount/drive choices of the SAME physical mechanism: mirror in x,
    mirror in y (which way is 'up'), and crank direction (reverses traversal).
    x is already centred; each variant is re-based so y.min() == 0."""
    xs, ys = [], []
    for sx in (1, -1):
        for sy in (1, -1):
            xv, yv = sx * x, sy * y
            yv = yv - yv.min()
            xs.append(xv); ys.append(yv)                 # crank one way
            xs.append(xv[::-1]); ys.append(yv[::-1])     # crank the other way
    return np.array(xs), np.array(ys)


def best_fit_curve(x, y):
    """(rmse, scale, x, y): x, y are scaled to the target's lift, centred,
    phase-aligned and in the best variant - i.e. exactly the curve that the
    RMSE was computed on. None if the curve is degenerate."""
    span = y.max() - y.min()
    if not np.isfinite(span) or span < 1e-6:
        return None
    scale = LIFT / span
    x = (x - x.mean()) * scale
    y = (y - y.min()) * scale
    Xv, Yv = _variants(x, y)

    # cross-correlation of (x,y) with the target for EVERY cyclic shift at once.
    # c[s] = sum_i roll(x, s)[i] * XT[i]  (+ same for y) - x and y roll together.
    cc = np.fft.irfft(np.conj(np.fft.rfft(Xv, axis=1)) * _FT_X
                      + np.conj(np.fft.rfft(Yv, axis=1)) * _FT_Y,
                      n=N_PTS, axis=1)
    e2 = (np.sum(Xv ** 2 + Yv ** 2, axis=1)[:, None] + _T2 - 2 * cc) / N_PTS
    v, s = np.unravel_index(np.argmin(e2), e2.shape)
    rmse = float(np.sqrt(max(e2[v, s], 0.0)))
    return rmse, scale, np.roll(Xv[v], s), np.roll(Yv[v], s)


def direct_rmse(x, y):
    """Brute-force RMSE of an already-aligned curve - used to audit best_fit_curve."""
    return float(np.sqrt(np.mean((x - XT_centered) ** 2 + (y - YT) ** 2)))


def objective(params, topo):
    viol = grashof_violation(*params[:4])
    if viol > 0:
        return PENALTY * (1 + viol)
    tip, infeasible, _ = solve(params, topo)
    if infeasible > 0 or not np.all(np.isfinite(tip)):
        return PENALTY * (1 + (infeasible if np.isfinite(infeasible) else 1.0))
    fit = best_fit_curve(tip[0], tip[1])
    return PENALTY * 2 if fit is None else fit[0]


# --------------------------------------------------------------------------
# 5. Self-tests (a few seconds; guards against the bug class that bit the
#    last version: a printed metric that is not the curve you actually plot)
# --------------------------------------------------------------------------
def run_selftests():
    rng = np.random.default_rng(0)

    # (a) Gruebler count is 1 for every chain type
    for nd in (0, 1, 2):
        assert all(t.dof == 1 for t in enumerate_topologies(nd)), "DOF != 1"

    # (b) FFT phase scan == brute force, and the returned (x, y) reproduce the RMSE
    t = THETA
    x = 260 * np.cos(t + 0.4) + 40 * np.cos(2 * t)
    y = 90 * np.sin(t) + 25 * np.sin(3 * t + 1.0)
    rmse, scale, xa, ya = best_fit_curve(x, y)
    xs, ys = (x - x.mean()) * scale, (y - y.min()) * scale
    Xv, Yv = _variants(xs, ys)
    brute = min(direct_rmse(np.roll(Xv[v], s), np.roll(Yv[v], s))
                for v in range(len(Xv)) for s in range(N_PTS))
    assert abs(rmse - brute) < 1e-6, f"phase scan {rmse} != brute force {brute}"
    assert abs(rmse - direct_rmse(xa, ya)) < 1e-6, "returned curve != scored curve"

    # (c) every pin really is the intersection of its two link circles
    for nd in (0, 1, 2):
        topo = enumerate_topologies(nd)[0]
        bnd = np.array(bounds_for(topo))
        for _ in range(60000):
            p = rng.uniform(bnd[:, 0], bnd[:, 1])
            if grashof_violation(*p[:4]) > 0:
                continue
            tip, infeasible, checks = solve(p, topo)
            if infeasible == 0:
                for C, c1, r1, c2, r2 in checks:
                    assert np.allclose(np.hypot(*(C - c1)), r1, atol=1e-6)
                    assert np.allclose(np.hypot(*(C - c2)), r2, atol=1e-6)
                break
        else:
            print(f"  selftest note: no random feasible {topo.bars}-bar found "
                  f"for the joint check (not an error)")
    print("self-tests passed\n")


# --------------------------------------------------------------------------
# 6. Search
# --------------------------------------------------------------------------
def optimise(topo, seed, cfg, pool_map):
    res = differential_evolution(
        objective, bounds_for(topo), args=(topo,), seed=seed,
        maxiter=cfg["maxiter"], popsize=cfg["popsize"], tol=1e-4,
        mutation=(0.4, 1.3), recombination=0.8, polish=True,
        workers=pool_map if pool_map else 1,
        updating="deferred" if pool_map else "immediate")
    tip, infeasible, _ = solve(res.x, topo)
    fit = None if (infeasible > 0 or grashof_violation(*res.x[:4]) > 0) \
        else best_fit_curve(tip[0], tip[1])
    if fit is None:
        return dict(rmse=np.inf, scale=np.nan, stride=np.nan, params=res.x, xy=None)
    rmse, scale, x, y = fit
    assert abs(rmse - direct_rmse(x, y)) < 1e-6, "reported RMSE != RMSE of returned curve"
    return dict(rmse=rmse, scale=scale, stride=float(x.max() - x.min()),
                params=res.x, xy=(x, y))


def main():
    if RUN_SELFTESTS:
        run_selftests()
    cfgs = SETTINGS[PRESET]
    pool = mp.Pool(None if WORKERS == -1 else WORKERS) if WORKERS != 1 else None
    pool_map = pool.map if pool else None

    print(f"preset '{PRESET}', {N_PTS} target points, lift {LIFT:.0f} mm, "
          f"workers={'all' if WORKERS == -1 else WORKERS}\n")
    t_start = time.time()
    results = []
    try:
        for bars in BARS_TO_RUN:
            nd = (bars - 4) // 2
            cfg = cfgs[bars]
            full = enumerate_topologies(nd)
            picked = list(enumerate(full))
            if cfg["max_topos"] and len(full) > cfg["max_topos"]:
                picked = sorted(random.Random(SAMPLE_SEED).sample(picked, cfg["max_topos"]))
            print(f"=== {bars}-bar: {len(full)} valid topologies, searching {len(picked)}, "
                  f"{cfg['runs']} runs each (maxiter {cfg['maxiter']}, popsize {cfg['popsize']}) ===")
            for topo_idx, topo in picked:
                for run in range(cfg["runs"]):
                    seed = 100 + run + 1000 * topo_idx + 100000 * nd
                    t0 = time.time()
                    r = optimise(topo, seed, cfg, pool_map)
                    r.update(bars=bars, topo=topo, run=run, seed=seed)
                    results.append(r)
                    print(f"  {topo.name:<58s} run {run:2d} seed {seed:6d}  "
                          f"RMSE {r['rmse']:7.2f} mm  stride {r['stride']:6.1f} mm  "
                          f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        if pool:
            pool.close(); pool.join()

    # ---- summary ---------------------------------------------------------
    print(f"\n{'bars':>4s} {'topos':>5s} {'runs':>5s} {'best':>8s} {'median':>8s} "
          f"{'worst':>8s}   best topology   (elapsed {time.time() - t_start:.0f} s)")
    best_by_bars = {}
    for bars in BARS_TO_RUN:
        rs = [r for r in results if r["bars"] == bars and np.isfinite(r["rmse"])]
        if not rs:
            print(f"{bars:>4d}  no feasible solution found - raise maxiter/popsize")
            continue
        vals = np.array([r["rmse"] for r in rs])
        best = min(rs, key=lambda r: r["rmse"])
        best_by_bars[bars] = best
        n_topo = len({r["topo"] for r in rs})
        print(f"{bars:>4d} {n_topo:>5d} {len(rs):>5d} {vals.min():8.2f} "
              f"{np.median(vals):8.2f} {vals.max():8.2f}   {best['topo'].name}")

    print("\nbest parameters per class (ground, crank, coupler, rocker, "
          "then per dyad: dP, aP, dQ, aQ, L1, L2, branch; then traced dist, angle):")
    for bars, b in best_by_bars.items():
        print(f"  {bars}-bar [{b['topo'].name}] RMSE {b['rmse']:.2f} mm")
        print("    " + ", ".join(f"{v:.3f}" for v in b["params"]))

    # ---- log every run ---------------------------------------------------
    with open("linkage_runs_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bars", "topology", "run", "seed", "rmse_mm", "stride_mm", "params"])
        for r in results:
            w.writerow([r["bars"], r["topo"].name, r["run"], r["seed"],
                        f"{r['rmse']:.3f}", f"{r['stride']:.1f}",
                        ";".join(f"{v:.4f}" for v in r["params"])])
    print("\nsaved linkage_runs_log.csv")

    # ---- plot best of each class vs target -------------------------------
    if best_by_bars:
        fig, axes = plt.subplots(1, len(best_by_bars), figsize=(6 * len(best_by_bars), 4.6),
                                 squeeze=False)
        for ax, (bars, b) in zip(axes[0], best_by_bars.items()):
            x, y = b["xy"]
            x = x - x.mean() + XT.mean()
            ax.plot(XT, YT, "k--", lw=2, label="target")
            ax.plot(x, y, "r-", lw=1.6, label=f"{bars}-bar (RMSE {b['rmse']:.1f} mm)")
            ax.set_aspect("equal"); ax.grid(alpha=0.3)
            ax.set_ylim(-15, LIFT * 1.4); ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"{bars}-bar, best of {sum(r['bars'] == bars for r in results)} runs",
                         fontsize=10)
            ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        fig.tight_layout()
        fig.savefig("linkage_best_by_class.png", dpi=150)
        print("saved linkage_best_by_class.png")


if __name__ == "__main__":
    main()