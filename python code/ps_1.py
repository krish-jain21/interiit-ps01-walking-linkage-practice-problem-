"""
Walking-linkage evolution: four-bar -> Klann six-bar -> Jansen eight-bar
Problem 01 (Inter-IIT Tech Meet). One input revolution each, all paths
scaled to 200 mm lift and compared against target_foot_tip_path.csv.

Pure Python (numpy/scipy/matplotlib). If you want the MATLAB version of the
Jansen stage instead, see the note at the bottom (matlab.engine hook).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# ---------------------------------------------------------------- target
data = np.loadtxt("target_foot_tip_path.csv", delimiter=",", skiprows=1)
XT, YT = data[:, 1], data[:, 2]
YT = YT - YT.min()
XTc = XT - XT.mean()

def score(px, py):
    """RMSE of candidate path vs target: uniform height scale to 200 mm,
    center x, scan all phase shifts of the closed curve."""
    py = py - py.min()
    sc = 200.0 / (py.max() - py.min())
    px, py = px * sc, py * sc
    px = px - px.mean()
    n = len(px)
    best = np.inf
    for sh in range(0, n, 2):
        r = np.roll(np.arange(n), sh)
        best = min(best, np.sqrt(np.mean((px[r] - XTc) ** 2 + (py - YT) ** 2)))
    return best, sc

def circ(p1, r1, p2, r2, side):
    """Point at r1 from p1 and r2 from p2 (vectorised); side=+1/-1 branch."""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    if p1.ndim == 1:
        p1 = np.repeat(p1[:, None], p2.shape[1], axis=1)
    dd = p2 - p1
    L = np.linalg.norm(dd, axis=0, keepdims=True)
    u = dd / L
    x = (L**2 + r1**2 - r2**2) / (2 * L)
    y = np.sqrt(np.maximum(r1**2 - x**2, 0))
    nv = np.vstack([-u[1], u[0]])
    return p1 + u * x + nv * (side * y)

TH = np.linspace(0, 2 * np.pi, 360, endpoint=False)

# ================================================== STAGE 1: FOUR-BAR (ours)
def fourbar(p):
    g, r1, r2, r3, bp, beta = p
    B = np.vstack([r1 * np.cos(TH), r1 * np.sin(TH)])
    d = np.hypot(B[0] - g, B[1])
    if np.any(d > r2 + r3) or np.any(d < abs(r2 - r3)) or np.any(d < 1e-9):
        return None
    a = (r2**2 - r3**2 + d**2) / (2 * d)
    h2 = r2**2 - a**2
    if np.any(h2 < 0):
        return None
    h = np.sqrt(h2)
    mx, my = B[0] + a * (g - B[0]) / d, B[1] + a * (-B[1]) / d
    cx, cy = mx + h * (-(-B[1])) / d, my + h * ((g - B[0])) / d
    psi = np.arctan2(cy - B[1], cx - B[0]) + beta
    return B[0] + bp * np.cos(psi), B[1] + bp * np.sin(psi)

def obj4(p):
    L = sorted(p[:4])
    pen = 0.0 if (L[0] + L[3] <= L[1] + L[2] and abs(L[0] - p[1]) < 1e-9) else 500.0
    c = fourbar(p)
    if c is None:
        return 1e4
    return min(score(*c)[0], score(c[0], -c[1])[0]) + pen

print("Stage 1/3: synthesising four-bar (crank-rocker, differential evolution)...")
bnds = [(50, 300), (10, 90), (50, 400), (50, 400), (50, 600), (0, 2 * np.pi)]
res = differential_evolution(obj4, bnds, seed=7, maxiter=30, popsize=12,
                             tol=1e-3, mutation=(0.4, 1.2), recombination=0.8)
p4 = res.x
c = fourbar(p4)
if score(c[0], -c[1])[0] < score(*c)[0]:
    x4, y4 = c[0], -c[1]
else:
    x4, y4 = c
rmse4, s4 = score(x4, y4)

# ================================================== STAGE 2: KLANN SIX-BAR
# Joe Klann, US patent 6,260,862 example leg (all lengths derived from the
# patent's fixed-point + snapshot coordinates):
#   crank 15->29 (2.976)   rod 21 triangle 29-27-35 (5.506 / 6.236 / 11.679)
#   rocker7 11->27 (3.390) rocker5 9->37 (7.391)
#   leg 31 triangle 35-37-33 (10.137 / 22.188 / 13.442), foot = 33
P15 = np.array([17.607, 11.807]); P9 = np.array([17.818, 16.076]); P11 = np.array([12.101, 10.186])
L_CR, L_R7, L_R5 = 2.976, 3.390, 7.391
L_2927, L_2735, L_2935 = 5.506, 6.236, 11.679
L_3537, L_3733, L_3533 = 10.137, 22.188, 13.442

A29 = P15[:, None] + L_CR * np.vstack([np.cos(TH), np.sin(TH)])
J27 = circ(A29, L_2927, P11[:, None], L_R7, -1)
J35 = circ(A29, L_2935, J27, L_2735, -1)
J37 = circ(J35, L_3537, P9[:, None], L_R5, +1)
J33 = circ(J35, L_3533, J37, L_3733, -1)          # foot
# sanity: patent "fully extended" snapshot is at crank angle 180 deg
i180 = 180
snap = np.linalg.norm(J27[:, i180]-[9.125, 11.807]) + np.linalg.norm(J35[:, i180]-[3.024, 13.099]) \
           + np.linalg.norm(J37[:, i180]-[11.119, 19.200]) + np.linalg.norm(J33[:, i180]-[0.0, 0.0])
assert snap < 0.05, "Klann branch check failed"
rmse6, s6 = score(J33[0], J33[1])

# ================================================== STAGE 3: JANSEN EIGHT-BAR
# Theo Jansen holy numbers; fixed pivots O=(0,0), B=(-38,-7.8); toe = G
a, l, m = 38.0, 7.8, 15.0
jb, jc, jd, je = 41.5, 39.3, 40.1, 55.8
jf, jg, jh, ji, jj, jk = 39.4, 36.7, 65.7, 49.0, 50.0, 61.9
Ja = np.vstack([m * np.cos(TH), m * np.sin(TH)]); Jb = np.array([-a, -l])
Jc = circ(Jb, jb, Ja, jj, +1)
Jd = circ(Jb, jc, Ja, jk, -1)
Je = circ(Jb, jd, Jc, je, +1)
Jf = circ(Jd, jg, Je, jf, +1)
Jg = circ(Jd, ji, Jf, jh, +1)                    # toe
rmse8, s8 = score(Jg[0], Jg[1])

# ---------------------------------------------------------------- report
def stride_of(px, py, sc):
    return (px.max() - px.min()) * sc

y4 = y4 - y4.min();  x4n, y4n = x4 * s4, y4 * s4
y6 = J33[1] - J33[1].min(); x6n, y6n = J33[0] * s6, y6 * s6
y8 = Jg[1] - Jg[1].min();  x8n, y8n = Jg[0] * s8, y8 * s8
print("\n================ WALKING-LINKAGE EVOLUTION ================")
print(f"Stage 1  four-bar (synthesised)   RMSE {rmse4:7.2f} mm   stride {stride_of(x4,y4,s4):6.1f} mm   scale {s4:.3f}")
print(f"Stage 2  Klann six-bar (patent)   RMSE {rmse6:7.2f} mm   stride {stride_of(J33[0],J33[1],s6):6.1f} mm   scale {s6:.3f}")
print(f"Stage 3  Jansen eight-bar (holy)  RMSE {rmse8:7.2f} mm   stride {stride_of(Jg[0],Jg[1],s8):6.1f} mm   scale {s8:.3f}")
print(f"Target                                                          stride {XT.max()-XT.min():6.1f} mm")
print("\nDOF check  4-bar: 3(4-1)-2*4 = 1 | 6-bar Klann: 3(6-1)-2*7 = 1 | 8-bar Jansen: 3(8-1)-2*10 = 1")

fig, axes = plt.subplots(3, 1, figsize=(11, 12))
for ax, (xn, yn, name, rmse) in zip(axes, [
        (x4n, y4n, "Stage 1 - four-bar (our synthesis)", rmse4),
        (x6n, y6n, "Stage 2 - Klann six-bar (US 6,260,862)", rmse6),
        (x8n, y8n, "Stage 3 - Jansen eight-bar (holy numbers)", rmse8)]):
    ax.plot(XT, YT, "k--", lw=2, label="target")
    ax.plot(xn - xn.mean() + XT.mean(), yn, "r-", lw=1.4,
            label=f"{name.split(' - ')[1]}  (RMSE {rmse:.1f} mm)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.legend(); ax.grid(alpha=.3); ax.set_aspect("equal")
    ax.set_title(name)
fig.tight_layout()
fig.savefig("evolution_4bar_6bar_8bar.png", dpi=150)
print("\nsaved evolution_4bar_6bar_8bar.png")

# ---- OPTIONAL: call the MATLAB Jansen script from Python -----------------
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.cd(r"<folder containing jansen_walking_sim.m>"); eng.jansen_walking_sim(nargout=0)