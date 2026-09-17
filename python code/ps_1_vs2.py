"""
Walking-linkage evolution - REAL-TIME MOTION (4-bar -> Klann 6-bar -> Jansen 8-bar)
Problem 01, Inter-IIT Tech Meet.

Usage:
    python walking_evolution_realtime.py          # live animation window
    python walking_evolution_realtime.py --save   # also export GIF

Needs: numpy, scipy, matplotlib  (and target_foot_tip_path.csv next to this file)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import differential_evolution

SAVE_GIF = "--save" in sys.argv

# ------------------------------------------------------------ target path
data = np.loadtxt("target_foot_tip_path.csv", delimiter=",", skiprows=1)
XT, YT = data[:, 1], data[:, 2]
YT -= YT.min()
XTc = XT - XT.mean()
TH = np.linspace(0, 2 * np.pi, 360, endpoint=False)

def circ(p1, r1, p2, r2, side):
    """Point at distance r1 from p1 AND r2 from p2 (vectorised). side=+1/-1 picks branch."""
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

def score(px, py):
    """RMSE vs target: uniform height scale to 200 mm, centre x, scan phase shifts."""
    py = py - py.min()
    sc = 200.0 / (py.max() - py.min())
    px, py = px * sc, py * sc
    px -= px.mean()
    n = len(px)
    best = np.inf
    for sh in range(0, n, 2):
        r = np.roll(np.arange(n), sh)
        best = min(best, np.sqrt(np.mean((px[r] - XTc) ** 2 + (py - YT) ** 2)))
    return best, sc

# ============================================================ STAGE 1: four-bar
def fourbar(p):
    g, r1, r2, r3, bp, beta = p
    Bx, By = r1 * np.cos(TH), r1 * np.sin(TH)
    d = np.hypot(g - Bx, -By)
    if np.any(d > r2 + r3) or np.any(d < abs(r2 - r3)) or np.any(d < 1e-9):
        return None
    aa = (r2**2 - r3**2 + d**2) / (2 * d)
    h2 = r2**2 - aa**2
    if np.any(h2 < 0):
        return None
    h = np.sqrt(h2)
    mx, my = Bx + aa * (g - Bx) / d, By + aa * (-By) / d
    Cx, Cy = mx + h * By / d, my + h * (g - Bx) / d
    psi = np.arctan2(Cy - By, Cx - Bx) + beta
    return Bx + bp * np.cos(psi), By + bp * np.sin(psi), Bx, By, Cx, Cy

def obj4(p):
    L = sorted(p[:4])
    pen = 0.0 if (L[0] + L[3] <= L[1] + L[2] and abs(L[0] - p[1]) < 1e-9) else 500.0
    c = fourbar(p)
    if c is None:
        return 1e4
    return min(score(c[0], c[1])[0], score(c[0], -c[1])[0]) + pen

print("Stage 1/3: synthesising four-bar (differential evolution, ~30 s) ...")
res = differential_evolution(obj4, [(50,300),(10,90),(50,400),(50,400),(50,600),(0,2*np.pi)],
                             seed=7, maxiter=30, popsize=12, tol=1e-3,
                             mutation=(0.4, 1.2), recombination=0.8)
c = fourbar(res.x)
if score(c[0], -c[1])[0] < score(c[0], c[1])[0]:
    x4, y4 = c[0], -c[1]
else:
    x4, y4 = c[0], c[1]
rmse4, s4 = score(x4, y4)
Ax4, Ay4, Cx4, Cy4, g4 = c[2], c[3], c[4], c[5], res.x[0]

# ============================================================ STAGE 2: Klann six-bar
P15 = np.array([17.607, 11.807]); P9 = np.array([17.818, 16.076]); P11 = np.array([12.101, 10.186])
A29 = P15[:, None] + 2.976 * np.vstack([np.cos(TH), np.sin(TH)])
J27 = circ(A29, 5.506, P11[:, None], 3.390, -1)
J35 = circ(A29, 11.679, J27, 6.236, -1)
J37 = circ(J35, 10.137, P9[:, None], 7.391, +1)
J33 = circ(J35, 13.442, J37, 22.188, -1)          # foot
snap = (np.linalg.norm(J27[:, 180] - [9.125, 11.807]) +
        np.linalg.norm(J35[:, 180] - [3.024, 13.099]) +
        np.linalg.norm(J37[:, 180] - [11.119, 19.200]) +
        np.linalg.norm(J33[:, 180] - [0, 0]))
assert snap < 0.05, "Klann assembly check failed (branch choice)"
rmse6, s6 = score(J33[0], J33[1])

# ============================================================ STAGE 3: Jansen eight-bar
Ja = 15 * np.vstack([np.cos(TH), np.sin(TH)]); Jb = np.array([-38.0, -7.8])
Jc = circ(Jb, 41.5, Ja, 50, +1)
Jd = circ(Jb, 39.3, Ja, 61.9, -1)
Je = circ(Jb, 40.1, Jc, 55.8, +1)
Jf = circ(Jd, 36.7, Je, 39.4, +1)
Jg = circ(Jd, 49.0, Jf, 65.7, +1)                 # toe
rmse8, s8 = score(Jg[0], Jg[1])

# ============================================================ common frame
def frame(px, py, sc):
    X = px * sc; Y = py * sc
    off = np.array([(XT.min() + XT.max()) / 2 - (X.min() + X.max()) / 2, -Y.min()])
    return X + off[0], Y + off[1], off

X4, Y4, off4 = frame(x4, y4, s4)
X6, Y6, off6 = frame(J33[0], J33[1], s6)
X8, Y8, off8 = frame(Jg[0], Jg[1], s8)

def pose(i):
    p4 = np.stack([[0,0],[Ax4[i],Ay4[i]],[Cx4[i],Cy4[i]],[g4,0],[x4[i],y4[i]]],
                  axis=1) * s4 + off4[:, None]
    p6 = np.stack([P15, A29[:,i], J27[:,i], J35[:,i], J37[:,i], J33[:,i], P9, P11],
                  axis=1) * s6
    p6[1] -= (J33[1] * s6).min(); p6[0] += off6[0]
    p8 = np.stack([[0,0], Ja[:,i], Jb, Jc[:,i], Jd[:,i], Je[:,i], Jf[:,i], Jg[:,i]],
                  axis=1) * s8 + off8[:, None]
    return p4, p6, p8

# ============================================================ summary
print("\n================ EVOLUTION SUMMARY ================")
print(f"Stage 1  four-bar (synthesised)  RMSE {rmse4:7.2f} mm   stride {X4.max()-X4.min():6.1f} mm")
print(f"Stage 2  Klann six-bar (patent)  RMSE {rmse6:7.2f} mm   stride {X6.max()-X6.min():6.1f} mm")
print(f"Stage 3  Jansen eight-bar        RMSE {rmse8:7.2f} mm   stride {X8.max()-X8.min():6.1f} mm")
print(f"Target                                                    {XT.max()-XT.min():6.1f} mm")
print("DOF  4-bar: 3(4-1)-2*4=1 | 6-bar: 3(6-1)-2*7=1 | 8-bar: 3(8-1)-2*10=1")

# ============================================================ real-time animation
fig, axes = plt.subplots(1, 3, figsize=(19, 7))
segs1 = [(0,3),(0,1),(1,2),(2,3),(1,4)]                          # O2-O4,O2-A,A-C,C-O4,A-P
segs2 = [(0,1),(1,2),(2,3),(1,3),(7,2),(6,4),(3,4),(4,5),(3,5)]  # crank,rod tri,rockers,leg tri
segs3 = [(0,2),(0,1),(2,3),(1,3),(2,4),(1,4),(2,5),(3,5),(5,6),(4,6),(4,7),(6,7)]
titles = [f"Stage 1 - FOUR-BAR (synthesised)  RMSE {rmse4:.1f} mm",
          f"Stage 2 - KLANN SIX-BAR (patent)  RMSE {rmse6:.1f} mm",
          f"Stage 3 - JANSEN EIGHT-BAR  RMSE {rmse8:.1f} mm"]
for ax, t in zip(axes, titles):
    ax.plot(XT, YT, "k--", lw=1.6)
    ax.plot([XT.min()-50, XT.max()+50], [0, 0], "k-", lw=1.5)
    ax.set_xlim(XT.min()-80, XT.max()+80)
    ax.set_ylim(-120, 1350)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(t, fontsize=10)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
links, trails = [], []
for ax, ns in zip(axes, [5, 9, 12]):
    hl = [ax.plot([], [], "-o", lw=2, ms=4, color="tab:blue")[0] for _ in range(ns)]
    links.append(hl)
    trails.append(ax.plot([], [], "r-", lw=1.4)[0])

def update(f):
    i = f % 360
    p4, p6, p8 = pose(i)
    for P, segs, hl, tr, Xs, Ys in [(p4, segs1, links[0], trails[0], X4, Y4),
                                    (p6, segs2, links[1], trails[1], X6, Y6),
                                    (p8, segs3, links[2], trails[2], X8, Y8)]:
        for ln, (a, b) in zip(hl, segs):
            ln.set_data([P[0, a], P[0, b]], [P[1, a], P[1, b]])
        tr.set_data(Xs[:i+1], Ys[:i+1])
    return [x for hl in links for x in hl] + trails

ani = FuncAnimation(fig, update, frames=360, interval=40, blit=True, repeat=True)
if SAVE_GIF:
    print("\nExporting GIF (this takes a minute) ...")
    ani.save("evolution_realtime.gif", writer=PillowWriter(fps=25))
    print("saved evolution_realtime.gif")
plt.tight_layout()
plt.show()