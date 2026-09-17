# Single-DOF Walking Linkage — Mechanism Synthesis & Kinematics

**Inter-IIT Tech Meet · Mechanical Problem Statement 01**

Design of a leg mechanism with **exactly one degree of freedom and one actuator** whose
foot-tip traces a target path (605 mm stride, 200 mm peak lift, flat stance phase),
synthesised from first principles and verified numerically against the supplied data.

---

## Result at a glance

| Stage | Mechanism | Source | RMSE vs target | Stride @ 200 mm lift | DOF |
|-------|-----------|--------|---------------:|---------------------:|----:|
| 1 | Four-bar crank-rocker | **synthesised here from scratch** | 49.7 mm | 580.0 mm | 1 |
| 2 | Klann six-bar | US patent 6,260,862 | 133.4 mm | 328.6 mm | 1 |
| 3 | **Jansen eight-bar** | Theo Jansen "holy numbers" | **0.00 mm** | **604.8 mm** | 1 |
| — | Target (`target_foot_tip_path.csv`) | supplied | — | 604.8 mm | — |

**Key finding:** the supplied target is *exactly* Jansen's linkage toe curve, generated
with a single uniform scale factor **s = 8.9059414** (residual 4×10⁻⁵ mm, i.e. CSV
rounding). The CSV's `theta_deg` column is the crank angle directly. Our independent
four-bar attempt is what led us to recognise this family (see *Approach* below).

---

## Approach (as required by the problem statement)

1. **First attempt, from first principles.** A four-bar crank-rocker was synthesised
   *without consulting any walking-linkage literature*: differential evolution over
   six link parameters with the Grashof condition (crank shortest, s+l ≤ p+q) enforced
   as a penalty so the crank fully rotates. Result: 49.7 mm RMSE — right character
   (leaf shape, flat-ish stance), but a single coupler point cannot simultaneously
   flatten the stance *and* reach the 3.02:1 stride:lift ratio of the target.
2. **Iterate / understand the wall.** Multiple optimiser restarts converged to the
   same ~50 mm wall → the limitation is topological, not numerical. Two stacked
   four-bar loops on one crank are needed to reshape the curve.
3. **Only now, research (problem step 5).** The two canonical eight-bar walkers are
   Klann's and Jansen's linkages. Both were implemented from their published
   proportions and verified: Klann assembles correctly (patent-pose assert, error
   < 0.05 mm) but its native stride:lift ratio is ~1.64:1 — wrong animal for this
   target. Jansen reproduces the CSV to 0.00 mm.
4. **DOF re-derived at every stage** (Grübler–Kutzbach, see below) — no sub-loop was
   ever left under-constrained while links were added.

---

## Grübler–Kutzbach DOF proofs  (F = 3(n−1) − 2·j₁ − j₂)

| Stage | n | j₁ (revolute, with multiplicity) | F |
|-------|---|----------------------------------|---|
| Four-bar | 4 (ground, crank, coupler, rocker) | 4 | **3·3 − 2·4 = 1** |
| Klann six-bar | 6 | 7 | **3·5 − 2·7 = 1** |
| Jansen eight-bar | 8 | 10 (ternary pins at A and D each count twice) | **3·7 − 2·10 = 1** |

Jansen joint inventory: O(ground–crank); B(ground–b, ground–c); A(m–j, m–k);
C(b–j); D(c–k, k–foot); E(bde–f); F(f–foot).

**Branch defects:** every circle-intersection sub-assembly keeps one fixed branch for
the full cycle (Jansen: C:+1 D:−1 E:+1 F:+1 G:+1; Klann: 27:−1 35:−1 37:+1 33:−1),
selected by matching the published pose snapshot. No toggle/dead-centre is crossed by
the crank (Grashof crank-rocker).

---

## Repository structure

```
├── README.md                     ← this file
├── data/
│   └── target_foot_tip_path.csv  ← supplied target trajectory
├── simulations/
│   ├── walking_evolution_realtime.py   ← 4→6→8 evolution, LIVE animation (side-by-side)
│   ├── walking_evolution_4_6_8.py      ← same, static 3-panel figure + summary
│   ├── walking_linkage_sim.py          ← stage-1 four-bar synthesis (standalone)
│   ├── jansen_walking_sim.m            ← Jansen 8-bar, MATLAB (FK + overlay + anim)
│   └── walking_evolution_sim.m         ← full 4→6→8 evolution, MATLAB real-time
├── results/
│   ├── trajectory_overlay.png          ← four-bar vs target
│   ├── jansen_vs_target.png            ← exact reconstruction proof
│   ├── evolution_4bar_6bar_8bar.png    ← 3-stage path comparison
│   ├── linkage_phases.png / linkage_phases_fixed.png
│   └── *.gif                           ← one-revolution animations
├── cad/                                ← (add your CAD exports here)
└── report/                             ← (add report PDF / hand calcs here)
```

## How to run

**Python** (needs `numpy scipy matplotlib`):
```bash
cd simulations
python walking_evolution_realtime.py          # live 3-panel animation
python walking_evolution_realtime.py --save   # also exports evolution_realtime.gif
```
Put `target_foot_tip_path.csv` next to the script (or fix the path at the top).
Stage-1 synthesis takes ~30 s; stages 2–3 are instant.

**MATLAB** (R2016b+, no toolboxes):
```matlab
>> jansen_walking_sim        % Jansen FK, overlay, phase sketch, animation
>> walking_evolution_sim     % full 4->6->8 synthesis + real-time animation
```

---

## Synthesised dimensions (final, scaled so lift = 200 mm)

All Jansen "holy numbers" × 8.9059414: crank m = 133.6 mm, b = 369.6, c = 350.0,
d = 357.1, e = 497.0, f = 350.9, g = 326.9, h = 585.1, i = 436.4, j = 445.3,
k = 551.3, ground a/l = 338.4/69.5. Fixed pivots O = (0,0), B = (−338.4, −69.5) mm.
Achieved stride **604.8 mm** (target 604.8 mm) — ratio 3.02:1.

## Section 5 — Adapting the design (terrain)

1. **Frequent 10–15 mm rocks, closely spaced** → raise lift to ~220–230 mm by scaling
   the *whole* linkage uniformly (stride grows to ~660 mm — acceptable, legs are
   longer). Non-uniform crank-only scaling would distort the curve (the swing loop
   is far more crank-sensitive than the stance); uniform scaling is the safe move.
   Trade-off: leg mass/inertia and packaging height.
2. **Smooth flat gravel — efficiency over clearance** → shrink lift toward ~120 mm by
   the same uniform scaling (stride drops to ~360 mm at fixed crank speed, so for the
   same ground speed you'd raise crank RPM; actuator torque demand drops ~40%).
   Trade-off: any real obstacle now becomes a trip hazard — this config is a
   track-only mode.
3. **Sustained 15° incline — traction & mechanical advantage** → retune the coupler
   point: move the toe attachment (β, BP) so the stance segment stays normal to the
   slope and the crank passes its low-torque zone during push-off; do **not** scale
   lift (200 mm is wasted on a slope). Optionally phase-shift left/right crank pairs
   so one leg is always in stance. Trade-off: swing loop becomes asymmetric and the
   flat-terrain stride efficiency worsens — a mode you switch, not a compromise.

## Sources (read *after* our own first attempt, per problem rules)

- T. Jansen, *The Great Pretender* (holy numbers; ratio table widely republished)
- J. Klann, US Patent 6,260,862 "Walking Device" (link lengths + fixed-pivot coordinates)
- Grübler/Kutzbach criterion — any mechanism-text treatment (Norton, *Design of Machinery*)
- Target curve proven to equal Jansen × 8.9059414 by exact reconstruction
  (residual 4×10⁻⁵ mm) — see `results/jansen_vs_target.png`

## Known limitations / honest notes

- Stage-1 four-bar fit (49.7 mm RMSE) is a shape match, not a discovery — the target
  data *is* Jansen's curve, so Stage 3 is verification rather than optimisation.
- Interference: at 8.9× scale links are long and thin; pin joints at A and D carry
  three links each — check shoulder bolts and plate clearances in CAD before build.
- All motion is kinematic; no dynamics (inertia, friction, motor torque) is modelled.
```
