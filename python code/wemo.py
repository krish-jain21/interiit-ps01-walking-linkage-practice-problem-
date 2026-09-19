#!/usr/bin/env python3
"""
Checked walking-linkage synthesis and validation.

Stages:
  1. Synthesised four-bar crank-rocker
  2. Klann six-bar from US Patent 6,260,862
  3. Jansen eight-bar from the published holy numbers

Outputs:
  - optimizer error after every generation;
  - a summary of every mechanism run;
  - pointwise foot-tip error at every crank angle;
  - optional trajectory plot.

Examples:
  python walking_linkage_checked.py
  python walking_linkage_checked.py --no-synthesis
  python walking_linkage_checked.py --generations 10 --plot

Only numpy is required for fixed-linkage validation.
scipy is required for four-bar synthesis.
matplotlib is required only for --plot.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_NAME = "target_foot_tip_path.csv"


class InputError(RuntimeError):
    """Invalid input data or command-line selection."""


class GeometryError(ValueError):
    """A linkage cannot be assembled at one or more crank angles."""


@dataclass(frozen=True)
class TargetPath:
    theta_deg: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    lift_mm: float
    stride_mm: float


@dataclass
class PathMetrics:
    rmse_mm: float
    mae_mm: float
    max_error_mm: float
    max_error_theta_deg: float
    stride_mm: float
    stride_error_mm: float
    lift_error_mm: float
    scale: float
    phase_shift_samples: int
    display_x_mm: np.ndarray
    display_y_mm: np.ndarray
    pointwise_error_mm: np.ndarray


@dataclass
class RunResult:
    stage: str
    status: str
    rmse_mm: Optional[float] = None
    mae_mm: Optional[float] = None
    max_error_mm: Optional[float] = None
    max_error_theta_deg: Optional[float] = None
    stride_mm: Optional[float] = None
    stride_error_mm: Optional[float] = None
    lift_error_mm: Optional[float] = None
    scale: Optional[float] = None
    phase_shift_samples: Optional[int] = None
    assembly_margin_mm: Optional[float] = None
    message: str = ""


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def find_target_file(explicit_path: Optional[Path]) -> Path:
    """Find the target CSV without depending on the current directory."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.is_file():
            return path
        raise InputError(f"Target CSV not found: {path}")

    bases = [
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
        Path.cwd(),
    ]

    relatives = [
        TARGET_NAME,
        Path("data") / TARGET_NAME,
        Path("simulations") / TARGET_NAME,
        Path("python code") / TARGET_NAME,
    ]

    checked: List[Path] = []
    seen = set()

    for base in bases:
        for relative in relatives:
            candidate = (base / relative).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            checked.append(candidate)
            if candidate.is_file():
                return candidate

    tried = "\n  ".join(str(path) for path in checked)
    raise InputError(
        "Target CSV was not found. Tried these locations:\n  " + tried
    )


def load_target(path: Path) -> TargetPath:
    """Load and validate the supplied target trajectory."""
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    except Exception as exc:
        raise InputError(f"Could not read target CSV {path}: {exc}") from exc

    if data.ndim != 2 or data.shape[1] < 3:
        raise InputError(
            f"{path} must contain at least theta_deg, x_mm and y_mm columns."
        )

    if data.shape[0] < 8:
        raise InputError(f"{path} contains too few trajectory samples.")

    theta = np.asarray(data[:, 0], dtype=float)
    x = np.asarray(data[:, 1], dtype=float)
    y = np.asarray(data[:, 2], dtype=float)

    if not (
        np.all(np.isfinite(theta))
        and np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
    ):
        raise InputError("Target trajectory contains NaN or infinite values.")

    n = theta.size

    # A uniform grid is required because phase shifts are indexed by sample.
    wrapped = np.r_[theta, theta[0] + 360.0]
    steps = np.diff(wrapped)
    nominal_step = 360.0 / n

    if np.any(steps <= 0.0) or not np.allclose(
        steps, nominal_step, rtol=0.0, atol=0.02
    ):
        raise InputError(
            "Target theta_deg must be a uniform grid covering exactly 360°."
        )

    # Remove arbitrary vertical datum.
    y = y - np.min(y)

    lift = float(np.ptp(y))
    stride = float(np.ptp(x))

    return TargetPath(
        theta_deg=theta,
        x_mm=x,
        y_mm=y,
        lift_mm=lift,
        stride_mm=stride,
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def as_points(value: np.ndarray) -> np.ndarray:
    """Convert a single point or a (2, N) point array to shape (2, N)."""
    array = np.asarray(value, dtype=float)

    if array.ndim == 1 and array.size == 2:
        return array.reshape(2, 1)

    if array.ndim == 2 and array.shape[0] == 2:
        return array

    raise GeometryError(
        f"Expected a point or a (2, N) array, received shape {array.shape}."
    )


def circle_intersection(
    p1: np.ndarray,
    radius1: float,
    p2: np.ndarray,
    radius2: float,
    branch: float,
    label: str = "intersection",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Return points satisfying:
        |point-p1| = radius1
        |point-p2| = radius2

    branch must be +1 or -1. Failures are not silently clamped.
    """
    radius1 = float(radius1)
    radius2 = float(radius2)
    branch = float(branch)

    if radius1 <= 0.0 or radius2 <= 0.0:
        raise GeometryError(f"{label}: link lengths must be positive.")

    if branch not in (-1.0, 1.0):
        raise GeometryError(f"{label}: branch must be +1 or -1.")

    p1 = as_points(p1)
    p2 = as_points(p2)

    try:
        p1, p2 = np.broadcast_arrays(p1, p2)
    except ValueError as exc:
        raise GeometryError(
            f"{label}: incompatible point-array shapes."
        ) from exc

    if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
        raise GeometryError(f"{label}: non-finite joint coordinate.")

    delta = p2 - p1
    distance = np.linalg.norm(delta, axis=0)

    tolerance = 1.0e-10 * max(radius1, radius2, 1.0)
    minimum_distance = abs(radius1 - radius2)
    maximum_distance = radius1 + radius2

    invalid = (
        (distance < tolerance)
        | (distance < minimum_distance - tolerance)
        | (distance > maximum_distance + tolerance)
    )

    if np.any(invalid):
        index = int(np.flatnonzero(invalid)[0])
        raise GeometryError(
            f"{label}: impossible circle intersection at sample {index}; "
            f"centre distance={distance[index]:.9g}, "
            f"allowed range=[{minimum_distance:.9g}, "
            f"{maximum_distance:.9g}]."
        )

    unit = delta / distance

    along = (
        distance**2 + radius1**2 - radius2**2
    ) / (2.0 * distance)

    height_squared = radius1**2 - along**2

    if np.any(height_squared < -tolerance):
        index = int(np.flatnonzero(height_squared < -tolerance)[0])
        raise GeometryError(
            f"{label}: negative intersection height at sample {index}."
        )

    # Only numerical roundoff has been rejected at this point.
    height = np.sqrt(np.maximum(height_squared, 0.0))
    normal = np.vstack((-unit[1], unit[0]))

    point = p1 + unit * along + normal * (branch * height)

    reach_margin = np.minimum(
        distance - minimum_distance,
        maximum_distance - distance,
    )

    info = {
        "boundary_margin": float(np.min(reach_margin)),
        "intersection_height": float(np.min(height)),
    }

    return point, info


# ---------------------------------------------------------------------------
# Correct trajectory matching
# ---------------------------------------------------------------------------

def best_match(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    target: TargetPath,
) -> PathMetrics:
    """
    Uniformly scale the candidate to the target lift, remove translation,
    and scan every complete-curve phase shift.

    Unlike the original repository code, BOTH x and y are shifted together.
    """
    raw_x = np.asarray(raw_x, dtype=float)
    raw_y = np.asarray(raw_y, dtype=float)

    if raw_x.shape != raw_y.shape:
        raise GeometryError("Candidate x and y arrays have different shapes.")

    if not (np.all(np.isfinite(raw_x)) and np.all(np.isfinite(raw_y))):
        raise GeometryError("Candidate trajectory contains non-finite values.")

    candidate_lift = float(np.ptp(raw_y))

    if candidate_lift <= 0.0:
        raise GeometryError("Candidate trajectory has zero lift.")

    scale = target.lift_mm / candidate_lift

    candidate_x = (raw_x - np.mean(raw_x)) * scale
    candidate_y = (raw_y - np.mean(raw_y)) * scale

    target_x = target.x_mm - np.mean(target.x_mm)
    target_y = target.y_mm - np.mean(target.y_mm)

    n = candidate_x.size
    base_indices = np.arange(n)

    best_rmse = np.inf
    best_shift = 0
    best_indices = base_indices

    for shift in range(n):
        indices = np.roll(base_indices, shift)
        error = np.hypot(
            candidate_x[indices] - target_x,
            candidate_y[indices] - target_y,
        )
        rmse = float(np.sqrt(np.mean(error**2)))

        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = shift
            best_indices = indices

    aligned_x = candidate_x[best_indices]
    aligned_y = candidate_y[best_indices]
    pointwise_error = np.hypot(
        aligned_x - target_x,
        aligned_y - target_y,
    )

    max_index = int(np.argmax(pointwise_error))

    # Put the candidate in the same visual frame as the target.
    display_x = aligned_x + np.mean(target.x_mm)
    display_y = aligned_y - np.min(aligned_y)

    candidate_stride = float(np.ptp(display_x))
    candidate_lift = float(np.ptp(display_y))

    return PathMetrics(
        rmse_mm=best_rmse,
        mae_mm=float(np.mean(pointwise_error)),
        max_error_mm=float(np.max(pointwise_error)),
        max_error_theta_deg=float(target.theta_deg[max_index]),
        stride_mm=candidate_stride,
        stride_error_mm=candidate_stride - target.stride_mm,
        lift_error_mm=candidate_lift - target.lift_mm,
        scale=float(scale),
        phase_shift_samples=int(best_shift),
        display_x_mm=display_x,
        display_y_mm=display_y,
        pointwise_error_mm=pointwise_error,
    )


# ---------------------------------------------------------------------------
# Four-bar synthesis
# ---------------------------------------------------------------------------

def fourbar_path(
    parameters: np.ndarray,
    sample_count: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Ground pivots:
        O = (0, 0)
        Q = (g, 0)

    Input crank:
        B = O + r1*(cos(theta), sin(theta))

    Coupler point:
        P = B + bp at angle (angle O? no: angle B->C) + beta
    """
    g, crank, coupler, rocker, coupler_radius, beta = map(
        float, parameters
    )

    if min(g, crank, coupler, rocker, coupler_radius) <= 0.0:
        raise GeometryError("Four-bar lengths must all be positive.")

    theta = np.linspace(
        0.0, 2.0 * np.pi, sample_count, endpoint=False
    )

    crank_x = crank * np.cos(theta)
    crank_y = crank * np.sin(theta)
    crank_point = np.vstack((crank_x, crank_y))
    ground_point = np.array([[g], [0.0]])

    coupler_point, assembly = circle_intersection(
        crank_point,
        coupler,
        ground_point,
        rocker,
        branch=+1.0,
        label="four-bar coupler/rocker intersection",
    )

    link_lengths = np.sort(np.array([g, crank, coupler, rocker]))
    shortest = link_lengths[0]
    longest = link_lengths[-1]
    middle_sum = link_lengths[1] + link_lengths[2]

    grashof_tolerance = 1.0e-9 * np.sum(link_lengths)
    grashof = (
        crank <= shortest + grashof_tolerance
        and shortest + longest <= middle_sum + grashof_tolerance
    )

    coupler_angle = np.arctan2(
        coupler_point[1] - crank_y,
        coupler_point[0] - crank_x,
    )

    foot_x = crank_x + coupler_radius * np.cos(
        coupler_angle + beta
    )
    foot_y = crank_y + coupler_radius * np.sin(
        coupler_angle + beta
    )

    info = {
        "grashof": float(grashof),
        "assembly_margin": min(
            assembly["boundary_margin"],
            assembly["intersection_height"],
        ),
    }

    return foot_x, foot_y, info


def synthesize_fourbar(
    target: TargetPath,
    generations: int,
    population_size: int,
    seed: int,
) -> Tuple[RunResult, PathMetrics]:
    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for four-bar synthesis. "
            "Install it with: pip install scipy"
        ) from exc

    bounds = [
        (50.0, 300.0),       # ground
        (10.0, 90.0),        # crank
        (50.0, 400.0),       # coupler
        (50.0, 400.0),       # rocker
        (50.0, 600.0),       # coupler point radius
        (0.0, 2.0 * np.pi),  # coupler point angle
    ]

    generation_history: List[float] = []

    def objective(parameters: np.ndarray) -> float:
        try:
            raw_x, raw_y, info = fourbar_path(
                parameters, target.theta_deg.size
            )
        except GeometryError:
            return 1.0e6

        normal_score = best_match(raw_x, raw_y, target).rmse_mm
        reflected_score = best_match(raw_x, -raw_y, target).rmse_mm

        penalty = 0.0 if info["grashof"] > 0.5 else 500.0
        return min(normal_score, reflected_score) + penalty

    def optimizer_callback(
        best_candidate: np.ndarray,
        convergence: float = 0.0,
    ) -> None:
        value = objective(best_candidate)
        generation_history.append(value)

        lengths = np.sort(best_candidate[:4])
        shortest = lengths[0]
        longest = lengths[-1]
        middle = lengths[1] + lengths[2]

        feasible = (
            best_candidate[1] <= shortest + 1.0e-10
            and shortest + longest <= middle + 1.0e-10
        )

        print(
            f"[4-bar generation {len(generation_history):02d}] "
            f"objective={value:9.3f} mm  "
            f"Grashof={'yes' if feasible else 'no'}  "
            f"convergence={convergence:.3e}",
            flush=True,
        )

    print(
        f"Starting four-bar synthesis: generations={generations}, "
        f"population={population_size}, seed={seed}"
    )

    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=generations,
        popsize=population_size,
        tol=1.0e-3,
        mutation=(0.4, 1.2),
        recombination=0.8,
        polish=True,
        updating="immediate",
        workers=1,
        callback=optimizer_callback,
    )

    raw_x, raw_y, geometry_info = fourbar_path(
        result.x, target.theta_deg.size
    )

    normal_metrics = best_match(raw_x, raw_y, target)
    reflected_metrics = best_match(raw_x, -raw_y, target)

    if reflected_metrics.rmse_mm < normal_metrics.rmse_mm:
        metrics = reflected_metrics
        orientation = "reflected y"
    else:
        metrics = normal_metrics
        orientation = "normal y"

    status = "OK"
    messages = [
        f"orientation={orientation}",
        f"DE success={bool(result.success)}",
        str(result.message),
    ]

    if geometry_info["grashof"] < 0.5:
        status = "WARN"
        messages.append("final design violates the crank-rocker Grashof test")

    if not result.success:
        status = "WARN"

    run_result = RunResult(
        stage="Four-bar synthesis",
        status=status,
        rmse_mm=metrics.rmse_mm,
        mae_mm=metrics.mae_mm,
        max_error_mm=metrics.max_error_mm,
        max_error_theta_deg=metrics.max_error_theta_deg,
        stride_mm=metrics.stride_mm,
        stride_error_mm=metrics.stride_error_mm,
        lift_error_mm=metrics.lift_error_mm,
        scale=metrics.scale,
        phase_shift_samples=metrics.phase_shift_samples,
        assembly_margin_mm=geometry_info["assembly_margin"] * metrics.scale,
        message="; ".join(messages),
    )

    return run_result, metrics


# ---------------------------------------------------------------------------
# Fixed published mechanisms
# ---------------------------------------------------------------------------

def klann_path(sample_count: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Klann six-bar geometry derived from US Patent 6,260,862.
    Coordinates and dimensions are in the patent's original units.
    """
    p15 = np.array([17.607, 11.807])
    p9 = np.array([17.818, 16.076])
    p11 = np.array([12.101, 10.186])

    theta = np.linspace(
        0.0, 2.0 * np.pi, sample_count, endpoint=False
    )

    joint29 = p15[:, None] + 2.976 * np.vstack(
        (np.cos(theta), np.sin(theta))
    )

    joint27, info27 = circle_intersection(
        joint29, 5.506, p11[:, None], 3.390, -1,
        "Klann joint 27",
    )
    joint35, info35 = circle_intersection(
        joint29, 11.679, joint27, 6.236, -1,
        "Klann joint 35",
    )
    joint37, info37 = circle_intersection(
        joint35, 10.137, p9[:, None], 7.391, +1,
        "Klann joint 37",
    )
    joint33, info33 = circle_intersection(
        joint35, 13.442, joint37, 22.188, -1,
        "Klann joint 33",
    )

    # Compare with the patent's fully extended pose at 180 degrees.
    pose_index = int(round((np.pi / (2.0 * np.pi)) * sample_count))
    pose_index %= sample_count

    expected = [
        (joint27, np.array([9.125, 11.807])),
        (joint35, np.array([3.024, 13.099])),
        (joint37, np.array([11.119, 19.200])),
        (joint33, np.array([0.0, 0.0])),
    ]

    snapshot_error = sum(
        float(np.linalg.norm(actual[:, pose_index] - wanted))
        for actual, wanted in expected
    )

    if snapshot_error > 0.05:
        raise GeometryError(
            f"Klann patent-pose check failed: error={snapshot_error:.6f}."
        )

    assembly_margin = min(
        info["boundary_margin"]
        for info in (info27, info35, info37, info33)
    )

    assembly_margin = min(
        assembly_margin,
        info27["intersection_height"],
        info35["intersection_height"],
        info37["intersection_height"],
        info33["intersection_height"],
    )

    info = {
        "assembly_margin": assembly_margin,
        "snapshot_error": snapshot_error,
    }

    return joint33[0], joint33[1], info


def jansen_path(sample_count: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Jansen eight-bar using the published holy numbers.
    Toe point is joint G.
    """
    theta = np.linspace(
        0.0, 2.0 * np.pi, sample_count, endpoint=False
    )

    joint_a = np.vstack(
        (
            15.0 * np.cos(theta),
            15.0 * np.sin(theta),
        )
    )
    joint_b = np.array([[-38.0], [-7.8]])

    joint_c, info_c = circle_intersection(
        joint_b, 41.5, joint_a, 50.0, +1, "Jansen joint C"
    )
    joint_d, info_d = circle_intersection(
        joint_b, 39.3, joint_a, 61.9, -1, "Jansen joint D"
    )
    joint_e, info_e = circle_intersection(
        joint_b, 40.1, joint_c, 55.8, +1, "Jansen joint E"
    )
    joint_f, info_f = circle_intersection(
        joint_d, 36.7, joint_e, 39.4, +1, "Jansen joint F"
    )
    joint_g, info_g = circle_intersection(
        joint_d, 49.0, joint_f, 65.7, +1, "Jansen toe G"
    )

    assembly_margin = min(
        info["boundary_margin"]
        for info in (info_c, info_d, info_e, info_f, info_g)
    )

    assembly_margin = min(
        assembly_margin,
        info_c["intersection_height"],
        info_d["intersection_height"],
        info_e["intersection_height"],
        info_f["intersection_height"],
        info_g["intersection_height"],
    )

    return joint_g[0], joint_g[1], {
        "assembly_margin": assembly_margin
    }


def evaluate_fixed_stage(
    stage_name: str,
    target: TargetPath,
    builder: Callable[[int], Tuple[np.ndarray, np.ndarray, Dict[str, float]]],
    note: str = "",
) -> Tuple[RunResult, Optional[PathMetrics]]:
    try:
        raw_x, raw_y, geometry_info = builder(target.theta_deg.size)
        metrics = best_match(raw_x, raw_y, target)

        messages = []
        if note:
            messages.append(note)

        if stage_name == "Klann six-bar":
            messages.append(
                "patent-pose error="
                f"{geometry_info.get('snapshot_error', np.nan):.6f}"
            )

        run_result = RunResult(
            stage=stage_name,
            status="OK",
            rmse_mm=metrics.rmse_mm,
            mae_mm=metrics.mae_mm,
            max_error_mm=metrics.max_error_mm,
            max_error_theta_deg=metrics.max_error_theta_deg,
            stride_mm=metrics.stride_mm,
            stride_error_mm=metrics.stride_error_mm,
            lift_error_mm=metrics.lift_error_mm,
            scale=metrics.scale,
            phase_shift_samples=metrics.phase_shift_samples,
            assembly_margin_mm=(
                geometry_info["assembly_margin"] * metrics.scale
            ),
            message="; ".join(messages),
        )

        return run_result, metrics

    except Exception as exc:
        return (
            RunResult(
                stage=stage_name,
                status="FAIL",
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_value(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def print_run_result(result: RunResult) -> None:
    print(f"\n{result.stage}")
    print(f"  status                 : {result.status}")
    print(f"  RMSE                   : {format_value(result.rmse_mm)} mm")
    print(f"  mean absolute error    : {format_value(result.mae_mm)} mm")
    print(f"  maximum point error    : {format_value(result.max_error_mm)} mm")
    print(
        "  maximum error at theta : "
        f"{format_value(result.max_error_theta_deg)} deg"
    )
    print(f"  stride                 : {format_value(result.stride_mm)} mm")
    print(
        "  stride error           : "
        f"{format_value(result.stride_error_mm)} mm"
    )
    print(
        "  lift error             : "
        f"{format_value(result.lift_error_mm)} mm"
    )
    print(f"  trajectory scale       : {format_value(result.scale)}")
    print(
        "  best phase shift       : "
        f"{result.phase_shift_samples if result.phase_shift_samples is not None else 'n/a'} samples"
    )
    print(
        "  assembly margin        : "
        f"{format_value(result.assembly_margin_mm)} mm"
    )

    if result.message:
        print(f"  notes                  : {result.message}")


def write_summary(
    path: Path,
    results: List[RunResult],
) -> None:
    fields = [
        "stage",
        "status",
        "rmse_mm",
        "mae_mm",
        "max_error_mm",
        "max_error_theta_deg",
        "stride_mm",
        "stride_error_mm",
        "lift_error_mm",
        "scale",
        "phase_shift_samples",
        "assembly_margin_mm",
        "message",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    field: getattr(result, field)
                    for field in fields
                }
            )


def write_pointwise_errors(
    path: Path,
    target: TargetPath,
    traces: Dict[str, PathMetrics],
) -> None:
    if not traces:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["theta_deg"]
        stage_column_names: Dict[str, str] = {}

        for stage_name in traces:
            column = (
                stage_name.lower()
                .replace("-", "_")
                .replace(" ", "_")
                + "_error_mm"
            )
            stage_column_names[stage_name] = column
            fieldnames.append(column)

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, theta in enumerate(target.theta_deg):
            row = {"theta_deg": f"{theta:.6f}"}

            for stage_name, metrics in traces.items():
                column = stage_column_names[stage_name]
                row[column] = (
                    f"{metrics.pointwise_error_mm[index]:.6f}"
                )

            writer.writerow(row)


def save_plot(
    path: Path,
    target: TargetPath,
    traces: Dict[str, PathMetrics],
) -> None:
    if not traces:
        print("No successful mechanism trajectories are available for plotting.")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"PLOT WARNING: matplotlib is unavailable: {exc}")
        return

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        target.x_mm,
        target.y_mm,
        "k--",
        linewidth=2.0,
        label="Target",
    )

    for stage_name, metrics in traces.items():
        axis.plot(
            metrics.display_x_mm,
            metrics.display_y_mm,
            linewidth=1.4,
            label=(
                f"{stage_name} "
                f"(RMSE {metrics.rmse_mm:.2f} mm)"
            ),
        )

    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_title("Walking-linkage trajectory comparison")
    axis.grid(True, alpha=0.3)
    axis.set_aspect("equal")
    axis.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Program entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Checked walking-linkage synthesis and validation."
    )

    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        help=(
            "Optional path to target_foot_tip_path.csv. "
            "If omitted, common repository locations are searched."
        ),
    )
    parser.add_argument(
        "--no-synthesis",
        action="store_true",
        help="Skip the slow four-bar differential-evolution stage.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
        help="Maximum four-bar optimizer generations.",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=12,
        help="Differential-evolution population multiplier.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible four-bar synthesis.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a trajectory comparison PNG.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=SCRIPT_DIR / "walking_linkage_run_summary.csv",
        help="Output path for the run summary CSV.",
    )
    parser.add_argument(
        "--errors-out",
        type=Path,
        default=SCRIPT_DIR / "walking_linkage_pointwise_errors.csv",
        help="Output path for pointwise crank-angle errors.",
    )

    args = parser.parse_args(argv)

    try:
        target_path = find_target_file(args.target)
        target = load_target(target_path)
    except (InputError, OSError) as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Target file : {target_path}")
    print(f"Samples     : {target.theta_deg.size}")
    print(f"Target lift : {target.lift_mm:.4f} mm")
    print(f"Target stride: {target.stride_mm:.4f} mm")

    if not np.isclose(target.lift_mm, 200.0, atol=0.05):
        print(
            f"WARNING: target lift is {target.lift_mm:.4f} mm, "
            "not approximately 200 mm."
        )

    results: List[RunResult] = []
    traces: Dict[str, PathMetrics] = {}

    # Stage 1 -----------------------------------------------------------
    if args.no_synthesis:
        results.append(
            RunResult(
                stage="Four-bar synthesis",
                status="SKIP",
                message="Skipped by --no-synthesis.",
            )
        )
    else:
        try:
            result, metrics = synthesize_fourbar(
                target,
                generations=args.generations,
                population_size=args.popsize,
                seed=args.seed,
            )
            results.append(result)
            traces[result.stage] = metrics
        except Exception as exc:
            results.append(
                RunResult(
                    stage="Four-bar synthesis",
                    status="FAIL",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )

    # Stage 2 -----------------------------------------------------------
    klann_result, klann_metrics = evaluate_fixed_stage(
        "Klann six-bar",
        target,
        klann_path,
        note="US Patent 6,260,862 dimensions",
    )
    results.append(klann_result)
    if klann_metrics is not None:
        traces[klann_result.stage] = klann_metrics

    # Stage 3 -----------------------------------------------------------
    jansen_result, jansen_metrics = evaluate_fixed_stage(
        "Jansen eight-bar",
        target,
        jansen_path,
        note="Published Jansen holy numbers",
    )
    results.append(jansen_result)
    if jansen_metrics is not None:
        traces[jansen_result.stage] = jansen_metrics

    # Report -------------------------------------------------------------
    print("\n================ RUN SUMMARY ================")
    for result in results:
        print_run_result(result)

    print("\nDOF checks:")
    print("  Four-bar      : F = 3(4-1) - 2(4)  = 1")
    print("  Klann six-bar : F = 3(6-1) - 2(7)  = 1")
    print("  Jansen 8-bar  : F = 3(8-1) - 2(10) = 1")

    try:
        summary_path = args.summary_out.resolve()
        error_path = args.errors_out.resolve()

        write_summary(summary_path, results)
        write_pointwise_errors(error_path, target, traces)

        print("\nOutput files:")
        print(f"  Summary CSV  : {summary_path}")
        print(f"  Error CSV    : {error_path}")
    except OSError as exc:
        print(f"OUTPUT ERROR: {exc}", file=sys.stderr)
        return 1

    if args.plot:
        plot_path = (SCRIPT_DIR / "walking_linkage_comparison.png").resolve()
        save_plot(plot_path, target, traces)
        print(f"  Plot PNG     : {plot_path}")

    failed = any(result.status == "FAIL" for result in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())