"""潜热与近临界研究共享的记录、局部时钟、守恒和验收实现。"""

from __future__ import annotations
import os, sys, json, hashlib
from pathlib import Path
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
from functools import lru_cache
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import scipy
from scipy.integrate import BDF
from scipy.integrate._ivp.ivp import solve_event_equation
from scipy.optimize import brentq
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from 公共接口 import configure, section, paths, read_json, write_json, sha256, source_identity

ORDERS = (8, 16)
LIMITS = {
    "moisture_absolute": 5e-8,
    "heat_relative": 1e-7,
    "heat_B0_equivalent_K": 1e-7,
    "quadrature_moisture_absolute": 5e-8,
    "quadrature_heat_B0_equivalent_K": 1e-7,
}
COMPONENTS = (
    "moisture_flux",
    "heat_flux",
    "heat_abs_flux",
    "heat_rate_volume",
    "convective_signed",
    "latent_signed",
    "gross_absolute_channels",
)
MAIN = dict(
    method="BDF",
    rtol=1e-11,
    atol_temperature_K=1e-11,
    atol_moisture_C=1e-13,
    max_step_s=75.0,
    limit_initial_step=True,
)
TIGHT = dict(MAIN, rtol=1e-12, atol_temperature_K=1e-12, atol_moisture_C=1e-14)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity():
    """Bind actual new implementations and original input files, not old acceptance receipts."""
    p = paths()
    files = dict(source_identity()["files"])
    selected = {
        name: Path(__file__).with_name(name)
        for name in ("研究求解.py", "潜热矩阵.py", "事件覆盖.py")
    }
    selected.update(
        config=Path(p["contract"]),
        environment=Path(p["inputs"]) / "附件1.xlsx",
        radius=Path(p["inputs"]) / "附件2.xlsx",
        latent_csv=Path(p["latent_table"]) / "L_water_reference.csv",
        latent_manifest=Path(p["latent_table"]) / "manifest.json",
        near_critical_protocol=Path(p["near_critical_protocol"]),
    )
    files.update(
        {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in selected.items()
        }
    )
    return files


def plan_hash(plan):
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def verify_checks_receipt(
    receipt,
    value,
    *,
    schema,
    expected_names,
    stale_message="Missing or stale bounded checks",
    incomplete_message="Incomplete bounded check suite",
):
    """逐项绑定检查回执、计划、源码、运行环境和完整的具名检查集合。"""
    receipt = read_json(receipt) if isinstance(receipt, (str, Path)) else receipt
    require(
        receipt["schema"] == schema
        and receipt["passed"]
        and receipt["plan_sha256"] == plan_hash(value)
        and receipt["source_bindings"] == identity()
        and receipt["runtime"] == runtime(),
        stale_message,
    )
    require(
        receipt["expected_check_names"] == expected_names
        and sorted(row["name"] for row in receipt["checks"]) == expected_names
        and all(row["passed"] for row in receipt["checks"]),
        incomplete_message,
    )
    return receipt


class ProgressChecker:
    """记录每项检查后立即保存进度；失败由原调用位置的条件判定。"""

    def __init__(self, directory, failure_prefix):
        self.path = Path(directory) / "progress.json"
        self.failure_prefix = failure_prefix
        self.rows = []

    def __call__(self, name, ok, **details):
        self.rows.append(dict(name=name, passed=bool(ok), **details))
        write_json(self.path, dict(completed=False, checks=self.rows))
        require(ok, self.failure_prefix + name)


def expected_times(geometry, horizon, first_seconds=10800.0):
    if geometry == "fixed":
        return np.arange(1.0, np.floor(horizon) + 1.0)
    return np.unique(
        np.r_[
            np.arange(1.0, min(first_seconds, np.floor(horizon)) + 1.0),
            np.arange(first_seconds + 60.0, np.floor(horizon) + 1.0, 60.0),
        ]
    )


class Solver:
    """Explicit local-clock BDF; observers read accepted polynomials without patching SciPy."""

    def __init__(
        self, model, configuration, recorder, observer=None, event_guard=None, trace_only=None
    ):
        self.model, self.recorder, self.observer = model, recorder, observer
        self.trace_only = trace_only
        self.s = settings(section("moving_model"), configuration)
        self.event_time = self.event_state = self.event_steps = None
        self.step_count = self.nfev = self.njev = self.nlu = 0
        self.step_hash = hashlib.sha256()
        self.step_times, self.segments = [], []
        self.started = perf_counter()
        if event_guard is not None:
            event_guard.bind_solver(self)

    def make_bdf(self, a, b, y, right):
        """按局部时钟建立BDF，保留初始扩散步长限制。"""
        model, s = self.model, self.s
        first = None
        if s.limit_initial_step and a == 0:
            T, C = model.unpack(y)
            dd = float(model.properties.evaluate(C, T)[3].max())
            if dd > 0:
                first = min(0.01, 0.1 * model.grid.minimum_spacing_m**2 / dd, b - a)
        kwargs = {} if first is None else {"first_step": first}
        solver = BDF(
            lambda tau, u: model.rhs(a + tau, u, right_of_breakpoint=right),
            0.0,
            y,
            float(b - a),
            rtol=s.rtol,
            atol=s.atol_vector(model.nr),
            max_step=s.max_step_s,
            jac_sparsity=model.jacobian_sparsity(),
            **kwargs,
        )
        return solver, first

    def step_once(self, solver, a, context):
        """提交观察器、推进一步、记录稠密多项式，再挂起本步审计。"""
        model, observer = self.model, self.observer
        if observer is not None:
            observer.commit()
        tau_left = float(solver.t)
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError(f"Local-clock BDF failed at a={a}, tau={tau_left}: {message}")
        if observer is not None or self.trace_only is not None:
            poly = solver.dense_output()
            (observer if observer is not None else self.trace_only).note_trace(
                solver, poly, context
            )
        if observer is not None:
            observer.hold(model, poly, tau_left, float(solver.t), context)
        self.step_count += 1
        return tau_left

    def try_event(
        self, solver, a, tau_left, tau_right, local_dense, event, oldg, detect_event, right_t
    ):
        """仅在湿侧到干侧的符号夹逼内定位等号事件。"""
        terminated = False
        newg = event(right_t, solver.y)
        if detect_event and oldg >= 0 and newg <= 0:
            tau_root = float(
                solve_event_equation(
                    lambda tau, u: event(a + tau, u), local_dense, tau_left, tau_right
                )
            )
            right_t, y = a + tau_root, np.asarray(local_dense(tau_root))
            self.event_time, self.event_state = right_t, y.copy()
            self.event_steps, terminated = self.step_count, True
        else:
            y = np.asarray(solver.y)
        return right_t, y, terminated, newg

    def record_targets(self, targets, p, right_t, dense, detect_event):
        """依次采样已被当前接受步覆盖的目标时刻。"""
        while p < len(targets) and targets[p] <= right_t + 1e-10:
            t = float(targets[p])
            self.recorder.add(t, np.asarray(dense(t)), None if detect_event else "post_event_tick")
            p += 1
        return p

    def span(self, a, b, y, targets, detect_event=True):
        model, s, observer = self.model, self.s, self.observer
        right = bool(a >= model.ambient.breakpoint_s and np.isfinite(model.ambient.breakpoint_s))
        context = dict(
            model=model,
            start=float(a),
            detect=bool(detect_event),
            right_side=right,
            query_times=[] if detect_event else np.asarray(targets, dtype=float).tolist(),
        )
        if observer is not None:
            observer.begin(model, a, y)
            observer._last_context = context
            if self.event_time == float(a) and str(float(a)) in observer.checkpoints:
                observer.checkpoints[str(float(a))]["label"] = "equality_root"
        solver, first = self.make_bdf(a, b, y, right)
        event = lambda t, state: float(np.max(state[model.nr :]) - 0.15)
        oldg, p = event(a, y), 0
        while p < len(targets) and targets[p] <= a + 1e-10:
            p += 1
        step_start, terminated = self.step_count, False
        min_local_step, lost_absolute_clock_steps = float("inf"), 0
        while solver.status == "running":
            tau_left = self.step_once(solver, a, context)
            tau_right = float(solver.t)
            min_local_step = min(min_local_step, tau_right - tau_left)
            left, right_t = a + tau_left, a + tau_right
            lost_absolute_clock_steps += int(left == right_t)
            local_dense = solver.dense_output()
            dense = lambda t: local_dense(np.asarray(t) - a)
            right_t, y, terminated, newg = self.try_event(
                solver, a, tau_left, tau_right, local_dense, event, oldg, detect_event, right_t
            )
            self.step_hash.update(np.float64(right_t).tobytes())
            self.step_times.append(right_t)
            p = self.record_targets(targets, p, right_t, dense, detect_event)
            if terminated:
                self.recorder.add(right_t, y, "equality_root")
            oldg = newg
            if terminated:
                break
        self.nfev += solver.nfev
        self.njev += solver.njev
        self.nlu += solver.nlu
        self.segments.append(
            dict(
                start_s=float(a),
                requested_end_s=float(b),
                actual_end_s=right_t,
                first_step_s=first,
                accepted_steps=self.step_count - step_start,
                nfev=solver.nfev,
                njev=solver.njev,
                nlu=solver.nlu,
                event=terminated,
                local_time_origin=True,
                minimum_local_step_s=min_local_step,
                steps_unresolvable_by_absolute_clock=lost_absolute_clock_steps,
            )
        )
        if observer is not None:
            observer.finish_span(a, b, self.event_time if terminated else b, terminated)
        return y, terminated


def advance_window(solver, y, start, horizon, post_window, geometry, first_seconds=10800.0):
    model, recorder = solver.model, solver.recorder
    had_event = solver.event_time is not None
    require(np.isfinite(horizon) and horizon > start, "Window must extend the saved state")
    targets = expected_times(geometry, horizon, first_seconds)
    if not had_event:
        value = float(np.max(y[model.nr :]) - 0.15)
        require(value >= 0, "Saved state is dry without inherited event evidence")
        if value == 0:
            solver.event_time, solver.event_state = start, y.copy()
            solver.event_steps = solver.step_count
            recorder.add(start, y, "equality_root")
        else:
            points = model.ambient.breakpoints(start, horizon)
            for a, b in zip(points[:-1], points[1:]):
                y, event = solver.span(
                    float(a), float(b), y, targets[(targets > a) & (targets <= b)]
                )
                if event:
                    break
    root = solver.event_time
    if root is None:
        end = horizon
        recorder.add(end, y, "window_end_event_pending")
    else:
        end = max(float(root) + post_window, horizon if had_event else 0.0)
        after = start if had_event else float(root)
        targets = expected_times(geometry, end, first_seconds)
        extra = np.unique(
            np.r_[targets[(targets > after) & (targets <= end)], report_ticks(after, end), end]
        )
        points = model.ambient.breakpoints(after, end)
        for a, b in zip(points[:-1], points[1:]):
            y, _ = solver.span(float(a), float(b), y, extra[(extra > a) & (extra <= b)], False)
        recorder.add(end, y, "post_event_final")
    return y, float(end)


def fresh(path):
    path = Path(path).resolve()
    require(
        not path.exists(),
        "Output must be a new directory; failed/partial outputs are retained: " + str(path),
    )
    path.mkdir(parents=True)
    return path


def settings(module, j):
    return module.IntegrationSettings(
        j["method"],
        j["rtol"],
        j["atol_temperature_K"],
        j["atol_moisture_C"],
        j["max_step_s"],
        j["limit_initial_step"],
    )


def report_ticks(root, end):
    unit = Decimal("0.36")
    start = (
        int((Decimal.from_float(float(root)) / unit).to_integral_value(rounding=ROUND_FLOOR)) + 1
    )
    final = int((Decimal.from_float(float(end)) / unit).to_integral_value(rounding=ROUND_FLOOR))
    return np.asarray([float(Decimal(k) * unit) for k in range(start, final + 1)], dtype=float)


class Recorder:
    """Chunked prescribed samples plus complete PL radial fields; never dense history."""

    def __init__(
        self,
        model,
        directory,
        geometry,
        full_chunk=64,
        physical_chunk=1024,
        full_seconds_cutoff=1800.0,
    ):
        self.model = model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.geometry = geometry
        self.limits = {"full": full_chunk, "physical": physical_chunk}
        self.full_seconds_cutoff = float(full_seconds_cutoff)
        self.buffers = {"full": [], "physical": []}
        self.index = {"full": [], "physical": []}
        self.last = None
        self.rows = 0
        self.last_full = None
        self.special = {}
        self.observed = {
            "minimum_C": float("inf"),
            "maximum_C": float("-inf"),
            "minimum_T_K": float("inf"),
            "maximum_T_K": float("-inf"),
        }
        self.audit_max = {
            "moisture_mean_rate_residual": 0.0,
            "thermal_current_volume_rate_residual": 0.0,
        }
        self.r = np.arange(21, dtype=float) * 0.001
        self.w = 2 * model.grid.volume_weights_m2 / model.grid.radius_m**2

    def inherit(self, previous):
        """Reference immutable prior chunks explicitly; never claim their rerun."""
        previous = Path(previous).resolve()
        manifest = read_json(previous / "arrays.json")
        require(manifest["complete"], "Cannot inherit an unfinished storage window")
        for kind, entries in manifest["index"].items():
            for item in entries:
                path = (previous / item["path"]).resolve()
                require(sha256(path) == item["sha256"], "Changed inherited chunk")
                entry = dict(item)
                try:
                    entry["path"] = os.path.relpath(path, self.directory)
                except ValueError:
                    entry["path"] = str(path)
                self.index[kind].append(entry)
        self.rows = manifest["rows"]
        self.last = self.index["physical"][-1]["last_s"]
        self.last_full = self.index["full"][-1]["last_s"]
        self.observed = manifest["observed_saved_state_range"]
        self.audit_max = manifest["sampled_algebraic_rate_identity_max_abs"]
        self.special = manifest["special_times"]

    def add(self, t, y, full_label=None):
        t = float(t)
        y = np.asarray(y)
        T, C = self.model.unpack(y)
        R = float(self.model.geometry.radius(t))
        require(
            np.isfinite(y).all() and np.isfinite(R) and (R > 0) and (C.min() > 0),
            "Nonfinite/nonpositive recorded state; no clipping is permitted",
        )
        require(self.last is None or t >= self.last, "Recorder times are not chronological")
        if full_label:
            self.special[str(t)] = {
                "label": full_label,
                "time_s": t,
                "R_m": R,
                "max_C": float(C.max()),
                "min_C": float(C.min()),
            }
        if self.last is not None and t == self.last:
            if full_label and self.last_full != t:
                self.add_full(t, R, T, C)
            return
        self.last = t
        self.rows += 1
        for name, value in [
            ("minimum_C", float(C.min())),
            ("maximum_C", float(C.max())),
            ("minimum_T_K", float(T.min())),
            ("maximum_T_K", float(T.max())),
        ]:
            self.observed[name] = (min if name.startswith("minimum") else max)(
                self.observed[name], value
            )
        audit = self.model.audit(t, y)
        for k in self.audit_max:
            self.audit_max[k] = max(self.audit_max[k], abs(audit[k]))
        self.add_physical(t, R, T, C)
        if t <= self.full_seconds_cutoff or t % 60.0 == 0.0 or full_label:
            self.add_full(t, R, T, C)
        for kind in self.buffers:
            if len(self.buffers[kind]) >= self.limits[kind]:
                self.flush_kind(kind)

    def add_physical(self, t, R, T, C):
        """记录固定物理半径采样及真实表面值。"""
        valid = self.r <= R
        Ts = np.full(21, np.nan)
        Cs = Ts.copy()
        Ts[valid] = np.interp(self.r[valid] / R, self.model.xi, T)
        Cs[valid] = np.interp(self.r[valid] / R, self.model.xi, C)
        self.buffers["physical"].append(
            (t, R, Ts, Cs, valid, float(T[-1]), float(C[-1]), float(self.w @ C))
        )

    def add_full(self, t, R, T, C):
        """保留全部径向节点；重复时刻的补充记录沿用同一入口。"""
        self.buffers["full"].append((t, R, T.copy(), C.copy(), float(self.w @ C)))
        self.last_full = t

    def flush_kind(self, kind):
        rows = self.buffers[kind]
        if not rows:
            return
        arrays = {
            "t_s": np.asarray([v[0] for v in rows]),
            "R_m": np.asarray([v[1] for v in rows]),
            "T_K": np.asarray([v[2] for v in rows]),
            "C": np.asarray([v[3] for v in rows]),
        }
        if kind == "physical":
            arrays.update(
                valid=np.asarray([v[4] for v in rows]),
                T_surface_K=np.asarray([v[5] for v in rows]),
                C_surface=np.asarray([v[6] for v in rows]),
                mean_C=np.asarray([v[7] for v in rows]),
                r_fixed_m=self.r,
            )
        else:
            arrays.update(
                mean_C=np.asarray([v[4] for v in rows]), xi=self.model.xi, radial_weights=self.w
            )
        name = f"{kind}_{len(self.index[kind]):05d}.npz"
        target = self.directory / name
        require(not target.exists(), "Chunk overwrite is forbidden")
        with target.open("wb") as f:
            np.savez_compressed(f, **arrays)
        self.index[kind].append(
            {
                "path": name,
                "sha256": sha256(target),
                "rows": len(rows),
                "first_s": rows[0][0],
                "last_s": rows[-1][0],
            }
        )
        self.buffers[kind] = []

    def flush(self, final=False):
        for kind in self.buffers:
            self.flush_kind(kind)
        write_json(
            self.directory / "arrays.json",
            {
                "schema": "W6-chunked-fields-v1",
                "complete": bool(final),
                "geometry": self.geometry,
                "index": self.index,
                "coordinate": "xi=r/R(t); T_K absolute kelvin; dry-basis C",
                "r_fixed_m": self.r.tolist(),
                "fixed_radius_validity": "r<=R; true surface also stored separately; exterior NaN",
                "full_field_semantics": "Continuous piecewise-linear reconstruction through every radial node",
                "rows": self.rows,
                "special_times": self.special,
                "observed_saved_state_range": self.observed,
                "sampled_algebraic_rate_identity_max_abs": self.audit_max,
                "time_integrated_balance_claimed": False,
            },
        )


def load_rows(directory, kind):
    directory = Path(directory)
    manifest = read_json(directory / "arrays.json")
    require(manifest["complete"], "Incomplete recorder")
    previous = None
    for item in manifest["index"][kind]:
        p = directory / item["path"]
        require(sha256(p) == item["sha256"], "Changed stored field chunk")
        with np.load(p, allow_pickle=False) as d:
            require(
                len(d["t_s"]) == item["rows"]
                and d["t_s"][0] == item["first_s"]
                and (d["t_s"][-1] == item["last_s"]),
                "Chunk time coverage changed",
            )
            for i, t in enumerate(d["t_s"]):
                require(previous is None or t > previous, "Nonunique or unsorted recorded time")
                previous = float(t)
                row = {
                    k: d[k].copy() if k in ("xi", "radial_weights", "r_fixed_m") else d[k][i].copy()
                    for k in d.files
                }
                yield (float(t), row)


def runtime():
    from scipy.integrate._ivp import bdf, common, ivp
    from scipy.sparse.linalg._dsolve import _superlu

    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "runtime_code_sha256": {
            m.__name__: sha256(m.__file__) for m in (bdf, common, ivp, _superlu)
        },
    }


def choose_report(root, spatial, temporal):
    unit = Decimal(".36")
    r = Decimal.from_float(float(root))
    lower = r + Decimal.from_float(float(spatial)) + Decimal.from_float(float(temporal))
    k = int((lower / unit).to_integral_value(rounding=ROUND_CEILING))
    if Decimal(k) * unit <= r:
        k += 1
    return float(Decimal(k) * unit)


def effect_gate(comparisons, dry, level_balance_flags, report_balance):
    final = [c for c in comparisons if c["kind"] in ("space", "time")]
    return bool(
        len(final) == 2
        and {c["kind"] for c in final} == {"space", "time"}
        and all((c["target_met"] for c in final))
        and dry
        and (len(level_balance_flags) >= 4)
        and all(level_balance_flags)
        and (report_balance is not None)
        and report_balance["passed"]
    )


def diff_full(da, db):
    """在两组径向节点的并集上比较分段线性场。"""
    x = np.union1d(da["xi"], db["xi"])
    dT = np.abs(np.interp(x, da["xi"], da["T_K"]) - np.interp(x, db["xi"], db["T_K"]))
    dC = np.abs(np.interp(x, da["xi"], da["C"]) - np.interp(x, db["xi"], db["C"]))
    return x, dT, dC


def diff_physical(da, db):
    """比较相同有效物理位置及真实表面，完整保留掩码检查。"""
    require(
        np.array_equal(da["valid"], db["valid"])
        and np.array_equal(da["r_fixed_m"], db["r_fixed_m"]),
        "Physical-coordinate mask/positions differ",
    )
    valid = da["valid"]
    x = np.r_[da["r_fixed_m"][valid], da["R_m"]]
    dT = np.r_[
        np.abs(da["T_K"][valid] - db["T_K"][valid]),
        abs(float(da["T_surface_K"]) - float(db["T_surface_K"])),
    ]
    dC = np.r_[
        np.abs(da["C"][valid] - db["C"][valid]),
        abs(float(da["C_surface"]) - float(db["C_surface"])),
    ]
    return x, dT, dC


def compare_pair(a, b, kind):
    ia, ib = (iter(load_rows(a, kind)), iter(load_rows(b, kind)))
    ra = next(ia, None)
    rb = next(ib, None)
    rows = 0
    first = last = None
    maxT = maxC = 0.0
    worstT = worstC = None
    common_times = hashlib.sha256()
    while ra is not None and rb is not None:
        ta, da = ra
        tb, db = rb
        if ta < tb:
            ra = next(ia, None)
            continue
        if tb < ta:
            rb = next(ib, None)
            continue
        require(
            float(da["R_m"]) == float(db["R_m"]),
            "Same-factor comparison has different physical radius",
        )
        x, dT, dC = (diff_full if kind == "full" else diff_physical)(da, db)
        if float(dT.max()) > maxT:
            maxT = float(dT.max())
            worstT = {"t_s": ta, "coordinate": float(x[dT.argmax()])}
        if float(dC.max()) > maxC:
            maxC = float(dC.max())
            worstC = {"t_s": ta, "coordinate": float(x[dC.argmax()])}
        rows += 1
        first = ta if first is None else first
        last = ta
        common_times.update(np.float64(ta).tobytes())
        ra = next(ia, None)
        rb = next(ib, None)
    require(rows > 0 and first == 0.0, "No common initial field")
    return {
        "rows": rows,
        "first_s": first,
        "last_s": last,
        "common_time_sha256": common_times.hexdigest(),
        "max_T_K": maxT,
        "max_C": maxC,
        "worst_T": worstT,
        "worst_C": worstC,
        "coordinate": "xi" if kind == "full" else "physical radius m",
        "scope": "Exact common saved times; PL union-knot maximum is complete in space at those times only. No continuous-time/PDE bound.",
    }


@lru_cache(maxsize=2)
def quadrature(order):
    x, w = leggauss(order)
    x.setflags(write=False)
    w.setflags(write=False)
    return (x, w)


def polynomial_derivative(poly, tau):
    """Differentiate actual BDF Newton products; never call the RHS for Tdot."""
    tau = np.asarray(tau)
    products = np.ones(tau.shape)
    derivatives = np.zeros(tau.shape)
    dy = np.zeros((poly.D.shape[1],) + tau.shape)
    for degree in range(poly.order):
        x = (tau - poly.t_shift[degree]) / poly.denom[degree]
        derivatives = derivatives * x + products / poly.denom[degree]
        products = products * x
        dy += poly.D[degree + 1, :, None] * derivatives[None, :]
    return dy


def interval_integrals(model, poly, start, lo, hi, right_side, order):
    x, w = quadrature(order)
    width = (hi - lo) / 2
    tau = (lo + hi) / 2 + width * x
    t = start + tau
    y = poly(tau)
    n = model.nr
    theta = y[:n].T
    C = y[n:].T
    T = theta + model.temperature_reference_K
    R0 = model.grid.radius_m
    weights = 2 * model.grid.volume_weights_m2 / R0**2
    R = np.asarray(model.geometry.radius(t))
    J = (R / R0) ** 2
    ambient = model.ambient.values(t, right_of_breakpoint=right_side)
    mult = np.asarray(model.moisture_boundary_multiplier(t, R, R0))
    require(
        np.all(mult == 1.0),
        "B1 signed latent/current-dry-density closure requires boundary multiplier exactly one",
    )
    moisture = 2 * model.h_m_m_s * mult / R * (C[:, -1] - ambient.moisture_eq)
    conv = (
        J
        * 2
        * model.h_W_m2_K
        / R
        * (theta[:, -1] - (ambient.temperature_K - model.temperature_reference_K))
    )
    if model.latent_scale == 0:
        latent = np.zeros_like(conv)
    else:
        L, _ = model.table.values(T[:, -1])
        jw = model.rhod0 / J * model.h_m_m_s * (C[:, -1] - ambient.moisture_eq)
        latent = J * 2 / R * (model.latent_scale * L * jw)
    heat = conv + latent
    dy = polynomial_derivative(poly, tau)
    rho, cp, _, _ = model.properties.evaluate(C, T)
    volume = J * (rho * cp * dy[:n].T @ weights)
    result = width * (
        np.asarray(
            [moisture, heat, np.abs(heat), volume, conv, latent, np.abs(conv) + np.abs(latent)]
        )
        @ w
    )
    require(np.isfinite(result).all(), "Nonfinite accepted-step balance integral")
    return result


def surface_power(poly, index, left, right, offset=0.0):
    """Independent normalized power reconstruction of the actual Newton field."""
    q = Polynomial([float(poly.D[0, index]) + offset])
    prod = Polynomial([1.0])
    for k in range(poly.order):
        prod *= Polynomial(
            [(left - poly.t_shift[k]) / poly.denom[k], (right - left) / poly.denom[k]]
        )
        q += float(poly.D[k + 1, index]) * prod
    return q.trim()


def polynomial_roots_unit(q):
    """All numerical roots via recursive derivative monotonic partitions.

    This is a checked floating-point quadrature partition, not an outward root
    certificate. Flat zero intervals need no split. Tangencies are also kept.
    """
    q = q.trim()
    degree = q.degree()
    if degree < 1:
        return []
    stationary = polynomial_roots_unit(q.deriv()) if degree > 1 else []
    points = np.unique(np.r_[0.0, stationary, 1.0])
    scale = max(1.0, float(np.abs(q.coef).sum()))
    near = 128 * np.finfo(float).eps * scale
    roots = []
    for x in points:
        if abs(float(q(x))) <= near:
            roots.append(float(x))
    for a, b in zip(points[:-1], points[1:]):
        fa, fb = (float(q(a)), float(q(b)))
        if fa * fb < 0:
            roots.append(
                float(brentq(q, float(a), float(b), xtol=5e-324, rtol=4 * np.finfo(float).eps))
            )
    roots = sorted(set((x for x in roots if 0.0 < x < 1.0)))
    require(
        all((abs(float(q(x))) <= 4 * near for x in roots)),
        "Unresolved polynomial crossing residual",
    )
    return roots


def accepted_partition(model, poly, start, left, right, right_side):
    """Split every table-knot crossing and every absolute-flux sign change.

    Geometry and ambient must be affine on the actual unchanged parent span.
    Original environmental/radius breakpoints are therefore never crossed here.
    """
    if right <= left:
        return ([left, right], {"table_crossings": 0, "absolute_flux_crossings": 0})
    Ts = surface_power(poly, model.nr - 1, left, right, model.temperature_reference_K)
    C = surface_power(poly, 2 * model.nr - 1, left, right)
    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    tau = left + (right - left) * u
    actual = poly(tau)[model.nr - 1] + model.temperature_reference_K
    tolerance = 512 * np.finfo(float).eps * max(1.0, float(np.abs(actual).max()))
    require(
        np.max(abs(Ts(u) - actual)) <= tolerance, "Surface Newton/power reconstruction mismatch"
    )
    extrema = np.r_[0.0, polynomial_roots_unit(Ts.deriv()), 1.0]
    range_T = Ts(extrema)
    if model.latent_scale != 0:
        model.table.values(range_T)
    t = start + tau
    R = np.asarray(model.geometry.radius(t))
    a = model.ambient.values(t, right_of_breakpoint=right_side)

    def affine(values, name):
        values = np.asarray(values, dtype=float) + np.zeros_like(u)
        p = Polynomial([values[0], values[-1] - values[0]])
        tol = 512 * np.finfo(float).eps * max(1.0, float(abs(values).max()))
        require(np.max(abs(p(u) - values)) <= tol, "Audit interval crosses non-affine " + name)
        return p

    Rp = affine(R, "radius")
    Ta = affine(a.temperature_K, "temperature input")
    Ce = affine(a.moisture_eq, "moisture input")
    cuts = [0.0, 1.0]
    crossings = 0
    if model.latent_scale != 0:
        for knot in model.table.temperature_K:
            if min(range_T) <= knot <= max(range_T):
                roots = polynomial_roots_unit(Ts - float(knot))
                cuts += roots
                crossings += len(roots)
    tablecuts = np.unique(cuts)
    signcuts = []
    for a0, b0 in zip(tablecuts[:-1], tablecuts[1:]):
        mid = (a0 + b0) / 2
        L = Polynomial([0.0])
        if model.latent_scale != 0:
            lm, lp = model.table.values(float(Ts(mid)))
            L = model.latent_scale * (float(lm) + float(lp) * (Ts - float(Ts(mid))))
        conv = 2 * model.h_W_m2_K * Rp * Rp / model.grid.radius_m**2 * (Ts - Ta)
        latent = 2 * model.rhod0 * model.h_m_m_s * L * (C - Ce)
        for flux in (conv + latent, conv, latent):
            signcuts.extend((x for x in polynomial_roots_unit(flux) if a0 < x < b0))
    allcuts = np.unique(np.r_[tablecuts, signcuts])
    physical = left + (right - left) * allcuts
    physical[0] = left
    physical[-1] = right
    return (
        np.unique(physical),
        {
            "table_crossings": crossings,
            "absolute_flux_crossings": len(signcuts),
            "surface_polynomial_min_T_K": float(min(range_T)),
            "surface_polynomial_max_T_K": float(max(range_T)),
            "surface_power_max_difference_K": float(np.max(abs(Ts(u) - actual))),
        },
    )


class BalanceAudit:

    def __init__(self, B0, initial=None, on_used_interval=None):
        require(np.isfinite(B0) and B0 > 0, "Invalid original-initial-state heat scale B0")
        self.B0 = float(B0)
        self.initial = None
        self.acc = np.zeros((2, 7))
        self.compensation = np.zeros((2, 7))
        self.partition = {
            "table_crossings": 0,
            "absolute_flux_crossings": 0,
            "accepted_steps_partitioned": 0,
            "surface_power_max_difference_K": 0.0,
        }
        self.intervals = 0
        self.accepted_steps = 0
        self.pending = None
        self.last_s = None
        self.checkpoints = {}
        self.segment_records = []
        self.maxima = {
            "moisture_residual_abs": {"value": 0.0, "t_s": None},
            "heat_residual_B0_K": {"value": 0.0, "t_s": None},
            "quadrature_moisture_abs": {"value": 0.0, "t_s": None},
            "quadrature_heat_B0_K": {"value": 0.0, "t_s": None},
        }
        self.trace = hashlib.sha256()
        self.trace_steps = 0
        self.on_used_interval = on_used_interval
        if initial is not None:
            self.restore(initial)

    def restore(self, state):
        require(
            state["schema"] == "B1-coupled-accepted-step-balance-v1"
            and state["B0_J_m3_K"] == self.B0
            and (state["limits"] == LIMITS),
            "Changed balance continuation definition",
        )
        require(state["window_completed"], "Cannot continue incomplete balance observer")
        self.initial = state["initial"]
        self.acc = np.asarray(state["accumulators"], dtype=float)
        self.compensation = np.asarray(state["compensation"], dtype=float)
        self.intervals = state["subintervals"]
        self.accepted_steps = state["accepted_intervals"]
        self.last_s = state["last_s"]
        self.checkpoints = state["checkpoints"]
        self.segment_records = state["segments"]
        self.maxima = state["maxima"]
        self.partition = state["partition"]

    def begin(self, model, start, y):
        w = 2 * model.grid.volume_weights_m2 / model.grid.radius_m**2
        if self.initial is None:
            self.initial = {
                "time_s": float(start),
                "mean_C": float(w @ model.unpack(y)[1]),
                "state_sha256": hashlib.sha256(
                    np.asarray(y, dtype=np.float64).tobytes()
                ).hexdigest(),
            }
            self.last_s = float(start)
            self.checkpoints[str(float(start))] = {
                "label": "initial",
                **self.diagnostics(start, self.initial["mean_C"]),
            }
        require(
            self.pending is None and self.last_s == float(start),
            "Noncontiguous audit span or uncommitted accepted step",
        )

    def hold(self, model, poly, left, right, context):
        require(self.pending is None, "Previous accepted-step audit was not committed")
        self.pending = (model, poly, float(left), float(right), context)

    def note_trace(self, solver, poly, context):
        for value in [
            np.asarray([context["start"], solver.t, poly.t_old, poly.t], dtype=np.float64),
            np.asarray(solver.y, dtype=np.float64),
            np.asarray([poly.order], dtype=np.int64),
            np.asarray(poly.t_shift, dtype=np.float64),
            np.asarray(poly.denom, dtype=np.float64),
            np.asarray(poly.D, dtype=np.float64),
        ]:
            self.trace.update(value.tobytes())
        self.trace_steps += 1

    def diagnostics(self, time_s, mean_C):
        rows = []
        for order, total in zip(ORDERS, self.acc):
            residual_C = mean_C - self.initial["mean_C"] + total[0]
            residual_h = total[1] + total[3]
            rows.append(
                {
                    "order": order,
                    **dict(zip(COMPONENTS, total.tolist())),
                    "moisture_residual_abs": abs(float(residual_C)),
                    "heat_residual_abs_J_m3_ref": abs(float(residual_h)),
                    "heat_residual_B0_K": abs(float(residual_h)) / self.B0,
                    "heat_residual_relative": abs(float(residual_h)) / max(float(total[2]), 1e-30),
                }
            )
        diff = np.abs(self.acc[1] - self.acc[0])
        out = {
            "t_s": float(time_s),
            "mean_C": float(mean_C),
            "quadrature": rows,
            "quadrature_component_difference": dict(zip(COMPONENTS, diff.tolist())),
            "quadrature_moisture_abs": float(diff[0]),
            "quadrature_heat_B0_K": float(diff[1:].max()) / self.B0,
        }
        for key, value in [
            ("moisture_residual_abs", max((r["moisture_residual_abs"] for r in rows))),
            ("heat_residual_B0_K", max((r["heat_residual_B0_K"] for r in rows))),
            ("quadrature_moisture_abs", out["quadrature_moisture_abs"]),
            ("quadrature_heat_B0_K", out["quadrature_heat_B0_K"]),
        ]:
            if value > self.maxima[key]["value"]:
                self.maxima[key] = {"value": float(value), "t_s": float(time_s)}
        out["passed"] = bool(
            all(
                (
                    r["moisture_residual_abs"] < LIMITS["moisture_absolute"]
                    and r["heat_residual_relative"] < LIMITS["heat_relative"]
                    and (r["heat_residual_B0_K"] < LIMITS["heat_B0_equivalent_K"])
                    for r in rows
                )
            )
            and out["quadrature_moisture_abs"] < LIMITS["quadrature_moisture_absolute"]
            and (out["quadrature_heat_B0_K"] < LIMITS["quadrature_heat_B0_equivalent_K"])
        )
        return out

    def commit(self, cut=None, checkpoint_label=None):
        if self.pending is None:
            return
        model, poly, left, right, context = self.pending
        start = context["start"]
        if cut is not None:
            right = min(right, float(cut) - start)
        require(right >= left, "Audit event cut precedes accepted-step start")
        absolute_left = start + left
        absolute_right = start + right
        require(self.last_s == absolute_left, "Accepted-step audit has a gap or overlap")
        if self.on_used_interval is not None:
            self.on_used_interval(
                model=model,
                polynomial=poly,
                left_tau=left,
                right_tau=right,
                physical_origin=start,
                right_of_breakpoint=context["right_side"],
                event_detection_enabled=context["detect"],
                event_cut_applied=cut is not None and right < float(self.pending[3]),
            )
        query = np.asarray(context.get("query_times", []), dtype=float)
        query = query[(query > absolute_left) & (query <= absolute_right)]
        try:
            model_cuts, partition = accepted_partition(
                model, poly, start, left, right, context["right_side"]
            )
        except ValueError as error:
            error.failure_context = {
                "phase": "accepted BDF polynomial quadrature partition; includes interior extrema",
                "physical_interval_s": [absolute_left, absolute_right],
                "right_of_breakpoint": context["right_side"],
                "physical_freezing_established": False,
                "underlying_error": str(error),
            }
            raise
        for name in ("table_crossings", "absolute_flux_crossings"):
            self.partition[name] += partition[name]
        self.partition["accepted_steps_partitioned"] += 1
        self.partition["surface_power_max_difference_K"] = max(
            self.partition["surface_power_max_difference_K"],
            partition.get("surface_power_max_difference_K", 0.0),
        )
        cuts = np.unique(np.r_[model_cuts, query - start])
        w = 2 * model.grid.volume_weights_m2 / model.grid.radius_m**2
        if right == left:
            current = poly(np.asarray([right]))[:, 0]
            diag = self.diagnostics(absolute_right, float(w @ model.unpack(current)[1]))
        for a, b in zip(cuts[:-1], cuts[1:]):
            values = np.asarray(
                [
                    interval_integrals(model, poly, start, a, b, context["right_side"], order)
                    for order in ORDERS
                ]
            )
            adjusted = values - self.compensation
            new = self.acc + adjusted
            self.compensation = new - self.acc - adjusted
            self.acc = new
            self.intervals += 1
            t = start + b
            current = poly(np.asarray([b]))[:, 0]
            diag = self.diagnostics(t, float(w @ model.unpack(current)[1]))
            if t in query:
                self.checkpoints[str(float(t))] = {"label": "exact_post_event_query", **diag}
        self.last_s = absolute_right
        self.accepted_steps += 1
        if checkpoint_label:
            self.checkpoints[str(absolute_right)] = {"label": checkpoint_label, **diag}
        self.pending = None

    def finish_span(self, start, requested_end, end, event):
        self.commit(end, "equality_root" if event else "segment_end")
        self.segment_records.append(
            {
                "start_s": float(start),
                "requested_end_s": float(requested_end),
                "used_end_s": float(end),
                "event": bool(event),
                "right_of_breakpoint": bool(self._last_context["right_side"]),
                "query_count": len(self._last_context["query_times"]),
            }
        )

    def state(self, window_completed=True):
        require(self.pending is None, "Cannot close observer with an uncommitted step")
        absolute_pass = bool(
            self.maxima["moisture_residual_abs"]["value"] < LIMITS["moisture_absolute"]
            and self.maxima["heat_residual_B0_K"]["value"] < LIMITS["heat_B0_equivalent_K"]
            and (
                self.maxima["quadrature_moisture_abs"]["value"]
                < LIMITS["quadrature_moisture_absolute"]
            )
            and (
                self.maxima["quadrature_heat_B0_K"]["value"]
                < LIMITS["quadrature_heat_B0_equivalent_K"]
            )
        )
        last = self.checkpoints.get(str(self.last_s))
        last_pass = bool(last is not None and last["passed"])
        checkpoints_pass = bool(
            self.checkpoints and all((x["passed"] for x in self.checkpoints.values()))
        )
        return {
            "schema": "B1-coupled-accepted-step-balance-v1",
            "window_completed": bool(window_completed),
            "passed": bool(window_completed and absolute_pass and last_pass and checkpoints_pass),
            "B0_J_m3_K": self.B0,
            "limits": LIMITS,
            "initial": self.initial,
            "last_s": self.last_s,
            "accumulators": self.acc.tolist(),
            "compensation": self.compensation.tolist(),
            "subintervals": self.intervals,
            "accepted_intervals": self.accepted_steps,
            "maxima": self.maxima,
            "checkpoints": self.checkpoints,
            "segments": self.segment_records,
            "partition": self.partition,
            "absolute_cumulative_checks_passed": absolute_pass,
            "last_checkpoint_checks_passed": last_pass,
            "all_saved_checkpoint_checks_passed": checkpoints_pass,
            "new_window_trace_steps": self.trace_steps,
            "new_window_accepted_state_dense_trace_sha256": self.trace.hexdigest(),
            "scope": "C balances fixed dry-reference mean. Heat is integral J*b(C)*Tdot plus J times signed (convective+latent) boundary heat per current volume, normalized to original volume. Not d(J*b*T)/dt, total mixture enthalpy or a physical-temperature error bound.",
            "relative_heat_denominator": "Integral abs(signed convective+latent net heat rate), not abs(integrated net), and not the separately stored larger gross absolute-channel integral.",
            "partition_scope": "Floating-point roots of actual BDF surface polynomial at every pure-water table knot and each signed flux zero. Checked residuals/extrema; not an outward interval/PDE certificate.",
            "sampling_scope": "Cumulative absolute/8-versus16 differences at every used accepted-step end and saved post-event query. Relative heat criterion at requested checkpoint; no continuous-time supremum certification.",
            "continuous_PDE_certified": False,
        }


def verify_coverage(directory, record, plan):
    """Independent exact schedule checks, including when both comparisons omit a row."""
    root = record["root_s"]
    end = record.get("saved_end_s", record["post_event_end_s"])
    job = record["job"]
    require(np.isfinite(end) and end > 0, "Invalid saved-window end")
    if root is not None:
        require(
            np.isfinite(root) and 0 <= root and (end >= root + plan["post_event_window_s"]),
            "Invalid root/post-event interval",
        )
    require(job in plan["jobs"], "Case not in frozen matrix")
    ordinary = expected_times(job["scenario"]["geometry"], end, plan["physical_seconds_cutoff"])
    windows = np.asarray(record.get("window_boundaries_s", [end]))
    special = np.r_[windows, root, report_ticks(root, end)] if root is not None else windows
    expected = np.unique(np.r_[0.0, ordinary, special, end])
    expected_full = expected[
        (expected <= plan["full_seconds_cutoff"])
        | (expected % 60.0 == 0.0)
        | np.isin(expected, special)
    ]
    if root is not None:
        expected_full = np.union1d(expected_full, expected[expected >= root])
    counts = {}
    for kind, wanted in [("physical", expected), ("full", expected_full)]:
        n = 0
        h = hashlib.sha256()
        N = job["N"]
        for t, d in load_rows(directory, kind):
            require(
                n < len(wanted) and t == wanted[n],
                "Missing/extra/wrong exact " + kind + " time at row " + str(n),
            )
            require(np.isfinite(d["R_m"]) and d["R_m"] > 0, "Invalid saved physical radius")
            if kind == "full":
                require(
                    d["T_K"].shape == d["C"].shape == d["xi"].shape == (N + 1,)
                    and np.all(np.diff(d["xi"]) > 0)
                    and (d["xi"][0] == 0.0)
                    and (d["xi"][-1] == 1.0),
                    "Wrong full-grid shape/coordinates",
                )
                require(
                    np.isfinite(d["T_K"]).all()
                    and np.isfinite(d["C"]).all()
                    and (np.min(d["C"]) > 0),
                    "Nonfinite/nonpositive full field",
                )
                require(
                    hashlib.sha256(np.asarray(d["xi"], dtype=np.float64).tobytes()).hexdigest()
                    == record["xi_sha256"],
                    "Changed radial grid",
                )
                if t == 0.0:
                    require(
                        np.all(d["T_K"] == plan["initial_state"]["temperature_K"])
                        and np.all(d["C"] == plan["initial_state"]["moisture_C"]),
                        "Saved initial field differs from original uniform initial condition",
                    )
            else:
                valid = d["valid"]
                require(
                    valid.shape == (21,) and np.array_equal(valid, d["r_fixed_m"] <= d["R_m"]),
                    "Invalid physical mask",
                )
                for field in ("T_K", "C"):
                    require(
                        np.isfinite(d[field][valid]).all() and np.isnan(d[field][~valid]).all(),
                        "Wrong finite/exterior physical samples",
                    )
                require(
                    np.isfinite(d["T_surface_K"])
                    and np.isfinite(d["C_surface"])
                    and (d["C_surface"] > 0),
                    "Missing surface",
                )
            h.update(np.float64(t).tobytes())
            n += 1
        require(n == len(wanted), "Incomplete " + kind + " sampling coverage")
        counts[kind] = {
            "rows": n,
            "time_sha256": h.hexdigest(),
            "first_s": float(wanted[0]),
            "last_s": float(wanted[-1]),
        }
    return {"passed": True, "kinds": counts, "time_interpolation_used": False}


def window_record(directory, plan):
    directory = Path(directory)
    status = read_json(directory / "run_status.json")
    record = read_json(directory / "record.json")
    require(
        status["status"] in ("complete", "window_complete_event_pending")
        and status["record_sha256"] == sha256(directory / "record.json"),
        "Missing/stale window receipt",
    )
    require(
        record["window_completed"]
        and record["source_bindings"] == plan["source_bindings"]
        and (record["arrays_manifest_sha256"] == sha256(directory / "arrays.json")),
        "Incomplete/unbound window",
    )
    require(
        record["event_pending"] == (record["root_s"] is None)
        and record["completed"] == (record["root_s"] is not None)
        and (status["completed"] == record["completed"]),
        "Inconsistent pending/completed/event flags",
    )
    require(
        status["status"]
        == ("complete" if record["completed"] else "window_complete_event_pending"),
        "Wrong window status",
    )
    require(
        record["continuation_state_sha256"] == sha256(directory / "continuation_state.npz"),
        "Changed continuation state/accepted times",
    )
    require(
        record["balance_state_sha256"] == sha256(directory / "balance_state.json"),
        "Changed balance observer state",
    )
    audited = read_json(directory / "balance_state.json")
    require(
        audited["window_completed"]
        and audited["passed"] == record["balance_passed"]
        and (audited["last_s"] == record["saved_end_s"]),
        "Inconsistent balance completion/extent",
    )
    require(
        sha256(directory / "continuous_event_state.json") == record["continuous_event_state_sha256"]
        and sha256(directory / "continuous_event_verification.json")
        == record["continuous_event_verification_sha256"],
        "Changed first-event evidence",
    )
    require(
        audited["accepted_intervals"] == record["accepted_steps_including_post"],
        "Accepted-step balance count does not cover all actual accepted steps",
    )
    guard = read_json(directory / "continuous_event_state.json")
    old_verification = read_json(directory / "continuous_event_verification.json")
    expected_intervals = (
        record["accepted_steps_to_event"]
        if record["root_s"] is not None
        else record["accepted_steps_including_post"]
    )
    require(
        guard["window_completed"] and guard["intervals"] == expected_intervals,
        "Event witness does not cover every pre-event accepted step",
    )
    from 事件覆盖 import verify as verify_event

    actual_verification = verify_event(directory)
    require(
        actual_verification["source_state_sha256"] == record["continuous_event_state_sha256"]
        and actual_verification["verifier_sha256"]
        == plan["source_bindings"]["事件覆盖.py"]["sha256"],
        "Actual event replay is not bound to the executed state/verifier",
    )
    for key in ("wet_prefix_passed", "first_computed_crossing_supported"):
        require(
            actual_verification[key] == old_verification[key] == guard[key] == record[key],
            "Event replay/stored state/record flags disagree: " + key,
        )
    require(
        old_verification["source_state_sha256"] == actual_verification["source_state_sha256"]
        and old_verification["verifier_sha256"] == actual_verification["verifier_sha256"],
        "Stored independent event verification identity differs",
    )
    require(
        Fraction(actual_verification["first_audited_physical_s_exact"]) == 0,
        "Original-initial-state scenario event witness must start at physical time zero",
    )
    expected_end = record["root_s"] if record["root_s"] is not None else record["saved_end_s"]
    require(
        Fraction(actual_verification["last_used_physical_s_exact"])
        == Fraction.from_float(float(expected_end)),
        "Event witness physical endpoint differs from scenario record",
    )
    if record["root_s"] is not None:
        require(
            actual_verification["root_bracket"] is not None
            and actual_verification["root_bracket"]["parent_root_s"] == record["root_s"],
            "Scenario equality root differs from actual event polynomial",
        )
    require(
        record["runtime"] == runtime(),
        "The comparison runtime differs from the executed-case runtime",
    )
    coverage = verify_coverage(directory, record, plan)
    require(coverage == record["output_coverage"], "Case coverage changed after execution")
    return record


def case_record(directory, plan):
    record = window_record(directory, plan)
    status = read_json(Path(directory) / "run_status.json")
    require(
        status["status"] == "complete"
        and status["completed"]
        and record["completed"]
        and (not record["event_pending"]),
        "An event-pending window is not a completed scenario",
    )
    return record


def run_model(plan, job, model, directory, previous=None, until_s=None):
    """Common complete-field writer and resumable, fully observed numerical window."""
    from 事件覆盖 import FirstCrossingObserver, verify as verify_event

    directory = fresh(directory)
    previous = Path(previous).resolve() if previous is not None else None
    prior = window_record(previous, plan) if previous is not None else None
    require(job in plan["jobs"], "Job is not in this exact plan")
    if prior is not None:
        require(prior["job"] == job, "Continuation changes factors, grid or tolerances")
    initial = plan["initial_state"]
    rho, cp, _, _ = model.properties.evaluate(
        np.array([initial["moisture_C"]]), np.array([initial["temperature_K"]])
    )
    guard = FirstCrossingObserver(
        directory, read_json(previous / "continuous_event_state.json") if prior else None
    )
    observer = BalanceAudit(
        float((rho * cp)[0]), read_json(previous / "balance_state.json") if prior else None, guard
    )
    recorder = Recorder(
        model,
        directory,
        job["scenario"]["geometry"],
        full_seconds_cutoff=plan["full_seconds_cutoff"],
    )
    solver = Solver(model, job["settings"], recorder, observer, guard)
    start, windows, lineage = 0.0, [], []
    if prior is None:
        y = model.initial_state(initial["temperature_K"], initial["moisture_C"])
        recorder.add(0.0, y, "initial")
    else:
        start = prior["saved_end_s"]
        recorder.inherit(previous)
        windows, lineage = list(prior["window_boundaries_s"]), list(prior["continuation_lineage"])
        with np.load(previous / "continuation_state.npz", allow_pickle=False) as saved:
            require(float(saved["t_s"]) == start, "Continuation time differs")
            y = saved["y"].copy()
            old_steps = saved["accepted_step_times_s"]
            solver.step_hash.update(np.asarray(old_steps, dtype=np.float64).tobytes())
            solver.step_times = old_steps.tolist()
            require(
                solver.step_hash.hexdigest() == prior["accepted_step_time_sha256"],
                "Inherited accepted times changed",
            )
            if prior["root_s"] is not None:
                solver.event_time, solver.event_state = prior["root_s"], saved["event_state"].copy()
                solver.event_steps = prior["accepted_steps_to_event"]
        for key, field in [
            ("step_count", "accepted_steps_including_post"),
            ("nfev", "nfev"),
            ("njev", "njev"),
            ("nlu", "nlu"),
        ]:
            setattr(solver, key, prior[field])
        solver.segments = list(prior["segments"])
        lineage.append(
            dict(
                directory=str(previous),
                record_sha256=sha256(previous / "record.json"),
                inherited_through_s=start,
                reuse_scope="Saved state and immutable output prefix",
            )
        )
    horizon = plan["horizon_s"] if until_s is None else float(until_s)
    try:
        y, end = advance_window(
            solver,
            y,
            start,
            horizon,
            plan["post_event_window_s"],
            job["scenario"]["geometry"],
            plan["physical_seconds_cutoff"],
        )
        recorder.flush(final=True)
        require(
            identity() == plan["source_bindings"], "A source or input changed during computation"
        )
        np.savez_compressed(
            directory / "continuation_state.npz",
            t_s=np.float64(end),
            y=y,
            accepted_step_times_s=np.asarray(solver.step_times),
            event_state=solver.event_state if solver.event_time is not None else np.array([]),
        )
        balance = observer.state()
        write_json(directory / "balance_state.json", balance)
        write_json(directory / "continuous_event_state.json", guard.state())
        verification = verify_event(directory)
        write_json(directory / "continuous_event_verification.json", verification)
        root = solver.event_time
        record = dict(
            completed=root is not None,
            window_completed=True,
            event_pending=root is None,
            job=job,
            root_s=root,
            saved_end_s=end,
            post_event_end_s=end if root is not None else None,
            window_boundaries_s=windows + [end],
            continuation_lineage=lineage,
            N=job["N"],
            xi_sha256=hashlib.sha256(np.asarray(model.xi, dtype=np.float64).tobytes()).hexdigest(),
            accepted_steps_to_event=solver.event_steps,
            accepted_steps_including_post=solver.step_count,
            accepted_step_time_sha256=solver.step_hash.hexdigest(),
            nfev=solver.nfev,
            njev=solver.njev,
            nlu=solver.nlu,
            segments=solver.segments,
            source_bindings=plan["source_bindings"],
            runtime=runtime(),
            balance_passed=balance["passed"],
            first_computed_crossing_supported=verification["first_computed_crossing_supported"],
            wet_prefix_passed=verification["wet_prefix_passed"],
            effects_accepted=False,
            continuous_PDE_certified=False,
            observed_saved_state_range=recorder.observed,
            wall_seconds=perf_counter() - solver.started,
            test_only=plan.get("test_only", False),
        )
        for key, filename in [
            ("arrays_manifest", "arrays.json"),
            ("continuation_state", "continuation_state.npz"),
            ("balance_state", "balance_state.json"),
            ("continuous_event_state", "continuous_event_state.json"),
            ("continuous_event_verification", "continuous_event_verification.json"),
        ]:
            record[key + "_sha256"] = sha256(directory / filename)
        record["output_coverage"] = verify_coverage(directory, record, plan)
        write_json(directory / "record.json", record)
        write_json(
            directory / "run_status.json",
            dict(
                status="complete" if record["completed"] else "window_complete_event_pending",
                completed=record["completed"],
                record_sha256=sha256(directory / "record.json"),
            ),
        )
        return record
    except BaseException as error:
        write_json(
            directory / "run_status.json",
            dict(
                status="failed",
                completed=False,
                error=str(error),
                accepted_steps=solver.step_count,
                committed_through_s=observer.last_s,
            ),
        )
        raise


def summarize_matrix(plan, runs, case_map=None):
    """Require every planned grid/tolerance level and an exact dry-side report sample."""
    require(not plan.get("test_only"), "Short fixtures cannot supply production results")
    require(identity() == plan["source_bindings"], "Stale summary source/input identity")
    directories = {job["id"]: Path(runs) / job["id"] for job in plan["jobs"]}
    selected = read_json(case_map) if isinstance(case_map, (str, Path)) else case_map
    for key, item in (selected or {}).items():
        require(key in directories, "Unknown selected job: " + key)
        directories[key] = Path(item["directory"]).resolve()
        require(
            sha256(directories[key] / "record.json") == item["record_sha256"],
            "Changed selected continuation record: " + key,
        )
    records, receipts, missing = {}, {}, []
    for job in plan["jobs"]:
        key, directory = job["id"], directories[job["id"]]
        if not (directory / "record.json").is_file():
            missing.append(dict(job=key, reason="No completed window receipt"))
            continue
        receipts[key] = dict(
            directory=str(directory.resolve()), record_sha256=sha256(directory / "record.json")
        )
        record = window_record(directory, plan)
        require(
            record["job"] == job and not record["test_only"],
            "Selected case differs from the entire formal job: " + key,
        )
        records[key] = record
        if record["event_pending"]:
            missing.append(
                dict(
                    job=key,
                    reason="Event pending; continue the same settings in a new directory",
                    saved_end_s=record["saved_end_s"],
                )
            )
    if missing:
        require(identity() == plan["source_bindings"], "Source/input changed during summary")
        for key, receipt in receipts.items():
            require(
                sha256(directories[key] / "record.json") == receipt["record_sha256"],
                "Selected record changed during summary: " + key,
            )
        return dict(
            completed=False,
            effects_accepted=False,
            missing_or_pending=missing,
            case_records=receipts,
            source_bindings=plan["source_bindings"],
            continuous_PDE_certified=False,
        )
    rows = []
    for scenario in plan["scenarios"]:
        levels = {
            (j["N"], j["level"]): records[j["id"]]
            for j in plan["jobs"]
            if j["scenario"] == scenario
        }
        main = sorted(N for N, level in levels if level == "main")
        fine, coarse = main[-1], main[-2]
        require((fine, "tight") in levels, "Missing tight solve at the finest grid")
        comparisons = []
        pairs = [
            ((a, "main"), (b, "main"), "space" if b == fine else "space_aux")
            for a, b in zip(main[:-1], main[1:])
        ]
        pairs += [
            ((N, "main"), (N, "tight"), "time" if N == fine else "time_aux")
            for N in main
            if (N, "tight") in levels
        ]
        for ka, kb, kind in pairs:
            a, b = levels[ka], levels[kb]
            full = compare_pair(directories[a["job"]["id"]], directories[b["job"]["id"]], "full")
            physical = compare_pair(
                directories[a["job"]["id"]], directories[b["job"]["id"]], "physical"
            )
            T, C = max(full["max_T_K"], physical["max_T_K"]), max(full["max_C"], physical["max_C"])
            limit = plan["acceptance"]["temporal" if kind.startswith("time") else "spatial"]
            comparisons.append(
                dict(
                    kind=kind,
                    max_T_K=T,
                    max_C=C,
                    target_met=T <= limit["T_K"] and C <= limit["C"],
                    a=a["job"]["id"],
                    b=b["job"]["id"],
                    root_difference_s=b["root_s"] - a["root_s"],
                    full=full,
                    physical=physical,
                )
            )
        tight = levels[fine, "tight"]
        space = abs(levels[fine, "main"]["root_s"] - levels[coarse, "main"]["root_s"])
        time = abs(tight["root_s"] - levels[fine, "main"]["root_s"])
        tick = choose_report(tight["root_s"], space, time)
        directory = directories[tight["job"]["id"]]
        field = next((row for t, row in load_rows(directory, "full") if t == tick), None)
        dry = field is not None and float(field["C"].max()) < 0.15
        balance = read_json(directory / "balance_state.json")["checkpoints"].get(str(tick))
        accepted = effect_gate(
            comparisons, dry, [r["balance_passed"] for r in levels.values()], balance
        )
        accepted = accepted and all(r["first_computed_crossing_supported"] for r in levels.values())
        rows.append(
            dict(
                scenario=scenario,
                equality_root_s=tight["root_s"],
                spatial_root_difference_s=space,
                temporal_root_difference_s=time,
                empirical_margin_s=space + time,
                reported_time_s=tick,
                reported_max_C=float(field["C"].max()) if field is not None else None,
                exact_query_available=field is not None,
                reported_balance=balance,
                all_level_first_computed_crossings_supported=all(
                    r["first_computed_crossing_supported"] for r in levels.values()
                ),
                strict_dry=bool(dry),
                comparisons=comparisons,
                accepted_for_effects=bool(accepted),
            )
        )
    require(identity() == plan["source_bindings"], "Source/input changed during summary")
    for key, receipt in receipts.items():
        require(
            sha256(directories[key] / "record.json") == receipt["record_sha256"],
            "Selected record changed during summary: " + key,
        )
    return dict(
        completed=True,
        effects_accepted=all(r["accepted_for_effects"] for r in rows),
        scenarios=rows,
        continuous_PDE_certified=False,
        source_bindings=plan["source_bindings"],
        case_records=receipts,
    )


def sample_run(model, out, observed=True, start=0.0, end=1.0, initial=None, late=False):
    """Small verification trajectory; state/time/dense hashes support independent regressions."""
    from 事件覆盖 import FirstCrossingObserver, verify as verify_event

    out = fresh(out)
    rows = []
    recorder = SimpleNamespace(
        add=lambda t, y, full_label=None: rows.append((float(t), np.asarray(y).copy()))
    )
    rho, cp, _, _ = model.properties.evaluate(np.array([2.55]), np.array([301.15]))
    guard = FirstCrossingObserver(out) if observed else None
    audit = BalanceAudit(float((rho * cp)[0]), on_used_interval=guard)
    solver = Solver(
        model,
        TIGHT,
        recorder,
        audit if observed else None,
        guard,
        trace_only=None if observed else audit,
    )
    y = model.initial_state(301.15, 2.55) if initial is None else initial.copy()
    recorder.add(start, y)
    if late:
        y, t = advance_window(solver, y, start, start + 0.00000001, 2.0, "measured")
        require(solver.event_time is None, "Late-event first window should remain wet")
        y, t = advance_window(solver, y, t, end, 2.0, "measured")
        require(solver.event_time is not None, "Late fixture failed to cross")
        y, end = advance_window(solver, y, t, t + 2.0, 2.0, "measured")
    else:
        points = np.unique(
            np.r_[model.ambient.breakpoints(start, end), 60.0 if start < 60.0 < end else start]
        )
        for a, b in zip(points[:-1], points[1:]):
            y, event = solver.span(
                float(a), float(b), y, np.arange(np.floor(a) + 1.0, np.floor(b) + 1.0)
            )
            require(not event, "Unexpected event in ordinary short fixture")
    digest = lambda a: hashlib.sha256(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()
    result = dict(
        accepted_trace_sha256=audit.trace.hexdigest(),
        accepted_time_sha256=solver.step_hash.hexdigest(),
        accepted_steps=solver.step_count,
        nfev=solver.nfev,
        njev=solver.njev,
        nlu=solver.nlu,
        sample_times_sha256=digest([t for t, y in rows]),
        sample_states_sha256=digest([y for t, y in rows]),
        final_state_sha256=digest(y),
        root_s=solver.event_time,
        event_steps=solver.event_steps,
        segments=solver.segments,
        final_s=end,
    )
    np.savez_compressed(
        out / "states.npz", t_s=[t for t, y in rows], y=[y for t, y in rows], final_s=end, final_y=y
    )
    if observed:
        result["balance"] = audit.state()
        write_json(out / "balance_state.json", result["balance"])
        write_json(out / "continuous_event_state.json", guard.state())
        result["event_verification"] = verify_event(out)
        write_json(out / "event_verification.json", result["event_verification"])
    write_json(out / "record.json", result)
    return result


def equal_samples(a, b):
    keys = (
        "accepted_trace_sha256",
        "accepted_time_sha256",
        "accepted_steps",
        "nfev",
        "njev",
        "nlu",
        "sample_times_sha256",
        "sample_states_sha256",
        "final_state_sha256",
        "root_s",
        "event_steps",
        "segments",
        "final_s",
    )
    return all(a[key] == b[key] for key in keys)


def restart_check(model, out):
    """Serialize state and audit, then require identical continued BDF states and steps."""
    out = fresh(out)
    rho, cp, _, _ = model.properties.evaluate(np.array([2.55]), np.array([301.15]))
    B0 = float((rho * cp)[0])
    recorder = SimpleNamespace(add=lambda *args: None)
    first = BalanceAudit(B0)
    solver = Solver(model, TIGHT, recorder, first)
    y = model.initial_state(301.15, 2.55)
    y, _ = solver.span(0.0, 30.0, y, np.arange(1.0, 31.0))
    write_json(out / "saved_balance.json", first.state())
    np.savez_compressed(out / "saved_state.npz", y=y, t_s=30.0)
    first.trace, first.trace_steps = hashlib.sha256(), 0
    count = solver.step_count
    final, _ = solver.span(30.0, 31.0, y, np.array([31.0]))
    normal, times = first.state(), solver.step_times[count:]
    second = BalanceAudit(B0, read_json(out / "saved_balance.json"))
    with np.load(out / "saved_state.npz", allow_pickle=False) as data:
        y = data["y"].copy()
    repeated = Solver(model, TIGHT, recorder, second)
    after, _ = repeated.span(30.0, 31.0, y, np.array([31.0]))
    restored = second.state()
    keys = (
        "initial",
        "accumulators",
        "compensation",
        "subintervals",
        "accepted_intervals",
        "maxima",
        "checkpoints",
        "segments",
        "last_s",
        "partition",
    )
    ok = (
        np.array_equal(final, after)
        and np.array_equal(times, repeated.step_times)
        and first.trace.hexdigest() == second.trace.hexdigest()
        and all(normal[k] == restored[k] for k in keys)
        and normal["passed"]
        and restored["passed"]
    )
    require(ok, "Serialized restart does not reproduce state, steps or audit")
    return dict(
        passed=bool(ok), actual_JSON_and_NPZ_roundtrip=True, trace_sha256=second.trace.hexdigest()
    )


def zero_event_check(model, out):
    out = fresh(out)
    rho, cp, _, _ = model.properties.evaluate(np.array([2.55]), np.array([301.15]))
    audit = BalanceAudit(float((rho * cp)[0]))
    solver = Solver(model, TIGHT, SimpleNamespace(add=lambda *args: None), audit)
    start = 200000.0
    y = np.r_[np.zeros(model.nr), 0.149 + 0.001 * (1 - model.xi**2)]
    y[model.nr] = 0.15
    y, end = advance_window(solver, y, start, start + 5.0, 2.0, "measured")
    state = audit.state()
    passed = (
        solver.event_time == start
        and solver.event_steps == 0
        and state["passed"]
        and state["checkpoints"][str(start)]["label"] == "equality_root"
    )
    require(passed, "Exact left-end equality event fixture failed")
    write_json(out / "balance_state.json", state)
    return dict(
        passed=bool(passed),
        event_s=solver.event_time,
        steps_before_event=0,
        first_computed_crossing_supported=False,
        artificial_initial_state=True,
    )


def branch_cli(
    description, make_plan, validate, checks, run, summarize, argv=None, builder=None, verifier=None
):
    """One ordinary CLI implementation shared by both research branches."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--code-dir", type=Path)
    parser.add_argument("--research-root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "checks", "run", "continue", "summarize", "smoke"):
        item = sub.add_parser(command)
        item.add_argument("--out", type=Path, required=True)
        if command not in ("plan", "smoke"):
            item.add_argument("--plan", type=Path, required=True)
        if command in ("run", "continue"):
            item.add_argument("--checks", type=Path, required=True)
            item.add_argument("--until-s", type=float, required=command == "continue")
            item.add_argument("--job", required=command == "run")
        if command == "continue":
            item.add_argument("--previous", type=Path, required=True)
        if command == "summarize":
            item.add_argument("--runs", type=Path, required=True)
            item.add_argument("--case-map", type=Path)
        if command == "smoke":
            item.add_argument("--N", type=int, default=20)
            item.add_argument("--end-s", type=float, default=1.0)
    if verifier is not None:
        item = sub.add_parser("verify")
        item.add_argument("--run", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    configure(args.code_dir, args.research_root)
    if args.command == "plan":
        result = make_plan()
        write_json(args.out, result)
    elif args.command == "verify":
        result = verifier(args.run)
        write_json(args.out, result)
    elif args.command == "smoke":
        require(
            builder is not None and args.N >= 2 and 0 < args.end_s <= 120.0,
            "Bounded smoke requires N>=2 and duration<=120s",
        )
        out = fresh(args.out)
        value = make_plan()
        rows = []
        for scenario in value["scenarios"]:
            tested = sample_run(builder(scenario, args.N), out / scenario["id"], end=args.end_s)
            rows.append(
                dict(
                    scenario=scenario["id"],
                    balance_passed=tested["balance"]["passed"],
                    wet_prefix_passed=tested["event_verification"]["wet_prefix_passed"],
                    accepted_steps=tested["accepted_steps"],
                )
            )
        result = dict(
            passed=all(r["balance_passed"] and r["wet_prefix_passed"] for r in rows),
            test_only=True,
            production_matrix_accepted=False,
            rows=rows,
        )
        write_json(out / "summary.json", result)
    else:
        value = read_json(args.plan)
        validate(value)
        if args.command == "checks":
            result = checks(value, args.out)
        elif args.command == "summarize":
            result = summarize(value, args.runs, args.out, args.case_map)
        else:
            previous = args.previous if args.command == "continue" else None
            job_id = args.job or window_record(previous, value)["job"]["id"]
            result = run(value, job_id, args.out, args.checks, previous, args.until_s)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("passed", "completed", "effects_accepted", "event_pending", "test_only")
            },
            ensure_ascii=False,
        )
    )
    return result
