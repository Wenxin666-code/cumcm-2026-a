"""晚期精确初值的区间验证、C2连续屏障与局部轨迹生产。

数学流程：精确节点与参数 -> Decimal向外区间 -> 时间多项式 ->
空间样条与残差 -> 比较误差管 -> 条件干燥时刻。源码中所有算法均为
具名函数或数学对象，主模型的构造与右端由同包公共依赖提供。

这里验证明确指定的晚期初值。连续屏障的初值是精确节点所定义的C2
样条，既有早期演化误差属于单独问题。读取存档验收与重新计算证明
分别报告；首段检查仅代表首段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import partial, wraps
from pathlib import Path
import argparse
import bisect
import json
import random
import shutil
import time
import numpy as np

from 验证公共接口 import (
    DATA,
    SUPPORT,
    read_json,
    write_json,
    load_npz,
    run_cli_modes,
    sha256,
    science,
    prepare_output,
    source_identity,
    verify_data,
    origin_source_hash,
)

DECIMAL_DIGITS = 44
DEFAULT_BUNDLE = DATA / "evidence/endpoint"
_execution_identity = None


def decimal_arithmetic(function):
    """每次算法调用局部采用44位精度，返回后恢复调用者的Decimal上下文。"""

    @wraps(function)
    def run(*args, **kwargs):
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            return function(*args, **kwargs)

    return run


@dataclass(frozen=True)
class FrozenGrid:
    r_m: np.ndarray
    node_spacing_m: np.ndarray
    face_r_m: np.ndarray
    volume_weights_m2: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class RadiusHistory:
    t_s: np.ndarray
    radius_m: np.ndarray
    tail_radius_m: float

    def radius(self, t):
        return np.interp(t, self.t_s, self.radius_m)


@dataclass(frozen=True)
class AmbientValue:
    moisture_eq: float
    temperature_K: float


@dataclass(frozen=True)
class ConstantAmbient:
    moisture_eq: float
    temperature_K: float

    def values(self, t, **kwargs):
        return AmbientValue(self.moisture_eq, self.temperature_K)


@dataclass(frozen=True)
class DiffusionLaw:
    D_prefactor: float
    D_moisture_exponent: float
    D_temperature_exponent_K: float


@dataclass(frozen=True)
class FrozenLateModel:
    """证据中二进制常量所定义的数学模型；浮点值作为精确实数输入。"""

    grid: FrozenGrid
    geometry: RadiusHistory
    ambient: ConstantAmbient
    properties: DiffusionLaw
    nr: int
    xi: np.ndarray
    temperature_reference_K: float
    h_m_m_s: float
    h_W_m2_K: float

    @classmethod
    def from_bundle(cls, bundle, case):
        description = read_json(bundle / "proof_manifest.json")["cases"][case]
        constants = {
            k: float.fromhex(v) for k, v in description["numeric_constants_binary64_hex"].items()
        }
        with np.load(bundle / case / "proof_constants.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        grid = FrozenGrid(
            *(arrays[k] for k in ("r_m", "node_spacing_m", "face_r_m", "volume_weights_m2")),
            constants["R_reference_m"],
        )
        geometry = RadiusHistory(
            arrays["radius_knots_s"],
            arrays["radius_knots_m"],
            float(arrays["radius_knots_m"][-1]),
        )
        ambient = ConstantAmbient(constants["Ceq"], constants["T_reference_K"])
        material = DiffusionLaw(
            *(
                constants[k]
                for k in (
                    "D_prefactor",
                    "D_moisture_exponent",
                    "D_temperature_exponent_K",
                )
            )
        )
        return cls(
            grid,
            geometry,
            ambient,
            material,
            description["N"] + 1,
            arrays["xi"],
            constants["T_reference_K"],
            constants["hm"],
            constants["h"],
        )


@dataclass
class VerificationContext:
    """一次验证的显式输入与输出；不改变已导入模块中的函数。"""

    out: Path
    reuse: Path
    runs: Path
    bundle: Path
    contract: Path
    inputs: Path
    production: bool = False
    cap: str = "0.0001"
    sub_space: int = 4
    sub_time: int = 4
    first_slab_only: bool = False
    models: dict = field(default_factory=dict)


class SpatialPolynomial(list):
    """单元的时间多项式系数及Robin斜率余项的样条影响函数。"""

    def __init__(self, coefficients, remainder):
        super().__init__(coefficients)
        self.remainder = remainder


def load_model(context, case, intervals):
    if case not in context.models:
        if context.production:
            moving = science("moving_model")
            thermal = science("relative_temperature")
            original = moving.build_model(
                context.inputs,
                context.contract,
                N=intervals,
                properties="P23" if case == "p23" else "P4",
                geometry="fixed" if case == "p23" else "measured",
            )
            context.models[case] = thermal.RelativeTemperatureModel.from_model(original)
        else:
            context.models[case] = FrozenLateModel.from_bundle(context.bundle, case)
    model = context.models[case]
    if model.nr != intervals + 1:
        raise ValueError("轨迹节点数与指定模型不一致。")
    return model


def load_weight_values(context, case, xi):
    if context.production:
        return np.exp(6 * (xi**2 - 1))
    with np.load(context.bundle / case / "certificate_weights.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["xi"], xi) or not np.all(archive["w"] > 0):
            raise ValueError("正权函数或节点身份不一致。")
        return archive["w"].copy()


def current_input_hashes(context, case):
    paths = {
        "model.json": context.contract,
        "local_radau.npz": context.reuse / case / "local_radau.npz",
        "endpoint.json": context.runs / case / "endpoint.json",
        "proof_constants.npz": context.bundle / case / "proof_constants.npz",
    }
    weights = context.out / case / "certificate_weights.npz"
    paths["certificate_weights.npz"] = (
        weights if weights.is_file() else context.bundle / case / weights.name
    )
    return {name: sha256(path) for name, path in paths.items() if path.is_file()}


def write_record(path, value):
    """计算值与当前执行源码身份分列记录；历史来源由独立索引解释。"""
    global _execution_identity
    if _execution_identity is None:
        _execution_identity = source_identity()
    record = dict(value)
    record["current_source_identity"] = _execution_identity
    record["historical_source_index_sha256"] = sha256(DATA / "来源源码索引.json")
    write_json(path, record)


def check_frozen_bundle(bundle=DEFAULT_BUNDLE):
    """核对数据字节及证据所声明的历史源身份，二者均区别于现行源码。"""
    manifest = read_json(bundle / "proof_manifest.json")
    if manifest.get("schema") == "integrated-late-snapshot-v1":
        if sha256(bundle / "model.json") != manifest["model_sha256"]:
            raise ValueError("自产快照模型配置身份不一致。")
        checked = 0
        for case, description in manifest["cases"].items():
            for name, expected in description["proof_object_sha256"].items():
                if sha256(bundle / case / name) != expected:
                    raise ValueError("自产快照数据身份不一致：" + case + "/" + name)
                checked += 1
            if (
                sha256(bundle / case / "endpoint.json")
                != description["production_input_sha256"]["endpoint.json"]
            ):
                raise ValueError("自产快照终点记录不一致。")
        return {
            "passed": True,
            "data_files_checked": checked,
            "snapshot_source_identity": manifest["current_source_identity"],
            "current_code_is_distinct_from_historical_sources": True,
            "proof_values_recomputed": False,
            "snapshot_manifest_sha256": sha256(bundle / "proof_manifest.json"),
        }
    inventory = read_json(bundle / "bundle_inventory.json")
    historical = read_json(DATA / "来源源码索引.json")["files"]
    source_count = data_count = 0
    for relative, expected in inventory["files_sha256"].items():
        if relative.endswith(".py"):
            key = "evidence/endpoint/" + relative
            if historical[key]["sha256"] != expected:
                raise ValueError("历史源码身份记录不一致：" + relative)
            source_count += 1
        else:
            if sha256(bundle / relative) != expected:
                raise ValueError("晚期证据数据校验失败：" + relative)
            data_count += 1
    manifest = read_json(bundle / "proof_manifest.json")
    if sha256(bundle / "model.json") != manifest["input_sha256"]["model.json"]:
        raise ValueError("模型配置身份不一致。")
    for name, expected in manifest["source_sha256"].items():
        key = (
            "evidence/endpoint/code/" + Path(name).name
            if name.startswith("extension_code/")
            else "code/" + name
        )
        if historical[key]["sha256"] != expected:
            raise ValueError("证书的历史来源字段不一致：" + name)
    for case, entry in manifest["cases"].items():
        if (
            sha256(bundle / case / "endpoint.json")
            != entry["production_input_sha256"]["endpoint.json"]
        ):
            raise ValueError("终点记录身份不一致：" + case)
        for name, expected in entry["proof_object_sha256"].items():
            if sha256(bundle / case / name) != expected:
                raise ValueError("证明对象身份不一致：" + case + "/" + name)
        certificate = read_json(bundle / case / "continuous_certificate.json")
        for name, expected in certificate["source_sha256"].items():
            if name.endswith(".py"):
                actual = historical["evidence/endpoint/code/" + name]["sha256"]
            else:
                path = bundle / "model.json" if name == "model.json" else bundle / case / name
                actual = sha256(path)
            if actual != expected:
                raise ValueError("连续证书来源字段不一致：" + name)
    return {
        "passed": True,
        "data_files_checked": data_count,
        "historical_source_records_checked": source_count,
        "current_code_is_distinct_from_historical_sources": True,
        "proof_values_recomputed": False,
        "bundle_inventory_sha256": sha256(bundle / "bundle_inventory.json"),
    }


def comparable_proof(value):
    """只排除执行时间及明确的来源字段；数学条件、范围和数值均参与比较。"""
    provenance = {
        "elapsed_s",
        "source_sha256",
        "guard_source_sha256",
        "current_source_identity",
        "historical_source_index_sha256",
    }
    if isinstance(value, dict):
        return {k: comparable_proof(v) for k, v in value.items() if k not in provenance}
    if isinstance(value, list):
        return [comparable_proof(item) for item in value]
    return value


def require_equivalent(reference, calculated, label):
    if comparable_proof(reference) != comparable_proof(calculated):
        raise AssertionError("重算的数学值或条件与参照证据不同：" + label)
    return True


def as_decimal(x):
    """浮点数按其精确二进制值转为Decimal；字符串按十进制实数解释。"""
    return (
        x
        if isinstance(x, Decimal)
        else Decimal.from_float(x) if isinstance(x, float) else Decimal(x)
    )


class Interval:
    __slots__ = ("lo", "hi")

    def __init__(self, lo=0, hi=None):
        self.lo = as_decimal(lo)
        self.hi = self.lo if hi is None else as_decimal(hi)
        if self.lo > self.hi:
            raise ValueError("Reversed interval")

    def __add__(self, o):
        o = as_interval(o)
        return Interval((self.lo + o.lo).next_minus(), (self.hi + o.hi).next_plus())

    __radd__ = __add__

    def __neg__(self):
        return Interval(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, o):
        return self + -as_interval(o)

    def __rsub__(self, o):
        return as_interval(o) + -self

    def __mul__(self, o):
        o = as_interval(o)
        p = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return Interval(min(p).next_minus(), max(p).next_plus())

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError("Interval contains zero")
        return Interval((Decimal(1) / self.hi).next_minus(), (Decimal(1) / self.lo).next_plus())

    def __truediv__(self, o):
        return self * as_interval(o).reciprocal()

    def __rtruediv__(self, o):
        return as_interval(o) * self.reciprocal()

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("Nonnegative integer power only")
        ans = Interval(1)
        for _ in range(n):
            ans = ans * self
        return ans

    def exp(self):
        return Interval(self.lo.exp().next_minus(), self.hi.exp().next_plus())

    def absmax(self):
        return max(self.lo.copy_abs(), self.hi.copy_abs())

    def padded(self, e):
        return self + Interval(-as_decimal(e), as_decimal(e))

    def floats(self):
        return [float(self.lo), float(self.hi)]

    def strings(self):
        return [str(self.lo), str(self.hi)]

    def __repr__(self):
        return f"I({self.lo},{self.hi})"


def as_interval(x):
    """把标量提升为退化区间，供四则运算复用。"""
    return x if isinstance(x, Interval) else Interval(x)


def poly_add(a, b):
    """按升幂系数相加，缺失高次项视为零。"""
    return [
        (a[k] if k < len(a) else Interval()) + (b[k] if k < len(b) else Interval())
        for k in range(max(len(a), len(b)))
    ]


def poly_negate(a):
    return [-v for v in a]


def poly_subtract(a, b):
    """区间多项式相减。"""
    return poly_add(a, poly_negate(b))


def poly_scale(a, s):
    """区间多项式乘以标量或区间。"""
    return [v * s for v in a]


def poly_multiply(a, b):
    """逐项卷积得到乘积，每一步保留向外舍入。"""
    c = [Interval() for _ in range(len(a) + len(b) - 1)]
    for j, x in enumerate(a):
        for k, y in enumerate(b):
            c[j + k] = c[j + k] + x * y
    return c


def poly_value(a, x):
    """用Horner公式包围区间自变量上的多项式值。"""
    v = Interval()
    for c in a[::-1]:
        v = v * x + c
    return v


def poly_bound(a):
    """包围标准时间区间[0,1]上的多项式。"""
    return a[0] + sum(
        (Interval(min(Decimal(0), v.lo), max(Decimal(0), v.hi)) for v in a[1:]),
        Interval(),
    )


def poly_absmax(a):
    """返回[0,1]上绝对值的区间上界。"""
    return poly_bound(a).absmax()


def diffusivity_series(q, pref, a, b, T, order=10, radius=64):
    """Taylor coefficients and rigorous Cauchy tail on u in [0,1].

    q is a cubic polynomial. On complex |u|=rho, |q-q0|<=s<q0.
    Thus |pref exp(-a/q-b/T)|<=pref exp(a/(q0-s)-b/T).
    Cauchy's coefficient bound controls the omitted infinite series. The
    arithmetic recurrence encloses every retained coefficient.
    """
    inv = [1 / q[0]]
    for n in range(1, order + 1):
        inv.append(
            -sum(
                (q[j] * inv[n - j] for j in range(1, min(n, len(q) - 1) + 1)),
                Interval(),
            )
            / q[0]
        )
    z = [-a * v for v in inv]
    z[0] = z[0] - b / T
    d = [pref * z[0].exp()]
    for n in range(1, order + 1):
        d.append(sum((k * z[k] * d[n - k] for k in range(1, n + 1)), Interval()) / n)
    rho = Interval(radius)
    s = sum((Interval(q[j].absmax()) * rho**j for j in range(1, len(q))), Interval()).hi
    if q[0].lo <= s:
        raise ValueError("Cauchy disk reaches C=0")
    bound = pref * (a / (q[0] - s) - b / T).exp()
    tail = (bound / rho ** (order + 1) / (1 - 1 / rho)).hi
    return (d, tail)


def radius_scale_series(R, Rref, order=6):
    x = R[1] / R[0] if len(R) > 1 else Interval()
    scale = (Rref / R[0]) ** 2
    out = [scale * (-x) ** n * (n + 1) for n in range(order + 1)]
    z = Interval(x.absmax())
    if z.hi >= 1:
        raise ValueError("半径几何级数需要相对变化严格小于1。")
    tail = (Interval(scale.absmax()) * (order + 2) * z ** (order + 1) / (1 - z) ** 2).hi
    return (out, tail)


def produce_local_trajectory(args, case):
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    "从主程序检查点重启Radau/BDF；右端复用现有主模型。"
    out = args.out / case
    out.mkdir(parents=True, exist_ok=True)
    data = load_npz(args.runs / case / "internal.npz")
    ep = json.loads((args.runs / case / "endpoint.json").read_text())
    N = len(data["xi"]) - 1
    model = load_model(args, case, N)
    root = ep["equality_root_s"]
    end = ep["reported_time_s"] + 2
    rows = []
    for method, window in [("Radau", 0), ("BDF", 0), ("Radau", 600), ("Radau", 3600)]:
        indices = np.where(data["t_s"] < (root - window if window else root))[0]
        j = int(indices[-1])
        start = float(data["t_s"][j])
        T0 = data["T_K"][j]
        C0 = data["C"][j]
        y = np.r_[T0 - model.temperature_reference_K, C0]
        segments = []
        t0 = time.perf_counter()
        for left, right in zip(
            model.geometry.breakpoints(start, end)[:-1],
            model.geometry.breakpoints(start, end)[1:],
        ):
            result = solve_ivp(
                model.rhs,
                (left, right),
                y,
                method=method,
                rtol=2e-12,
                atol=np.r_[np.full(N + 1, 1e-14), np.full(N + 1, 2e-15)],
                max_step=2.0 if window == 0 else 20.0,
                jac_sparsity=model.jacobian_sparsity(),
                dense_output=True,
            )
            if not result.success:
                raise RuntimeError(result.message)
            segments.append(result)
            y = result.y[:, -1]

        def state(t):
            k = min(np.searchsorted([s.t[-1] for s in segments], t), len(segments) - 1)
            return segments[k].sol(t)

        rr = brentq(lambda t: np.max(state(t)[N + 1 :]) - 0.15, start, end, xtol=1e-09)
        common = data["t_s"][(data["t_s"] >= start) & (data["t_s"] <= end)]
        ys = np.column_stack([state(float(t)) for t in common])
        orig = data["C"][np.searchsorted(data["t_s"], common)]
        slope = model.rhs(rr, state(rr))[N + 1 :]
        v = state(rr)[N + 1 :]
        row = {
            "method": method,
            "requested_lookback_s": window,
            "checkpoint_s": start,
            "root_s": rr,
            "root_minus_production_s": rr - root,
            "maximum_common_C_difference": float(np.max(abs(ys[N + 1 :].T - orig))),
            "checkpoint_max_T_deviation_K": float(np.max(abs(T0 - model.temperature_reference_K))),
            "point_event_active_slope_per_s": float(slope[np.argmax(v)]),
            "near_active_slope_range_per_s": [
                float(slope[v >= v.max() - 1e-08].min()),
                float(slope[v >= v.max() - 1e-08].max()),
            ],
            "steps": sum((len(s.t) - 1 for s in segments)),
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        print(json.dumps({"case": case, **row}), flush=True)
        if method == "Radau" and window == 0:
            polys = [p for s in segments for p in s.sol.interpolants]
            np.savez_compressed(
                out / "local_radau.npz",
                left=np.array([p.t_old for p in polys]),
                right=np.array([p.t for p in polys]),
                y_old=np.stack([p.y_old for p in polys]),
                y_new=np.stack([s.y[:, j + 1] for s in segments for j in range(len(s.t) - 1)]),
                Q=np.stack([p.Q for p in polys]),
                xi=model.xi,
            )
    write_record(
        out / "restart_comparisons.json",
        {
            "rows": rows,
            "scope": "Independent numerical restarts; inherited checkpoint error NOT enclosed.",
            "production_endpoint": ep,
            "source_sha256": current_input_hashes(args, case),
        },
    )


def radius_polynomial(model, left, right):
    """在单个半径观测区间内构造精确线性半径多项式。"""
    g = model.geometry
    if not hasattr(g, "t_s"):
        return ([Interval(float(g.radius(left)))], [])
    j = int(np.searchsorted(g.t_s, left, side="right") - 1)
    if j >= len(g.t_s) - 1:
        return ([Interval(float(g.tail_radius_m))], [])
    if right > g.t_s[j + 1] + 1e-09:
        raise ValueError("Dense interval crosses a radius knot")
    slope = (Interval(float(g.radius_m[j + 1])) - Interval(float(g.radius_m[j]))) / (
        Interval(float(g.t_s[j + 1])) - Interval(float(g.t_s[j]))
    )
    r = Interval(float(g.radius_m[j])) + slope * (Interval(float(left)) - Interval(float(g.t_s[j])))
    return (
        [r, slope * (Interval(float(right)) - Interval(float(left)))],
        [float(g.t_s[j]), float(g.t_s[j + 1])],
    )


def diffusivity_derivatives(q, T, p):
    """共同包围扩散率及其对含水率、温度的一阶导数。"""
    d = (
        Interval(float(p.D_prefactor))
        * (
            -Interval(float(p.D_moisture_exponent)) / q
            - Interval(float(p.D_temperature_exponent_K)) / T
        ).exp()
    )
    dc = d * Interval(float(p.D_moisture_exponent)) / (q * q)
    dt = d * Interval(float(p.D_temperature_exponent_K)) / (T * T)
    return (d, dc, dt)


def semidiscrete_slab_bounds(model, q, left, right, cap, temperature_cap):
    """All-time bounds over one continuous cubic Radau reconstruction slab."""
    n = len(q)
    grid = model.grid
    p = model.properties
    R, _ = radius_polynomial(model, left, right)
    rbox = poly_bound(R)
    Rref = Interval(float(grid.radius_m))
    scalebox = (Rref / rbox) ** 2
    scale, scale_tail = radius_scale_series(R, Rref, 0 if len(R) == 1 else 6)
    T = Interval(float(model.temperature_reference_K))
    Ceq = Interval(float(model.ambient.values(left, right_of_breakpoint=True).moisture_eq))
    hm = Interval(float(model.h_m_m_s))
    h = Interval(float(right)) - Interval(float(left))
    a = Interval(float(p.D_moisture_exponent))
    b = Interval(float(p.D_temperature_exponent_K))
    pref = Interval(float(p.D_prefactor))
    diff = []
    derr = []
    boxes = []
    dval = []
    dcval = []
    dtval = []
    for qi in q:
        di, ei = diffusivity_series(qi, pref, a, b, T)
        diff.append(di)
        derr.append(ei)
        box = poly_bound(qi).padded(cap)
        boxes.append(box)
        dd, dc, dt = diffusivity_derivatives(box, T.padded(temperature_cap), p)
        dval.append(dd)
        dcval.append(dc)
        dtval.append(dt)
    flux = []
    ferr = []
    shift_flux = []
    thermal_flux = []
    coop = []
    for j in range(n - 1):
        fac = Interval(float(grid.face_r_m[j])) / (2 * Interval(float(grid.node_spacing_m[j])))
        dq = poly_subtract(q[j + 1], q[j])
        dqbox = poly_bound(dq)
        flux.append(poly_scale(poly_multiply(poly_add(diff[j], diff[j + 1]), dq), fac))
        ferr.append(
            (fac * (Interval(derr[j]) + Interval(derr[j + 1])) * Interval(dqbox.absmax())).hi
        )
        shift_flux.append(fac * (dcval[j] + dcval[j + 1]) * dqbox)
        thermal_flux.append((fac * (dtval[j] + dtval[j + 1]) * Interval(dqbox.absmax())).hi)
        delta = Interval(dqbox.absmax()) + 2 * Interval(cap)
        coop.append(
            (dval[j] + dval[j + 1] - Interval(max(dcval[j].hi, dcval[j + 1].hi)) * delta).lo
        )
    outer = poly_scale(poly_multiply(R, poly_subtract(q[-1], [Ceq])), -hm)
    defects = []
    mu = []
    thermal = []
    for j in range(n):
        incoming = flux[j - 1] if j else [Interval()]
        outgoing = flux[j] if j < n - 1 else outer
        w = Interval(float(grid.volume_weights_m2[j]))
        div = poly_scale(poly_subtract(outgoing, incoming), 1 / w)
        err = Interval(ferr[j - 1] if j else Decimal(0)) + Interval(
            ferr[j] if j < n - 1 else Decimal(0)
        )
        rhs = poly_multiply(scale, div)
        deriv = [q[j][k] * k / h for k in range(1, len(q[j]))]
        rem = (
            Interval(scalebox.absmax()) * err / w
            + Interval(scale_tail) * Interval(poly_absmax(div))
        ).hi
        defects.append((Interval(poly_absmax(poly_subtract(deriv, rhs))) + Interval(rem)).hi)
        shift = (shift_flux[j] if j < n - 1 else -rbox * hm) - (
            shift_flux[j - 1] if j else Interval()
        )
        mu.append((scalebox * shift / w).hi)
        thermal.append(
            (
                scalebox
                * (
                    Interval(thermal_flux[j - 1] if j else Decimal(0))
                    + Interval(thermal_flux[j] if j < n - 1 else Decimal(0))
                )
                / w
            ).hi
        )
    return {
        "defect": defects,
        "mu": mu,
        "thermal_gain": thermal,
        "cooperative_numerator_lower": min(coop),
        "minimum_C": min((v.lo for v in boxes)),
        "minimum_D": min((v.lo for v in dval)),
        "radius": rbox,
        "cauchy_tail_max": max(derr),
        "scale_tail": scale_tail,
    }


def nodal_polynomials(data, j, n):
    """用旧/新节点值精确修正三次项，保证相邻时间段无跳跃。"""
    q = []
    for i in range(n):
        q0 = Interval(float(data["y_old"][j, n + i]))
        q1 = Interval(float(data["Q"][j, n + i, 0]))
        q2 = Interval(float(data["Q"][j, n + i, 1]))
        q3 = Interval(float(data["y_new"][j, n + i])) - q0 - q1 - q2
        q.append([q0, q1, q2, q3])
    return q


def initial_error_scenarios():
    """统一管和细化管采用同一组已声明的初始误差条件。"""
    return [
        {"C0": Decimal(c), "T0": Decimal(t), "e": Decimal(c), "rows": []}
        for c, t in [("0", "0"), ("1e-9", "0"), ("1e-8", "1e-12"), ("1e-7", "1e-9")]
    ]


def bracket_threshold(maximum, left, right, threshold_lower, threshold_upper, iterations):
    """按调用者的精确阈值及轮数分别夹逼左右端；只在已证实夹住事件时调用。"""
    a, b = left, right
    for _ in range(iterations):
        middle = (a + b) / 2
        if maximum(middle).lo > threshold_upper:
            a = middle
        else:
            b = middle
    lower = a
    a, b = left, right
    for _ in range(iterations):
        middle = (a + b) / 2
        if maximum(middle).hi < threshold_lower:
            b = middle
        else:
            a = middle
    return lower, b


@decimal_arithmetic
def verify_uniform_tube(args, case):
    """对四组给定初始误差传播统一误差管并包围条件终点。"""
    out = args.out / case
    data = load_npz(out / "local_radau.npz")
    n = len(data["xi"])
    model = load_model(args, case, n - 1)
    ep = json.loads((args.runs / case / "endpoint.json").read_text())
    cap = Decimal("0.00001")
    tcap = Decimal("0.000000001")
    scenarios = initial_error_scenarios()
    slab_records = []
    started = time.perf_counter()
    qq = []
    for j, (left, right) in enumerate(zip(data["left"], data["right"])):
        q = nodal_polynomials(data, j, n)
        qq.append(q)
        bounds = semidiscrete_slab_bounds(model, q, float(left), float(right), cap, tcap)
        mu = max(Decimal(0), max(bounds["mu"]))
        hh = Interval(float(right)) - Interval(float(left))
        ex = (Interval(mu) * hh).exp().hi
        rec = {
            "index": j,
            "left_s": float(left),
            "right_s": float(right),
            "defect_max": str(max(bounds["defect"])),
            "uniform_shift_logarithmic_bound_per_s": str(mu),
            "thermal_gain_max_per_s_K": str(max(bounds["thermal_gain"])),
            "cooperative_numerator_lower": str(bounds["cooperative_numerator_lower"]),
            "minimum_C_in_assumed_tube": str(bounds["minimum_C"]),
            "minimum_D_in_assumed_tube": str(bounds["minimum_D"]),
            "cauchy_D_tail_max": str(bounds["cauchy_tail_max"]),
        }
        if bounds["cooperative_numerator_lower"] <= 0 or bounds["minimum_C"] <= 0:
            raise RuntimeError("Comparison assumptions failed")
        for s in scenarios:
            eta = max(
                (
                    (Interval(a) + Interval(b) * Interval(s["T0"])).hi
                    for a, b in zip(bounds["defect"], bounds["thermal_gain"])
                )
            )
            e0 = s["e"]
            e1 = ((Interval(e0) + Interval(eta) * hh) * Interval(ex)).hi
            s["e"] = e1
            s["rows"].append({"e0": str(e0), "e1": str(e1), "eta": str(eta), "mu": str(mu)})
        slab_records.append(rec)
        if j % 5 == 0:
            print(
                json.dumps(
                    {
                        "case": case,
                        "verified_slab": j + 1,
                        "total": len(data["left"]),
                        "eta": rec["defect_max"],
                        "mu": str(mu),
                        "conditional_errors": [str(s["e"]) for s in scenarios],
                        "elapsed_s": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
            write_record(
                out / "verification_progress.json",
                {
                    "slabs_completed": j + 1,
                    "slabs_total": len(data["left"]),
                    "last_slab": rec,
                },
            )

    def error_at(s, j, t):
        r = s["rows"][j]
        h = Interval(t) - Interval(float(data["left"][j]))
        return (
            (Interval(Decimal(r["e0"])) + Interval(Decimal(r["eta"])) * h)
            * (Interval(Decimal(r["mu"])) * h).exp()
        ).hi

    def maximum_interval(s, t):
        j = min(
            bisect.bisect_left([as_decimal(float(v)) for v in data["right"]], as_decimal(t)),
            len(qq) - 1,
        )
        u = (Interval(t) - Interval(float(data["left"][j]))) / (
            Interval(float(data["right"][j])) - Interval(float(data["left"][j]))
        )
        vv = [poly_value(p, u) for p in qq[j]]
        e = error_at(s, j, t)
        return Interval(max((v.lo for v in vv)), max((v.hi for v in vv))).padded(e)

    results = []
    threshold = Interval("0.15")
    start = as_decimal(float(data["left"][0]))
    end = as_decimal(float(data["right"][-1]))
    for s in scenarios:
        cap_ok = s["e"] < cap
        lower = start
        upper = end
        valid = (
            cap_ok
            and maximum_interval(s, start).lo > threshold.hi
            and (maximum_interval(s, end).hi < threshold.lo)
        )
        if valid:
            lower, upper = bracket_threshold(
                lambda t: maximum_interval(s, t), start, end, threshold.lo, threshold.hi, 45
            )
        report = Decimal(str(ep["reported_time_s"]))
        report_C = maximum_interval(s, report)
        results.append(
            {
                "initial_C_error_bound": str(s["C0"]),
                "initial_T_error_bound_K": str(s["T0"]),
                "checkpoint_s": str(start),
                "last_time_s": str(end),
                "final_C_error_bound": str(s["e"]),
                "tube_cap_satisfied": cap_ok,
                "conditional_semidiscrete_event_enclosed": valid,
                "root_interval_s_decimal": [str(lower), str(upper)] if valid else None,
                "root_interval_width_s": float(upper - lower) if valid else None,
                "production_report_C_interval_decimal": report_C.strings(),
                "production_report_conditionally_strictly_dry": bool(
                    cap_ok and report_C.hi < threshold.lo
                ),
                "per_slab_error_bounds": s["rows"],
            }
        )
    record = {
        "completed": True,
        "conditional_semidiscrete_certificate_passed": all(
            (r["conditional_semidiscrete_event_enclosed"] for r in results[:3])
        ),
        "continuous_PDE_certified": False,
        "global_t0_semidiscrete_error_certified": False,
        "case": case,
        "N": n - 1,
        "precision_decimal_digits": 44,
        "source_sha256": current_input_hashes(args, case),
        "scope": (
            "Exact real arithmetic semidiscrete equations with recorded binary64 "
            "grid/property constants; radius is exact piecewise-linear interpolation of "
            "binary64 input knots. Conditional checkpoint T/C bounds only. No "
            "spatial/PDE certification."
        ),
        "mathematical_conditions": {
            "positive_C_D_k_b_R": True,
            "moisture_cooperative_on_tube": True,
            "temperature_scalar_maximum_principle": True,
            "constant_tail_environment": True,
            "continuous_reconstruction_by_endpoint_correction": True,
            "full_time_residual_enclosed_not_sampled": True,
            "global_max_nonincreasing_above_equilibrium": True,
        },
        "uniform_C_tube_cap": str(cap),
        "temperature_tube_cap_K": str(tcap),
        "slabs": slab_records,
        "scenarios": results,
        "elapsed_s": time.perf_counter() - started,
    }
    write_record(out / "conditional_certificate.json", record)
    print(
        json.dumps(
            {
                "case": case,
                "conditional_passed": record["conditional_semidiscrete_certificate_passed"],
                "widths_s": [r["root_interval_width_s"] for r in results],
                "report_dry": [r["production_report_conditionally_strictly_dry"] for r in results],
            }
        ),
        flush=True,
    )


def uniform_stability_bound(model, q, left, right, cap, Tcap):
    """在指定误差管上包围比较系统的稳定性常数。"""
    n = len(q)
    p = model.properties
    R, _ = radius_polynomial(model, left, right)
    r = poly_bound(R)
    scale = (Interval(float(model.grid.radius_m)) / r) ** 2
    T = Interval(float(model.temperature_reference_K)).padded(Tcap)
    boxes = [poly_bound(x).padded(cap) for x in q]
    coeff = [diffusivity_derivatives(c, T, p) for c in boxes]
    flux = []
    cooperative = []
    for j in range(n - 1):
        g = Interval(float(model.grid.face_r_m[j])) / (
            2 * Interval(float(model.grid.node_spacing_m[j]))
        )
        dc = poly_bound(poly_subtract(q[j + 1], q[j]))
        flux.append(g * (coeff[j][1] + coeff[j + 1][1]) * dc)
        cooperative.append(
            (
                coeff[j][0]
                + coeff[j + 1][0]
                - Interval(max(coeff[j][1].hi, coeff[j + 1][1].hi))
                * (Interval(dc.absmax()) + 2 * Interval(cap))
            ).lo
        )
    rows = []
    for j in range(n):
        f = (flux[j] if j < n - 1 else -r * Interval(float(model.h_m_m_s))) - (
            flux[j - 1] if j else Interval()
        )
        rows.append((scale * f / Interval(float(model.grid.volume_weights_m2[j]))).hi)
    assert min(cooperative) > 0 and min((c.lo for c in boxes)) > 0
    return (max(Decimal(0), max(rows)), min(cooperative))


@decimal_arithmetic
def verify_refined_tube(args, case):
    """保留已有残差上界，在较窄误差管上重算稳定性和横截性。"""
    out = args.out / case
    base = json.loads((out / "conditional_certificate.json").read_text())
    data = load_npz(out / "local_radau.npz")
    n = len(data["xi"])
    model = load_model(args, case, n - 1)
    rights = [as_decimal(float(t)) for t in data["right"]]
    lefts = [as_decimal(float(t)) for t in data["left"]]
    q = [nodal_polynomials(data, j, n) for j in range(len(rights))]
    cap = Decimal(args.cap)
    Tcap = Decimal("1e-9")
    slabs = []
    start = time.perf_counter()
    scenarios = initial_error_scenarios()
    for j, (left, right) in enumerate(zip(lefts, rights)):
        mu, coop = uniform_stability_bound(model, q[j], float(left), float(right), cap, Tcap)
        h = Interval(right) - Interval(left)
        b = base["slabs"][j]
        for s in scenarios:
            eta = (
                Interval(Decimal(b["defect_max"]))
                + Interval(Decimal(b["thermal_gain_max_per_s_K"])) * Interval(s["T0"])
            ).hi
            e0 = s["e"]
            s["e"] = ((Interval(e0) + Interval(eta) * h) * (Interval(mu) * h).exp()).hi
            s["rows"].append({"eta": eta, "mu": mu, "e0": e0, "e1": s["e"]})
        slabs.append(
            {
                "left_s": str(left),
                "right_s": str(right),
                "mu_upper_per_s": str(mu),
                "cooperative_numerator_lower": str(coop),
            }
        )
        if j % 5 == 0:
            print(
                json.dumps(
                    {
                        "case": case,
                        "refined_stability_slab": j + 1,
                        "cap": str(cap),
                        "mu": str(mu),
                        "elapsed_s": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )

    def e_at(s, j, t):
        row = s["rows"][j]
        h = Interval(t) - Interval(lefts[j])
        return (
            (Interval(row["e0"]) + Interval(row["eta"]) * h) * (Interval(row["mu"]) * h).exp()
        ).hi

    def boxes_at(s, t):
        j = min(bisect.bisect_left(rights, t), len(q) - 1)
        u = (Interval(t) - Interval(lefts[j])) / (Interval(rights[j]) - Interval(lefts[j]))
        e = e_at(s, j, t)
        return [poly_value(v, u).padded(e) for v in q[j]]

    def mx(s, t):
        vv = boxes_at(s, t)
        return Interval(max((v.lo for v in vv)), max((v.hi for v in vv)))

    def transversality(s, left, right):
        vv = [None] * n
        emax = Decimal(0)
        rr = []
        for j in range(len(q)):
            lo = max(left, lefts[j])
            hi = min(right, rights[j])
            if lo > hi:
                continue
            u = (Interval(lo, hi) - Interval(lefts[j])) / (Interval(rights[j]) - Interval(lefts[j]))
            ee = e_at(s, j, hi)
            emax = max(emax, ee)
            R, _ = radius_polynomial(model, float(lefts[j]), float(rights[j]))
            rr.append(poly_bound(R))
            for i in range(n):
                v = poly_value(q[j][i], u).padded(ee)
                vv[i] = v if vv[i] is None else Interval(min(vv[i].lo, v.lo), max(vv[i].hi, v.hi))
        target = max((v.lo for v in vv))
        active = [i for i, v in enumerate(vv) if v.hi >= target]
        R = Interval(min((x.lo for x in rr)), max((x.hi for x in rr)))
        scale = (Interval(float(model.grid.radius_m)) / R) ** 2
        T = Interval(float(model.temperature_reference_K)).padded(s["T0"])
        cc = [diffusivity_derivatives(v, T, model.properties)[0] for v in vv]
        rhs = []
        for i in active:
            f = Interval()
            for j in [i - 1, i + 1]:
                if j < 0 or j >= n:
                    continue
                e = min(i, j)
                fac = Interval(float(model.grid.face_r_m[e])) / (
                    2 * Interval(float(model.grid.node_spacing_m[e]))
                )
                f = f + fac * (cc[i] + cc[j]) * (vv[j] - vv[i])
            if i == n - 1:
                ceq = Interval(
                    float(model.ambient.values(float(left), right_of_breakpoint=True).moisture_eq)
                )
                f = f - R * Interval(float(model.h_m_m_s)) * (vv[i] - ceq)
            rhs.append(scale * f / Interval(float(model.grid.volume_weights_m2[i])))
        bound = Interval(min((x.lo for x in rhs)), max((x.hi for x in rhs)))
        return {
            "possible_active_nodes": active,
            "possible_active_xi": [float(model.xi[i]) for i in active],
            "Dini_derivative_interval_per_s": bound.strings(),
            "negative_upper_bound_verified": bound.hi < 0,
            "state_error_bound_used": str(emax),
        }

    ep = json.loads((args.runs / case / "endpoint.json").read_text())
    reports = []
    threshold = Decimal(".15")
    for s in scenarios:
        valid = (
            s["e"] < cap and mx(s, lefts[0]).lo > threshold and (mx(s, rights[-1]).hi < threshold)
        )
        lo = lefts[0]
        hi = rights[-1]
        if valid:
            lo, hi = bracket_threshold(
                lambda t: mx(s, t), lefts[0], rights[-1], threshold, threshold, 42
            )
        rpt = mx(s, Decimal(str(ep["reported_time_s"])))
        reports.append(
            {
                "initial_C_error_bound": str(s["C0"]),
                "initial_T_error_bound_K": str(s["T0"]),
                "tube_cap_satisfied": s["e"] < cap,
                "final_error_bound": str(s["e"]),
                "conditional_semidiscrete_event_enclosed": valid,
                "root_interval_s_decimal": [str(lo), str(hi)] if valid else None,
                "width_s": float(hi - lo) if valid else None,
                "production_report_C_interval_decimal": rpt.strings(),
                "production_report_conditionally_strictly_dry": bool(valid and rpt.hi < threshold),
                "transversality": transversality(s, lo, hi) if valid else None,
                "per_slab_error_bounds": [{k: str(v) for k, v in r.items()} for r in s["rows"]],
            }
        )
    result = {
        "completed": True,
        "conditional_semidiscrete_certificate_passed": all(
            (s["conditional_semidiscrete_event_enclosed"] for s in reports)
        ),
        "continuous_PDE_certified": False,
        "global_t0_semidiscrete_error_certified": False,
        "case": case,
        "N": n - 1,
        "checkpoint_s": str(lefts[0]),
        "C_tube_cap": str(cap),
        "slabs": slabs,
        "scenarios": reports,
        "source_sha256": current_input_hashes(args, case),
        "scope": (
            "Conditional late-checkpoint real semidiscrete IVP only; narrower comparison "
            "tube, original all-time residual enclosures reused unchanged."
        ),
        "elapsed_s": time.perf_counter() - start,
    }
    write_record(out / "refined_conditional_certificate.json", result)
    print(
        json.dumps(
            {
                "case": case,
                "widths_s": [s["width_s"] for s in reports],
                "report_dry": [s["production_report_conditionally_strictly_dry"] for s in reports],
                "transverse": [
                    (
                        s["transversality"]["negative_upper_bound_verified"]
                        if s["transversality"]
                        else None
                    )
                    for s in reports
                ],
            }
        ),
        flush=True,
    )


@decimal_arithmetic
def verify_weighted_tube(args, case):
    """正权函数将局部残差转为加权幅值，包围中心误差与可能活跃节点。"""
    out = args.out / case
    data = load_npz(out / "local_radau.npz")
    n = len(data["xi"])
    model = load_model(args, case, n - 1)
    weights = load_weight_values(args, case, model.xi)
    w = [Interval(float(x)) for x in weights]
    cap = Decimal("1e-7")
    lefts = [as_decimal(float(t)) for t in data["left"]]
    rights = [as_decimal(float(t)) for t in data["right"]]
    q = [nodal_polynomials(data, j, n) for j in range(len(rights))]
    records = []
    amplitude = Decimal(0)
    started = time.perf_counter()
    for k, (left, right) in enumerate(zip(lefts, rights)):
        qq = q[k]
        raw = semidiscrete_slab_bounds(model, qq, float(left), float(right), cap, Decimal(0))
        R, _ = radius_polynomial(model, float(left), float(right))
        R = poly_bound(R)
        scale = (Interval(float(model.grid.radius_m)) / R) ** 2
        T = Interval(float(model.temperature_reference_K))
        boxes = [poly_bound(qq[i]).padded((Interval(cap) * w[i]).hi) for i in range(n)]
        coeff = [diffusivity_derivatives(c, T, model.properties) for c in boxes]
        faces = []
        coop = []
        for i in range(n - 1):
            g = Interval(float(model.grid.face_r_m[i])) / (
                2 * Interval(float(model.grid.node_spacing_m[i]))
            )
            dw = w[i + 1] - w[i]
            dq = poly_bound(poly_subtract(qq[i + 1], qq[i]))
            grad = dq + Interval(-cap, cap) * dw
            faces.append(
                g
                * (
                    (coeff[i][0] + coeff[i + 1][0]) * dw
                    + (coeff[i][1] * w[i] + coeff[i + 1][1] * w[i + 1]) * grad
                )
            )
            coop.append(
                (
                    coeff[i][0]
                    + coeff[i + 1][0]
                    - Interval(max(coeff[i][1].hi, coeff[i + 1][1].hi))
                    * (Interval(dq.absmax()) + Interval(cap) * (w[i] + w[i + 1]))
                ).lo
            )
        rows = []
        for i in range(n):
            f = (faces[i] if i < n - 1 else -R * Interval(float(model.h_m_m_s)) * w[i]) - (
                faces[i - 1] if i else Interval()
            )
            rows.append((scale * f / Interval(float(model.grid.volume_weights_m2[i])) / w[i]).hi)
        assert min(coop) > 0 and min((c.lo for c in boxes)) > 0
        mu = max(Decimal(0), max(rows))
        eta = max(((Interval(v) / wi).hi for v, wi in zip(raw["defect"], w)))
        h = Interval(right) - Interval(left)
        a0 = amplitude
        amplitude = ((Interval(a0) + Interval(eta) * h) * (Interval(mu) * h).exp()).hi
        records.append(
            {
                "left_s": str(left),
                "right_s": str(right),
                "a0": str(a0),
                "a1": str(amplitude),
                "mu_weighted_upper_per_s": str(mu),
                "eta_weighted_upper_per_s": str(eta),
                "cooperative_numerator_lower": str(min(coop)),
                "raw_defect_max": str(max(raw["defect"])),
            }
        )
        if k % 5 == 0:
            print(
                json.dumps(
                    {
                        "case": case,
                        "weighted_slab": k + 1,
                        "amplitude": str(amplitude),
                        "mu": str(mu),
                        "elapsed_s": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    def a_at(k, t):
        z = records[k]
        h = Interval(t) - Interval(lefts[k])
        return (
            (Interval(Decimal(z["a0"])) + Interval(Decimal(z["eta_weighted_upper_per_s"])) * h)
            * (Interval(Decimal(z["mu_weighted_upper_per_s"])) * h).exp()
        ).hi

    def field(t):
        k = min(bisect.bisect_left(rights, t), len(q) - 1)
        u = (Interval(t) - Interval(lefts[k])) / (Interval(rights[k]) - Interval(lefts[k]))
        a = a_at(k, t)
        return [poly_value(v, u).padded((Interval(a) * wi).hi) for v, wi in zip(q[k], w)]

    def mx(t):
        box = field(t)
        return Interval(max((v.lo for v in box)), max((v.hi for v in box)))

    threshold = Decimal(".15")
    valid = amplitude < cap and mx(lefts[0]).lo > threshold and (mx(rights[-1]).hi < threshold)
    lo = lefts[0]
    hi = rights[-1]
    if valid:
        lo, hi = bracket_threshold(mx, lefts[0], rights[-1], threshold, threshold, 42)
    poly_derivative = None
    if valid:
        vv = [None] * n
        rr = []
        for k in range(len(q)):
            a = max(lo, lefts[k])
            b = min(hi, rights[k])
            if a > b:
                continue
            u = (Interval(a, b) - Interval(lefts[k])) / (Interval(rights[k]) - Interval(lefts[k]))
            aa = a_at(k, b)
            R, _ = radius_polynomial(model, float(lefts[k]), float(rights[k]))
            rr.append(poly_bound(R))
            for i in range(n):
                box = poly_value(q[k][i], u).padded((Interval(aa) * w[i]).hi)
                vv[i] = (
                    box if vv[i] is None else Interval(min(vv[i].lo, box.lo), max(vv[i].hi, box.hi))
                )
        R = Interval(min((x.lo for x in rr)), max((x.hi for x in rr)))
        scale = (Interval(float(model.grid.radius_m)) / R) ** 2
        active = [i for i, x in enumerate(vv) if x.hi >= max((v.lo for v in vv))]
        dd = [
            diffusivity_derivatives(
                v, Interval(float(model.temperature_reference_K)), model.properties
            )[0]
            for v in vv
        ]
        vals = []
        for i in active:
            f = Interval()
            for j in [i - 1, i + 1]:
                if j < 0 or j >= n:
                    continue
                k = min(i, j)
                g = Interval(float(model.grid.face_r_m[k])) / (
                    2 * Interval(float(model.grid.node_spacing_m[k]))
                )
                f = f + g * (dd[i] + dd[j]) * (vv[j] - vv[i])
            if i == n - 1:
                ceq = Interval(
                    float(model.ambient.values(float(lo), right_of_breakpoint=True).moisture_eq)
                )
                f = f - R * Interval(float(model.h_m_m_s)) * (vv[i] - ceq)
            vals.append(scale * f / Interval(float(model.grid.volume_weights_m2[i])))
        bound = Interval(min((v.lo for v in vals)), max((v.hi for v in vals)))
        poly_derivative = {
            "possible_active_nodes": active,
            "derivative_interval_per_s_decimal": bound.strings(),
            "strictly_negative_verified": bound.hi < 0,
        }
    ep = json.loads((args.runs / case / "endpoint.json").read_text())
    report = mx(Decimal(str(ep["reported_time_s"])))
    np.savez_compressed(out / "certificate_weights.npz", xi=model.xi, w=weights)
    result = {
        "completed": True,
        "conditional_semidiscrete_certificate_passed": valid,
        "continuous_PDE_certified": False,
        "global_t0_semidiscrete_error_certified": False,
        "case": case,
        "N": n - 1,
        "initial_C_error_bound": "0",
        "initial_T_error_bound_K": "0",
        "checkpoint_s": str(lefts[0]),
        "weight_definition": "binary64 exp(6*(xi^2-1)), stored exactly as input constants in certificate_weights.npz",
        "amplitude_cap": str(cap),
        "final_amplitude_bound": str(amplitude),
        "final_centre_C_error_bound": str((Interval(amplitude) * w[0]).hi),
        "root_interval_s_decimal": [str(lo), str(hi)] if valid else None,
        "root_interval_width_s": float(hi - lo) if valid else None,
        "transversality": poly_derivative,
        "production_report_C_interval_decimal": report.strings(),
        "production_report_conditionally_strictly_dry": bool(valid and report.hi < threshold),
        "slabs": records,
        "source_sha256": current_input_hashes(args, case),
        "scope": (
            "Only the semidiscrete IVP starting from the exact recorded late checkpoint. "
            "No inherited checkpoint or spatial/PDE error is certified."
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    write_record(out / "weighted_conditional_certificate.json", result)
    print(
        json.dumps(
            {
                "case": case,
                "passed": valid,
                "width_s": result["root_interval_width_s"],
                "transversality": poly_derivative,
            }
        ),
        flush=True,
    )


def exact_radius(model, t):
    """以有理数重构半径节点之间的精确线性插值。"""
    g = model.geometry
    if not hasattr(g, "t_s"):
        return Fraction(float(g.radius(t)))
    j = int(np.searchsorted(g.t_s, t, side="right") - 1)
    if j >= len(g.t_s) - 1:
        return Fraction(float(g.tail_radius_m))
    return Fraction(float(g.radius_m[j])) + (
        Fraction(float(g.radius_m[j + 1])) - Fraction(float(g.radius_m[j]))
    ) * (Fraction(float(t)) - Fraction(float(g.t_s[j]))) / (
        Fraction(float(g.t_s[j + 1])) - Fraction(float(g.t_s[j]))
    )


@decimal_arithmetic
def audit_interfaces(args, case, reuse=None):
    """核对全状态/时间接口、恒温常量、节点单调性和样条接口条件。"""
    path = (reuse or args.reuse) / case / "local_radau.npz"
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    n = len(d["xi"])
    m = load_model(args, case, n - 1)
    ep = json.loads((args.runs / case / "endpoint.json").read_text())
    report = Decimal(str(ep["reported_time_s"]))
    first = Decimal.from_float(float(d["left"][0]))
    last = Decimal.from_float(float(d["right"][-1]))
    adjacent = np.array_equal(d["right"][:-1], d["left"][1:]) and np.array_equal(
        d["y_new"][:-1], d["y_old"][1:]
    )
    assert adjacent and first <= report <= last
    assert (
        np.array_equal(d["xi"], m.xi)
        and np.all(np.diff(m.xi) > 0)
        and (m.xi[0] == 0)
        and (m.xi[-1] == 1)
    )
    assert np.all(d["y_old"][:, :n] == 0) and np.all(d["y_new"][:, :n] == 0)
    derivative_upper = None
    centre_upper = None
    ambient = []
    radii = []
    for k in range(len(d["left"])):
        q = nodal_polynomials(d, k, n)
        h = Interval(float(d["right"][k])) - Interval(float(d["left"][k]))
        for i, v in enumerate(q):
            bb = poly_bound([v[j] * j / h for j in range(1, 4)]).hi
            derivative_upper = bb if derivative_upper is None else max(derivative_upper, bb)
            if i == 0:
                centre_upper = bb if centre_upper is None else max(centre_upper, bb)
        env = m.ambient.values(float(d["left"][k]), right_of_breakpoint=True)
        ambient.append((float(env.moisture_eq).hex(), float(m.temperature_reference_K).hex()))
        if k + 1 < len(d["left"]):
            t = float(d["right"][k])
            assert exact_radius(m, t) == exact_radius(m, float(d["left"][k + 1]))
            radii.append(str(exact_radius(m, t)))
    constant = len(set(ambient)) == 1
    result = {
        "passed": bool(adjacent and constant and (centre_upper < 0) and (derivative_upper < 0)),
        "case": case,
        "N": n - 1,
        "window_s_decimal": [str(first), str(last)],
        "report_time_s_decimal": str(report),
        "report_inside_covered_window": True,
        "all_time_interfaces_same_binary_time_and_full_state": bool(adjacent),
        "thermal_theta_zero_at_all_stored_interfaces": True,
        "ambient_constants_identical_across_slabs": constant,
        "centre_polynomial_time_derivative_upper_per_s": str(centre_upper),
        "all_nodal_polynomial_time_derivative_upper_per_s": str(derivative_upper),
        "lower_barrier_maximum_nonincreasing": bool(derivative_upper < 0),
        "continuous_spline_interface_identity": {
            "verified": bool(adjacent and constant),
            "reason": (
                "The right endpoint polynomial equals y_new by its exact algebraic "
                "definition; this is bitwise the next y_old. At each shared exact time the "
                "affine radius, Ce, T and material constants coincide. The fixed strictly "
                "diagonally dominant clamped spline matrix and exact Robin slope therefore "
                "have identical defining right-hand sides. Uniqueness gives equality of the "
                "entire spatial spline, its first and second spatial derivatives; this is "
                "stronger than equality only at nodes."
            ),
            "shared_exact_radius_fraction": radii,
        },
        "parameter_scope": (
            "Frozen binary64 input, derived tail mean, grid and material constants "
            "treated as exact reals. Decimal problem inputs and tail-mean computation "
            "roundoff are not enclosed."
        ),
        "local_radau_sha256": sha256(path),
        "endpoint_sha256": sha256(args.runs / case / "endpoint.json"),
        "guard_source_sha256": sha256(Path(__file__)),
    }
    return result


def poly_derivative(p, h):
    """对时间多项式求导，并除以当前时间段长度。"""
    return [p[j] * j / h for j in range(1, len(p))] or [Interval()]


def inverse_diffusivity_series(q, p, T, order=10, rho=64):
    """计算Robin斜率中的扩散率倒数级数及Cauchy余项。"""
    inv = [1 / q[0]]
    for n in range(1, order + 1):
        inv.append(-sum((q[j] * inv[n - j] for j in range(1, min(n, 3) + 1)), Interval()) / q[0])
    a = Interval(float(p.D_moisture_exponent))
    b = Interval(float(p.D_temperature_exponent_K))
    pref = Interval(float(p.D_prefactor))
    z = [a * v for v in inv]
    z[0] = z[0] + b / T
    series = [z[0].exp() / pref]
    for n in range(1, order + 1):
        series.append(sum((k * z[k] * series[n - k] for k in range(1, n + 1)), Interval()) / n)
    rad = Interval(rho)
    s = sum((Interval(q[j].absmax()) * rad**j for j in range(1, len(q))), Interval()).hi
    assert q[0].lo > s
    B = (a / (q[0] - s) + b / T).exp() / pref
    x = 1 / rad
    tail = (B * x ** (order + 1) / (1 - x)).hi
    derivative_tail = (B * x ** (order + 1) * ((order + 1) / (1 - x) + x / (1 - x) ** 2)).hi
    return (series, tail, derivative_tail)


def build_hermite_cells(model, q, left, right):
    """构造C1三次Hermite空间单元及边界斜率余项。"""
    n = len(q)
    x = [Interval(float(v)) for v in model.xi]
    slopes = [[Interval()]]
    for i in range(1, n - 1):
        hl = x[i] - x[i - 1]
        hr = x[i + 1] - x[i]
        slopes.append(
            poly_add(
                poly_scale(poly_subtract(q[i + 1], q[i]), hl / (hr * (hl + hr))),
                poly_scale(poly_subtract(q[i], q[i - 1]), hr / (hl * (hl + hr))),
            )
        )
    R, _ = radius_polynomial(model, left, right)
    T = Interval(float(model.temperature_reference_K))
    hm = Interval(float(model.h_m_m_s))
    ceq = Interval(float(model.ambient.values(left, right_of_breakpoint=True).moisture_eq))
    ht = Interval(right) - Interval(left)
    inverse, tail, dtail = inverse_diffusivity_series(q[-1], model.properties, T)
    A = poly_multiply(R, poly_subtract(q[-1], [ceq]))
    sp = poly_scale(poly_multiply(A, inverse), -hm)
    slope_error = (hm * Interval(poly_absmax(A)) * Interval(tail)).hi
    slope_time_error = (
        hm
        * (
            Interval(poly_absmax(poly_derivative(A, Interval(1)))) * Interval(tail)
            + Interval(poly_absmax(A)) * Interval(dtail)
        )
        / ht
    ).hi
    slopes.append(sp)
    cells = []
    for i in range(n - 1):
        h = x[i + 1] - x[i]
        dq = poly_subtract(q[i + 1], q[i])
        coeff = [
            q[i],
            poly_scale(slopes[i], h),
            poly_subtract(
                poly_scale(dq, 3),
                poly_scale(poly_add(poly_scale(slopes[i], 2), slopes[i + 1]), h),
            ),
            poly_add(poly_scale(dq, -2), poly_scale(poly_add(slopes[i], slopes[i + 1]), h)),
        ]
        cells.append((coeff, [poly_derivative(c, ht) for c in coeff], h))
    return (cells, slope_error, slope_time_error, R)


def evaluate_hermite_cell(coeff, dtcoeff, h, u, v, xi_left, is_last, es, est):
    """共同包围含水率、时间导数、径向导数、拉普拉斯和轴线极限。"""
    a = [poly_value(c, u) for c in coeff]
    at = [poly_value(c, u) for c in dtcoeff]
    value = poly_value(a, v)
    tt = poly_value(at, v)
    dx = (a[1] + 2 * a[2] * v + 3 * a[3] * v * v) / h
    dxx = (2 * a[2] + 6 * a[3] * v) / (h * h)
    xx = xi_left + h * v
    if xi_left.lo == 0:
        ratio = (2 * a[2] + 3 * a[3] * v) / (h * h)
        lap = (4 * a[2] + 9 * a[3] * v) / (h * h)
    else:
        ratio = dx / xx
        lap = dxx + ratio
    if is_last:
        value = value.padded((h * Interval(es)).hi)
        tt = tt.padded((h * Interval(est)).hi)
        dx = dx.padded((5 * Interval(es)).hi)
        extra = 4 * Interval(es) / h + 5 * Interval(es) / xx
        lap = lap.padded(extra.hi)
        ratio = ratio.padded((5 * Interval(es) / xx).hi)
    return (value, tt, dx, lap, ratio, xx)


@decimal_arithmetic
def bound_direct_slab(model, q, left, right, cap, sub_t, sub_x, reconstruction="hermite"):
    builder = build_spline_cells if reconstruction == "spline" else build_hermite_cells
    evaluator = evaluate_spline_cell if reconstruction == "spline" else evaluate_hermite_cell
    "在全部空间/时间矩形上直接评价连续残差和比较条件。"
    cells, es, est, Rpoly = builder(model, q, left, right)
    T = Interval(float(model.temperature_reference_K))
    p = model.properties
    a = Interval(float(p.D_moisture_exponent))
    hm = Interval(float(model.h_m_m_s))
    ceq = Interval(float(model.ambient.values(left, right_of_breakpoint=True).moisture_eq))
    eta = Decimal(0)
    mu = Decimal(0)
    beta = None
    monotone = None
    minC = None
    minD = None
    where = None
    for jt in range(sub_t):
        u = Interval(Decimal(jt) / Decimal(sub_t), Decimal(jt + 1) / Decimal(sub_t))
        R = poly_value(Rpoly, u)
        qs = poly_value(q[-1], u)
        ds = diffusivity_derivatives(qs, T, p)[0]
        s_exact = -R * hm * (qs - ceq) / ds
        us = qs + Interval(-cap, cap)
        du, dcu, _ = diffusivity_derivatives(us, T, p)
        bb = dcu * (s_exact + 12 * Interval(-cap, cap)) + 12 * du + R * hm
        beta = bb.lo if beta is None else min(beta, bb.lo)
        for i, (coeff, dtcoeff, h) in enumerate(cells):
            for jx in range(sub_x):
                v = Interval(Decimal(jx) / Decimal(sub_x), Decimal(jx + 1) / Decimal(sub_x))
                xi = Interval(float(model.xi[i]))
                c, ct, cx, lap, ratio, x = evaluator(
                    coeff, dtcoeff, h, u, v, xi, i == len(cells) - 1, es, est
                )
                w = (6 * (x * x - 1)).exp()
                d, dc, _ = diffusivity_derivatives(c, T, p)
                residual = ct - (d * lap + dc * cx * cx) / (R * R)
                ee = (Interval(residual.absmax()) / w).hi
                if ee > eta:
                    eta = ee
                    where = {
                        "space_cell": i,
                        "time_part": jt,
                        "space_part": jx,
                        "xi_interval": x.strings(),
                    }
                U = c + Interval(-cap, cap) * w
                DD, DC, _ = diffusivity_derivatives(U, T, p)
                DCC = DD * (a * a / U**4 - 2 * a / U**3)
                log_derivative = 12 * x
                L = 24 + 144 * x * x
                shifted_gradient = cx + Interval(-cap, cap) * w * log_derivative
                deriv = (
                    DC * (lap + Interval(-cap, cap) * w * L)
                    + DD * L
                    + DCC * shifted_gradient * shifted_gradient
                    + 2 * DC * shifted_gradient * log_derivative
                ) / (R * R)
                mu = max(mu, deriv.hi)
                mm = (ratio + 12 * Interval(cap) * w).hi
                monotone = mm if monotone is None else max(monotone, mm)
                minC = U.lo if minC is None else min(minC, U.lo)
                minD = DD.lo if minD is None else min(minD, DD.lo)
    return {
        "eta": eta,
        "mu": mu,
        "beta_lower": beta,
        "upper_radial_derivative_over_xi_upper": monotone,
        "minimum_C": minC,
        "minimum_D": minD,
        "maximum_weighted_residual_location": where,
        "Robin_slope_series_error": es,
        "Robin_slope_time_derivative_series_error": est,
    }


@decimal_arithmetic
def verify_continuous_tube(args, case, slab_verifier):
    """逐时间段传播连续屏障，检查闭合、报告时刻和条件根区间。"""
    dest = args.out / case
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(args.reuse / case / "local_radau.npz") as z:
        data = {k: z[k] for k in z.files}
    n = len(data["xi"])
    model = load_model(args, case, n - 1)
    cap = Decimal(args.cap)
    records = []
    amplitude = Decimal(0)
    lefts = [as_decimal(float(t)) for t in data["left"]]
    rights = [as_decimal(float(t)) for t in data["right"]]
    q = [nodal_polynomials(data, j, n) for j in range(len(rights))]
    started = time.perf_counter()
    for j, (left, right) in enumerate(zip(lefts, rights)):
        b = slab_verifier(
            model, q[j], float(left), float(right), cap, args.sub_time, args.sub_space
        )
        h = Interval(right) - Interval(left)
        a0 = amplitude
        amplitude = ((Interval(a0) + Interval(b["eta"]) * h) * (Interval(b["mu"]) * h).exp()).hi
        row = {k: str(v) if isinstance(v, Decimal) else v for k, v in b.items()}
        row.update(
            {
                "left_s": str(left),
                "right_s": str(right),
                "a0": str(a0),
                "a1": str(amplitude),
            }
        )
        records.append(row)
        write_record(
            dest / "progress.json",
            {
                "slabs_completed": j + 1,
                "slabs_total": len(q),
                "last": row,
                "elapsed_s": time.perf_counter() - started,
            },
        )
        print(
            json.dumps(
                {
                    "case": case,
                    "continuous_slab": j + 1,
                    "eta": str(b["eta"]),
                    "mu": str(b["mu"]),
                    "beta": str(b["beta_lower"]),
                    "monotone": str(b["upper_radial_derivative_over_xi_upper"]),
                    "amplitude": str(amplitude),
                    "elapsed_s": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        if args.first_slab_only:
            break
    full = len(records) == len(q)
    assumptions = all(
        (
            Decimal(r["beta_lower"]) >= 0
            and Decimal(r["upper_radial_derivative_over_xi_upper"]) <= 0
            and (Decimal(r["minimum_C"]) > 0)
            and (Decimal(r["minimum_D"]) > 0)
            for r in records
        )
    )
    state_closed = full and assumptions and (amplitude < cap)
    w0 = Interval(-6).exp()

    def mx(t):
        j = min(bisect.bisect_left(rights, t), len(q) - 1)
        u = (Interval(t) - Interval(lefts[j])) / (Interval(rights[j]) - Interval(lefts[j]))
        r = records[j]
        h = Interval(t) - Interval(lefts[j])
        amp = (
            (Interval(Decimal(r["a0"])) + Interval(Decimal(r["eta"])) * h)
            * (Interval(Decimal(r["mu"])) * h).exp()
        ).hi
        return poly_value(q[j][0], u).padded((Interval(amp) * w0).hi)

    bracketed = False
    lo = lefts[0]
    hi = rights[-1]
    report = None
    if state_closed:
        bracketed = mx(lo).lo > Decimal(".15") and mx(hi).hi < Decimal(".15")
        if bracketed:
            a = lo
            b = hi
            for _ in range(40):
                mid = (a + b) / 2
                if mx(mid).lo > Decimal(".15"):
                    a = mid
                else:
                    b = mid
            lo = a
            a = lefts[0]
            b = rights[-1]
            for _ in range(40):
                mid = (a + b) / 2
                if mx(mid).hi < Decimal(".15"):
                    b = mid
                else:
                    a = mid
            hi = b
        ep = json.loads((args.runs / case / "endpoint.json").read_text())
        report = mx(Decimal(str(ep["reported_time_s"])))
    result = {
        "completed": full,
        "trial_first_slab_only": args.first_slab_only,
        "case": case,
        "N": n - 1,
        "conditional_continuous_late_PDE_barrier_verified": state_closed,
        "event_bracketed": bracketed,
        "original_global_coupled_PDE_certified": False,
        "inherited_initial_profile_error_certified": False,
        "conditions": [
            "Exact isothermal temperature T=T_tail at the late initial time; not merely numerically close.",
            (
                "Continuous initial moisture equals the explicitly defined C1 Hermite "
                "function, with exact Robin surface slope."
            ),
            (
                "Bounds apply to regular solutions of this scalar uniformly parabolic late "
                "problem; original early evolution is not enclosed."
            ),
        ],
        "space_time_interval_subdivisions": [args.sub_space, args.sub_time],
        "amplitude_cap": str(cap),
        "final_amplitude_bound": str(amplitude),
        "final_centre_error_bound": str((Interval(amplitude) * w0).hi),
        "root_interval_s_decimal": [str(lo), str(hi)] if bracketed else None,
        "root_interval_width_s": float(hi - lo) if bracketed else None,
        "original_report_C_interval_decimal": report.strings() if report else None,
        "original_report_strict_under_late_conditions": bool(
            report is not None and report.hi < Decimal(".15")
        ),
        "slabs": records,
        "source_sha256": current_input_hashes(args, case),
        "elapsed_s": time.perf_counter() - started,
    }
    write_record(
        dest / ("trial.json" if args.first_slab_only else "continuous_certificate.json"),
        result,
    )
    print(
        json.dumps(
            {
                "case": case,
                "continuous_state_barrier": state_closed,
                "bracketed": bracketed,
                "width_s": result["root_interval_width_s"],
                "report_dry": result["original_report_strict_under_late_conditions"],
            }
        ),
        flush=True,
    )


def solve_tridiagonal(lower, diag, upper, rhs):
    """Outward interval Thomas solve of a strictly diagonally dominant system."""
    d = list(diag)
    b = list(rhs)
    n = len(d)
    for i in range(n):
        assert diag[i].lo > (Interval(lower[i].absmax()) + Interval(upper[i].absmax())).hi
    for i in range(1, n):
        assert d[i - 1].lo > 0
        f = lower[i] / d[i - 1]
        d[i] = d[i] - f * upper[i - 1]
        b[i] = b[i] - f * b[i - 1]
    assert d[-1].lo > 0
    x = [Interval() for _ in d]
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - upper[i] * x[i + 1]) / d[i]
    return x


def build_spline_cells(model, q, left, right):
    """用区间Thomas解构造C2样条，余项与每个空间多项式共同保存。"""
    n = len(q)
    x = [Interval(float(v)) for v in model.xi]
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    R, _ = radius_polynomial(model, left, right)
    hm = Interval(float(model.h_m_m_s))
    ceq = Interval(float(model.ambient.values(left, right_of_breakpoint=True).moisture_eq))
    T = Interval(float(model.temperature_reference_K))
    ht = Interval(right) - Interval(left)
    inv, tail, dtail = inverse_diffusivity_series(q[-1], model.properties, T)
    A = poly_multiply(R, poly_subtract(q[-1], [ceq]))
    slope = poly_scale(poly_multiply(A, inv), -hm)
    es = (hm * Interval(poly_absmax(A)) * Interval(tail)).hi
    est = (
        hm
        * (
            Interval(poly_absmax(poly_derivative(A, Interval(1)))) * Interval(tail)
            + Interval(poly_absmax(A)) * Interval(dtail)
        )
        / ht
    ).hi
    lower = [Interval()] + h[:-1] + [h[-1]]
    upper = [h[0]] + h[1:] + [Interval()]
    diag = [2 * h[0]] + [2 * (h[i - 1] + h[i]) for i in range(1, n - 1)] + [2 * h[-1]]
    diff = [poly_scale(poly_subtract(q[i + 1], q[i]), 1 / h[i]) for i in range(n - 1)]
    rhs = (
        [poly_scale(diff[0], 6)]
        + [poly_scale(poly_subtract(diff[i], diff[i - 1]), 6) for i in range(1, n - 1)]
        + [poly_scale(poly_subtract(slope, diff[-1]), 6)]
    )
    degree = max(map(len, rhs))
    moments = [[] for _ in range(n)]
    for k in range(degree):
        sol = solve_tridiagonal(
            lower, diag, upper, [r[k] if k < len(r) else Interval() for r in rhs]
        )
        for i in range(n):
            moments[i].append(sol[i])
    influence = solve_tridiagonal(
        lower, diag, upper, [Interval() for _ in range(n - 1)] + [Interval(6)]
    )
    cells = []
    for i in range(n - 1):
        H = h[i]
        hh = H * H
        mi = moments[i]
        mj = moments[i + 1]
        dq = poly_subtract(q[i + 1], q[i])
        coeff = [
            q[i],
            poly_subtract(dq, poly_scale(poly_add(poly_scale(mi, 2), mj), hh / 6)),
            poly_scale(mi, hh / 2),
            poly_scale(poly_subtract(mj, mi), hh / 6),
        ]
        z0 = influence[i]
        z1 = influence[i + 1]
        basis = [Interval(), -hh * (2 * z0 + z1) / 6, hh * z0 / 2, hh * (z1 - z0) / 6]
        data = {"basis": basis}
        if i > 0 and i < 32:
            b1 = poly_scale(coeff[3], 3 / hh)
            b0 = poly_subtract(poly_scale(coeff[2], 2 / hh), poly_scale(b1, x[i] / H))
            rem = poly_subtract(poly_scale(coeff[1], 1 / H), poly_scale(b0, x[i]))
            data["ratio"] = [b0, b1, rem]
        coeff = SpatialPolynomial(coeff, data)
        cells.append((coeff, [poly_derivative(c, ht) for c in coeff], H))
    return (cells, es, est, R)


def evaluate_spline_cell(coeff, dtcoeff, h, u, v, xi_left, is_last, es, est):
    """计算C2样条及Robin斜率余项传播后的共同区间界。"""
    value, tt, dx, lap, ratio, x = evaluate_hermite_cell(
        coeff, dtcoeff, h, u, v, xi_left, False, Decimal(0), Decimal(0)
    )
    d = coeff.remainder
    b = d["basis"]
    hh = h * h
    if "ratio" in d:
        b0, b1, rem = d["ratio"]
        ratio = poly_value(b0, u) + poly_value(b1, u) * v + poly_value(rem, u) / x
        a2 = poly_value(coeff[2], u)
        a3 = poly_value(coeff[3], u)
        lap = (2 * a2 + 6 * a3 * v) / hh + ratio
    bv = poly_value(b, v)
    bx = (b[1] + 2 * b[2] * v + 3 * b[3] * v * v) / h
    if xi_left.lo == 0:
        bratio = (2 * b[2] + 3 * b[3] * v) / hh
        blap = (4 * b[2] + 9 * b[3] * v) / hh
    else:
        bratio = bx / x
        blap = (2 * b[2] + 6 * b[3] * v) / hh + bratio
    return (
        value.padded((Interval(bv.absmax()) * Interval(es)).hi),
        tt.padded((Interval(bv.absmax()) * Interval(est)).hi),
        dx.padded((Interval(bx.absmax()) * Interval(es)).hi),
        lap.padded((Interval(blap.absmax()) * Interval(es)).hi),
        ratio.padded((Interval(bratio.absmax()) * Interval(es)).hi),
        x,
    )


def meanvalue_residual(cells, i, u, v, model, R, es, est):
    """取直接区间残差与空间均值形式的交集，以减小区间过估计。"""
    coeff, dtcoeff, h = cells[i]
    x0 = Interval(float(model.xi[i]))
    T = Interval(float(model.temperature_reference_K))
    c, ct, cx, lap, ratio, x = evaluate_spline_cell(coeff, dtcoeff, h, u, v, x0, False, es, est)
    p = model.properties
    DD, DC, _ = diffusivity_derivatives(c, T, p)
    aa = Interval(float(p.D_moisture_exponent))
    DCC = DD * (aa * aa / c**4 - 2 * aa / c**3)
    direct = ct - (DD * lap + DC * cx * cx) / (R * R)
    mid = (v.lo + v.hi) / 2
    cc, ctc, cxc, lapc, _, _ = evaluate_spline_cell(
        coeff, dtcoeff, h, u, Interval(mid), x0, False, es, est
    )
    dd, dc, _ = diffusivity_derivatives(cc, T, p)
    center = ctc - (dd * lapc + dc * cxc * cxc) / (R * R)
    a = [poly_value(z, u) for z in coeff]
    at = [poly_value(z, u) for z in dtcoeff]
    h2 = h * h
    h3 = h2 * h
    b = coeff.remainder["basis"]
    cxx = (2 * a[2] + 6 * a[3] * v) / h2
    ctx = (at[1] + 2 * at[2] * v + 3 * at[3] * v * v) / h
    bx = (b[1] + 2 * b[2] * v + 3 * b[3] * v * v) / h
    bxx = (2 * b[2] + 6 * b[3] * v) / h2
    cxx = cxx.padded((Interval(bxx.absmax()) * Interval(es)).hi)
    ctx = ctx.padded((Interval(bx.absmax()) * Interval(est)).hi)
    if x0.lo == 0:
        lapx = 9 * a[3] / h3
        blapx = 9 * b[3] / h3
    elif "ratio" in coeff.remainder:
        b0, b1, rem = coeff.remainder["ratio"]
        lapx = 6 * a[3] / h3 + poly_value(b1, u) / h - poly_value(rem, u) / (x * x)
        blapx = 6 * b[3] / h3 + bxx / x - bx / (x * x)
    else:
        lapx = 6 * a[3] / h3 + cxx / x - cx / (x * x)
        blapx = 6 * b[3] / h3 + bxx / x - bx / (x * x)
    lapx = lapx.padded((Interval(blapx.absmax()) * Interval(es)).hi)
    rx = ctx - (DD * lapx + DC * cx * (lap + 2 * cxx) + DCC * cx * cx * cx) / (R * R)
    meanvalue = center + rx * h * (v - Interval(mid))
    lo = max(direct.lo, meanvalue.lo)
    hi = min(direct.hi, meanvalue.hi)
    assert lo <= hi
    return (Interval(lo, hi), (c, ct, cx, lap, ratio, x))


@decimal_arithmetic
def bound_meanvalue_slab(model, q, left, right, cap, sub_t, sub_x):
    """覆盖所有矩形并汇总C2残差、边界正性与径向单调性条件。"""
    cells, es, est, Rpoly = build_spline_cells(model, q, left, right)
    T = Interval(float(model.temperature_reference_K))
    p = model.properties
    a = Interval(float(p.D_moisture_exponent))
    hm = Interval(float(model.h_m_m_s))
    ceq = Interval(float(model.ambient.values(left, right_of_breakpoint=True).moisture_eq))
    eta = Decimal(0)
    mu = Decimal(0)
    beta = None
    mono = None
    cmin = None
    dmin = None
    where = None
    for jt in range(sub_t):
        u = Interval(Decimal(jt) / Decimal(sub_t), Decimal(jt + 1) / Decimal(sub_t))
        R = poly_value(Rpoly, u)
        qs = poly_value(q[-1], u)
        s = -R * hm * (qs - ceq) / diffusivity_derivatives(qs, T, p)[0]
        us = qs + Interval(-cap, cap)
        du, dcu, _ = diffusivity_derivatives(us, T, p)
        bb = dcu * (s + 12 * Interval(-cap, cap)) + 12 * du + R * hm
        beta = bb.lo if beta is None else min(beta, bb.lo)
        for i in range(len(cells)):
            for j in range(sub_x):
                v = Interval(Decimal(j) / Decimal(sub_x), Decimal(j + 1) / Decimal(sub_x))
                res, values = meanvalue_residual(cells, i, u, v, model, R, es, est)
                c, ct, cx, lap, ratio, x = values
                w = (6 * (x * x - 1)).exp()
                ee = (Interval(res.absmax()) / w).hi
                if ee > eta:
                    eta = ee
                    where = {
                        "cell": i,
                        "xi_interval": x.strings(),
                        "time_part": jt,
                        "space_part": j,
                    }
                U = c + Interval(-cap, cap) * w
                DD, DC, _ = diffusivity_derivatives(U, T, p)
                DCC = DD * (a * a / U**4 - 2 * a / U**3)
                l = 12 * x
                K = 24 + 144 * x * x
                g = cx + Interval(-cap, cap) * w * l
                mm = (
                    DC * (lap + Interval(-cap, cap) * w * K) + DD * K + DCC * g * g + 2 * DC * g * l
                ) / (R * R)
                mu = max(mu, mm.hi)
                mon = (ratio + 12 * Interval(cap) * w).hi
                mono = mon if mono is None else max(mono, mon)
                cmin = U.lo if cmin is None else min(cmin, U.lo)
                dmin = DD.lo if dmin is None else min(dmin, DD.lo)
    return {
        "eta": eta,
        "mu": mu,
        "beta_lower": beta,
        "upper_radial_derivative_over_xi_upper": mono,
        "minimum_C": cmin,
        "minimum_D": dmin,
        "maximum_weighted_residual_location": where,
        "Robin_slope_series_error": es,
        "Robin_slope_time_derivative_series_error": est,
    }


@decimal_arithmetic
def bound_adaptive_slab(model, q, left, right, cap, sub_t, sub_x):
    """细分残差过宽的矩形，每个终止矩形仍保留完整向外界。"""
    cells, es, est, Rpoly = build_hermite_cells(model, q, left, right)
    T = Interval(float(model.temperature_reference_K))
    p = model.properties
    a = Interval(float(p.D_moisture_exponent))
    hm = Interval(float(model.h_m_m_s))
    ceq = Interval(float(model.ambient.values(left, right_of_breakpoint=True).moisture_eq))
    eta = Decimal(0)
    mu = Decimal(0)
    beta = None
    mono = None
    cmin = None
    dmin = None
    where = None
    leaf_count = 0
    refined = 0
    deepest = 0
    target = Decimal("1e-7")

    def evaluate(i, u, v):
        coeff, dtcoeff, h = cells[i]
        R = poly_value(Rpoly, u)
        c, ct, cx, lap, ratio, x = evaluate_hermite_cell(
            coeff,
            dtcoeff,
            h,
            u,
            v,
            Interval(float(model.xi[i])),
            i == len(cells) - 1,
            es,
            est,
        )
        w = (6 * (x * x - 1)).exp()
        d, dc, _ = diffusivity_derivatives(c, T, p)
        residual = ct - (d * lap + dc * cx * cx) / (R * R)
        ee = (Interval(residual.absmax()) / w).hi
        lower_abs = (
            min(abs(residual.lo), abs(residual.hi)) if residual.lo * residual.hi > 0 else Decimal(0)
        )
        lower_weighted = (Interval(lower_abs) / w).lo
        U = c + Interval(-cap, cap) * w
        DD, DC, _ = diffusivity_derivatives(U, T, p)
        DCC = DD * (a * a / U**4 - 2 * a / U**3)
        ld = 12 * x
        L = 24 + 144 * x * x
        grad = cx + Interval(-cap, cap) * w * ld
        deriv = (
            DC * (lap + Interval(-cap, cap) * w * L)
            + DD * L
            + DCC * grad * grad
            + 2 * DC * grad * ld
        ) / (R * R)
        return (
            ee,
            lower_weighted,
            deriv.hi,
            (ratio + 12 * Interval(cap) * w).hi,
            U.lo,
            DD.lo,
            x,
        )

    for jt in range(sub_t):
        u0 = Interval(Decimal(jt) / Decimal(sub_t), Decimal(jt + 1) / Decimal(sub_t))
        R = poly_value(Rpoly, u0)
        qs = poly_value(q[-1], u0)
        ds = diffusivity_derivatives(qs, T, p)[0]
        s = -R * hm * (qs - ceq)
        s = s / ds
        us = qs + Interval(-cap, cap)
        du, dcu, _ = diffusivity_derivatives(us, T, p)
        bb = dcu * (s + 12 * Interval(-cap, cap)) + 12 * du + R * hm
        beta = bb.lo if beta is None else min(beta, bb.lo)
        for i in range(len(cells)):
            pending = [
                (
                    u0,
                    Interval(Decimal(j) / Decimal(sub_x), Decimal(j + 1) / Decimal(sub_x)),
                    0,
                    0,
                )
                for j in range(sub_x)
            ]
            while pending:
                u, v, dx, dt = pending.pop()
                ev = evaluate(i, u, v)
                ee, lower, mm, mon, cc, dd, x = ev
                if ee > target and ee > max(Decimal(0), lower) * Decimal("1.05") and (dx + dt < 12):
                    if dx < 6:
                        mid = (v.lo + v.hi) / 2
                        pending.extend(
                            [
                                (u, Interval(v.lo, mid), dx + 1, dt),
                                (u, Interval(mid, v.hi), dx + 1, dt),
                            ]
                        )
                    else:
                        mid = (u.lo + u.hi) / 2
                        pending.extend(
                            [
                                (Interval(u.lo, mid), v, dx, dt + 1),
                                (Interval(mid, u.hi), v, dx, dt + 1),
                            ]
                        )
                    refined += 1
                    continue
                leaf_count += 1
                deepest = max(deepest, dx + dt)
                if ee > eta:
                    eta = ee
                    where = {
                        "space_cell": i,
                        "xi_interval": x.strings(),
                        "time_fraction": u.strings(),
                    }
                mu = max(mu, mm)
                mono = mon if mono is None else max(mono, mon)
                cmin = cc if cmin is None else min(cmin, cc)
                dmin = dd if dmin is None else min(dmin, dd)
    return {
        "eta": eta,
        "mu": mu,
        "beta_lower": beta,
        "upper_radial_derivative_over_xi_upper": mono,
        "minimum_C": cmin,
        "minimum_D": dmin,
        "maximum_weighted_residual_location": where,
        "Robin_slope_series_error": es,
        "Robin_slope_time_derivative_series_error": est,
        "accepted_interval_rectangles": leaf_count,
        "bisections": refined,
        "maximum_added_depth": deepest,
        "adaptive_target_weighted_residual_per_s": str(target),
    }


def exact_time_coefficients(data, slab, node):
    """精确有理数形式的节点三次多项式，与区间核独立构造。"""
    n = len(data["xi"])
    initial = Fraction(float(data["y_old"][slab, n + node]))
    linear = Fraction(float(data["Q"][slab, n + node, 0]))
    quadratic = Fraction(float(data["Q"][slab, n + node, 1]))
    cubic = Fraction(float(data["y_new"][slab, n + node])) - initial - linear - quadratic
    return initial, linear, quadratic, cubic


def exact_quadratic_maximum(a, b, c):
    """解析比较区间[0,1]端点及凹二次式内部顶点。"""
    values = [c, a + b + c]
    if a < 0:
        vertex = -b / (2 * a)
        if 0 < vertex < 1:
            values.append(a * vertex * vertex + b * vertex + c)
    return max(values)


def check_continuous_consistency(bundle, case):
    """用有理数极值和80位算术独立核对已存连续证书；空间残差不在此重算。"""
    with localcontext() as arithmetic:
        arithmetic.prec = 80
        certificate_path = bundle / case / "continuous_certificate.json"
        certificate = read_json(certificate_path)
        assert certificate["completed"] and certificate["event_bracketed"]
        assert not certificate["original_global_coupled_PDE_certified"]
        assert not certificate["inherited_initial_profile_error_certified"]
        with np.load(bundle / case / "local_radau.npz", allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        rows, n = certificate["slabs"], len(data["xi"])
        assert n == certificate["N"] + 1
        assert len(rows) == len(data["left"]) == len(data["right"])
        assert np.array_equal(data["y_new"][:-1], data["y_old"][1:])
        assert np.array_equal(data["right"][:-1], data["left"][1:])
        assert np.all(data["y_old"][:, :n] == 0) and np.all(data["y_new"][:, :n] == 0)
        lefts = [Decimal(row["left_s"]) for row in rows]
        rights = [Decimal(row["right_s"]) for row in rows]
        cap, previous = Decimal(certificate["amplitude_cap"]), Decimal(0)
        min_slack = maximum_derivative = maximum_centre_derivative = None
        checks = 0
        for j, row in enumerate(rows):
            left, right = lefts[j], rights[j]
            assert left == Decimal.from_float(float(data["left"][j]))
            assert right == Decimal.from_float(float(data["right"][j])) and right > left
            eta, mu, initial, final = (Decimal(row[key]) for key in ("eta", "mu", "a0", "a1"))
            assert initial == previous
            assert eta >= 0 and mu >= 0 and 0 <= initial <= final < cap
            length = right - left
            slack = final - (initial + eta * length) * (mu * length).exp()
            assert slack >= 0
            min_slack = slack if min_slack is None else min(min_slack, slack)
            assert Decimal(row["beta_lower"]) >= 0
            assert Decimal(row["upper_radial_derivative_over_xi_upper"]) <= 0
            assert Decimal(row["minimum_C"]) > 0 and Decimal(row["minimum_D"]) > 0
            assert Decimal(row["Robin_slope_series_error"]) >= 0
            assert Decimal(row["Robin_slope_time_derivative_series_error"]) >= 0
            previous = final
            exact_length = Fraction(float(data["right"][j])) - Fraction(float(data["left"][j]))
            for i in range(n):
                q = exact_time_coefficients(data, j, i)
                upper = exact_quadratic_maximum(3 * q[3], 2 * q[2], q[1]) / exact_length
                assert upper < 0
                maximum_derivative = (
                    upper if maximum_derivative is None else max(maximum_derivative, upper)
                )
                if i == 0:
                    maximum_centre_derivative = (
                        upper
                        if maximum_centre_derivative is None
                        else max(maximum_centre_derivative, upper)
                    )
                checks += 1

        def decimal_fraction(value):
            return Decimal(value.numerator) / Decimal(value.denominator)

        assert previous == Decimal(certificate["final_amplitude_bound"])
        assert decimal_fraction(maximum_derivative) <= Decimal(
            certificate["guards"]["all_nodal_polynomial_time_derivative_upper_per_s"]
        )
        assert decimal_fraction(maximum_centre_derivative) <= Decimal(
            certificate["guards"]["centre_polynomial_time_derivative_upper_per_s"]
        )
        assert previous * Decimal(-6).exp() <= Decimal(certificate["final_centre_error_bound"])

        def independent_bounds(t):
            assert lefts[0] <= t <= rights[-1]
            j = min(bisect.bisect_left(rights, t), len(rows) - 1)
            row = rows[j]
            u = Fraction(t - lefts[j]) / Fraction(rights[j] - lefts[j])
            q = exact_time_coefficients(data, j, 0)
            value = decimal_fraction(sum((q[k] * u**k for k in range(4)), Fraction(0)))
            elapsed = t - lefts[j]
            error = (Decimal(row["a0"]) + Decimal(row["eta"]) * elapsed) * (
                Decimal(row["mu"]) * elapsed - Decimal(6)
            ).exp()
            return value - error, value + error

        low, high = map(Decimal, certificate["root_interval_s_decimal"])
        low_bounds, high_bounds = independent_bounds(low), independent_bounds(high)
        assert low_bounds[0] > Decimal(".15") and high_bounds[1] < Decimal(".15")
        report_bounds = independent_bounds(Decimal(certificate["guards"]["report_time_s_decimal"]))
        saved_report = list(map(Decimal, certificate["original_report_C_interval_decimal"]))
        assert (
            saved_report[0]
            <= report_bounds[0]
            <= report_bounds[1]
            <= saved_report[1]
            < Decimal(".15")
        )
        query_path = bundle / "reports" / f"{case}_candidate_time_conditional_query.json"
        candidate_checked = False
        if query_path.is_file():
            query = read_json(query_path)
            assert query["source_sha256"]["continuous_certificate.json"] == sha256(certificate_path)
            bounds = independent_bounds(Decimal(query["query_time_s_decimal"]))
            saved = list(map(Decimal, query["true_maximum_interval_under_stated_conditions"]))
            assert saved[0] <= bounds[0] <= bounds[1] <= saved[1] < Decimal(".15")
            candidate_checked = True
        return {
            "passed": True,
            "slabs": len(rows),
            "exact_node_derivative_extrema_checks": checks,
            "minimum_amplitude_outward_slack_decimal": str(min_slack),
            "exact_maximum_node_time_derivative_decimal": str(decimal_fraction(maximum_derivative)),
            "root_lower_endpoint_above_threshold_decimal": str(low_bounds[0] - Decimal(".15")),
            "root_upper_endpoint_below_threshold_decimal": str(Decimal(".15") - high_bounds[1]),
            "original_report_contains_independent_80_digit_evaluation": True,
            "candidate_report_contains_independent_80_digit_evaluation": candidate_checked,
            "spatial_residual_enclosures_recomputed": False,
        }


@decimal_arithmetic
def check_uniform_domains(context, case):
    """逐节点核对Cauchy展开区域、正物性及统一管的条件范围。"""
    certificate = read_json(context.bundle / case / "conditional_certificate.json")
    with np.load(context.bundle / case / "local_radau.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    n = len(data["xi"])
    model = load_model(context, case, n - 1)
    cap = Decimal(certificate["uniform_C_tube_cap"])
    temperature_cap = Decimal(certificate["temperature_tube_cap_K"])
    powers = [Interval(64) ** j for j in range(4)]
    minimum_margin = minimum_moisture = Decimal("Infinity")
    maximum_ratio, count = Decimal(0), 0
    for j in range(len(data["left"])):
        for q in nodal_polynomials(data, j, n):
            variation = sum(
                (Interval(q[k].absmax()) * powers[k] for k in range(1, len(q))),
                Interval(),
            ).hi
            minimum_margin = min(minimum_margin, (q[0] - Interval(variation)).lo)
            maximum_ratio = max(maximum_ratio, (Interval(variation) / q[0]).hi)
            minimum_moisture = min(minimum_moisture, poly_bound(q).padded(cap).lo)
            count += 1
    description = read_json(context.bundle / "proof_manifest.json")["cases"][case]
    constants = {
        key: float.fromhex(value)
        for key, value in description["numeric_constants_binary64_hex"].items()
    }
    positive_thermal = all(constants[key] > 0 for key in ("rho_offset", "cp_offset", "k_offset"))
    positive_thermal &= all(
        constants[key] >= 0
        for key in (
            "rho_C_coefficient",
            "cp_fraction_coefficient",
            "k_fraction_coefficient",
        )
    )
    checks = {
        "all_arrays_finite": all(np.isfinite(value).all() for value in data.values()),
        "all_slabs_positive_length": bool(np.all(data["right"] > data["left"])),
        "positive_grid_weights_and_spacings": bool(
            np.all(model.grid.volume_weights_m2 > 0) and np.all(model.grid.node_spacing_m > 0)
        ),
        "all_cauchy_disks_avoid_zero": minimum_margin > 0,
        "whole_assumed_C_box_positive": minimum_moisture > 0,
        "positive_thermal_properties_on_positive_C": positive_thermal,
        "positive_T_box": Interval(float(model.temperature_reference_K)).padded(temperature_cap).lo
        > 0,
        "threshold_above_equilibrium": Interval(".15").lo > Interval(constants["Ceq"]).hi,
        "cooperative_numerator_lower_positive_in_all_recorded_slabs": all(
            Decimal(row["cooperative_numerator_lower"]) > 0 for row in certificate["slabs"]
        ),
        "finite_dimensional_only_flags_explicit": not certificate["continuous_PDE_certified"]
        and not certificate["global_t0_semidiscrete_error_certified"],
        "all_assumed_initial_boxes_fit_declared_caps": all(
            Decimal(row["initial_C_error_bound"]) <= cap
            and Decimal(row["initial_T_error_bound_K"]) <= temperature_cap
            for row in certificate["scenarios"]
        ),
    }
    assert all(checks.values()), checks
    return {
        "passed": True,
        "checks": checks,
        "polynomials_independently_domain_checked": count,
        "minimum_Cauchy_disk_margin": str(minimum_margin),
        "maximum_Cauchy_relative_variation": str(maximum_ratio),
        "minimum_C_in_whole_assumed_tube": str(minimum_moisture),
        "space_time_residual_enclosures_recomputed": False,
    }


@decimal_arithmetic
def arithmetic_checks():
    """用精确有理数核对四则运算，并独立检查Cauchy展开和C2轴线极限。"""
    random.seed(61925)
    for _ in range(1000):
        a, b = random.uniform(-10, 10), random.uniform(0.1, 10)
        left, right = Interval(a), Interval(b)
        exact_a, exact_b = Fraction.from_float(a), Fraction.from_float(b)
        for enclosure, exact in (
            (left + right, exact_a + exact_b),
            (left - right, exact_a - exact_b),
            (left * right, exact_a * exact_b),
            (left / right, exact_a / exact_b),
        ):
            assert Fraction(enclosure.lo) <= exact <= Fraction(enclosure.hi)
    for _ in range(100):
        q = [
            Interval(random.uniform(0.08, 0.16)),
            Interval(random.uniform(-2e-6, 0)),
            Interval(random.uniform(-1e-10, 1e-10)),
            Interval(random.uniform(-1e-14, 1e-14)),
        ]
        poly, remainder = diffusivity_series(
            q,
            Interval(0.0024),
            Interval(0.45),
            Interval(3850),
            Interval(323.14591666666666),
        )
        for x in (Decimal(0), Decimal(".125"), Decimal(".5"), Decimal(1)):
            box = poly_value(poly, Interval(x)).padded(remainder)
            with localcontext() as context:
                context.prec = 100
                c = q[0].lo + sum(q[j].lo * x**j for j in range(1, len(q)))
                truth = (
                    as_decimal(0.0024)
                    * (-as_decimal(0.45) / c - Decimal(3850) / as_decimal(323.14591666666666)).exp()
                )
            assert box.lo < truth < box.hi
    spline = spline_reproduction_checks()
    return {
        "passed": True,
        "exact_Fraction_operation_checks": 4000,
        "Cauchy_Taylor_direct_100_digit_checks": 400,
        "precision": DECIMAL_DIGITS,
        "spline_kernel": spline,
        "full_window_recomputed": False,
    }


@decimal_arithmetic
def spline_reproduction_checks():
    """精确二次解同时检验样条插值、径向拉普拉斯与均值残差包围。"""
    from types import SimpleNamespace

    xi = 1 - (1 - np.arange(17) / 16.0) ** 2
    model = SimpleNamespace(
        xi=xi,
        geometry=SimpleNamespace(radius=lambda t: 1.0),
        properties=DiffusionLaw(1.0, 0.0, 0.0),
        h_m_m_s=1.0,
        temperature_reference_K=300.0,
        ambient=ConstantAmbient(0.5, 300.0),
    )
    rate = Decimal(1) / Decimal(16384)
    q = []
    for x in xi:
        xx = Decimal.from_float(float(x))
        shape = Decimal("1.5") - xx * xx / 2
        q.append(
            [
                Interval(Decimal(".5") + shape),
                Interval(rate * shape),
                Interval(),
                Interval(),
            ]
        )
    cells, slope_error, slope_time_error, _ = build_spline_cells(model, q, 0.0, 2.0)
    checked = 0
    largest_width = Decimal(0)
    for i, (coeff, dtcoeff, h) in enumerate(cells):
        for u in (Decimal(0), Decimal(".125"), Decimal(".5"), Decimal(1)):
            for v in (Decimal(0), Decimal(".125"), Decimal(".5"), Decimal(1)):
                x0, x1 = Decimal.from_float(float(xi[i])), Decimal.from_float(float(xi[i + 1]))
                x = x0 + (x1 - x0) * v
                amplitude, shape = 1 + rate * u, Decimal("1.5") - x * x / 2
                expected = [
                    Decimal(".5") + amplitude * shape,
                    rate * shape / 2,
                    -amplitude * x,
                    -2 * amplitude,
                    -amplitude,
                    x,
                ]
                values = evaluate_spline_cell(
                    coeff,
                    dtcoeff,
                    h,
                    Interval(u),
                    Interval(v),
                    Interval(float(xi[i])),
                    i == len(cells) - 1,
                    slope_error,
                    slope_time_error,
                )
                for enclosure, target in zip(values, expected):
                    assert enclosure.lo <= target <= enclosure.hi
                    largest_width = max(largest_width, enclosure.hi - enclosure.lo)
                    checked += 1
    rectangles = 0
    for i in range(len(cells)):
        for ul, ur in ((Decimal(0), Decimal(".5")), (Decimal(".5"), Decimal(1))):
            for vl, vr in ((Decimal(0), Decimal(".5")), (Decimal(".5"), Decimal(1))):
                box, _ = meanvalue_residual(
                    cells,
                    i,
                    Interval(ul, ur),
                    Interval(vl, vr),
                    model,
                    Interval(1),
                    slope_error,
                    slope_time_error,
                )
                x0 = Decimal.from_float(float(xi[i]))
                width = Decimal.from_float(float(xi[i + 1])) - x0
                xmin, xmax = x0 + width * vl, x0 + width * vr
                lower = rate * (Decimal("1.5") - xmax * xmax / 2) / 2 + 2 * (1 + rate * ul)
                upper = rate * (Decimal("1.5") - xmin * xmin / 2) / 2 + 2 * (1 + rate * ur)
                assert box.lo <= lower <= upper <= box.hi
                rectangles += 1
    assert largest_width < Decimal("1e-12")
    return {
        "passed": True,
        "quantities_enclosed": checked,
        "meanvalue_rectangles": rectangles,
        "largest_interval_width": str(largest_width),
    }


@decimal_arithmetic
def query_maximum(bundle, case, time_decimal, certificates=None):
    """在已覆盖窗口内评价指定晚期C2初值的连续最大含水率界。"""
    certificate_path = (certificates or bundle) / case / "continuous_certificate.json"
    certificate = read_json(certificate_path)
    if not (
        certificate["completed"]
        and certificate["conditional_continuous_late_PDE_barrier_verified"]
        and certificate["guards"]["passed"]
    ):
        raise ValueError("查询需要完成且通过的连续屏障证书。")
    trajectory = bundle / case / "local_radau.npz"
    if sha256(trajectory) != certificate["source_sha256"]["local_radau.npz"]:
        raise ValueError("证书与轨迹身份不一致。")
    with np.load(trajectory, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    t = Decimal(time_decimal)
    left = [Decimal(row["left_s"]) for row in certificate["slabs"]]
    right = [Decimal(row["right_s"]) for row in certificate["slabs"]]
    if not left[0] <= t <= right[-1]:
        raise ValueError("查询时刻超出证书覆盖窗口。")
    j = min(bisect.bisect_left(right, t), len(right) - 1)
    slab = certificate["slabs"][j]
    h = Interval(t) - Interval(left[j])
    u = h / (Interval(right[j]) - Interval(left[j]))
    amplitude = (
        (Interval(Decimal(slab["a0"])) + Interval(Decimal(slab["eta"])) * h)
        * (Interval(Decimal(slab["mu"])) * h).exp()
    ).hi
    q = nodal_polynomials(data, j, len(data["xi"]))[0]
    maximum = poly_value(q, u).padded((Interval(amplitude) * Interval(-6).exp()).hi)
    return {
        "completed": True,
        "case": case,
        "query_time_s_decimal": str(t),
        "inside_covered_window": True,
        "true_maximum_interval_under_stated_conditions": maximum.strings(),
        "strictly_dry_under_stated_late_conditions": maximum.hi < Decimal(".15"),
        "conditions": certificate["conditions"],
        "original_global_t0_error_certified": False,
        "continuous_certificate_sha256": sha256(certificate_path),
        "local_radau_sha256": sha256(trajectory),
    }


def stage_case_data(bundle, output, case):
    dest = output / case
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "local_radau.npz",
        "certificate_weights.npz",
        "endpoint.json",
        "conditional_certificate.json",
    ):
        if (bundle / case / name).is_file():
            shutil.copyfile(bundle / case / name, dest / name)


def continuous_result(context, case, method="meanvalue"):
    """显式选择重构与残差算法，消除模块间函数替换。"""
    guards = audit_interfaces(context, case)
    if not guards["passed"]:
        raise ValueError("节点、接口或单调性前置检查未通过。")
    methods = {
        "meanvalue": bound_meanvalue_slab,
        "hermite": bound_direct_slab,
        "spline": partial(bound_direct_slab, reconstruction="spline"),
        "adaptive": bound_adaptive_slab,
    }
    verify_continuous_tube(context, case, methods[method])
    name = "trial.json" if context.first_slab_only else "continuous_certificate.json"
    path = context.out / case / name
    result = read_json(path)
    if method == "meanvalue":
        result["guards"] = guards
        result["conditions"][1] = (
            "Continuous initial moisture equals the unique clamped C2 cubic spline "
            "through the saved checkpoint; exact nonlinear Robin surface derivative."
        )
        result["conditions"].append(guards["parameter_scope"])
        result["residual_enclosure"] = (
            "Intersection of outward direct and spatial mean-value interval forms; all "
            "cells and all time fractions covered."
        )
    elif method == "spline":
        result["conditions"][1] = (
            "Continuous initial moisture equals the unique clamped C2 cubic spline "
            "through the recorded checkpoint, with exact nonlinear Robin surface slope."
        )
        result["reconstruction"] = (
            "C2 clamped cubic spline; exact tridiagonal defining equations enclosed by "
            "outward interval elimination; analytic slope and slope-time-derivative "
            "Cauchy remainders propagated by the spline influence function."
        )
    elif method == "adaptive":
        result["interval_partition"] = (
            "Adaptive bisection of initial rectangles; every final rectangle retains its "
            "outward enclosure, even if its width target is not reached."
        )
    write_record(path, result)
    return result


def make_context(output, bundle=DEFAULT_BUNDLE):
    return VerificationContext(
        output, output, output, bundle, bundle / "model.json", SUPPORT / "inputs/附件"
    )


def bind_snapshot_to_main_model(context, case):
    """逐常量、逐网格核对同包主模型与证明快照，并检查连续证书的固定半径窗口。"""
    manifest = read_json(context.bundle / "proof_manifest.json")
    description = manifest["cases"][case]
    current = make_context(context.out, context.bundle)
    current.production = True
    current.contract = SUPPORT / "code/config/model.json"
    if "input_sha256" in manifest:
        assert sha256(current.contract) == manifest["input_sha256"]["model.json"]
        for name in ("附件1.xlsx", "附件2.xlsx"):
            assert sha256(current.inputs / name) == manifest["input_sha256"][name]
    else:
        assert sha256(current.contract) == manifest["model_sha256"]
    model = load_model(current, case, description["N"])
    assert description["checkpoint_s"] > model.ambient.breakpoint_s
    constants = {
        name: float(getattr(model.properties, name)).hex()
        for name in (
            "rho_offset",
            "rho_C_coefficient",
            "cp_offset",
            "cp_fraction_coefficient",
            "k_offset",
            "k_fraction_coefficient",
            "D_prefactor",
            "D_moisture_exponent",
            "D_temperature_exponent_K",
        )
    }
    constants.update(
        R_reference_m=float(model.grid.radius_m).hex(),
        h=float(model.h_W_m2_K).hex(),
        hm=float(model.h_m_m_s).hex(),
        T_reference_K=float(model.temperature_reference_K).hex(),
        Ceq=float(
            model.ambient.values(description["checkpoint_s"], right_of_breakpoint=True).moisture_eq
        ).hex(),
    )
    assert constants == description["numeric_constants_binary64_hex"]
    with np.load(context.bundle / case / "proof_constants.npz", allow_pickle=False) as frozen:
        for name in ("r_m", "node_spacing_m", "face_r_m", "volume_weights_m2"):
            assert np.array_equal(frozen[name], getattr(model.grid, name))
        assert np.array_equal(frozen["xi"], model.xi)
        if hasattr(model.geometry, "t_s"):
            assert np.array_equal(frozen["radius_knots_s"], model.geometry.t_s)
            assert np.array_equal(frozen["radius_knots_m"], model.geometry.radius_m)
        assert np.all(frozen["initial_theta"] == 0)
        with np.load(context.bundle / case / "local_radau.npz", allow_pickle=False) as trajectory:
            assert np.array_equal(frozen["initial_C"], trajectory["y_old"][0, model.nr :])
            assert np.array_equal(frozen["initial_theta"], trajectory["y_old"][0, : model.nr])
    radii = []
    certificate_path = context.bundle / case / "continuous_certificate.json"
    if certificate_path.is_file():
        for slab in read_json(certificate_path)["slabs"]:
            left, right = float(slab["left_s"]), float(slab["right_s"])
            geometry = model.geometry
            if hasattr(geometry, "t_s"):
                j = int(np.searchsorted(geometry.t_s, left, side="right") - 1)
                constant = j >= len(geometry.t_s) - 1 or float(geometry.radius_m[j]) == float(
                    geometry.radius_m[j + 1]
                )
            else:
                constant = True
            with localcontext() as arithmetic:
                arithmetic.prec = DECIMAL_DIGITS
                polynomial, _ = radius_polynomial(model, left, right)
                assert constant and polynomial[0].lo > 0
                radii.append(
                    {
                        "left_s": slab["left_s"],
                        "right_s": slab["right_s"],
                        "constant_radius_m_interval": polynomial[0].strings(),
                    }
                )
    return {
        "passed": True,
        "current_configuration_and_recorded_inputs_match": True,
        "binary64_constants_identical": True,
        "all_grid_arrays_identical": True,
        "exact_initial_node_state_identical": True,
        "exact_isothermal_initial_condition": True,
        "main_model_environment_constant_after_checkpoint": True,
        "constant_radius_windows": radii,
        "current_main_model_source_identity": source_identity(),
    }


def produce_snapshot(context, case):
    """由同包模型生成晚期轨迹与独立快照，主模型右端始终复用公共依赖。"""
    produce_local_trajectory(context, case)
    dest = context.out / case
    shutil.copyfile(context.runs / case / "endpoint.json", dest / "endpoint.json")
    with np.load(dest / "local_radau.npz", allow_pickle=False) as trajectory:
        model = load_model(context, case, len(trajectory["xi"]) - 1)
        t = float(trajectory["left"][0])
        grid, geometry = model.grid, model.geometry
        np.savez_compressed(
            dest / "proof_constants.npz",
            r_m=grid.r_m,
            node_spacing_m=grid.node_spacing_m,
            face_r_m=grid.face_r_m,
            volume_weights_m2=grid.volume_weights_m2,
            xi=model.xi,
            radius_knots_s=(geometry.t_s if hasattr(geometry, "t_s") else np.array([0.0])),
            radius_knots_m=(
                geometry.radius_m if hasattr(geometry, "t_s") else np.array([geometry.radius_m])
            ),
            initial_C=trajectory["y_old"][0, model.nr :],
            initial_theta=trajectory["y_old"][0, : model.nr],
        )
        weights = np.exp(6 * (model.xi**2 - 1))
        np.savez_compressed(dest / "certificate_weights.npz", xi=model.xi, w=weights)
        constants = {
            name: float(getattr(model.properties, name)).hex()
            for name in (
                "rho_offset",
                "rho_C_coefficient",
                "cp_offset",
                "cp_fraction_coefficient",
                "k_offset",
                "k_fraction_coefficient",
                "D_prefactor",
                "D_moisture_exponent",
                "D_temperature_exponent_K",
            )
        }
        constants.update(
            R_reference_m=float(grid.radius_m).hex(),
            h=float(model.h_W_m2_K).hex(),
            hm=float(model.h_m_m_s).hex(),
            T_reference_K=float(model.temperature_reference_K).hex(),
            Ceq=float(model.ambient.values(t, right_of_breakpoint=True).moisture_eq).hex(),
        )
        return {
            "N": model.nr - 1,
            "checkpoint_s": t,
            "numeric_constants_binary64_hex": constants,
            "proof_object_sha256": {
                name: sha256(dest / name)
                for name in (
                    "local_radau.npz",
                    "proof_constants.npz",
                    "certificate_weights.npz",
                )
            },
            "production_input_sha256": {
                name: sha256(context.runs / case / name)
                for name in ("internal.npz", "endpoint.json")
            },
        }


def stored_certificate_mode(args, context, case, row):
    """执行一种晚期验证职责，保留其原有成功条件与范围字段。"""
    output, bundle = context.out, context.bundle
    weighted_path = bundle / case / "weighted_conditional_certificate.json"
    continuous_path = bundle / case / "continuous_certificate.json"
    row["stored_certificates_present"] = weighted_path.is_file() and continuous_path.is_file()
    if weighted_path.is_file():
        weighted = read_json(weighted_path)
        row["weighted_root_interval_s_decimal"] = weighted["root_interval_s_decimal"]
    if continuous_path.is_file():
        continuous = read_json(continuous_path)
        assert (
            continuous["conditional_continuous_late_PDE_barrier_verified"]
            and continuous["event_bracketed"]
        )
        row["continuous_root_interval_s_decimal"] = continuous["root_interval_s_decimal"]
        row["independent_continuous_consistency"] = check_continuous_consistency(bundle, case)
    if (bundle / case / "conditional_certificate.json").is_file():
        row["uniform_domain_checks"] = check_uniform_domains(context, case)


def model_binding_mode(args, context, case, row):
    """执行一种晚期验证职责，保留其原有成功条件与范围字段。"""
    output, bundle = context.out, context.bundle
    row["main_model_binding"] = bind_snapshot_to_main_model(context, case)


def semidiscrete_mode(args, context, case, row):
    """执行一种晚期验证职责，保留其原有成功条件与范围字段。"""
    output, bundle = context.out, context.bundle
    functions = {
        "uniform": verify_uniform_tube,
        "refined": verify_refined_tube,
        "weighted": verify_weighted_tube,
    }
    filenames = {
        "uniform": "conditional_certificate.json",
        "refined": "refined_conditional_certificate.json",
        "weighted": "weighted_conditional_certificate.json",
    }
    reference_path = bundle / case / filenames[args.mode]
    reference = read_json(reference_path) if reference_path.is_file() else None
    context.cap = args.cap or (
        reference["C_tube_cap"] if args.mode == "refined" and reference else "5e-7"
    )
    if args.mode == "refined":
        if args.uniform_base is not None:
            path = args.uniform_base.resolve() / case / "conditional_certificate.json"
            base = read_json(path)
            if base["source_sha256"].get("local_radau.npz") != sha256(
                output / case / "local_radau.npz"
            ):
                raise ValueError("细化的基础误差证书与当前轨迹身份不一致。")
            shutil.copyfile(path, output / case / path.name)
        if not (output / case / "conditional_certificate.json").is_file():
            raise ValueError("refined需要uniform证书；请通过--uniform-base指定。")
    functions[args.mode](context, case)
    calculated = read_json(output / case / filenames[args.mode])
    if reference is not None and not (
        args.mode == "refined" and context.cap != reference["C_tube_cap"]
    ):
        row["all_proof_values_identical"] = require_equivalent(
            reference, calculated, case + "/" + args.mode
        )
    row.update(
        full_window_recomputed=True,
        proof_values_recomputed=True,
        conditional_semidiscrete_certificate_passed=calculated[
            "conditional_semidiscrete_certificate_passed"
        ],
    )
    if not calculated["conditional_semidiscrete_certificate_passed"]:
        row["passed"] = False


def continuous_mode(args, context, case, row):
    """执行一种晚期验证职责，保留其原有成功条件与范围字段。"""
    output, bundle = context.out, context.bundle
    reference_path = bundle / case / "continuous_certificate.json"
    reference = read_json(reference_path) if reference_path.is_file() else None
    context.first_slab_only = args.mode == "continuous-first"
    context.cap = args.cap or (reference["amplitude_cap"] if reference else "1e-4")
    context.sub_space = (
        args.sub_space
        if args.sub_space is not None
        else (reference["space_time_interval_subdivisions"][0] if reference else 4)
    )
    context.sub_time = (
        args.sub_time
        if args.sub_time is not None
        else (reference["space_time_interval_subdivisions"][1] if reference else 4)
    )
    if context.sub_space < 1 or context.sub_time < 1 or Decimal(context.cap) <= 0:
        raise ValueError("空间/时间划分与误差管上限必须为正。")
    calculated = continuous_result(context, case, args.method)
    same_settings = reference is not None and (
        context.cap == reference["amplitude_cap"]
        and [context.sub_space, context.sub_time] == reference["space_time_interval_subdivisions"]
    )
    if args.method == "meanvalue" and same_settings:
        equivalent = (
            require_equivalent(
                reference["slabs"][0],
                calculated["slabs"][0],
                case + "/continuous-first",
            )
            if context.first_slab_only
            else require_equivalent(reference, calculated, case + "/continuous-full")
        )
        row["all_recomputed_proof_values_identical"] = equivalent
    row.update(
        full_window_recomputed=not context.first_slab_only,
        proof_values_recomputed=True,
        conditional_continuous_late_PDE_certified=calculated[
            "conditional_continuous_late_PDE_barrier_verified"
        ],
    )
    if not context.first_slab_only and not (
        calculated["conditional_continuous_late_PDE_barrier_verified"]
        and calculated["event_bracketed"]
    ):
        row["passed"] = False


def query_mode(args, context, case, row):
    """执行一种晚期验证职责，保留其原有成功条件与范围字段。"""
    output, bundle = context.out, context.bundle
    t = args.time or str(read_json(bundle / case / "endpoint.json")["reported_time_s"])
    row["query"] = query_maximum(
        bundle,
        case,
        t,
        args.certificates.resolve() if args.certificates else None,
    )


def produce_mode(args, output, bundle, cases):
    """从主轨迹生成局部晚期数据，生产本身不构成证明。"""
    context = make_context(output, bundle)
    context.production = True
    context.runs = args.runs.resolve()
    context.contract = SUPPORT / "code/config/model.json"
    snapshot = {
        "schema": "integrated-late-snapshot-v1",
        "cases": {},
        "model_sha256": sha256(context.contract),
        "scope": "Explicit late trajectories and binary constants; generation alone is not a mathematical certificate.",
    }
    for case in cases:
        snapshot["cases"][case] = produce_snapshot(context, case)
    shutil.copyfile(context.contract, output / "model.json")
    write_record(output / "proof_manifest.json", snapshot)
    return {"snapshot": snapshot}


def arithmetic_mode(args, output, bundle, cases):
    return {"arithmetic": arithmetic_checks()}


def bundle_mode(args, output, bundle, cases):
    result = {"bundle_check": check_frozen_bundle(bundle), "cases": [], "passed": True}
    for case in cases:
        stage_case_data(bundle, output, case)
        context = make_context(output, bundle)
        guards = audit_interfaces(context, case)
        if not guards["passed"]:
            raise AssertionError("精确节点接口检查失败：" + case)
        row = {
            "case": case,
            "passed": True,
            "guards": guards,
            "full_window_recomputed": False,
            "proof_values_recomputed": False,
        }
        run_cli_modes(args.mode, LATE_CASE_MODES, args, context, case, row)
        result["passed"] = result["passed"] and row["passed"]
        result["cases"].append(row)
    return result


LATE_CASE_MODES = {
    "verify": stored_certificate_mode,
    "model-bindings": model_binding_mode,
    "uniform": semidiscrete_mode,
    "refined": semidiscrete_mode,
    "weighted": semidiscrete_mode,
    "continuous-first": continuous_mode,
    "continuous-full": continuous_mode,
    "query": query_mode,
}


@decimal_arithmetic
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=(
            "verify",
            "model-bindings",
            "arithmetic",
            "uniform",
            "refined",
            "weighted",
            "continuous-first",
            "continuous-full",
            "query",
            "produce",
        ),
        default="verify",
    )
    parser.add_argument("--case", choices=("p23", "q4", "both"), default="both")
    parser.add_argument("--out", type=Path, required=True, help="支撑包外的新输出目录")
    parser.add_argument(
        "--bundle", type=Path, default=DEFAULT_BUNDLE, help="精确节点及证书所在目录"
    )
    parser.add_argument(
        "--method",
        choices=("meanvalue", "spline", "hermite", "adaptive"),
        default="meanvalue",
    )
    parser.add_argument("--cap", help="refined和continuous模式的误差管上限；默认采用对应证据参数")
    parser.add_argument("--sub-space", type=int)
    parser.add_argument("--sub-time", type=int)
    parser.add_argument("--time", help="query使用的十进制秒数；省略时读取该案报告时刻")
    parser.add_argument("--runs", type=Path, help="produce所需的主程序轨迹目录，含p23/q4子目录")
    parser.add_argument("--uniform-base", type=Path, help="refined使用的uniform输出根目录")
    parser.add_argument("--certificates", type=Path, help="query使用的continuous-full输出根目录")
    args = parser.parse_args(argv)
    output = prepare_output(args.out)
    bundle = args.bundle.resolve()
    cases = ["p23", "q4"] if args.case == "both" else [args.case]
    report = {
        "mode": args.mode,
        "passed": True,
        "cases": [],
        "data_check": verify_data(),
        "original_global_t0_error_certified": False,
    }
    if args.mode == "produce" and args.runs is None:
        parser.error("produce需要--runs提供现有主程序轨迹。")
    handlers = {mode: bundle_mode for mode in LATE_CASE_MODES}
    handlers.update(arithmetic=arithmetic_mode, produce=produce_mode)
    report.update(run_cli_modes(args.mode, handlers, args, output, bundle, cases))
    write_record(output / "晚期验证报告.json", report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "mode": args.mode,
                "report": str(output / "晚期验证报告.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
