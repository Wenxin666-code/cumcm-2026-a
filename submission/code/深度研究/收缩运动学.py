"""非仿射收缩：一个物理类、一套局部时间积分/采样、五场景收敛矩阵。

网格、物性、附件插值、温度平移和稀疏结构直接复用原 code/公共依赖.py。
本文件只实现内部物质映射及对应通量；不复制 vendor 或旧运行脚本。
run 可限制时段；matrix 保留 1e-5 场差、0.36 秒事件差和 1e-7 守恒门槛。
这些是数值研究判据，不是连续 PDE 误差证明。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from 研究求解 import polynomial_derivative

from 公共接口 import (
    configure,
    paths,
    read_json,
    section,
    sha256,
    source_identity,
    write_json,
)


class NonAffineModel:
    """持有原 RelativeTemperatureModel，仅重写非仿射几何、通量和守恒审计。"""

    def __init__(self, base, beta):
        if not np.isfinite(beta) or not -1 <= beta <= 1:
            raise ValueError("beta 必须在 [-1, 1]。")
        self.base, self.beta = base, float(beta)
        faces = base.grid.face_r_m
        self.x2_left = (np.r_[0.0, faces] / base.grid.radius_m) ** 2
        self.x2_right = (np.r_[faces, base.grid.radius_m] / base.grid.radius_m) ** 2
        self.reference_weights = 2 * base.grid.volume_weights_m2 / base.grid.radius_m**2

    def __getattr__(self, name):
        return getattr(self.base, name)

    @classmethod
    def create(
        cls,
        inputs=None,
        contract=None,
        N=2560,
        beta=0.0,
        power=2.0,
        geometry="measured",
        ambient=None,
    ):
        data = paths()
        original = section("moving_model").build_model(
            inputs or data["inputs"],
            contract or data["kinematics_contract"],
            N=N,
            power=power,
            geometry=geometry,
            ambient=ambient,
        )
        return cls(
            section("relative_temperature").RelativeTemperatureModel.from_model(original),
            beta,
        )

    def mapping(self, t, a):
        a = np.asarray(a, dtype=float)
        R0 = self.grid.radius_m
        scale = np.asarray(self.geometry.radius(t)) / R0
        b, x = self.beta * (1 - scale), a / R0
        q, p = 1 + b * (1 - x * x), 1 + b * (1 - 3 * x * x)
        chi, gradient, jacobian = scale * a * q, scale * p, scale * scale * q * p
        if np.any(gradient <= 0) or np.any(jacobian <= 0):
            raise ValueError("物质映射失去正 Jacobian。")
        return chi, gradient, jacobian

    def current_weights(self, t):
        scale = float(self.geometry.radius(t)) / self.grid.radius_m
        b = self.beta * (1 - scale)
        q, left, right = 1 + b, self.x2_left, self.x2_right
        ratio = (
            scale
            * scale
            * (
                q * q
                - 2 * b * q * (right + left)
                + b * b * (right * right + right * left + left * left)
            )
        )
        return self.grid.volume_weights_m2 * ratio

    def fluxes(self, t, y, right_of_breakpoint=False):
        T, C = self.unpack(y)
        theta = np.asarray(y)[: self.nr]
        rho, cp, k, D = self.properties.evaluate(C, T)
        ambient = self.ambient.values(t, right_of_breakpoint=right_of_breakpoint)
        chi, gradient, _ = self.mapping(t, self.grid.face_r_m)
        _, surface_gradient, _ = self.mapping(t, self.grid.radius_m)
        radius = float(self.geometry.radius(t))
        FC = (
            self.grid.face_r_m
            * (D[:-1] + D[1:])
            / 2
            / gradient**2
            * np.diff(C)
            / self.grid.node_spacing_m
        )
        FT = chi / gradient * (k[:-1] + k[1:]) / 2 * np.diff(theta) / self.grid.node_spacing_m
        outerC = (
            -self.grid.radius_m
            * self.h_m_m_s
            / surface_gradient
            * (C[-1] - float(ambient.moisture_eq))
        )
        outerT = (
            -radius
            * self.h_W_m2_K
            * (theta[-1] - (float(ambient.temperature_K) - self.temperature_reference_K))
        )
        return FC, FT, float(outerC), float(outerT), rho * cp

    def rhs(self, t, y, *, right_of_breakpoint=False):
        FC, FT, outerC, outerT, capacity = self.fluxes(t, y, right_of_breakpoint)
        dC = np.r_[FC[0], np.diff(FC), outerC - FC[-1]] / self.grid.volume_weights_m2
        dT = np.r_[FT[0], np.diff(FT), outerT - FT[-1]] / (self.current_weights(t) * capacity)
        result = np.r_[dT, dC]
        if not np.isfinite(result).all():
            raise FloatingPointError("非仿射导数出现非有限值。")
        return result

    def audit(self, t, y, *, right_of_breakpoint=False):
        _, C = self.unpack(y)
        dy = self.rhs(t, y, right_of_breakpoint=right_of_breakpoint)
        _, _, outerC, outerT, capacity = self.fluxes(t, y, right_of_breakpoint)
        water = float(self.grid.volume_weights_m2 @ dy[self.nr :])
        heat = float(self.current_weights(t) @ (capacity * dy[: self.nr]))
        chi, _, jacobian = self.mapping(t, self.grid.r_m)
        index = int(np.argmax(C))
        return {
            "moisture_rate_residual": 2 * (water - outerC) / self.grid.radius_m**2,
            "heat_rate_residual": 2 * (heat - outerT) / self.grid.radius_m**2,
            "moisture_relative_residual": abs(water - outerC) / max(abs(water), abs(outerC), 1e-30),
            "heat_relative_residual": abs(heat - outerT) / max(abs(heat), abs(outerT), 1e-30),
            "mean_C": float(self.reference_weights @ C),
            "max_C": float(C[index]),
            "maximum_a_over_R0": float(self.xi[index]),
            "maximum_r_m": float(chi[index]),
            "J_min": float(jacobian.min()),
            "J_max": float(jacobian.max()),
            "dry_density_ratio_min": float(1 / jacobian.max()),
            "dry_density_ratio_max": float(1 / jacobian.min()),
        }

    def sample(self, times, states, radii):
        times, states, radii = (
            np.atleast_1d(times),
            np.asarray(states),
            np.asarray(radii),
        )
        if states.ndim == 1:
            states = states[:, None]
        T = np.full((len(times), len(radii) + 1), np.nan)
        C = T.copy()
        for row, t in enumerate(times):
            chi, _, _ = self.mapping(float(t), self.grid.r_m)
            valid = radii < chi[-1]
            tv, cv = self.unpack(states[:, row])
            T[row, :-1][valid] = np.interp(radii[valid], chi, tv)
            C[row, :-1][valid] = np.interp(radii[valid], chi, cv)
            T[row, -1], C[row, -1] = tv[-1], cv[-1]
        return T, C


def solve_segment(model, t_span, y0, settings, *, right_of_breakpoint=False, detect_event=False):
    """BDF 用 tau=t-origin；系数仍用物理时间，保留小于绝对时间 ULP 的步。"""
    origin, duration = float(t_span[0]), float(t_span[1] - t_span[0])
    if duration <= 0:
        raise ValueError("积分区间必须严格递增。")
    first_step = None
    if settings.limit_initial_step and origin == 0.0:
        T, C = model.unpack(y0)
        diffusivity = float(np.max(model.properties.evaluate(C, T)[3]))
        if diffusivity > 0:
            first_step = min(0.01, 0.1 * model.grid.minimum_spacing_m**2 / diffusivity, duration)
    upstream = section("_vendor_host2_model")
    result = solve_ivp(
        lambda tau, y: model.rhs(origin + float(tau), y, right_of_breakpoint=right_of_breakpoint),
        (0.0, duration),
        y0,
        method=settings.method,
        rtol=settings.rtol,
        atol=settings.atol_vector(model.nr),
        max_step=settings.max_step_s,
        jac_sparsity=model.jacobian_sparsity(),
        dense_output=True,
        events=upstream._event_function(model) if detect_event else None,
        **({} if first_step is None else {"first_step": first_step}),
    )
    if not result.success:
        raise RuntimeError(f"局部时间积分失败 {t_span}: {result.message}")
    wrapped = SimpleNamespace(
        t=origin + result.t,
        y=result.y,
        sol=lambda t: result.sol(np.asarray(t, dtype=float) - origin),
        t_events=(None if result.t_events is None else [origin + t for t in result.t_events]),
        nfev=result.nfev,
        njev=result.njev,
        nlu=result.nlu,
        success=True,
        local_result=result,
        time_origin_s=origin,
    )
    return upstream.SegmentSolution(
        origin, origin + float(result.t[-1]), right_of_breakpoint, first_step, wrapped
    )


def integrate_balance(model, segment, sums):
    """在每个实际 BDF 步微分稠密多项式，独立积分水分/有效显热通量。"""
    local = segment.result.local_result
    for order, totals in sums.items():
        nodes, weights = leggauss(order)
        for left, right, poly in zip(local.sol.ts[:-1], local.sol.ts[1:], local.sol.interpolants):
            right = min(float(right), float(local.t[-1]))
            if right <= left:
                continue
            tau = (left + right) / 2 + (right - left) / 2 * nodes
            y = poly(tau)
            dy = polynomial_derivative(poly, tau)
            t = segment.result.time_origin_s + tau
            theta, C = y[: model.nr].T, y[model.nr :].T
            rho, cp, _, _ = model.properties.evaluate(C, theta + model.temperature_reference_K)
            ambient = model.ambient.values(t, right_of_breakpoint=segment.right_of_breakpoint)
            R0, radius = model.grid.radius_m, np.asarray(model.geometry.radius(t))
            scale = radius / R0
            surface_gradient = scale * (1 - 2 * model.beta * (1 - scale))
            water = 2 * model.h_m_m_s / (R0 * surface_gradient) * (C[:, -1] - ambient.moisture_eq)
            heat = (
                2
                * radius
                * model.h_W_m2_K
                / R0**2
                * (theta[:, -1] - (ambient.temperature_K - model.temperature_reference_K))
            )
            volumes = np.array([model.current_weights(float(tt)) for tt in t])
            stored = 2 * np.sum(volumes * (rho * cp) * dy[: model.nr].T, axis=1) / R0**2
            totals += (right - left) / 2 * (weights @ np.array([water, heat, stored, abs(heat)]).T)


def run_identity(inputs, contract):
    """绑定程序、附件和实际模型配置。"""
    identity = source_identity()
    identity["files"].update(
        {
            str(p): {"path": str(p.resolve()), "sha256": sha256(p)}
            for p in (
                Path(__file__),
                contract,
                inputs / "附件1.xlsx",
                inputs / "附件2.xlsx",
            )
        }
    )
    return identity


def retain_samples(storage, key, times, values):
    if len(times):
        storage[key][0].extend(np.asarray(times).tolist())
        storage[key][1].extend(np.asarray(values).T)


def write_sample_outputs(out, model, storage, radii, event, event_state, state):
    """统一保存物理半径、参考坐标与事件后展示刻度。"""
    for key in ("samples", "internal", "near_event"):
        times, rows = storage[key]
        times, indices = np.unique(times, return_index=True)
        values = np.asarray(rows)[indices] if len(rows) else np.empty((0, 2 * model.nr))
        if key == "internal":
            T, C = (
                values[:, : model.nr] + model.temperature_reference_K,
                values[:, model.nr :],
            )
            extra = {
                "xi": model.xi,
                "a_m": model.grid.r_m,
                "mean_C": C @ model.reference_weights,
            }
        else:
            T, C = model.sample(times, values.T, radii)
            extra = {"r_fixed_m": radii}
        np.savez_compressed(
            out / f"{key}.npz",
            t_s=times,
            T_K=T,
            C=C,
            R_m=model.geometry.radius(times),
            **extra,
        )
    if event is not None:
        np.savez_compressed(
            out / "event_states.npz",
            root_state=event_state,
            post_state=state,
            tick_t_s=storage["ticks"][0],
            tick_states=storage["ticks"][1],
        )


def balance_report(model, state, initial_mean, sums):
    change = float(model.reference_weights @ state[model.nr :]) - initial_mean
    balances = [
        {
            "order": q,
            "moisture_residual_abs": abs(change + v[0]),
            "heat_residual_abs": abs(v[1] + v[2]),
            "heat_residual_relative": abs(v[1] + v[2]) / max(v[3], 1e-30),
            "integrals": v.tolist(),
        }
        for q, v in sums.items()
    ]
    return balances


def verify_run_identity(identity):
    for item in identity["files"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("积分期间源码或输入发生变化。")


def run_case(
    out,
    *,
    beta=0.0,
    N=2560,
    end_s=None,
    horizon_s=864000.0,
    tight=False,
    inputs=None,
    contract=None,
):
    """统一时限/事件运行；事件模式另保存根后 2 秒，用于四位小时干侧检查。"""
    out, data = Path(out), paths()
    inputs, contract = Path(inputs or data["inputs"]), Path(contract or data["kinematics_contract"])
    horizon = float(horizon_s if end_s is None else end_s)
    if N < 2 or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("N 至少为 2，终止时间必须为正且有限。")
    model = NonAffineModel.create(inputs, contract, N, beta)
    scale = 0.1 if tight else 1.0
    settings = section("moving_model").IntegrationSettings(
        "BDF", 1e-10 * scale, 1e-9 * scale, 1e-12 * scale, 600.0, True
    )
    identity = run_identity(inputs, contract)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    state = model.initial_state()
    initial_mean = float(model.reference_weights @ state[model.nr :])
    storage = {key: [[], []] for key in ("samples", "internal", "near_event", "ticks")}
    next_sample = next_profile = 0.0
    sums, segments, recent = {8: np.zeros(4), 16: np.zeros(4)}, [], []
    radii, event, event_state = np.arange(20) * 0.001, None, None

    def advance(left, right, detect):
        nonlocal state
        segment = solve_segment(
            model,
            (left, right),
            state,
            settings,
            right_of_breakpoint=left >= model.ambient.breakpoint_s,
            detect_event=detect,
        )
        integrate_balance(model, segment, sums)
        state = segment.result.y[:, -1].copy()
        segments.append(
            {
                "start_s": float(left),
                "stop_s": segment.t_end_s,
                "steps": len(segment.result.t) - 1,
                "nfev": segment.result.nfev,
            }
        )
        return segment

    try:
        cuts = model.ambient.breakpoints(0.0, horizon)
        for left, right in zip(cuts[:-1], cuts[1:]):
            segment = advance(float(left), float(right), end_s is None)
            stop = segment.t_end_s
            sample = []
            while next_sample <= stop + 1e-8:
                if next_sample >= left - 1e-8:
                    sample.append(next_sample)
                next_sample += 1 if next_sample < 10800 else 60
            if sample:
                retain_samples(storage, "samples", sample, segment.result.sol(sample))
            sample = []
            while next_profile <= stop + 1e-8:
                if next_profile >= left - 1e-8:
                    sample.append(next_profile)
                next_profile += 60
            if sample:
                retain_samples(storage, "internal", sample, segment.result.sol(sample))
            tail = np.arange(
                max(math.ceil(left), math.ceil(stop) - 120),
                math.floor(stop) + 1,
                dtype=float,
            )
            if len(tail):
                recent.append((tail, segment.result.sol(tail)))
            recent = recent[-2:]
            if end_s is None and len(segment.result.t_events[0]):
                event, event_state = float(segment.result.t_events[0][0]), state.copy()
                retain_samples(storage, "internal", [event], state[:, None])
                for times, values in recent:
                    keep = times >= event - 120
                    retain_samples(storage, "near_event", times[keep], values[:, keep])
                break
            write_json(
                out / "progress.json",
                {
                    "status": "running",
                    "time_s": stop,
                    "maximum_C": float(state[model.nr :].max()),
                },
            )
            del segment
            gc.collect()
        if event is not None:
            ticks = 0.36 * np.arange(
                math.floor(event / 0.36) + 1, math.floor((event + 2.0) / 0.36) + 1
            )
            cuts = model.ambient.breakpoints(event, event + 2.0)
            for left, right in zip(cuts[:-1], cuts[1:]):
                segment = advance(float(left), float(right), False)
                times = ticks[(ticks > left) & (ticks <= right + 1e-8)]
                if len(times):
                    retain_samples(storage, "ticks", times, segment.result.sol(times))
            stop = event + 2.0
        retain_samples(storage, "internal", [stop], state[:, None])
        write_sample_outputs(out, model, storage, radii, event, event_state, state)
        balances = balance_report(model, state, initial_mean, sums)
        verify_run_identity(identity)
        result = {
            "completed": end_s is not None or event is not None,
            "beta": beta,
            "N": N,
            "tight": tight,
            "root_s": event,
            "root_h": None if event is None else event / 3600,
            "end_time_s": stop,
            "settings": asdict(settings),
            "balance": balances,
            "segments": segments,
            "final_audit": model.audit(
                stop, state, right_of_breakpoint=stop > model.ambient.breakpoint_s
            ),
            "source_identity": identity,
            "elapsed_s": time.perf_counter() - start,
            "time_coordinate": "逐段局部 tau；物性和外界系数使用原物理时间",
            "scope": "新数值运行；热审计积分当前体积 rho*cp*T_t，未包含潜热。",
        }
        write_json(out / "record.json", result)
        return result
    except Exception as error:
        write_json(out / "failure.json", {"completed": False, "error": str(error)})
        raise


def compare(left, right, filenames=("samples", "near_event", "internal")):
    left, right = Path(left), Path(right)
    records = [read_json(p / "record.json") for p in (left, right)]
    if records[0]["beta"] != records[1]["beta"]:
        raise ValueError("收敛比较必须使用相同 beta。")
    if records[0]["source_identity"] != records[1]["source_identity"]:
        raise ValueError("收敛比较的源码/输入身份不同。")
    differences = {}
    for filename in filenames:
        with np.load(left / f"{filename}.npz", allow_pickle=False) as a, np.load(
            right / f"{filename}.npz", allow_pickle=False
        ) as b:
            times, ia, ib = np.intersect1d(a["t_s"], b["t_s"], return_indices=True)
            if not len(times):
                if filename == "near_event" and all(r["root_s"] is None for r in records):
                    continue
                raise ValueError(f"{filename} 没有公共采样时刻。")
            for field in ("T_K", "C"):
                aa, bb = a[field][ia], b[field][ib]
                if filename == "internal" and not np.array_equal(a["xi"], b["xi"]):
                    aa = np.array([np.interp(b["xi"], a["xi"], row) for row in aa])
                if aa.shape != bb.shape or not np.array_equal(np.isfinite(aa), np.isfinite(bb)):
                    raise ValueError("采样字段形状或几何掩码不同。")
                differences[f"{filename}_{field}"] = float(np.nanmax(abs(aa - bb)))
    differences["max_T_K"] = max(v for k, v in differences.items() if k.endswith("_T_K"))
    differences["max_C"] = max(v for k, v in differences.items() if k.endswith("_C"))
    differences["root_difference_s"] = (
        abs(records[0]["root_s"] - records[1]["root_s"])
        if all(r["root_s"] is not None for r in records)
        else None
    )
    return differences


def check_mapping(model, beta):
    scales = np.unique(
        np.r_[1.0, model.geometry.radius_m / 0.02, model.geometry.tail_radius_m / 0.02]
    )
    b = beta * (1 - scales)
    qmin, pmin = float(min(1.0, np.min(1 + b))), float(np.min(np.minimum(1 + b, 1 - 2 * b)))
    if min(qmin, pmin) <= 0:
        raise ValueError("完整观测半径范围不能保持正物质映射。")
    return {"beta": beta, "q_min": qmin, "p_min": pmin}


def check_affine_limit(model, reference, state, beta, times):
    affine = []
    for t in times:
        if beta == 0:
            target = reference.rhs(t, state, right_of_breakpoint=t > 14400)
            affine.append(
                float(
                    np.max(abs(model.rhs(t, state, right_of_breakpoint=t > 14400) - target))
                    / max(np.max(abs(target)), 1.0)
                )
            )
    return affine


def check_dry_mass(model, gauss, weights, times):
    mass = []
    for t in times:
        chi, gradient, jacobian = model.mapping(t, 0.01 * (gauss + 1))
        for length in (0.7, 1.0, 1.3):
            integral = 0.01 * np.dot(weights, 1 / (length * jacobian) * length * chi * gradient)
            mass.append(abs(integral / (0.5 * 0.02**2) - 1))
    return mass


def check_fixed_radius(inputs, contract, beta, state):
    fixed_model = NonAffineModel.create(inputs, contract, 40, beta, geometry="fixed")
    return float(np.max(abs(fixed_model.rhs(160.0, state) - fixed_model.base.rhs(160.0, state))))


def check_closed_constant(model):
    model.h_W_m2_K = model.h_m_m_s = 0.0
    return list(
        float(np.max(abs(model.rhs(t, model.initial_state(310.0, 1.2)))))
        for t in (0.0, 160.0, 1800.0, 14401.0, 180000.0, 300000.0)
    )


def validate_manufactured_solution(inputs, contract):
    moving = section("moving_model")
    props = moving.properties_from_contract(read_json(contract))

    def exact(t, a):
        x = np.asarray(a) / 0.02

        def field(base, amplitude, tau, eta):
            A = amplitude * np.exp(-t / tau)
            shape = 1 - x * x + eta * (1 - x**4)
            return (
                base + A * shape,
                -A * shape / tau,
                A * (-2 * x - 4 * eta * x**3) / 0.02,
                A * (-2 - 12 * eta * x * x) / 0.02**2,
            )

        return field(310.0, 8.0, 4000.0, 0.2), field(1.1, 0.15, 3000.0, 0.35)

    def properties(C, T):
        rho = props.rho_offset + props.rho_C_coefficient * C
        cp = props.cp_offset + props.cp_fraction_coefficient * C / (1 + C)
        k = props.k_offset + props.k_fraction_coefficient * C / (1 + C)
        D = props.D_prefactor * np.exp(
            -props.D_moisture_exponent / C - props.D_temperature_exponent_K / T
        )
        return rho * cp, k, D

    def ambient_values(t, **kwargs):
        (T, _, Ta, _), (C, _, Ca, _) = exact(t, 0.02)
        _, k, D = properties(C, T)
        scale = 1 / (1 + 1e-4 * np.asarray(t))
        gradient = scale * (1 - (1 - scale))
        return moving.AmbientValues(
            T + k * Ta / (gradient * 25.0), C + D * Ca / (gradient * 8e-7), False
        )

    def forcing(model, t):
        a = model.grid.r_m
        (T, Tt, Ta, Taa), (C, Ct, Ca, Caa) = exact(t, a)
        B, k, D = properties(C, T)
        scale, x = 1 / (1 + 1e-4 * t), a / 0.02
        b = 0.5 * (1 - scale)
        gradient = scale * (1 + b - 3 * b * x * x)
        ga, chi = -6 * scale * b * a / 0.02**2, scale * a * (1 + b - b * x * x)
        logDa = props.D_moisture_exponent / C**2 * Ca + props.D_temperature_exponent_K / T**2 * Ta
        ka = props.k_fraction_coefficient / (1 + C) ** 2 * Ca
        axisC, axisT = np.empty_like(a), np.empty_like(a)
        axisC[0], axisT[0] = Caa[0], Taa[0]
        axisC[1:], axisT[1:] = Ca[1:] / a[1:], gradient[1:] * Ta[1:] / chi[1:]
        LC = D / gradient**2 * (Caa + axisC + (logDa - 2 * ga / gradient) * Ca)
        LT = k / (gradient * gradient * B) * (Taa + axisT + (ka / k - ga / gradient) * Ta)
        return np.r_[Tt - LT, Ct - LC]

    ambient = SimpleNamespace(
        breakpoint_s=np.inf,
        values=ambient_values,
        breakpoints=lambda start, end: np.array([start, end]),
    )
    geometry = moving.CallableRadius(lambda t: 0.02 / (1 + 1e-4 * np.asarray(t)))
    rows = []
    for N in (20, 40, 80, 160):
        model = NonAffineModel.create(
            inputs, contract, N, 0.5, power=1.0, geometry=geometry, ambient=ambient
        )
        (T, *_), (C, *_) = exact(0.0, model.grid.r_m)
        times = np.linspace(0.0, 1200.0, 41)
        solution = solve_ivp(
            lambda t, y: model.rhs(t, y) + forcing(model, t),
            (0.0, 1200.0),
            np.r_[T - model.temperature_reference_K, C],
            method="BDF",
            rtol=1e-11,
            atol=np.r_[np.full(model.nr, 1e-9), np.full(model.nr, 1e-12)],
            jac_sparsity=model.jacobian_sparsity(),
            t_eval=times,
            max_step=20.0,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        expectedT = np.array([exact(t, model.grid.r_m)[0][0] for t in times])
        expectedC = np.array([exact(t, model.grid.r_m)[1][0] for t in times])
        errorT = float(
            np.max(abs(solution.y[: model.nr].T + model.temperature_reference_K - expectedT))
        )
        errorC = float(np.max(abs(solution.y[model.nr :].T - expectedC)))
        rows.append(
            {
                "N": N,
                "T_error": errorT,
                "C_error": errorC,
                "T_order": (None if not rows else math.log2(rows[-1]["T_error"] / errorT)),
                "C_order": (None if not rows else math.log2(rows[-1]["C_error"] / errorC)),
            }
        )
    if rows[-1]["T_order"] <= 1.7 or rows[-1]["C_order"] <= 1.7:
        raise ValueError("独立解析制造解未达到原有 >1.7 收敛阶门槛。")
    return rows


def validate(out, *, inputs=None, contract=None):
    """新源码的映射/退化/守恒及独立解析制造解检查；不借用旧通过标志。"""
    out, data = Path(out), paths()
    inputs, contract = Path(inputs or data["inputs"]), Path(contract or data["kinematics_contract"])
    moving = section("moving_model")
    relative = section("relative_temperature").RelativeTemperatureModel
    affine, fixed, constant, mass, mapping = [], [], [], [], []
    gauss, weights = leggauss(64)
    for beta in (-1.0, -0.5, 0.0, 0.5, 1.0):
        model = NonAffineModel.create(inputs, contract, 40, beta)
        reference = relative.from_model(moving.build_model(inputs, contract, N=40))
        state = np.r_[
            310 + 3 * model.xi**2 - model.temperature_reference_K,
            1.2 + 0.2 * (1 - model.xi**2),
        ]
        mapping.append(check_mapping(model, beta))
        times = (0.0, 160.0, 1800.0, 14401.0, 180000.0)
        affine.extend(check_affine_limit(model, reference, state, beta, times))
        mass.extend(check_dry_mass(model, gauss, weights, times))
        fixed.append(check_fixed_radius(inputs, contract, beta, state))
        constant.extend(check_closed_constant(model))
    if max(affine) >= 1e-12 or max(fixed) >= 1e-9 or max(constant) != 0 or max(mass) >= 1e-13:
        raise ValueError("仿射/固定半径/常量场或干质量退化检查失败。")
    rows = validate_manufactured_solution(inputs, contract)
    result = {
        "passed": True,
        "mapping": mapping,
        "affine_relative_rhs_max": max(affine),
        "fixed_radius_rhs_max": max(fixed),
        "closed_constant_rhs_max": max(constant),
        "dry_mass_relative_error": max(mass),
        "manufactured_solution": rows,
        "source_identity": source_identity(),
        "scope": "解析源项独立于离散 RHS；仅验证所用数学模型。",
    }
    write_json(out / "validation.json", result)
    return result


def matrix(
    out,
    *,
    betas=(0.0, -0.5, 0.5, -1.0, 1.0),
    grids=(320, 640, 1280),
    max_N=20480,
    horizon_s=864000.0,
    **inputs,
):
    """逐级空间与时间收敛；达不到原门槛便报错并保留计算结果。"""
    out, rows = Path(out), []
    if len(grids) < 3 or sorted(set(grids)) != list(grids) or grids[-1] > max_N:
        raise ValueError("至少三个递增初始网格，且不超过 max_N。")
    out.mkdir(parents=True, exist_ok=False)
    validate(out / "validation", **inputs)
    for beta in betas:
        previous, history = None, []
        pending = list(grids)
        while pending:
            N = pending.pop(0)
            folder = out / f"beta_{beta:g}_N{N}"
            record = run_case(folder, beta=beta, N=N, horizon_s=horizon_s, **inputs)
            if not record["completed"]:
                raise RuntimeError("给定 horizon 内未到首次干燥事件。")
            if previous is not None:
                history.append(compare(previous, folder))
            previous = folder
            if not pending and (
                history[-1]["max_C"] >= 1e-5
                or history[-1]["max_T_K"] >= 1e-5
                or history[-1]["root_difference_s"] >= 0.36
            ):
                if 2 * N > max_N:
                    raise RuntimeError("达到 max_N 仍未满足空间收敛门槛；未降低标准。")
                pending.append(2 * N)
        tight = out / f"beta_{beta:g}_N{N}_tight"
        tight_record = run_case(tight, beta=beta, N=N, tight=True, horizon_s=horizon_s, **inputs)
        if not tight_record["completed"]:
            raise RuntimeError("加严时间容差运行未到事件。")
        temporal = compare(folder, tight)
        margin = history[-1]["root_difference_s"] + temporal["root_difference_s"]
        if temporal["max_C"] >= 1e-5 or temporal["max_T_K"] >= 1e-5 or margin >= 0.36:
            raise RuntimeError("时间收敛或合并事件余量未通过。")
        report_time = 0.36 * math.ceil((record["root_s"] + margin) / 0.36)
        with np.load(folder / "event_states.npz", allow_pickle=False) as values:
            index = int(np.argmin(abs(values["tick_t_s"] - report_time)))
            maximum = float(np.max(values["tick_states"][index, N + 1 :]))
            if (
                abs(values["tick_t_s"][index] - report_time) >= 1e-8
                or maximum >= 0.15
                or report_time <= record["root_s"]
            ):
                raise RuntimeError("报告时刻未严格处于干侧。")
        balance = record["balance"][-1]
        if balance["moisture_residual_abs"] >= 1e-7 or balance["heat_residual_relative"] >= 1e-7:
            raise RuntimeError("积分守恒门槛未通过。")
        row = {
            "beta": beta,
            "N": N,
            "root_s": record["root_s"],
            "reported_time_h": report_time / 3600,
            "reported_max_C": maximum,
            "empirical_margin_s": margin,
            "space": history[-1],
            "time": temporal,
            "balance": balance,
        }
        rows.append(row)
        write_json(
            out / f"beta_{beta:g}_convergence.json",
            {"history": history, "accepted": row},
        )
    result = {
        "passed": True,
        "rows": rows,
        "scope": "指定 beta 矩阵的数值收敛及守恒验收，不是连续 PDE 证明。",
    }
    write_json(out / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--contract", type=Path)
    commands = parser.add_subparsers(dest="mode", required=True)
    checks = commands.add_parser("validate")
    checks.add_argument("--out", type=Path, required=True)
    one = commands.add_parser("run")
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--beta", type=float, default=0.0)
    one.add_argument("--N", type=int, default=2560)
    one.add_argument("--end", dest="end_s", type=float, help="省略即计算首次事件")
    one.add_argument("--horizon", dest="horizon_s", type=float, default=864000.0)
    one.add_argument("--tight", action="store_true")
    many = commands.add_parser("matrix")
    many.add_argument("--out", type=Path, required=True)
    many.add_argument("--betas", nargs="+", type=float, default=[0.0, -0.5, 0.5, -1.0, 1.0])
    many.add_argument("--grids", nargs="+", type=int, default=[320, 640, 1280])
    many.add_argument("--max-N", dest="max_N", type=int, default=20480)
    many.add_argument("--horizon", dest="horizon_s", type=float, default=864000.0)
    pair = commands.add_parser("compare")
    pair.add_argument("left", type=Path)
    pair.add_argument("right", type=Path)
    args = vars(parser.parse_args(argv))
    configure(args.pop("code_dir"), args.pop("research_root"))
    mode = args.pop("mode")
    if mode == "compare":
        args.pop("inputs")
        args.pop("contract")
    result = {
        "run": run_case,
        "matrix": matrix,
        "compare": compare,
        "validate": validate,
    }[
        mode
    ](**args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
