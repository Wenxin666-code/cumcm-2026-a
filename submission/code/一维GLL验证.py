"""一维三次 GLL 的独立弱式、完整求解及 FV 配对验证。

输入：code/data/验证数据/inputs 中两份题目附件、模型配置、9组原始GLL数组；
compare 模式接收未舍入 FV 数组。输出统一写入包外目录。
空间弱式和物性独立实现，作为主有限体积模型的独立对照。
"""

import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
import gc
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import j0, j1
from scipy.sparse import coo_matrix, bmat
from openpyxl import load_workbook
from 验证公共接口 import (
    DATA,
    read_json,
    load_npz,
    write_csv,
    run_cli_modes,
    write_json,
    sha256,
    prepare_output,
    source_identity,
    verify_data,
    science,
    research,
)

EVIDENCE = DATA / "evidence/numerics/independent_fem"
SOURCE_REPORT = DATA / "data/processed/final_numeric_sections/validation_1d_sources.json"


@dataclass
class CubicGLL:
    elements: int
    radial_power: float = 2.0

    def __post_init__(self):
        z = np.array([-1.0, -1 / np.sqrt(5), 1 / np.sqrt(5), 1.0])
        self.local_nodes = z
        self.local_weights = np.array([1.0, 5.0, 5.0, 1.0]) / 6
        self.Vinverse = np.linalg.inv(np.vander(z, 4, increasing=True))
        bases = [
            Polynomial.fromroots(np.delete(z, j)) / np.prod(z[j] - np.delete(z, j))
            for j in range(4)
        ]
        self.D = np.column_stack([b.deriv()(z) for b in bases])
        radial_edges = 1 - (1 - np.linspace(0, 1, self.elements + 1)) ** self.radial_power
        self.edges = radial_edges**2
        self.width = np.diff(self.edges)
        self.local_s = (self.edges[:-1, None] + self.edges[1:, None]) / 2 + self.width[
            :, None
        ] * z / 2
        self.connectivity = 3 * np.arange(self.elements)[:, None] + np.arange(4)[None, :]
        self.nodes = 3 * self.elements + 1
        self.s = np.zeros(self.nodes)
        self.s[self.connectivity] = self.local_s
        self.qweights = self.width[:, None] / 2 * self.local_weights
        self.mass = np.bincount(
            self.connectivity.ravel(), weights=self.qweights.ravel(), minlength=self.nodes
        )
        pairs = np.broadcast_arrays(self.connectivity[:, :, None], self.connectivity[:, None, :])
        self.scalar_sparsity = coo_matrix(
            (np.ones(self.elements * 16), (pairs[0].ravel(), pairs[1].ravel())),
            shape=(self.nodes, self.nodes),
        ).tocsr()
        self.coupled_sparsity = bmat(
            [
                [self.scalar_sparsity, self.scalar_sparsity],
                [self.scalar_sparsity, self.scalar_sparsity],
            ],
            format="csr",
        )

    def stiffness_action(self, u, coefficient):
        local = u[self.connectivity]
        # Subtract the local constant before differentiating: constants vanish exactly.
        gradient = ((local - local[:, :1]) @ self.D.T) * (2 / self.width[:, None])
        flux = self.qweights * self.local_s * coefficient[self.connectivity] * gradient
        local_action = (flux @ self.D) * (2 / self.width[:, None])
        return np.bincount(
            self.connectivity.ravel(), weights=local_action.ravel(), minlength=self.nodes
        )

    def evaluate(self, fields, query_s):
        """Evaluate cubic elements, never linear-interpolate a high-order field."""
        query = np.asarray(query_s, dtype=float)
        if np.any(query < 0) or np.any(query > 1):
            raise ValueError("Reference coordinate outside the physical domain")
        e = np.clip(np.searchsorted(self.edges, query, side="right") - 1, 0, self.elements - 1)
        z = 2 * (query - self.edges[e]) / self.width[e] - 1
        weights = np.vander(z.ravel(), 4, increasing=True) @ self.Vinverse
        weights = weights.reshape(query.shape + (4,))
        if np.ndim(fields) == 1:
            return np.sum(fields[self.connectivity[e]] * weights, axis=-1)
        if query.ndim == 1:
            return np.einsum("tij,ij->ti", fields[:, self.connectivity[e]], weights)
        if query.ndim == 2 and query.shape[0] == fields.shape[0]:
            values = fields[np.arange(len(fields))[:, None, None], self.connectivity[e]]
            return np.sum(values * weights, axis=-1)
        raise ValueError("Unsupported field/query shape")

    def extrema(self, u):
        """Global extrema including every real stationary point inside each cubic."""
        local = u[self.connectivity]
        offset = local[:, 0]
        a = (local - offset[:, None]) @ self.Vinverse.T
        qa, qb, qc = 3 * a[:, 3], 2 * a[:, 2], a[:, 1]
        discriminant = qb * qb - 4 * qa * qc
        roots = []
        nonlinear = np.abs(qa) > 1e-24
        for sign in [-1.0, 1.0]:
            z = np.full(self.elements, np.nan)
            good = nonlinear & (discriminant >= 0)
            z[good] = (-qb[good] + sign * np.sqrt(discriminant[good])) / (2 * qa[good])
            roots.append(z)
        linear = (~nonlinear) & (np.abs(qb) > 1e-24)
        roots[0][linear] = -qc[linear] / qb[linear]
        lo = float(u.min())
        hi = float(u.max())
        for z in roots:
            good = np.isfinite(z) & (z > -1) & (z < 1)
            val = (
                offset[good]
                + ((a[good, 3] * z[good] + a[good, 2]) * z[good] + a[good, 1]) * z[good]
                + a[good, 0]
            )
            if len(val):
                lo = min(lo, float(val.min()))
                hi = max(hi, float(val.max()))
        return lo, hi


def read_inputs(directory):
    wb = load_workbook(directory / "附件1.xlsx", read_only=True, data_only=True)
    try:
        ambient = np.array(
            list(wb["Sheet1"].iter_rows(min_row=2, max_col=3, values_only=True)), dtype=float
        )
    finally:
        wb.close()
    wb = load_workbook(directory / "附件2.xlsx", read_only=True, data_only=True)
    try:
        raw = np.array(
            list(wb.active.iter_rows(min_row=2, max_col=2, values_only=True)), dtype=float
        )
    finally:
        wb.close()
    # Input columns use seconds and centimetres, as verified against the attachment header.
    radius = raw.copy()
    radius[:, 1] /= 100
    if ambient.shape != (241, 3) or ambient[-1, 0] != 14400 or radius[-1, 0] != 259200:
        raise ValueError("Unexpected input layout or time units")
    mask = ambient[:, 0] >= 10800
    tails = np.trapezoid(ambient[mask, 1:], ambient[mask, 0], axis=0) / 3600
    return ambient, radius, tails


class IndependentModel:
    def __init__(self, mesh, properties, ambient, radius, tails, moving=False):
        self.mesh = mesh
        self.p = properties
        self.ambient = ambient
        self.radius = radius
        self.tails = tails
        self.moving = moving
        self.h = 25.0
        self.hm = 8e-7

    def environment(self, t, right_tail=False):
        if t > 14400 or (right_tail and t >= 14400):
            return self.tails[0] + 273.15, self.tails[1]
        return np.interp(t, self.ambient[:, 0], self.ambient[:, 1]) + 273.15, np.interp(
            t, self.ambient[:, 0], self.ambient[:, 2]
        )

    def R(self, t):
        return float(np.interp(t, self.radius[:, 0], self.radius[:, 1])) if self.moving else 0.02

    def coefficients(self, T, C):
        if np.any(C <= 0) or np.any(T <= 0):
            raise FloatingPointError("Nonpositive nodal state; no clipping is performed")
        p = self.p
        rho = p["rho_offset"] + p["rho_C_coefficient"] * C
        cp = p["cp_offset"] + p["cp_fraction_coefficient"] * C / (1 + C)
        k = p["k_offset"] + p["k_fraction_coefficient"] * C / (1 + C)
        D = p["D_prefactor"] * np.exp(
            -p["D_moisture_exponent"] / C - p["D_temperature_exponent_K"] / T
        )
        return rho * cp, k, D

    def rhs(self, t, y, right_tail=False):
        m = self.mesh
        n = m.nodes
        T, C = y[:n], y[n:]
        b, k, D = self.coefficients(T, C)
        R = self.R(t)
        Ta, Ce = self.environment(t, right_tail)
        heat = -4 / R**2 * m.stiffness_action(T, k)
        water = -4 / R**2 * m.stiffness_action(C, D)
        heat[-1] -= 2 * self.h / R * (T[-1] - Ta)
        water[-1] -= 2 * self.hm / R * (C[-1] - Ce)
        return np.r_[heat / (m.mass * b), water / m.mass]

    def knots(self, end):
        values = [np.array([0.0, end]), self.ambient[:, 0]]
        if self.moving:
            values.append(self.radius[:, 0])
        values = np.unique(np.concatenate(values))
        return values[(values >= 0) & (values <= end)]


def independent_checks(out):
    rows = []
    algebra = []
    # Variable coefficient + moving radius MMS, and an independent exact Robin eigenmode.
    for family in ["variable_coefficient_moving_MMS", "Robin_Bessel_eigenmode"]:
        for ne in [8, 16, 32]:
            m = CubicGLL(ne)
            s = m.s
            constant_residual = float(
                np.max(np.abs(m.stiffness_action(np.ones(m.nodes), np.ones(m.nodes))))
            )
            test_u = np.sin(2 * s) + 0.3 * s
            weak_sum = float(abs(m.stiffness_action(test_u, 1 + s).sum()))
            algebra.append(
                {
                    "case": family,
                    "elements": ne,
                    "mass_sum": float(m.mass.sum()),
                    "minimum_mass": float(m.mass.min()),
                    "constant_action_max": constant_residual,
                    "weak_internal_flux_sum_abs": weak_sum,
                }
            )
            if family == "variable_coefficient_moving_MMS":

                def exact(t):
                    return 1 + 0.1 * np.exp(-t) * np.cos(np.pi * s)

                def rhs(t, u):
                    radius = 1 - 0.1 * t
                    D = 0.05 * (1 + 0.2 * s) * (1 + 0.1 * t)
                    Ds = 0.01 * (1 + 0.1 * t)
                    truth = exact(t)
                    b = 1 + 0.2 * truth
                    ut = -0.1 * np.exp(-t) * np.cos(np.pi * s)
                    us = -0.1 * np.pi * np.exp(-t) * np.sin(np.pi * s)
                    uss = -0.1 * np.pi**2 * np.exp(-t) * np.cos(np.pi * s)
                    forcing = b * ut - 4 / radius**2 * ((D + s * Ds) * us + s * D * uss)
                    return (-4 / radius**2 * m.stiffness_action(u, D) / m.mass + forcing) / b

            else:
                kappa = 0.1
                h = 0.2
                eq = 0.4
                lam = brentq(lambda z: z * j1(z) - h / kappa * j0(z), 1e-10, 2.4048255577)

                def exact(t):
                    return eq + np.exp(-kappa * lam**2 * t) * j0(lam * np.sqrt(s))

                def rhs(t, u):
                    weak = -4 * m.stiffness_action(u, np.full(m.nodes, kappa))
                    weak[-1] -= 2 * h * (u[-1] - eq)
                    return weak / m.mass

            tic = time.perf_counter()
            sol = solve_ivp(
                rhs,
                (0, 0.3),
                exact(0),
                method="BDF",
                rtol=1e-11,
                atol=1e-13,
                max_step=0.005,
                jac_sparsity=m.scalar_sparsity,
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            error = sol.y[:, -1] - exact(0.3)
            rows.append(
                {
                    "check": family,
                    "elements": ne,
                    "nodes": m.nodes,
                    "max_abs_error": float(np.max(np.abs(error))),
                    "mass_L2_error": float(np.sqrt(np.dot(m.mass, error**2))),
                    "wall_seconds": time.perf_counter() - tic,
                }
            )
    passed = all(
        x["minimum_mass"] > 0
        and abs(x["mass_sum"] - 1) < 1e-14
        and x["constant_action_max"] == 0
        and x["weak_internal_flux_sum_abs"] < 1e-10
        for x in algebra
    )
    for family in ["variable_coefficient_moving_MMS", "Robin_Bessel_eigenmode"]:
        group = [r for r in rows if r["check"] == family]
        for a, b in zip(group[:-1], group[1:]):
            b["observed_max_error_order"] = float(np.log2(a["max_abs_error"] / b["max_abs_error"]))
        passed = (
            passed
            and group[-1]["max_abs_error"] < 1e-7
            and group[-1]["max_abs_error"] < group[0]["max_abs_error"] / 8
        )
    report = {
        "passed": bool(passed),
        "rows": rows,
        "algebra_checks": algebra,
        "MMS_definition": (
            "u=1+0.1 exp(-t) cos(pi s), R=1-0.1t, D=.05(1+.2s)(1+.1t), b=1+.2u_exact; "
            "forcing=b*u_t-4/R^2*d_s(s D u_s), zero outer derivative."
        ),
        "analytic_definition": (
            "u=.4+exp(-.1 lambda^2 t) J0(lambda sqrt(s)); lambda J1(lambda)=2 "
            "J0(lambda), R=1, diffusivity=.1, Robin h=.2."
        ),
        "interpretation": (
            "Checks test independent weak assembly, moving-radius factor, "
            "variable-coefficient divergence, and Robin boundary sign/scaling. They do "
            "not validate physical assumptions."
        ),
    }
    write_json(out / "method_checks.json", report)
    if not passed:
        raise RuntimeError(
            "Independent manufactured/analytic checks failed; no production pass can be claimed"
        )
    return report


def evaluate_segments(segments, times):
    ends = np.array([b for a, b, sol in segments])
    values = []
    for t in times:
        i = min(int(np.searchsorted(ends, t, side="left")), len(segments) - 1)
        a, b, sol = segments[i]
        if t < a - 1e-8 or t > b + 1e-8:
            raise ValueError("Dense output extrapolation is not permitted")
        values.append(sol.sol(t))
    return np.asarray(values)


def integrate_segment(model, start, end, state, *, event=None, right_tail=None):
    """按统一精度推进一段；环境跳点和事件后的补算由调用方显式指定。"""
    mesh = model.mesh
    options = dict(
        method="BDF",
        rtol=1e-10,
        atol=np.r_[np.full(mesh.nodes, 1e-10), np.full(mesh.nodes, 1e-12)],
        jac_sparsity=mesh.coupled_sparsity,
        max_step=75.0,
        dense_output=True,
    )
    if start == 0:
        capacity, conductivity, diffusivity = model.coefficients(
            state[: mesh.nodes], state[mesh.nodes :]
        )
        spacing = np.diff(np.sqrt(mesh.s)) * 0.02
        options["first_step"] = min(
            0.001,
            0.03
            * float(spacing.min()) ** 2
            / float(max(np.max(conductivity / capacity), diffusivity.max())),
        )
    right_tail = bool(start >= 14400) if right_tail is None else right_tail
    solution = solve_ivp(
        lambda t, u: model.rhs(t, u, right_tail),
        (float(start), float(end)),
        state,
        events=event,
        **options,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def run_identity(reference):
    """绑定当前源码与该次计算使用的原始参考数组。"""
    identity = source_identity(extra_files=(Path(__file__),))
    identity["reference_file"] = sha256(reference)
    return identity


def run_case(out, case, ne, ambient, radius, tails, cfg):
    label = f"{case}_E{ne}"
    dest = out / label
    ref_path, ref, event_ref = reference_data(case)
    protected = run_identity(ref_path)
    if (dest / "record.json").exists():
        old = read_json(dest / "record.json")
        if (
            old["source_hashes_before"] != protected
            or sha256(dest / "samples.npz") != old["samples_sha256"]
        ):
            raise RuntimeError("Existing independent run changed: " + label)
        print(json.dumps({"run": label, "reused_verified_run": True}), flush=True)
        return old
    mesh = CubicGLL(ne)
    model = IndependentModel(mesh, cfg["property_sets"][case], ambient, radius, tails, case == "P4")
    initial = np.r_[np.full(mesh.nodes, 301.15), np.full(mesh.nodes, 2.55)]
    end = 1800.0 if case == "P1" else max(event_ref * 1.02, event_ref + 3600)
    knots = model.knots(end)
    segments = []
    y = initial
    root_time = None
    root_state = None
    start = time.perf_counter()
    nfev = 0
    steps = 0

    def event(t, u):
        return mesh.extrema(u[mesh.nodes :])[1] - 0.15

    event.terminal = True
    event.direction = -1
    for a, b in zip(knots[:-1], knots[1:]):
        sol = integrate_segment(model, a, b, y, event=event if case != "P1" else None)
        actual_end = float(sol.t[-1])
        segments.append((float(a), actual_end, sol))
        nfev += sol.nfev
        steps += len(sol.t) - 1
        y = sol.y[:, -1]
        if case != "P1" and len(sol.t_events[0]):
            root_time = float(sol.t_events[0][0])
            root_state = y.copy()
            break
    if case != "P1" and root_time is None:
        raise RuntimeError("No polynomial-global-maximum event within diagnostic horizon")
    if segments[-1][1] < ref["t_s"][-1]:
        a = segments[-1][1]
        b = float(ref["t_s"][-1])
        sol = integrate_segment(model, a, b, y, right_tail=True)
        segments.append((a, b, sol))
        nfev += sol.nfev
        steps += len(sol.t) - 1
    states = evaluate_segments(segments, ref["t_s"])
    T = states[:, : mesh.nodes]
    C = states[:, mesh.nodes :]
    sampled_T = mesh.evaluate(T, ref["xi"] ** 2)
    sampled_C = mesh.evaluate(C, ref["xi"] ** 2)
    # Inspect early subsecond layers and all comparison times, including cubic interior extrema.
    diagnostic_times = np.unique(
        np.r_[
            ref["t_s"],
            [1e-6, 1e-4, 0.001, 0.01, 0.1, 10.0],
            root_time if root_time is not None else 1800.0,
        ]
    )
    diagnostic_states = evaluate_segments(segments, diagnostic_times)
    extrema = np.array([mesh.extrema(v[mesh.nodes :]) for v in diagnostic_states])
    root_global = None
    root_nodal = None
    if root_state is not None:
        root_global = mesh.extrema(root_state[mesh.nodes :])[1]
        root_nodal = float(root_state[mesh.nodes :].max())
    dest.mkdir(parents=True, exist_ok=True)
    data = {
        "t_s": ref["t_s"],
        "xi": ref["xi"],
        "T_K": sampled_T,
        "C": sampled_C,
        "reference_T_K": ref["T_K"],
        "reference_C": ref["C"],
        "diagnostic_t_s": diagnostic_times,
        "diagnostic_C_min": extrema[:, 0],
        "diagnostic_C_max": extrema[:, 1],
        "mesh_s": mesh.s,
        "mass_weights": mesh.mass,
        "event_state": root_state if root_state is not None else states[-1],
    }
    np.savez_compressed(dest / "samples.npz", **data)
    after = run_identity(ref_path)
    if protected != after:
        raise RuntimeError("Independent run protected source changed")
    record = {
        "label": label,
        "case": case,
        "elements": ne,
        "nodes": mesh.nodes,
        "degree": 3,
        "radial_element_power": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - start,
        "method": "Independent P3 GLL lumped-mass weak form in s=(r/R)^2; independent sparse BDF wrapper",
        "settings": {"rtol": 1e-10, "atol_T_K": 1e-10, "atol_C": 1e-12, "max_step_s": 75},
        "reference_file": str(ref_path.relative_to(DATA)).replace("\\", "/"),
        "source_hashes_before": protected,
        "source_hashes_after": after,
        "samples_sha256": sha256(dest / "samples.npz"),
        "reference_event_root_s": event_ref,
        "event_root_s": root_time,
        "event_root_h": root_time / 3600 if root_time else None,
        "event_difference_s": root_time - event_ref if root_time else None,
        "event_global_polynomial_max_C": root_global,
        "event_nodal_max_C": root_nodal,
        "event_definition": (
            "First decreasing maximum of the full piecewise-cubic interpolant crossing "
            "0.15; all real interior derivative roots are checked. Equality infimum, not "
            "earliest attained strict time."
        ),
        "max_T_difference_to_reference_K": float(np.max(np.abs(sampled_T - ref["T_K"]))),
        "max_C_difference_to_reference": float(np.max(np.abs(sampled_C - ref["C"]))),
        "comparison_rows": len(ref["t_s"]),
        "comparison_radii": len(ref["xi"]),
        "minimum_sampled_polynomial_C": float(extrema[:, 0].min()),
        "maximum_sampled_polynomial_C": float(extrema[:, 1].max()),
        "maximum_sampled_C_overshoot_above_initial": float(max(0.0, extrema[:, 1].max() - 2.55)),
        "all_sampled_fields_finite": bool(np.isfinite(states).all()),
        "no_clipping_or_limiter_used": True,
        "steps": steps,
        "nfev": nfev,
        "limitations": (
            "Cubic GLL discretization is not a general positivity/monotonicity theorem; "
            "diagnostic extrema are checked at saved early and full-timespan samples. "
            "Agreement with FV alone is not proof."
        ),
    }
    write_json(dest / "record.json", record)
    print(
        json.dumps(
            {
                "run": label,
                "event_h": record["event_root_h"],
                "event_diff_s": record["event_difference_s"],
                "max_C_reference_difference": record["max_C_difference_to_reference"],
                "overshoot": record["maximum_sampled_C_overshoot_above_initial"],
                "seconds": record["wall_seconds"],
            }
        ),
        flush=True,
    )
    del segments, states, diagnostic_states
    gc.collect()
    return record


def reference_data(case):
    """读取原始查询节点和参考场，绑定对应的终点记录。"""
    directory = EVIDENCE / f"{case}_E256"
    path = directory / "samples.npz"
    arrays = load_npz(path, ("t_s", "xi", "reference_T_K", "reference_C"))
    reference = {
        "t_s": arrays["t_s"],
        "xi": arrays["xi"],
        "T_K": arrays["reference_T_K"],
        "C": arrays["reference_C"],
    }
    record = read_json(directory / "receipt.json")["source_record"]
    return path, reference, record["reference_event_root_s"]


def readback(out):
    """Check frozen data identity and recompute stored reference differences."""
    integrity = verify_data()
    source = read_json(SOURCE_REPORT)
    rows = []
    for case in ("P1", "P23", "P4"):
        for elements in (64, 128, 256):
            label = f"{case}_E{elements}"
            file = EVIDENCE / label / "samples.npz"
            expected = source["source_files"][f"fem:{label}/samples.npz"]["sha256"]
            if sha256(file) != expected:
                raise ValueError("原始GLL数据身份不符: " + label)
            z = load_npz(file)
            if not all(
                np.isfinite(z[k]).all()
                for k in ("t_s", "xi", "T_K", "C", "reference_T_K", "reference_C")
            ):
                raise ValueError("非有限字段")
            row = {
                "case": case,
                "elements": elements,
                "time_rows": len(z["t_s"]),
                "radii": len(z["xi"]),
                "samples_sha256": expected,
                "reference_T_difference_K": float(abs(z["T_K"] - z["reference_T_K"]).max()),
                "reference_C_difference": float(abs(z["C"] - z["reference_C"]).max()),
            }
            receipt = read_json(file.parent / "receipt.json")
            if (
                row["reference_T_difference_K"] != receipt["recomputed_max_T_to_stored_reference_K"]
                or row["reference_C_difference"] != receipt["recomputed_max_C_to_stored_reference"]
            ):
                raise ValueError("参考场差异记录不符")
            rows.append(row)
    result = {
        "passed": True,
        "rows": rows,
        "data_integrity": integrity,
        "archived_final_FV_pairing": source["recomputed_high_order_comparison"]["rows"],
        "current_execution_identity": source_identity(extra_files=(Path(__file__),)),
        "scope": "保存数组及其内含参考场的重算。最终FV配对摘要单独读回；compare模式计算实际提供的FV数组。",
    }
    write_json(out / "readback.json", result)
    return result


def compare_saved_fields(left, right):
    """比较相同时间与材料节点上的两组保存场，拒绝错位或无效数组。"""
    a, b = load_npz(left), load_npz(right)
    if any(not np.array_equal(a[key], b[key]) for key in ("t_s", "xi")):
        raise ValueError("共同查询位置改变")
    shape = (len(a["t_s"]), len(a["xi"]))
    differences = {}
    for key, output in (("T_K", "max_T_K"), ("C", "max_C")):
        if a[key].shape != shape or b[key].shape != shape:
            raise ValueError("保存场形状与查询节点不符")
        if not np.isfinite(a[key]).all() or not np.isfinite(b[key]).all():
            raise ValueError("保存场包含非有限值")
        differences[output] = float(abs(a[key] - b[key]).max())
    return differences


def solve_all(out, cases, elements):
    """Integrate each requested physical GLL problem and compare its complete saved queries."""
    checks = independent_checks(out / "method_checks")
    ambient, radius, tails = read_inputs(DATA / "inputs/附件")
    config = read_json(DATA / "config/model.json")
    records = []
    comparisons = []
    for case in cases:
        for count in elements:
            record = run_case(out, case, count, ambient, radius, tails, config)
            records.append(record)
            source = EVIDENCE / f"{case}_E{count}" / "samples.npz"
            if source.exists():
                difference = compare_saved_fields(source, out / record["label"] / "samples.npz")
                comparisons.append(
                    {
                        "case": case,
                        "elements": count,
                        **difference,
                        "passed": all(value < 1e-7 for value in difference.values()),
                    }
                )
    convergence = []
    for case in cases:
        ordered = sorted((r for r in records if r["case"] == case), key=lambda r: r["elements"])
        for coarse, fine in zip(ordered[:-1], ordered[1:]):
            convergence.append(
                {
                    "case": case,
                    "coarse": coarse["elements"],
                    "fine": fine["elements"],
                    **compare_saved_fields(
                        out / coarse["label"] / "samples.npz", out / fine["label"] / "samples.npz"
                    ),
                    "root_difference_s": (
                        None if case == "P1" else fine["event_root_s"] - coarse["event_root_s"]
                    ),
                }
            )
    report = {
        "completed": True,
        "passed": checks["passed"] and all(r["passed"] for r in comparisons),
        "records": records,
        "comparison_to_archived_GLL": comparisons,
        "grid_convergence": convergence,
        "current_execution_identity": source_identity(extra_files=(Path(__file__),)),
        "scope": "完整独立GLL数值解、原始保存数组一致性和网格差异；1e-7为跨运行一致性阈值。",
    }
    write_json(out / "summary.json", report)
    if not report["passed"]:
        raise RuntimeError("一致性检查失败，请检查报告。")
    return report


def compare_fv(out, paths):
    """Pair every exact common time and existing material node, and report unmatched rows."""
    rows = []
    for case, path in zip(("P1", "P23", "P4"), paths):
        if path is None:
            raise ValueError("compare需要三个FV文件参数")
        path = Path(path).resolve()
        fv = load_npz(path)
        ft = fv["t_s"]
        fx = fv["xi"] if "xi" in fv else fv["r_m"] / 0.02
        if fx.ndim != 1 or not np.all(np.diff(ft) > 0):
            raise ValueError("FV坐标或时间顺序无效")
        for count in (64, 128, 256):
            fe = load_npz(EVIDENCE / f"{case}_E{count}" / "samples.npz")
            times, i, j = np.intersect1d(fe["t_s"], ft, return_indices=True)
            nearest = np.argmin(abs(fx[:, None] - fe["xi"][None, :]), axis=0)
            if not len(times) or np.max(abs(fx[nearest] - fe["xi"])) > 1e-14:
                raise ValueError("缺少共同时间或材料节点")
            differences = {}
            for field in ("T_K", "C"):
                a = fe[field][i]
                b = fv[field][j][:, nearest]
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("场存在非有限值")
                delta = abs(a - b)
                ix = np.unravel_index(np.argmax(delta), delta.shape)
                differences[field] = {
                    "maximum": float(delta[ix]),
                    "t_s": float(times[ix[0]]),
                    "xi": float(fe["xi"][ix[1]]),
                    "FE": float(a[ix]),
                    "FV": float(b[ix]),
                }
            rows.append(
                {
                    "case": case,
                    "elements": count,
                    "common_time_rows": len(times),
                    "first_time_s": float(times[0]),
                    "last_time_s": float(times[-1]),
                    "unmatched_FE_times_s": np.setdiff1d(fe["t_s"], ft).tolist(),
                    "differences": differences,
                    "FV_sha256": sha256(path),
                    "time_interpolation": False,
                    "space_interpolation": False,
                }
            )
    result = {
        "completed": True,
        "rows": rows,
        "current_execution_identity": source_identity(extra_files=(Path(__file__),)),
    }
    write_json(out / "FV_GLL_comparison.json", result)
    return result


def observer_checks(out):
    """有理系数时间多项式独立核对移动域热量观察器和终点截断。"""
    F = Fraction
    shift = [F(1, 4), F(-1, 4), F(1, 2)]
    denominators = [F(2), F(4), F(8)]
    data = [
        [F(1, 2), F(3, 4), F(5, 4), F(3, 2)],
        [F(1, 4), F(-1, 8), F(-1, 4), F(-1, 8)],
        [F(1, 8), F(1, 16), F(1, 32), F(-1, 32)],
        [F(1, 16), F(-1, 32), F(-1, 64), F(1, 64)],
    ]
    # Polynomial的object系数保持Fraction精确有理运算。
    fields = [Polynomial([data[0][j]]) for j in range(4)]
    basis = Polynomial([F(1)])
    for k in range(3):
        basis *= Polynomial([-shift[k] / denominators[k], 1 / denominators[k]])
        for j in range(4):
            fields[j] += basis * data[k + 1][j]

    class KnownDensePolynomial:
        order = 3
        t_shift = np.asarray(shift, dtype=float)
        denom = np.asarray(denominators, dtype=float)
        D = np.asarray(data, dtype=float)

        def __call__(self, t):
            return np.array(
                [
                    np.polynomial.polynomial.polyval(t, np.asarray(f.coef, dtype=float))
                    for f in fields
                ]
            )

    model = SimpleNamespace(
        nr=2,
        grid=SimpleNamespace(radius_m=1.0, volume_weights_m2=np.array([0.25, 0.25])),
        temperature_reference_K=300.0,
        h_m_m_s=1 / 16,
        h_W_m2_K=1 / 8,
        geometry=SimpleNamespace(radius=lambda t: 1 + np.asarray(t) / 8),
        ambient=SimpleNamespace(
            values=lambda t, **kw: SimpleNamespace(
                moisture_eq=np.asarray(t) * 0 + 1 / 8, temperature_K=np.asarray(t) * 0 + 300.25
            )
        ),
        moisture_boundary_multiplier=lambda t, R, R0: R,
        properties=SimpleNamespace(
            evaluate=lambda C, T: (
                np.ones_like(C) * 2,
                np.ones_like(C) * 3,
                np.ones_like(C),
                np.ones_like(C),
            )
        ),
    )
    observer = science("finest_balance_audit").Observer(0.75, 1.25)
    exact = {key: F(0) for key in ("moisture_flux", "heat_flux", "heat_store")}
    polynomial = KnownDensePolynomial()
    for origin, left, right, detect in [
        (F(0), F(1, 8), F(3, 4), True),
        (F(3, 4), F(0), F(1, 2), False),
    ]:
        solver = SimpleNamespace(t=1.0, dense_output=lambda: polynomial)
        observer.observe(
            solver,
            float(left),
            {"model": model, "start": float(origin), "detect": detect, "right_side": True},
        )
        radius = Polynomial([1 + origin / 8, F(1, 8)])
        coefficients = {
            "moisture_flux": (fields[3] - F(1, 8)) * F(1, 8),
            "heat_flux": radius * (fields[1] - F(1, 4)) * F(1, 4),
            "heat_store": radius * radius * (fields[0].deriv() + fields[1].deriv()) * F(3),
        }
        for name, coefficient in coefficients.items():
            exact[name] += sum(
                value * (right ** (i + 1) - left ** (i + 1)) / F(i + 1)
                for i, value in enumerate(coefficient.coef)
            )
    rows = []
    for order, row in observer.rows.items():
        errors = {key: abs(row[key] - float(value)) for key, value in exact.items()}
        rows.append(
            {
                "quadrature_order": order,
                "intervals": row["intervals"],
                "absolute_errors_against_exact_fraction_integrals": errors,
                "passed": bool(row["intervals"] == 2 and max(errors.values()) < 2e-14),
            }
        )
    result = {
        "completed": True,
        "passed": all(r["passed"] for r in rows),
        "checks": rows,
        "exact_integrals": {key: str(value) for key, value in exact.items()},
        "current_execution_identity": source_identity(),
        "scope": "独立有理幂基导数和精确积分，覆盖移动几何热权重及root/report时间截断。",
    }
    write_json(out / "observer_checks.json", result)
    if not result["passed"]:
        raise RuntimeError("一维守恒观察器精确多项式核验未通过")
    return result


class RefinementComparison:
    """比较已有细化轨迹，保留物理时间、半径列与四位小数逐格差异。

    这里只做离散结果之间的经验比较。共同时间直接配对，空间场使用
    有限体积节点的分段线性重构；终点处明确记录保存时刻的位移。
    """

    def __init__(self, production: Path):
        self.production = production

    load = staticmethod(load_npz)

    @staticmethod
    def rounded(value):
        return Decimal(str(float(value))).quantize(Decimal(".0001"), rounding=ROUND_HALF_UP)

    @classmethod
    def changed_cells(cls, original, refined, times, field):
        if original.shape != refined.shape or not np.array_equal(
            np.isfinite(original), np.isfinite(refined)
        ):
            raise ValueError("表格形状或有效物理位置掩码不一致。")
        changed = []
        for row, column in zip(*np.where(np.isfinite(original))):
            left, right = original[row, column], refined[row, column]
            before, after = cls.rounded(left), cls.rounded(right)
            if before != after:
                changed.append(
                    {
                        "field": field,
                        "row": int(row),
                        "column": int(column),
                        "time_s": float(times[row]),
                        "formal_value": float(left),
                        "fine_value": float(right),
                        "formal_four_decimal": str(before),
                        "fine_four_decimal": str(after),
                    }
                )
        return changed

    @classmethod
    def compare_arrays(cls, original, refined, times, coordinates):
        if not len(times):
            raise ValueError("两条轨迹没有完全共同的时间节点。")
        result = {
            "time_rows": len(times),
            "space_nodes": len(coordinates),
            "values_per_field": len(times) * len(coordinates),
            "time_interpolation_used": False,
        }
        for field in ("T_K", "C"):
            before, after = original[field], refined[field]
            if before.shape != (len(times), len(coordinates)) or before.shape != after.shape:
                raise ValueError("比较字段形状与时间/位置节点不一致。")
            if not np.isfinite(before).all() or not np.isfinite(after).all():
                raise ValueError("完整场比较包含非有限值。")
            difference = np.abs(before - after)
            index = np.unravel_index(np.argmax(difference), difference.shape)
            result[field] = {
                "maximum_absolute_difference": float(difference[index]),
                "at_time_s": float(times[index[0]]),
                "at_xi": float(coordinates[index[1]]),
                "four_decimal_changed_values": len(cls.changed_cells(before, after, times, field)),
            }
        return result

    @classmethod
    def sample_comparison(cls, coarse, fine):
        a, b = cls.load(coarse / "samples.npz"), cls.load(fine / "samples.npz")
        if not np.array_equal(a["xi"], b["xi"]):
            raise ValueError("细化比较的输出位置节点不一致。")
        times, ia, ib = np.intersect1d(a["t_s"], b["t_s"], return_indices=True)
        return cls.compare_arrays(
            {k: a[k][ia] for k in ("T_K", "C")}, {k: b[k][ib] for k in ("T_K", "C")}, times, a["xi"]
        )

    @staticmethod
    def time_index(times, target):
        index = int(np.argmin(np.abs(times - target)))
        displacement = float(times[index] - target)
        if abs(displacement) >= 5e-8:
            raise ValueError(f"缺少指定物理时刻的保存场：{target} s。")
        return index, displacement

    def table_comparisons(self, case, fine):
        samples = self.load(fine / "samples.npz")
        checkpoints = self.load(fine / "checkpoints.npz")
        tables = []
        for filename in ("paper_first3h.npz", "paper_full.npz"):
            path = self.production / case / filename
            if not path.exists():
                continue
            original = self.load(path)
            fields = {k: [] for k in ("T_K", "C")}
            displacements = []
            for row, target in enumerate(original["t_s"]):
                source = samples if case == "p23" else checkpoints
                index, displacement = self.time_index(source["t_s"], target)
                displacements.append(displacement)
                for field in fields:
                    if case == "p23":
                        values = np.interp(
                            [0, 0.25, 0.5, 0.75, 1], source["xi"], source[field][index]
                        )
                    else:
                        radii = original["r_sample_m"]
                        valid = original["valid_interior"][row]
                        values = np.full(len(radii) + 1, np.nan)
                        values[:-1][valid] = np.interp(
                            radii[valid] / original["R_m"][row], source["xi"], source[field][index]
                        )
                        values[-1] = source[field][index, -1]
                    fields[field].append(values)
            if case == "p23":
                original_T, original_C = original["T_K"], original["C"]
            else:
                original_T = np.column_stack([original["T_K_sample"], original["T_K_surface"]])
                original_C = np.column_stack([original["C_sample"], original["C_surface"]])
            refined_T, refined_C = np.array(fields["T_K"]), np.array(fields["C"])
            temperature_changes = self.changed_cells(
                original_T - 273.15, refined_T - 273.15, original["t_s"], "temperature_Celsius"
            )
            moisture_changes = self.changed_cells(
                original_C, refined_C, original["t_s"], "moisture_C"
            )
            tables.append(
                {
                    "file": filename,
                    "T_max_difference_K": float(np.nanmax(abs(original_T - refined_T))),
                    "C_max_difference": float(np.nanmax(abs(original_C - refined_C))),
                    "T_four_decimal_changes": len(temperature_changes),
                    "C_four_decimal_changes": len(moisture_changes),
                    "changed_paper_cells": temperature_changes + moisture_changes,
                    "actual_sample_minus_original_time_s": displacements,
                    "physical_radius_columns_preserved": case == "q4",
                    "source_sha256": sha256(path),
                }
            )
        return tables

    def fixed_grid_comparison(self, fine):
        samples = self.load(fine / "samples.npz")
        index_path = self.production / "p23/data/q2_fixed_grid_1s_index.json"
        index = read_json(index_path)
        original = {k: [] for k in ("T_K", "C")}
        refined = {k: [] for k in original}
        times, sources = [], []
        for entry in index["chunks"]:
            path = self.production / "p23" / entry["file"]
            chunk = self.load(path)
            common, ia, ib = np.intersect1d(chunk["t_s"], samples["t_s"], return_indices=True)
            if not len(common):
                continue
            xi = chunk["r_m"] / chunk["r_m"][-1]
            if xi.shape != samples["xi"].shape or np.max(abs(xi - samples["xi"])) >= 1e-14:
                raise ValueError("第二问固定半径列与细化输出节点不对应。")
            for field in original:
                original[field].append(chunk[field][ia])
                refined[field].append(
                    np.array([np.interp(xi, samples["xi"], samples[field][j]) for j in ib])
                )
            times.append(common)
            sources.append({"file": entry["file"], "sha256": sha256(path)})
        if not times:
            raise ValueError("第二问分块场与细化场没有共同时间。")
        times = np.concatenate(times)
        result = self.compare_arrays(
            {k: np.concatenate(v) for k, v in original.items()},
            {k: np.concatenate(v) for k, v in refined.items()},
            times,
            samples["xi"],
        )
        result.update(
            all_seconds_1_through_10800_included=bool(
                np.all(np.isin(np.arange(1.0, 10801.0), times))
            ),
            original_chunk_sources=sources,
            index_sha256=sha256(index_path),
        )
        return result

    def compare_case(self, case, root):
        coarse_n = 1280 if case == "p23" else 2560
        directories = [
            root / f"{case}_N{coarse_n}_rtol1e-11",
            root / f"{case}_N{2 * coarse_n}_rtol1e-11",
            root / f"{case}_N{2 * coarse_n}_rtol1e-12",
        ]
        records = [read_json(d / "record.json") for d in directories]
        endpoint_path = self.production / case / "endpoint.json"
        endpoint = read_json(endpoint_path)
        fine = directories[-1]
        checkpoints = self.load(fine / "checkpoints.npz")
        target = float(endpoint["reported_time_s"])
        refined_index, displacement = self.time_index(checkpoints["t_s"], target)
        maximum = float(checkpoints["C"][refined_index].max())
        endpoint_check = {
            "requested_report_time_s": target,
            "stored_state_time_s": float(checkpoints["t_s"][refined_index]),
            "stored_minus_report_s": displacement,
            "unrounded_full_nodal_max_C": maximum,
            "strictly_dry_at_stored_time": maximum < 0.15,
            "scope": "在明确保存的物理时刻直接检查未舍入场，并记录时刻位移。",
        }
        original_path = self.production / case / "internal.npz"
        original = self.load(original_path)
        original_index, original_displacement = self.time_index(original["t_s"], target)
        union = np.union1d(original["xi"], checkpoints["xi"])
        endpoint_difference = {}
        for field in ("T_K", "C"):
            refined_values = np.interp(union, checkpoints["xi"], checkpoints[field][refined_index])
            original_values = np.interp(union, original["xi"], original[field][original_index])
            endpoint_difference[field] = float(np.max(abs(refined_values - original_values)))
        samples = self.load(fine / "samples.npz")
        times, io, iff = np.intersect1d(original["t_s"], samples["t_s"], return_indices=True)
        original_fields = {
            k: np.array([np.interp(samples["xi"], original["xi"], original[k][j]) for j in io])
            for k in ("T_K", "C")
        }
        run_records = []
        for directory, record in zip(directories, records):
            run_records.append(
                {
                    "directory": directory.name,
                    "record_sha256": sha256(directory / "record.json"),
                    "samples_sha256": sha256(directory / "samples.npz"),
                    "checkpoints_sha256": sha256(directory / "checkpoints.npz"),
                    "N": record["N"],
                    "settings": record["settings"],
                    "equality_root_s": record["event_root_s"],
                    "accepted_steps": record["accepted_steps_including_post"],
                    "local_steps_unresolvable_by_absolute_clock": sum(
                        s.get("steps_unresolvable_by_absolute_clock", 0) for s in record["segments"]
                    ),
                }
            )
        result = {
            "case": case,
            "completed": True,
            "continuous_PDE_certified": False,
            "runs": run_records,
            "matched_tolerance_fine_minus_coarse_root_s": records[1]["event_root_s"]
            - records[0]["event_root_s"],
            "same_fine_grid_tighter_minus_main_root_s": records[2]["event_root_s"]
            - records[1]["event_root_s"],
            "tight_fine_minus_formal_root_s": records[2]["event_root_s"]
            - endpoint["equality_root_s"],
            "remaining_formal_report_delay_after_refined_root_s": target
            - records[2]["event_root_s"],
            "matched_tolerance_spatial_samples": self.sample_comparison(
                directories[0], directories[1]
            ),
            "same_fine_grid_tolerance_samples": self.sample_comparison(
                directories[1], directories[2]
            ),
            "tight_fine_vs_formal_common_minute_samples": self.compare_arrays(
                original_fields, {k: samples[k][iff] for k in ("T_K", "C")}, times, samples["xi"]
            ),
            "formal_report_endpoint": endpoint_check,
            "formal_stored_minus_report_s": original_displacement,
            "at_report_piecewise_linear_union_grid_difference": endpoint_difference,
            "paper_table_comparisons": self.table_comparisons(case, fine),
            "formal_endpoint_sha256": sha256(endpoint_path),
            "formal_internal_sha256": sha256(original_path),
            "limitations": [
                "差异属于离散结果对照，并非连续PDE空间误差上界。",
                "21个归一化材料位置的采样与收缩域工作簿的固定物理半径列分别处理。",
                "完整径向场按实际保存时刻比较；时间位移单独记录。",
            ],
        }
        if case == "p23":
            result["tight_fine_vs_formal_Q2_available_seconds_and_minutes"] = (
                self.fixed_grid_comparison(fine)
            )
        return result


def compare_rounding(runs: Path, out: Path) -> dict:
    """从production/p23/q4三路径清单比较网格、容差、终点和表格舍入。

    --runs接收JSON文件，或含rounding_comparison_inputs.json的目录。
    三个路径可以是绝对路径，也可以相对于该JSON文件所在目录。
    """
    manifest_path = runs / "rounding_comparison_inputs.json" if runs.is_dir() else runs
    manifest = read_json(manifest_path)
    directories = {
        name: (manifest_path.parent / manifest[name]).resolve()
        for name in ("production", "p23", "q4")
    }
    comparison = RefinementComparison(directories["production"])
    cases = [comparison.compare_case(case, directories[case]) for case in ("p23", "q4")]
    result = {
        "completed": True,
        "continuous_PDE_certified": False,
        "cases": cases,
        "input_manifest_sha256": sha256(manifest_path),
        "source_identity": source_identity(),
    }
    write_json(out / "global_refinement_comparison.json", result)
    write_csv(
        out / "global_refinement_roots.csv",
        [
            {
                "case": case["case"],
                "N": run["N"],
                "rtol": run["settings"]["rtol"],
                "root_s": run["equality_root_s"],
            }
            for case in cases
            for run in case["runs"]
        ],
        columns=("case", "N", "rtol", "root_s"),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("readback", "checks", "solve", "fem", "compare", "compare-1d", "compare-rounding"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="+", choices=("P1", "P23", "P4"), default=["P1", "P23", "P4"]
    )
    parser.add_argument("--elements", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--runs", type=Path)
    for name in ("fv-p1", "fv-p23", "fv-p4"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    out = prepare_output(args.out)
    if any(n < 1 for n in args.elements):
        raise ValueError("GLL单元数应为正整数")
    if args.mode == "compare-rounding" and args.runs is None:
        parser.error("compare-rounding需要--runs配置JSON或目录")
    handlers = {
        "readback": lambda: readback(out),
        "checks": lambda: (
            readback(out),
            independent_checks(out / "method_checks"),
            observer_checks(out / "observer_checks"),
        ),
        "solve": lambda: solve_all(out, args.cases, args.elements),
        "compare": lambda: compare_fv(out, [args.fv_p1, args.fv_p23, args.fv_p4]),
        "compare-rounding": lambda: compare_rounding(args.runs, out),
    }
    mode = {"fem": "solve", "compare-1d": "compare"}.get(args.mode, args.mode)
    run_cli_modes(mode, handlers)


if __name__ == "__main__":
    main()
