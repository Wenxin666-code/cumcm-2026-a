"""二维解析验证、制造解、物理计算与完整连续重构场比较。

空间模型、稀疏时间推进、热量审计及向外Bernstein算术直接复用
code/深度研究/二维模型.py。本文件实现实验定义、数据读取、比较和验收。
输出全部位于指定的包外目录；数据来源身份与当前执行源码分别记录。
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.integrate._ivp.bdf import BdfDenseOutput
from scipy.optimize import brentq
from scipy.special import j0, j1
from 验证公共接口 import (
    DATA,
    SUPPORT,
    research,
    science,
    read_json,
    load_npz,
    write_csv,
    run_cli_modes,
    write_json,
    sha256,
    prepare_output,
    source_identity,
    verify_data,
)
from 一维GLL验证 import CubicGLL, IndependentModel, read_inputs

backend = research("二维模型")
TensorFE = backend.TensorFE


def continuous_range(mesh, field, tolerance=1e-8):
    """Enclose a saved bicubic polynomial with the shared outward interval engine."""
    bounds = backend.PolynomialBounds(mesh)
    maximum = bounds.bound((field, field), tolerance=tolerance)
    minimum = bounds.bound((-field, -field), tolerance=tolerance)
    for row in (maximum, minimum):
        row["lower_decimal"] = str(Decimal.from_float(row["lower"]))
        row["upper_decimal"] = str(Decimal.from_float(row["upper"]))
    return {
        "maximum_lower": maximum["lower"],
        "maximum_upper": maximum["upper"],
        "minimum_lower": -minimum["upper"],
        "minimum_upper": -minimum["lower"],
        "maximum": maximum,
        "negative_minimum": minimum,
        "scope": "Exact saved bicubic field, distinct from continuous-PDE error.",
    }


def save_trajectory(out, mesh, times, amplitude, T0=300.0, Tscale=5.0, C0=1.0, Cscale=0.1):
    """Use the physical solver's coordinate and chunk format for test trajectories."""
    writer = backend.FieldWriter(out, mesh)
    for t, a in zip(times, amplitude):
        writer.add(float(t), np.r_[(T0 + Tscale * a).ravel(), (C0 + Cscale * a).ravel()])
    writer.finish()


def analytic_case(out, Er, Ez, family, gamma):
    mesh = TensorFE(Er, Ez)
    s = mesh.radial.x[:, None]
    z = mesh.axial.x[None, :]
    R0 = 0.02
    H = 0.125
    D = 1e-8
    stop = 5000.0
    hm = 8e-7
    if family == "fixed_robin":
        lr = brentq(lambda x: x * j1(x) - hm * R0 / D * j0(x), 1e-12, 2.40482555769, xtol=1e-14)
        lz = brentq(lambda x: x * np.tan(x) - hm * H / D, 1e-12, np.pi / 2 - 1e-12, xtol=1e-14)
        rate = 0.0
        gamma = 0.0
    else:
        lr = brentq(j1, 3.0, 4.0, xtol=1e-14)
        lz = np.pi
        hm = 0.0
        rate = np.log(0.02 / 0.012) / stop
    mesh.geometry = lambda t: (R0 * np.exp(-rate * t), np.exp(-rate * gamma * t))
    phi = j0(lr * np.sqrt(s)) * np.cos(lz * z)
    integral = lambda t, a: np.asarray(t) if a == 0 else np.expm1(2 * a * np.asarray(t)) / (2 * a)

    def exact(t):
        decay = np.exp(
            -D * (lr**2 / R0**2 * integral(t, rate) + lz**2 / H**2 * integral(t, rate * gamma))
        )
        return decay[..., None, None] * phi

    def rhs(t, u):
        R = R0 * np.exp(-rate * t)
        ell = np.exp(-rate * gamma * t)
        return (mesh.weak_rhs(u, D, R, H, ell, hm, 0.0) / mesh.mass).ravel()

    times = np.linspace(0, stop, 51)
    started = time.perf_counter()
    sol = solve_ivp(
        rhs,
        (0.0, stop),
        phi.ravel(),
        method="BDF",
        rtol=2e-11,
        atol=2e-13,
        max_step=25.0,
        t_eval=times,
        jac_sparsity=mesh.sparsity,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    numerical = sol.y.T.reshape(-1, *mesh.shape)
    error = numerical - exact(times)
    amplitude_error = float(np.max(abs(error)))
    label = f"{family}_gamma{gamma:g}_Er{Er}_Ez{Ez}"
    dest = out / label
    save_trajectory(dest, mesh, times, numerical)
    bounds = continuous_range(mesh, numerical[-1], tolerance=1e-8)
    # Test the two independent affine scalar reconstructions against the coupled
    # weak operator algebra. No absolute-temperature integrator change is hidden.
    T = 300 + 5 * phi
    C = 1 + 0.1 * phi
    heat = mesh.weak_rhs(T, D, R0, H, 1.0, hm, 300.0) / mesh.mass
    water = mesh.weak_rhs(C, D, R0, H, 1.0, hm, 1.0) / mesh.mass
    amp = rhs(0.0, phi.ravel()).reshape(mesh.shape)
    row = {
        "family": family,
        "gamma": gamma,
        "radial_elements": Er,
        "axial_elements": Ez,
        "radial_nodes": mesh.shape[0],
        "axial_nodes": mesh.shape[1],
        "total_nodes": mesh.size,
        "shared_output_times": len(times),
        "max_amplitude_error_all_times": amplitude_error,
        "max_T_error_K_all_times": 5 * amplitude_error,
        "max_C_error_all_times": 0.1 * amplitude_error,
        "final_mass_L2_amplitude_error": float(np.sqrt(np.sum(mesh.mass * error[-1] ** 2))),
        "max_reference_mass_drift": float(
            np.max(abs(np.sum(numerical * mesh.mass, axis=(1, 2)) - np.sum(phi * mesh.mass)))
        ),
        "affine_T_rhs_consistency_max": float(np.max(abs(heat - 5 * amp))),
        "affine_C_rhs_consistency_max": float(np.max(abs(water - 0.1 * amp))),
        "time_settings": {"method": "BDF", "rtol": 2e-11, "atol": 2e-13, "max_step_s": 25.0},
        "nfev": sol.nfev,
        "njev": sol.njev,
        "nlu": sol.nlu,
        "elapsed_seconds": time.perf_counter() - started,
        "continuous_final_amplitude_range": bounds,
        "field_index": f"{label}/field_index.json",
        "root_parameters": {
            "lambda_r": lr,
            "lambda_z": lz,
            "diffusivity": D,
            "R0": R0,
            "H0": H,
            "radial_exponential_rate": rate,
        },
        "scope": (
            "Actual linear analytic verification; all saved times and nodes checked. No "
            "nonlinear physical-model accuracy or general high-order maximum principle "
            "is established."
        ),
    }
    write_json(dest / "record.json", row)
    print(
        json.dumps(
            {
                "case": label,
                "max_amplitude_error": amplitude_error,
                "wall_s": row["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return row


def structural_checks(ambient, radius, tails, cfg, out):
    rows = []
    for Er, Ez in [(4, 2), (8, 4), (16, 8), (32, 16)]:
        m = TensorFE(Er, Ez)
        s = m.radial.x[:, None]
        z = m.axial.x[None, :]
        constant = m.stiffness(np.ones(m.shape), np.ones(m.shape))
        u = 1 + 0.1 * np.sin(2 * s) + 0.2 * np.cos(2 * z) + 0.1 * s * z
        k = 1 + 0.2 * s + 0.1 * z
        ks, kz = m.stiffness(u, k)
        R = 0.016
        ell = 0.8**0.5
        H = 0.125
        hm = 8e-7
        Ce = 0.05
        weak = m.weak_rhs(u, k * 1e-8, R, H, ell, hm, Ce)
        expected = -2 * hm / R * np.dot(m.axial.mass, u[-1] - Ce) - hm / (H * ell) * np.dot(
            m.radial.mass, u[:, -1] - Ce
        )
        rows.append(
            {
                "Er": Er,
                "Ez": Ez,
                "mass_sum": float(m.mass.sum()),
                "minimum_mass": float(m.mass.min()),
                "constant_stiffness_max": float(max(np.max(abs(v)) for v in constant)),
                "radial_internal_action_sum": float(ks.sum()),
                "axial_internal_action_sum": float(kz.sum()),
                "boundary_rate_balance_abs": float(abs(weak.sum() - expected)),
                "boundary_rate_balance_scaled": float(
                    abs(weak.sum() - expected) / max(abs(expected), 1e-30)
                ),
            }
        )
    # Nonlinear physical RHS, no end exchange and no axial variation, compared
    # directly with the previously verified independent 1D GLL weak method.
    m = TensorFE(8, 4)
    one = CubicGLL(8)
    comparisons = []
    if not np.array_equal(m.radial.x, one.s):
        raise RuntimeError("1D reduction mesh differs")
    for material in ["P1", "P23", "P4"]:
        moving = material == "P4"
        model = TensorFE(
            8,
            4,
            properties=cfg["property_sets"][material],
            ambient=ambient,
            radius=radius,
            tails=tails,
            moving=moving,
            gamma=1.0,
            end_factor=0.0,
        )
        baseline = IndependentModel(
            one, cfg["property_sets"][material], ambient, radius, tails, moving
        )
        T = 301 + 0.7 * np.cos(one.s)
        C = 1.5 + 0.1 * np.sin(one.s)
        y = np.r_[
            np.broadcast_to(T[:, None], m.shape).ravel(),
            np.broadcast_to(C[:, None], m.shape).ravel(),
        ]
        for t in [0.0, 1234.0, 14400.0, 200000.0]:
            right = t >= 14400
            actual = model.rhs(t, y, right).reshape(2, *m.shape)
            truth = baseline.rhs(t, np.r_[T, C], right).reshape(2, -1)
            diff = actual - truth[:, :, None]
            comparisons.append(
                {
                    "material": material,
                    "t_s": t,
                    "max_T_rhs_difference": float(np.max(abs(diff[0]))),
                    "max_C_rhs_difference": float(np.max(abs(diff[1]))),
                    "max_scaled_difference": float(
                        np.max(abs(diff) / (1 + abs(truth[:, :, None])))
                    ),
                }
            )
    # Transient no-exchange test with spatially and state-dependent diffusion,
    # radial and axial motion. Uniform dry-reference mass must be conserved.
    u0 = 1 + 0.08 * np.cos(2 * m.radial.x[:, None]) + 0.04 * np.sin(3 * m.axial.x[None, :])

    def no_exchange(t, v):
        u = v.reshape(m.shape)
        R = 0.02 * np.exp(-0.0001 * t)
        ell = (R / 0.02) ** 0.7
        D = 1e-8 * (1 + 0.2 * m.radial.x[:, None] + 0.1 * m.axial.x[None, :] + 0.05 * u)
        return (m.weak_rhs(u, D, R, 0.125, ell, 0.0, 0.0) / m.mass).ravel()

    times = np.linspace(0, 1000, 51)
    sol = solve_ivp(
        no_exchange,
        (0.0, 1000.0),
        u0.ravel(),
        method="BDF",
        rtol=2e-11,
        atol=2e-13,
        max_step=10.0,
        t_eval=times,
        jac_sparsity=m.sparsity,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    values = sol.y.T.reshape(-1, *m.shape)
    mass = np.sum(values * m.mass, axis=(1, 2))
    save_trajectory(out / "zero_exchange", m, times, values, T0=0, Tscale=1, C0=0, Cscale=1)
    conservation = {
        "max_reference_mass_drift": float(np.max(abs(mass - mass[0]))),
        "minimum_saved_state": float(values.min()),
        "nfev": sol.nfev,
        "nlu": sol.nlu,
        "scope": "State-dependent scalar diffusion with radial and axial motion; no source or exchange; no clipping.",
    }
    # An off-node peak must not be relabelled as the nodal maximum.
    test = TensorFE(1, 1, 1.0, 1.0)
    s = test.radial.x[:, None]
    z = test.axial.x[None, :]
    poly = 1 - (s - 0.37) ** 2 - 2 * (z - 0.61) ** 2
    extrema = continuous_range(test, poly, tolerance=1e-10)
    extrema.update(
        nodal_maximum=float(poly.max()),
        known_polynomial_maximum=1.0,
        note=(
            "Nodal float evaluation induces at most roundoff discrepancy from the "
            "intended quadratic; comparison allows 1e-12."
        ),
    )
    checks = {
        "positive_mass": all(
            r["minimum_mass"] > 0 and abs(r["mass_sum"] - 1) < 1e-14 for r in rows
        ),
        "constants_exactly_annihilated": all(r["constant_stiffness_max"] == 0 for r in rows),
        "internal_weak_fluxes_cancel": all(
            max(abs(r["radial_internal_action_sum"]), abs(r["axial_internal_action_sum"])) < 1e-10
            for r in rows
        ),
        "boundary_mean_rate_identity": all(r["boundary_rate_balance_scaled"] < 1e-10 for r in rows),
        "one_dimensional_physical_rhs_reduction": all(
            r["max_scaled_difference"] < 1e-10 for r in comparisons
        ),
        "zero_exchange_transient_reference_mass": conservation["max_reference_mass_drift"] < 1e-10,
        "continuous_peak_contains_true_maximum": extrema["maximum_lower"] <= 1 + 1e-12
        and extrema["maximum_upper"] >= 1 - 1e-12,
        "continuous_peak_bounds_converged": extrema["maximum"]["converged"]
        and extrema["negative_minimum"]["converged"],
        "float_bounds_are_outward": all(
            Decimal.from_float(b["lower"]) <= Decimal(b["lower_decimal"])
            and Decimal.from_float(b["upper"]) >= Decimal(b["upper_decimal"])
            for b in [extrema["maximum"], extrema["negative_minimum"]]
        ),
        "continuous_peak_exceeds_nodal_maximum": extrema["maximum_lower"]
        > extrema["nodal_maximum"] + 1e-3,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "algebra": rows,
        "one_dimensional_reduction": comparisons,
        "zero_exchange": conservation,
        "off_node_extremum_check": extrema,
    }
    write_json(out / "structural_checks.json", result)
    return result


def manufactured_case(out, material, gamma, Er, cfg, ambient, radius, tails):
    Ez = Er // 2
    m = TensorFE(Er, Ez)
    n = m.size
    s = m.radial.x[:, None]
    z = m.axial.x[None, :]
    phi = np.cos(np.pi * s) * np.cos(np.pi * z)
    phis = -np.pi * np.sin(np.pi * s) * np.cos(np.pi * z)
    phiz = -np.pi * np.cos(np.pi * s) * np.sin(np.pi * z)
    phiss = -np.pi**2 * phi
    phizz = phiss
    R0 = 0.02
    H = 0.125
    stop = 1000.0
    rate = np.log(0.02 / 0.018) / stop
    prop = cfg["property_sets"][material]
    model = TensorFE(
        Er, Ez, properties=prop, ambient=ambient, radius=radius, tails=tails, h=0.0, hm=0.0
    )
    model.geometry = lambda t: (R0 * np.exp(-rate * t), np.exp(-gamma * rate * t))

    def exact(t):
        a = np.exp(-t / 1000.0)
        return 310 + 0.5 * a * phi, 1.8 + 0.1 * a * phi

    def continuous_source(t):
        a = np.exp(-t / 1000.0)
        T, C = exact(t)
        Ts = 0.5 * a * phis
        Tz = 0.5 * a * phiz
        Tss = 0.5 * a * phiss
        Tzz = 0.5 * a * phizz
        Cs = 0.1 * a * phis
        Cz = 0.1 * a * phiz
        Css = 0.1 * a * phiss
        Czz = 0.1 * a * phizz
        Tt = -0.5 * a * phi / 1000.0
        Ct = -0.1 * a * phi / 1000.0
        b, k, D = model.coefficients(T, C)
        kC = prop["k_fraction_coefficient"] / (1 + C) ** 2
        DC = D * prop["D_moisture_exponent"] / C**2
        DT = D * prop["D_temperature_exponent_K"] / T**2
        ks = kC * Cs
        kz = kC * Cz
        Ds = DC * Cs + DT * Ts
        Dz = DC * Cz + DT * Tz
        R, ell = model.geometry(t)
        heat = 4 / R**2 * ((k + s * ks) * Ts + s * k * Tss) + (kz * Tz + k * Tzz) / (H * ell) ** 2
        water = 4 / R**2 * ((D + s * Ds) * Cs + s * D * Css) + (Dz * Cz + D * Czz) / (H * ell) ** 2
        return b * Tt - heat, Ct - water

    def rhs(t, y):
        T = y[:n].reshape(m.shape)
        C = y[n:].reshape(m.shape)
        b, k, D = model.coefficients(T, C)
        R, ell = model.geometry(t)
        fT, fC = continuous_source(t)
        heat = (m.weak_rhs(T, k, R, H, ell, 0.0, 0.0) / m.mass + fT) / b
        water = m.weak_rhs(C, D, R, H, ell, 0.0, 0.0) / m.mass + fC
        return np.r_[heat.ravel(), water.ravel()]

    T, C = exact(0.0)
    times = np.linspace(0, stop, 51)
    started = time.perf_counter()
    solution = solve_ivp(
        rhs,
        (0.0, stop),
        np.r_[T.ravel(), C.ravel()],
        method="BDF",
        rtol=2e-11,
        atol=np.r_[np.full(n, 2e-10), np.full(n, 2e-12)],
        max_step=10.0,
        t_eval=times,
        jac_sparsity=m.coupled_sparsity,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    values = solution.y.T.reshape(-1, 2, *m.shape)
    errors = []
    for i, t in enumerate(times):
        truth = exact(t)
        errors.append([np.max(abs(values[i, j] - truth[j])) for j in range(2)])
    errors = np.array(errors)
    dest = out / f"{material}_gamma{gamma:g}_Er{Er}_Ez{Ez}"
    dest.mkdir()
    np.savez_compressed(
        dest / "fields.npz",
        t_s=times,
        s=m.radial.x,
        zeta=m.axial.x,
        T_K=values[:, 0],
        C=values[:, 1],
    )
    report = {
        "material": material,
        "gamma": gamma,
        "Er": Er,
        "Ez": Ez,
        "nodes": n,
        "max_T_error_K": float(errors[:, 0].max()),
        "max_C_error": float(errors[:, 1].max()),
        "nfev": solution.nfev,
        "njev": solution.njev,
        "nlu": solution.nlu,
        "elapsed_seconds": time.perf_counter() - started,
        "fields_sha256": sha256(dest / "fields.npz"),
        "settings": {
            "method": "BDF",
            "rtol": 2e-11,
            "atol_T": 2e-10,
            "atol_C": 2e-12,
            "max_step_s": 10.0,
            "stop_s": 1000.0,
        },
    }
    write_json(dest / "record.json", report)
    print(json.dumps(report), flush=True)
    return report


def input_model():
    ambient, radius, tails = read_inputs(DATA / "inputs/附件")
    return ambient, radius, tails, read_json(DATA / "config/model.json")


def manufactured_suite(out, family="all"):
    """Execute all declared meshes and preserve per-case fields and error criteria."""
    ambient, radius, tails, cfg = input_model()
    reports = {}
    if family in ("all", "analytic"):
        dest = out / "analytic"
        dest.mkdir()
        structure = structural_checks(ambient, radius, tails, cfg, dest)
        rows = []
        groups = []
        for kind, gamma in [
            ("fixed_robin", 0.0),
            ("moving_neumann", 0.0),
            ("moving_neumann", 0.5),
            ("moving_neumann", 1.0),
        ]:
            group = []
            for er in (4, 8, 16, 32):
                row = analytic_case(dest, er, er // 2, kind, gamma)
                if group:
                    row["observed_order"] = float(
                        np.log2(
                            group[-1]["max_amplitude_error_all_times"]
                            / row["max_amplitude_error_all_times"]
                        )
                    )
                rows.append(row)
                group.append(row)
            groups.append(
                {
                    "family": kind,
                    "gamma": gamma,
                    "passed": group[-1]["max_amplitude_error_all_times"] < 1e-6
                    and group[-1]["max_amplitude_error_all_times"]
                    < group[0]["max_amplitude_error_all_times"] / 16,
                }
            )
        checks = {
            "structure": structure["passed"],
            "analytic_convergence": all(g["passed"] for g in groups),
            "moving_mass": all(
                r["max_reference_mass_drift"] < 1e-10
                for r in rows
                if r["family"] == "moving_neumann"
            ),
            "polynomial_ranges": all(
                r["continuous_final_amplitude_range"]["maximum"]["converged"]
                and r["continuous_final_amplitude_range"]["negative_minimum"]["converged"]
                for r in rows
            ),
        }
        reports["analytic"] = {
            "completed": True,
            "passed": all(checks.values()),
            "checks": checks,
            "groups": groups,
            "records": rows,
        }
        write_json(dest / "summary.json", reports["analytic"])
    if family in ("all", "nonlinear"):
        dest = out / "nonlinear"
        dest.mkdir()
        rows = []
        groups = []
        for material, gamma in [("P23", 0.0), ("P4", 0.0), ("P4", 0.5), ("P4", 1.0)]:
            group = []
            for er in (4, 8, 16, 32):
                row = manufactured_case(dest, material, gamma, er, cfg, ambient, radius, tails)
                if group:
                    row["observed_T_order"] = float(
                        np.log2(group[-1]["max_T_error_K"] / row["max_T_error_K"])
                    )
                    row["observed_C_order"] = float(
                        np.log2(group[-1]["max_C_error"] / row["max_C_error"])
                    )
                rows.append(row)
                group.append(row)
            groups.append(
                {
                    "material": material,
                    "gamma": gamma,
                    "passed": group[-1]["max_T_error_K"] < 2e-6
                    and group[-1]["max_C_error"] < 1e-7
                    and group[-1]["max_T_error_K"] < group[0]["max_T_error_K"] / 8
                    and group[-1]["max_C_error"] < group[0]["max_C_error"] / 8,
                }
            )
        reports["nonlinear"] = {
            "completed": True,
            "passed": all(g["passed"] for g in groups),
            "groups": groups,
            "records": rows,
            "exact_fields": "T=310+.5 exp(-t/1000)cos(pi*s)cos(pi*zeta); C=1.8+.1 exp(-t/1000)cos(pi*s)cos(pi*zeta)",
            "geometry": "R=.02 exp(-k*t), ell=exp(-gamma*k*t), k=log(.02/.018)/1000; H0=.125",
            "source": "Explicit derivatives of continuous formulas in manufactured_case; homogeneous Neumann boundaries.",
        }
        write_json(dest / "summary.json", reports["nonlinear"])
    result = {
        "completed": True,
        "passed": all(r["passed"] for r in reports.values()),
        "families": reports,
        "current_execution_identity": source_identity(extra_files=(Path(__file__),)),
    }
    write_json(out / "summary.json", result)
    if not result["passed"]:
        raise RuntimeError("制造解验收未通过")
    return result


def readback(out):
    integrity = verify_data()
    records = []
    origin = read_json(DATA / "来源源码索引.json")["files"]
    for name in ("analytic_mms", "nonlinear_mms"):
        summary = read_json(DATA / "evidence/numerics" / name / "summary.json")
        if not summary["passed"] or len(summary["records"]) != 16:
            raise ValueError("保存制造解摘要不完整")
        provenance = []
        for relative, digest in summary["source_sha256"].items():
            if relative.endswith(".py"):
                expected = origin[relative]["sha256"]
            else:
                expected = sha256(DATA / relative)
            if expected != digest:
                raise ValueError("保存证据来源身份不符: " + relative)
            provenance.append(
                {
                    "path": relative,
                    "recorded_sha256": digest,
                    "kind": "origin_source" if relative.endswith(".py") else "data",
                }
            )
        records.append(
            {
                "family": name,
                "records": summary["records"],
                "provenance": provenance,
                "passed": True,
            }
        )
    report = {
        "passed": True,
        "data_integrity": integrity,
        "families": records,
        "current_execution_identity": source_identity(extra_files=(Path(__file__),)),
        "scope": "保存制造解证据与其来源身份读回。当前执行源码单独记录；mms模式实际计算全部制造解。",
    }
    write_json(out / "readback.json", report)
    return report


def bilinear(coarse, radial, axial, fine_radial, fine_axial):
    """Evaluate the complete coarse bilinear reconstruction on all fine corners."""
    if np.any(np.diff(radial) <= 0) or np.any(np.diff(axial) <= 0):
        raise ValueError("网格坐标须严格递增")
    if (
        fine_radial[0] < radial[0]
        or fine_radial[-1] > radial[-1]
        or fine_axial[0] < axial[0]
        or fine_axial[-1] > axial[-1]
    ):
        raise ValueError("比较域超出原场")
    ir = np.clip(np.searchsorted(radial, fine_radial, side="right") - 1, 0, len(radial) - 2)
    iz = np.clip(np.searchsorted(axial, fine_axial, side="right") - 1, 0, len(axial) - 2)
    wr = (fine_radial - radial[ir]) / (radial[ir + 1] - radial[ir])
    wz = (fine_axial - axial[iz]) / (axial[iz + 1] - axial[iz])
    along = (1 - wr[:, None]) * coarse[ir, :] + wr[:, None] * coarse[ir + 1, :]
    return (1 - wz[None, :]) * along[:, iz] + wz[None, :] * along[:, iz + 1]


class FieldSeries:
    """负责全场索引、身份和逐块读取；compare_fields负责重构与差异判据。"""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.index = read_json(self.directory / "field_index.json")
        if not self.index.get("completed"):
            raise ValueError("全场索引未完成")
        z = load_npz(self.directory / "coordinates.npz")
        self.coordinates = z
        self.kind = "fe" if "s" in self.coordinates else "fv"
        expected_shape = (
            len(self.coordinates["s" if self.kind == "fe" else "xi"]),
            len(self.coordinates["zeta"]),
        )
        self.times = []
        self.chunks = []
        for row in self.index["chunks"]:
            path = (self.directory / row["file"]).resolve()
            if not path.is_relative_to(self.directory) or not path.is_file():
                raise ValueError("场分块路径越界或缺失")
            if sha256(path) != row["sha256"]:
                raise ValueError("场分块身份不符")
            z = load_npz(path)
            t = z["t_s"]
            T = z["T_K"]
            C = z["C"]
            if (
                len(t) != row["rows"]
                or T.shape != C.shape
                or T.shape[0] != len(t)
                or T.shape[1:] != expected_shape
                or not all(np.isfinite(x).all() for x in (t, T, C))
            ):
                raise ValueError("场分块形状或有限性错误")
            self.times.extend(t.tolist())
            self.chunks.append((path, row["sha256"]))
        self.times = np.asarray(self.times)
        if len(self.times) != self.index["rows"] or not np.all(np.diff(self.times) > 0):
            raise ValueError("全场时间覆盖错误")
        self.summary = (
            read_json(self.directory / "summary.json")
            if (self.directory / "summary.json").is_file()
            else {}
        )
        self.identity = {
            "field_index": sha256(self.directory / "field_index.json"),
            "coordinates": sha256(self.directory / "coordinates.npz"),
            "chunks": {p.relative_to(self.directory).as_posix(): h for p, h in self.chunks},
        }
        if self.summary:
            self.identity["summary"] = sha256(self.directory / "summary.json")

    def rows(self, wanted):
        for path, digest in self.chunks:
            if sha256(path) != digest:
                raise ValueError("比较过程中分块改变")
            z = load_npz(path)
            for i, t in enumerate(z["t_s"]):
                if float(t) in wanted:
                    geometry = {
                        key: float(z[key][i]) for key in ("R_m", "ell", "half_length_m") if key in z
                    }
                    yield float(t), z["T_K"][i], z["C"][i], geometry


def mesh_from_coordinates(coordinates):
    mesh = TensorFE(len(coordinates["radial_edges"]) - 1, len(coordinates["axial_edges"]) - 1)
    actual = mesh.coordinates()
    for key in (
        "s",
        "zeta",
        "radial_edges",
        "axial_edges",
        "local_gll_nodes",
        "radial_connectivity",
        "axial_connectivity",
    ):
        if key not in coordinates or not np.array_equal(coordinates[key], actual[key]):
            raise ValueError("完整FE场网格与指定三次单元不符: " + key)
    return mesh


def refined_difference(bounds, coarse, fine, gap):
    result = bounds.difference(coarse, fine)
    if gap is None:
        return result
    lo, hi = bounds.difference_coefficients(coarse, fine)
    lower_plus = float(lo[:, :, [0, 3], :][:, :, :, [0, 3]].max())
    lower_minus = float((-hi)[:, :, [0, 3], :][:, :, :, [0, 3]].max())
    positive = backend.refine_maximum(lo, hi, lower_plus, tolerance=gap)
    negative = backend.refine_maximum(-hi, -lo, lower_minus, tolerance=gap)
    result.update(
        continuous_absolute_difference_lower=max(0.0, positive["lower"], negative["lower"]),
        continuous_absolute_difference_upper=max(0.0, positive["upper"], negative["upper"]),
        positive=positive,
        negative=negative,
        refinement_converged=positive["converged"] and negative["converged"],
    )
    return result


def compare_fields(out, coarse, fine, times=None, gap=None):
    """Compare every declared common time over the full continuous spatial reconstruction."""
    a, b = FieldSeries(coarse), FieldSeries(fine)
    if a.kind != b.kind:
        raise ValueError("本入口比较同类空间重构；一维与二维模型对照需另设一致问题。")
    for key in ("material", "gamma", "axial_power", "moving", "end_factor"):
        left = a.summary.get("settings", a.summary).get(key)
        right = b.summary.get("settings", b.summary).get(key)
        if left is not None and right is not None and left != right:
            raise ValueError("比较场的物理条件不同: " + key)
    for key in ("input_sha256", "refactored_source_sha256"):
        if key in a.summary and key in b.summary and a.summary[key] != b.summary[key]:
            raise ValueError("比较场的输入或物理算子身份不同: " + key)
    common = np.intersect1d(a.times, b.times)
    if times is not None:
        requested = np.asarray(times, dtype=float)
        if not np.isin(requested, common).all():
            raise ValueError("指定时刻须为双方准确保存的共同时间")
        common = np.unique(requested)
    if len(common) == 0:
        raise ValueError("没有共同保存时刻")
    if a.kind == "fe":
        bounds = backend.PolynomialBounds(
            mesh_from_coordinates(b.coordinates), coarse=mesh_from_coordinates(a.coordinates)
        )
    else:
        ar, az = a.coordinates["xi"], a.coordinates["zeta"]
        br, bz = b.coordinates["xi"], b.coordinates["zeta"]
        # On nested grids the difference is bilinear on each fine rectangle.
        if max(min(abs(br - v)) for v in ar) > 1e-14 or max(min(abs(bz - v)) for v in az) > 1e-14:
            raise ValueError("完整FV场差要求嵌套网格")
        ci = np.argmin(abs(br[:, None] - ar[None, :]), axis=0)
        cj = np.argmin(abs(bz[:, None] - az[None, :]), axis=0)
    rows = []
    maxima = {k: {"lower": -1.0, "upper": -1.0} for k in ("T_K", "C")}
    for left, right in zip(a.rows(set(common)), b.rows(set(common))):
        if left[0] != right[0]:
            raise ValueError("共同物理时刻错配")
        if set(left[3]) != set(right[3]) or any(
            abs(left[3][k] - right[3][k]) > 1e-13 * max(1, abs(left[3][k])) for k in left[3]
        ):
            raise ValueError("同一时刻几何不同")
        if not left[3]:
            raise ValueError("物理完整场比较需要保存每行几何；解析实验请采用其独立验收。")
        row = {"t_s": left[0]}
        for k, c, f in zip(("T_K", "C"), left[1:3], right[1:3]):
            if a.kind == "fe":
                r = refined_difference(bounds, c, f, gap)
                lower = r["continuous_absolute_difference_lower"]
                upper = r["continuous_absolute_difference_upper"]
            else:
                reconstructed = bilinear(c, ar, az, br, bz)
                delta = abs(reconstructed - f)
                ix = np.unravel_index(np.argmax(delta), delta.shape)
                lower = upper = float(delta[ix])
                r = {
                    "fine_corner_maximum": upper,
                    "xi": float(br[ix[0]]),
                    "zeta": float(bz[ix[1]]),
                    "shared_coarse_node_difference": float(abs(c - f[np.ix_(ci, cj)]).max()),
                }
            row[k] = r
            if upper > maxima[k]["upper"]:
                maxima[k].update(upper=upper, upper_time_s=left[0])
            if lower > maxima[k]["lower"]:
                maxima[k].update(lower=lower, lower_time_s=left[0])
        rows.append(row)
    if len(rows) != len(common):
        raise ValueError("比较遗漏了共同时间")
    endpoint = None
    if a.summary.get("event") and b.summary.get("event"):
        ce, fe = a.summary["event"], b.summary["event"]
        lower = Fraction.from_float(fe["root_lower_s"]) - Fraction.from_float(ce["root_upper_s"])
        upper = Fraction.from_float(fe["root_upper_s"]) - Fraction.from_float(ce["root_lower_s"])
        endpoint = {
            "fine_minus_coarse_computed_root_difference_interval_s": [
                backend.fbound(lower)[0],
                backend.fbound(upper)[1],
            ],
            "coarse_root_interval_s": [ce["root_lower_s"], ce["root_upper_s"]],
            "fine_root_interval_s": [fe["root_lower_s"], fe["root_upper_s"]],
            "coarse_report_time_s": ce.get("report_time_s"),
            "fine_report_time_s": fe.get("report_time_s"),
        }
    elif a.summary.get("event_time_s") is not None and b.summary.get("event_time_s") is not None:
        endpoint = {
            "fine_minus_coarse_event_s": b.summary["event_time_s"] - a.summary["event_time_s"]
        }
    report = {
        "completed": True,
        "engine": a.kind,
        "common_time_rows": len(rows),
        "maxima": maxima,
        "rows": rows,
        "unmatched_coarse_times_s": np.setdiff1d(a.times, b.times).tolist(),
        "unmatched_fine_times_s": np.setdiff1d(b.times, a.times).tolist(),
        "coarse_event": a.summary.get("event"),
        "fine_event": b.summary.get("event"),
        "endpoint_comparison": endpoint,
        "source_fields": {"coarse": a.identity, "fine": b.identity},
        "current_execution_identity": source_identity(extra_files=(Path(__file__),)),
        "scope": "全部选定共同时间的完整空间重构差；FE为向外区间，FV为binary64双线性比较。时间间隔与连续PDE误差分别判断。",
    }
    write_json(out / "comparison.json", report)
    history = []
    for row in rows:
        record = {"t_s": row["t_s"]}
        for field, prefix in (("T_K", "T"), ("C", "C")):
            for side in ("lower", "upper"):
                record[f"{prefix}_{side}"] = row[field].get(
                    f"continuous_absolute_difference_{side}", row[field].get("fine_corner_maximum")
                )
        history.append(record)
    write_csv(out / "differences.csv", history, encoding="utf8")
    return report


class RadialBaseline:
    """读取Q1完整节点剖面或P23/Q4的internal未舍入场及metadata.json。"""

    def __init__(self, directory):
        directory = Path(directory).resolve()
        metadata = directory / "metadata.json"
        self.metadata = read_json(metadata)
        native_q1 = self.metadata.get("problem") == "q1"
        question = "q1" if native_q1 else self.metadata.get("question")
        materials = {"q1": "P1", "p23": "P23", "q4": "P4"}
        if question not in materials:
            raise ValueError("一维基准metadata.question须为q1、p23或q4")
        self.material = materials[question]
        hashes = self.metadata.get("input_sha256", {})
        aliases = (
            {"model.json": "model_contract", "附件1.xlsx": "附件1.xlsx"}
            if native_q1
            else {
                "model.json": "model_config",
                "附件1.xlsx": "attachment1",
                "附件2.xlsx": "attachment2",
            }
        )
        if set(hashes) != set(aliases.values()):
            raise ValueError("一维基准缺少对应模型实际使用的配置/附件来源哈希")
        self.input_hashes = {name: hashes[alias] for name, alias in aliases.items()}
        fields = directory / ("grid_profiles.npz" if native_q1 else "internal.npz")
        data = load_npz(fields)
        required = ("t_s", "r_m", "T_K", "C") if native_q1 else ("t_s", "xi", "R_m", "T_K", "C")
        if any(name not in data for name in required):
            raise ValueError("一维未舍入场缺少相应时间、半径或温湿度字段")
        self.data = {name: data[name].copy() for name in required}
        if native_q1:
            radial = self.data.pop("r_m")
            if (
                self.metadata.get("property_set") != "P1"
                or radial.ndim != 1
                or len(radial) < 2
                or len(radial) != self.metadata.get("grid_nodes")
                or radial[-1] <= 0
            ):
                raise ValueError("Q1需与metadata一致的完整径向节点剖面")
            self.data["xi"] = radial / radial[-1]
            self.data["R_m"] = np.full(len(self.data["t_s"]), radial[-1])
        required = ("t_s", "xi", "R_m", "T_K", "C")
        t, x, radius, temperature, moisture = (self.data[name] for name in required)
        if (
            t.ndim != 1
            or x.ndim != 1
            or len(t) == 0
            or len(x) < 2
            or not all(np.isfinite(value).all() for value in self.data.values())
            or not np.all(np.diff(t) > 0)
            or not np.all(np.diff(x) > 0)
            or x[0] != 0
            or x[-1] != 1
            or radius.shape != t.shape
            or not np.all(radius > 0)
            or temperature.shape != (len(t), len(x))
            or moisture.shape != temperature.shape
        ):
            raise ValueError("一维基准时间、归一化半径、字段维度或有限性错误")
        self.identity = {fields.name: sha256(fields), "metadata.json": sha256(metadata)}


def compare_baseline(out, full_field, baseline, times=None, centre_tolerance=1e-10):
    """共同保存时刻的二维减一维有符号差、极值位置和中心最大值诊断。"""
    if not np.isfinite(centre_tolerance) or centre_tolerance <= 0:
        raise ValueError("中心最大值诊断容差须为正有限数")
    two, one = FieldSeries(full_field), RadialBaseline(baseline)
    settings = two.summary.get("settings", {})
    material = two.summary.get("material", settings.get("material"))
    if material != one.material:
        raise ValueError("一维和二维材料不同")
    two_hashes = two.summary.get("input_sha256", {})
    if any(two_hashes.get(name) != digest for name, digest in one.input_hashes.items()):
        raise ValueError("一维和二维配置、环境或半径输入来源不同")
    gamma = two.summary.get("axial_power", settings.get("gamma"))
    if gamma != 0:
        raise ValueError("径向一维基准对应固定长度；本对照要求二维axial_power/gamma为0")
    coordinates = two.coordinates
    radial = coordinates["xi"] if two.kind == "fv" else np.sqrt(coordinates["s"])
    axial = coordinates["zeta"]
    if (
        radial.ndim != 1
        or axial.ndim != 1
        or not np.all(np.diff(radial) > 0)
        or not np.all(np.diff(axial) > 0)
        or radial[0] != 0
        or radial[-1] != 1
        or axial[0] != 0
        or axial[-1] != 1
    ):
        raise ValueError("二维坐标须覆盖完整径向和中面至端面，并严格递增")
    common = np.intersect1d(two.times, one.data["t_s"])
    if times is not None:
        requested = np.asarray(times, dtype=float)
        if not np.isfinite(requested).all() or not np.isin(requested, common).all():
            raise ValueError("指定时刻须在双方原始未舍入场中准确保存")
        common = np.unique(requested)
    if len(common) == 0:
        raise ValueError("一维与二维未舍入场没有共同保存时刻")
    mesh = mesh_from_coordinates(coordinates) if two.kind == "fe" else None
    sample_radius = np.union1d(radial, one.data["xi"]) if two.kind == "fv" else radial
    rows, maximum_positive, minimum_negative = [], None, None
    for t, temperature, moisture, geometry in two.rows(set(common)):
        j = int(np.searchsorted(one.data["t_s"], t))
        if one.data["t_s"][j] != t:
            raise ValueError("一维和二维保存时刻错配")
        if any(name not in geometry for name in ("R_m", "ell", "half_length_m")):
            raise ValueError("二维场缺少逐时刻半径、轴向伸缩和半长度")
        radius, half_length = geometry["R_m"], geometry["half_length_m"]
        radius_difference = abs(radius - float(one.data["R_m"][j]))
        if (
            not all(np.isfinite(value) for value in geometry.values())
            or radius <= 0
            or half_length <= 0
            or geometry["ell"] != 1
            or radius_difference > 64 * np.finfo(float).eps * max(radius, one.data["R_m"][j])
        ):
            raise ValueError("共同时间的一维和二维实际几何不同")
        T1 = np.interp(sample_radius, one.data["xi"], one.data["T_K"][j])
        C1 = np.interp(sample_radius, one.data["xi"], one.data["C"][j])
        T2 = (
            bilinear(temperature, radial, axial, sample_radius, axial)
            if two.kind == "fv"
            else temperature
        )
        C2 = (
            bilinear(moisture, radial, axial, sample_radius, axial)
            if two.kind == "fv"
            else moisture
        )
        difference = C2 - C1[:, None]
        positive = np.unravel_index(np.argmax(difference), difference.shape)
        negative = np.unravel_index(np.argmin(difference), difference.shape)

        def location(index):
            r, z = index
            return {
                "t_s": t,
                "xi": float(sample_radius[r]),
                "zeta": float(axial[z]),
                "physical_r_m": float(sample_radius[r] * radius),
                "z_from_midplane_m": float(axial[z] * half_length),
                "distance_from_end_m": float((1 - axial[z]) * half_length),
                "C_1D": float(C1[r]),
                "C_2D": float(C2[r, z]),
                "T_1D_K": float(T1[r]),
                "T_2D_K": float(T2[r, z]),
                "signed_C2D_minus_C1D": float(difference[r, z]),
            }

        if two.kind == "fe":
            enclosure = backend.PolynomialBounds(mesh).bound(
                (moisture, moisture), tolerance=centre_tolerance / 4
            )
            lower, upper = enclosure["lower"], enclosure["upper"]
        else:
            lower = upper = float(moisture.max())
        centre = float(moisture[0, 0])
        gap_lower = max(Fraction(0), Fraction.from_float(lower) - Fraction.from_float(centre))
        gap_upper = max(Fraction(0), Fraction.from_float(upper) - Fraction.from_float(centre))
        exact_tolerance = Fraction.from_float(centre_tolerance)
        centred = (
            True
            if gap_upper <= exact_tolerance
            else (False if gap_lower > exact_tolerance else None)
        )
        row = {
            "t_s": t,
            "maximum_signed_C2D_minus_C1D": float(difference.max()),
            "minimum_signed_C2D_minus_C1D": float(difference.min()),
            "maximum_positive_location": location(positive),
            "minimum_negative_location": location(negative),
            "maximum_C_2D_interval": [lower, upper],
            "geometric_centre_C": centre,
            "maximum_minus_centre_interval": [
                backend.fbound(gap_lower)[0],
                backend.fbound(gap_upper)[1],
            ],
            "centre_contains_maximum_within_tolerance": centred,
            "actual_radius_difference_m": radius_difference,
            "comparison_radial_positions": len(sample_radius),
            "axial_positions": len(axial),
        }
        rows.append(row)
        if (
            maximum_positive is None
            or row["maximum_signed_C2D_minus_C1D"] > maximum_positive["signed_C2D_minus_C1D"]
        ):
            maximum_positive = row["maximum_positive_location"]
        if (
            minimum_negative is None
            or row["minimum_signed_C2D_minus_C1D"] < minimum_negative["signed_C2D_minus_C1D"]
        ):
            minimum_negative = row["minimum_negative_location"]
    if len(rows) != len(common):
        raise ValueError("二维-一维对照遗漏共同保存时刻")
    report = {
        "completed": True,
        "engine": two.kind,
        "material": material,
        "common_time_rows": len(rows),
        "rows": rows,
        "maximum_signed_difference_location": maximum_positive,
        "minimum_signed_difference_location": minimum_negative,
        "unmatched_2D_times_s": np.setdiff1d(two.times, one.data["t_s"]).tolist(),
        "unmatched_1D_times_s": np.setdiff1d(one.data["t_s"], two.times).tolist(),
        "input_hashes_identical": one.input_hashes,
        "time_interpolation_used": False,
        "centre_tolerance": centre_tolerance,
        "signed_difference_scope": (
            "保存时刻双线性二维场与分段线性一维场之差，径向节点并集及轴向节点覆盖完整空间极值；binary64诊断。"
            if two.kind == "fv"
            else "保存时刻二维GLL节点与一维分段线性场的有符号差；不是节点间有符号差界。"
        ),
        "maximum_scope": (
            "保存的双线性场最大值位于网格角点。"
            if two.kind == "fv"
            else "保存的分片双三次场最大值采用向外Bernstein界；中心判定为null表示容差附近尚未判定。"
        ),
        "continuous_time_or_continuum_ordering_proved": False,
        "source_fields": {"two_dimensional": two.identity, "one_dimensional": one.identity},
        "current_execution_identity": source_identity(),
    }
    write_json(out / "baseline_comparison.json", report)
    return report


def algebra_checks(out):
    """Check off-node extrema, exact polynomial restriction and FV interior peaks."""
    coarse = TensorFE(2, 1)
    fine = TensorFE(4, 2)
    c = 1 - (coarse.radial.x[:, None] - 0.37) ** 2 - 2 * (coarse.axial.x[None, :] - 0.61) ** 2
    peak = continuous_range(coarse, c, 1e-10)
    f = 1 - (fine.radial.x[:, None] - 0.37) ** 2 - 2 * (fine.axial.x[None, :] - 0.61) ** 2
    bounds = backend.PolynomialBounds(fine, coarse=coarse)
    identical = bounds.difference(c, f)
    target = f + 1e-3 * (fine.radial.x[:, None] * (1 - fine.radial.x[:, None])) * (
        fine.axial.x[None, :] * (1 - fine.axial.x[None, :])
    )
    enclosure = refined_difference(bounds, c, target, 1e-10)
    r = np.array([0.0, 0.5, 1.0])
    z = np.array([0.0, 0.5, 1.0])
    fr = np.linspace(0, 1, 9)
    fz = np.linspace(0, 1, 7)
    u = 1 + r[:, None] + 2 * z[None, :] + 3 * r[:, None] * z[None, :]
    expected = 1 + fr[:, None] + 2 * fz[None, :] + 3 * fr[:, None] * fz[None, :]
    interpolation = float(abs(bilinear(u, r, z, fr, fz) - expected).max())
    rng = np.random.default_rng(314159)
    excess = 0.0
    for _ in range(500):
        corners = rng.normal(size=(2, 2))
        x, y = rng.random(2)
        value = (
            (1 - x) * (1 - y) * corners[0, 0]
            + x * (1 - y) * corners[1, 0]
            + (1 - x) * y * corners[0, 1]
            + x * y * corners[1, 1]
        )
        excess = max(excess, float(abs(value) - abs(corners).max()))
    checks = {
        "off_node_peak": peak["maximum_lower"] <= 1 + 1e-12
        and peak["maximum_upper"] >= 1 - 1e-12
        and peak["maximum_lower"] > float(c.max()) + 1e-3,
        "same_polynomial_restriction": identical["continuous_absolute_difference_upper"] < 1e-11,
        "known_difference_enclosed": enclosure["continuous_absolute_difference_lower"]
        <= 6.25e-5
        <= enclosure["continuous_absolute_difference_upper"],
        "bilinear_exactness": interpolation < 2e-14,
        "convex_corner_bound": excess == 0.0,
    }
    paths = []
    for name, times, root in [("coarse", [0.0, 1.0, 2.0], 10.0), ("fine", [0.0, 1.0, 3.0], 11.0)]:
        directory = out / "alignment" / name
        directory.mkdir(parents=True)
        writer = backend.FieldWriter(directory, coarse)
        for t in times:
            writer.add(t, np.r_[np.full(coarse.size, 300 + t), np.full(coarse.size, 1 - 0.01 * t)])
        writer.finish()
        write_json(
            directory / "summary.json",
            {
                "completed": True,
                "settings": {"material": "synthetic", "gamma": 0.0},
                "event": {
                    "root_lower_s": root,
                    "root_upper_s": root + 0.00005,
                    "report_time_s": root + 0.08,
                },
            },
        )
        paths.append(directory)
    alignment = compare_fields(out / "alignment/comparison", *paths)
    root_difference = alignment["endpoint_comparison"][
        "fine_minus_coarse_computed_root_difference_interval_s"
    ]
    checks["exact_common_time_alignment"] = (
        alignment["common_time_rows"] == 2
        and alignment["unmatched_coarse_times_s"] == [2.0]
        and alignment["unmatched_fine_times_s"] == [3.0]
        and all(r["upper"] < 1e-11 for r in alignment["maxima"].values())
        and root_difference[0] <= 1 <= root_difference[1]
    )
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "off_node_peak": peak,
        "same_polynomial": identical,
        "known_difference": enclosure,
        "bilinear_error": interpolation,
        "max_corner_excess": excess,
        "alignment_comparison": alignment,
        "current_execution_identity": source_identity(extra_files=(Path(__file__),)),
    }
    write_json(out / "algebra_checks.json", result)
    if not result["passed"]:
        raise RuntimeError("多项式代数检查未通过")
    return result


def audit_checks(out):
    """已知空间/时间多项式独立核对守恒求积、BDF导数和离节点事件。"""
    mesh = backend.build_model(
        "fe", DATA / "inputs/附件", DATA / "config/model.json", "P1", elements=4
    )
    s, z = mesh.radial.x[:, None], mesh.axial.x[None, :]
    T, C = 300 + s * s + z**3, 1 + 0.1 * s + 0.2 * z * z
    Td, Cd = 0.2 + s * s - z * z, 0.01 * (1 + s * z)
    spatial = backend.HeatAudit(mesh, j_heat=False).audit(
        0.0, np.r_[T.ravel(), C.ravel()], np.r_[Td.ravel(), Cd.ravel()]
    )
    Ta, Ce = mesh.environment(0.0)
    capacity = 820 * 2600
    exact = {
        "mean_C": 1 + 0.1 / 2 + 0.2 / 3,
        "mean_T_K": 300 + 1 / 3 + 1 / 4,
        "volume_moisture_rate": 0.0125,
        "volume_heat_rate": 0.2 * capacity,
        "side_moisture_rate": 2 * mesh.hm / 0.02 * (1.1 + 0.2 / 3 - Ce),
        "end_moisture_rate": mesh.hm / 0.125 * (1.2 + 0.1 / 2 - Ce),
        "side_heat_rate": 2 * mesh.h / 0.02 * (301.25 - Ta),
        "end_heat_rate": mesh.h / 0.125 * (301 + 1 / 3 - Ta),
    }
    errors = {
        name: {key: abs(spatial[name][key] - value) for key, value in exact.items()}
        for name in ("gauss8", "gauss16")
    }
    space_pass = spatial["valid"] and all(
        row[key] < 1e-7 * max(1, abs(exact[key])) for row in errors.values() for key in exact
    )
    n = mesh.size
    s, z = np.broadcast_to(s, mesh.shape), np.broadcast_to(z, mesh.shape)
    data = np.stack(
        [
            np.r_[(310 + 0.3 * s + 0.1 * z).ravel(), (1.8 + 0.2 * s - 0.05 * z).ravel()],
            np.r_[(0.5 + 0.01 * s).ravel(), (-0.02 + 0.001 * z).ravel()],
            np.r_[(0.07 + 0.003 * z).ravel(), (0.002 + 0.0001 * s).ravel()],
            np.r_[(0.002 + 0.0001 * s).ravel(), (-0.0001 + 0.00001 * z).ravel()],
        ]
    )
    dense = BdfDenseOutput(0.0, 60.0, 60.0, 3, data)
    bases = backend.polynomial_bases(dense, 0.0, 60.0)
    power = np.zeros((4, 2 * n))
    for k, basis in enumerate(bases):
        for j, value in enumerate(basis):
            power[j] += float(value) * data[k]

    def values(t):
        return np.polynomial.polynomial.polyval(t / 60.0, power)

    def derivative(t):
        return (
            np.polynomial.polynomial.polyval(t / 60.0, np.arange(1, 4)[:, None] * power[1:]) / 60.0
        )

    temporal = []
    for material, gamma in [("P1", 0.0), ("P4", 0.5)]:
        model = backend.build_model(
            "fe",
            DATA / "inputs/附件",
            DATA / "config/model.json",
            material,
            elements=4,
            axial_power=gamma,
        )
        audit = backend.HeatAudit(model, j_heat=False)
        audit.integrate(dense, 0.0, 60.0, False)

        def component(t, j):
            y, dy = values(t), derivative(t)
            temp, moisture = y[:n].reshape(model.shape), y[n:].reshape(model.shape)
            R, ell = model.geometry(t)
            Ta, Ce = model.environment(t)
            if j == 0:
                return 2 * model.hm / R * np.sum(model.axial.mass * (moisture[-1] - Ce))
            if j == 1:
                return (
                    model.hm / (model.H * ell) * np.sum(model.radial.mass * (moisture[:, -1] - Ce))
                )
            if j == 2:
                return 2 * model.h / R * np.sum(model.axial.mass * (temp[-1] - Ta))
            if j == 3:
                return model.h / (model.H * ell) * np.sum(model.radial.mass * (temp[:, -1] - Ta))
            if j == 4:
                return np.sum(model.mass * dy[n:].reshape(model.shape))
            properties = model.p
            rho = properties["rho_offset"] + properties["rho_C_coefficient"] * moisture
            cp = properties["cp_offset"] + properties["cp_fraction_coefficient"] * moisture / (
                1 + moisture
            )
            return np.sum(model.mass * rho * cp * dy[:n].reshape(model.shape))

        reference = np.array(
            [
                quad(lambda t: component(t, j), 0.0, 60.0, epsabs=1e-9, epsrel=2e-13)[0]
                for j in range(6)
            ]
        )
        error = abs(audit.gauss[16] - reference) / np.maximum(1, abs(reference))
        y0, yf = values(0.0), values(60.0)
        b0 = model.coefficients(y0[:n].reshape(model.shape), y0[n:].reshape(model.shape))[0]
        bf = model.coefficients(yf[:n].reshape(model.shape), yf[n:].reshape(model.shape))[0]
        wrong = float(
            np.sum(
                model.mass * (bf * yf[:n].reshape(model.shape) - b0 * y0[:n].reshape(model.shape))
            )
        )
        row = {
            "material": material,
            "maximum_scaled_difference": float(error.max()),
            "reference_integrals": reference.tolist(),
            "gauss16": audit.gauss[16].tolist(),
            "wrong_delta_bT": wrong,
            "correct_integral_b_Tdot": float(reference[5]),
            "passed": bool(error.max() < 2e-12),
        }
        if material == "P1":
            exactC = float(np.sum(model.mass * (yf[n:] - y0[n:]).reshape(model.shape)))
            exactT = float(np.sum(model.mass * b0 * (yf[:n] - y0[:n]).reshape(model.shape)))
            row["constant_b_volume_errors"] = [
                float(audit.gauss[16][4] - exactC),
                float(audit.gauss[16][5] - exactT),
            ]
            row["passed"] &= (
                abs(row["constant_b_volume_errors"][0]) < 1e-13
                and abs(row["constant_b_volume_errors"][1]) < 1e-6
            )
        else:
            row["passed"] &= abs(wrong - reference[5]) > 1e3
        row["passed"] = bool(row["passed"])
        temporal.append(row)

    event_mesh = TensorFE(2, 2, radial_power=1, axial_power=1)
    n = event_mesh.size
    s, z = event_mesh.radial.x[:, None], event_mesh.axial.x[None, :]
    c11 = 0.149 - 0.0001 * ((s - 0.37) ** 2 + (z - 0.61) ** 2)
    data = np.stack([np.r_[np.full(n, 300.0), c11.ravel()], np.r_[np.zeros(n), np.full(n, -0.002)]])
    crossing = BdfDenseOutput(9.0, 11.0, 2.0, 1, data)
    bracket = backend.bracket_event(crossing, backend.PolynomialBounds(event_mesh), n, width=1e-4)
    witness = backend.scalar_time_bound(crossing, n + int(np.argmax(c11)), 9.0, 9.5)
    nodal_crossing = 11 + float(c11.max() - 0.15) / 0.001
    data = np.array([[3, -0.1, 1], [0.1, 0.2, -0.3], [0.02, -0.04, 0.06], [-0.001, 0.002, -0.004]])
    sample = BdfDenseOutput(0.0, 0.4, 0.4, 3, data)
    t = 0.1234
    interval = backend.dense_interval(sample, t, slice(None))
    bases = backend.polynomial_bases(sample, 0.0, 0.4)
    u = Fraction.from_float(t) / Fraction.from_float(0.4)
    exact_values, exact_derivatives = [], []
    for j in range(3):
        exact_values.append(
            sum(
                Fraction.from_float(float(data[k, j])) * sum(v * u**i for i, v in enumerate(poly))
                for k, poly in enumerate(bases)
            )
        )
        exact_derivatives.append(
            sum(
                Fraction.from_float(float(data[k, j]))
                * sum(
                    i * poly[i] * u ** (i - 1) / Fraction.from_float(0.4)
                    for i in range(1, len(poly))
                )
                for k, poly in enumerate(bases)
            )
        )
    checks = {
        "off_node_true_crossing_enclosed": bracket["root_lower_s"] <= 10 <= bracket["root_upper_s"],
        "time_width_target": bracket["root_width_s"] <= 1e-4,
        "event_step_continuous_derivative_negative": bracket[
            "whole_event_step_strict_decrease_verified"
        ],
        "different_from_nodal_crossing": bracket["root_lower_s"] > nodal_crossing + 1e-4,
        "prior_scalar_witness": witness["strictly_above_threshold"],
        "rational_temporal_values_enclosed": all(
            Fraction.from_float(float(interval[0][j]))
            <= v
            <= Fraction.from_float(float(interval[1][j]))
            for j, v in enumerate(exact_values)
        ),
        "dense_derivative_matches_rational_formula": float(
            np.max(
                abs(
                    backend.dense_derivative(sample, t)
                    - np.array([float(v) for v in exact_derivatives])
                )
            )
        )
        < 1e-12,
    }
    report = {
        "completed": True,
        "passed": bool(space_pass and all(r["passed"] for r in temporal) and all(checks.values())),
        "spatial": {"passed": bool(space_pass), "exact_integrals": exact, "errors": errors},
        "temporal": temporal,
        "event": {"checks": checks, "bracket": bracket, "nodal_crossing_s": nodal_crossing},
        "current_execution_identity": source_identity(),
        "scope": "独立已知多项式和标量自适应求积；这些测试场仅用于算法验证。",
    }
    write_json(out / "audit_checks.json", report)
    if not report["passed"]:
        raise RuntimeError("时空守恒/连续事件独立核验未通过")
    return report


def physical(out, args):
    """Delegate the full physical run to the existing checked FE/FV implementation."""
    options = SimpleNamespace(
        engine=args.engine,
        inputs=DATA / "inputs/附件",
        contract=DATA / "config/model.json",
        out=out / "run",
        material=args.material,
        nr=args.nr,
        nz=args.nz,
        elements=args.elements,
        axial_power=args.axial_power,
        event=args.event,
        end=args.end,
        horizon=args.horizon,
        tight=args.tight,
        default_lu=args.default_lu,
        unweighted=args.unweighted,
    )
    result = backend.run(options)
    write_json(
        out / "execution.json",
        {
            "completed": True,
            "model_result": result,
            "current_execution_identity": source_identity(extra_files=(Path(__file__),)),
        },
    )
    return result


def backend_checks(out):
    """Actual FE/FV same-grid checks for checked LU and garbage-collection policy."""
    rows = []
    for engine in ("fe", "fv"):
        for material in ("P1", "P23", "P4"):
            runs = []
            for checked, collect, label in (
                (False, False, "default"),
                (True, False, "checked"),
                (True, True, "collected"),
            ):
                model = backend.build_model(
                    engine,
                    DATA / "inputs/附件",
                    DATA / "config/model.json",
                    material,
                    nr=8,
                    nz=4,
                    elements=4,
                    axial_power=0.5 if material == "P4" else 0.0,
                )
                directory = out / f"{engine}_{material}_{label}"
                directory.mkdir()
                solver = backend.solve_fe if engine == "fe" else backend.solve_fv
                result, state = solver(
                    model, directory, 1.0, event=False, checked=checked, collect=collect
                )
                write_json(directory / "summary.json", result)
                runs.append((directory, state))
            first = compare_fields(out / f"{engine}_{material}_LU", runs[0][0], runs[1][0])
            second = compare_fields(out / f"{engine}_{material}_collection", runs[1][0], runs[2][0])
            same = bool(np.array_equal(runs[1][1], runs[2][1]))
            rows.append(
                {
                    "engine": engine,
                    "material": material,
                    "LU_maxima": first["maxima"],
                    "collection_maxima": second["maxima"],
                    "collection_final_state_bitwise_equal": same,
                    "passed": first["maxima"]["T_K"]["upper"] < 1e-7
                    and first["maxima"]["C"]["upper"] < 1e-8
                    and same,
                }
            )
    report = {
        "completed": True,
        "passed": all(r["passed"] for r in rows),
        "rows": rows,
        "scope": "完整1秒小网格实际物理计算，验证后端和资源管理的数值一致性；大网格及干燥终点另行计算。",
    }
    write_json(out / "backend_checks.json", report)
    if not report["passed"]:
        raise RuntimeError("求解后端检查未通过")
    return report


class ExponentialRadius:
    """解析检验专用半径；分段接口与物理求解器一致。"""

    def __init__(self, radius, rate):
        self.initial = float(radius)
        self.rate = float(rate)

    def radius(self, t):
        return self.initial * np.exp(-self.rate * t)

    def breakpoints(self, start, end):
        return np.asarray([start, end], dtype=float)


def fv_analytic_case(out, family, gamma, nr):
    """5000秒Robin/运动Neumann解析解，调用生产FV算子与时间推进。"""
    core = science("moving_model")
    model = backend.build_model(
        "fv",
        DATA / "inputs/附件",
        DATA / "config/model.json",
        "P1",
        nr=nr,
        nz=nr // 2,
        axial_power=gamma,
    )
    diffusivity, stop = 1e-8, 5000.0
    model.properties = core.MaterialProperties.constant(
        rho=1.0, cp=1.0, k=diffusivity, diffusivity=diffusivity
    )
    model.ambient = core.ConstantAmbient(300.0, 1.0)
    radius, half_length = model.grid.radius_m, model.H
    if family == "moving_neumann":
        rate = np.log(0.02 / 0.012) / stop
        model.geometry = ExponentialRadius(radius, rate)
        model.h = model.hm = model.end_factor = 0.0
        radial_root, axial_root = brentq(j1, 3.0, 4.0), np.pi
        radial_clock = np.expm1(2 * rate * stop) / (2 * rate)
        axial_clock = stop if gamma == 0 else np.expm1(2 * rate * gamma * stop) / (2 * rate * gamma)
    elif family == "fixed_robin":
        if gamma != 0:
            raise ValueError("固定Robin解析解要求gamma=0")
        model.geometry = ExponentialRadius(radius, 0.0)
        model.h = model.hm = 8e-7
        model.end_factor = 1.0
        radial_biot, axial_biot = (
            model.hm * radius / diffusivity,
            model.hm * half_length / diffusivity,
        )
        radial_root = brentq(lambda x: x * j1(x) - radial_biot * j0(x), 1e-10, 2.40482555769)
        axial_root = brentq(lambda x: x * np.tan(x) - axial_biot, 1e-10, np.pi / 2 - 1e-10)
        radial_clock = axial_clock = stop
    else:
        raise ValueError("未知解析解族")
    phi = (
        j0(radial_root * model.grid.r_m / radius)[:, None]
        * np.cos(axial_root * model.z_m / half_length)[None, :]
    )
    initial = model.state(300.0 + 5 * phi, 1.0 + 0.1 * phi)
    decay = np.exp(
        -diffusivity
        * (radial_root**2 / radius**2 * radial_clock + axial_root**2 / half_length**2 * axial_clock)
    )
    out.mkdir(parents=True)
    settings = backend.Settings(2e-11, 1e-10, 2e-13, 30.0)
    result, state = backend.solve_fv(
        model,
        out,
        stop,
        settings=settings,
        event=False,
        initial=initial,
        sample_times=np.linspace(0.0, stop, 51),
        checked=True,
        collect=False,
    )
    temperature, moisture = model.unpack(state)
    row = {
        "family": family,
        "gamma": gamma,
        "Nr": nr,
        "Nz": nr // 2,
        "max_T_error_K": float(np.max(abs(temperature - (300.0 + 5 * phi * decay)))),
        "max_C_error": float(np.max(abs(moisture - (1.0 + 0.1 * phi * decay)))),
        "uniform_zero_flux_rhs_max": float(np.max(abs(model.rhs(1234.0, model.state(300.0, 1.0))))),
        "mass_residual": result["cumulative_moisture_balance_abs"],
        "settings": {
            "rtol": settings.rtol,
            "atol_T_K": settings.atol_t,
            "atol_C": settings.atol_c,
            "max_step_s": settings.max_step,
            "stop_s": stop,
        },
    }
    write_json(out / "summary.json", {**result, "analytic": row})
    return row


def fv_analytic_suite(out):
    """四类情景各24/48/96网格，检验二阶收敛和守恒。"""
    families = []
    for family, gamma in [
        ("fixed_robin", 0.0),
        ("moving_neumann", 0.0),
        ("moving_neumann", 0.5),
        ("moving_neumann", 1.0),
    ]:
        rows = []
        for nr in (24, 48, 96):
            row = fv_analytic_case(out / f"{family}_g{gamma:g}_N{nr}", family, gamma, nr)
            if rows:
                for field in ("T", "C"):
                    key = "max_T_error_K" if field == "T" else "max_C_error"
                    row[field + "_order"] = float(np.log2(rows[-1][key] / row[key]))
            rows.append(row)
            print(json.dumps(row), flush=True)
        passed = (
            rows[-1]["T_order"] > 1.75
            and rows[-1]["C_order"] > 1.75
            and max(x["uniform_zero_flux_rhs_max"] for x in rows) == 0
            and max(x["mass_residual"] for x in rows) < 1e-7
        )
        families.append({"family": family, "gamma": gamma, "rows": rows, "passed": passed})
    report = {
        "completed": True,
        "passed": all(f["passed"] for f in families),
        "families": families,
        "current_execution_identity": source_identity(),
    }
    write_json(out / "fv_analytic.json", report)
    if not report["passed"]:
        raise RuntimeError("FV解析收敛检验未通过")
    return report


def heat_observer_checks(out, stop=1800.0):
    """在相同真实输入上比较J加权审计开关：逐步轨迹、稠密系数和完整保存场。"""
    rows = []
    for material, gamma in [("P23", 0.0), ("P4", 0.0), ("P4", 0.5), ("P4", 1.0)]:
        runs, traces, records = [], [], []
        for weighted in (False, True):
            model = backend.build_model(
                "fe",
                DATA / "inputs/附件",
                DATA / "config/model.json",
                material,
                elements=4,
                axial_power=gamma,
            )
            directory = out / f"{material}_g{gamma:g}_J{int(weighted)}"
            directory.mkdir()
            trace = []
            record, state = backend.solve_fe(
                model, directory, stop, event=False, j_heat=weighted, trace=trace
            )
            write_json(directory / "summary.json", record)
            traces.append(
                [
                    {k: v.hexdigest() if k.endswith("_hash") else v for k, v in r.items()}
                    for r in trace
                ]
            )
            records.append(record)
            runs.append(directory)
        a, b = FieldSeries(runs[0]), FieldSeries(runs[1])
        equal = np.array_equal(a.times, b.times)
        for left, right in zip(a.rows(set(a.times)), b.rows(set(b.times))):
            equal &= all(np.array_equal(x, y) for x, y in zip(left[1:3], right[1:3]))
            equal &= left[0] == right[0] and left[3] == right[3]
        original_balance = records[0]["time_balance"]
        weighted_balance = dict(records[1]["time_balance"])
        weighted_heat = weighted_balance.pop("initial_volume_heat_audit")
        space_equal, space_count = True, 0
        with (runs[0] / "spatial_audit.jsonl").open(encoding="utf8") as sa, (
            runs[1] / "spatial_audit.jsonl"
        ).open(encoding="utf8") as sb:
            from itertools import zip_longest

            for la, lb in zip_longest(sa, sb):
                if la is None or lb is None:
                    space_equal = False
                    break
                x, y = json.loads(la), json.loads(lb)
                for key in (
                    "current_to_initial_volume_ratio_J",
                    "initial_volume_heat_rates",
                    "initial_volume_heat_scope",
                ):
                    y.pop(key, None)
                space_equal &= x == y
                space_count += 1
        row = {
            "material": material,
            "gamma": gamma,
            "stop_s": stop,
            "all_accepted_steps_states_dense_coefficients_bitwise_equal": traces[0] == traces[1],
            "accepted_step_traces": traces,
            "all_output_arrays_bitwise_equal": bool(equal),
            "output_rows": len(a.times),
            "original_time_audit_bitwise_equal": original_balance == weighted_balance,
            "original_spatial_audits_equal": space_equal,
            "spatial_rows": space_count,
            "J_weighted_heat_audit": weighted_heat,
            "field_identities": [a.identity, b.identity],
        }
        row["passed"] = bool(
            equal
            and traces[0] == traces[1]
            and original_balance == weighted_balance
            and space_equal
        )
        rows.append(row)
    report = {
        "completed": True,
        "passed": all(r["passed"] for r in rows),
        "rows": rows,
        "current_execution_identity": source_identity(),
        "scope": "J加权项是观测量；验证其在给定真实输入及时间范围内保持离散轨迹。",
    }
    write_json(out / "heat_observer_checks.json", report)
    if not report["passed"]:
        raise RuntimeError("J热审计轨迹等价性未通过")
    return report


def fixed_heat_identity(out, path):
    """固定P23的J恒为1，将逐步热积分直接换算到初始体积口径。"""
    fields = FieldSeries(path)
    if fields.kind != "fe":
        raise ValueError("固定热量恒等式要求FE场及逐步审计")
    record = read_json(Path(path) / "record.json")
    settings = record["settings"]
    if (
        not record["completed"]
        or settings["material"] != "P23"
        or settings["moving"]
        or settings["gamma"] != 0
    ):
        raise ValueError("要求完成的固定几何P23轨迹")
    initial_radius = None
    for _, _, _, geometry in fields.rows(set(fields.times)):
        if initial_radius is None:
            initial_radius = geometry["R_m"]
        if geometry["R_m"] != initial_radius or geometry["ell"] != 1.0:
            raise ValueError("保存几何与固定域条件矛盾")
    balance = record["time_balance"]
    g8, g16 = (np.asarray(balance[key])[[2, 3, 5]] for key in ("time_gauss8", "time_gauss16"))
    result = {
        "completed": True,
        "passed": True,
        "J_identically_one": True,
        "retained_geometry_rows": len(fields.times),
        "R0_m": initial_radius,
        "argument": "TensorFE.geometry在moving=False返回R0，gamma=0给出ell=1；因此J=(R/R0)^2*ell在全部时刻恒为1。",
        "initial_volume_heat_audit": {
            "gauss8": g8.tolist(),
            "gauss16": g16.tolist(),
            "gauss16_minus_gauss8": (g16 - g8).tolist(),
            "components": ["J_side_heat", "J_end_heat", "J_volume_bC_Tdot"],
            "heat_balance_abs_J_per_initial_m3": abs(float(g16.sum())),
            "heat_balance_normalized": abs(float(g16.sum()))
            / max(abs(float(g16[2])), abs(float(g16[:2].sum())), 1.0),
        },
        "field_identity": fields.identity,
        "record_sha256": sha256(Path(path) / "record.json"),
        "current_execution_identity": source_identity(),
        "scope": "给定离散轨迹的热量口径恒等变换；误差与时间积分仍由该轨迹审计给出。",
    }
    write_json(out / "fixed_heat_identity.json", result)
    return result


def plan(out, full_matrix=False):
    """Create an executable plan whose every command calls this integrated entry."""
    script = Path(__file__).resolve()
    jobs = []

    def add(label, mode, *arguments):
        jobs.append(
            {
                "label": label,
                "argv": [sys.executable, "-B", str(script), mode, *map(str, arguments)],
            }
        )

    add("algebra", "checks", "--out", out / "algebra")
    add("audit", "audit-check", "--out", out / "audit")
    add("backend", "backend-check", "--out", out / "backend")
    if full_matrix:
        add("fv_analytic", "fv-analytic", "--out", out / "fv_analytic")
        add("heat_observer", "heat-check", "--out", out / "heat_observer")
    cases = []

    def add_physical(label, engine, material, size, tight, *, gamma=None):
        mesh = ["--elements", size] if engine == "fe" else ["--nr", size, "--nz", size // 2]
        period = (
            ["--end", 1800]
            if gamma is None
            else ["--axial-power", gamma, "--event", "--horizon", 360000]
        )
        add(
            label,
            "physical",
            "--engine",
            engine,
            "--material",
            material,
            *mesh,
            *period,
            "--out",
            out / label,
            *(["--tight"] if tight else []),
        )
        cases.append(label)

    def add_comparison(coarse, fine, *, label=None, extra=(), destination=None):
        label = label or coarse + "__" + fine
        add(
            label,
            "compare",
            "--coarse",
            out / coarse / "run",
            "--fine",
            out / fine / "run",
            *extra,
            "--out",
            out / (destination or label),
        )

    # 网格、容差配对集中列出；命令格式仅在上面的两处构造。
    initial_grids = {
        "fe": ([(16, False), (32, False)] if full_matrix else [])
        + [(64, False), (128, False)]
        + (
            [(128, True), (256, False), (256, True), (512, False), (512, True)]
            if full_matrix
            else []
        ),
        "fv": ([(96, False)] if full_matrix else []) + [(192, False), (384, False), (384, True)],
    }
    for engine, grids in initial_grids.items():
        prefix = "FE_P1_E" if engine == "fe" else "FV_P1_N"
        for size, tight in grids:
            add_physical(
                prefix + str(size) + ("_tight" if tight else ""), engine, "P1", size, tight
            )
    add_comparison(
        "FE_P1_E64",
        "FE_P1_E128",
        label="FE_t1",
        extra=("--times", 1, "--gap", 1e-12),
        destination="FE_t1_comparison",
    )
    for coarse, fine, label in (
        ("FV_P1_N192", "FV_P1_N384", "FV_space"),
        ("FV_P1_N384", "FV_P1_N384_tight", "FV_time"),
    ):
        add_comparison(coarse, fine, label=label)
    if full_matrix:
        pairs = [(f"FE_P1_E{n}", f"FE_P1_E{2*n}") for n in (16, 32, 64, 128, 256)]
        pairs += [(f"FE_P1_E{n}", f"FE_P1_E{n}_tight") for n in (128, 256, 512)]
        pairs += [("FE_P1_E256_tight", "FE_P1_E512_tight"), ("FV_P1_N96", "FV_P1_N192")]
        for coarse, fine in pairs:
            add_comparison(coarse, fine)
        drying_grids = {
            "fe": [(32, False), (64, False), (128, False), (128, True)],
            "fv": [(96, False), (192, False), (384, False), (384, True)],
        }
        for material, gamma in (("P23", 0.0), ("P4", 0.0), ("P4", 0.5), ("P4", 1.0)):
            for engine, grids in drying_grids.items():
                labels = []
                for size, tight in grids:
                    label = f"{engine}_{material}_g{gamma:g}_{size}" + ("_tight" if tight else "")
                    add_physical(label, engine, material, size, tight, gamma=gamma)
                    labels.append(label)
                for coarse, fine in zip(labels[:-1], labels[1:]):
                    add_comparison(coarse, fine)
    comparison_matrix = []
    for job in jobs:
        argv = job["argv"]
        if argv[3] != "compare" or job["label"] == "FE_t1":
            continue
        coarse = Path(argv[argv.index("--coarse") + 1]).parent.name
        fine = Path(argv[argv.index("--fine") + 1]).parent.name
        engine = "fe" if coarse.lower().startswith("fe_") else "fv"
        kind = "time_settings" if fine == coarse + "_tight" else "spatial"
        final = (
            engine == "fv"
            or (coarse, fine)
            in [
                ("FE_P1_E256", "FE_P1_E512"),
                ("FE_P1_E512", "FE_P1_E512_tight"),
                ("FE_P1_E256_tight", "FE_P1_E512_tight"),
            ]
            or (coarse.startswith("fe_") and (fine.endswith("_128") or fine.endswith("_128_tight")))
        )
        comparison_matrix.append(
            {"label": job["label"], "engine": engine, "kind": kind, "final_accuracy_pair": final}
        )
    report = {
        "executed": False,
        "full_matrix": full_matrix,
        "jobs": jobs,
        "physical_cases": cases,
        "comparison_matrix": comparison_matrix,
        "entry_sha256": sha256(script),
        "current_execution_identity": source_identity(extra_files=(script,)),
        "scope": "按原网格序列和主/紧容差执行完整二维物理场及比较；事件模式复用已检查的首次事件实现。",
    }
    write_json(out / "production_plan.json", report)
    return report


def execute_plan(out, path):
    """Execute only commands for this fixed scientific entry; shell parsing is absent."""
    document = read_json(path)
    script = Path(__file__).resolve()
    if document["entry_sha256"] != sha256(script):
        raise ValueError("计划绑定的当前源码不同")
    completed = []
    for job in document["jobs"]:
        argv = job["argv"]
        if (
            len(argv) < 4
            or Path(argv[2]).resolve() != script
            or argv[1] != "-B"
            or argv[3]
            not in (
                "physical",
                "compare",
                "checks",
                "audit-check",
                "backend-check",
                "fv-analytic",
                "heat-check",
            )
            or Path(job["label"]).name != job["label"]
        ):
            raise ValueError("计划包含外部命令")
        started = time.perf_counter()
        with (out / (job["label"] + ".log")).open("w", encoding="utf8") as stream:
            result = subprocess.run(
                [sys.executable, *argv[1:]], stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        completed.append(
            {
                "label": job["label"],
                "returncode": result.returncode,
                "elapsed_s": time.perf_counter() - started,
            }
        )
        write_json(
            out / "execution.json",
            {
                "completed": len(completed) == len(document["jobs"]) and result.returncode == 0,
                "jobs": completed,
            },
        )
        if result.returncode:
            raise RuntimeError("计划任务失败: " + job["label"])
    return completed


def matrix_status(out, plan_path, criteria_path=None):
    """按绑定计划检查全部算例、比较覆盖和独立精度门槛；缺项逐项列出。"""
    document = read_json(plan_path)
    if not document.get("full_matrix"):
        raise ValueError("完整矩阵核验要求--full-matrix生成的计划")
    if document["entry_sha256"] != sha256(Path(__file__)):
        raise ValueError("计划绑定源码与本次核验入口不同")
    policy = None if criteria_path is None else read_json(criteria_path)
    if policy is not None:
        if policy.get("approved_for_final_use") is not True:
            raise ValueError("FV精度准则需approved_for_final_use=true")
        for kind in ("spatial", "time_settings"):
            for key in ("max_T_difference_K", "max_C_difference"):
                if not np.isfinite(policy[kind][key]) or policy[kind][key] <= 0:
                    raise ValueError("FV精度准则应为有限正数")
    jobs = {job["label"]: job for job in document["jobs"]}
    fields, cases, comparisons, failures = {}, [], [], []
    expected_inputs = {
        p.name: sha256(p)
        for p in (
            DATA / "config/model.json",
            DATA / "inputs/附件/附件1.xlsx",
            DATA / "inputs/附件/附件2.xlsx",
        )
    }
    for label in document["physical_cases"]:
        argv = jobs[label]["argv"]
        root = Path(argv[argv.index("--out") + 1]) / "run"
        try:
            field = FieldSeries(root)
            record = field.summary
            engine = argv[argv.index("--engine") + 1]
            if not record.get("completed") or record.get("input_sha256") != expected_inputs:
                raise ValueError("算例未完成或输入身份不同")
            if record.get("refactored_source_sha256") != sha256(Path(backend.__file__)):
                raise ValueError("算例物理程序身份不同")
            if (
                record.get("material") != argv[argv.index("--material") + 1]
                or record.get("engine") != engine
            ):
                raise ValueError("物性或离散类型不符")
            setting = record["settings"] if engine == "fe" else record
            if engine == "fe":
                if setting["radial_elements"] != int(
                    argv[argv.index("--elements") + 1]
                ) or not record.get("numerical_checks_passed"):
                    raise ValueError("FE网格或数值审计不符")
                expected = (
                    (2e-12, 1e-11, 2e-14, 60.0) if "--tight" in argv else (2e-10, 1e-9, 2e-12, 60.0)
                )
                actual = tuple(setting[k] for k in ("rtol", "atol_t", "atol_c", "max_step"))
            else:
                if (
                    field.coordinates["xi"].size != int(argv[argv.index("--nr") + 1]) + 1
                    or field.coordinates["zeta"].size != int(argv[argv.index("--nz") + 1]) + 1
                ):
                    raise ValueError("FV网格不符")
                expected = (
                    (2e-11, 1e-10, 2e-13, 30.0) if "--tight" in argv else (2e-10, 1e-9, 2e-12, 60.0)
                )
                actual = tuple(setting[k] for k in ("rtol", "atol_T_K", "atol_C", "max_step_s"))
            if not np.allclose(actual, expected, rtol=1e-14, atol=0):
                raise ValueError("时间容差与计划不同")
            if "--end" in argv:
                if not np.array_equal(field.times, np.arange(1801, dtype=float)):
                    raise ValueError("P1逐秒完整覆盖不足")
            elif not record.get("event") and record.get("event_time_s") is None:
                raise ValueError("长时算例缺少干燥事件")
            fields[label] = field
            cases.append(
                {
                    "label": label,
                    "engine": engine,
                    "rows": len(field.times),
                    "identity": field.identity,
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"label": label, "kind": "physical", "reason": str(exc)})
    for pair in document["comparison_matrix"]:
        argv = jobs[pair["label"]]["argv"]
        root = Path(argv[argv.index("--out") + 1])
        coarse = Path(argv[argv.index("--coarse") + 1]).parent.name
        fine = Path(argv[argv.index("--fine") + 1]).parent.name
        try:
            a, b = fields[coarse], fields[fine]
            report = read_json(root / "comparison.json")
            if report["source_fields"] != {"coarse": a.identity, "fine": b.identity}:
                raise ValueError("比较报告绑定了不同原始场")
            rows = report["rows"]
            if not report["completed"] or not np.array_equal(
                [r["t_s"] for r in rows], np.intersect1d(a.times, b.times)
            ):
                raise ValueError("比较遗漏共同时间")
            with (root / "differences.csv").open(encoding="utf8") as stream:
                csv_rows = list(csv.DictReader(stream))
            if len(csv_rows) != len(rows) or not np.array_equal(
                [float(r["t_s"]) for r in csv_rows], [r["t_s"] for r in rows]
            ):
                raise ValueError("逐时CSV覆盖与报告不同")
            for field, prefix in (("T_K", "T"), ("C", "C")):
                lower = max(float(r[prefix + "_lower"]) for r in csv_rows)
                upper = max(float(r[prefix + "_upper"]) for r in csv_rows)
                if (
                    not 0 <= lower <= upper
                    or not np.isfinite(upper)
                    or report["maxima"][field]["upper"] != upper
                    or report["maxima"][field]["lower"] != lower
                ):
                    raise ValueError("区间上下界或逐时极值不符")
            limits = (
                {"max_T_difference_K": 1e-5, "max_C_difference": 1e-5}
                if pair["engine"] == "fe"
                else (None if policy is None else policy[pair["kind"]])
            )
            outcomes = {}
            if limits is not None:
                for field, key in (("T_K", "max_T_difference_K"), ("C", "max_C_difference")):
                    value, limit = report["maxima"][field], limits[key]
                    outcomes[field] = (
                        "within_declared_bound"
                        if value["upper"] <= limit
                        else (
                            "definite_exceedance"
                            if value["lower"] > limit
                            else "upper_enclosure_does_not_establish_target"
                        )
                    )
                if "max_abs_root_difference_s" in limits:
                    difference = (report.get("endpoint_comparison") or {}).get(
                        "fine_minus_coarse_event_s"
                    )
                    if difference is not None:
                        outcomes["root"] = (
                            "within_declared_bound"
                            if abs(difference) <= limits["max_abs_root_difference_s"]
                            else "definite_exceedance"
                        )
            comparisons.append(
                {
                    **pair,
                    "outcomes": outcomes,
                    "evaluated": limits is not None,
                    "passed_accuracy": limits is not None
                    and all(v == "within_declared_bound" for v in outcomes.values()),
                    "summary_sha256": sha256(root / "comparison.json"),
                    "history_sha256": sha256(root / "differences.csv"),
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"label": pair["label"], "kind": "compare", "reason": str(exc)})
    gates = {}
    for engine, required_cases, required_pairs, required_final in (
        ("fe", 25, 21, 11),
        ("fv", 20, 15, 15),
    ):
        selected = [r for r in comparisons if r["engine"] == engine]
        final = [r for r in selected if r["final_accuracy_pair"]]
        count = sum(r["engine"] == engine for r in cases)
        gates[engine] = {
            "required_cases": required_cases,
            "completed_cases": count,
            "required_comparisons": required_pairs,
            "completed_comparisons": len(selected),
            "required_final_accuracy_pairs": required_final,
            "coverage_completed": count == required_cases and len(selected) == required_pairs,
            "accuracy_evaluated": len(final) == required_final
            and all(r["evaluated"] for r in final),
            "accuracy_passed": len(final) == required_final
            and all(r["passed_accuracy"] for r in final),
        }
    result = {
        "completed": all(g["coverage_completed"] for g in gates.values()),
        "passed": all(g["coverage_completed"] and g["accuracy_passed"] for g in gates.values()),
        "gates": gates,
        "cases": cases,
        "comparisons": comparisons,
        "failures_or_pending": failures,
        "plan_sha256": sha256(plan_path),
        "FV_criteria": policy,
        "current_execution_identity": source_identity(),
        "scope": "完成性与精度分别核验；FE采用各最终配对1e-5上界门槛，FV按提供的冻结准则评估。",
    }
    write_json(out / "matrix_status.json", result)
    return result


def summarize(out, paths):
    cases = []
    comparisons = []
    for path in paths:
        path = Path(path)
        if (path / "comparison.json").exists():
            r = read_json(path / "comparison.json")
            checks = {k: r["maxima"][k]["upper"] <= 1e-5 for k in ("T_K", "C")}
            comparisons.append(
                {
                    "path": str(path),
                    "completed": r["completed"],
                    "maxima": r["maxima"],
                    "accuracy_at_1e_5": checks,
                    "passes_declared_pair_target": all(checks.values()),
                    "sha256": sha256(path / "comparison.json"),
                }
            )
        else:
            root = path / "run" if (path / "run").is_dir() else path
            fields = FieldSeries(root)
            r = fields.summary
            cases.append(
                {
                    "path": str(path),
                    "rows": len(fields.times),
                    "completed": r.get("completed", False),
                    "engine": fields.kind,
                    "event": r.get("event"),
                    "time_balance": r.get("time_balance", r.get("balance")),
                    "spatial_audit": r.get("spatial_audit"),
                    "identity": fields.identity,
                }
            )
    report = {
        "completed": bool(cases or comparisons)
        and all(r["completed"] for r in cases + comparisons),
        "cases": cases,
        "comparisons": comparisons,
        "scope": "完成性、来源、全场覆盖与逐对数值精度分别报告。输入目录集合决定汇总范围。",
    }
    write_json(out / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "readback",
            "checks",
            "audit-check",
            "mms",
            "physical",
            "compare",
            "compare-baseline",
            "backend-check",
            "fv-analytic",
            "heat-check",
            "fixed-heat",
            "plan",
            "plan-2d",
            "execute-plan",
            "summarize",
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", choices=("all", "analytic", "nonlinear"), default="all")
    parser.add_argument("--engine", choices=("fe", "fv"), default="fe")
    parser.add_argument("--material", choices=("P1", "P23", "P4"), default="P1")
    parser.add_argument("--nr", type=int, default=96)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--elements", type=int, default=32)
    parser.add_argument("--axial-power", type=float, choices=(0.0, 0.5, 1.0), default=0.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--end", type=float)
    mode.add_argument("--event", action="store_true")
    parser.add_argument("--horizon", type=float, default=360000.0)
    parser.add_argument("--tight", action="store_true")
    parser.add_argument("--default-lu", action="store_true")
    parser.add_argument("--unweighted", action="store_true")
    parser.add_argument("--coarse", type=Path)
    parser.add_argument("--fine", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="一维主程序目录：Q1为grid_profiles.npz，其余为internal.npz；均需metadata.json",
    )
    parser.add_argument("--centre-tolerance", type=float, default=1e-10)
    parser.add_argument("--times", nargs="+", type=float)
    parser.add_argument("--gap", type=float)
    parser.add_argument("--full-matrix", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--runs", nargs="+", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--criteria", type=Path)
    args = parser.parse_args()
    out = prepare_output(args.out)
    required = {
        "compare": ("coarse", "fine"),
        "compare-baseline": ("run", "baseline"),
        "fixed-heat": ("run",),
        "execute-plan": ("plan",),
    }
    missing = [name for name in required.get(args.mode, ()) if getattr(args, name) is None]
    if missing:
        parser.error(args.mode + "需要" + "和".join("--" + name for name in missing))
    if (
        args.mode == "compare"
        and args.gap is not None
        and not (np.isfinite(args.gap) and args.gap > 0)
    ):
        parser.error("--gap应为有限正数")
    if (
        args.mode == "heat-check"
        and args.end is not None
        and not (np.isfinite(args.end) and args.end > 0)
    ):
        parser.error("--end应为有限正数")
    if args.mode == "summarize" and not (args.plan or args.runs):
        parser.error("summarize需要--runs或--plan")
    handlers = {
        "readback": lambda: readback(out),
        "checks": lambda: algebra_checks(out),
        "audit-check": lambda: audit_checks(out),
        "mms": lambda: manufactured_suite(out, args.family),
        "physical": lambda: physical(out, args),
        "compare": lambda: compare_fields(out, args.coarse, args.fine, args.times, args.gap),
        "compare-baseline": lambda: compare_baseline(
            out, args.run, args.baseline, args.times, args.centre_tolerance
        ),
        "backend-check": lambda: backend_checks(out),
        "fv-analytic": lambda: fv_analytic_suite(out),
        "heat-check": lambda: heat_observer_checks(out, 1800.0 if args.end is None else args.end),
        "fixed-heat": lambda: fixed_heat_identity(out, args.run),
        "plan": lambda: plan(out, args.full_matrix),
        "execute-plan": lambda: execute_plan(out, args.plan),
        "summarize": lambda: (
            matrix_status(out, args.plan, args.criteria) if args.plan else summarize(out, args.runs)
        ),
    }
    run_cli_modes("plan" if args.mode == "plan-2d" else args.mode, handlers)


if __name__ == "__main__":
    main()
