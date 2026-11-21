"""四种质量/几何假设、20任务的潜热耦合；共用显式求解器与事件认证。"""

from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from 公共接口 import configure, section, paths, read_json, write_json, sha256
from 研究求解 import (
    MAIN,
    TIGHT,
    LIMITS,
    require,
    identity,
    plan_hash,
    fresh,
    run_model,
    window_record,
    summarize_matrix,
    runtime,
    ProgressChecker,
    sample_run,
    equal_samples,
    restart_check,
    zero_event_check,
    effect_gate,
    verify_checks_receipt,
)


class LatentModel:
    """Combine the NIST table, dry-mass convention and additive surface heat operator."""

    def __init__(self, base, config, property_set, mass_mode="common_wet820", latent_scale=1.0):
        self.base = base
        self.p = config["property_sets"][property_set]
        c0 = config["initial_state"]["moisture_C"]
        require(mass_mode in ("common_wet820", "appendix_wet"), "Unknown dry-mass convention")
        rho = (
            config["property_sets"]["P1"]["rho_offset"]
            if mass_mode == "common_wet820"
            else self.p["rho_offset"] + self.p["rho_C_coefficient"] * c0
        )
        self.rhod0, self.latent_scale = float(rho / (1 + c0)), float(latent_scale)
        require(
            np.isfinite(self.latent_scale) and self.latent_scale >= 0, "Invalid latent multiplier"
        )
        self.n = base.grid.r_m.size
        folder = Path(paths()["latent_table"])
        receipt = read_json(folder / "manifest.json")
        require(
            receipt["completed"]
            and receipt["original_pdf_sha256"]
            == "b48928ace1e5a902fe69f3e674537e12cddd23af127483ddcd9543a54d845bf4"
            and sha256(folder / "L_water_reference.csv") == receipt["csv_sha256"],
            "NIST reference identity differs",
        )
        with (folder / "L_water_reference.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.temperature_K = np.array([float(r["temperature_C"]) for r in rows]) + 273.15
        self.L = np.array([float(r["delta_h_J_kg"]) for r in rows])
        require(
            len(rows) == 101
            and np.all(np.diff(self.temperature_K) > 0)
            and np.all(np.diff(self.L) < 0),
            "Invalid latent table",
        )
        self.slopes = np.diff(self.L) / np.diff(self.temperature_K)
        self.table = self

    def __getattr__(self, name):
        return getattr(self.base, name)

    def values(self, temperature_K):
        t = np.asarray(temperature_K, dtype=float)
        require(
            np.isfinite(t).all()
            and np.all(t >= self.temperature_K[0])
            and np.all(t <= self.temperature_K[-1]),
            "Outside NIST liquid reference 0.01..100 C; this does not prove freezing",
        )
        i = np.minimum(
            np.searchsorted(self.temperature_K, t, side="right") - 1, len(self.slopes) - 1
        )
        return self.L[i] + self.slopes[i] * (t - self.temperature_K[i]), self.slopes[i]

    def increment(self, t, y, *, right_of_breakpoint=False):
        R0 = self.base.grid.radius_m
        R = float(self.base.geometry.radius(t)) if hasattr(self.base, "geometry") else R0
        require(
            float(self.base.moisture_boundary_multiplier(t, R, R0)) == 1.0,
            "Latent closure requires moisture multiplier one",
        )
        ambient = self.base.ambient.values(t, right_of_breakpoint=right_of_breakpoint)
        T, C = self.base.unpack(y)
        Ts, Cs, Ceq = float(T[-1]), float(C[-1]), float(ambient.moisture_eq)
        p = self.p
        rho = p["rho_offset"] + p["rho_C_coefficient"] * Cs
        cp = p["cp_offset"] + p["cp_fraction_coefficient"] * Cs / (1 + Cs)
        B = rho * cp
        Bp = p["rho_C_coefficient"] * cp + rho * p["cp_fraction_coefficient"] / (1 + Cs) ** 2
        weight, hm = float(self.base.grid.volume_weights_m2[-1]), self.base.h_m_m_s
        require(
            min(R, R0, weight, B, self.rhod0, hm) > 0,
            "Positive geometry, capacity and transport required",
        )
        J = (R / R0) ** 2
        jw = self.rhod0 / J * hm * (Cs - Ceq)
        if self.latent_scale == 0:
            return SimpleNamespace(
                temperature_rhs=0.0,
                derivative_T=0.0,
                derivative_C=0.0,
                jw_kg_m2_s=jw,
                latent_flux_W_m2=0.0,
            )
        L, Lp = map(float, self.values(Ts))
        L *= self.latent_scale
        Lp *= self.latent_scale
        gain, q = R0**2 / (R * weight), L * jw
        return SimpleNamespace(
            temperature_rhs=-gain * q / B,
            derivative_T=-gain * jw * Lp / B,
            derivative_C=-gain * (L * self.rhod0 / J * hm / B - q * Bp / B**2),
            jw_kg_m2_s=jw,
            latent_flux_W_m2=q,
        )

    def rhs(self, t, y, *, right_of_breakpoint=False):
        result = self.base.rhs(t, y, right_of_breakpoint=right_of_breakpoint)
        if self.latent_scale != 0:
            try:
                result[self.n - 1] += self.increment(
                    t, y, right_of_breakpoint=right_of_breakpoint
                ).temperature_rhs
            except ValueError as error:
                error.failure_context = dict(
                    phase="RHS evaluation; may be an implicit trial",
                    t_s=float(t),
                    surface_T_K=float(self.unpack(y)[0][-1]),
                    right_of_breakpoint=bool(right_of_breakpoint),
                    physical_freezing_established=False,
                )
                raise
        return result

    def audit(self, t, y, *, right_of_breakpoint=False):
        T, C = self.base.unpack(y)
        d = self.rhs(t, y, right_of_breakpoint=right_of_breakpoint)
        ambient = self.base.ambient.values(t, right_of_breakpoint=right_of_breakpoint)
        R0 = self.base.grid.radius_m
        R = float(self.base.geometry.radius(t))
        J, w = (R / R0) ** 2, 2 * self.base.grid.volume_weights_m2 / R0**2
        rho, cp, _, _ = self.base.properties.evaluate(C, T)
        inc = self.increment(t, y, right_of_breakpoint=right_of_breakpoint)
        conv = self.base.h_W_m2_K * (float(T[-1]) - float(ambient.temperature_K))
        volume = float(w @ (rho * cp * d[: self.n]))
        boundary = 2 / R * (conv + inc.latent_flux_W_m2)
        return dict(
            moisture_mean_rate_residual=float(
                w @ d[self.n :] + 2 * self.base.h_m_m_s / R * (C[-1] - ambient.moisture_eq)
            ),
            thermal_current_volume_rate_residual=volume + boundary,
            thermal_initial_volume_rate_residual=J * (volume + boundary),
            mean_C=float(w @ C),
            radius_m=R,
            J=J,
            jw_kg_m2_s=inc.jw_kg_m2_s,
            latent_flux_W_m2=inc.latent_flux_W_m2,
        )


MATRIX_LEVELS = {
    "P23": (
        (640, "main"),
        (1280, "main"),
        (2560, "main"),
        (2560, "tight"),
        (5120, "main"),
        (5120, "tight"),
    ),
    "P4": ((1280, "main"), (2560, "main"), (5120, "main"), (5120, "tight")),
}


def matrix():
    """每种物性组合的网格与容差层级由矩阵配置明确给出。"""
    scenarios, jobs = [], []
    for prop, geometry in (("P23", "fixed"), ("P4", "measured")):
        for mass in ("common_wet820", "appendix_wet"):
            scenario = dict(
                id=f"{prop}_{mass}",
                property_set=prop,
                heat_set=prop,
                D_set=prop,
                geometry=geometry,
                mass_mode=mass,
                latent_scale=1.0,
            )
            scenarios.append(scenario)
            for N, level in MATRIX_LEVELS[prop]:
                jobs.append(
                    dict(
                        id=f'{scenario["id"]}_N{N}_{level}',
                        scenario=dict(scenario),
                        N=N,
                        level=level,
                        settings=dict(TIGHT if level == "tight" else MAIN),
                    )
                )
    require(len(scenarios) == 4 and len(jobs) == 20, "Incomplete coupled matrix")
    return scenarios, jobs


def plan():
    scenarios, jobs = matrix()
    return dict(
        schema="coupled20-direct-v1",
        scenarios=scenarios,
        jobs=jobs,
        initial_state=read_json(paths()["contract"])["initial_state"],
        source_bindings=identity(),
        horizon_s=864000.0,
        post_event_window_s=2.0,
        physical_seconds_cutoff=10800.0,
        full_seconds_cutoff=1800.0,
        acceptance=dict(
            spatial=dict(T_K=1e-5, C=1e-5), temporal=dict(T_K=1e-5, C=1e-5), integral_balance=LIMITS
        ),
        test_only=False,
        continuous_PDE_certified=False,
        effects_accepted=False,
    )


def validate(value):
    require(value == plan(), "Plan/source/input/factors/tolerances differ; generate a new plan")
    return dict(passed=True, jobs=20, production_PDE_started=False)


def build(scenario, N):
    moving, relative = section("moving_model"), section("relative_temperature")
    p = paths()
    base = moving.build_model(
        p["inputs"],
        p["contract"],
        N=N,
        power=2.0,
        properties=scenario["property_set"],
        geometry=scenario["geometry"],
    )
    return LatentModel(
        relative.RelativeTemperatureModel.from_model(base),
        read_json(p["contract"]),
        scenario["property_set"],
        scenario["mass_mode"],
        scenario["latent_scale"],
    )


def run(value, job_id, out, checks, previous=None, until_s=None):
    validate(value)
    verify_checks_receipt(
        checks, value, schema="coupled-direct-checks-v1", expected_names=CHECK_NAMES
    )
    jobs = [j for j in value["jobs"] if j["id"] == job_id]
    require(len(jobs) == 1, "Unknown or duplicate job")
    return run_model(
        value, jobs[0], build(jobs[0]["scenario"], jobs[0]["N"]), out, previous, until_s
    )


def summarize(value, runs, out=None, case_map=None):
    validate(value)
    result = summarize_matrix(value, runs, case_map)
    if result["effects_accepted"]:
        roots = {row["scenario"]["id"]: row["equality_root_s"] for row in result["scenarios"]}
        result["mass_hypothesis_effects"] = {
            prop: dict(
                appendix_minus_common_s=roots[prop + "_appendix_wet"]
                - roots[prop + "_common_wet820"],
                appendix_to_common_ratio=roots[prop + "_appendix_wet"]
                / roots[prop + "_common_wet820"],
            )
            for prop in ("P23", "P4")
        }
    else:
        result["mass_hypothesis_effects"] = None
    if out is not None:
        write_json(out, result)
    return result


def polynomial_check(moving):
    from fractions import Fraction as F
    from numpy.polynomial import Polynomial
    from 研究求解 import accepted_partition, polynomial_roots_unit, interval_integrals

    def add(a, c):
        q = [F(0)] * max(len(a), len(c))
        for i, x in enumerate(a):
            q[i] += x
        for i, x in enumerate(c):
            q[i] += x
        return q

    def scale(a, k):
        return [x * k for x in a]

    def mul(a, c):
        q = [F(0)] * (len(a) + len(c) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(c):
                q[i + j] += x * y
        return q

    def integ(a, lo, hi):
        return sum((x * (hi ** (i + 1) - lo ** (i + 1)) / F(i + 1) for i, x in enumerate(a)))

    class PowerDense:

        def __init__(self, columns):
            self.order = max(map(len, columns)) - 1
            self.t_old = 0.0
            self.t = 1.0
            self.t_shift = np.zeros(self.order)
            self.denom = np.ones(self.order)
            self.D = np.array(
                [list(x) + [F(0)] * (self.order + 1 - len(x)) for x in columns], dtype=float
            ).T

        def __call__(self, t):
            return np.polynomial.polynomial.polyval(np.asarray(t), self.D)

    theta = [F(0), F(4), F(-4)]
    R = [F(1), F(1, 10) if moving else F(0)]
    q = [F(1, 10), F(1, 100)]
    Cs = add([F(1, 20)], mul(R, q))
    poly = PowerDense([theta] * 3 + [Cs] * 3)

    class Table:
        temperature_K = np.array([300.0, 300.75, 301.0])
        slopes = np.array([-8 / 3, -4.0])
        L = np.array([10.0, 8.0, 7.0])

        def values(self, t):
            t = np.asarray(t)
            require(np.all((t >= 300) & (t <= 301)), "Synthetic table domain")
            i = np.minimum(np.searchsorted(self.temperature_K, t, side="right") - 1, 1)
            return (self.L[i] + self.slopes[i] * (t - self.temperature_K[i]), self.slopes[i])

    model = SimpleNamespace(
        nr=3,
        temperature_reference_K=300.0,
        grid=SimpleNamespace(radius_m=1.0, volume_weights_m2=np.array([0.125, 0.25, 0.125])),
        properties=SimpleNamespace(
            evaluate=lambda C, T: (np.full_like(C, 2.0), np.full_like(C, 3.0), C * 0 + 1, C * 0 + 1)
        ),
        geometry=SimpleNamespace(radius=lambda t: 1 + float(R[1]) * np.asarray(t)),
        ambient=SimpleNamespace(
            values=lambda t, right_of_breakpoint=False: SimpleNamespace(
                temperature_K=np.asarray(t) * 0 + 300.25, moisture_eq=np.asarray(t) * 0 + 0.05
            )
        ),
        h_W_m2_K=0.1,
        h_m_m_s=0.01,
        rhod0=1.0,
        latent_scale=1.0,
        table=Table(),
        moisture_boundary_multiplier=lambda t, R, R0: 1.0,
    )
    cuts, info = accepted_partition(model, poly, 0.0, 0.0, 1.0, False)
    require(
        all((np.min(abs(cuts - x)) < 1e-13 for x in (0.25, 0.5, 0.75))),
        "Missing back-and-forth/tangent table roots",
    )
    require(
        not polynomial_roots_unit(Polynomial([0.0])), "Identically-zero polynomial needs no split"
    )
    moisture = scale(q, F(1, 50))
    conv = scale(mul(R, add(theta, [F(-1, 4)])), F(1, 5))
    volume = scale(mul(mul(R, R), [F(4), F(-8)]), F(6))
    exact = np.zeros(4)
    for lo, hi, upper in [(F(0), F(1, 4), False), (F(1, 4), F(3, 4), True), (F(3, 4), F(1), False)]:
        L = add([F(11) if upper else F(10)], scale(theta, F(-4) if upper else F(-8, 3)))
        latent = scale(mul(L, q), F(1, 50))
        exact += np.array(
            [float(integ(p, lo, hi)) for p in (moisture, add(conv, latent), volume, latent)]
        )
    rows = []
    for order in (8, 16):
        got = sum(
            (
                interval_integrals(model, poly, 0.0, a, c, False, order)
                for a, c in zip(cuts[:-1], cuts[1:])
            ),
            np.zeros(7),
        )
        error = float(max(abs(got[[0, 1, 3, 5]] - exact)))
        bound = 1024 * np.finfo(float).eps * max(1.0, float(abs(exact).max()))
        require(error < bound, "Fraction signed polynomial flux/J-rate integral differs")
        require(
            got[2] >= abs(got[1]) - bound and got[6] >= got[2] - bound,
            "Wrong net/gross denominator ordering",
        )
        rows.append(
            {
                "order": order,
                "maximum_abs_difference": error,
                "roundoff_limit": bound,
                "actual": got.tolist(),
                "independent_fraction_selected_components": exact.tolist(),
            }
        )
    return {
        "passed": True,
        "moving": moving,
        "cuts": cuts.tolist(),
        "partition": info,
        "integrals": rows,
        "scope": "Independent rational signed integrals; absolute-channel branch partition checked numerically, not outward certification.",
    }


CHECK_NAMES = sorted(
    [
        "matrix_and_input_binding",
        "polynomial_fixed",
        "polynomial_moving",
        "domain_rejections",
        "restore",
        "late_event_and_extension",
        "exact_zero_left_event",
        "new_directory_pending_continuation",
        "breakpoint_semantics",
        "rejection_gates",
    ]
    + [f"operator_{p}_{m}" for p in ("P23", "P4") for m in ("common_wet820", "appendix_wet")]
    + [f"zero_L_{p}" for p in ("P23", "P4")]
    + [f"latent_{p}_{m}" for p in ("P23", "P4") for m in ("common_wet820", "appendix_wet")]
)


def check_operators(value, out, check):
    for scenario in value["scenarios"]:
        model = build(scenario, 32)
        n = model.nr
        Cdiff = Cerror = Herror = 0.0
        for t in (0.0, 60.0, 14400.0, 14400.0001, 259200.0, 864000.0):
            y = np.r_[np.linspace(-20.0, -12.0, n), np.linspace(0.5, 0.2, n)]
            got, reference = model.rhs(t, y), model.base.rhs(t, y)
            Cdiff = max(Cdiff, float(abs(got[n:] - reference[n:]).max()))
            audit = model.audit(t, y)
            Cerror = max(Cerror, abs(audit["moisture_mean_rate_residual"]))
            Herror = max(Herror, abs(audit["thermal_initial_volume_rate_residual"]))
        prop = model.p
        rhod = (
            820.0
            if scenario["mass_mode"] == "common_wet820"
            else prop["rho_offset"] + prop["rho_C_coefficient"] * 2.55
        ) / 3.55
        check(
            "operator_" + scenario["id"],
            Cdiff == 0.0 and model.rhod0 == rhod and Cerror < 1e-12 and Herror < 1e-7,
        )


def check_fixtures(value, out, check):
    for moving in (False, True):
        result = polynomial_check(moving)
        check("polynomial_" + ("moving" if moving else "fixed"), result["passed"])
    model = build(value["scenarios"][0], 16)
    invalid = []
    for temperature in (273.15, 373.1501, float("nan")):
        try:
            model.values(temperature)
        except ValueError:
            invalid.append(True)
        else:
            invalid.append(False)
    try:
        model.rhs(1.0, model.initial_state(270.0, 2.55))
    except ValueError as error:
        context = getattr(error, "failure_context", None)
    else:
        context = None
    check(
        "domain_rejections",
        all(invalid) and context is not None and not context["physical_freezing_established"],
    )


def check_short_runs(value, out, check):
    for prop in ("P23", "P4"):
        scenario = next(s for s in value["scenarios"] if s["property_set"] == prop)
        model = build(dict(scenario, latent_scale=0.0), 32)
        reference = sample_run(model.base, out / ("zero_" + prop + "_reference"), False, end=120.0)
        got = sample_run(model, out / ("zero_" + prop + "_audit"), True, end=120.0)
        check(
            "zero_L_" + prop,
            equal_samples(reference, got)
            and got["balance"]["passed"]
            and got["event_verification"]["wet_prefix_passed"],
        )
    for scenario in value["scenarios"]:
        model = build(scenario, 32)
        reference = sample_run(model, out / (scenario["id"] + "_reference"), False, end=120.0)
        got = sample_run(model, out / (scenario["id"] + "_audit"), True, end=120.0)
        check(
            "latent_" + scenario["id"],
            equal_samples(reference, got)
            and got["balance"]["passed"]
            and got["event_verification"]["wet_prefix_passed"],
        )


def check_events(value, out, check):
    model = build(value["scenarios"][-1], 16)
    check("restore", restart_check(model, out / "restore")["passed"])
    initial = np.r_[np.zeros(model.nr), 0.149 + 0.0010000001 * (1 - model.xi**2)]
    reference = sample_run(model, out / "late_reference", False, 200000.0, 200060.0, initial, True)
    got = sample_run(model, out / "late_audit", True, 200000.0, 200060.0, initial, True)
    check(
        "late_event_and_extension",
        equal_samples(reference, got)
        and got["balance"]["passed"]
        and got["event_verification"]["first_computed_crossing_supported"],
        artificial_initial_state=True,
    )
    check("exact_zero_left_event", zero_event_check(model, out / "exact_zero")["passed"])
    fixture = copy.deepcopy(value)
    job = copy.deepcopy(value["jobs"][-1])
    job.update(N=16, id="TEST_ONLY_N16_pending")
    fixture.update(jobs=[job], horizon_s=3.0, test_only=True)
    first = run_model(fixture, job, build(job["scenario"], 16), out / "pending_3", until_s=3.0)
    second = run_model(
        fixture,
        job,
        build(job["scenario"], 16),
        out / "pending_6",
        previous=out / "pending_3",
        until_s=6.0,
    )
    checked = window_record(out / "pending_6", fixture)
    try:
        validate(fixture)
    except ValueError:
        rejected = True
    else:
        rejected = False
    check(
        "new_directory_pending_continuation",
        rejected
        and first["event_pending"]
        and second["event_pending"]
        and checked["balance_passed"]
        and checked["wet_prefix_passed"],
    )
    reference = sample_run(model, out / "break_reference", False, 14399.0, 14401.0)
    got = sample_run(model, out / "break_audit", True, 14399.0, 14401.0)
    sides = [segment["right_of_breakpoint"] for segment in got["balance"]["segments"]]
    check(
        "breakpoint_semantics",
        equal_samples(reference, got) and got["balance"]["passed"] and sides == [False, True],
    )


def check_acceptance_gates(value, out, check):
    bad = copy.deepcopy(value)
    bad["acceptance"]["spatial"]["T_K"] = 1.0
    try:
        validate(bad)
    except ValueError:
        rejected = True
    else:
        rejected = False
    final = [dict(kind="space", target_met=True), dict(kind="time", target_met=True)]
    negatives = [
        effect_gate(final, True, [True, False, True, True], dict(passed=True)),
        effect_gate(
            [dict(kind="space", target_met=False), final[1]], True, [True] * 4, dict(passed=True)
        ),
        effect_gate(final, True, [True] * 4, None),
    ]
    check("rejection_gates", rejected and not any(negatives))


def checks(value, out):
    """The complete 20-check gate, including real short solves and independent polynomial integrals."""
    validate(value)
    out = fresh(out)
    check = ProgressChecker(out, "Bounded coupled check failed: ")
    rows = check.rows

    check("matrix_and_input_binding", len(value["jobs"]) == 20 and len(value["scenarios"]) == 4)
    check_operators(value, out, check)
    check_fixtures(value, out, check)
    check_short_runs(value, out, check)
    check_events(value, out, check)
    check_acceptance_gates(value, out, check)
    require(sorted(r["name"] for r in rows) == CHECK_NAMES, "Incomplete named coupled suite")
    validate(value)
    result = dict(
        schema="coupled-direct-checks-v1",
        passed=all(r["passed"] for r in rows),
        expected_check_names=CHECK_NAMES,
        checks=rows,
        plan_sha256=plan_hash(value),
        source_bindings=identity(),
        runtime=runtime(),
        production_PDE_started=False,
        scope="Bounded numerical fixtures only; no full 20-job accuracy or physical closure acceptance.",
    )
    write_json(out / "checks.json", result)
    return result


def main(argv=None):
    from 研究求解 import branch_cli

    return branch_cli("潜热20任务", plan, validate, checks, run, summarize, argv, builder=build)


if __name__ == "__main__":
    main()
