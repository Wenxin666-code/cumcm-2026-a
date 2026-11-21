"""环境平台识别、解释性假设、物性与几何交叉情景验证。

environment 从环境观测识别描述性末段；checks 独立核对最小二乘和边界输入。
plan/run 提供24种单因素情景，factorial 计算物性与几何的交叉项，extra
补充细网格时间精度；summarize 汇总62次完整事件求解并复核干侧报告时刻。

模型构造、物性、时间推进与移动半径采样直接调用已有公共依赖。相邻
网格/容差差属于经验精度指标，情景结果属于确定性敏感性分析。
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import replace, asdict
from datetime import datetime, timezone
from time import perf_counter
from itertools import combinations
import argparse
import gc
import json
import math
import numpy as np
from 验证公共接口 import (
    SUPPORT,
    prepare_output,
    read_json,
    write_json,
    sha256,
    science,
    source_hashes as current_source_hashes,
    write_csv,
    load_npz,
    audit_header,
)


def source_hashes(root=SUPPORT):
    """记录全部当前源码、两份附件与模型配置的身份。"""
    inputs = ("inputs/附件/附件1.xlsx", "inputs/附件/附件2.xlsx", "code/config/model.json")
    return current_source_hashes((root / name for name in inputs), root=root)


def initialize_models():
    """按需取得原有模型对象；帮助和纯代数函数可以独立读取。"""
    global core, moving, AmbientProfile
    core = science("_vendor_host2_model")
    moving = science("moving_model")
    AmbientProfile = core.AmbientProfile


def segment_costs(t, y):
    n = len(t)
    prefix = [np.r_[0.0, np.cumsum(a)] for a in [t, y, t * t, t * y, y * y]]
    linear = np.full((n + 1, n + 1), np.inf)
    constant = linear.copy()
    for i in range(n):
        for j in range(i + 2, n + 1):
            sx, sy, sxx, sxy, syy = (a[j] - a[i] for a in prefix)
            size = j - i
            xx = sxx - sx * sx / size
            xy = sxy - sx * sy / size
            yy = max(0.0, syy - sy * sy / size)
            constant[i, j] = yy
            linear[i, j] = max(0.0, yy - xy * xy / xx)
    return (linear, constant)


def identify_platform(t_s, temperature_C, min_points=12, max_segments=None):
    t = np.asarray(t_s) / 3600
    y = np.asarray(temperature_C)
    n = len(t)
    max_segments = n // min_points if max_segments is None else max_segments
    lin, const = segment_costs(t, y)
    dp = np.full((max_segments + 1, n + 1), np.inf)
    previous = np.full(dp.shape, -1, dtype=int)
    dp[0, 0] = 0.0
    for m in range(1, max_segments + 1):
        for j in range(m * min_points, n + 1):
            candidates = np.arange((m - 1) * min_points, j - min_points + 1)
            vals = dp[m - 1, candidates] + lin[candidates, j]
            k = int(np.argmin(vals))
            dp[m, j] = vals[k]
            previous[m, j] = candidates[k]
    records = []
    for m in range(1, max_segments + 1):
        starts = [0] if m == 1 else range((m - 1) * min_points, n - min_points + 1)
        for i in starts:
            sse = dp[m - 1, i] + const[i, n]
            if np.isfinite(sse):
                parameters = 3 * m - 1
                records.append(
                    {
                        "model": "terminal_constant",
                        "segments": m,
                        "tail_start_index": i,
                        "tail_start_s": float(t_s[i]),
                        "SSE_temperature_C2": float(sse),
                        "parameter_count_including_knots_and_variance": parameters,
                        "BIC_like": float(n * np.log(sse / n) + parameters * np.log(n)),
                    }
                )
        if np.isfinite(dp[m, n]):
            parameters = 3 * m
            records.append(
                {
                    "model": "all_segments_linear",
                    "segments": m,
                    "tail_start_index": int(previous[m, n]),
                    "tail_start_s": float(t_s[previous[m, n]]),
                    "SSE_temperature_C2": float(dp[m, n]),
                    "parameter_count_including_knots_and_variance": parameters,
                    "BIC_like": float(n * np.log(dp[m, n] / n) + parameters * np.log(n)),
                }
            )
    flat = min(
        (r for r in records if r["model"] == "terminal_constant"), key=lambda r: r["BIC_like"]
    )
    unrestricted = min(
        (r for r in records if r["model"] == "all_segments_linear"), key=lambda r: r["BIC_like"]
    )
    tail = int(flat["tail_start_index"])
    intervals = [(tail, n)]
    j = tail
    for m in range(int(flat["segments"]) - 1, 0, -1):
        i = int(previous[m, j])
        intervals.append((i, j))
        j = i
    intervals.reverse()
    segments = []
    for k, (i, j) in enumerate(intervals):
        slope, intercept = np.polyfit(t[i:j], y[i:j], 1)
        segments.append(
            {
                "first_index": i,
                "last_index": j - 1,
                "first_s": float(t_s[i]),
                "last_s": float(t_s[j - 1]),
                "points": j - i,
                "fitted_type": "constant" if k == len(intervals) - 1 else "linear",
                "diagnostic_OLS_slope_C_per_h": float(slope),
                "arithmetic_mean_C": float(np.mean(y[i:j])),
                "sample_sd_C": float(np.std(y[i:j], ddof=1)),
            }
        )
    return {
        "min_segment_points": min_points,
        "max_segments": max_segments,
        "selected_terminal_constant": flat,
        "best_all_linear": unrestricted,
        "terminal_constant_has_lower_score": flat["BIC_like"] < unrestricted["BIC_like"],
        "segments": segments,
        "candidate_scores": records,
    }


def analyze_environment(input_path, processed_dir):
    before = sha256(input_path)
    profile = AmbientProfile.from_xlsx(input_path)
    fit = identify_platform(profile.t_s, profile.temperature_C)
    i = fit["selected_terminal_constant"]["tail_start_index"]
    t0 = float(profile.t_s[i])
    end = float(profile.t_s[-1])
    mean_T = float(np.trapezoid(profile.temperature_C[i:], profile.t_s[i:]) / (end - t0))
    mean_C = float(np.trapezoid(profile.moisture[i:], profile.t_s[i:]) / (end - t0))
    candidates = [
        {
            "scenario": "ambient_mean_1h",
            "averaging_start_s": 10800.0,
            "averaging_end_s": end,
            "tail_temperature_C": profile.tail_temperature_C,
            "tail_Croom": profile.tail_moisture,
            "definition": "Piecewise-linear time mean of the last observed hour; regression anchor, not future observations",
        },
        {
            "scenario": "ambient_nominal",
            "averaging_start_s": None,
            "averaging_end_s": None,
            "tail_temperature_C": 50.0,
            "tail_Croom": 0.05,
            "definition": "Nominal candidate only; not specified by the problem statement",
        },
        {
            "scenario": "ambient_last",
            "averaging_start_s": end,
            "averaging_end_s": end,
            "tail_temperature_C": float(profile.temperature_C[-1]),
            "tail_Croom": float(profile.moisture[-1]),
            "definition": "Last observed node held constant after 4h",
        },
        {
            "scenario": "ambient_plateau",
            "averaging_start_s": t0,
            "averaging_end_s": end,
            "tail_temperature_C": mean_T,
            "tail_Croom": mean_C,
            "definition": (
                "Time mean over terminal segment selected from temperature alone, using the "
                "preregistered segmented-score rule"
            ),
        },
    ]
    stability = []
    for minimum in [8, 12, 18]:
        for maximum in [5, 6, 7, len(profile.t_s) // minimum]:
            alt = identify_platform(profile.t_s, profile.temperature_C, minimum, maximum)
            index = alt["selected_terminal_constant"]["tail_start_index"]
            duration = profile.t_s[-1] - profile.t_s[index]
            stability.append(
                {
                    "min_segment_points": minimum,
                    "max_segments": maximum,
                    "onset_s": float(profile.t_s[index]),
                    "segments": alt["selected_terminal_constant"]["segments"],
                    "terminal_constant_preferred": alt["terminal_constant_has_lower_score"],
                    "tail_temperature_C": float(
                        np.trapezoid(profile.temperature_C[index:], profile.t_s[index:]) / duration
                    ),
                    "tail_Croom": float(
                        np.trapezoid(profile.moisture[index:], profile.t_s[index:]) / duration
                    ),
                }
            )
    write_csv(
        processed_dir / "ambient_observed.csv",
        [
            {"t_s": float(t), "temperature_C": float(T), "Croom": float(C)}
            for t, T, C in zip(profile.t_s, profile.temperature_C, profile.moisture)
        ],
    )
    write_csv(processed_dir / "ambient_platform_candidates.csv", candidates)
    write_csv(processed_dir / "ambient_changepoint_scores.csv", fit.pop("candidate_scores"))
    write_csv(processed_dir / "ambient_changepoint_rule_sensitivity.csv", stability)
    result = {
        "source_file": Path(input_path).name,
        "source_sha256_before": before,
        "source_sha256_after": sha256(input_path),
        "observations": len(profile.t_s),
        "observed_interval_s": [0.0, end],
        "selection_protocol": {
            "temperature_only": True,
            "future_PDE_outcomes_used": False,
            "criterion": "n*log(SSE/n)+k*log(n); piecewise independent linear segments, optional terminal constant",
            "parameters": (
                "k=3m-1 for m segments with terminal constant; k=3m for all-linear segments; "
                "includes changepoint locations and one residual variance"
            ),
            "primary_min_segment_points": 12,
            "primary_max_segments": len(profile.t_s) // 12,
            "search_scope": (
                "All segment counts allowed by the minimum segment size; capped 5/6/7 "
                "searches are sensitivity diagnostics only. An initial six-segment "
                "computational probe selected its upper bound, so the primary search removes "
                "that active cap; no PDE output entered this choice."
            ),
            "noise_interpretation": (
                "BIC-like descriptive ranking corresponds to a working common-variance "
                "independent Gaussian residual model. Serial dependence and model-form error "
                "are not established; no change-time confidence interval, physical "
                "discontinuity, or future-platform guarantee is claimed."
            ),
            "mean_definition": (
                "Integrate measured piecewise-linear T and Croom over the same selected "
                "temperature-terminal interval; retain all original observed nodes for t<=4h."
            ),
        },
        "primary_fit": fit,
        "rule_sensitivity": stability,
        "candidates": candidates,
        "source_unchanged": before == sha256(input_path),
        "limitations": [
            (
                "A smoothly approaching thermal plateau can be approximated by a changepoint "
                "model without a physical change point."
            ),
            "Neither future ambient constancy nor Ceq=Croom is identified by this input-only fit.",
            "The 50 C / 0.05 nominal pair is a comparison candidate, not a supplied setpoint.",
        ],
    }
    path = processed_dir / "ambient_platform_detection.json"
    write_json(path, result)
    return result


def build_cases(ambient_analysis):
    ambient_by_name = {r["scenario"]: r for r in ambient_analysis["candidates"]}
    cases = []
    for question in ["P23", "P4"]:
        base = {
            "question": question,
            "geometry": "fixed" if question == "P23" else "measured",
            "ambient": "ambient_mean_1h",
            "alpha": 1.0,
            "h_factor": 1.0,
            "hm_factor": 1.0,
            "phase_switch_s": None,
            "boundary_normalization": "current_dry_density",
            "category": "ambient",
        }
        for name in ["ambient_mean_1h", "ambient_nominal", "ambient_last", "ambient_plateau"]:
            cases.append(dict(base, case_id=f"{question}_{name}", ambient=name))
        for alpha in [0.5, 1.5, 2.0]:
            cases.append(
                dict(
                    base,
                    case_id=f"{question}_alpha_{str(alpha).replace('.', 'p')}",
                    alpha=alpha,
                    category="Ceq_scale",
                )
            )
        for coeff in ["h", "hm"]:
            for factor in [0.9, 1.1]:
                cases.append(
                    dict(
                        base,
                        case_id=f"{question}_{coeff}_{str(factor).replace('.', 'p')}",
                        **{coeff + "_factor": factor},
                        category="exchange_coefficient",
                    )
                )
        if question == "P23":
            cases.append(
                dict(
                    base,
                    case_id="P23_P1_then_P23",
                    phase_switch_s=1800.0,
                    category="property_stage",
                )
            )
        else:
            cases.append(
                dict(
                    base,
                    case_id="P4_boundary_initial_density",
                    boundary_normalization="initial_dry_density_external_only",
                    category="boundary_closure",
                )
            )
    for case in cases:
        a = ambient_by_name[case["ambient"]]
        case["tail_temperature_C"] = a["tail_temperature_C"]
        case["tail_Croom"] = a["tail_Croom"]
        case["tail_Ceq"] = case["alpha"] * a["tail_Croom"]
        case["h_W_m2_K"] = 25.0 * case["h_factor"]
        case["hm_m_s"] = 8e-07 * case["hm_factor"]
    return cases


def build_case_model(root, case, N):
    cfg = read_json(root / "code/config/model.json")
    base = core.AmbientProfile.from_xlsx(root / "inputs/附件/附件1.xlsx")
    ambient = replace(
        base,
        tail_temperature_C=case["tail_temperature_C"],
        moisture=base.moisture * case["alpha"],
        tail_moisture=case["tail_Ceq"],
    )
    prop = core.MaterialProperties(**cfg["property_sets"][case["question"]], name=case["question"])
    h, hm = (case["h_W_m2_K"], case["hm_m_s"])
    if case["question"] == "P23":
        model = core.FixedRadialModel(core.RadialGrid.surface_stretched(N), ambient, prop, h, hm)
    else:
        multiplier = (
            moving.initial_dry_density_boundary
            if case["boundary_normalization"] == "initial_dry_density_external_only"
            else moving.current_dry_density_boundary
        )
        radius = moving.RadiusProfile.from_xlsx(root / "inputs/附件/附件2.xlsx")
        radius = replace(radius, tail_radius_m=float(radius.radius_m[-1]))
        model = moving.build_model(
            root / "inputs/附件",
            root / "code/config/model.json",
            N=N,
            properties=prop,
            geometry=radius,
            ambient=ambient,
            h_W_m2_K=h,
            h_m_m_s=hm,
            moisture_boundary_multiplier=multiplier,
        )
    return (model, cfg)


def solve_case(model, cfg, case, settings, horizon_s):
    if case["phase_switch_s"] is None:
        return (core.solve_to_event(model, settings, horizon_s=horizon_s), None)
    switch = case["phase_switch_s"]
    p1 = core.MaterialProperties(**cfg["property_sets"]["P1"], name="P1_before_1800s")
    early = replace(model, properties=p1)
    y = model.initial_state(301.15, 2.55)
    points = np.unique(np.r_[model.ambient.breakpoints(0.0, horizon_s), switch])
    segments = []
    switch_state = None
    jump = None
    early_end = None
    for start, stop in zip(points[:-1], points[1:]):
        active = early if start < switch else model
        segment = core._solve_segment(
            active,
            (float(start), float(stop)),
            y,
            settings,
            right_of_breakpoint=bool(start >= model.ambient.breakpoint_s),
            detect_event=True,
        )
        if start == switch:
            jump = float(np.max(np.abs(segment.result.y[:, 0] - switch_state)))
        segments.append(segment)
        event = segment.result.t_events[0]
        y = np.asarray(segment.result.y[:, -1], dtype=float)
        if stop == switch:
            switch_state = y.copy()
            T, C = model.unpack(y)
            early_end = {
                "center_T_C": float(T[0] - 273.15),
                "surface_T_C": float(T[-1] - 273.15),
                "center_C": float(C[0]),
                "surface_C": float(C[-1]),
            }
        if len(event):
            event_s = float(event[0])
            before = segment.result.t[segment.result.t < event_s - 1e-09]
            return (
                core.PiecewiseTrajectory(
                    model,
                    segments,
                    event_s,
                    (float(before[-1]) if len(before) else float(start), event_s),
                ),
                {
                    "switch_time_s": switch,
                    "state_jump_max_abs": jump,
                    "continuous_to_machine_equality": jump == 0.0,
                    "P1_end_state_summary": early_end,
                    "P1_properties": asdict(p1),
                    "interpretation": (
                        "True P1 integration from original uniform initial condition to 1800s, then "
                        "continuous T/C state handed to P23. Not a shifted or reset P23 trajectory."
                    ),
                },
            )
    raise RuntimeError("No drying event within the declared horizon")


def sample_run(trajectory, case):
    event = float(trajectory.event_time_s)
    regular = np.arange(0, event, 300.0)
    special = np.array(
        [
            1.0,
            60.0,
            100.0,
            600.0,
            900.0,
            1200.0,
            1500.0,
            1800.0,
            3600.0,
            5400.0,
            7200.0,
            9000.0,
            10800.0,
            14400.0,
        ]
    )
    times = np.unique(np.r_[regular, special[special < event], event])
    T, C = trajectory.evaluate_fields(times)
    model = trajectory.model
    weights = model.grid.volume_weights_m2
    xi = np.linspace(0, 1, 21)
    relative_grid = model.grid.r_m / model.grid.radius_m
    result = {
        "t_s": times,
        "xi": xi,
        "T_K": core.sample_radii(T, relative_grid, xi),
        "C": core.sample_radii(C, relative_grid, xi),
        "max_C": C.max(axis=1),
        "mean_C": C @ weights / weights.sum(),
        "minimum_C": C.min(axis=1),
        "center_C": C[:, 0],
        "surface_C": C[:, -1],
        "center_T_K": T[:, 0],
        "surface_T_K": T[:, -1],
        "event_node_r_ref_m": model.grid.r_m,
        "event_T_K": T[-1],
        "event_C": C[-1],
    }
    result["R_m"] = (
        np.asarray(model.geometry.radius(times))
        if case["question"] == "P4"
        else np.full(len(times), 0.02)
    )
    if case["question"] == "P4":
        physical = moving.sample_physical(trajectory, times, np.arange(20) * 0.001)
        for name in ["r_sample_m", "valid_interior", "C_sample", "T_K_sample"]:
            result[name] = physical[name]
    else:
        result["r_m"] = xi * 0.02
    diagnostics = {
        "min_sampled_C": float(C.min()),
        "max_sampled_C": float(C.max()),
        "min_sampled_T_K": float(T.min()),
        "max_sampled_T_K": float(T.max()),
        "maximum_positive_radial_C_increment": float(np.max(np.diff(C, axis=1))),
        "maximum_positive_increment_of_max_C": float(np.max(np.diff(C.max(axis=1)))),
        "all_fields_finite": bool(np.isfinite(T).all() and np.isfinite(C).all()),
        "sample_rows": len(times),
        "sample_step_s": 300.0,
        "extra_initial_and_stage_times": special.tolist(),
    }
    return (result, diagnostics)


def factorial_case(base, property_set, geometry):
    """交叉情景的物性与几何显式分开，question 仅选择几何采样接口。"""
    return dict(
        base,
        case_id=f"factorial_{property_set}_{geometry}",
        question="P23" if geometry == "fixed" else "P4",
        geometry=geometry,
        property_set=property_set,
        category="property_geometry_factorial",
        question_field_role="Shared helper geometry/sampling selector; property_set is the actual material.",
    )


def execute_case(root, out, case, N, kind, hashes, *, property_set=None, geometry=None):
    """统一执行单因素或交叉情景；读回时完整校验已有样本和来源。"""
    label = f"{case['case_id']}__n{N}__{kind}"
    destination = out / label
    record_path = destination / "record.json"
    if record_path.exists():
        record = verified_study_record(out, case, N, kind, hashes, property_set, geometry)
        print(json.dumps({"run": label, "reused_verified_run": True}), flush=True)
        return record
    settings = core.IntegrationSettings(**study_settings(kind))
    model, cfg = build_case_model(root, case, N)
    if property_set is not None:
        if kind != "main" or case["phase_switch_s"] is not None:
            raise ValueError("交叉情景要求 main 参数和统一物性阶段。")
        prop = core.MaterialProperties(**cfg["property_sets"][property_set], name=property_set)
        model = replace(model, properties=prop)
    start = perf_counter()
    trajectory, switch_audit = solve_case(model, cfg, case, settings, 2592000.0)
    data, diagnostics = sample_run(trajectory, case)
    event = float(trajectory.event_time_s)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "samples.npz", **data)
    protected_after = source_hashes(root)
    if property_set is not None:
        protected_after["factorial_program"] = sha256(Path(__file__))
    if protected_after != hashes:
        raise RuntimeError(
            "An input, shared source, config, or study source changed during integration"
        )
    record = {
        "label": label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_run": True,
        "case": case,
        "grid_intervals": N,
        "run_kind": kind,
        "settings": asdict(settings),
        "event_root_s": event,
        "event_root_h": event / 3600.0,
        "event_max_C": float(data["event_C"].max()),
        "event_R_m": float(data["R_m"][-1]),
        "crosses_72h": event > 259200.0,
        "uses_radius_extrapolation": case["question"] == "P4" and event > 259200.0,
        "diagnostics": diagnostics,
        "wall_seconds": perf_counter() - start,
        "source_hashes": hashes,
        "source_hashes_after": protected_after,
        "samples_sha256": sha256(destination / "samples.npz"),
    }
    if property_set is not None:
        record.update(
            actual_property_set=property_set,
            actual_geometry=geometry,
            properties=asdict(prop),
            event_kind="Equality root/infimum; no earliest attained time under strict dryness is claimed.",
        )
    else:
        record.update(
            {
                "event_step_bracket_s": trajectory.event_step_bracket_s,
                "event_kind": "First decreasing nodal maximum C crossing 0.15. Equality-root infimum; not "
                "an attained minimum time for strict C<0.15.",
                "strict_report_status": "Separate postprocessing will add a verified dry-side display time using "
                "declared empirical grid/time margins.",
                "uses_ambient_extrapolation": event > 14400.0,
                "radius_tail_rule": (
                    (
                        "Hold the actual last measured radius after 72h; assumption beyond "
                        "observation, not observational support."
                    )
                    if case["question"] == "P4"
                    else None
                ),
                "radius_tail_m": (
                    float(model.geometry.radius_m[-1]) if case["question"] == "P4" else None
                ),
                "stage_switch_audit": switch_audit,
                "accepted_steps": trajectory.accepted_steps,
                "function_evaluations": trajectory.function_evaluations,
                "sample_array_scope": "Full time span at 300s plus early/stage times and equality endpoint. 21 "
                "relative-radius profiles; P4 also fixed physical radii with strict interior "
                "mask. No full per-second Excel export.",
            }
        )
    write_json(record_path, record)
    print(
        json.dumps(
            {
                "run": label,
                "event_h": event / 3600.0,
                "crosses_72h": event > 259200.0,
                "seconds": record["wall_seconds"],
            }
        ),
        flush=True,
    )
    del trajectory, model, data
    gc.collect()
    return record


def least_squares_sse(x, y, constant=False):
    design = np.ones((len(x), 1)) if constant else np.column_stack((x, np.ones(len(x))))
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(residual @ residual)


def comparison_arrays(out, a, b):
    x = load_npz(out / a["label"] / "samples.npz")
    y = load_npz(out / b["label"] / "samples.npz")
    t, ix, iy = np.intersect1d(x["t_s"], y["t_s"], return_indices=True)
    result = {
        "common_time_rows": len(t),
        "common_time_max_s": float(t[-1]),
        "max_abs_C_at_21_relative_radii": float(np.max(np.abs(x["C"][ix] - y["C"][iy]))),
        "max_abs_T_K_at_21_relative_radii": float(np.max(np.abs(x["T_K"][ix] - y["T_K"][iy]))),
        "endpoint_abs_difference_s": abs(a["event_root_s"] - b["event_root_s"]),
    }
    if "valid_interior" in x:
        mask = x["valid_interior"][ix] & y["valid_interior"][iy]
        result["max_abs_C_common_physical_interior"] = float(
            np.max(np.abs(x["C_sample"][ix] - y["C_sample"][iy])[mask])
        )
        result["max_abs_T_K_common_physical_interior"] = float(
            np.max(np.abs(x["T_K_sample"][ix] - y["T_K_sample"][iy])[mask])
        )
    return result


def verify_report(root, out, fine, grid_margin_s, time_margin_s, time_source, report_directory):
    start = perf_counter()
    case = fine["case"]
    event = fine["event_root_s"]
    margin = grid_margin_s + time_margin_s
    report_s = math.ceil((event + max(margin, 1e-06)) / 0.36) * 0.36
    if report_s <= event:
        report_s += 0.36
    model, _ = build_case_model(root, case, fine["grid_intervals"])
    data = load_npz(out / fine["label"] / "samples.npz")
    state = np.r_[data["event_T_K"], data["event_C"]]
    settings = core.IntegrationSettings(**fine["settings"])
    points = model.ambient.breakpoints(event, report_s)
    for a, b in zip(points[:-1], points[1:]):
        seg = core._solve_segment(
            model,
            (float(a), float(b)),
            state,
            settings,
            right_of_breakpoint=bool(a >= model.ambient.breakpoint_s),
            detect_event=False,
        )
        state = np.asarray(seg.result.y[:, -1])
    T, C = model.unpack(state)
    passed = bool(np.isfinite(state).all() and C.max() < 0.15)
    if not passed:
        raise RuntimeError("Dry-side report failed: " + fine["label"])
    path = report_directory / fine["label"] / "report_state.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t_s=np.array([report_s]), r_reference_m=model.grid.r_m, T_K=T, C=C)
    record = {
        "case_id": case["case_id"],
        "source_run": fine["label"],
        "source_record_sha256": sha256(out / fine["label"] / "record.json"),
        "source_samples_sha256": fine["samples_sha256"],
        "report_state_sha256": sha256(path),
        "event_root_s": event,
        "event_root_h": event / 3600,
        "event_definition": "Equality crossing infimum for the strict drying condition; strict earliest time is not attained.",
        "reported_time_s": report_s,
        "reported_time_h": report_s / 3600,
        "reported_time_h_4dp": f"{report_s / 3600:.4f}",
        "reported_max_C_unrounded": float(C.max()),
        "strictly_dry_on_fine_discrete_model": passed,
        "report_tick_s": 0.36,
        "report_minus_root_s": report_s - event,
        "empirical_grid_margin_s": grid_margin_s,
        "empirical_time_margin_s": time_margin_s,
        "time_margin_source": time_source,
        "total_empirical_margin_s": margin,
        "margin_interpretation": (
            "Full adjacent-grid endpoint difference at matched solver settings plus a "
            "same-grid tolerance difference. P4 baseline fine reports use the additional "
            "N1280 tight run and N640 tight spatial reference. For nonbaseline scenarios "
            "the time component is an explicitly declared matched-geometry baseline "
            "proxy. Not a rigorous PDE/model/physical error bound."
        ),
        "verification": (
            "Restart fine-grid state at its equality root and integrate to the next "
            "selected 0.0001h tick; test all nodal C without rounding."
        ),
        "wall_seconds": perf_counter() - start,
    }
    write_json(report_directory / fine["label"] / "report.json", record)
    return record


def method_checks(out, cases):
    profile = core.AmbientProfile.from_xlsx(SUPPORT / "inputs/附件/附件1.xlsx")
    x = profile.t_s / 3600
    y = profile.temperature_C
    linear, constant = segment_costs(x, y)
    rng = np.random.default_rng(1729)
    errors = []
    for _ in range(200):
        a = int(rng.integers(0, len(x) - 3))
        b = int(rng.integers(a + 3, len(x) + 1))
        errors.extend(
            [
                abs(linear[a, b] - least_squares_sse(x[a:b], y[a:b])),
                abs(constant[a, b] - least_squares_sse(x[a:b], y[a:b], True)),
            ]
        )
    xx = np.arange(13, dtype=float)
    yy = 2 + np.sin(xx / 4) + rng.normal(0, 0.02, len(xx))
    small = identify_platform(xx * 3600, yy, min_points=3, max_segments=4)
    brute = []
    for m in range(1, 5):
        for cuts in combinations(range(3, len(xx) - 2), m - 1):
            pts = (0,) + cuts + (len(xx),)
            if min(np.diff(pts)) < 3:
                continue
            for terminal in [True, False]:
                sse = sum(
                    (
                        least_squares_sse(xx[a:b], yy[a:b], terminal and j == m - 1)
                        for j, (a, b) in enumerate(zip(pts[:-1], pts[1:]))
                    )
                )
                k = 3 * m - 1 if terminal else 3 * m
                brute.append(
                    (len(xx) * np.log(sse / len(xx)) + k * np.log(len(xx)), terminal, m, pts[-2])
                )
    brute_errors = {}
    for terminal, key in [(True, "selected_terminal_constant"), (False, "best_all_linear")]:
        expected = min((r for r in brute if r[1] == terminal))
        got = small[key]
        assert (got["segments"], got["tail_start_index"]) == (expected[2], expected[3])
        brute_errors[key] = abs(got["BIC_like"] - expected[0])
    fit = identify_platform(profile.t_s, profile.temperature_C)
    kelvin_fit = identify_platform(profile.t_s, profile.temperature_C + 273.15)
    same_onset = (
        fit["selected_terminal_constant"]["tail_start_s"]
        == kelvin_fit["selected_terminal_constant"]["tail_start_s"]
    )
    scaled = next((c for c in cases if c["case_id"] == "P4_alpha_2p0"))
    model, _ = build_case_model(SUPPORT, scaled, 320)
    node_times = [0.0, 1800.0, 10800.0, 14400.0]
    alpha_errors = [
        abs(float(model.ambient.values(t).moisture_eq) - 2 * float(profile.values(t).moisture_eq))
        for t in node_times
    ]
    radius_tail = float(model.geometry.radius(300000.0))
    radius_last = float(model.geometry.radius_m[-1])
    assert max(errors) < 1e-07 and max(brute_errors.values()) < 1e-07 and same_onset
    assert max(alpha_errors) < 1e-14 and radius_tail == radius_last
    report = {
        "passed": True,
        "prefix_cost_random_interval_comparisons": 400,
        "maximum_SSE_difference_from_direct_lstsq": max(errors),
        "exhaustive_partition_score_errors": brute_errors,
        "Celsius_Kelvin_onset_identical": same_onset,
        "alpha_scaling_max_error_at_observed_nodes": max(alpha_errors),
        "radius_after_72h_m": radius_tail,
        "radius_last_observed_m": radius_last,
        "scope": "Independent algebraic input/segmentation checks; no claim of physical validation.",
    }
    write_json(out / "method_checks.json", report)
    print(report)
    return report


def study_settings(kind):
    """用于生成和读回校验的两组时间积分参数。"""
    settings = core.IntegrationSettings(
        rtol=1e-9, atol_temperature_K=1e-9, atol_moisture_C=1e-11, max_step_s=150.0
    )
    if kind == "time_tight":
        settings = replace(
            settings, rtol=1e-10, atol_temperature_K=1e-10, atol_moisture_C=1e-12, max_step_s=75.0
        )
    elif kind != "main":
        raise ValueError("未知积分参数组。")
    return asdict(settings)


def verified_study_record(directory, case, N, kind, hashes, property_set=None, geometry=None):
    """从原始文件核对情景、网格、时间设置、源码、端点与样本身份。"""
    label = f"{case['case_id']}__n{N}__{kind}"
    path = directory / label / "record.json"
    record = read_json(path)
    expected = {
        "label": label,
        "case": case,
        "grid_intervals": N,
        "run_kind": kind,
        "actual_run": True,
        "settings": study_settings(kind),
        "source_hashes": hashes,
        "source_hashes_after": hashes,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{label}: {key}与指定情景不符。")
    if property_set is not None:
        if (
            record.get("actual_property_set") != property_set
            or record.get("actual_geometry") != geometry
        ):
            raise ValueError(label + ": 交叉项物性/几何不符。")
        cfg = read_json(SUPPORT / "code/config/model.json")
        properties = asdict(
            core.MaterialProperties(**cfg["property_sets"][property_set], name=property_set)
        )
        if record.get("properties") != properties:
            raise ValueError(label + ": 交叉项实际物性参数不符。")
    sample_path = directory / label / "samples.npz"
    if sha256(sample_path) != record.get("samples_sha256"):
        raise ValueError(label + ": 样本哈希不符。")
    event = record.get("event_root_s")
    if not isinstance(event, (int, float)) or not np.isfinite(event) or event <= 0:
        raise ValueError(label + ": 干燥事件时刻无效。")
    data = load_npz(sample_path)
    times, temperature, moisture = data["t_s"], data["T_K"], data["C"]
    if (
        times.ndim != 1
        or len(times) < 2
        or times[0] != 0
        or times[-1] != event
        or not np.all(np.diff(times) > 0)
        or not np.isfinite(times).all()
        or temperature.shape != (len(times), 21)
        or moisture.shape != temperature.shape
        or not np.isfinite(temperature).all()
        or not np.isfinite(moisture).all()
        or not np.array_equal(data["xi"], np.linspace(0, 1, 21))
    ):
        raise ValueError(label + ": 保存时间或采样场无效。")
    event_C, event_T = data["event_C"], data["event_T_K"]
    if (
        event_C.shape != (N + 1,)
        or event_T.shape != (N + 1,)
        or not np.isfinite(event_C).all()
        or not np.isfinite(event_T).all()
        or abs(float(event_C.max()) - 0.15) > 1e-8
        or record.get("event_max_C") != float(event_C.max())
        or record.get("event_root_h") != event / 3600
    ):
        raise ValueError(label + ": 端点状态与记录不符。")
    if not record.get("diagnostics", {}).get("all_fields_finite"):
        raise ValueError(label + ": 场有限性审查失败。")
    return record


def expected_study_cases(directory):
    """按本次环境候选生成24项明确情景，检查输入身份和完整集合。"""
    environment = read_json(directory / "environment/ambient_platform_detection.json")
    source = sha256(SUPPORT / "inputs/附件/附件1.xlsx")
    if (
        environment.get("source_sha256_before") != source
        or environment.get("source_sha256_after") != source
        or not environment.get("source_unchanged")
    ):
        raise ValueError("环境识别记录与当前附件身份不符。")
    cases = build_cases(environment)
    if len(cases) != 24 or len({c["case_id"] for c in cases}) != 24:
        raise ValueError("环境候选没有产生24个独立情景。")
    manifest = read_json(directory / "study_manifest.json")
    actual = manifest.get("cases", [])
    if len(actual) != 24 or {c["case_id"]: c for c in actual} != {c["case_id"]: c for c in cases}:
        raise ValueError("情景清单缺项、重复或参数不符。")
    return cases


def factorial_cells(directory, cases, hashes):
    """读回2物性×2几何×2网格的八个文件记录。"""
    by_name = {case["case_id"]: case for case in cases}
    values = {}
    for prop, geometry, selector in [
        ("P23", "fixed", "P23"),
        ("P4", "measured", "P4"),
        ("P4", "fixed", "P23"),
        ("P23", "measured", "P4"),
    ]:
        base = by_name[selector + "_ambient_mean_1h"]
        diagonal = prop == selector
        case = base if diagonal else factorial_case(base, prop, geometry)
        expected_hashes = (
            hashes if diagonal else dict(hashes, factorial_program=sha256(Path(__file__)))
        )
        for N in (320, 640):
            values[prop, geometry, N] = verified_study_record(
                directory,
                case,
                N,
                "main",
                expected_hashes,
                property_set=None if diagonal else prop,
                geometry=None if diagonal else geometry,
            )
    return values


def verify_factorial_report(directory, report, cases, hashes):
    """汇总时重新读取八格证据，防止报告代替实际交叉算例。"""
    if not report.get("passed") or report.get("program_sha256") != sha256(Path(__file__)):
        raise ValueError("交叉试验报告未通过或程序身份不符。")
    values = factorial_cells(directory, cases, hashes)
    cells = report.get("cells", [])
    keys = [(c["property_set"], c["geometry"], c["N"]) for c in cells]
    if len(keys) != 8 or len(set(keys)) != 8 or set(keys) != set(values):
        raise ValueError("交叉试验报告不是完整2×2×2集合。")
    for cell in cells:
        key = cell["property_set"], cell["geometry"], cell["N"]
        record = values[key]
        expected = {
            "source_run": record["label"],
            "samples_sha256": record["samples_sha256"],
            "source_record_sha256": sha256(directory / record["label"] / "record.json"),
            "event_root_s": record["event_root_s"],
            "event_root_h": record["event_root_h"],
        }
        if any(cell.get(k) != v for k, v in expected.items()):
            raise ValueError(record["label"] + ": 交叉汇总与原始证据不符。")
    return values


def factorial(out, runs_only=False):
    root = SUPPORT
    cases = expected_study_cases(out)
    hashes = source_hashes(root)
    bases = {
        c["question"]: c
        for c in cases
        if c["case_id"] in ["P23_ambient_mean_1h", "P4_ambient_mean_1h"]
    }
    for base in bases.values():
        for N in (320, 640):
            verified_study_record(out, base, N, "main", hashes)
    extra = []
    for prop, geometry, selector in [("P4", "fixed", "P23"), ("P23", "measured", "P4")]:
        for N in [320, 640]:
            case = factorial_case(bases[selector], prop, geometry)
            identity = dict(hashes, factorial_program=sha256(Path(__file__)))
            extra.append(
                execute_case(
                    root, out, case, N, "main", identity, property_set=prop, geometry=geometry
                )
            )
    if runs_only:
        return
    values = factorial_cells(out, cases, hashes)
    cells = []
    for (prop, geometry, N), r in sorted(values.items()):
        if sha256(out / r["label"] / "samples.npz") != r["samples_sha256"]:
            raise RuntimeError("Factorial cell samples changed")
        if not r["diagnostics"]["all_fields_finite"] or abs(r["event_max_C"] - 0.15) > 1e-08:
            raise RuntimeError("Factorial cell invalid")
        cells.append(
            dict(
                property_set=prop,
                geometry=geometry,
                N=N,
                event_root_s=r["event_root_s"],
                event_root_h=r["event_root_h"],
                source_run=r["label"],
                source_record_sha256=sha256(out / r["label"] / "record.json"),
                samples_sha256=r["samples_sha256"],
                ver2_actual_run=True,
                crosses_72h=r["crosses_72h"],
            )
        )
    effects = []
    for N in [320, 640]:
        a = values["P23", "fixed", N]["event_root_s"]
        b = values["P4", "fixed", N]["event_root_s"]
        c = values["P23", "measured", N]["event_root_s"]
        d = values["P4", "measured", N]["event_root_s"]
        effects.append(
            dict(
                N=N,
                P23_fixed_h=a / 3600,
                P4_fixed_h=b / 3600,
                P23_measured_h=c / 3600,
                P4_measured_h=d / 3600,
                shrinkage_percent_under_P23=100 * (a - c) / a,
                shrinkage_percent_under_P4=100 * (b - d) / b,
                property_effect_fixed_s=b - a,
                property_effect_measured_s=d - c,
                geometry_effect_P23_s=c - a,
                geometry_effect_P4_s=d - b,
                additive_interaction_s=d - b - c + a,
                additive_interaction_h=(d - b - c + a) / 3600,
                total_diagonal_change_s=d - a,
                total_diagonal_percent=100 * (d - a) / a,
                additive_reconstruction_residual_s=d - a - (b - a + (c - a) + (d - b - c + a)),
            )
        )
    numerical = []
    for prop in ["P23", "P4"]:
        for geometry in ["fixed", "measured"]:
            numerical.append(
                dict(
                    property_set=prop,
                    geometry=geometry,
                    **comparison_arrays(
                        out, values[prop, geometry, 320], values[prop, geometry, 640]
                    ),
                )
            )
    grid_effects = {
        key: abs(effects[1][key] - effects[0][key])
        for key in [
            "shrinkage_percent_under_P23",
            "shrinkage_percent_under_P4",
            "additive_interaction_s",
        ]
    }
    tables = out / "tables"
    for suffix, rows in (("cells", cells), ("effects", effects), ("grid_checks", numerical)):
        write_csv(tables / f"assumption_factorial_{suffix}.csv", rows)
    report = {
        "passed": True,
        "design": (
            "2x2 property set P23/P4 by fixed/measured geometry; matched last-hour "
            "ambient, original initial state, current-dry-density boundary, inherited "
            "h/hm."
        ),
        "new_off_diagonal_runs": 4,
        "diagonal_runs_reused": 4,
        "old_version_arrays_reused": False,
        "cells": cells,
        "effects": effects,
        "numerical_checks": numerical,
        "effect_difference_between_grids": grid_effects,
        "sign_convention": (
            "Shrinking benefit is positive 100*(fixed-measured)/fixed; property and "
            "geometry additive terms are changed-minus-reference seconds. Interaction = "
            "P4moving-P4fixed-P23moving+P23fixed."
        ),
        "endpoint_interpretation": (
            "All values are equality roots/infima for strict dryness; no dry-side "
            "display ticks are used in this decomposition."
        ),
        "new_run_wall_seconds": sum((r["wall_seconds"] for r in extra)),
        "program_sha256": sha256(Path(__file__)),
    }
    write_json(out / "factorial_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def build_endpoint_rows(out, records, hashes):
    """核验样本、事件和阶段连续性，形成逐次端点表。"""
    failures = []
    endpoint_rows = []
    for r in records:
        if r["source_hashes"] != hashes or r["source_hashes_after"] != hashes:
            failures.append(r["label"] + ": source hashes differ")
        if sha256(out / r["label"] / "samples.npz") != r["samples_sha256"]:
            failures.append(r["label"] + ": samples hash differs")
        d = r["diagnostics"]
        if (
            not d["all_fields_finite"]
            or d["min_sampled_C"] <= 0
            or abs(r["event_max_C"] - 0.15) > 1e-08
        ):
            failures.append(r["label"] + ": invalid field or event")
        if (
            d["maximum_positive_radial_C_increment"] > 1e-08
            or d["maximum_positive_increment_of_max_C"] > 1e-08
        ):
            failures.append(r["label"] + ": sampled moisture monotonicity violated")
        sw = r["stage_switch_audit"]
        if sw and (not sw["continuous_to_machine_equality"]):
            failures.append(r["label"] + ": property-switch state jump")
        endpoint_rows.append(
            dict(
                case_id=r["case"]["case_id"],
                N=r["grid_intervals"],
                kind=r["run_kind"],
                question=r["case"]["question"],
                category=r["case"]["category"],
                event_root_s=r["event_root_s"],
                event_root_h=r["event_root_h"],
                event_max_C=r["event_max_C"],
                event_R_m=r["event_R_m"],
                crosses_72h=r["crosses_72h"],
                uses_radius_extrapolation=r["uses_radius_extrapolation"],
                wall_seconds=r["wall_seconds"],
                accepted_steps=r["accepted_steps"],
                function_evaluations=r["function_evaluations"],
                rtol=r["settings"]["rtol"],
                atol_T_K=r["settings"]["atol_temperature_K"],
                atol_C=r["settings"]["atol_moisture_C"],
                max_step_s=r["settings"]["max_step_s"],
                sample_rows=d["sample_rows"],
                **{
                    k: r["case"][k]
                    for k in [
                        "ambient",
                        "alpha",
                        "h_W_m2_K",
                        "hm_m_s",
                        "tail_temperature_C",
                        "tail_Croom",
                        "tail_Ceq",
                        "phase_switch_s",
                        "boundary_normalization",
                    ]
                },
            )
        )
    if failures:
        raise RuntimeError("; ".join(failures))
    return endpoint_rows


def build_numerical_checks(out, case, indexed):
    """选取配对网格与容差结果，返回数值差及干侧报告的经验裕量。"""
    cid, q = case["case_id"], case["question"]
    coarse, fine = indexed[cid, 320, "main"], indexed[cid, 640, "main"]
    numerical = []
    numerical.append(
        dict(case_id=cid, comparison="N320_vs_N640", **comparison_arrays(out, coarse, fine))
    )
    finest = indexed.get((cid, 1280, "main"), fine)
    grid_reference = fine if finest is not fine else coarse
    gridmargin = abs(finest["event_root_s"] - grid_reference["event_root_s"])
    if finest is not fine:
        numerical.append(
            dict(case_id=cid, comparison="N640_vs_N1280", **comparison_arrays(out, fine, finest))
        )
    timebasecid = cid if (cid, 640, "time_tight") in indexed else q + "_ambient_mean_1h"
    timebase = indexed[timebasecid, 640, "main"]
    tight = indexed[timebasecid, 640, "time_tight"]
    timemargin = abs(timebase["event_root_s"] - tight["event_root_s"])
    if (cid, 640, "time_tight") in indexed:
        numerical.append(
            dict(
                case_id=cid,
                comparison="N640_main_vs_time_tight",
                **comparison_arrays(out, timebase, tight),
            )
        )
    report_source = finest
    if (cid, 1280, "time_tight") in indexed:
        report_source = indexed[cid, 1280, "time_tight"]
        numerical.append(
            dict(
                case_id=cid,
                comparison="N1280_main_vs_time_tight",
                **comparison_arrays(out, finest, report_source),
            )
        )
        numerical.append(
            dict(
                case_id=cid,
                comparison="N640_tight_vs_N1280_tight",
                **comparison_arrays(out, tight, report_source),
            )
        )
        gridmargin = abs(report_source["event_root_s"] - tight["event_root_s"])
        timemargin = abs(report_source["event_root_s"] - finest["event_root_s"])
        timebasecid = cid + "_N1280_main_vs_time_tight"
    return numerical, finest, report_source, gridmargin, timemargin, timebasecid


def build_comparisons(case, indexed, finest, report_source, report):
    """计算相对基准的情景效应，并明确报告时刻使用的网格。"""
    cid = case["case_id"]
    q = case["question"]
    coarse = indexed[cid, 320, "main"]
    fine = indexed[cid, 640, "main"]
    base640 = indexed[q + "_ambient_mean_1h", 640, "main"]
    base320 = indexed[q + "_ambient_mean_1h", 320, "main"]
    d320 = 100 * (coarse["event_root_s"] / base320["event_root_s"] - 1)
    d640 = 100 * (fine["event_root_s"] / base640["event_root_s"] - 1)
    row = dict(
        case_id=cid,
        question=q,
        category=case["category"],
        root_N320_h=coarse["event_root_h"],
        root_N640_h=fine["event_root_h"],
        root_N1280_h=finest["event_root_h"] if finest is not fine else None,
        root_N1280_time_tight_h=(
            report_source["event_root_h"] if report_source is not finest else None
        ),
        delta_from_reference_N320_percent=d320,
        delta_from_reference_N640_percent=d640,
        delta_N640_s=fine["event_root_s"] - base640["event_root_s"],
        N320_N640_root_difference_s=abs(coarse["event_root_s"] - fine["event_root_s"]),
        effect_grid_difference_percentage_points=abs(d640 - d320),
        model_effect_over_adjacent_grid_difference=abs(
            fine["event_root_s"] - base640["event_root_s"]
        )
        / max(1e-30, abs(coarse["event_root_s"] - fine["event_root_s"])),
        crosses_72h=fine["crosses_72h"],
        uses_radius_extrapolation=fine["uses_radius_extrapolation"],
        report_grid_N=finest["grid_intervals"],
        report_time_h=report["reported_time_h"],
        report_max_C_unrounded=report["reported_max_C_unrounded"],
        report_margin_s=report["total_empirical_margin_s"],
    )
    return row


def build_timeseries(out, fine, cid):
    """导出一个情景的中心、表面和平均温湿过程。"""
    rows = []
    data = load_npz(out / fine["label"] / "samples.npz")
    for i, t in enumerate(data["t_s"]):
        rows.append(
            dict(
                case_id=cid,
                t_s=float(t),
                t_h=float(t / 3600),
                equality_endpoint=bool(i == len(data["t_s"]) - 1),
                **{
                    name: float(data[name][i])
                    for name in [
                        "center_C",
                        "surface_C",
                        "max_C",
                        "mean_C",
                        "minimum_C",
                        "center_T_K",
                        "surface_T_K",
                        "R_m",
                    ]
                },
            )
        )
    return rows


def summarize(study_dir, output):
    root = SUPPORT
    out = study_dir
    tables = output / "tables"
    manifest = read_json(out / "study_manifest.json")
    expected_cases = expected_study_cases(out)
    expected_plan = make_plan(expected_cases)
    plan_keys = [(p["case_id"], p["N"], p["kind"]) for p in manifest.get("plan", [])]
    expected_keys = {(p["case_id"], p["N"], p["kind"]) for p in expected_plan}
    if len(plan_keys) != 56 or len(set(plan_keys)) != 56 or set(plan_keys) != expected_keys:
        raise ValueError("主研究计划需要完整且无重复的56项情景/网格/容差组合。")
    if manifest.get("completed_run_count") != 56 or not manifest.get("protected_sources_unchanged"):
        raise RuntimeError(
            "All 56 declared runs must finish with unchanged protected sources first"
        )
    factorial = read_json(out / "factorial_report.json")
    method_checks = read_json(out / "method_checks.json")
    extra_resolution = read_json(out / "extra_resolution_manifest.json")
    if not factorial["passed"] or not method_checks["passed"] or (not extra_resolution["passed"]):
        raise RuntimeError("Factorial and independent method checks must pass")
    hashes = source_hashes(root)
    if hashes != manifest["source_hashes_before"]:
        raise RuntimeError("Run source contract changed")
    if manifest.get("source_hashes_after") != hashes:
        raise ValueError("主研究计算后的来源身份不符。")
    by_case = {case["case_id"]: case for case in expected_cases}
    records = [
        verified_study_record(out, by_case[p["case_id"]], p["N"], p["kind"], hashes)
        for p in expected_plan
    ]
    expected_extra = [
        f"{cid}__n1280__time_tight" for cid in ("P4_ambient_mean_1h", "P4_ambient_plateau")
    ]
    labels = extra_resolution.get("labels", [])
    if (
        len(labels) != 2
        or len(set(labels)) != 2
        or set(labels) != set(expected_extra)
        or extra_resolution.get("additional_run_count") != 2
        or extra_resolution.get("source_hashes_before") != hashes
        or extra_resolution.get("source_hashes_after") != hashes
    ):
        raise ValueError("细网格时间精度核验需要完整两项与同一来源身份。")
    records.extend(
        verified_study_record(out, by_case[cid], 1280, "time_tight", hashes)
        for cid in ("P4_ambient_mean_1h", "P4_ambient_plateau")
    )
    verify_factorial_report(out, factorial, expected_cases, hashes)
    indexed = {(r["case"]["case_id"], r["grid_intervals"], r["run_kind"]): r for r in records}
    endpoint_rows = build_endpoint_rows(out, records, hashes)
    numerical = []
    comparisons = []
    reports = []
    timeseries = []
    env_comparisons = []
    for case in manifest["cases"]:
        cid = case["case_id"]
        checks, finest, report_source, gridmargin, timemargin, timebasecid = build_numerical_checks(
            out, case, indexed
        )
        numerical.extend(checks)
        report = verify_report(
            root, out, report_source, gridmargin, timemargin, timebasecid, output / "report_states"
        )
        reports.append(report)
        row = build_comparisons(case, indexed, finest, report_source, report)
        comparisons.append(row)
        if case["category"] == "ambient":
            env_comparisons.append(
                dict(
                    row,
                    tail_temperature_C=case["tail_temperature_C"],
                    tail_Croom=case["tail_Croom"],
                )
            )
        timeseries.extend(build_timeseries(out, indexed[cid, 640, "main"], cid))
    for suffix, rows in (
        ("endpoints", endpoint_rows),
        ("comparisons", comparisons),
        ("numerical_checks", numerical),
        ("environment_comparison", env_comparisons),
        ("time_series", timeseries),
        ("report_ticks", reports),
    ):
        write_csv(tables / f"assumption_{suffix}.csv", rows)
    stage = []
    for N in [320, 640]:
        b = indexed["P23_ambient_mean_1h", N, "main"]
        s = indexed["P23_P1_then_P23", N, "main"]
        base = load_npz(out / b["label"] / "samples.npz")
        switched = load_npz(out / s["label"] / "samples.npz")
        for t in [1800.0, 3600.0, 5400.0, 7200.0, 9000.0, 10800.0]:
            ib = int(np.flatnonzero(base["t_s"] == t)[0])
            isw = int(np.flatnonzero(switched["t_s"] == t)[0])
            for ir in [0, 5, 10, 15, 20]:
                stage.append(
                    dict(
                        N=N,
                        t_s=t,
                        r_cm=float(base["r_m"][ir] * 100),
                        uniform_T_C=float(base["T_K"][ib, ir] - 273.15),
                        switched_T_C=float(switched["T_K"][isw, ir] - 273.15),
                        delta_T_K=float(switched["T_K"][isw, ir] - base["T_K"][ib, ir]),
                        uniform_C=float(base["C"][ib, ir]),
                        switched_C=float(switched["C"][isw, ir]),
                        delta_C=float(switched["C"][isw, ir] - base["C"][ib, ir]),
                    )
                )
    write_csv(tables / "assumption_phase_switch.csv", stage)
    bycase = {r["case_id"]: r for r in comparisons}
    alpha = []
    for q in ["P23", "P4"]:
        for scale, cid in [
            (0.5, q + "_alpha_0p5"),
            (1.0, q + "_ambient_mean_1h"),
            (1.5, q + "_alpha_1p5"),
            (2.0, q + "_alpha_2p0"),
        ]:
            alpha.append(dict(alpha=scale, **bycase[cid]))
    write_csv(tables / "assumption_alpha_response.csv", alpha)
    env = read_json(out / "environment/ambient_platform_detection.json")
    baseline_regression = []
    for q, old_root, old_report in [
        ("P23", 206926.36144106067, 57.4796),
        ("P4", 183948.83670493803, 51.097),
    ]:
        r = indexed.get(
            (q + "_ambient_mean_1h", 1280, "time_tight"),
            indexed[q + "_ambient_mean_1h", 1280, "main"],
        )
        baseline_regression.append(
            {
                "question": q,
                "current_N1280_root_h": r["event_root_h"],
                "run_kind": r["run_kind"],
                "historic_equality_root_s": old_root,
                "root_difference_s": r["event_root_s"] - old_root,
                "historic_display_time_h": old_report,
                "role": (
                    "Regression reference only; no old arrays reused; report ticks can differ "
                    "under an explicitly revised empirical margin."
                ),
            }
        )
    derivative = []
    for q in ["P23", "P4"]:
        base = indexed[q + "_ambient_mean_1h", 640, "main"]["event_root_s"]
        for coeff in ["h", "hm"]:
            minus = indexed[q + "_" + coeff + "_0p9", 640, "main"]["event_root_s"]
            plus = indexed[q + "_" + coeff + "_1p1", 640, "main"]["event_root_s"]
            derivative.append(
                dict(
                    question=q,
                    parameter=coeff,
                    central_dimensionless_sensitivity=(plus - minus) / (0.2 * base),
                    minus10_percent_change=100 * (minus / base - 1),
                    plus10_percent_change=100 * (plus / base - 1),
                    interpretation="Central finite difference across chosen ±10% interval; not an inferred parameter uncertainty.",
                )
            )
    write_csv(tables / "assumption_exchange_sensitivity.csv", derivative)
    after = source_hashes(root)
    products = {
        str(p.resolve()).replace("\\", "/"): sha256(p)
        for p in sorted(tables.glob("assumption_*.csv"))
    }
    for p in sorted((out / "environment").glob("ambient_*")):
        products[str(p.resolve()).replace("\\", "/")] = sha256(p)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": after == hashes
        and all((x["strictly_dry_on_fine_discrete_model"] for x in reports)),
        "physical_scenario_count": 26,
        "core_sensitivity_scenario_count": 24,
        "actual_full_event_run_count": 62,
        "core_sensitivity_run_count": 56,
        "additional_factorial_run_count": 4,
        "additional_resolution_run_count": 2,
        "current_runs_verified": True,
        "source_hashes_before": hashes,
        "source_hashes_after": after,
        "protected_inputs_and_shared_sources_unchanged": after == hashes,
        "runner_manifest_sha256": sha256(out / "study_manifest.json"),
        "summary_program_sha256": sha256(Path(__file__)),
        "sum_recorded_run_seconds": sum((r["wall_seconds"] for r in records))
        + factorial["new_run_wall_seconds"],
        "sum_report_extension_seconds": sum((r["wall_seconds"] for r in reports)),
        "ambient_protocol": env["selection_protocol"],
        "ambient_primary_fit": env["primary_fit"],
        "ambient_candidates": env["candidates"],
        "ambient_rule_sensitivity": env["rule_sensitivity"],
        "comparisons": comparisons,
        "numerical_checks": numerical,
        "report_ticks": reports,
        "exchange_sensitivities": derivative,
        "baseline_regression_only": baseline_regression,
        "property_geometry_factorial": factorial,
        "independent_method_checks": method_checks,
        "additional_resolution_manifest": extra_resolution,
        "crosses_72h_cases": sorted(
            set(
                [r["case_id"] for r in comparisons if r["crosses_72h"]]
                + [c["source_run"].split("__")[0] for c in factorial["cells"] if c["crosses_72h"]]
            )
        ),
        "core_sensitivity_crosses_72h_cases": [
            r["case_id"] for r in comparisons if r["crosses_72h"]
        ],
        "radius_extrapolation_cases": [
            r["case_id"] for r in comparisons if r["uses_radius_extrapolation"]
        ],
        "maximum_adjacent_320_640_root_difference_s": max(
            (r["N320_N640_root_difference_s"] for r in comparisons)
        ),
        "maximum_effect_grid_difference_percentage_points": max(
            (r["effect_grid_difference_percentage_points"] for r in comparisons)
        ),
        "sampled_validation_scope": (
            "Whole time span at 300s plus listed early/stage times and event endpoints, "
            "21 relative-radius profiles and P4 valid fixed physical radii. Diagnostic "
            "monotonicity refers to saved samples, not an interval proof."
        ),
        "limitations": [
            "All post-4h environments are assumptions; nominal50C/.05 is not supplied by the problem.",
            (
                "BIC-like ranking uses a working independent equal-variance Gaussian "
                "residual model; temporal dependence, smooth thermal approach, and model "
                "bias prevent a statistical or physical platform guarantee."
            ),
            (
                "Ceq=alpha*Croom is a bounded scenario family; alpha cannot be identified "
                "from only ambient/radius inputs without sorption or specimen moisture "
                "measurements. No probabilities or adsorption isotherm are asserted."
            ),
            (
                "P23 appendix-3 properties uniformly from time zero is the literal main "
                "interpretation; the true P1-to-P23 switch is a tested alternative."
            ),
            "h/hm inheritance and ±10% are declared choices, not measured uncertainty bounds.",
            (
                "Initial-versus-current dry-density closure changes only external moisture "
                "boundary flux; it does not validate either convention physically."
            ),
            (
                "For P4 scenarios beyond72h, the last measured radius .01198m is held "
                "constant; this continuation has no observations beyond72h."
            ),
            (
                "Event equality roots are infima for strict dryness, never attained earliest "
                "strict times. Dry-side ticks are checked in unrounded fine-grid arrays."
            ),
            (
                "Adjacent-grid/tolerance comparisons and margins are empirical, not rigorous "
                "PDE error bounds; numerical accuracy does not identify model assumptions."
            ),
            (
                "Scenario effects are compared at matched N640 and root-to-root; report "
                "ticks have separate margins and are not used to estimate effects."
            ),
        ],
        "artifact_sha256": products,
    }
    write_json(output / "summary.json", report)

    return report


REFINEMENT_PLAN = ((1280, "main"), (640, "time_tight"))


def make_plan(cases, grids=(320, 640), basic_only=False, selected=()):
    """共同网格比较与基准空间/时间细化构成56项默认计划。"""
    known = {case["case_id"] for case in cases}
    if selected and not set(selected) <= known:
        raise ValueError("情景名称不在计划中。")
    plan = []
    for case in cases:
        if selected and case["case_id"] not in selected:
            continue
        plan.extend({"case_id": case["case_id"], "N": n, "kind": "main"} for n in grids)
        if (
            not basic_only
            and case["ambient"] in ("ambient_mean_1h", "ambient_plateau")
            and case["category"] == "ambient"
        ):
            plan.extend(
                {"case_id": case["case_id"], "N": n, "kind": kind} for n, kind in REFINEMENT_PLAN
            )
    return list({(p["case_id"], p["N"], p["kind"]): p for p in plan}.values())


def run_plan(out, cases, plan, execute):
    hashes = source_hashes()
    manifest = {
        "source_hashes_before": hashes,
        "cases": cases,
        "plan": plan,
        "case_count": len(cases),
        "planned_run_count": len(plan),
        "completed_run_count": 0,
        "execution_requested": execute,
    }
    write_json(out / "study_manifest.json", manifest)
    if execute:
        by_name = {case["case_id"]: case for case in cases}
        records = [
            execute_case(SUPPORT, out, by_name[p["case_id"]], p["N"], p["kind"], hashes)
            for p in plan
        ]
        after = source_hashes()
        manifest.update(
            completed_run_count=len(records),
            source_hashes_after=after,
            protected_sources_unchanged=hashes == after,
        )
        write_json(out / "study_manifest.json", manifest)
    return manifest


def extra_resolution(out):
    manifest = read_json(out / "study_manifest.json")
    cases = [
        case
        for case in manifest["cases"]
        if case["case_id"] in ("P4_ambient_mean_1h", "P4_ambient_plateau")
    ]
    hashes = source_hashes()
    records = [execute_case(SUPPORT, out, case, 1280, "time_tight", hashes) for case in cases]
    result = {
        "passed": hashes == source_hashes(),
        "additional_run_count": len(records),
        "labels": [r["label"] for r in records],
        "source_hashes_before": hashes,
        "source_hashes_after": source_hashes(),
        "sum_recorded_run_seconds": sum(r["wall_seconds"] for r in records),
    }
    write_json(out / "extra_resolution_manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("environment", "checks", "plan", "run", "factorial", "extra", "summarize")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path, help="summarize读取的完整情景计算目录")
    parser.add_argument("--grids", type=int, nargs="+", default=[320, 640])
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--basic-only", action="store_true")
    parser.add_argument("--runs-only", action="store_true", help="factorial仅完成交叉项计算")
    args = parser.parse_args(argv)
    if args.mode == "summarize" and args.runs is None:
        parser.error("summarize需要--runs指定完整情景目录。")
    if any(n < 2 for n in args.grids):
        parser.error("网格单元数至少为2。")
    out = prepare_output(
        args.out, allow_existing=args.mode in ("run", "factorial", "extra", "checks")
    )
    initialize_models()
    if args.mode == "summarize":
        runs = args.runs.resolve()
        if runs.is_relative_to(SUPPORT):
            parser.error("计算轨迹目录应位于支撑材料之外。")
        result = summarize(runs, out)
    elif args.mode == "factorial":
        result = factorial(out, args.runs_only)
    elif args.mode == "extra":
        result = extra_resolution(out)
    else:
        environment = analyze_environment(SUPPORT / "inputs/附件/附件1.xlsx", out / "environment")
        cases = build_cases(environment)
        if args.mode == "checks":
            result = method_checks(out, cases)
        elif args.mode == "environment":
            result = environment
        else:
            plan = make_plan(cases, args.grids, args.basic_only, args.cases)
            result = run_plan(out, cases, plan, args.mode == "run")
    write_json(
        out / (args.mode + "_verification.json"),
        audit_header(
            mode=args.mode,
            identity_key="source_identity",
            result=result,
            continuous_PDE_certified=False,
        ),
    )
    print(f"{args.mode}完成：{out}")


if __name__ == "__main__":
    main()
