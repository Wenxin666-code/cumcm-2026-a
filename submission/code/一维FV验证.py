"""一维有限体积算子的制造解、守恒结构及退化验证。

输入：数据/config/model.json 中的模型系数，以及现有公共科学分区。
输出：包外新目录中的 JSON、Markdown 和四份 CSV 核验记录。

三种制造解各运行三个网格、0～1200 s 的 41 个时刻，并对最细 P4
收紧时间容差。源项由连续解析导数计算，验证范围是数值实现。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from 验证公共接口 import (
    DATA,
    origin_source_hash,
    prepare_output,
    read_json,
    science,
    sha256,
    source_identity,
    verify_data,
    write_json,
    write_csv,
)

moving = science("moving_model")
core = science("radial_core")
fixed_model = science("_vendor_host2_model")

END = 1200.0
SAMPLE_TIMES = np.linspace(0.0, END, 41)
SHRINK_RATE = 1.0e-4
HEAT_TRANSFER = 25.0
MASS_TRANSFER = 8.0e-7


def maxabs(values) -> float:
    return float(np.max(np.abs(values)))


def analytic_properties(parameters, moisture, temperature):
    """独立写出物性及连续链式导数；系数来自同一指定物理模型。"""
    density = parameters["rho_offset"] + parameters["rho_C_coefficient"] * moisture
    specific_heat = parameters["cp_offset"] + parameters["cp_fraction_coefficient"] * (
        1.0 - 1.0 / (1.0 + moisture)
    )
    conductivity = parameters["k_offset"] + parameters["k_fraction_coefficient"] * (
        1.0 - 1.0 / (1.0 + moisture)
    )
    diffusivity = (
        parameters["D_prefactor"]
        * np.exp(-parameters["D_moisture_exponent"] / moisture)
        * np.exp(-parameters["D_temperature_exponent_K"] / temperature)
    )
    conductivity_C = parameters["k_fraction_coefficient"] / (1.0 + moisture) ** 2
    diffusivity_C = diffusivity * parameters["D_moisture_exponent"] / moisture**2
    diffusivity_T = diffusivity * parameters["D_temperature_exponent_K"] / temperature**2
    return (
        density * specific_heat,
        conductivity,
        diffusivity,
        conductivity_C,
        diffusivity_C,
        diffusivity_T,
    )


class ExactManufactured:
    """偶四次多项式制造场，中心拉普拉斯极限显式计算。

    f_g(x)=1-x²+g(1-x⁴)，圆柱拉普拉斯为 -4-16g x²。
    x=r/R(t)，固定域取 R=R0，收缩域取 R=R0/(1+alpha t)。
    """

    def __init__(self, parameters, radius_m, is_moving):
        self.parameters = parameters
        self.R0 = radius_m
        self.is_moving = is_moving

    def radius(self, time_s):
        if self.is_moving:
            return self.R0 / (1.0 + SHRINK_RATE * np.asarray(time_s))
        return self.R0 + 0 * np.asarray(time_s)

    def fields(self, time_s, position):
        x = np.asarray(position)
        amplitude_T = 2 * np.exp(-time_s / 4000.0)
        amplitude_C = 0.15 * np.exp(-time_s / 3000.0)
        profile_T = 1 - x * x + 0.20 * (1 - x**4)
        profile_C = 1 - x * x + 0.35 * (1 - x**4)
        temperature = 310.0 + amplitude_T * profile_T
        moisture = 1.1 + amplitude_C * profile_C
        derivative_T_t = -amplitude_T * profile_T / 4000.0
        derivative_C_t = -amplitude_C * profile_C / 3000.0
        derivative_T_x = amplitude_T * (-2 * x - 0.80 * x**3)
        derivative_C_x = amplitude_C * (-2 * x - 1.40 * x**3)
        laplacian_T = amplitude_T * (-4 - 3.20 * x * x)
        laplacian_C = amplitude_C * (-4 - 5.60 * x * x)
        capacity, conductivity, diffusivity, k_C, D_C, D_T = analytic_properties(
            self.parameters, moisture, temperature
        )
        radius = self.radius(time_s)
        diffusion_T = (
            conductivity * laplacian_T + k_C * derivative_C_x * derivative_T_x
        ) / radius**2
        diffusion_C = (
            diffusivity * laplacian_C
            + (D_C * derivative_C_x + D_T * derivative_T_x) * derivative_C_x
        ) / radius**2
        return {
            "T": temperature,
            "C": moisture,
            "Tt": derivative_T_t,
            "Ct": derivative_C_t,
            "Tx": derivative_T_x,
            "Cx": derivative_C_x,
            "b": capacity,
            "k": conductivity,
            "D": diffusivity,
            "LT": diffusion_T,
            "LC": diffusion_C,
            "heat_source": capacity * derivative_T_t - diffusion_T,
            "water_source": derivative_C_t - diffusion_C,
        }

    def ambient(self, time_s):
        """由 -k T_x/R=h(T_s-T_a) 及水分 Robin 条件反求环境。"""
        surface = self.fields(time_s, np.array(1.0))
        radius = self.radius(time_s)
        temperature = surface["T"] + surface["k"] * surface["Tx"] / (radius * HEAT_TRANSFER)
        moisture = surface["C"] + surface["D"] * surface["Cx"] / (radius * MASS_TRANSFER)
        return float(temperature), float(moisture)


class ManufacturedAmbient:
    """把解析 Robin 环境提供给主模型的环境接口。"""

    breakpoint_s = np.inf

    def __init__(self, exact):
        self.exact = exact

    def values(self, time_s, *, right_of_breakpoint=False):
        temperature, moisture = self.exact.ambient(time_s)
        return moving.AmbientValues(temperature, moisture, False)

    def breakpoints(self, start, end):
        return np.array([start, end])


class CoreFixedModel:
    """以当前主程序 Q1 有限体积核心装配固定半径制造解。"""

    def __init__(self, contract, label, count, ambient):
        self.contract = contract
        self.label = label
        self.ambient = ambient
        self.grid = core.NodeRadialGrid(contract["initial_state"]["radius_m"], count)
        self.nr = count + 1
        self.xi = self.grid.r_m / self.grid.radius_m

    def unpack(self, state):
        return state[: self.nr], state[self.nr :]

    def rhs(self, time_s, state):
        temperature, moisture = self.unpack(state)
        properties = core.property_values(self.contract, self.label, moisture, temperature)
        ambient = self.ambient.values(time_s)
        temperature_rate = core.diffusion_rhs(
            self.grid,
            temperature,
            core.face_arithmetic(properties["k"]),
            HEAT_TRANSFER * (temperature[-1] - ambient.temperature_K),
            properties["rho"] * properties["cp"],
        )
        moisture_rate = core.diffusion_rhs(
            self.grid,
            moisture,
            core.face_arithmetic(properties["D"]),
            MASS_TRANSFER * (moisture[-1] - ambient.moisture_eq),
        )
        return np.r_[temperature_rate, moisture_rate]

    def jacobian_sparsity(self):
        model = fixed_model.FixedRadialModel(
            moving.RadialGrid.uniform(self.nr - 1, self.grid.radius_m),
            moving.ConstantAmbient(310, 1.1),
            moving.MaterialProperties.p23(),
        )
        return model.jacobian_sparsity()


def integrate(model, rhs, end, initial, times, *, tight=False, method="Radau"):
    """使用生产空间算子和严格时间容差，避免时间误差掩盖空间收敛。"""
    tolerance = np.r_[
        np.full(model.nr, 1e-12 if tight else 1e-10),
        np.full(model.nr, 1e-14 if tight else 1e-12),
    ]
    result = solve_ivp(
        rhs,
        (0, end),
        initial,
        method=method,
        rtol=2e-12 if tight else 2e-10,
        atol=tolerance,
        t_eval=times,
        jac_sparsity=model.jacobian_sparsity(),
        max_step=100.0,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result


def make_case(contract, label, count, case):
    radius = contract["initial_state"]["radius_m"]
    exact = ExactManufactured(contract["property_sets"][label], radius, case == "P4_moving")
    ambient = ManufacturedAmbient(exact)
    if case == "P1_core_fixed":
        model = CoreFixedModel(contract, label, count, ambient)
        weights = model.grid.weights_m2
        power = 1.0
    else:
        grid = moving.RadialGrid.surface_stretched(count, radius, power=2)
        properties = moving.properties_from_contract(contract, label)
        if case == "P4_moving":
            model = moving.MovingRadialModel(
                grid,
                ambient,
                properties,
                HEAT_TRANSFER,
                MASS_TRANSFER,
                moving.CallableRadius(exact.radius),
                moving.current_dry_density_boundary,
            )
        else:
            model = fixed_model.FixedRadialModel(
                grid,
                ambient,
                properties,
                HEAT_TRANSFER,
                MASS_TRANSFER,
            )
        weights, power = grid.volume_weights_m2, 2.0
    return exact, model, model.grid.r_m / radius, weights, power


def run_manufactured(contract, label, count, case, *, tight=False):
    start = perf_counter()
    exact, model, positions, weights, power = make_case(contract, label, count, case)
    initial = exact.fields(0, positions)

    def rhs(time_s, state):
        manufactured = exact.fields(time_s, positions)
        temperature, moisture = model.unpack(state)
        capacity = analytic_properties(exact.parameters, moisture, temperature)[0]
        source = np.r_[manufactured["heat_source"] / capacity, manufactured["water_source"]]
        return model.rhs(time_s, state) + source

    result = integrate(
        model,
        rhs,
        END,
        np.r_[initial["T"], initial["C"]],
        SAMPLE_TIMES,
        tight=tight,
    )
    expected = np.column_stack(
        [
            np.r_[exact.fields(t, positions)["T"], exact.fields(t, positions)["C"]]
            for t in SAMPLE_TIMES
        ]
    )
    error = result.y - expected
    error_T, error_C = error[: model.nr], error[model.nr :]
    midpoint = exact.fields(600.0, positions)
    truncation = (
        rhs(600.0, np.r_[midpoint["T"], midpoint["C"]]) - np.r_[midpoint["Tt"], midpoint["Ct"]]
    )
    surface = exact.fields(600.0, np.array(1.0))
    ambient_T, ambient_C = exact.ambient(600.0)
    radius = float(exact.radius(600.0))
    row = {
        "case": case,
        "properties": label,
        "N": count,
        "grid_power": power,
        "end_s": END,
        "R_end_m": float(exact.radius(END)),
        "J_end": float((exact.radius(END) / exact.R0) ** 2),
        "T_Linf_K": maxabs(error_T),
        "C_Linf": maxabs(error_C),
        "T_center_K": maxabs(error_T[0]),
        "C_center": maxabs(error_C[0]),
        "T_surface_K": maxabs(error_T[-1]),
        "C_surface": maxabs(error_C[-1]),
        "T_L2_weighted_K": float(np.max(np.sqrt((2 * weights / exact.R0**2) @ error_T**2))),
        "C_L2_weighted": float(np.max(np.sqrt((2 * weights / exact.R0**2) @ error_C**2))),
        "T_center_truncation_K_s": float(abs(truncation[0])),
        "C_center_truncation_s": float(abs(truncation[model.nr])),
        "T_surface_truncation_K_s": float(abs(truncation[model.nr - 1])),
        "C_surface_truncation_s": float(abs(truncation[-1])),
        "analytic_Robin_heat_residual": float(
            abs(-surface["k"] * surface["Tx"] / radius - HEAT_TRANSFER * (surface["T"] - ambient_T))
        ),
        "analytic_Robin_moisture_residual": float(
            abs(-surface["D"] * surface["Cx"] / radius - MASS_TRANSFER * (surface["C"] - ambient_C))
        ),
        "T_order": None,
        "C_order": None,
        "nfev": result.nfev,
        "njev": result.njev,
        "nlu": result.nlu,
        "tighter_time_tolerance": tight,
        "success": bool(result.success),
        "elapsed_s": perf_counter() - start,
    }
    return row, result.y


def center_and_flux_structure(contract):
    """中心二次函数极限、常值保持、外向通量符号及体积权重。"""
    radius = contract["initial_state"]["radius_m"]
    grid = core.NodeRadialGrid(radius, 40)
    quadratic = core.diffusion_rhs(grid, grid.r_m**2, np.ones(40), -2 * radius)
    constant = core.diffusion_rhs(grid, np.ones(41), np.ones(40), 0)
    outward = core.diffusion_rhs(grid, np.ones(41), np.ones(40), 0.25)
    return {
        "center_quadratic_laplacian": float(quadratic[0]),
        "center_quadratic_error": float(abs(quadratic[0] - 4)),
        "constant_zero_flux_error": maxabs(constant),
        "outward_flux_gives_negative_surface_rate": bool(outward[-1] < 0),
        "weights_sum_error": float(abs(grid.weights_m2.sum() - radius**2 / 2)),
        "weights_positive": bool(np.all(grid.weights_m2 > 0)),
        "scope": "Cylindrical Laplacian r^2=4 at centre, constant preservation and outward-flux sign.",
    }


def constant_pure_shrink(contract):
    """零交换常值场收缩至 J=0.16，检查干基水分没有虚假稀释项。"""
    radius = contract["initial_state"]["radius_m"]
    properties = moving.properties_from_contract(contract, "P4")
    geometry = moving.CallableRadius(lambda t: radius / (1 + SHRINK_RATE * np.asarray(t)))
    model = moving.MovingRadialModel(
        moving.RadialGrid.surface_stretched(40, radius),
        moving.ConstantAmbient(301.15, 2.55),
        properties,
        0,
        0,
        geometry,
    )
    initial = model.initial_state(301.15, 2.55)
    times = np.linspace(0, 15000, 31)
    result = integrate(model, model.rhs, 15000, initial, times, method="BDF")
    return {
        "end_s": 15000.0,
        "R_ratio_end": float(geometry.radius(15000) / radius),
        "J_end": float((geometry.radius(15000) / radius) ** 2),
        "T_max_change_K": maxabs(result.y[: model.nr] - initial[: model.nr, None]),
        "C_max_change": maxabs(result.y[model.nr :] - initial[model.nr :, None]),
        "maximum_rhs": max(maxabs(model.rhs(t, initial)) for t in times),
        "success": bool(result.success),
        "meaning": "C is kg water/kg dry solid; homogeneous contraction with zero exchange preserves C.",
    }


def constant_radius_degeneracy(contract):
    """对同一空间模板比较运动适配器的固定半径极限与固定域模型。"""
    radius = contract["initial_state"]["radius_m"]
    properties = moving.properties_from_contract(contract, "P4")
    ambient = moving.ConstantAmbient(315.15, 0.04)
    grid = moving.RadialGrid.surface_stretched(80, radius)
    adapter = moving.MovingRadialModel(
        grid,
        ambient,
        properties,
        HEAT_TRANSFER,
        MASS_TRANSFER,
        moving.FixedRadius(radius),
    )
    native = fixed_model.FixedRadialModel(grid, ambient, properties, HEAT_TRANSFER, MASS_TRANSFER)
    x = grid.r_m / radius
    initial = np.r_[301.15 + 5 * x * x, 0.2 + 0.9 * (1 - x * x)]
    times = np.linspace(0, 1200, 41)
    adapted_result = integrate(adapter, adapter.rhs, 1200, initial, times)
    native_result = integrate(native, native.rhs, 1200, initial, times)
    return {
        "end_s": 1200.0,
        "N": 80,
        "initial_rhs_max_difference": maxabs(adapter.rhs(0, initial) - native.rhs(0, initial)),
        "T_max_difference_K": maxabs(
            adapted_result.y[: adapter.nr] - native_result.y[: adapter.nr]
        ),
        "C_max_difference": maxabs(adapted_result.y[adapter.nr :] - native_result.y[adapter.nr :]),
        "success": bool(adapted_result.success and native_result.success),
        "scope": "Fixed-radius adapter regression on the same spatial stencil.",
    }


def fixed_operator_alignment(contract):
    """两个现有固定域核心在同网格与同状态下进行代数接口核对。"""
    radius = contract["initial_state"]["radius_m"]
    ambient = moving.ConstantAmbient(315.15, 0.04)
    rows = []
    for label in ("P1", "P23", "P4"):
        grid = moving.RadialGrid.uniform(40, radius)
        vendor = fixed_model.FixedRadialModel(
            grid,
            ambient,
            moving.properties_from_contract(contract, label),
            HEAT_TRANSFER,
            MASS_TRANSFER,
        )
        alternate = CoreFixedModel(contract, label, 40, ambient)
        x = grid.r_m / radius
        state = np.r_[301.15 + 6 * x * x, 0.25 + 1.2 * (1 - 0.8 * x * x)]
        first, second = vendor.rhs(250, state), alternate.rhs(250, state)
        rows.append(
            {
                "properties": label,
                "T_rhs_max_difference_K_s": maxabs(first[:41] - second[:41]),
                "C_rhs_max_difference_s": maxabs(first[41:] - second[41:]),
                "maximum_scaled_difference": maxabs((first - second) / (1 + abs(second))),
            }
        )
    return rows


def reference_volume_balance(contract):
    """检查 J=(R/R0)^2 的参考体积守恒与物理表面因子 2/R。"""
    radius = contract["initial_state"]["radius_m"]
    properties = moving.properties_from_contract(contract, "P4")
    ambient = moving.ConstantAmbient(315.15, 0.04)
    grid = moving.RadialGrid.surface_stretched(80, radius)
    geometry = moving.CallableRadius(lambda t: radius / (1 + SHRINK_RATE * np.asarray(t)))
    model = moving.MovingRadialModel(
        grid,
        ambient,
        properties,
        HEAT_TRANSFER,
        MASS_TRANSFER,
        geometry,
    )
    x = grid.r_m / radius
    state = np.r_[301.15 + 5 * x * x, 0.2 + 0.9 * (1 - x * x)]
    rows = []
    for time_s in (0.0, 600.0, 1200.0):
        temperature, moisture = model.unpack(state)
        rate_T, rate_C = model.unpack(model.rhs(time_s, state))
        capacity = analytic_properties(contract["property_sets"]["P4"], moisture, temperature)[0]
        weights = 2 * grid.volume_weights_m2 / radius**2
        current_radius = float(geometry.radius(time_s))
        water_storage = float(weights @ rate_C)
        water_out = 2 * MASS_TRANSFER / current_radius * (moisture[-1] - ambient.moisture_eq)
        heat_storage = float(weights @ (capacity * rate_T))
        heat_out = 2 * HEAT_TRANSFER / current_radius * (temperature[-1] - ambient.temperature_K)
        rows.append(
            {
                "t_s": time_s,
                "J": float((current_radius / radius) ** 2),
                "water_absolute_residual": abs(water_storage + water_out),
                "heat_absolute_residual": abs(heat_storage + heat_out),
                "water_scaled_residual": abs(water_storage + water_out)
                / max(abs(water_storage), abs(water_out), 1e-300),
                "heat_scaled_residual": abs(heat_storage + heat_out)
                / max(abs(heat_storage), abs(heat_out), 1e-300),
            }
        )
    return rows


REPORT_TEMPLATE = """# 一维有限体积算子验证

全部判据通过：{report[passed]}。

{report[manuscript_method]}

|检验|N|温度最大误差/K|干基含水率最大误差|温度阶|水分阶|
|---|---:|---:|---:|---:|---:|
{table}

纯收缩至 J={pure[J_end]:.4f}，温度/水分最大漂移：{pure[T_max_change_K]:.6e} K / {pure[C_max_change]:.6e}。
固定半径 1200 s 退化：温度/水分差 {fixed[T_max_difference_K]:.6e} K / {fixed[C_max_difference]:.6e}。
最细 P4 时间容差收紧 100 倍：温度/水分差占制造解误差 {temporal[T_difference_error_ratio]:.3e} / {temporal[C_difference_error_ratio]:.3e}。

x=r/R(t)，R=R0/(1+10^-4 t)，J=(R/R0)^2；固定域取 R=R0。
T*=310+2 exp(-t/4000)[1-x²+0.20(1-x⁴)]，C*=1.1+0.15 exp(-t/3000)[1-x²+0.35(1-x⁴)]。
解析源项为 S_T=(rho cp)T*_t-L_T、S_C=C*_t-L_C；物性导数按连续链式法则求出，能量源项在时间推进时除以当前 rho cp。
Robin 环境由解析表面梯度反求。参考体积方程中的 J 归一化后，内部算子系数为 1/R²，表面因子为 2/R。

{limitations}
运行入口：python code/一维FV验证.py --out <支撑包外新目录>
Python {report[environment][python]}；NumPy {report[environment][numpy]}；SciPy {report[environment][scipy]}。
运行耗时 {report[elapsed_s]:.3f} s，当前源码与数据身份记录在 JSON 中。
"""


def write_report(out, report):
    write_json(out / "operator_validation.json", report)
    finest = max(row["N"] for row in report["manufactured_solutions"])
    columns = ("case", "N", "grid_power", "T_Linf_K", "C_Linf", "T_order", "C_order")
    tables = {
        "mms": report["manufactured_solutions"],
        "summary": [
            {key: row[key] for key in columns}
            for row in report["manufactured_solutions"]
            if row["N"] == finest
        ],
        "balances": report["reference_balance"],
        "alignment": report["fixed_operator_alignment"],
    }
    for suffix, rows in tables.items():
        write_csv(out / f"operator_validation_{suffix}.csv", rows, encoding="utf-8")
    table = []
    for row in report["manufactured_solutions"]:
        order_T = "—" if row["T_order"] is None else f"{row['T_order']:.3f}"
        order_C = "—" if row["C_order"] is None else f"{row['C_order']:.3f}"
        table.append(
            f"|{row['case']}|{row['N']}|{row['T_Linf_K']:.6e}|{row['C_Linf']:.6e}|{order_T}|{order_C}|"
        )
    text = REPORT_TEMPLATE.format(
        report=report,
        table="\n".join(table),
        limitations="".join("- " + item + "\n" for item in report["limitations"]),
        pure=report["constant_pure_shrink"],
        fixed=report["constant_radius_degeneracy"],
        temporal=report["time_tolerance_check"],
    )
    (out / "operator_validation.md").write_text(text, encoding="utf-8")


# 各判据名称、字段与门槛集中列示；解析源项和结构试验本身保持独立。
SCALAR_LIMITS = {
    "time_tolerance_subdominant_below_1percent": (
        "time_tolerance_check",
        {"T_difference_error_ratio": 0.01, "C_difference_error_ratio": 0.01},
    ),
    "constant_field_no_shrink_dilution": (
        "constant_pure_shrink",
        {"T_max_change_K": 1e-10, "C_max_change": 1e-12, "maximum_rhs": 1e-14},
    ),
    "constant_radius_degeneracy": (
        "constant_radius_degeneracy",
        {"T_max_difference_K": 1e-8, "C_max_difference": 1e-10},
    ),
}
ROW_LIMITS = {
    "manufactured_Robin_alignment": (
        "manufactured_solutions",
        {"analytic_Robin_heat_residual": 1e-10, "analytic_Robin_moisture_residual": 1e-14},
    ),
    "fixed_core_vendor_alignment": (
        "fixed_operator_alignment",
        {"maximum_scaled_difference": 1e-11},
    ),
    "reference_J_flux_balance": (
        "reference_balance",
        {"water_scaled_residual": 1e-10, "heat_scaled_residual": 1e-10},
    ),
}


def evaluate_criteria(report, finest_grid):
    """按同一门槛验证逐行误差、标量误差及中心结构。"""
    rows, structure = report["manufactured_solutions"], report["structure"]
    criteria = {
        "all_manufactured_integrations_success": all(row["success"] for row in rows),
        "finest_C_and_T_orders_above_1p75": all(
            row["T_order"] > 1.75 and row["C_order"] > 1.75
            for row in rows
            if row["N"] == finest_grid
        ),
        "center_boundary_structure": (
            structure["center_quadratic_error"] < 1e-12
            and structure["constant_zero_flux_error"] == 0
            and structure["outward_flux_gives_negative_surface_rate"]
            and structure["weights_positive"]
        ),
    }
    for name, (section, limits) in SCALAR_LIMITS.items():
        criteria[name] = all(report[section][key] < limit for key, limit in limits.items())
    for name, (section, limits) in ROW_LIMITS.items():
        criteria[name] = all(
            max(row[key] for row in report[section]) < limit for key, limit in limits.items()
        )
    return {name: bool(passed) for name, passed in criteria.items()}


def run_validation(contract, grids):
    """执行全部制造解与结构用例，并按来源测试的相同门槛验收。"""
    rows = []
    last_solutions = {}
    for case, label in (("P1_core_fixed", "P1"), ("P23_fixed", "P23"), ("P4_moving", "P4")):
        previous = None
        for count in grids:
            row, state = run_manufactured(contract, label, count, case)
            if previous is not None:
                ratio = np.log(count / previous["N"])
                row["T_order"] = float(np.log(previous["T_Linf_K"] / row["T_Linf_K"]) / ratio)
                row["C_order"] = float(np.log(previous["C_Linf"] / row["C_Linf"]) / ratio)
            rows.append(row)
            previous = row
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in ("case", "N", "T_Linf_K", "C_Linf", "T_order", "C_order")
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        last_solutions[case] = state
    tight_row, tight_state = run_manufactured(contract, "P4", grids[-1], "P4_moving", tight=True)
    difference = tight_state - last_solutions["P4_moving"]
    nodes = grids[-1] + 1
    temporal = {
        "N": grids[-1],
        "T_max_difference_K": maxabs(difference[:nodes]),
        "C_max_difference": maxabs(difference[nodes:]),
        "T_difference_error_ratio": maxabs(difference[:nodes]) / rows[-1]["T_Linf_K"],
        "C_difference_error_ratio": maxabs(difference[nodes:]) / rows[-1]["C_Linf"],
        "tight_run": tight_row,
    }
    structural_tests = {
        "structure": center_and_flux_structure,
        "constant_pure_shrink": constant_pure_shrink,
        "constant_radius_degeneracy": constant_radius_degeneracy,
        "fixed_operator_alignment": fixed_operator_alignment,
        "reference_balance": reference_volume_balance,
    }
    report = {name: check(contract) for name, check in structural_tests.items()}
    report.update(manufactured_solutions=rows, time_tolerance_check=temporal)
    criteria = evaluate_criteria(report, grids[-1])
    return dict(report, passed=all(criteria.values()), criteria=criteria)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="支撑材料包外的新结果目录")
    parser.add_argument(
        "--config", type=Path, default=DATA / "config/model.json", help="模型系数 JSON"
    )
    parser.add_argument(
        "--grids", type=int, nargs="+", default=[20, 40, 80], help="至少三个递增网格，默认 20 40 80"
    )
    args = parser.parse_args(argv)
    if len(args.grids) < 3 or args.grids != sorted(set(args.grids)) or min(args.grids) < 4:
        parser.error("至少需要三个严格递增、均不少于 4 的网格。")
    out = prepare_output(args.out)
    data_check = verify_data()
    identity = source_identity([args.config])
    start = perf_counter()
    report = run_validation(read_json(args.config), args.grids)
    report.update(
        settings={
            "method": "Radau",
            "rtol": 2e-10,
            "atol_T_K": 1e-10,
            "atol_C": 1e-12,
            "max_step_s": 100,
            "output_times_s": SAMPLE_TIMES.tolist(),
            "grids": args.grids,
            "source_construction": "Continuous analytic derivatives independent of discrete RHS",
        },
        manuscript_method=(
            "在运动参考坐标中指定偶四次多项式温湿场，由解析链式法则计算变物性下的热湿源项，"
            "表面环境按 Robin 条件反求。固定 P1、固定 P23 及收缩 P4 各细化三个网格，"
            "比较 0～1200 s 内 41 个时刻的全节点误差；最细 P4 收紧时间容差。"
            "另检查零交换常值收缩、固定半径退化、两固定域核心对齐及参考体积守恒。"
        ),
        limitations=[
            "中心使用偶函数解析极限，并以圆柱 Laplacian(r²)=4 独立检查。",
            "表面是半控制体积边界节点；局部截断残差与全局解误差的收敛阶分别报告。",
            "连续源项直接使用解析导数；物性和 Robin 闭合来自同一指定模型，验证的是数值实现。",
            "J=(R/R0)² 按当前干密度归一化；干基水分没有单独的 -C Jdot/J 项。",
            "固定半径退化与两核心对齐属于共享空间模板的结构及回归检查。",
            "平滑制造解中 C 远离零，覆盖温和非线性和短时运动算子；真实干燥薄层和终点事件由其他验证任务处理。",
            "收敛阶是本次网格序列的经验观测，不是连续 PDE 的严格统一误差界。",
        ],
        data_inventory=data_check,
        current_execution=identity,
        origin_source_sha256=origin_source_hash("code/validation/validate_operators.py"),
        config_sha256=sha256(args.config),
        environment={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        elapsed_s=perf_counter() - start,
    )
    if source_identity([args.config]) != identity:
        raise RuntimeError("运行期间当前源码或输入身份发生变化，请在文件稳定后重跑。")
    write_report(out, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "criteria": report["criteria"],
                "elapsed_s": report["elapsed_s"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
