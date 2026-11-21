"""累计水分屏障：充分干燥时间上界与条件不可达性。

verify：核对既有证据、输入及来源字段。
upper：从附件和参数重新生成上界的全部区间箱，并独立逐箱复核。
lower：重新计算下屏障常数，并以90位Decimal交叉核验。
checks：检查输入与有理向外算术。

本文件只进行标量证明计算。结论限于正文声明的B0一维模型及解的
正则性条件；它既不是主PDE求解器，也不证明该PDE解的存在性。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache, partial
from pathlib import Path
import time

from 验证公共接口 import (
    DATA,
    SUPPORT,
    origin_source_hash,
    prepare_output,
    read_json,
    research,
    sha256,
    source_identity,
    verify_data,
    write_json,
    write_csv,
    run_cli_modes,
)

F = Fraction
AUDIT = DATA / "new_audit_branch/PROJECT/extension_physics_audit"
UPPER_EVIDENCE = AUDIT / "boundary/math_upper/results/mass_upper_summary.json"
LOWER_EVIDENCE = AUDIT / "boundary/math/results/mass_lower_constants.json"


@lru_cache(maxsize=1)
def arithmetic():
    """直接复用连续误差模块已有的70位有理向外算术。"""
    return research("连续PDE")


def parameter_interval(value) -> tuple[Fraction, Fraction]:
    """同时包围参数的十进制文本解释与精确binary64解释。"""
    decimal = F(str(value))
    binary = F.from_float(float(value))
    return min(decimal, binary), max(decimal, binary)


def decimal_fraction(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def interval_record(lower: Fraction, upper: Fraction, places: int = 60) -> dict:
    math = arithmetic()
    return {
        "lower": math.text(math.down(lower, places), places),
        "upper": math.text(math.up(upper, places), places),
        "outward_decimal_places": places,
    }


def require(condition: bool, message: str) -> None:
    """证明前提检查即使在Python优化模式下也保持有效。"""
    if not condition:
        raise ValueError(message)


def read_input_table(path: Path, headings, times):
    """按指定表头和完整时间网格读取一份附件，并关闭只读工作簿。"""
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(book["Sheet1"].values)
    finally:
        book.close()
    require(list(rows[0]) == headings, path.name + "表头不符")
    require([row[0] for row in rows[1:]] == list(times), path.name + "时间节点不符")
    return rows[1:]


@dataclass
class BoundInputs:
    """两类屏障共用的模型参数、原始节点及恒定尾段。"""

    config: dict
    environment: list
    radius: list
    tail_moisture: float
    tail_temperature_c: float

    @classmethod
    def load(cls) -> "BoundInputs":
        import numpy as np

        config = read_json(SUPPORT / "code/config/model.json")
        tables = []
        for filename, headings, times in (
            ("附件1.xlsx", ["时间", "温度", "水分浓度"], range(0, 14401, 60)),
            ("附件2.xlsx", ["时间", "半径"], range(0, 259201, 1800)),
        ):
            tables.append(read_input_table(SUPPORT / "inputs/附件" / filename, headings, times))
        fixed = config["fixed_domain_model"]
        moving = config["moving_domain_contract"]
        require(
            fixed["heat_equation"] == "rho(C)*cp(C)*T_t = (1/r)*d_r(r*k(C)*T_r)",
            "时间界限要求正文B0热方程",
        )
        require("latent_heat" in fixed["omitted_heat_terms"], "时间界限要求B0模型")
        require(moving["motion"] == "homogeneous_radial_shrinkage", "要求均匀径向收缩")
        require(moving["length"] == "constant_at_initial_value", "要求长度固定")
        values = np.asarray(tables[0], dtype=np.float64)
        tail = values[values[:, 0] >= 10800]
        moisture = float(np.trapezoid(tail[:, 2], tail[:, 0]) / 3600)
        temperature = float(np.trapezoid(tail[:, 1], tail[:, 0]) / 3600)
        registered = config["input_environment"]["post_observation"]
        require(moisture == registered["ambient_moisture"], "尾段含水率与配置不符")
        require(
            temperature + 273.15 == registered["temperature_degC"] + 273.15, "尾段温度与配置不符"
        )
        return cls(config, tables[0], tables[1], moisture, temperature)

    def temperature_bounds(self) -> tuple[Fraction, Fraction]:
        offset = parameter_interval(self.config["input_environment"]["temperature_to_K_offset"])
        pairs = [parameter_interval(row[1]) for row in self.environment]
        pairs.append(parameter_interval(self.tail_temperature_c))
        initial = parameter_interval(self.config["initial_state"]["temperature_K"])
        return (
            min([p[0] + offset[0] for p in pairs] + [initial[0]]),
            max([p[1] + offset[1] for p in pairs] + [initial[1]]),
        )

    def radius_bounds(self) -> tuple[Fraction, Fraction, Fraction]:
        settings = self.config["input_radius"]
        scale = parameter_interval(settings["cm_to_m_factor"])
        lower = [(F(row[0]), parameter_interval(row[1])[0] * scale[0]) for row in self.radius]
        upper = [parameter_interval(row[1])[1] * scale[1] for row in self.radius]
        initial = parameter_interval(self.config["initial_state"]["radius_m"])
        tail = parameter_interval(settings["post_observation"]["radius_m"])
        return (
            min(r for t, r in lower if t <= 14400),
            min([r for _, r in lower] + [tail[0]]),
            max(upper + [initial[1], tail[1]]),
        )

    def validate_properties(self, material: str) -> dict:
        prop = self.config["property_sets"][material]
        positive = (
            "rho_offset",
            "cp_offset",
            "k_offset",
            "D_prefactor",
            "D_moisture_exponent",
            "D_temperature_exponent_K",
        )
        nonnegative = ("rho_C_coefficient", "cp_fraction_coefficient", "k_fraction_coefficient")
        require(all(prop[k] > 0 for k in positive), material + "物性参数必须为正")
        require(all(prop[k] >= 0 for k in nonnegative), material + "物性增量必须非负")
        return prop


class UpperBarrier:
    """二次累计水分上屏障，幅度U按有理分箱的安全速率衰减。

    shape是空间形状参数a，initial/final为起止幅度。rates同时受扩散
    不等式和Robin边界不等式约束，selected向下保留40位十进制数。
    """

    H = F(".051")
    R_MAX = F(".020000000000000001")
    T_MIN = F(301)
    FINAL = F(".098")

    def __init__(self, inputs: BoundInputs, material: str, shape: Fraction):
        prop = inputs.validate_properties(material)
        self.shape = shape
        self.initial = (
            parameter_interval(inputs.config["initial_state"]["moisture_C"])[1] - self.H
        ) / (1 - shape / 2)
        self.prefactor = parameter_interval(prop["D_prefactor"])[0]
        self.moisture = parameter_interval(prop["D_moisture_exponent"])[1]
        self.thermal = parameter_interval(prop["D_temperature_exponent_K"])[1]
        h_m = parameter_interval(inputs.config["fixed_domain_model"]["h_m_m_s"])[0]
        self.geometric = 4 * shape / self.R_MAX**2
        self.boundary = 2 * h_m / self.R_MAX * (1 - shape) / (1 - shape / 2)
        self._rates = {}

    def rates(self, amplitude: Fraction):
        if amplitude not in self._rates:
            math = arithmetic()
            slope = self.H + (1 - self.shape) * amplitude
            lo, hi = math.exp_negative(self.moisture / slope + self.thermal / self.T_MIN)
            diffusion_lo, diffusion_hi = self.prefactor * lo, self.prefactor * hi
            rate_lo = min(self.geometric * diffusion_lo, self.boundary)
            rate_hi = min(self.geometric * diffusion_hi, self.boundary)
            selected = math.down(rate_lo, 40)
            require(0 < selected <= rate_lo <= rate_hi, "上屏障速率界不成立")
            self._rates[amplitude] = selected, rate_hi, diffusion_lo, diffusion_hi
        return self._rates[amplitude]

    def nodes(self, bins: int):
        z0, z1 = 1 / self.initial, 1 / self.FINAL
        return [1 / (z0 + (z1 - z0) * F(i, bins)) for i in range(bins + 1)]

    def evaluate(self, bins: int, *, keep_rows: bool = False) -> dict:
        math = arithmetic()
        nodes = self.nodes(bins)
        upper = lower = F(0)
        rows = []
        for i, (top, bottom) in enumerate(zip(nodes[:-1], nodes[1:])):
            rate_lo, _, diffusion_lo, _ = self.rates(bottom)
            _, rate_hi, _, _ = self.rates(top)
            log_lo, log_hi = math.log_positive(top / bottom)
            duration_upper = math.up(log_hi / rate_lo)
            duration_lower = math.down(log_lo / rate_hi)
            require(0 <= duration_lower <= duration_upper, "分箱时长区间不成立")
            upper += duration_upper
            lower += duration_lower
            if keep_rows:
                rows.append(
                    {
                        "bin": i,
                        "U_top_lower": math.text(math.down(top)),
                        "U_bottom_upper": math.text(math.up(bottom)),
                        "rate_selected_s_inverse": math.text(rate_lo),
                        "D_expression_lower_at_bin_bottom": math.text(math.down(diffusion_lo)),
                        "duration_upper_s": math.text(duration_upper),
                        "ideal_integral_duration_lower_s": math.text(duration_lower),
                        "cumulative_safe_duration_upper_s": math.text(upper),
                    }
                )
        return {
            "bins": bins,
            "time_lower_s": lower,
            "time_upper_s": upper,
            "relative_bracket_gap": (upper - lower) / upper,
            "rows": rows,
        }


def display_upper(result: dict) -> dict:
    math = arithmetic()
    return {
        "bins": result["bins"],
        "ideal_continuous_rate_time_lower_s": math.text(result["time_lower_s"]),
        "safe_staged_time_upper_s": math.text(result["time_upper_s"]),
        "relative_bracket_gap_upper": math.text(math.up(result["relative_bracket_gap"])),
        "safe_staged_time_upper_h": math.text(math.up(result["time_upper_s"] / 3600)),
    }


def select_upper_shape(inputs: BoundInputs, material: str):
    """保持论文的有限候选筛选；不将其解释为全局最优性证明。"""
    barriers = {}
    records = []

    def scan(shapes, bins, label):
        results = []
        for shape in shapes:
            barrier = barriers.setdefault(shape, UpperBarrier(inputs, material, shape))
            evaluated = barrier.evaluate(bins)
            results.append((evaluated["time_upper_s"], shape))
            records.append(
                {
                    "material": material,
                    "shape_a": str(shape),
                    "search": label,
                    **display_upper(evaluated),
                }
            )
        return min(results)[1]

    center = scan([F(i, 20) for i in range(1, 20)], 128, "finite coarse rational grid")
    shapes = sorted({center + F(j, 1000) for j in range(-50, 51, 5) if 0 < center + F(j, 1000) < 1})
    selected = scan(shapes, 256, "finite local rational grid")
    return barriers[selected], records


def check_upper_schedule(
    barrier: UpperBarrier, result: dict, csv_path: Path, gap_target: Fraction
) -> dict:
    """逐箱有理审查，再用100位Decimal的exp/ln独立交叉核验。"""
    math = arithmetic()
    nodes = barrier.nodes(result["final"]["bins"])
    upper = lower = F(0)
    count = 0
    minimum_margin = None
    with localcontext() as context, csv_path.open(encoding="utf-8-sig", newline="") as stream:
        context.prec = 100

        def reference_diffusion(amplitude):
            argument = (
                barrier.moisture / (barrier.H + (1 - barrier.shape) * amplitude)
                + barrier.thermal / barrier.T_MIN
            )
            return decimal_fraction(barrier.prefactor) * (-decimal_fraction(argument)).exp()

        for i, row in enumerate(csv.DictReader(stream)):
            top, bottom = nodes[i : i + 2]
            require(int(row["bin"]) == i, "分箱序号不连续")
            require(0 <= top - F(row["U_top_lower"]) < F(1, 10**70), "上端节点向外界不符")
            require(0 <= F(row["U_bottom_upper"]) - bottom < F(1, 10**70), "下端节点向外界不符")
            rate = F(row["rate_selected_s_inverse"])
            diffusion_lo = F(row["D_expression_lower_at_bin_bottom"])
            require(
                0 < rate <= min(barrier.boundary, barrier.geometric * diffusion_lo),
                "序列化速率不满足两项约束",
            )
            duration = F(row["duration_upper_s"])
            lower_duration = F(row["ideal_integral_duration_lower_s"])
            require(duration * rate >= math.log_positive(top / bottom)[1], "时长未包住向外对数上界")
            upper += duration
            lower += lower_duration
            require(upper == F(row["cumulative_safe_duration_upper_s"]), "累计时长不一致")
            reference_diffusions = [reference_diffusion(amplitude) for amplitude in (bottom, top)]
            reference_rate = min(
                decimal_fraction(barrier.geometric) * reference_diffusions[0],
                decimal_fraction(barrier.boundary),
            )
            rate_decimal = decimal_fraction(rate)
            require(rate_decimal <= reference_rate, "Decimal100速率交叉核验失败")
            margin = reference_rate - rate_decimal
            minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
            require(
                decimal_fraction(diffusion_lo) <= reference_diffusions[0],
                "Decimal100扩散下界核验失败",
            )
            log_ratio = (decimal_fraction(top) / decimal_fraction(bottom)).ln()
            top_rate = min(
                decimal_fraction(barrier.geometric) * reference_diffusions[1],
                decimal_fraction(barrier.boundary),
            )
            require(
                decimal_fraction(duration) >= log_ratio / rate_decimal, "Decimal100安全时长核验失败"
            )
            require(
                decimal_fraction(lower_duration) <= log_ratio / top_rate, "Decimal100下和核验失败"
            )
            count += 1
    require(count == result["final"]["bins"], "分箱数目不符")
    require(upper == F(result["final"]["safe_staged_time_upper_s"]), "上和不符")
    require(lower == F(result["final"]["ideal_continuous_rate_time_lower_s"]), "下和不符")
    require((upper - lower) / upper <= gap_target, "标量积分间隙超过目标")
    require(
        upper <= F(result["integer_hour_safe_time"]) * 3600 < upper + 3600, "整小时向上取整不符"
    )
    return {
        "material": result["material"],
        "bins_checked": count,
        "passed": True,
        "minimum_Decimal100_rate_margin": str(minimum_margin),
        "decimal_crosscheck_is_outward_proof": False,
    }


def compute_upper(inputs: BoundInputs, out: Path, gap_target: Fraction = F(".0001")) -> dict:
    require(0 < gap_target < 1, "标量间隙目标应在0与1之间")
    temperature_lower, _ = inputs.temperature_bounds()
    _, _, radius_upper = inputs.radius_bounds()
    environment_upper = max(parameter_interval(row[2])[1] for row in inputs.environment)
    require(
        max(environment_upper, parameter_interval(inputs.tail_moisture)[1]) < UpperBarrier.H,
        "上屏障环境上界不足",
    )
    require(temperature_lower > UpperBarrier.T_MIN, "上屏障温度下界不足")
    require(radius_upper < UpperBarrier.R_MAX, "上屏障半径上界不足")
    require(UpperBarrier.H + UpperBarrier.FINAL == F(".149") < F(".15"), "终止幅度未满足严格阈值")
    expected = read_json(UPPER_EVIDENCE)
    results, candidates, checks = [], [], []
    for material in ("P23", "P4"):
        barrier, screened = select_upper_shape(inputs, material)
        candidates.extend(screened)
        sequence = []
        bins = 512
        while True:
            evaluated = barrier.evaluate(bins)
            sequence.append(display_upper(evaluated))
            if evaluated["relative_bracket_gap"] <= gap_target:
                break
            bins *= 2
        evaluated = barrier.evaluate(bins, keep_rows=True)
        csv_path = out / (material + "_certified_bins.csv")
        write_csv(csv_path, evaluated["rows"], columns=list(evaluated["rows"][0]))
        hours = (evaluated["time_upper_s"] / 3600).__ceil__()
        record = {
            "material": material,
            "shape_a_exact": str(barrier.shape),
            "U_initial_exact_fraction": [
                str(barrier.initial.numerator),
                str(barrier.initial.denominator),
            ],
            "box_refinement": sequence,
            "final": display_upper(evaluated),
            "integer_hour_safe_time": hours,
            "bin_csv": csv_path.name,
            "bin_csv_sha256": sha256(csv_path),
        }
        old = next(r for r in expected["results"] if r["material"] == material)
        if gap_target == F(".0001"):
            for key in (
                "shape_a_exact",
                "U_initial_exact_fraction",
                "box_refinement",
                "final",
                "integer_hour_safe_time",
                "bin_csv_sha256",
            ):
                require(record[key] == old[key], "上界重算与既有证据不符：" + material + "/" + key)
        checks.append(check_upper_schedule(barrier, record, csv_path, gap_target))
        results.append(record)
        print(f"{material}：{bins}箱，上界{hours}小时，逐箱复核通过。", flush=True)
    require(candidates == expected["finite_rational_shape_search"], "有限形状搜索结果不符")
    return {
        "passed": True,
        "PDE_solved": False,
        "results": results,
        "finite_rational_shape_search": candidates,
        "schedule_checks": checks,
        "scalar_gap_target": str(gap_target),
        "matches_paper_configuration": gap_target == F(".0001"),
    }


def lower_exponential(argument: Fraction, order: int = 48):
    """下屏障的60位证书约定：有理整次幂完成后才舍入。

    上界算法在每次平方后向外舍入70位；这里保留另一证书的明确
    算术约定，使用相同的整数向外取整接口，并给出交错级数见证。
    """
    require(argument >= 0 and order % 2 == 0, "指数参数或级数阶数不符")
    reduced, halvings = argument, 0
    while reduced > F(1, 2):
        reduced /= 2
        halvings += 1
    term = total = F(1)
    alternating_term = alternating_sum = F(1)
    for j in range(1, order + 1):
        term *= reduced / j
        total += term
        alternating_term *= -reduced / j
        alternating_sum += alternating_term
    remainder = term * reduced / (order + 1) / (1 - reduced / (order + 2))
    power = 2**halvings
    math = arithmetic()
    lower = math.down((1 / (total + remainder)) ** power, 60)
    upper = math.up((1 / total) ** power, 60)
    alternate_lower = math.down(
        (alternating_sum - alternating_term * reduced / (order + 1)) ** power, 60
    )
    alternate_upper = math.up(alternating_sum**power, 60)
    require(max(lower, alternate_lower) <= min(upper, alternate_upper), "两种指数包围不相交")
    witness = {
        "x_exact": [str(argument.numerator), str(argument.denominator)],
        "range_halvings": halvings,
        "Taylor_order": order,
        "method": "positive Taylor + geometric remainder, reciprocal and power",
        "alternating_Taylor_crosscheck": interval_record(alternate_lower, alternate_upper),
        "intervals_overlap": True,
    }
    return lower, upper, witness


def compute_lower_case(inputs: BoundInputs, material: str) -> dict:
    math = arithmetic()
    prop = inputs.validate_properties(material)
    prefactor_hi = parameter_interval(prop["D_prefactor"])[1]
    argument = parameter_interval(prop["D_moisture_exponent"])[0] / F("2.55")
    argument += parameter_interval(prop["D_temperature_exponent_K"])[0] / F("323.4")
    exp_lo, exp_hi, exp_witness = lower_exponential(argument)
    diffusion = prefactor_hi * exp_lo, prefactor_hi * exp_hi
    diffusion_used = math.up(diffusion[1], 16)
    radii = (F(".02"), F(".02")) if material == "P23" else (F(".0147"), F(".0119"))
    h_m = parameter_interval(inputs.config["fixed_domain_model"]["h_m_m_s"])[1]
    shape = F(".5")
    phases = []
    for radius in radii:
        diffusion_rate = 4 * shape * diffusion_used / (radius**2 * (1 - shape / 2))
        boundary_rate = 2 * h_m / radius * (1 - shape) / (1 - shape / 2)
        rate = math.up(max(diffusion_rate, boundary_rate), 12)
        phases.append(
            {
                "R_lower_m": str(float(radius)),
                "lambda_s_inverse": math.text(rate, 60),
                "required_diffusion": interval_record(diffusion_rate, diffusion_rate),
                "required_boundary": interval_record(boundary_rate, boundary_rate),
                "both_inequalities_passed": True,
            }
        )
    exp4_lo, exp4_hi, witness4 = lower_exponential(F(phases[0]["lambda_s_inverse"]) * 14400)
    amplitude_lo, amplitude_hi = F("2.49") * exp4_lo, F("2.49") * exp4_hi
    center_lower = F(".05") + amplitude_lo
    cumulative_lower = F(".05") + (1 - shape / 2) * amplitude_lo
    tail_amplitude = math.down(cumulative_lower - F(".15"), 10)
    require(center_lower > F(".15") and tail_amplitude > 0, "下屏障未证明严格不可达")
    require(F(".15") + tail_amplitude <= cumulative_lower, "尾段初始序不成立")
    return {
        "material": material,
        "D_bound_scope": "D evaluated at the positive subsolution slope <=2.55 and actual T<=323.4K",
        "D_conservative_upper_interval": interval_record(*diffusion),
        "D_used_upper_m2_s": math.text(diffusion_used, 60),
        "D_exponential_enclosure": exp_witness,
        "phases": phases,
        "exp_minus_lambda_4h": interval_record(exp4_lo, exp4_hi),
        "four_hour_exponential_enclosure": witness4,
        "U_at4h": interval_record(amplitude_lo, amplitude_hi),
        "center_C_at4h_lower": math.text(math.down(center_lower, 60), 60),
        "all_y_M_over_y_at4h_lower": math.text(math.down(cumulative_lower, 60), 60),
        "tail_positive_amplitude_V0": math.text(tail_amplitude, 60),
        "finite_time_tail_center_lower_formula": "0.15 + V0 * exp(-lambda_tail*(t-14400))",
        "numerical_constant_inequalities_passed": True,
    }


def check_lower_decimal(inputs: BoundInputs, rows: list[dict]) -> list[dict]:
    checks = []
    with localcontext() as context:
        context.prec = 90
        for row in rows:
            pair = row["D_exponential_enclosure"]["x_exact"]
            argument = F(int(pair[0]), int(pair[1]))
            prefactor = parameter_interval(
                inputs.config["property_sets"][row["material"]]["D_prefactor"]
            )[1]
            diffusion = decimal_fraction(prefactor) * (-decimal_fraction(argument)).exp()
            enclosure = row["D_conservative_upper_interval"]
            require(
                Decimal(enclosure["lower"]) <= diffusion <= Decimal(enclosure["upper"]),
                "Decimal90扩散区间不符",
            )
            require(diffusion <= Decimal(row["D_used_upper_m2_s"]), "Decimal90扩散上界不足")
            amplitude = (
                Decimal("2.49") * (-Decimal(row["phases"][0]["lambda_s_inverse"]) * 14400).exp()
            )
            require(
                Decimal(row["U_at4h"]["lower"]) <= amplitude <= Decimal(row["U_at4h"]["upper"]),
                "Decimal90幅度区间不符",
            )
            require(
                Decimal(row["center_C_at4h_lower"]) <= Decimal(".05") + amplitude, "中心下界不足"
            )
            for phase in row["phases"]:
                rate = F(phase["lambda_s_inverse"])
                require(
                    rate
                    >= max(
                        F(phase["required_diffusion"]["upper"]),
                        F(phase["required_boundary"]["upper"]),
                    ),
                    "下屏障速率约束不成立",
                )
            checks.append({"material": row["material"], "passed": True, "precision_digits": 90})
    return checks


def compute_lower(inputs: BoundInputs) -> dict:
    config = inputs.config
    initial_lower = parameter_interval(config["initial_state"]["moisture_C"])[0]
    require(F(".05") + F("2.49") <= initial_lower, "下屏障初始序不成立")
    require(inputs.temperature_bounds()[1] < F("323.4"), "下屏障温度上界不足")
    room_lower = min(parameter_interval(row[2])[0] for row in inputs.environment)
    require(room_lower > F(".0196") and 3 * F(".0196") > F(".05"), "早期边界下界不足")
    require(parameter_interval(inputs.tail_moisture)[1] < F(".05"), "临界平台条件不符")
    early_radius, radius_lower, _ = inputs.radius_bounds()
    require(early_radius > F(".0147") and radius_lower > F(".0119"), "下屏障半径下界不足")
    require(
        parameter_interval(config["initial_state"]["radius_m"])[0] >= F(".02"), "初始半径下界不足"
    )
    critical = F(".15") / F.from_float(inputs.tail_moisture)
    classification = []
    for q in (F(1), F("1.001"), F("1.01")):
        alpha = float(q * critical)
        product = alpha * inputs.tail_moisture
        margin = F.from_float(product) - F(".15")
        require(margin >= 0, "实际binary64平台低于严格阈值")
        classification.append(
            {
                "q_exact": str(q),
                "alpha_binary64": alpha,
                "mathematical_tail_exact": str(q * F(".15")),
                "tail_product_binary64": product,
                "actual_binary64_tail_minus_exact_threshold": interval_record(margin, margin),
                "tail_not_below_exact_threshold": True,
            }
        )
    rows = [compute_lower_case(inputs, material) for material in ("P23", "P4")]
    original = read_json(LOWER_EVIDENCE)
    require(rows == original["results"], "下屏障全部科学常数与既有证据不符")
    require(
        classification == original["critical_tail_implementations"], "临界平台分类与既有证据不符"
    )
    return {
        "passed": True,
        "PDE_solved": False,
        "results": rows,
        "critical_tail_implementations": classification,
        "independent_checks": check_lower_decimal(inputs, rows),
        "all_scientific_result_fields_match": True,
        "scope": "在B0一维固定域或定长仿射收缩模型的正则性前提下，alpha大于等于临界值时，任意有限时刻中心含水率仍严格大于0.15。",
    }


def verify_stored_evidence() -> dict:
    data_check = verify_data()
    records = []
    prefix = "new_audit_branch/PROJECT/extension_physics_audit/"
    source_paths = {
        "upper": {
            "program": prefix + "boundary/math_upper/code/certify_mass_upper.py",
            "rational_arithmetic": prefix + "boundary/math_upper/code/rational_bounds.py",
        },
        "lower": {"executed_script": prefix + "boundary/code/certify_mass_lower_barrier.py"},
    }
    actual_inputs = {
        "contract": SUPPORT / "code/config/model.json",
        "ambient": SUPPORT / "inputs/附件/附件1.xlsx",
        "radius": SUPPORT / "inputs/附件/附件2.xlsx",
    }
    for kind, path in (("upper", UPPER_EVIDENCE), ("lower", LOWER_EVIDENCE)):
        summary = read_json(path)
        sources = summary["sources_sha256"]
        for name, actual_path in actual_inputs.items():
            require(sha256(actual_path) == sources[name], "时间界限输入与证据绑定不符：" + name)
        for name, original_path in source_paths[kind].items():
            require(
                origin_source_hash(original_path) == sources[name], "来源源码身份记录不符：" + name
            )
        for filename in ("_vendor_host2_model.py", "moving_model.py", "relative_temperature.py"):
            require(
                origin_source_hash("code/solvers/" + filename)
                == sources["frozen_model/" + filename],
                "来源模型身份不符",
            )
        require(summary["PDE_started"] is False, "证据的计算范围字段不符")
        if kind == "upper":
            for row in summary["results"]:
                hours = (F(row["final"]["safe_staged_time_upper_s"]) / 3600).__ceil__()
                require(hours == row["integer_hour_safe_time"], "存储整小时上界不符")
        records.append(
            {
                "kind": kind,
                "data_sha256": sha256(path),
                "stored_origin_identifiers_consistent": True,
            }
        )
    return {
        "passed": True,
        "data": data_check,
        "records": records,
        "historical_source_files_executed": False,
        "scope": "数据与来源字段读回；重新计算由upper/lower模式执行。",
    }


def input_arithmetic_checks(inputs):
    """BoundInputs.load 已核对输入；此处补充有理向外算术自检。"""
    return {"passed": True, "input_checks": True, "arithmetic_checks": arithmetic().self_checks()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "checks", "upper", "lower"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gap", default="0.0001", help="上界标量积分的相对间隙目标")
    args = parser.parse_args(argv)
    out = prepare_output(args.out)
    started = time.perf_counter()
    inputs = None if args.mode == "verify" else BoundInputs.load()
    handlers = {
        "verify": verify_stored_evidence,
        "checks": partial(input_arithmetic_checks, inputs),
        "upper": lambda: compute_upper(inputs, out, F(args.gap)),
        "lower": partial(compute_lower, inputs),
    }
    result = run_cli_modes(args.mode, handlers)
    result.update(
        mode=args.mode,
        elapsed_s=time.perf_counter() - started,
        source_identity=source_identity(),
        PDE_existence_proved=False,
    )
    write_json(out / "verification.json", result)
    print(f"{args.mode}验证完成：{out / 'verification.json'}")


if __name__ == "__main__":
    main()
