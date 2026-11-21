"""问题一温度及部分含水率的正确舍入验证。

输入：data/验证数据/certificates 中的参数、谱根、向外区间和来源记录；附件1；
      支撑材料/results/result1.xlsx；解析重建另需 python-flint 0.9.0。
输出：包外新目录中的核验报告、可选数值容器、谱根记录或解析区间。

本文件直接实现区间读取、谱根隔离和解析比较计算。数据中的旧源码
哈希说明证据来源；当前执行身份由本文件及验证公共接口单独记录。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import lzma
import math
import sys
import zipfile
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Iterator
import xml.etree.ElementTree as ElementTree

from 验证公共接口 import (
    DATA,
    SUPPORT,
    prepare_output,
    read_json,
    sha256,
    source_identity,
    verify_data,
    write_json,
    write_jsonl_gz,
    run_cli_modes,
)

sys.dont_write_bytecode = True

CERTIFICATES = DATA / "certificates"
ARCHIVE = CERTIFICATES / "archive/extension_physics_audit"
INPUT_SHA256 = "7ef32870abeef420b89560b2530ff60dfe4255917805151d89988d0311af9dd7"
COLUMNS = [
    "time_s",
    "radius_index",
    "decimal_places",
    "lower_integer",
    "upper_integer",
    "certified_bin",
    "selected_bin",
    "selected_hex",
]
EXPECTED_CERTIFIED = {"temperature": 37800, "moisture": 17357}


def require(condition: bool, explanation: str) -> None:
    """检查证据中的确定条件，失败时保留已经生成的诊断文件。"""
    if not condition:
        raise ValueError(explanation)


def worksheet_numeric_cells(path: Path) -> tuple[bytes, dict[str, str]]:
    """保留附件 XML 的十进制字面值，供有理核查和 Arb 包围共同使用。"""
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as workbook:
        raw = workbook.read("xl/worksheets/sheet1.xml")
    sheet = ElementTree.fromstring(raw)
    values = {
        cell.attrib["r"]: cell.find("s:v", namespace).text
        for cell in sheet.findall(".//s:c", namespace)
        if cell.find("s:v", namespace) is not None
    }
    return raw, values


def exact_input_checks(path: Path) -> dict:
    """核对初边值假设及纯有理谱尾界；这些条件检查本身不是存在性证明。"""
    raw, cells = worksheet_numeric_cells(path)
    config = json.loads(
        (CERTIFICATES / "archive/package/config/model.json").read_text(encoding="utf-8"),
        parse_float=Fraction,
    )
    require(
        [Fraction(cells[f"A{row}"]) for row in range(2, 243)]
        == list(map(Fraction, range(0, 14401, 60))),
        "环境节点未覆盖规定的 241 个时间",
    )
    moisture_boxes = []
    for row in range(2, 33):
        value = cells[f"C{row}"]
        variants = [Fraction(value), Fraction.from_float(float(value)), Fraction(str(float(value)))]
        moisture_boxes.append((min(variants), max(variants)))
    ambient_min = min(pair[0] for pair in moisture_boxes)
    ambient_max = max(pair[1] for pair in moisture_boxes)
    initial = config["initial_state"]["moisture_C"]
    properties = config["property_sets"]["P1"]
    problem = config["problem_bindings"]["q1"]
    require(0 < ambient_min <= ambient_max < initial, "初始含水率与正边界范围不符")
    require(
        properties["D_prefactor"] > 0
        and properties["D_moisture_exponent"] > 0
        and properties["D_temperature_exponent_K"] == 0,
        "问题一扩散系数的正性或温度独立条件不符",
    )
    require(
        problem["duration_s"] == 1800
        and problem["geometry"] == "fixed_radial"
        and config["initial_state"]["spatially_uniform"] is True,
        "问题一固定半径及均匀初值的模型绑定不符",
    )
    n, p, s = Fraction(2), Fraction(4), Fraction(1)
    require(n / p < s < min(1 + 1 / p, 2 - n / p), "初值空间的指数条件不符")
    temperatures = [Fraction(cells[f"B{row}"]) for row in range(2, 33)]
    require(temperatures[0] == 28, "初温的 XML 精确值不符")
    slopes = [(right - left) / 60 for left, right in zip(temperatures[:-1], temperatures[1:])]
    jumps = [slopes[0]] + [right - left for left, right in zip(slopes[:-1], slopes[1:])]
    require(sum(jumps, Fraction(0)) == slopes[-1], "斜率跳的伸缩和恒等式不符")
    variation = sum(map(abs, jumps), Fraction(0))
    minimum_age = min(
        Fraction(t) - knot for t in range(1, 1801) for knot in range(0, 1800, 60) if knot < t
    )
    require(minimum_age == 1, "谱尾界的最小滞后时间不符")
    diffusivity = Fraction("0.36") / (820 * 2600 * Fraction("0.02") ** 2)
    biot = 25 * Fraction("0.02") / Fraction("0.36")
    require(diffusivity == Fraction(9, 21320) and biot == Fraction(25, 18), "精确热参数不符")
    omitted = 511
    c = diffusivity * minimum_age
    d = 4 * c * (omitted + 1)
    # e > 8/3，e^d >= 1+d 给出只涉及有理数的向上尾界。
    semigroup_tail = (
        4
        * biot
        / diffusivity
        / omitted**3
        * Fraction(3, 8) ** math.floor(c * omitted**2)
        * (1 + d)
        / d
    )
    temperature_tail = variation * semigroup_tail
    require(temperature_tail < Fraction(1, 10**50), "纯有理温度尾界未达到记录阈值")
    return {
        "passed": True,
        "worksheet_xml_sha256": hashlib.sha256(raw).hexdigest(),
        "environment_knots": 241,
        "q1_segments": 30,
        "output_times_checked": 1800,
        "ambient_moisture_exact_range": [str(ambient_min), str(ambient_max)],
        "initial_moisture": str(initial),
        "initial_space": {"n": 2, "p": 4, "s": 1},
        "minimum_age_s": str(minimum_age),
        "diffusivity_exact": str(diffusivity),
        "Bi_exact": str(biot),
        "slope_jump_variation_exact": str(variation),
        "semigroup_tail_exact": str(semigroup_tail),
        "temperature_tail_exact": str(temperature_tail),
        "temperature_tail_below_1e_minus_50": True,
        "input_convention_for_rational_tail": "Worksheet XML decimal values interpreted as exact rationals",
        "finite_modes_evaluated": False,
        "existence_theorem_proved_by_this_check": False,
    }


def exact_bin4(value: Fraction) -> int:
    """精确实数按四位 HALF_UP 舍入后的整数编码。"""
    scaled = Fraction(value) * 10000
    if scaled < 0:
        return -math.floor(-scaled + Fraction(1, 2))
    return math.floor(scaled + Fraction(1, 2))


def selected_bin4(hex_value: str) -> int:
    """数值结果遵循原工作簿的最短十进制表示及 HALF_UP 规则。"""
    value = Decimal(str(float.fromhex(hex_value)))
    return int(value.quantize(Decimal(".0001"), rounding=ROUND_HALF_UP) * 10000)


def outward_decimal_interval(lower, upper, claimed_bin):
    """向外扩张有理区间，同时保留其已经唯一确定的四位舍入格。"""
    lower, upper = Fraction(lower), Fraction(upper)
    require(lower <= upper, "区间端点次序错误")
    for places in range(12, 81):
        scale = 10**places
        lower_integer = math.floor(lower * scale)
        upper_integer = math.ceil(upper * scale)
        low = Fraction(lower_integer, scale)
        high = Fraction(upper_integer, scale)
        require(low <= lower <= upper <= high, "十进制区间未向外包含原区间")
        if claimed_bin is None or exact_bin4(low) == exact_bin4(high) == claimed_bin:
            return places, lower_integer, upper_integer
    raise ValueError("80位十进制端点仍不能保留指定舍入格")


def compact_rows(path: Path) -> Iterator[dict]:
    with lzma.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == COLUMNS, "紧凑区间的列定义不符")
        yield from reader


def check_compact_intervals(path: Path, field: str) -> dict:
    claims = 0
    count = 0
    for index, row in enumerate(compact_rows(path)):
        coordinate = int(row["time_s"]), int(row["radius_index"])
        require(coordinate == (index // 21 + 1, index % 21), "查询顺序或覆盖范围不符")
        places = int(row["decimal_places"])
        require(12 <= places <= 80, "紧凑端点的十进制位数越界")
        lower = Fraction(int(row["lower_integer"]), 10**places)
        upper = Fraction(int(row["upper_integer"]), 10**places)
        require(lower <= upper, "紧凑区间端点次序错误")
        chosen = int(row["selected_bin"])
        require(selected_bin4(row["selected_hex"]) == chosen, "选定binary64与舍入格不符")
        if row["certified_bin"]:
            claim = int(row["certified_bin"])
            require(
                exact_bin4(lower) == exact_bin4(upper) == chosen == claim,
                "声称认证的区间跨越四位舍入格",
            )
            require(field != "moisture" or claim == 25500, "含水率认证格应为2.5500")
            claims += 1
        count += 1
    require(count == 37800 and claims == EXPECTED_CERTIFIED[field], "认证格数量不符")
    return {
        "rows": count,
        "certified_bins": claims,
        "selected_bin_mismatches": 0,
        "all_compact_claims_verified": True,
    }


def original_log_rows(path: Path, field: str):
    """把两种解析日志转换成共同的精确查询记录。保留端点字面值。"""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if field == "temperature":
                yield (
                    row["time_s"],
                    row["radius_index"],
                    row["lower_C"],
                    row["upper_C"],
                    row["bin_integer"],
                    row["selected_source_bin_integer"],
                    row["source_binary64_C_hex"],
                )
            else:
                yield (
                    row["time_s"],
                    row["radius_index"],
                    *row["C_interval_exact"],
                    row["conditional_unique_bin"],
                    row["selected_bin"],
                    row["selected_hex"],
                )


def compact_exact_log(source: Path, field: str, destination: Path) -> dict:
    """逐项证明包含关系，并输出与既有紧凑格式相同的无损xz文件。"""
    require(not destination.exists(), "紧凑区间输出文件已存在")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    projection = hashlib.sha256()
    histogram = {}
    max_added_width = Fraction(0)
    count = 0
    claims = 0
    for index, row in enumerate(original_log_rows(source, field)):
        t, radius_index, lower, upper, claim, chosen, hex_value = row
        require((t, radius_index) == (index // 21 + 1, index % 21), "原日志查询覆盖不符")
        require(selected_bin4(hex_value) == chosen, "原日志选定数值舍入不符")
        if claim is not None:
            require(
                exact_bin4(lower) == exact_bin4(upper) == claim == chosen,
                "原解析区间的认证格不成立",
            )
            claims += 1
        places, lo, hi = outward_decimal_interval(lower, upper, claim)
        histogram[str(places)] = histogram.get(str(places), 0) + 1
        added_width = Fraction(hi - lo, 10**places) - (Fraction(upper) - Fraction(lower))
        max_added_width = max(max_added_width, added_width)
        projection.update((json.dumps(list(row), separators=(",", ":")) + "\n").encode())
        writer.writerow(
            [t, radius_index, places, lo, hi, "" if claim is None else claim, chosen, hex_value]
        )
        count += 1
    require(count == 37800 and claims == EXPECTED_CERTIFIED[field], "原日志认证数量不符")
    raw = buffer.getvalue().encode("utf-8")
    packed = lzma.compress(raw, preset=9)
    destination.write_bytes(packed)
    require(lzma.decompress(packed) == raw, "xz编码解码不一致")
    return {
        "name": field,
        "rows": count,
        "certified_bins": claims,
        "original_log_sha256": sha256(source),
        "original_exact_projection_sha256": projection.hexdigest(),
        "path": destination.name,
        "compressed_sha256": sha256(destination),
        "uncompressed_csv_sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_bytes": len(packed),
        "decimal_places_histogram": histogram,
        "maximum_added_width_exact": str(max_added_width),
        "all_original_intervals_contained": True,
        "all_claimed_bins_preserved": True,
    }


def data_path(relative: str) -> Path:
    """只接受证书目录内现存的数据文件。"""
    require(
        isinstance(relative, str) and "\\" not in relative and ":" not in relative,
        "证书清单路径格式错误",
    )
    name = PurePosixPath(relative)
    require(
        not name.is_absolute() and name.parts and all(p not in {".", ".."} for p in name.parts),
        "证书清单路径越界",
    )
    path = (CERTIFICATES / name).resolve()
    require(
        path.is_relative_to(CERTIFICATES.resolve()) and path.is_file(), "证书数据缺失：" + relative
    )
    return path


def check_provenance_and_data() -> dict:
    """分别核对保留的数据字节与来源源码身份，不将两者混为当前执行源码。"""
    data_check = verify_data()
    source_index = read_json(DATA / "来源源码索引.json")["files"]
    manifest = read_json(CERTIFICATES / "manifest.json")
    records = manifest["files"]
    names = [r["path"] for r in records]
    require(len(names) == len(set(names)), "证书来源清单存在重复项")
    data_names = {name for name in names if not name.endswith(".py")}
    actual_names = {
        p.relative_to(CERTIFICATES).as_posix()
        for p in CERTIFICATES.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    require(actual_names == data_names, "问题一证书数据文件集合不符")
    code_count = 0
    for record in records:
        relative = record["path"]
        if relative.endswith(".py"):
            key = "certificates/" + relative
            require(
                key in source_index and source_index[key]["sha256"] == record["sha256"],
                "证书所引用的来源源码身份不符：" + key,
            )
            code_count += 1
        else:
            path = data_path(relative)
            require(
                path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"],
                "证书数据哈希不符：" + relative,
            )
    archive_index = read_json(CERTIFICATES / "archive_inventory.json")["records"]
    for record in archive_index:
        relative = record["packaged_path"]
        if relative.endswith(".py"):
            digest = source_index["certificates/" + relative]["sha256"]
        else:
            digest = sha256(data_path(relative))
        require(digest == record["packaged_sha256"], "归档来源身份不符：" + relative)
    require(sha256(data_path("archive/inputs/附件/附件1.xlsx")) == INPUT_SHA256, "原附件1身份不符")
    return {
        "data_files_checked": len(data_names),
        "origin_source_records_checked": code_count,
        "origin_archive_records_checked": len(archive_index),
        "all_bindings_verified": True,
        "data_inventory": data_check,
        "current_execution": source_identity(),
        "origin_source_files_executed": False,
    }


def check_stored_roots(field: str, index: dict) -> dict:
    """读取全部根、端点符号、导数排零和交错关系；本函数不求超越函数。"""
    count = 256 if field == "temperature" else 768
    decoded = {}
    for kind in ("J0_roots", "Robin_roots"):
        relative = f"evidence/{field}_{kind}.json.xz"
        record = next(r for r in index["records"] if r["path"] == relative)
        raw = lzma.decompress(data_path(relative).read_bytes())
        require(hashlib.sha256(raw).hexdigest() == record["decoded_sha256"], "谱根解码哈希不符")
        decoded[kind] = json.loads(raw)
    zeros = decoded["J0_roots"]["roots"]
    robin = decoded["Robin_roots"]["roots"]
    require(len(zeros) == len(robin) == count, "谱根数量不符")

    def interval_sign(pair):
        lower, upper = map(Fraction, pair)
        require(lower <= upper, "符号区间端点次序错误")
        require(lower > 0 or upper < 0, "符号区间包含零")
        return 1 if lower > 0 else -1

    previous = Fraction(0)
    for number, (zero, root) in enumerate(zip(zeros, robin), 1):
        require(zero["index"] == root["index"] == number, "谱根编号不符")
        zl, zh, rl, rh = map(Fraction, (zero["lo"], zero["hi"], root["lo"], root["hi"]))
        require(previous < rl < rh < zl < zh, "J0与Robin根未严格交错")
        require(
            interval_sign(zero["left_J0"]) * interval_sign(zero["right_J0"]) == -1,
            "J0根端点符号未反向",
        )
        interval_sign(zero["derivative_over_entire_bracket"])
        require(
            interval_sign(root["left_residual"]) * interval_sign(root["right_residual"]) == -1,
            "Robin根端点残差未反向",
        )
        interval_sign(root["derivative"])
        previous = zh
    report = {
        "J0_roots": count,
        "Robin_roots": count,
        "stored_root_sign_and_derivative_intervals_checked": True,
        "stored_interlacing_order_checked": True,
        "transcendental_Arb_evaluations_rerun": False,
    }
    if not index.get("gap_projections"):
        report.update(
            full_gap_records_in_package=False,
            complete_gap_coverage_rechecked=False,
            source_zero_free_leaf_count=decoded["J0_roots"]["zero_free_intervals"],
        )
        return report
    gap = next(
        r
        for r in index["gap_projections"]
        if r["path"] == f"evidence/{field}_zero_free_gaps.csv.xz"
    )
    raw = lzma.decompress(data_path(gap["path"]).read_bytes())
    require(
        hashlib.sha256(raw).hexdigest() == gap["uncompressed_csv_sha256"], "排零区间解码哈希不符"
    )
    leaves = list(csv.DictReader(io.StringIO(raw.decode())))
    require(len(leaves) == gap["count"], "排零区间数量不符")
    position = 0
    previous = Fraction(0)
    for zero in zeros:
        target = Fraction(zero["lo"])
        while previous < target:
            require(position < len(leaves), "排零区间出现缺口")
            leaf = leaves[position]
            lower, upper = map(Fraction, (leaf["lo"], leaf["hi"]))
            require(
                lower == previous and lower < upper <= target and int(leaf["J0_sign"]) in (-1, 1),
                "排零覆盖或符号记录错误",
            )
            previous = upper
            position += 1
        previous = Fraction(zero["hi"])
    require(position == len(leaves), "存在多余排零区间")
    report.update(exact_gap_leaves=position, complete_stored_gap_coverage=True)
    return report


def verify_evidence() -> dict:
    report = check_provenance_and_data()
    receipt = read_json(CERTIFICATES / "evidence/conversion_receipt.json")
    require(
        receipt["all_intervals_examined"] == 75600
        and not receipt["mathematical_certification_rerun"],
        "紧凑转换收据的范围不符",
    )
    intervals = {}
    for record in receipt["records"]:
        require(
            record["all_original_intervals_contained"] and record["all_claimed_bins_preserved"],
            "紧凑转换未保留包含关系及认证格",
        )
        path = data_path(record["path"])
        require(sha256(path) == record["compressed_sha256"], "紧凑文件身份不符")
        raw = lzma.decompress(path.read_bytes())
        require(
            hashlib.sha256(raw).hexdigest() == record["uncompressed_csv_sha256"], "紧凑CSV身份不符"
        )
        intervals[record["name"]] = check_compact_intervals(path, record["name"])
    root_index = read_json(CERTIFICATES / "evidence/root_encoding.json")
    temperature_acceptance = read_json(
        ARCHIVE / "q1_temperature_certificate_01/root_acceptance_01/acceptance.json"
    )
    moisture_acceptance = read_json(
        ARCHIVE / "q1_C_linear_comparison_01/root_acceptance_01/acceptance.json"
    )
    require(
        temperature_acceptance["accepted"] and temperature_acceptance["certified_values"] == 37800,
        "温度来源验收范围不符",
    )
    require(
        moisture_acceptance["accepted"]
        and moisture_acceptance["correctly_rounded_required_C_values"] == 17357
        and not moisture_acceptance["all_Q1_C_values_certified"],
        "含水率来源验收范围不符",
    )
    report.update(
        passed=True,
        intervals=intervals,
        root_records={name: check_stored_roots(name, root_index) for name in EXPECTED_CERTIFIED},
        accepted_Q1_T_bins=37800,
        accepted_Q1_C_bins=17357,
        remaining_unproved_Q1_C_bins=20443,
        all_questions_certified=False,
        mathematical_certificate_regenerated=False,
        PDE_solved=False,
    )
    return report


@dataclass
class SelectedFields:
    """已选数值场，温度以摄氏度保存，避免读回时混淆单位。"""

    times_s: object
    radii_m: object
    temperature_C: object
    moisture: object


def recover_selected_fields(out: Path) -> tuple[SelectedFields, dict]:
    import numpy as np
    import openpyxl

    workbook = SUPPORT / "results/result1.xlsx"
    selection = read_json(CERTIFICATES / "archive/work/reviews/formal_result_source_selection.json")
    require(
        sha256(workbook) == selection["workbooks"]["result1.xlsx"]["sha256"], "结果1工作簿身份不符"
    )
    arrays = {}
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        for field, sheet_name in (("temperature", "温度"), ("moisture", "水分浓度")):
            array = np.empty((1801, 21), dtype=np.float64)
            array[0] = 28.0 if field == "temperature" else 2.55
            sheet = list(book[sheet_name].values)
            records = list(compact_rows(CERTIFICATES / f"evidence/{field}_outward.csv.xz"))
            require(len(sheet) == 1801 and len(records) == 37800, "已选数值场或工作簿形状不符")
            for index, row in enumerate(records):
                t, j = int(row["time_s"]), int(row["radius_index"])
                require((t, j) == (index // 21 + 1, index % 21), "数值恢复坐标不符")
                value = float.fromhex(row["selected_hex"])
                require(value.hex() == row["selected_hex"], "binary64精确恢复失败")
                chosen = selected_bin4(row["selected_hex"])
                cell = int(Decimal(str(sheet[t][j + 1])) * 10000)
                require(
                    chosen == int(row["selected_bin"]) == cell and float(sheet[t][0]) == t,
                    f"选定数值与工作簿不符：{field}, t={t}, j={j}",
                )
                array[t, j] = value
            arrays[field] = array
    finally:
        book.close()
    temperature_K = arrays["temperature"] + 273.15
    require(
        np.array_equal(temperature_K - 273.15, arrays["temperature"]), "温标往返改变binary64数值"
    )
    fields = SelectedFields(
        np.arange(1801, dtype=float),
        np.linspace(0.0, 0.02, 21),
        arrays["temperature"],
        arrays["moisture"],
    )
    path = out / "selected_fields_from_certificate.npz"
    np.savez_compressed(
        path, t_s=fields.times_s, r_m=fields.radii_m, T_K=temperature_K, C=fields.moisture
    )
    with np.load(path, allow_pickle=False) as saved:
        require(
            np.array_equal(saved["T_K"] - 273.15, fields.temperature_C)
            and np.array_equal(saved["C"], fields.moisture),
            "恢复容器写入后读回不符",
        )
    report = {
        "passed": True,
        "selected_binary64_values_checked": 75600,
        "all_bins_match_result1": True,
        "workbook_sha256": sha256(workbook),
        "new_container_sha256": sha256(path),
        "original_solution_npz_recreated": False,
        "PDE_solved": False,
    }
    write_json(out / "selected_projection_check.json", report)
    return fields, report


class ArbArithmetic:
    """256位Arb算术与精确有理端点；SciPy仅提供待验证的数值初猜。"""

    def __init__(self, runtime: Path | None = None):
        if runtime is not None:
            runtime = runtime.resolve()
            require(runtime.is_dir(), "指定Arb导入目录不存在")
            sys.path.insert(0, str(runtime))
        import flint
        import numpy
        import scipy
        from scipy import optimize, special

        require(flint.__version__ == "0.9.0", "解析证书要求python-flint 0.9.0")
        if runtime is not None:
            require(
                Path(flint.__file__).resolve().is_relative_to(runtime),
                "实际Arb导入位置与指定目录不符",
            )
        flint.ctx.prec = 256
        self.arb = flint.arb
        self.special = special
        self.optimize = optimize
        self.runtime = {
            "python_flint": flint.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "precision_bits": 256,
            "actual_imports": [
                {
                    "package": Path(m.__file__).parent.name,
                    "file": Path(m.__file__).name,
                    "sha256": sha256(Path(m.__file__)),
                }
                for m in (flint, numpy, scipy)
            ],
        }

    def number(self, value):
        rational = Fraction(value)
        return self.arb(str(rational.numerator)) / self.arb(str(rational.denominator))

    @staticmethod
    def exact_endpoint(value) -> Fraction:
        require(value.is_finite() and value.is_exact(), "Arb端点必须有限且精确")
        mantissa, exponent = map(int, value.man_exp())
        if exponent >= 0:
            return Fraction(mantissa * (1 << exponent))
        return Fraction(mantissa, 1 << -exponent)

    def endpoints(self, ball) -> tuple[Fraction, Fraction]:
        require(ball.is_finite(), "Arb区间不是有限数")
        return self.exact_endpoint(ball.lower()), self.exact_endpoint(ball.upper())

    def strings(self, ball) -> list[str]:
        return list(map(str, self.endpoints(ball)))

    def interval(self, lower, upper):
        lower, upper = Fraction(lower), Fraction(upper)
        require(lower <= upper, "Arb输入端点次序错误")
        return self.number(lower).union(self.number(upper))

    def hull(self, values):
        rational_values = list(map(Fraction, values))
        return self.interval(min(rational_values), max(rational_values))

    @staticmethod
    def sign(ball) -> int:
        if ball > 0:
            return 1
        if ball < 0:
            return -1
        return 0


class RobinSpectrum:
    """J0零点隔离、完整间隙排零和Robin特征根的身份验证。"""

    def __init__(self, arithmetic: ArbArithmetic):
        self.math = arithmetic

    def bracket_guess(self, function, guess, radius=Fraction(1, 10**8)):
        center = Fraction.from_float(float(guess))
        lower, upper = center - radius, center + radius
        require(lower > 0, "初猜根区间必须位于正半轴")
        sign_lower = self.math.sign(function(self.math.number(lower)))
        sign_upper = self.math.sign(function(self.math.number(upper)))
        require(sign_lower * sign_upper == -1, "根的初始区间没有严格异号端点")
        return lower, upper, sign_lower, sign_upper

    def bisect_parameter_root(self, function, lower, upper, sign_lower, sign_upper):
        for _ in range(180):
            if upper - lower < Fraction(1, 10**45):
                break
            midpoint = (lower + upper) / 2
            middle_sign = self.math.sign(function(self.math.number(midpoint)))
            if middle_sign == sign_lower:
                lower = midpoint
            elif middle_sign == sign_upper:
                upper = midpoint
            else:
                # 参数区间内的不同真根可能落在中点两侧，只移动已严格定号的端点。
                quarter_left, quarter_right = (lower + midpoint) / 2, (midpoint + upper) / 2
                moved = False
                if self.math.sign(function(self.math.number(quarter_left))) == sign_lower:
                    lower = quarter_left
                    moved = True
                if self.math.sign(function(self.math.number(quarter_right))) == sign_upper:
                    upper = quarter_right
                    moved = True
                if not moved:
                    break
        require(
            self.math.sign(function(self.math.number(lower))) == sign_lower
            and self.math.sign(function(self.math.number(upper))) == sign_upper,
            "二分后的根区间丢失严格端点符号",
        )
        return lower, upper

    def verify_zero_free(self, lower, upper, stream, counts, depth=0):
        value = self.math.interval(lower, upper).bessel_j(0)
        if self.math.sign(value):
            stream.write(
                json.dumps({"lo": str(lower), "hi": str(upper), "J0": self.math.strings(value)})
                + "\n"
            )
            counts["zero_free_intervals"] += 1
            return
        require(depth < 100, "J0排零自适应细分达到深度上限")
        midpoint = (lower + upper) / 2
        self.verify_zero_free(lower, midpoint, stream, counts, depth + 1)
        self.verify_zero_free(midpoint, upper, stream, counts, depth + 1)

    def reconstruct(self, count: int, biot, out: Path):
        guesses = self.math.special.jn_zeros(0, count)
        zero_records = []
        zero_intervals = []
        counts = {"zero_free_intervals": 0, "simple_J0_roots": 0}
        previous = Fraction(0)
        with gzip.open(out / "zero_free_intervals.jsonl.gz", "wt", encoding="utf-8") as stream:
            for number, guess in enumerate(guesses, 1):
                lower, upper, _, _ = self.bracket_guess(lambda z: z.bessel_j(0), guess)
                derivative = -self.math.interval(lower, upper).bessel_j(1)
                require(
                    self.math.sign(derivative) != 0 and previous < lower, "J0根简单性或区间次序错误"
                )
                self.verify_zero_free(previous, lower, stream, counts)
                zero_records.append(
                    {
                        "index": number,
                        "lo": str(lower),
                        "hi": str(upper),
                        "left_J0": self.math.strings(self.math.number(lower).bessel_j(0)),
                        "right_J0": self.math.strings(self.math.number(upper).bessel_j(0)),
                        "derivative_over_entire_bracket": self.math.strings(derivative),
                    }
                )
                zero_intervals.append((lower, upper))
                previous = upper
                counts["simple_J0_roots"] += 1
        write_json(
            out / "J0_roots.json",
            {
                "roots": zero_records,
                **counts,
                "coverage_start": "0",
                "coverage_end": str(previous),
                "every_gap_excluded_by_interval_evaluation": True,
            },
        )
        result = []
        records = []
        residual = lambda z: z * z.bessel_j(1) - biot * z.bessel_j(0)
        for index, right_guess in enumerate(guesses):
            left_guess = 0.0 if index == 0 else guesses[index - 1]
            guess = self.math.optimize.brentq(
                lambda z: z * self.math.special.j1(z) - float(biot.mid()) * self.math.special.j0(z),
                left_guess,
                right_guess,
            )
            lower, upper, sign_lower, sign_upper = self.bracket_guess(residual, guess)
            lower, upper = self.bisect_parameter_root(
                residual, lower, upper, sign_lower, sign_upper
            )
            require(
                (lower > 0 if index == 0 else lower > zero_intervals[index - 1][1])
                and upper < zero_intervals[index][0],
                "Robin根未处于相邻J0根之间",
            )
            ball = self.math.interval(lower, upper)
            derivative = ball * ball.bessel_j(0) + biot * ball.bessel_j(1)
            require(self.math.sign(derivative) != 0, "Robin根导数区间包含零")
            result.append(ball)
            records.append(
                {
                    "index": index + 1,
                    "lo": str(lower),
                    "hi": str(upper),
                    "left_residual": self.math.strings(residual(self.math.number(lower))),
                    "right_residual": self.math.strings(residual(self.math.number(upper))),
                    "derivative": self.math.strings(derivative),
                    "strictly_between_consecutive_certified_J0_roots": True,
                }
            )
        write_json(
            out / "Robin_roots.json",
            {
                "roots": records,
                "Bi": self.math.strings(biot),
                "root_count": count,
                "identity_argument": "H=lambda J1/J0 has Hprime=lambda+H^2/lambda>0 between consecutive J0 zeros.",
            },
        )
        return result


def temperature_input_boxes(arithmetic: ArbArithmetic, path: Path):
    """同时包围工作簿XML十进制、binary64及温标平移解释。"""
    raw, values = worksheet_numeric_cells(path)
    knots = []
    temperatures = []
    records = []
    for row in range(2, 243):
        t = Fraction(values[f"A{row}"])
        xml_value = values[f"B{row}"]
        numeric = float(xml_value)
        admitted = [
            Fraction(xml_value),
            Fraction.from_float(numeric),
            Fraction(str(numeric)),
            Fraction.from_float(numeric + 273.15) - Fraction.from_float(273.15),
        ]
        require(t == 60 * (row - 2), "环境时间节点不符")
        knots.append(t)
        temperatures.append(arithmetic.hull(admitted))
        records.append(
            {
                "row": row,
                "time": str(t),
                "XML_temperature_C": xml_value,
                "binary64_hex": numeric.hex(),
                "admitted_exact_C": sorted(set(map(str, admitted))),
                "enclosure_C": arithmetic.strings(temperatures[-1]),
            }
        )
    require(temperatures[0].is_exact() and temperatures[0] == 28, "初始环境温度不等于28摄氏度")
    return knots, temperatures, {"raw_xml_sha256": hashlib.sha256(raw).hexdigest(), "rows": records}


class MoistureComparison:
    """Kirchhoff势的I0、erfc启动层及Robin热比较场所需常数与运算。"""

    def __init__(self, arithmetic: ArbArithmetic):
        self.math = arithmetic
        self.parameters = read_json(
            ARCHIVE / "q1_C_interior_certificate_01/actual_02/parameters.json"
        )
        boxes = self.parameters
        radius, transfer, prefactor, initial, exponent = (
            arithmetic.interval(*boxes[name]) for name in ("R", "hm", "D_prefactor", "C0", "q")
        )
        self.beta = arithmetic.endpoints(
            radius * transfer / prefactor / (-exponent / initial).exp()
        )[0]
        self.max_diffusivity = Fraction(boxes["a_max"])
        self.boundary_amplitude = Fraction(boxes["A_max"])

    def choose_lambda(self, x: float, time_s: int) -> Fraction:
        diffusion_time = float(self.max_diffusivity) * time_s
        beta = float(self.beta)

        def objective(value):
            return (
                diffusion_time * value * value
                + math.log(float(self.math.special.i0e(value * x)))
                + value * x
                - value
                - math.log(
                    value * float(self.math.special.i1e(value))
                    + beta * float(self.math.special.i0e(value))
                )
            )

        result = self.math.optimize.minimize_scalar(
            objective,
            bounds=(0.01, 128.0),
            method="bounded",
            options={"xatol": 1e-10},
        )
        return Fraction(round(float(result.x) * 2**20), 2**20)

    def bessel_drop_upper(self, x: Fraction, time_s: int, lambda_value: Fraction) -> Fraction:
        value = self.math.number(lambda_value)
        drop = (
            self.math.number(self.boundary_amplitude)
            * (self.math.number(self.max_diffusivity * time_s) * value * value).exp()
            * (value * self.math.number(x)).bessel_i(0)
            / (value * value.bessel_i(1) + self.math.number(self.beta) * value.bessel_i(0))
        )
        require(drop > 0, "I0比较场的势降必须为正")
        return self.math.endpoints(drop)[1]

    def erfc_profile(self, time_s: int):
        """在一个时刻构造并检查erfc启动层的公共时间系数。"""
        arb = self.math.arb
        diffusivity = self.math.number(self.max_diffusivity)
        time_radius = (
            diffusivity * (arb.pi() * time_s).sqrt()
            + (diffusivity * diffusivity * arb.pi() * time_s + diffusivity).sqrt()
        )
        coefficient = self.math.endpoints(time_radius * time_radius)[1] * (1 + Fraction(1, 2**180))
        residual = (
            diffusivity
            - self.math.number(coefficient)
            + 2 * diffusivity * (arb.pi() * self.math.number(coefficient * time_s)).sqrt()
        )
        require(self.math.endpoints(residual)[1] < 0, "erfc时间屏障残差未严格为负")
        diffusion_time = self.math.number(coefficient * time_s)
        amplitude = (
            2 * self.math.number(self.boundary_amplitude) * (diffusion_time / arb.pi()).sqrt()
        )
        return coefficient, diffusion_time, amplitude

    def erfc_drop_upper(self, x: Fraction, time_s: int, profile) -> Fraction:
        coefficient, diffusion_time, amplitude = profile
        distance = (1 - x * x) / 2
        argument = self.math.number(distance) / (2 * diffusion_time.sqrt())
        drop = (
            amplitude * (-self.math.number(distance * distance / (4 * coefficient * time_s))).exp()
            - self.math.number(self.boundary_amplitude * distance) * argument.erfc()
        )
        return self.math.endpoints(drop)[1]


def temperature_parameters(arithmetic: ArbArithmetic):
    recorded = read_json(ARCHIVE / "q1_temperature_certificate_01/actual_01/parameters.json")
    radius = arithmetic.hull([Fraction(1, 50), Fraction.from_float(0.02)])
    conductivity = arithmetic.hull([Fraction(9, 25), Fraction.from_float(0.36)])
    diffusivity = (
        (conductivity / (820 * 2600 * radius * radius))
        .union(arithmetic.number(Fraction(9, 21320)))
        .union(arithmetic.arb((0.36 / (820 * 2600)) / 0.02**2))
    )
    biot = (
        (25 * radius / conductivity)
        .union(arithmetic.number(Fraction(25, 18)))
        .union(arithmetic.arb(25 * 0.02 / 0.36))
    )
    require(
        arithmetic.strings(diffusivity) == recorded["a"]
        and arithmetic.strings(biot) == recorded["Bi"],
        "重建的无量纲温度参数包围不符",
    )
    return diffusivity, biot, recorded


def temperature_log(arithmetic, eigenvalues, diffusivity, biot, parameters, inputs, fields, output):
    """逐秒递推Robin谱有限和，叠加解析无限尾界，输出37,800个严格区间。"""
    arb = arithmetic.arb
    radii = fields.radii_m
    positions = [
        arithmetic.hull(
            [
                Fraction(i, 20),
                Fraction.from_float(float(radii[i])) / Fraction.from_float(0.02),
                Fraction.from_float(float(radii[i] / 0.02)),
            ]
        )
        for i in range(21)
    ]
    require(
        list(map(arithmetic.strings, positions)) == parameters["x"], "径向位置包围与证书参数不符"
    )
    _, temperatures, input_record = temperature_input_boxes(arithmetic, inputs)
    write_json(output.parent / "input_semantics.json", input_record)
    rates = [diffusivity * value * value for value in eigenvalues]
    decay = [(-value).exp() for value in rates]
    coefficients = [
        2
        * biot
        / (value.bessel_j(0) * (value * value + biot * biot))
        / (diffusivity * value * value)
        for value in eigenvalues
    ]
    basis = [[(value * x).bessel_j(0) for value in eigenvalues] for x in positions]
    particular = [(x * x - 1 - 2 / biot) / (4 * diffusivity) for x in positions]
    slopes = [(temperatures[j + 1] - temperatures[j]) / 60 for j in range(30)]
    jumps = [slopes[0]] + [slopes[j] - slopes[j - 1] for j in range(1, 30)]
    jump_norm = sum((abs(value).upper() for value in jumps), arb(0))
    lower_diffusivity = diffusivity.lower()
    upper_biot = biot.upper()
    omitted_root_lower = arb(511)
    ratio = (-4 * lower_diffusivity * (omitted_root_lower + 1)).exp()
    tail = (
        (4 * upper_biot / lower_diffusivity)
        * (-lower_diffusivity * omitted_root_lower * omitted_root_lower).exp()
        / (omitted_root_lower**3 * (1 - ratio))
        * jump_norm
    ).upper()

    def records():
        transient = [arb(0) for _ in eigenvalues]
        for interval_index in range(30):
            transient = [
                value + jumps[interval_index] * coefficient
                for value, coefficient in zip(transient, coefficients)
            ]
            for offset in range(1, 61):
                time_s = interval_index * 60 + offset
                transient = [value * factor for value, factor in zip(transient, decay)]
                ambient = temperatures[interval_index] + slopes[interval_index] * offset
                for radius_index in range(21):
                    value = (
                        ambient
                        + slopes[interval_index] * particular[radius_index]
                        + sum((v * w for v, w in zip(transient, basis[radius_index])), arb(0))
                    )
                    lower, upper = arithmetic.endpoints(value + arb(0, tail))
                    low_bin, high_bin = exact_bin4(lower), exact_bin4(upper)
                    selected = float(fields.temperature_C[time_s, radius_index])
                    row = {
                        "time_s": time_s,
                        "radius_index": radius_index,
                        "nominal_radius_m": str(Fraction(radius_index, 1000)),
                        "lower_C": str(lower),
                        "upper_C": str(upper),
                        "correct_rounding_bin_unique": low_bin == high_bin,
                        "bin_integer": low_bin if low_bin == high_bin else None,
                        "selected_source_bin_integer": selected_bin4(selected.hex()),
                        "source_binary64_C_hex": selected.hex(),
                    }
                    yield row

    write_jsonl_gz(output, records(), separators=(",", ":"))


def moisture_log(comparison: MoistureComparison, eigenvalues, fields, output):
    """合并I0、erfc和线性Robin下解，判定含水率的2.5500舍入格。"""
    arithmetic = comparison.math
    arb = arithmetic.arb
    parameters = comparison.parameters
    beta = arithmetic.number(comparison.beta)
    diffusivity = arithmetic.number(comparison.max_diffusivity)
    boundary_amplitude = arithmetic.number(comparison.boundary_amplitude)
    omitted_root_lower = Fraction(1535)
    tail_upper = arithmetic.endpoints(
        4
        * beta
        / arithmetic.number(omitted_root_lower)
        * (-diffusivity * arithmetic.number(omitted_root_lower * omitted_root_lower)).exp()
        / (1 - (-4 * diffusivity * arithmetic.number(omitted_root_lower + 1)).exp())
    )[1]
    positions = [arithmetic.interval(*pair) for pair in parameters["x_intervals"]]
    decay = [(-diffusivity * root * root).exp() for root in eigenvalues]
    coefficients = [
        2 * beta / (root.bessel_j(0) * (root * root + beta * beta)) for root in eigenvalues
    ]
    basis = [[(root * x).bessel_j(0) for root in eigenvalues] for x in positions]
    initial_lower, initial_upper = map(Fraction, parameters["C0"])
    ambient_minimum = Fraction(parameters["Ceq_min"])
    rounding_boundary = Fraction("2.54995")
    slope_lower = Fraction(parameters["positive_slope_lower_at_rounding_boundary"])
    threshold_potential_upper = Fraction(parameters["F_rounding_lower_boundary"][1])
    position_boxes = [list(map(Fraction, pair)) for pair in parameters["x_intervals"]]

    def records():
        state = coefficients[:]
        for time_s in range(1, 1801):
            state = [value * factor for value, factor in zip(state, decay)]
            erfc_profile = comparison.erfc_profile(time_s)
            for radius_index in range(21):
                x_upper = position_boxes[radius_index][1]
                erfc_drop = comparison.erfc_drop_upper(x_upper, time_s, erfc_profile)
                lambda_value = comparison.choose_lambda(float(x_upper), time_s)
                bessel_drop = comparison.bessel_drop_upper(x_upper, time_s, lambda_value)
                previous_drop = min(bessel_drop, erfc_drop)
                finite_sum = sum(
                    (value * weight for value, weight in zip(state, basis[radius_index])), arb(0)
                )
                spectral_sum = finite_sum + arb(0, arithmetic.number(tail_upper))
                linear_drop = boundary_amplitude / beta * (1 - spectral_sum)
                linear_upper = arithmetic.endpoints(linear_drop)[1]
                drop_upper = min(linear_upper, previous_drop)
                require(linear_upper >= 0 and erfc_drop >= 0, "含水率比较场出现负势降上界")
                unique = -drop_upper > threshold_potential_upper
                lower = (
                    max(rounding_boundary, initial_lower - drop_upper / slope_lower)
                    if unique
                    else ambient_minimum
                )
                selected = float(fields.moisture[time_s, radius_index])
                row = {
                    "time_s": time_s,
                    "radius_index": radius_index,
                    "linear_drop_interval_exact": arithmetic.strings(linear_drop),
                    "combined_drop_upper_exact": str(drop_upper),
                    "C_interval_exact": [str(lower), str(initial_upper)],
                    "conditional_unique_bin": 25500 if unique else None,
                    "selected_bin": exact_bin4(Fraction.from_float(selected)),
                    "selected_hex": selected.hex(),
                }
                yield row

    write_jsonl_gz(output, records())


def reconstruct(mode: str, out: Path, runtime: Path | None, inputs: Path, fields=None) -> dict:
    arithmetic = ArbArithmetic(runtime)
    comparison = MoistureComparison(arithmetic)
    require(
        arithmetic.endpoints(arithmetic.number(Fraction(1, 2))) == (Fraction(1, 2), Fraction(1, 2)),
        "有理数Arb表示检查失败",
    )
    diffusivity, biot, parameters = temperature_parameters(arithmetic)
    report = {
        "passed": True,
        "runtime": arithmetic.runtime,
        "PDE_solved": False,
        "current_execution": source_identity(),
    }
    if mode == "runtime-check":
        report.update(
            kernel_load_smoke_passed=True,
            parameter_boxes_match=True,
            complete_root_reconstruction=False,
            field_intervals_recomputed=False,
        )
        return report
    spectrum = RobinSpectrum(arithmetic)
    roots = {}
    for name, count, coefficient in (
        ("temperature", 256, biot),
        ("moisture", 768, arithmetic.number(comparison.beta)),
    ):
        destination = out / name
        destination.mkdir()
        roots[name] = spectrum.reconstruct(count, coefficient, destination)
    report.update(
        complete_root_reconstruction=True, root_counts={"temperature": 256, "moisture": 768}
    )
    if mode == "reconstruct-roots":
        report["field_intervals_recomputed"] = False
        return report
    require(fields is not None, "解析区间重建需要选定数值场")
    temperature_path = out / "temperature_intervals.jsonl.gz"
    moisture_path = out / "all_query_bounds.jsonl.gz"
    temperature_log(
        arithmetic,
        roots["temperature"],
        diffusivity,
        biot,
        parameters,
        inputs,
        fields,
        temperature_path,
    )
    moisture_log(comparison, roots["moisture"], fields, moisture_path)
    receipt = read_json(CERTIFICATES / "evidence/conversion_receipt.json")
    comparisons = []
    for name, path in (("temperature", temperature_path), ("moisture", moisture_path)):
        generated = compact_exact_log(path, name, out / (name + "_outward.csv.xz"))
        expected = next(record for record in receipt["records"] if record["name"] == name)
        exact = (
            generated["original_exact_projection_sha256"]
            == expected["original_exact_projection_sha256"]
        )
        compact = generated["compressed_sha256"] == expected["compressed_sha256"]
        comparisons.append(
            {"name": name, "exact_endpoint_projection_same": exact, "compact_bytes_same": compact}
        )
        require(exact and compact, "完整解析重建与绑定证书不同；输出保留供检查")
    report.update(
        field_intervals_recomputed=True,
        all_exact_endpoint_projections_same=True,
        comparisons=comparisons,
        new_formal_acceptance=False,
    )
    return report


def rebuild_compact_logs(temperature_path: Path, moisture_path: Path, out: Path) -> list[dict]:
    receipt = read_json(CERTIFICATES / "evidence/conversion_receipt.json")
    reports = []
    for name, path in (("temperature", temperature_path), ("moisture", moisture_path)):
        expected = next(record for record in receipt["records"] if record["name"] == name)
        require(sha256(path) == expected["original_log_sha256"], "输入精确日志与原证书身份不符")
        generated = compact_exact_log(path, name, out / Path(expected["path"]).name)
        require(
            generated["compressed_sha256"] == expected["compressed_sha256"]
            and generated["original_exact_projection_sha256"]
            == expected["original_exact_projection_sha256"],
            "紧凑证书逐项重建不符",
        )
        reports.append(generated)
    return reports


def stored_robin_roots(arithmetic: ArbArithmetic, field: str):
    """读入已绑定根区间；直接求值复核与完整谱根重建分别记录。"""
    path = CERTIFICATES / f"evidence/{field}_Robin_roots.json.xz"
    record = json.loads(lzma.decompress(path.read_bytes()))
    count = 256 if field == "temperature" else 768
    require(len(record["roots"]) == count, "直接求值所用根数量不符")
    return [arithmetic.interval(root["lo"], root["hi"]) for root in record["roots"]]


def compact_endpoints(row):
    scale = 10 ** int(row["decimal_places"])
    return Fraction(int(row["lower_integer"]), scale), Fraction(int(row["upper_integer"]), scale)


def direct_temperature_profiles(arithmetic, inputs, fields, out):
    """按完整斜率跳历史直接求和，复核温度剖面及其舍入格。"""
    arb = arithmetic.arb
    diffusivity, biot, parameters = temperature_parameters(arithmetic)
    _, environment, _ = temperature_input_boxes(arithmetic, inputs)
    temperature_roots = stored_robin_roots(arithmetic, "temperature")
    positions = [
        arithmetic.hull(
            [
                Fraction(i, 20),
                Fraction.from_float(float(fields.radii_m[i])) / Fraction.from_float(0.02),
                Fraction.from_float(float(fields.radii_m[i] / 0.02)),
            ]
        )
        for i in range(21)
    ]
    require(list(map(arithmetic.strings, positions)) == parameters["x"], "直接求值的位置包围不符")
    slopes = [(environment[j + 1] - environment[j]) / 60 for j in range(30)]
    jumps = [slopes[0]] + [slopes[j] - slopes[j - 1] for j in range(1, 30)]
    variation = sum((abs(value).upper() for value in jumps), arb(0))
    a_min, bi_max, omitted = diffusivity.lower(), biot.upper(), arb(511)
    ratio = (-4 * a_min * (omitted + 1)).exp()
    temperature_tail = (
        (4 * bi_max / a_min)
        * (-a_min * omitted * omitted).exp()
        / (omitted**3 * (1 - ratio))
        * variation
    ).upper()
    coefficients = [
        2 * biot / (root.bessel_j(0) * (root * root + biot * biot)) / (diffusivity * root * root)
        for root in temperature_roots
    ]
    basis = [[(root * x).bessel_j(0) for root in temperature_roots] for x in positions]
    particular = [(x * x - 1 - 2 / biot) / (4 * diffusivity) for x in positions]
    rates = [diffusivity * root * root for root in temperature_roots]
    temperature_times = sorted(
        set(
            [60 * j + 1 for j in range(30)]
            + [60 * (j + 1) for j in range(30)]
            + [2, 59, 119, 121, 736, 1799]
        )
    )
    temperature_stored = list(compact_rows(CERTIFICATES / "evidence/temperature_outward.csv.xz"))
    temperature_values = []

    for time_s in temperature_times:
        j = (time_s - 1) // 60
        ambient = environment[j] + slopes[j] * (time_s - 60 * j)
        modal = [
            coefficients[n]
            * sum((jumps[k] * (-rates[n] * (time_s - 60 * k)).exp() for k in range(j + 1)), arb(0))
            for n in range(256)
        ]
        for radius_index in range(21):
            value = (
                ambient
                + slopes[j] * particular[radius_index]
                + sum((v * w for v, w in zip(modal, basis[radius_index])), arb(0))
                + arb(0, temperature_tail)
            )
            lower, upper = arithmetic.endpoints(value)
            row = temperature_stored[(time_s - 1) * 21 + radius_index]
            reference_lower, reference_upper = compact_endpoints(row)
            require(
                exact_bin4(lower) == exact_bin4(upper) == int(row["certified_bin"])
                and max(lower, reference_lower) <= min(upper, reference_upper),
                "直接温度历史和与证书区间或舍入格不符",
            )
            temperature_values.append(
                {
                    "time_s": time_s,
                    "radius_index": radius_index,
                    "interval_C": [str(lower), str(upper)],
                    "bin_integer": exact_bin4(lower),
                }
            )
    write_json(out / "temperature_direct_profiles.json", temperature_values)

    return temperature_times, len(temperature_values)


def direct_moisture_profiles(arithmetic, out):
    """按直接指数与 I0/erfc 上界复核含水率剖面。"""
    arb = arithmetic.arb
    comparison = MoistureComparison(arithmetic)
    moisture_parameters = comparison.parameters
    beta = arithmetic.number(comparison.beta)
    a = arithmetic.number(comparison.max_diffusivity)
    boundary_amplitude = arithmetic.number(comparison.boundary_amplitude)
    moisture_roots = stored_robin_roots(arithmetic, "moisture")
    x_boxes = [list(map(Fraction, pair)) for pair in moisture_parameters["x_intervals"]]
    x_values = [arithmetic.interval(*pair) for pair in x_boxes]
    coefficients = [
        2 * beta / (root.bessel_j(0) * (root * root + beta * beta)) for root in moisture_roots
    ]
    basis = [[(root * x).bessel_j(0) for root in moisture_roots] for x in x_values]
    omitted = Fraction(1535)
    moisture_tail = arithmetic.endpoints(
        4
        * beta
        / arithmetic.number(omitted)
        * (-a * arithmetic.number(omitted * omitted)).exp()
        / (1 - (-4 * a * arithmetic.number(omitted + 1)).exp())
    )[1]
    stored = list(compact_rows(CERTIFICATES / "evidence/moisture_outward.csv.xz"))
    counts = [sum(bool(row["certified_bin"]) for row in stored[j::21]) for j in range(21)]
    moisture_times = sorted(
        set(
            [1, 2, 59, 60, 61, 300, 600, 900, 1200, 1359, 1500, 1799, 1800]
            + [max(count, 1) for count in counts]
            + [min(count + 1, 1800) for count in counts]
        )
    )
    potential_threshold = Fraction(moisture_parameters["F_rounding_lower_boundary"][1])
    initial_lower, initial_upper = map(Fraction, moisture_parameters["C0"])
    slope_lower = Fraction(moisture_parameters["positive_slope_lower_at_rounding_boundary"])
    ambient_min = Fraction(moisture_parameters["Ceq_min"])
    moisture_values = []
    for time_s in moisture_times:
        state = [
            coefficient * (-a * root * root * time_s).exp()
            for coefficient, root in zip(coefficients, moisture_roots)
        ]
        erfc_profile = comparison.erfc_profile(time_s)
        for radius_index in range(21):
            finite = sum(
                (value * weight for value, weight in zip(state, basis[radius_index])), arb(0)
            )
            linear_drop = (
                boundary_amplitude
                / beta
                * (1 - (finite + arb(0, arithmetic.number(moisture_tail))))
            )
            x_upper = x_boxes[radius_index][1]
            lambda_value = comparison.choose_lambda(float(x_upper), time_s)
            bessel_drop = comparison.bessel_drop_upper(x_upper, time_s, lambda_value)
            erfc_drop = comparison.erfc_drop_upper(x_upper, time_s, erfc_profile)
            combined_drop = min(arithmetic.endpoints(linear_drop)[1], bessel_drop, erfc_drop)
            unique = -combined_drop > potential_threshold
            lower = (
                max(Fraction("2.54995"), initial_lower - combined_drop / slope_lower)
                if unique
                else ambient_min
            )
            row = stored[(time_s - 1) * 21 + radius_index]
            reference_lower, reference_upper = compact_endpoints(row)
            require(
                unique == bool(row["certified_bin"])
                and max(lower, reference_lower) <= min(initial_upper, reference_upper),
                "直接水分指数求值与证书分类或区间不符",
            )
            if unique:
                require(
                    exact_bin4(lower) == exact_bin4(initial_upper) == int(row["certified_bin"]),
                    "直接水分指数求值未确定同一舍入格",
                )
            moisture_values.append(
                {
                    "time_s": time_s,
                    "radius_index": radius_index,
                    "linear_drop_interval_exact": arithmetic.strings(linear_drop),
                    "moisture_interval": [str(lower), str(initial_upper)],
                    "same_unique_classification": True,
                }
            )
    write_json(out / "moisture_direct_profiles.json", moisture_values)
    return moisture_times, len(moisture_values)


def direct_check(out: Path, runtime: Path | None, inputs: Path, fields: SelectedFields) -> dict:
    """分别求值温度和含水率，汇总其实际复核范围。"""
    arithmetic = ArbArithmetic(runtime)
    temperature_times, temperature_count = direct_temperature_profiles(
        arithmetic, inputs, fields, out
    )
    moisture_times, moisture_count = direct_moisture_profiles(arithmetic, out)
    require(temperature_count == 1365 and moisture_count == 987, "直接复核剖面数量不符")
    return {
        "passed": True,
        "temperature_profiles": len(temperature_times),
        "temperature_values": 1365,
        "moisture_profiles": len(moisture_times),
        "moisture_values": 987,
        "temperature_times_s": temperature_times,
        "moisture_times_s": moisture_times,
        "all_direct_intervals_overlap_compact_evidence": True,
        "all_selected_rounding_classifications_match": True,
        "root_source": "Bound stored Robin intervals",
        "complete_root_isolation_repeated": False,
        "evaluation_graph": "Full slope-jump history and direct exp(-rate*t), independent of one-second recurrence.",
        "comparison_reference": (
            "Stored outward compact intervals; the omitted original tighter logs are not "
            "reconstructed by this mode."
        ),
        "runtime": arithmetic.runtime,
        "PDE_solved": False,
    }


def readback_mode(args, out):
    """公共证据读回已在调度之前完成，本模式没有追加计算。"""
    return {}


def compact_mode(args, out):
    return {
        "compact_reconstruction": rebuild_compact_logs(args.temperature_log, args.moisture_log, out)
    }


def analytic_mode(args, out):
    return {"analytic_reconstruction": reconstruct(args.mode, out, args.arb_runtime, args.inputs)}


def selected_field_mode(args, out):
    fields, projection = recover_selected_fields(out)
    result = {"selected_fields": projection}
    if args.mode == "regenerate-exact":
        result["analytic_reconstruction"] = reconstruct(
            args.mode, out, args.arb_runtime, args.inputs, fields
        )
    elif args.mode == "direct-check":
        result["direct_evaluation"] = direct_check(out, args.arb_runtime, args.inputs, fields)
    return result


Q1_MODES = {
    "readback": readback_mode,
    "verify": selected_field_mode,
    "runtime-check": analytic_mode,
    "reconstruct-roots": analytic_mode,
    "regenerate-exact": selected_field_mode,
    "rebuild-compact": compact_mode,
    "direct-check": selected_field_mode,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "readback: 标准库证据和有理输入检查；verify: 增加结果对表与新数值容器；"
            "runtime-check: 256 位 Arb 与参数检查；reconstruct-roots: 完整谱根及间隙重建；"
            "regenerate-exact: 完整谱根及 75600 个解析区间重建；"
            "direct-check: 选定剖面的直接指数复核；"
            "rebuild-compact: 将给定的原精确日志转换成紧凑格式。"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[
            "verify",
            "readback",
            "runtime-check",
            "reconstruct-roots",
            "regenerate-exact",
            "rebuild-compact",
            "direct-check",
        ],
        default="verify",
    )
    parser.add_argument("--out", type=Path, required=True, help="支撑包外的新输出目录")
    parser.add_argument("--arb-runtime", type=Path, help="可选python-flint 0.9.0导入目录")
    parser.add_argument(
        "--inputs",
        type=Path,
        default=CERTIFICATES / "archive/inputs/附件/附件1.xlsx",
        help="附件1副本；须与包内绑定文件具有相同 SHA256",
    )
    parser.add_argument("--temperature-log", type=Path, help="rebuild-compact 使用的原温度精确日志")
    parser.add_argument("--moisture-log", type=Path, help="rebuild-compact 使用的原水分精确日志")
    args = parser.parse_args(argv)
    require(sha256(args.inputs) == INPUT_SHA256, "附件1哈希与问题一证书不符")
    if args.mode == "rebuild-compact":
        require(
            args.temperature_log is not None and args.moisture_log is not None,
            "重建紧凑证书需要两份原精确日志",
        )
    out = prepare_output(args.out)
    report = {
        "mode": args.mode,
        "evidence": verify_evidence(),
        "PDE_solved": False,
        "exact_input_checks": exact_input_checks(args.inputs),
    }
    report.update(run_cli_modes(args.mode, Q1_MODES, args, out))
    report["passed"] = True
    report["current_execution"] = source_identity()
    write_json(out / "q1_verification.json", report)
    print(
        json.dumps(
            {"passed": True, "mode": args.mode, "report": str(out / "q1_verification.json")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
