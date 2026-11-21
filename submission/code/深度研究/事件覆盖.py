"""精确有理数事件覆盖及近临界24任务；验证对象为存储的数值多项式。"""

from __future__ import annotations
import json, hashlib
from pathlib import Path
from fractions import Fraction as F
from math import comb
from types import SimpleNamespace
from decimal import Decimal, localcontext
import numpy as np
from 公共接口 import configure, section, paths, read_json, write_json, sha256
from 研究求解 import (
    require,
    identity,
    plan_hash,
    fresh,
    run_model,
    summarize_matrix,
    runtime,
    MAIN,
    TIGHT,
    LIMITS,
    verify_checks_receipt,
    ProgressChecker,
)


def rat(x):
    return x if isinstance(x, F) else F.from_float(float(x))


def multiply_linear(poly, constant, linear):
    result = [F(0)] * (len(poly) + 1)
    for i, value in enumerate(poly):
        result[i] += value * constant
        result[i + 1] += value * linear
    return result


def node_power_on_unit(dense, node, left, right):
    """Independent expansion of SciPy's Newton product at tau=left+(right-left)u."""
    left, right = (rat(left), rat(right))
    n = int(dense.order)
    result = [F(0)] * (n + 1)
    result[0] = rat(dense.D[0, node])
    product = [F(1)]
    for j in range(n):
        denominator = rat(dense.denom[j])
        if denominator == 0:
            raise ValueError("Zero BDF Newton denominator")
        product = multiply_linear(
            product, (left - rat(dense.t_shift[j])) / denominator, (right - left) / denominator
        )
        coefficient = rat(dense.D[j + 1, node])
        for k, value in enumerate(product):
            result[k] += coefficient * value
    return result


def bernstein(power):
    degree = len(power) - 1
    return [
        sum((power[k] * F(comb(i, k), comb(degree, k)) for k in range(i + 1)), F(0))
        for i in range(degree + 1)
    ]


def split_half(coefficients):
    rows = [list(coefficients)]
    while len(rows[-1]) > 1:
        rows.append([(a + b) / 2 for a, b in zip(rows[-1][:-1], rows[-1][1:])])
    return ([r[0] for r in rows], [r[-1] for r in rows[::-1]])


def rational_text(value):
    return str(value.numerator) + "/" + str(value.denominator)


def witness_candidates(dense, nr, left, right, candidate_limit):
    selection_times = [float(left), float((left + right) / 2), float(right)]
    nodes = [0, nr - 1]
    for t in selection_times:
        values = np.asarray(dense(t))[nr:]
        nodes.extend(np.argsort(values)[-candidate_limit:].tolist())
    nodes = list(dict.fromkeys((int(n) for n in nodes)))
    coefficients = {
        node: bernstein(node_power_on_unit(dense, nr + node, left, right)) for node in nodes
    }
    return nodes, coefficients


def witness_cover(
    dense,
    nr,
    tau_left,
    tau_right,
    *,
    threshold=F.from_float(0.15),
    min_width_s=F(1, 10**7),
    max_depth=40,
    candidate_limit=6,
):
    """Cover the whole interval by subintervals with at least one wet PL node.

    All Bernstein comparisons are Fraction-exact; heuristic node selection can
    only make the result unresolved, never create a false positive.
    """
    left, right = (rat(tau_left), rat(tau_right))
    if right < left or dense.D.shape[1] != 2 * nr:
        raise ValueError("Invalid interval or two-field shape")
    nodes, coefficients = witness_candidates(dense, nr, left, right, candidate_limit)
    stack = [(left, right, coefficients, 0)]
    covered, unresolved, counterexamples = ([], [], [])
    while stack:
        a, b, coeffs, depth = stack.pop()
        witness = next(((node, min(v)) for node, v in coeffs.items() if min(v) >= threshold), None)
        if witness is not None:
            covered.append({"left": a, "right": b, "node": witness[0], "lower": witness[1]})
        elif np.max(np.asarray(dense(float((a + b) / 2)))[nr:]) < float(threshold):
            midpoint = (a + b) / 2
            maximum, node = exact_maximum(dense, nr, midpoint)
            if maximum < threshold:
                counterexamples.append({"tau": midpoint, "maximum_C": maximum, "max_node": node})
                unresolved.append({"left": a, "right": b})
                unresolved.extend(({"left": v[0], "right": v[1]} for v in stack))
                stack.clear()
            else:
                unresolved.append({"left": a, "right": b})
        elif b - a <= min_width_s or depth >= max_depth:
            unresolved.append({"left": a, "right": b})
        else:
            midpoint = (a + b) / 2
            splits = {node: split_half(v) for node, v in coeffs.items()}
            stack.append((midpoint, b, {node: v[1] for node, v in splits.items()}, depth + 1))
            stack.append((a, midpoint, {node: v[0] for node, v in splits.items()}, depth + 1))
    covered.sort(key=lambda v: v["left"])
    unresolved.sort(key=lambda v: v["left"])
    return {
        "all_covered": not unresolved,
        "candidate_nodes": nodes,
        "exact_dry_counterexamples": counterexamples,
        "covered": covered,
        "unresolved": unresolved,
        "threshold_exact_rational": rational_text(threshold),
        "scope": "At every covered time a complete PL nodal field has a node >= threshold. Exact stored-polynomial arithmetic, not a PDE bound.",
    }


def exact_node_value(poly, node, tau):
    tau = rat(tau)
    total, product = (rat(poly.D[0, node]), F(1))
    for j in range(poly.order):
        product *= (tau - rat(poly.t_shift[j])) / rat(poly.denom[j])
        total += rat(poly.D[j + 1, node]) * product
    return total


def exact_maximum(poly, nr, tau):
    values = [exact_node_value(poly, nr + i, tau) for i in range(nr)]
    node = max(range(nr), key=values.__getitem__)
    return (values[node], node)


def tree_json(value):
    if isinstance(value, F):
        return rational_text(value)
    if isinstance(value, dict):
        return {k: tree_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_json(v) for v in value]
    return value


class FirstCrossingObserver:
    """Read-only callback for W6 BalanceObserver.on_used_interval.

    Pre-event intervals must all have exact-rational wet-witness covers. The
    root-containing accepted polynomial must have a <=1e-4 s bracket enclosing
    its first possible dry point; any earlier unresolved piece rejects the gate.
    """

    def __init__(self, directory, initial=None):
        self.directory = Path(directory)
        self.path = self.directory / "continuous_event_intervals.jsonl"
        if self.path.exists():
            raise ValueError("Do not overwrite an event audit")
        self.intervals = self.covered = self.unresolved = 0
        self.bracket = None
        self.solver = None
        self.threshold = F(3, 20)
        self.prefix_receipts = []
        if initial is not None:
            if not initial["window_completed"]:
                raise ValueError("Incomplete event-audit prefix")
            self.intervals, self.covered, self.unresolved = (
                initial[k] for k in ("intervals", "covered_intervals", "unresolved_intervals")
            )
            self.bracket = initial["first_crossing_bracket"]
            self.prefix_receipts = list(initial["all_trace_receipts"])

    def bind_solver(self, solver):
        self.solver = solver

    def __call__(
        self,
        *,
        model,
        polynomial,
        left_tau,
        right_tau,
        physical_origin,
        right_of_breakpoint,
        event_cut_applied,
        event_detection_enabled=True,
    ):
        if self.solver is None:
            raise ValueError("Event observer must be bound to the actual solver")
        left, right = (rat(left_tau), rat(right_tau))
        root = self.solver.event_time
        if not event_detection_enabled or (
            root is not None and physical_origin + float(left) >= root
        ):
            return
        is_root = root is not None and physical_origin + float(right) >= root
        nr = model.nr
        cutoff = right
        bracket = None
        if is_root:
            cutoff, bracket = self._find_root_bracket(
                polynomial, model, left, right, physical_origin, left_tau, root
            )
        audit = witness_cover(polynomial, nr, left, cutoff, threshold=self.threshold)
        self.intervals += 1
        ok = audit["all_covered"] and (not is_root or bracket is not None)
        self.covered += int(ok)
        self.unresolved += int(not ok)
        if bracket is not None:
            self.bracket = tree_json(bracket)
        self._write_interval(
            polynomial,
            audit,
            {
                "interval": self.intervals,
                "physical_origin_s": physical_origin,
                "left_tau": left,
                "used_right_tau": right,
                "audited_prefix_right_tau": cutoff,
                "order": int(polynomial.order),
                "nr": nr,
                "right_of_breakpoint": bool(right_of_breakpoint),
                "event_cut_applied": bool(event_cut_applied),
                "root_interval": bool(is_root),
                "passed": bool(ok),
            },
        )

    def state(self, window_completed=True):
        receipts = list(self.prefix_receipts)
        if self.path.exists():
            receipts.append(
                {
                    "path": str(self.path.resolve()),
                    "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                }
            )
        return {
            "schema": "W3-exact-stored-polynomial-first-event-v1",
            "window_completed": bool(window_completed),
            "intervals": self.intervals,
            "covered_intervals": self.covered,
            "unresolved_intervals": self.unresolved,
            "wet_prefix_passed": self.unresolved == 0,
            "first_crossing_bracket": self.bracket,
            "first_computed_crossing_supported": bool(
                window_completed and self.unresolved == 0 and (self.bracket is not None)
            ),
            "all_trace_receipts": receipts,
            "audited_threshold_exact_rational": rational_text(self.threshold),
            "parent_threshold_binary64_exact_rational": rational_text(F.from_float(0.15)),
            "mathematical_threshold_exact_rational": "3/20",
            "scope": "Exact Fraction arithmetic for stored binary64-coefficient numerical interpolants. No t=0 PDE integration-error enclosure. A finite wet prefix with no root is pending, not nonattainment.",
        }

    def _find_root_bracket(self, polynomial, model, left, right, physical_origin, left_tau, root):
        """寻找精确湿端与干端，宽度门槛维持 1e-4 秒。"""
        nr, cutoff, bracket = model.nr, right, None
        for epsilon in (F(1, 10**7), F(1, 10**6), F(1, 10**5), F(1, 20000)):
            lo, hi = (max(left, right - epsilon), min(rat(polynomial.t), right + epsilon))
            max_lo, wet_node = exact_maximum(polynomial, nr, lo)
            max_hi, dry_node = exact_maximum(polynomial, nr, hi)
            if max_lo >= self.threshold and max_hi < self.threshold and (hi - lo <= F(1, 10000)):
                bracket = {
                    "physical_origin_s": physical_origin,
                    "left_tau": lo,
                    "right_tau": hi,
                    "width_s_exact": hi - lo,
                    "wet_node_at_left": wet_node,
                    "max_C_at_left_exact": max_lo,
                    "max_C_at_right_exact": max_hi,
                    "max_node_at_right": dry_node,
                    "parent_root_s": root,
                    "scope": "First possible dry transition of stored numerical polynomial lies in this bracket if the complete earlier witness cover passes. No uniqueness or PDE error bound.",
                }
                cutoff = lo
                bracket["root_polynomial_sha256"] = self._save_root_polynomial(
                    polynomial, model, physical_origin, left_tau, root
                )
                break
        return cutoff, bracket

    def _save_root_polynomial(self, polynomial, model, physical_origin, left_tau, root):
        """保存完整根多项式及数值域，供读回时逐字节核验。"""
        target = self.directory / "continuous_root_polynomial.npz"
        if target.exists():
            raise ValueError("Duplicate root polynomial")
        with target.open("wb") as f:
            np.savez_compressed(
                f,
                D=polynomial.D,
                t_shift=polynomial.t_shift,
                denom=polynomial.denom,
                order=polynomial.order,
                physical_origin_s=physical_origin,
                accepted_left_tau=left_tau,
                accepted_right_tau=polynomial.t,
                parent_root_s=root,
                xi=model.xi,
            )
        return sha256(target)

    def _write_interval(self, polynomial, audit, row):
        """记录完整多项式哈希、候选节点十六进制系数和精确见证。"""
        nr = row["nr"]
        candidates = audit["candidate_nodes"]
        source_digest = hashlib.sha256()
        for a in (polynomial.D, polynomial.t_shift, polynomial.denom):
            source_digest.update(np.asarray(a, dtype=np.float64).tobytes())
        row.update(
            {
                "stored_full_dense_sha256": source_digest.hexdigest(),
                "t_shift_hex": [float(v).hex() for v in polynomial.t_shift],
                "denom_hex": [float(v).hex() for v in polynomial.denom],
                "candidate_D_hex": {
                    str(i): [float(v).hex() for v in polynomial.D[:, nr + i]] for i in candidates
                },
                "wet_witness_cover": audit,
            }
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(tree_json(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def fixture_checks():

    class Polynomial:

        def __init__(self, moisture_power):
            self.order = len(moisture_power) - 1
            self.t_shift, self.denom = (np.zeros(self.order), np.ones(self.order))
            self.D = np.zeros((self.order + 1, 4))
            self.D[:, 2] = moisture_power
            self.D[0, 3] = 0.14

        def __call__(self, t):
            return np.polynomial.polynomial.polyval(t, self.D)

    positive = Polynomial([0.41, -1.0, 1.0])
    hidden_dip = Polynomial([0.36, -1.0, 1.0])
    one = witness_cover(positive, 2, 0.0, 1.0)
    two = witness_cover(hidden_dip, 2, 0.0, 1.0, min_width_s=F(1, 64))
    endpoint_signs = all((np.max(hidden_dip(t)[2:]) > 0.15 for t in (0.0, 1.0)))
    hidden_dry = np.max(hidden_dip(0.5)[2:]) < 0.15
    shifted = SimpleNamespace(
        order=3,
        t_shift=np.array([0.75, 0.25, -0.25]),
        denom=np.array([0.5, 1.0, 1.5]),
        D=np.array([[1.0], [0.25], [-0.125], [0.0625]]),
    )
    power = node_power_on_unit(shifted, 0, 0.125, 0.875)
    expansion_ok = True
    for u in (F(0), F(1, 7), F(1, 2), F(1)):
        t = F(1, 8) + F(3, 4) * u
        product, exact = (F(1), rat(shifted.D[0, 0]))
        for j in range(shifted.order):
            product *= (t - rat(shifted.t_shift[j])) / rat(shifted.denom[j])
            exact += rat(shifted.D[j + 1, 0]) * product
        expansion_ok &= sum((v * u**j for j, v in enumerate(power)), F(0)) == exact
    return {
        "passed": bool(
            one["all_covered"]
            and (not two["all_covered"])
            and endpoint_signs
            and hidden_dry
            and expansion_ok
        ),
        "strict_wet_polynomial_covered": one["all_covered"],
        "same_sign_endpoints_hidden_dry_dip_rejected": bool(
            endpoint_signs and hidden_dry and (not two["all_covered"])
        ),
        "independent_newton_expansion_exact": bool(expansion_ok),
        "production_PDE_started": False,
        "scope": "Exact algebraic fixtures; actual accepted-step integration hook is a separate check.",
    }


def verify_state_header(directory):
    """核对事件证据结构、布尔值、计数与精确阈值。"""
    directory = Path(directory).resolve()
    state = read_json(directory / "continuous_event_state.json")
    require(
        state["schema"] == "W3-exact-stored-polynomial-first-event-v1",
        "Wrong first-event evidence schema",
    )
    for key in ("window_completed", "wet_prefix_passed", "first_computed_crossing_supported"):
        require(type(state[key]) is bool, "Invalid state boolean " + key)
    for key in ("intervals", "covered_intervals", "unresolved_intervals"):
        require(type(state[key]) is int and state[key] >= 0, "Invalid event counter " + key)
    threshold = F(3, 20)
    require(
        F(state["audited_threshold_exact_rational"]) == threshold
        and F(state["mathematical_threshold_exact_rational"]) == threshold,
        "The audited and mathematical threshold must be exactly 3/20",
    )
    require(
        F(state["parent_threshold_binary64_exact_rational"]) == F.from_float(0.15),
        "Wrong recorded parent binary threshold",
    )
    return state, threshold


def verify_witness_row(row, state, threshold, start, end, shifts, denom):
    audit = row["wet_witness_cover"]
    require(type(audit["all_covered"]) is bool, "Invalid all-covered boolean")
    require(
        F(audit["threshold_exact_rational"]) == threshold,
        "Per-row witness threshold differs from exact 3/20",
    )
    if "threshold_exact_rational" in row:
        require(F(row["threshold_exact_rational"]) == threshold, "Row threshold differs")
    nodes = audit["candidate_nodes"]
    require(
        len(nodes) > 0
        and len(nodes) == len(set(nodes))
        and all((type(n) is int and 0 <= n < row["nr"] for n in nodes)),
        "Invalid/duplicate candidate node",
    )
    require(
        set(row["candidate_D_hex"]) == {str(n) for n in nodes},
        "Candidate coefficient keys differ from witness candidates",
    )
    for values in row["candidate_D_hex"].values():
        a = np.asarray([float.fromhex(v) for v in values])
        require(
            a.shape == (row["order"] + 1,) and np.isfinite(a).all(),
            "Invalid candidate Newton coefficients",
        )
    pieces = []
    for item in audit["covered"]:
        left, right = (F(item["left"]), F(item["right"]))
        require(
            start <= left <= right <= end and item["node"] in nodes,
            "Witness outside used prefix/candidate set",
        )
        D = np.asarray([[float.fromhex(v)] for v in row["candidate_D_hex"][str(item["node"])]])
        poly = SimpleNamespace(order=row["order"], D=D, t_shift=shifts, denom=denom)
        lower = min(bernstein(node_power_on_unit(poly, 0, left, right)))
        require(
            lower == F(item["lower"]) and lower >= threshold,
            "Invalid exact wet witness",
        )
        pieces.append((left, right))
    for item in audit["unresolved"]:
        left, right = (F(item["left"]), F(item["right"]))
        require(start <= left <= right <= end, "Unresolved piece outside prefix")
        pieces.append((left, right))
    pieces.sort()
    cursor = start
    for left, right in pieces:
        require(left == cursor and right >= left, "Event-cover gap/overlap")
        cursor = right
    require(cursor == end, "Event cover does not span the used prefix")
    require(
        audit["all_covered"] == (len(audit["unresolved"]) == 0),
        "Incorrect all-covered flag",
    )
    ok = audit["all_covered"] and (
        not row["root_interval"] or state["first_crossing_bracket"] is not None
    )
    require(ok == row["passed"], "Incorrect interval flag")
    return ok


def verify_trace_rows(state, threshold):
    rows = covered = unresolved = 0
    previous_end = None
    previous_origin = None
    first_start = None
    last_row = None
    root_row = None
    root_rows = 0
    nr_fixed = None
    for receipt in state["all_trace_receipts"]:
        path = Path(receipt["path"])
        require(sha256(path) == receipt["sha256"], "Event trace SHA differs")
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rows += 1
                require(row["interval"] == rows, "Event interval index gap/duplicate")
                require(type(row["interval"]) is int, "Invalid interval index type")
                for key in ("right_of_breakpoint", "event_cut_applied", "root_interval", "passed"):
                    require(type(row[key]) is bool, "Invalid row boolean " + key)
                require(root_row is None, "A root row must be the unique terminal event-audit row")
                require(
                    type(row["order"]) is int
                    and 1 <= row["order"] <= 5
                    and (type(row["nr"]) is int)
                    and (row["nr"] >= 2),
                    "Invalid BDF order or radial shape",
                )
                if nr_fixed is None:
                    nr_fixed = row["nr"]
                require(row["nr"] == nr_fixed, "Radial node count changed within event evidence")
                start, end = (F(row["left_tau"]), F(row["audited_prefix_right_tau"]))
                used_end = F(row["used_right_tau"])
                origin_float = float(row["physical_origin_s"])
                require(
                    np.isfinite(origin_float)
                    and origin_float >= 0
                    and (F(0) <= start <= end <= used_end),
                    "Invalid event interval/order/physical origin",
                )
                origin = F.from_float(origin_float)
                physical_start = origin + start
                physical_end = origin + used_end
                if previous_origin is None or origin != previous_origin:
                    require(start == 0, "A new local-clock origin must begin at local time zero")
                if previous_end is not None:
                    gap = physical_start - previous_end
                    require(
                        gap == 0, "Physical event-cover gap/overlap (exact seconds): " + str(gap)
                    )
                else:
                    first_start = physical_start
                previous_end = physical_end
                previous_origin = origin
                if row["root_interval"]:
                    root_rows += 1
                    root_row = row
                else:
                    require(end == used_end, "Non-root row truncates its audited prefix")
                shifts = np.asarray([float.fromhex(v) for v in row["t_shift_hex"]])
                denom = np.asarray([float.fromhex(v) for v in row["denom_hex"]])
                require(
                    shifts.shape == denom.shape == (row["order"],)
                    and np.isfinite(shifts).all()
                    and np.isfinite(denom).all()
                    and np.all(denom > 0),
                    "Invalid Newton coordinate arrays",
                )
                dense_end = F.from_float(float(shifts[0]))
                require(
                    start < dense_end and used_end <= dense_end,
                    "Used row extrapolates beyond its actual BDF dense endpoint",
                )
                require(
                    row["event_cut_applied"] == (used_end < dense_end),
                    "Incorrect actual dense-domain clipping flag",
                )
                if not row["root_interval"]:
                    require(
                        used_end == dense_end, "Non-root row must use the full accepted BDF step"
                    )
                ok = verify_witness_row(row, state, threshold, start, end, shifts, denom)
                covered += int(ok)
                unresolved += int(not ok)
                last_row = row
    require(
        (rows, covered, unresolved)
        == (state["intervals"], state["covered_intervals"], state["unresolved_intervals"]),
        "Event-audit totals differ",
    )
    return {
        "rows": rows,
        "covered": covered,
        "unresolved": unresolved,
        "first_start": first_start,
        "previous_end": previous_end,
        "last_row": last_row,
        "root_row": root_row,
        "root_rows": root_rows,
    }


def verify_root_polynomial(directory, state, trace, threshold):
    last_row = trace["last_row"]
    root_row = trace["root_row"]
    root_rows = trace["root_rows"]
    bracket = state["first_crossing_bracket"]
    if bracket is not None:
        require(
            root_rows == 1 and root_row is last_row, "A bracket requires exactly one final root row"
        )
        require(
            F.from_float(float(bracket["physical_origin_s"]))
            == F.from_float(float(root_row["physical_origin_s"])),
            "Root bracket origin differs from terminal row",
        )
        require(
            F(bracket["left_tau"]) == F(root_row["audited_prefix_right_tau"]),
            "Root bracket and wet prefix do not join exactly",
        )
        candidates = [directory / "continuous_root_polynomial.npz"]
        candidates += [
            Path(item["path"]).parent / "continuous_root_polynomial.npz"
            for item in state["all_trace_receipts"]
        ]
        paths = [
            p for p in candidates if p.is_file() and sha256(p) == bracket["root_polynomial_sha256"]
        ]
        require(paths, "Root polynomial not found or changed")
        with np.load(paths[0], allow_pickle=False) as d:
            for name in ("D", "t_shift", "denom", "xi"):
                require(
                    d[name].dtype == np.dtype("float64") and np.isfinite(d[name]).all(),
                    "Invalid/nonfinite root array " + name,
                )
            require(
                d["order"].shape == () and d["order"].dtype.kind in "iu",
                "Invalid root order scalar",
            )
            poly = SimpleNamespace(
                order=int(d["order"]), D=d["D"], t_shift=d["t_shift"], denom=d["denom"]
            )
            nr = len(d["xi"])
            require(
                poly.order == root_row["order"]
                and nr == root_row["nr"]
                and (poly.D.shape == (poly.order + 1, 2 * nr))
                and (poly.t_shift.shape == poly.denom.shape == (poly.order,)),
                "Root polynomial shape differs from terminal row",
            )
            require(
                d["xi"].shape == (nr,)
                and d["xi"][0] == 0.0
                and (d["xi"][-1] == 1.0)
                and np.all(np.diff(d["xi"]) > 0)
                and np.all(poly.denom > 0),
                "Invalid root radial coordinate or Newton denominators",
            )
            for name in (
                "physical_origin_s",
                "accepted_left_tau",
                "accepted_right_tau",
                "parent_root_s",
            ):
                require(
                    d[name].shape == ()
                    and d[name].dtype == np.dtype("float64")
                    and np.isfinite(float(d[name])),
                    "Invalid root scalar " + name,
                )
            origin = F.from_float(float(d["physical_origin_s"]))
            left = F.from_float(float(d["accepted_left_tau"]))
            right = F.from_float(float(d["accepted_right_tau"]))
            require(
                origin == F.from_float(float(root_row["physical_origin_s"]))
                and left == F(root_row["left_tau"]),
                "Root saved origin/accepted-left does not match terminal row",
            )
            require(
                F(0) <= left <= F(root_row["used_right_tau"]) <= right,
                "Root used interval is outside actual saved polynomial",
            )
            require(
                right == F.from_float(float(poly.t_shift[0])) and left < right,
                "Root declared endpoint differs from actual BDF dense anchor",
            )
            require(
                float(d["parent_root_s"]) == float(bracket["parent_root_s"])
                and F.from_float(float(d["parent_root_s"]))
                == origin + F(root_row["used_right_tau"]),
                "Parent root is not the terminal used endpoint",
            )
            digest = hashlib.sha256()
            for value in (poly.D, poly.t_shift, poly.denom):
                digest.update(np.asarray(value, dtype=np.float64).tobytes())
            require(
                digest.hexdigest() == root_row["stored_full_dense_sha256"],
                "Root full polynomial is not the terminal-row source polynomial",
            )
            require(
                poly.t_shift.tobytes()
                == np.asarray(
                    [float.fromhex(v) for v in root_row["t_shift_hex"]], dtype=np.float64
                ).tobytes()
                and poly.denom.tobytes()
                == np.asarray(
                    [float.fromhex(v) for v in root_row["denom_hex"]], dtype=np.float64
                ).tobytes(),
                "Root Newton coordinates differ from terminal witness",
            )
            for node, values in root_row["candidate_D_hex"].items():
                require(
                    poly.D[:, nr + int(node)].tobytes()
                    == np.asarray([float.fromhex(v) for v in values], dtype=np.float64).tobytes(),
                    "Root candidate coefficients differ from terminal witness",
                )
            lo, hi = (F(bracket["left_tau"]), F(bracket["right_tau"]))
            require(
                left <= lo <= F(root_row["used_right_tau"]) <= hi <= right,
                "Root bracket exceeds accepted polynomial domain",
            )
            wet, wi = exact_maximum(poly, nr, lo)
            dry, di = exact_maximum(poly, nr, hi)
            require(hi - lo == F(bracket["width_s_exact"]) <= F(1, 10000), "Root bracket too wide")
            require(
                wet == F(bracket["max_C_at_left_exact"]) >= threshold
                and dry == F(bracket["max_C_at_right_exact"]) < threshold,
                "Invalid exact wet/dry endpoints",
            )
            require(
                wi == bracket["wet_node_at_left"] and di == bracket["max_node_at_right"],
                "Root maximum location differs",
            )
    else:
        require(root_rows <= 1, "More than one unresolved terminal root row")
        if root_rows:
            require(not root_row["passed"], "Root row without a bracket cannot pass")
    return bracket


def verify_final_gates(directory, state, trace, bracket):
    rows = trace["rows"]
    unresolved = trace["unresolved"]
    first_start = trace["first_start"]
    previous_end = trace["previous_end"]
    root_rows = trace["root_rows"]
    expected = state["window_completed"] and unresolved == 0 and (bracket is not None)
    require(state["first_computed_crossing_supported"] == expected, "Incorrect first-crossing gate")
    require(state["wet_prefix_passed"] == (unresolved == 0), "Incorrect wet-prefix flag")
    return {
        "passed": True,
        "certificate_arithmetic_passed": True,
        "first_computed_crossing_supported": expected,
        "wet_prefix_passed": unresolved == 0,
        "rows": rows,
        "continuous_PDE_certified": False,
        "source_state_sha256": sha256(directory / "continuous_event_state.json"),
        "verifier_sha256": sha256(__file__),
        "root_bracket": bracket,
        "threshold_exact_rational": "3/20",
        "all_physical_joins_exact": True,
        "root_rows": root_rows,
        "actual_BDF_endpoint_anchors_checked": True,
        "first_audited_physical_s_exact": str(first_start) if first_start is not None else None,
        "last_used_physical_s_exact": str(previous_end) if previous_end is not None else None,
        "coverage_scope": "All stored pre-event rows join exactly in physical time. The first origin is reported, not assumed to be zero; an artificial late initial-state fixture does not prove a t=0 trajectory.",
        "source_binding_scope": "Terminal full root polynomial is bound to its row hash and candidate coefficients. Other rows retain only their selected witness coefficients; full solver provenance must also be checked by the calling case reader.",
    }


def verify(directory):
    """依次校验证据头、连续轨迹、精确见证、根多项式和最终结论。"""
    directory = Path(directory).resolve()
    state, threshold = verify_state_header(directory)
    trace = verify_trace_rows(state, threshold)
    bracket = verify_root_polynomial(directory, state, trace, threshold)
    return verify_final_gates(directory, state, trace, bracket)


class CeqMultiplier:
    """Evaluate base interpolation first, then multiply only effective solid Ceq.

    Pre-scaling input nodes changes rounding at intermediate times; this wrapper
    deliberately matches the declared alpha * base.values(t) convention.
    """

    def __init__(self, base, alpha):
        self.base, self.alpha = (base, float(alpha))
        require(np.isfinite(self.alpha) and self.alpha > 0, "Invalid positive Ceq multiplier")

    def __getattr__(self, name):
        return getattr(self.base, name)

    def values(self, t_s, *, right_of_breakpoint=False):
        value = self.base.values(t_s, right_of_breakpoint=right_of_breakpoint)
        return type(value)(value.temperature_K, self.alpha * value.moisture_eq, value.extrapolated)

    def breakpoints(self, start_s, end_s):
        return self.base.breakpoints(start_s, end_s)


def plan(horizon_s=864000.0):
    """Six q<1 material scenarios and their exact 24 specified jobs."""
    protocol = read_json(paths()["near_critical_protocol"])
    require(
        np.isfinite(horizon_s) and horizon_s >= 14400,
        "Initial window must cover the first four hours",
    )
    base_tail = protocol["alpha_critical"]["base_tail_binary64"]
    scenarios, jobs = [], []
    for q in protocol["scenarios"]:
        qd = Decimal(q["q_tail_over_threshold_exact"])
        if qd >= 1:
            continue
        with localcontext() as context:
            context.prec = 80
            exact = qd * Decimal(".15") / Decimal.from_float(base_tail)
            alpha = float(exact)
            difference = Decimal.from_float(alpha * base_tail) - qd * Decimal(".15")
        require(alpha == q["alpha_binary64"], "Multiplier differs from original protocol")
        for material, geometry, case in [("P23", "fixed", "p23"), ("P4", "measured", "q4")]:
            scenario = dict(
                id=material + "_" + q["case_id"],
                property_set=material,
                geometry=geometry,
                question=case,
                q_decimal=str(qd),
                alpha_decimal_80digits=str(exact),
                alpha_binary64=alpha,
                base_tail_binary64=base_tail,
                target_tail_exact_decimal=str(qd * Decimal(".15")),
                actual_binary64_tail=alpha * base_tail,
                tail_binary64_minus_exact_target=str(difference),
            )
            scenarios.append(scenario)
            levels = protocol["proposed_initial_numerical_configurations"][case]
            for index, row in enumerate(levels):
                level = "tight" if index == len(levels) - 1 else "main"
                jobs.append(
                    dict(
                        id=f'{scenario["id"]}_N{row["N"]}_{level}',
                        scenario=dict(scenario),
                        N=row["N"],
                        level=level,
                        settings=dict(row["settings"]),
                    )
                )
    require(len(scenarios) == 6 and len(jobs) == 24, "Incomplete near-critical matrix")
    return dict(
        schema="nearcritical24-direct-v1",
        scenarios=scenarios,
        jobs=jobs,
        initial_state=read_json(paths()["contract"])["initial_state"],
        source_bindings=identity(),
        horizon_s=float(horizon_s),
        post_event_window_s=2.0,
        physical_seconds_cutoff=14400.0,
        full_seconds_cutoff=14400.0,
        acceptance=dict(
            spatial=dict(T_K=1e-5, C=1e-5), temporal=dict(T_K=1e-5, C=1e-5), integral_balance=LIMITS
        ),
        q_ge_one_scope="No finite numerical window proves nonattainment; exact critical decimal and binary product remain distinct.",
        test_only=False,
        continuous_PDE_certified=False,
        effects_accepted=False,
    )


def validate(value):
    require(
        value == plan(value["horizon_s"]), "Source, protocol, matrix or numerical criteria changed"
    )
    return dict(passed=True, jobs=24, production_PDE_started=False)


def build(scenario, N):
    from 潜热矩阵 import LatentModel

    moving, relative = section("moving_model"), section("relative_temperature")
    p = paths()
    ambient = CeqMultiplier(
        moving.AmbientProfile.from_xlsx(Path(p["inputs"]) / "附件1.xlsx"),
        scenario["alpha_binary64"],
    )
    original = moving.build_model(
        p["inputs"],
        p["contract"],
        N=N,
        power=2.0,
        properties=scenario["property_set"],
        geometry=scenario["geometry"],
        ambient=ambient,
    )
    return LatentModel(
        relative.RelativeTemperatureModel.from_model(original),
        read_json(p["contract"]),
        scenario["property_set"],
        latent_scale=0.0,
    )


def run(value, job_id, out, checks, previous=None, until_s=None):
    validate(value)
    receipt = read_json(checks) if isinstance(checks, (str, Path)) else checks
    verify_checks_receipt(
        receipt,
        value,
        schema="events-direct-checks-v1",
        expected_names=CHECK_NAMES,
        stale_message="Missing or stale near-critical bounded checks",
        incomplete_message="Incomplete near-critical check suite",
    )
    jobs = [job for job in value["jobs"] if job["id"] == job_id]
    require(len(jobs) == 1, "Unknown or duplicate job")
    job = jobs[0]
    return run_model(value, job, build(job["scenario"], job["N"]), out, previous, until_s)


def summarize(value, runs, out=None, case_map=None):
    validate(value)
    result = summarize_matrix(value, runs, case_map)
    if out is not None:
        write_json(out, result)
    return result


CHECK_NAMES = sorted(
    [
        "base_tail_exact_protocol_identity",
        "P23_alpha1_bitwise_RHS",
        "P4_alpha1_bitwise_RHS",
        "critical_exact_and_float_not_conflated",
        "schedule_first4h_seconds",
        "schedule_minutes_after4h",
        "matrices_original_precision",
        "exact_polynomial_guard_fixtures",
        "P23_continuation",
        "P4_continuation",
        "polynomial_fixed",
        "polynomial_moving",
        "restore",
        "late_event_and_extension",
        "breakpoint_semantics",
        "rejection_gates",
    ]
    + [
        prop + "_" + q + suffix
        for prop in ("P23", "P4")
        for q in ("q_0p90", "q_0p99", "q_0p999")
        for suffix in (
            "_ambient_False",
            "_ambient_True",
            "_initial",
            "_all_input_knots",
            "_only_Robin_0.0",
            "_only_Robin_14400.0",
            "_only_Robin_14401.0",
        )
    ]
)


def checks(value, out):
    """Original boundary identities, exact-polynomial fixtures, real restart and event checks."""
    import copy
    from 潜热矩阵 import polynomial_check
    from 研究求解 import (
        sample_run,
        equal_samples,
        restart_check,
        expected_times,
        window_record,
        effect_gate,
    )

    validate(value)
    out = fresh(out)
    checked = ProgressChecker(out, "Near-critical check failed: ")
    rows = checked.rows

    moving, relative = section("moving_model"), section("relative_temperature")
    base = moving.AmbientProfile.from_xlsx(Path(paths()["inputs"]) / "附件1.xlsx")
    checked(
        "base_tail_exact_protocol_identity",
        all(s["base_tail_binary64"] == base.tail_moisture for s in value["scenarios"]),
    )
    times = np.unique(
        np.r_[
            base.t_s,
            (base.t_s[:-1] + base.t_s[1:]) / 2,
            14400.0,
            np.nextafter(14400.0, np.inf),
            259200.0,
            259201.0,
            864000.0,
        ]
    )
    for scenario in value["scenarios"]:
        wrapped = CeqMultiplier(base, scenario["alpha_binary64"])
        for right in (False, True):
            original, scaled = base.values(times, right_of_breakpoint=right), wrapped.values(
                times, right_of_breakpoint=right
            )
            checked(
                scenario["id"] + "_ambient_" + str(right),
                np.array_equal(
                    scaled.moisture_eq, scenario["alpha_binary64"] * original.moisture_eq
                )
                and np.array_equal(scaled.temperature_K, original.temperature_K)
                and np.array_equal(scaled.extrapolated, original.extrapolated),
            )
        model = build(scenario, 32)
        y = model.initial_state(301.15, 2.55)
        checked(
            scenario["id"] + "_initial",
            np.all(model.unpack(y)[0] == 301.15)
            and np.all(model.unpack(y)[1] == 2.55)
            and model.geometry.radius(0.0) == 0.02,
        )
        knots = model.ambient.breakpoints(0.0, 864000.0)
        expected = np.union1d(
            base.breakpoints(0.0, 864000.0), model.geometry.breakpoints(0.0, 864000.0)
        )
        checked(scenario["id"] + "_all_input_knots", np.array_equal(knots, expected))
        reference = build(dict(scenario, alpha_binary64=1.0), 32)
        for t in (0.0, 14400.0, 14401.0):
            a, b = model.rhs(t, y), reference.rhs(t, y)
            R, R0 = model.geometry.radius(t), model.grid.radius_m
            delta = (
                (R0 / R) ** 2
                * R
                * model.h_m_m_s
                * (
                    float(model.ambient.values(t).moisture_eq)
                    - float(reference.ambient.values(t).moisture_eq)
                )
                / model.grid.volume_weights_m2[-1]
            )
            checked(
                scenario["id"] + "_only_Robin_" + str(t),
                np.array_equal(a[:-1], b[:-1])
                and abs((a[-1] - b[-1]) - delta)
                <= 128 * np.finfo(float).eps * max(1.0, abs(delta)),
            )
    for prop, geometry in [("P23", "fixed"), ("P4", "measured")]:
        scenario = next(s for s in value["scenarios"] if s["property_set"] == prop)
        scaled = build(dict(scenario, alpha_binary64=1.0), 32)
        reference = relative.RelativeTemperatureModel.from_model(
            moving.build_model(
                paths()["inputs"],
                paths()["contract"],
                N=32,
                power=2.0,
                properties=prop,
                geometry=geometry,
            )
        )
        y = np.r_[-10.0 + 4 * reference.xi, 1.0 + 0.1 * reference.xi**2]
        checked(
            prop + "_alpha1_bitwise_RHS",
            all(np.array_equal(scaled.rhs(t, y), reference.rhs(t, y)) for t in times),
        )
    critical = next(
        q
        for q in read_json(paths()["near_critical_protocol"])["scenarios"]
        if q["case_id"] == "q_1"
    )
    checked(
        "critical_exact_and_float_not_conflated",
        Decimal(critical["target_tail_exact_decimal"]) == Decimal(".15")
        and critical["product_tail_binary64"] > 0.15,
    )
    checked(
        "schedule_first4h_seconds",
        np.array_equal(expected_times("measured", 14400.0, 14400.0), np.arange(1.0, 14401.0)),
    )
    checked(
        "schedule_minutes_after4h",
        np.array_equal(expected_times("measured", 14521.0, 14400.0)[-2:], [14460.0, 14520.0]),
    )
    checked(
        "matrices_original_precision",
        len(value["jobs"]) == 24
        and next(j for j in value["jobs"] if j["scenario"]["property_set"] == "P4")["settings"][
            "max_step_s"
        ]
        == 600.0,
    )
    checked("exact_polynomial_guard_fixtures", fixture_checks()["passed"])
    for prop in ("P23", "P4"):
        original = next(
            j
            for j in value["jobs"]
            if j["scenario"]["property_set"] == prop
            and j["scenario"]["q_decimal"] == "0.999"
            and j["level"] == "tight"
        )
        job = copy.deepcopy(original)
        job.update(id=prop + "_TEST_ONLY_N32", N=32)
        fixture = copy.deepcopy(value)
        fixture.update(jobs=[job], horizon_s=3.0, test_only=True)
        first = run_model(
            fixture, job, build(job["scenario"], 32), out / (prop + "_3s"), until_s=3.0
        )
        prefix = {p.name: sha256(p) for p in (out / (prop + "_3s")).iterdir() if p.is_file()}
        second = run_model(
            fixture,
            job,
            build(job["scenario"], 32),
            out / (prop + "_6s"),
            previous=out / (prop + "_3s"),
            until_s=6.0,
        )
        checked(
            prop + "_continuation",
            all(sha256(out / (prop + "_3s") / k) == v for k, v in prefix.items())
            and first["event_pending"]
            and second["event_pending"]
            and second["balance_passed"]
            and window_record(out / (prop + "_6s"), fixture)["wet_prefix_passed"],
        )
    for moving_geometry in (False, True):
        checked(
            "polynomial_" + ("moving" if moving_geometry else "fixed"),
            polynomial_check(moving_geometry)["passed"],
        )
    scenario = next(
        s for s in value["scenarios"] if s["property_set"] == "P4" and s["q_decimal"] == "0.999"
    )
    model = build(scenario, 16)
    checked("restore", restart_check(model, out / "restore")["passed"])
    initial = np.r_[np.zeros(model.nr), 0.149 + 0.0010000001 * (1 - model.xi**2)]
    ref = sample_run(model, out / "late_reference", False, 200000.0, 200060.0, initial, True)
    got = sample_run(model, out / "late_audit", True, 200000.0, 200060.0, initial, True)
    checked(
        "late_event_and_extension",
        equal_samples(ref, got)
        and got["balance"]["passed"]
        and got["event_verification"]["first_computed_crossing_supported"],
    )
    ref = sample_run(model, out / "break_reference", False, 14399.0, 14401.0)
    got = sample_run(model, out / "break_audit", True, 14399.0, 14401.0)
    checked(
        "breakpoint_semantics",
        equal_samples(ref, got)
        and got["balance"]["passed"]
        and [s["right_of_breakpoint"] for s in got["balance"]["segments"]] == [False, True],
    )
    bad = copy.deepcopy(value)
    bad["acceptance"]["spatial"]["C"] = 1.0
    try:
        validate(bad)
    except ValueError:
        rejected = True
    else:
        rejected = False
    gate = [dict(kind="space", target_met=True), dict(kind="time", target_met=True)]
    checked(
        "rejection_gates",
        rejected
        and not effect_gate(gate, False, [True] * 4, dict(passed=True))
        and not effect_gate(gate, True, [True] * 4, None),
    )
    require(sorted(r["name"] for r in rows) == CHECK_NAMES, "Incomplete near-critical check names")
    validate(value)
    result = dict(
        schema="events-direct-checks-v1",
        passed=True,
        expected_check_names=CHECK_NAMES,
        checks=rows,
        plan_sha256=plan_hash(value),
        source_bindings=identity(),
        runtime=runtime(),
        production_PDE_started=False,
        continuous_PDE_certified=False,
    )
    write_json(out / "checks.json", result)
    return result


def main(argv=None):
    from 研究求解 import branch_cli

    return branch_cli(
        "近临界24任务与精确事件覆盖",
        plan,
        validate,
        checks,
        run,
        summarize,
        argv,
        builder=build,
        verifier=verify,
    )


if __name__ == "__main__":
    main()
