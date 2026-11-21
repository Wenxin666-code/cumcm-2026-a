"""连续 PDE：直接复用公共径向算子，融合稠密轨迹、误差管和局部屏障。

此模块只含正常的 Python 类与函数。配置、附件与证明轨迹从指定路径读取。
数值轨迹与连续 PDE 误差认证是两个独立步骤；未经闭合的区间推进不会
被标为已证明。新源码产生自己的来源记录，不冒用旧运行的源码哈希。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from math import comb
from pathlib import Path
import argparse
import gzip
import hashlib
import inspect
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.radau import RadauDenseOutput
from scipy.integrate._ivp.common import OdeSolution
from 公共接口 import (
    configure,
    paths,
    section,
    read_json,
    write_json,
    sha256,
    source_identity,
)

PLACES = 70


@lru_cache(maxsize=1)
def flint_runtime():
    """仅局部 Gaussian/Kirchhoff 区间推进需要 python-flint；数值推进不需要。"""
    try:
        import flint
    except ImportError as error:
        raise RuntimeError("局部严格区间推进需要 python-flint（原研究使用0.9.0）") from error
    return flint


def down(x, places=PLACES):
    scale = 10**places
    return F(x.numerator * scale // x.denominator, scale)


def up(x, places=PLACES):
    return -down(-x, places)


def text(x, places=PLACES):
    z = x * 10**places
    if z.denominator != 1:
        raise ValueError("Text requires a previously rounded rational endpoint")
    value = abs(z.numerator)
    return (
        ("-" if z < 0 else "")
        + str(value // 10**places)
        + "."
        + str(value % 10**places).zfill(places)
    )


def bounds(lo, hi):
    assert lo <= hi
    return {"lower": text(down(lo)), "upper": text(up(hi))}


def exp_negative(x, order=48):
    """e^-x for x>=0. Reduce y=x/2**k<=1/2.

    With S_n=sum_{j=0}^n y^j/j!,
    S_n <= e^y <= S_n + [y^(n+1)/(n+1)!]/[1-y/(n+2)].
    Reciprocate, round outward, then square k times with outward rounding.
    """
    assert x >= 0
    k = 0
    y = x
    while y > F(1, 2):
        k += 1
        y /= 2
    term = total = F(1)
    for j in range(1, order + 1):
        term = term * y / j
        total += term
    remainder = term * y / (order + 1) / (1 - y / (order + 2))
    lo, hi = (down(1 / (total + remainder)), up(1 / total))
    for _ in range(k):
        lo, hi = (down(lo * lo), up(hi * hi))
    assert 0 <= lo <= hi <= 1
    return (lo, hi)


def log_positive(z):
    """log z for z>0, using log z = 2*atanh((z-1)/(z+1)).

    Range reduction gives 1<=r<=2. With w=(r-1)/(r+1),
    the positive odd-power series truncated after 2n+1 has remainder
    <= 2*w^(2n+3)/[(2n+3)*(1-w*w)].
    """
    assert z > 0
    if z < 1:
        lo, hi = log_positive(1 / z)
        return (-hi, -lo)
    k = 0
    while z > 2:
        z /= 2
        k += 1
    w = (z - 1) / (z + 1)
    power = w
    total = F(0)
    n = 0
    tolerance = F(1, 10 ** (PLACES + 8))
    while True:
        total += 2 * power / (2 * n + 1)
        next_power = power * w * w
        remainder = 2 * next_power / ((2 * n + 3) * (1 - w * w))
        if remainder <= tolerance:
            break
        n += 1
        power = next_power
    lo, hi = (down(total), up(total + remainder))
    if k:
        l2, u2 = log_positive(F(2))
        lo, hi = (down(lo + k * l2), up(hi + k * u2))
    return (lo, hi)


def self_checks():
    exact_cases = (
        ("exp(0) exact", exp_negative, F(0), (F(1), F(1))),
        ("log(1) exact", log_positive, F(1), (F(0), F(0))),
    )
    checks = [
        {"name": name, "passed": evaluate(argument) == expected}
        for name, evaluate, argument, expected in exact_cases
    ]
    l2, u2 = log_positive(F(2))
    l8, u8 = log_positive(F(8))
    checks.append(
        {
            "name": "log(8) versus 3 log(2) interval identity",
            "passed": max(l8, 3 * l2) <= min(u8, 3 * u2),
        }
    )
    e_lo, _ = exp_negative(u2)
    _, e_hi = exp_negative(l2)
    checks.append({"name": "exp(-log(2)) contains exact 1/2", "passed": e_lo <= F(1, 2) <= e_hi})
    for x in [F("0.1"), F(1), F(13), F(18)]:
        y = x
        k = 0
        while y > F(1, 2):
            y /= 2
            k += 1
        term = total = F(1)
        for j in range(1, 49):
            term = -term * y / j
            total += term
        alt_lo = down((total - term * y / 49) ** 2**k)
        alt_hi = up(total**2**k)
        lo, hi = exp_negative(x)
        checks.append(
            {
                "name": "independent alternating Taylor at " + str(x),
                "passed": max(lo, alt_lo) <= min(hi, alt_hi),
            }
        )
    assert all((r["passed"] for r in checks))
    return checks


@dataclass(frozen=True)
class Interval:
    lo: F
    hi: F

    def __init__(self, lo=0, hi=None):
        if isinstance(lo, Interval):
            hi = lo.hi if hi is None else hi
            lo = lo.lo
        lo = F(lo)
        hi = lo if hi is None else F(hi)
        if lo > hi:
            raise ValueError("Reversed interval")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    def __add__(self, other):
        other = Interval(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -Interval(other)

    def __rsub__(self, other):
        return Interval(other) + -self

    def __mul__(self, other):
        other = Interval(other)
        terms = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        return Interval(min(terms), max(terms))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Interval(other)
        if other.lo <= 0 <= other.hi:
            raise ValueError("Division by an interval containing zero")
        return self * Interval(1 / other.hi, 1 / other.lo)

    def __rtruediv__(self, other):
        return Interval(other) / self

    def square(self):
        if self.lo <= 0 <= self.hi:
            return Interval(0, max(self.lo**2, self.hi**2))
        return Interval(min(self.lo**2, self.hi**2), max(self.lo**2, self.hi**2))

    @property
    def absmax(self):
        return max(abs(self.lo), abs(self.hi))

    def json(self):
        return bounds(self.lo, self.hi)


class Polynomial:

    def __init__(self, coefficients=None):
        if isinstance(coefficients, Polynomial):
            coefficients = coefficients.c
        if isinstance(coefficients, (int, str, F)):
            coefficients = {(0, 0): F(coefficients)}
        self.c = {tuple(k): F(v) for k, v in (coefficients or {}).items() if v}
        if any((i < 0 or j < 0 for i, j in self.c)):
            raise ValueError("Negative polynomial exponent")

    def __add__(self, other):
        other = Polynomial(other)
        out = dict(self.c)
        for key, value in other.c.items():
            out[key] = out.get(key, F(0)) + value
        return Polynomial(out)

    __radd__ = __add__

    def __neg__(self):
        return Polynomial({k: -v for k, v in self.c.items()})

    def __sub__(self, other):
        return self + -Polynomial(other)

    def __rsub__(self, other):
        return Polynomial(other) + -self

    def __mul__(self, other):
        other = Polynomial(other)
        out = {}
        for (i, j), v in self.c.items():
            for (k, l), w in other.c.items():
                key = (i + k, j + l)
                out[key] = out.get(key, F(0)) + v * w
        return Polynomial(out)

    __rmul__ = __mul__

    def __pow__(self, n):
        if n < 0:
            raise ValueError("Negative polynomial power")
        out = Polynomial(1)
        for _ in range(n):
            out *= self
        return out

    def derivative(self, axis):
        out = {}
        for (i, j), v in self.c.items():
            power = i if axis == "y" else j
            if power:
                out[(i - 1, j) if axis == "y" else (i, j - 1)] = power * v
        return Polynomial(out)

    def divide_y(self):
        if any((i == 0 for i, _ in self.c)):
            raise ValueError("Axis residual is not exactly divisible by y")
        return Polynomial({(i - 1, j): v for (i, j), v in self.c.items()})

    def at(self, axis, value):
        value, out = (F(value), {})
        for (i, j), v in self.c.items():
            key = (0, j) if axis == "y" else (i, 0)
            out[key] = out.get(key, F(0)) + v * value ** (i if axis == "y" else j)
        return Polynomial(out)

    def value(self, y, t):
        y, t = (F(y), F(t))
        return sum((v * y**i * t**j for (i, j), v in self.c.items()), F(0))

    def key(self):
        return tuple(sorted(((i, j, v.numerator, v.denominator) for (i, j), v in self.c.items())))

    def json(self):
        return [[i, j, str(v)] for (i, j), v in sorted(self.c.items())]

    def bounds(self, rectangle):
        coefficients = bernstein_coefficients(self.c, rectangle)
        return Interval(min(coefficients), max(coefficients))


def bernstein_coefficients(power_coefficients, rectangle):
    """将有理二元幂系数在指定矩形上精确转换成 Bernstein 系数。"""
    ya, yb, ta, tb = map(F, rectangle)
    if ya > yb or ta > tb:
        raise ValueError("Reversed polynomial rectangle")
    if not power_coefficients:
        return [F(0)]
    ny, nt = (max((i for i, _ in power_coefficients)), max((j for _, j in power_coefficients)))
    power = {}
    for (i, j), value in power_coefficients.items():
        for a in range(i + 1):
            for b in range(j + 1):
                v = (
                    value
                    * comb(i, a)
                    * ya ** (i - a)
                    * (yb - ya) ** a
                    * comb(j, b)
                    * ta ** (j - b)
                    * (tb - ta) ** b
                )
                power[a, b] = power.get((a, b), F(0)) + v
    coefficients = []
    for i in range(ny + 1):
        for j in range(nt + 1):
            coefficients.append(
                sum(
                    (
                        v * F(comb(i, a), comb(ny, a)) * F(comb(j, b), comb(nt, b))
                        for (a, b), v in power.items()
                        if a <= i and b <= j
                    ),
                    F(0),
                )
            )
    return coefficients


def exp_negative_range(x):
    x = Interval(x)
    if x.lo < 0:
        raise ValueError("This frozen exponential routine requires nonnegative x")
    lower, _ = exp_negative(x.hi)
    _, upper = exp_negative(x.lo)
    return Interval(lower, upper)


@dataclass(frozen=True)
class DiffusionBounds:
    prefactor: Interval
    moisture_exponent: Interval
    temperature_exponent: Interval
    name: str

    def __post_init__(self):
        if (
            self.prefactor.lo <= 0
            or self.moisture_exponent.lo < 0
            or self.temperature_exponent.lo < 0
        ):
            raise ValueError(
                "This comparison implementation requires D>0 and nonnegative registered moisture/temperature exponents"
            )

    def value(self, C, T=None):
        C = Interval(C)
        if C.lo <= 0:
            raise ValueError("Nonpositive concentration interval")
        x = self.moisture_exponent / C
        if self.temperature_exponent.hi:
            if T is None or Interval(T).lo <= 0:
                raise ValueError("A genuinely enclosing positive temperature interval is required")
            x += self.temperature_exponent / Interval(T)
        return self.prefactor * exp_negative_range(x)

    def derivative_C_upper(self, C, T=None):
        C = Interval(C)
        return (self.value(C, T) * self.moisture_exponent / C.square()).hi


Y, TIME = (Polynomial({(1, 0): 1}), Polynomial({(0, 1): 1}))


class DiffusionExpression:
    """Finite sum coefficient(y,t)*D(Cpoly(y,t),Tspec).

    Structural cancellation is exact algebra, before interval evaluation.
    Distinct temperature specifications never cancel for a T-dependent law.
    """

    def __init__(self, diffusion):
        self.diffusion, self.terms = (diffusion, {})

    def add(self, coefficient, concentration, temperature="actual"):
        coefficient = Polynomial(coefficient)
        if not coefficient.c:
            return self
        tkey = (
            ("irrelevant",)
            if self.diffusion.temperature_exponent.hi == 0
            else (
                ("actual",) if temperature == "actual" else ("fixed_polynomial", temperature.key())
            )
        )
        key = (concentration.key(), tkey)
        if key in self.terms:
            old = self.terms[key]
            coefficient = old[0] + coefficient
        if coefficient.c:
            self.terms[key] = (coefficient, concentration, temperature)
        elif key in self.terms:
            del self.terms[key]
        return self

    def subtract(self, other):
        for coefficient, concentration, temperature in other.terms.values():
            self.add(-coefficient, concentration, temperature)
        return self

    def bounds(self, rectangle, actual_temperature):
        grouped = {}
        for coefficient, concentration, temperature in self.terms.values():
            if self.diffusion.temperature_exponent.hi == 0:
                key = (concentration.key(), ("irrelevant",))
            elif temperature == "actual":
                tkey = (
                    ("point", actual_temperature.lo)
                    if actual_temperature.lo == actual_temperature.hi
                    else ("actual",)
                )
                key = (concentration.key(), tkey)
            else:
                values = temperature.bounds(rectangle)
                tkey = (
                    ("point", values.lo)
                    if values.lo == values.hi
                    else ("fixed_polynomial", temperature.key())
                )
                key = (concentration.key(), tkey)
            if key in grouped:
                coefficient = coefficient + grouped[key][0]
            grouped[key] = (coefficient, concentration, temperature)
        total = Interval(0)
        for coefficient, concentration, temperature in grouped.values():
            if not coefficient.c:
                continue
            T = actual_temperature if temperature == "actual" else temperature.bounds(rectangle)
            total += coefficient.bounds(rectangle) * self.diffusion.value(
                concentration.bounds(rectangle), T
            )
        return total


@dataclass
class MassSource:
    interior_plain: Polynomial
    interior_flux: DiffusionExpression
    boundary_plain: Polynomial
    boundary_beta: Polynomial
    description: str

    @classmethod
    def zero(cls, diffusion):
        return cls(
            Polynomial(),
            DiffusionExpression(diffusion),
            Polynomial(),
            Polynomial(),
            "Original homogeneous PDE; S=G=0.",
        )


def exact_equal(a, b):
    return not (a - b).c


def validate_patchwork(patches):
    """One time slab; exact axis and M/M_y/M_yy matching at every spatial seam."""
    if not patches or patches[0]["y0"] != 0 or patches[-1]["y1"] != F(1, 2):
        raise ValueError("Patchwork must cover exactly y in [0,1/2]")
    if patches[0]["M"].at("y", 0).c:
        raise ValueError("M(0,t) is not the exact zero polynomial")
    for i, p in enumerate(patches):
        if p["y0"] >= p["y1"]:
            raise ValueError("Nonpositive spatial cell")
        if i:
            a = patches[i - 1]
            if a["y1"] != p["y0"]:
                raise ValueError("Spatial coverage gap/overlap")
            left, right = (a["M"], p["M"])
            for derivative in range(3):
                if not exact_equal(left.at("y", p["y0"]), right.at("y", p["y0"])):
                    raise ValueError("Spatial C2 trace mismatch at derivative " + str(derivative))
                left, right = (left.derivative("y"), right.derivative("y"))
    return True


def validate_time_seam(previous, following, time):
    """A time derivative jump is allowed, but the whole M trace must agree."""
    if len(previous) != len(following):
        raise ValueError("This minimal seam checker needs the same fixed y grid")
    for a, b in zip(previous, following):
        if (a["y0"], a["y1"]) != (b["y0"], b["y1"]) or not exact_equal(
            a["M"].at("t", time), b["M"].at("t", time)
        ):
            raise ValueError("Time seam does not inherit the entire cumulative field")
    return True


def continuous_residual_bounds(
    M, diffusion, radius, hm, Ceq, source, rectangle, actual_temperature, error_cap
):
    slope, curvature = (M.derivative("y"), M.derivative("y").derivative("y"))
    mt = M.derivative("t")
    try:
        plain_bound = (mt.divide_y() - source.interior_plain).bounds(rectangle)
    except ValueError:
        if rectangle[0] <= 0:
            raise
        plain_bound = mt.bounds(rectangle) / Interval(
            rectangle[0], rectangle[1]
        ) - source.interior_plain.bounds(rectangle)
    flux = DiffusionExpression(diffusion).add(-2 * curvature, slope).subtract(source.interior_flux)
    R = radius.bounds(rectangle)
    if R.lo <= 0:
        raise ValueError("Nonpositive prescribed radius")
    residual = plain_bound + flux.bounds(rectangle, actual_temperature) / R.square()
    q, kappa = (slope.bounds(rectangle), curvature.bounds(rectangle))
    if q.lo <= error_cap:
        raise ValueError("The entire lower-barrier slope tube must stay in C>0")
    DC = diffusion.derivative_C_upper(
        Interval(q.lo - error_cap, q.hi + error_cap), actual_temperature
    )
    K = 2 * max(F(0), kappa.hi) * DC / R.lo**2
    return {
        "residual_over_y": residual,
        "rho": residual.absmax,
        "K": K,
        "slope": q,
        "curvature": kappa,
        "radius": R,
        "concavity_certified": kappa.hi <= 0,
    }


def phi(a):
    a = F(a)
    if not F(0) <= a < 1:
        raise ValueError("This proof requires 0 <= a < 1 and increasing phi")
    return Y - a * Y * Y


def exponential_enclosure(e0, rho, rate, dt):
    """Enclose the exact solution E'=rho-rate*E over one constant slab.

    E is monotone towards its nonnegative equilibrium, or increasing when
    rate<=0. Its endpoint upper bound plus E(0) controls the full time slab.
    The next slab inherits the upper endpoint, which is an outward jump.
    """
    e0, rho, rate, dt = map(F, (e0, rho, rate, dt))
    if e0 < 0 or rho < 0 or dt <= 0:
        raise ValueError("Invalid initial error, residual or physical duration")
    if rate == 0:
        value = Interval(e0 + rho * dt)
        multiplier = Interval(1)
    elif rate > 0:
        multiplier = Interval(*exp_negative(rate * dt))
        stationary = rho / rate
        value = Interval(stationary) + (e0 - stationary) * multiplier
    else:
        multiplier = 1 / Interval(*exp_negative(-rate * dt))
        value = e0 * multiplier + rho / -rate * (multiplier - 1)
    endpoint = up(value.hi)
    if endpoint < 0:
        raise AssertionError("Outward endpoint contradicts nonnegative ODE")
    return (
        endpoint,
        {
            "e0_exact": str(e0),
            "rho_exact": str(rho),
            "rate_exact": str(rate),
            "duration_exact": str(dt),
            "multiplier_exact_interval": [str(multiplier.lo), str(multiplier.hi)],
            "endpoint_before_rounding_exact_interval": [str(value.lo), str(value.hi)],
            "e1_upper_exact": str(endpoint),
            "slab_error_upper_exact": str(max(e0, endpoint)),
            "meaning": "Exact constant-coefficient scalar comparison ODE, with outward exponential enclosure; not a sampled error ODE.",
        },
    )


def advance_dissipative(
    patches,
    diffusion,
    radius,
    hm,
    Ceq,
    source,
    time0,
    time1,
    initial_error,
    error_cap,
    temperature_bounds,
    shape_parameter,
):
    a, t0, t1, e0, cap = map(F, (shape_parameter, time0, time1, initial_error, error_cap))
    shape = phi(a)
    if t0 >= t1 or not 0 <= e0 <= cap or cap <= 0:
        raise ValueError("Invalid physical time interval or inherited error/cap")
    validate_patchwork(patches)
    rows = []
    rho, rate = (F(0), None)
    for p in patches:
        rect = (p["y0"], p["y1"], t0, t1)
        quotient = (Polynomial(1) - a * Y).bounds(rect)
        derivative = shape.derivative("y").bounds(rect)
        assert quotient.lo > 0 and derivative.lo > 0
        T = temperature_bounds(rect)
        fields = continuous_residual_bounds(
            p["M"], diffusion, radius, hm, Ceq, source, rect, T, cap * derivative.hi
        )
        q, kappa, R = (fields["slope"], fields["curvature"], fields["radius"])
        C_tube = Interval(q.lo - cap * derivative.hi, q.hi + cap * derivative.hi)
        if C_tube.lo <= 0:
            raise ValueError("A barrier slope may leave the physical C>0 domain")
        D_min = diffusion.value(C_tube, T).lo
        DC_max = diffusion.derivative_C_upper(C_tube, T)
        dissipation = 4 * a * D_min / (R.hi**2 * quotient.hi)
        harmful = 2 * derivative.hi * max(F(0), kappa.hi) * DC_max / (R.lo**2 * quotient.lo)
        local_rate = dissipation - harmful
        local_residual = fields["residual_over_y"] / quotient
        local_rho = local_residual.absmax
        rho = max(rho, local_rho)
        rate = local_rate if rate is None else min(rate, local_rate)
        rows.append(
            {
                "rectangle_exact": list(map(str, rect)),
                "residual_over_phi_exact": [
                    str(local_residual.lo),
                    str(local_residual.hi),
                ],
                "rho_exact": str(local_rho),
                "rate_lower_exact": str(local_rate),
                "dissipation_lower_exact": str(dissipation),
                "harmful_upper_exact": str(harmful),
                "slope_tube_exact": [str(C_tube.lo), str(C_tube.hi)],
                "D_lower_exact": str(D_min),
                "DC_upper_exact": str(DC_max),
            }
        )
    rim = patches[-1]["M"]
    rect = (F(1, 2), F(1, 2), t0, t1)
    beta = hm / radius.bounds(rect)
    boundary = (rim.derivative("t") - source.boundary_plain).bounds(rect)
    boundary += beta * (rim.derivative("y") - Ceq - source.boundary_beta).bounds(rect)
    phi_b, derivative_b = (F(1, 2) - a / 4, 1 - a)
    boundary_rate = beta.lo * derivative_b / phi_b
    boundary_rho = boundary.absmax / phi_b
    rate, rho = (min(rate, boundary_rate), max(rho, boundary_rho))
    e1, scalar = exponential_enclosure(e0, rho, rate, t1 - t0)
    if max(e0, e1) > cap:
        raise ValueError("The complete comparison tube exceeds the declared cap")
    return (
        e1,
        {
            "time_s_exact": [str(t0), str(t1)],
            "shape_parameter_exact": str(a),
            "shape": "phi(y)=y*(1-a*y)",
            "e0_exact": str(e0),
            "e1_upper_exact": str(e1),
            "error_cap_exact": str(cap),
            "rho_exact": str(rho),
            "rate_lower_exact": str(rate),
            "boundary_rate_lower_exact": str(boundary_rate),
            "boundary_rho_exact": str(boundary_rho),
            "boundary_residual_exact": [str(boundary.lo), str(boundary.hi)],
            "rows": rows,
            "scalar_comparison": scalar,
            "continuous_space_and_time_coefficients": True,
            "requires_prior_initial_enclosure": True,
            "production_PDE_started": False,
        },
    )


def f64(x):
    return F.from_float(float(x))


def combine(a, b, scale_a=F(1), scale_b=F(1)):
    return [scale_a * x + scale_b * y for x, y in zip(a, b)]


def scaled(a, x):
    return [v * x for v in a]


def cumulative_cells(node_coefficients, radii, R):
    radius = f64(R)
    y = [(f64(r) / radius) ** 2 / 2 for r in radii]
    n = len(y) - 1
    if n < 2 or y[0] != 0 or y[-1] != F(1, 2) or (len(node_coefficients) != n + 1):
        raise ValueError("Full exact axis-to-surface nodal polynomial field required")
    if any((a >= b for a, b in zip(y[:-1], y[1:]))):
        raise ValueError("Nonincreasing spatial grid")
    weights, slopes = ([], [])
    for i in range(n + 1):
        lo, hi = (0, 1) if i == 0 else (n - 1, n) if i == n else (i - 1, i + 1)
        weight = f64(1 / (y[hi] - y[lo]))
        weights.append(weight)
        slopes.append(combine(node_coefficients[hi], node_coefficients[lo], weight, -weight))
    prefix = [F(0)] * 4
    cells = []
    for i in range(n):
        h = y[i + 1] - y[i]
        c0, c1 = (node_coefficients[i], node_coefficients[i + 1])
        m0, m1 = (slopes[i], slopes[i + 1])
        delta = combine(c1, c0, 1 / h, -1 / h)
        A = [(3 * d - 2 * a - b) / h for d, a, b in zip(delta, m0, m1)]
        B = [(a + b - 2 * d) / h**2 for d, a, b in zip(delta, m0, m1)]
        assert [c + m * h + a * h**2 + b * h**3 for c, m, a, b in zip(c0, m0, A, B)] == c1
        assert [m + 2 * a * h + 3 * b * h**2 for m, a, b in zip(m0, A, B)] == m1
        increment = [h * (a + b) / 2 + h * h * (u - v) / 12 for a, b, u, v in zip(c0, c1, m0, m1)]
        cells.append(
            {
                "y0": y[i],
                "y1": y[i + 1],
                "M_prefix_u_coefficients": prefix[:],
                "C_local_y_power_coefficients_each_in_u": [c0[:], m0[:], A, B],
            }
        )
        prefix = combine(prefix, increment)
    return (
        cells,
        {
            "space_C1_and_cumulative_C2_traces_checked": True,
            "axis_cumulative_exact_zero": True,
            "derivative_rule_linear_in_all_nodal_time_coefficients": True,
            "fixed_weights_exact": list(map(str, weights)),
            "monotonicity_guaranteed": False,
            "global_K_zero_claimed": False,
            "time_continuity_condition": "Adjacent nodal time-polynomial endpoints must equal the same accepted states; fixed linear slopes and integration then preserve entire C/M traces.",
            "PDE_residual_bound_computed": False,
        },
    )


def array_sha(x):
    return hashlib.sha256(np.asarray(x, dtype="<f8").tobytes(order="C")).hexdigest()


def native_record(result, left, right):
    h = hashlib.sha256()
    for item in result.sol.interpolants:
        for x in (item.y_old, item.Q):
            h.update(np.asarray(x, dtype="<f8").tobytes(order="C"))
    return {
        "left_s": float(left),
        "right_s": float(right),
        "steps": len(result.t) - 1,
        "nfev": int(result.nfev),
        "njev": int(result.njev),
        "nlu": int(result.nlu),
        "accepted_times_sha256": array_sha(result.t),
        "accepted_states_time_major_sha256": array_sha(result.y.T),
        "dense_y_old_then_Q_sha256": h.hexdigest(),
    }


class DenseTrajectory:

    def __init__(self, folder):
        self.folder = Path(folder)
        self.record = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        r = self.record
        if (
            r.get("completed") is not True
            or r.get("permanent_retention") is not True
            or r.get("cleanup_allowed") is not False
        ):
            raise ValueError("Incomplete/nonpermanent dense span")
        names = {
            "accepted_t.npy",
            "accepted_C.npy",
            "Q.npy",
            "y_old.npy",
            "t_old.npy",
            "t.npy",
            "h.npy",
            "t_min.npy",
            "t_max.npy",
            "order.npy",
        }
        if set(r["arrays_sha256"]) != names:
            raise ValueError("Required permanent polynomial fields absent or extra")
        for name, digest in r["arrays_sha256"].items():
            if sha256(self.folder / name) != digest:
                raise ValueError("Dense array SHA changed: " + name)
        if (
            sha256(inspect.getsourcefile(RadauDenseOutput)) != r["RadauDenseOutput_source_sha256"]
            or sha256(inspect.getsourcefile(OdeSolution)) != r["OdeSolution_source_sha256"]
        ):
            raise ValueError("Actual polynomial evaluator source changed")
        self.arrays = {
            name[:-4]: np.load(self.folder / name, mmap_mode="r", allow_pickle=False)
            for name in names
        }
        z = self.arrays
        nstep, nodes = (r["native"]["steps"], r["nodes"])
        if (
            z["accepted_C"].shape != (nstep + 1, nodes)
            or z["Q"].shape != (nstep, nodes, 3)
            or z["y_old"].shape != (nstep, nodes)
        ):
            raise ValueError("Dense array shape mismatch")
        if not all((np.isfinite(v).all() for v in z.values())) or not np.all(
            np.diff(z["accepted_t"]) > 0
        ):
            raise ValueError("Nonfinite or nonincreasing permanent history")
        if not np.array_equal(z["accepted_t"][:-1], z["t_old"]) or not np.array_equal(
            z["accepted_t"][1:], z["t"]
        ):
            raise ValueError("Dense physical endpoint association failed")
        if not np.array_equal(z["accepted_C"][:-1], z["y_old"]) or not np.all(z["order"] == 2):
            raise ValueError("Accepted-left/dense identity or polynomial order failed")
        if (
            z["accepted_t"][0] != r["native"]["left_s"]
            or z["accepted_t"][-1] != r["native"]["right_s"]
        ):
            raise ValueError("Span domain differs from native record")
        items = []
        for i in range(nstep):
            item = RadauDenseOutput(
                float(z["t_old"][i]), float(z["t"][i]), z["y_old"][i], z["Q"][i]
            )
            if any((getattr(item, key) != z[key][i] for key in ("h", "t_min", "t_max", "order"))):
                raise ValueError("Serialized Radau scalar metadata mismatch")
            items.append(item)
        if r["native_OdeSolution_ascending"] is not True or r["native_OdeSolution_side"] not in (
            "left",
            "right",
        ):
            raise ValueError("Unsupported native solution orientation")
        self.sol = OdeSolution(
            z["accepted_t"], items, alt_segment=r["native_OdeSolution_side"] == "right"
        )

    def __call__(self, time):
        t = np.asarray(time)
        if (
            not np.isfinite(t).all()
            or np.min(t) < self.arrays["accepted_t"][0]
            or np.max(t) > self.arrays["accepted_t"][-1]
        ):
            raise ValueError("Query outside the stored physical domain")
        return self.sol(t)

    @staticmethod
    def _write_native_arrays(folder, result, nstep, nodes):
        """核对每步原生 Radau 多项式并保存接受节点、系数与标量元数据。"""
        np.save(folder / "accepted_t.npy", result.t)
        np.save(folder / "accepted_C.npy", np.asarray(result.y.T))
        q = np.lib.format.open_memmap(
            folder / "Q.npy", mode="w+", dtype="<f8", shape=(nstep, nodes, 3)
        )
        old = np.lib.format.open_memmap(
            folder / "y_old.npy", mode="w+", dtype="<f8", shape=(nstep, nodes)
        )
        scalar = {
            key: np.empty(nstep, dtype=float) for key in ("t_old", "t", "h", "t_min", "t_max")
        }
        orders = np.empty(nstep, dtype=np.int64)
        native_endpoint_defect = 0.0
        for i, item in enumerate(result.sol.interpolants):
            if type(item) is not RadauDenseOutput or item.Q.shape != (nodes, 3):
                raise ValueError("Unexpected dense basis/type; do not guess its polynomial")
            if not np.array_equal(item.y_old, result.y[:, i]):
                raise ValueError("Native dense y_old does not equal its accepted left state")
            if item.t_old != result.t[i] or item.t != result.t[i + 1]:
                raise ValueError("Native polynomial endpoints differ from accepted step")
            q[i], old[i] = (item.Q, item.y_old)
            for key in scalar:
                scalar[key][i] = getattr(item, key)
            orders[i] = item.order
            native_endpoint_defect = max(
                native_endpoint_defect,
                float(np.max(np.abs(item(item.t) - result.y[:, i + 1]))),
            )
        q.flush()
        old.flush()
        del q, old
        for key, v in scalar.items():
            np.save(folder / (key + ".npy"), v)
        np.save(folder / "order.npy", orders)
        return native_endpoint_defect

    @staticmethod
    def save(folder, result, left, right):
        folder = Path(folder)
        if folder.exists():
            raise FileExistsError(folder)
        if not result.success:
            raise ValueError("Failed solver result cannot become a completed dense span")
        folder.mkdir(parents=True)
        nstep, nodes = (len(result.t) - 1, result.y.shape[0])
        if (
            nstep <= 0
            or len(result.sol.interpolants) != nstep
            or (not np.array_equal(result.sol.ts, result.t))
        ):
            raise ValueError("Native step/dense coverage differs")
        native_endpoint_defect = DenseTrajectory._write_native_arrays(folder, result, nstep, nodes)
        files = {p.name: sha256(p) for p in sorted(folder.glob("*.npy"))}
        record = {
            "completed": True,
            "permanent_retention": True,
            "cleanup_allowed": False,
            "schema": "native-radau-moisture-dense-v1",
            "nodes": nodes,
            "accepted_rows": nstep + 1,
            "native_OdeSolution_side": result.sol.side,
            "native_OdeSolution_ascending": bool(result.sol.ascending),
            "native": native_record(result, left, right),
            "arrays_sha256": files,
            "RadauDenseOutput_source_sha256": sha256(inspect.getsourcefile(RadauDenseOutput)),
            "OdeSolution_source_sha256": sha256(inspect.getsourcefile(OdeSolution)),
            "basis": "C(t)=y_old+Q@[x,x^2,x^3], x=(t-t_old)/h; there is no additional h multiplier",
            "native_float_endpoint_to_accepted_state_max_abs": native_endpoint_defect,
            "no_claim": "Preserved numerical polynomial/state data are not a continuous PDE error bound.",
        }
        write_json(folder / "manifest.json", record)
        return record

    def polynomials(self, step):
        z = self.arrays
        left, right = (f64(z["t_old"][step]), f64(z["t"][step]))
        H, native_h = (right - left, f64(z["h"][step]))
        if H <= 0 or native_h <= 0:
            raise ValueError("Positive exact physical interval required")
        ratio = H / native_h
        result, corrections = ([], [])
        for node in range(z["y_old"].shape[1]):
            old = f64(z["y_old"][step, node])
            q = [f64(z["Q"][step, node, k]) * ratio ** (k + 1) for k in range(3)]
            target = f64(z["accepted_C"][step + 1, node])
            correction = target - old - sum(q)
            coeff = [old, q[0] + correction, q[1], q[2]]
            assert coeff[0] == f64(z["accepted_C"][step, node]) and sum(coeff) == target
            result.append(coeff)
            corrections.append(correction)
        return (
            result,
            {
                "physical_left_exact": str(left),
                "physical_right_exact": str(right),
                "physical_duration_exact": str(H),
                "native_h_exact": str(native_h),
                "basis": "u=(t-left)/(right-left); nodal C is degree<=3 in u",
                "endpoint_correction_max_abs_exact": str(max(map(abs, corrections), default=F(0))),
                "endpoint_correction_bounds_entire_interval": True,
                "raw_native_data_changed": False,
                "scope": "Correction bounds the mathematical native-polynomial change, not all floating evaluation roundoff and not PDE error.",
            },
        )


def ball(q):
    q = F(q)
    return flint_runtime().arb(str(q.numerator)) / flint_runtime().arb(str(q.denominator))


def dyadic(x):
    if not x.is_finite() or not x.is_exact():
        raise ValueError("A finite exact outward endpoint is required")
    m, e = map(int, x.man_exp())
    return F(m * (1 << e)) if e >= 0 else F(m, 1 << -e)


def endpoints(x):
    if not x.is_finite():
        raise ValueError("Nonfinite Arb enclosure")
    return (dyadic(x.lower()), dyadic(x.upper()))


class KirchhoffTransform:
    """Kirchhoff势变换：采用最终Arb闭式原函数及向外取整的区间Newton逆变换。"""

    def __init__(self, C0, q, prefactor, precision=256, panels=16):
        self.C0, self.q, self.prefactor = map(F, (C0, q, prefactor))
        self.precision, self.panels = (precision, panels)
        if (
            self.C0 <= 0
            or self.q < 0
            or self.prefactor <= 0
            or (not isinstance(precision, int))
            or (precision < 128)
            or (panels < 2)
            or panels % 2
        ):
            raise ValueError(
                "Positive concentration/diffusivity, 128-bit precision and even panels required"
            )

    def slope(self, C):
        if C.lo <= 0:
            raise ValueError("Positive concentration interval required")
        return Interval(self.slope_point(C.lo).lo, self.slope_point(C.hi).hi)

    def inverse(self, U, domain, iterations=10):
        """Interval Newton using strictly positive F'; no empirical root bound."""
        box = Interval(domain.lo, domain.hi)
        if box.lo <= 0 or self.primitive(box.lo).hi > U.lo or self.primitive(box.hi).lo < U.hi:
            raise ValueError("Potential range not enclosed by the declared positive C domain")
        for _ in range(iterations):
            mid = (box.lo + box.hi) / 2
            image = mid + (U - self.primitive(mid)) / self.slope(box)
            raw_lo = max(box.lo, image.lo)
            raw_hi = min(box.hi, image.hi)
            if raw_lo > raw_hi:
                raise ValueError("Empty rigorous inverse interval")
            fresh = Interval(max(box.lo, down(raw_lo)), min(box.hi, up(raw_hi)))
            if fresh.lo == box.lo and fresh.hi == box.hi:
                break
            box = fresh
        return box

    def diffusion(self, C):
        return self.prefactor * self.slope(C)

    def potential_derivative_diffusion(self, C):
        return self.prefactor * self.q / (C * C)

    @lru_cache(maxsize=32768)
    def slope_point(self, C):
        C = F(C)
        if C <= 0:
            raise ValueError("Kirchhoff primitive outside positive C")
        if self.q == 0:
            return Interval(1)
        with flint_runtime().ctx.workprec(self.precision):
            return Interval(*endpoints((-ball(self.q) / ball(C)).exp()))

    @lru_cache(maxsize=8192)
    def primitive(self, C, panels=None):
        C = F(C)
        n = self.panels if panels is None else panels
        if C <= 0 or n < 2 or n % 2:
            raise ValueError("Positive primitive argument and even panels required")
        if C == self.C0:
            return Interval(0)
        if self.q == 0:
            return Interval(C - self.C0)
        with flint_runtime().ctx.workprec(self.precision):
            q = ball(self.q)

            def H(v):
                c = ball(v)
                z = -q / c
                return c * z.exp() + q * z.ei()

            return Interval(*endpoints(H(C) - H(self.C0)))


def time_interpolant(values, knots):
    if len(values) != len(knots) or len(set(knots)) != len(knots):
        raise ValueError("Distinct complete temporal nodes required")
    result = Polynomial()
    for i, value in enumerate(values):
        basis = Polynomial(1)
        for j, k in enumerate(knots):
            if i != j:
                basis *= (TIME - k) * (1 / (knots[i] - k))
        result += value * basis
    return result


def quintic(u0, u1, m0, m1, k0, k1, h):
    z = Y
    return (
        (1 - 10 * z**3 + 15 * z**4 - 6 * z**5) * u0
        + (10 * z**3 - 15 * z**4 + 6 * z**5) * u1
        + (z - 6 * z**3 + 8 * z**4 - 3 * z**5) * (h * m0)
        + (-4 * z**3 + 7 * z**4 - 3 * z**5) * (h * m1)
        + (z * z - 3 * z**3 + 3 * z**4 - z**5) * (h * h * F(1, 2) * k0)
        + (z**3 - 2 * z**4 + z**5) * (h * h * F(1, 2) * k1)
    )


def nodal_derivatives(U, x, rim_slope):
    n = len(x) - 1
    m = [Polynomial() for _ in x]
    k = [Polynomial() for _ in x]
    k[0] = (U[1] - U[0]) * (2 / (x[1] - x[0]) ** 2)
    for i in range(1, n):
        hl, hr = (x[i] - x[i - 1], x[i + 1] - x[i])
        m[i] = (
            U[i - 1] * (-hr / (hl * (hl + hr)))
            + U[i] * ((hr - hl) / (hl * hr))
            + U[i + 1] * (hl / (hr * (hl + hr)))
        )
        k[i] = ((U[i + 1] - U[i]) * (1 / hr) - (U[i] - U[i - 1]) * (1 / hl)) * (2 / (hl + hr))
    m[-1] = rim_slope
    k[-1] = k[-2]
    return (m, k)


def validate_cell(p, u0, u1, m0, m1, k0, k1, h):
    for level, ends in ((0, (u0, u1)), (1, (m0, m1)), (2, (k0, k1))):
        q = p
        for _ in range(level):
            q = q.derivative("y") * (1 / h)
        for z, end in enumerate(ends):
            if (q.at("y", z) - end).c:
                raise ValueError("Hermite exact endpoint jet failed")


class GaussianEnvelope:

    def __init__(self, tr, R, domain, v0, eta=F(1, 10**12), factor=F(9, 2)):
        self.tr = tr
        self.R, self.v0, self.eta = map(F, (R, v0, eta))
        if self.R <= 0 or self.v0 <= 0 or self.eta <= 0 or (factor <= 4) or (domain.lo <= 0):
            raise ValueError("Positive full envelope domain/length/floor and factor>4 required")
        self.amin = tr.diffusion(domain).lo / self.R**2
        self.amax = tr.diffusion(domain).hi / self.R**2
        self.kappa = factor * self.amax
        self.p = 2 * self.amin / self.kappa
        self.gamma = 16 * self.amax**2 / (self.kappa - 4 * self.amax)
        self.apmax = tr.prefactor * tr.q / (self.R**2 * domain.lo**2)

    def b(self, v):
        return ball(F(v))

    def box(self, lo, hi):
        return self.b((lo + hi) / 2) + flint_runtime().arb(0, self.b((hi - lo) / 2))

    def interval(self, b):
        return Interval(*endpoints(b))

    @lru_cache(maxsize=200000)
    def shape(self, xlo, xhi, tlo, thi):
        if not 0 <= xlo <= xhi <= 1 or not 0 <= tlo <= thi:
            raise ValueError("Physical envelope range")
        with flint_runtime().ctx.workprec(256):
            x = self.box(xlo, xhi)
            t = self.box(tlo, thi)
            v = self.b(self.v0) + self.b(self.kappa) * t
            d = (1 - x * x) / 2
            psi = (
                (self.b(self.v0) / v).log() * self.b(self.p) + self.b(self.gamma) * t - d * d / v
            ).exp() + self.b(self.eta)
            ans = self.interval(psi)
            return Interval(max(self.eta, ans.lo), ans.hi)

    def amplitude(self, A0, S, L, dt):
        if min(A0, S, L, dt) < 0:
            raise ValueError("Nonnegative envelope evolution")
        if L == 0:
            return Interval(A0 + S * dt)
        with flint_runtime().ctx.workprec(256):
            z = self.b(L) * self.b(dt)
            return self.interval(self.b(A0) * z.exp() + self.b(S) * z.expm1() / self.b(L))

    def parameters(self):
        return {
            k: str(getattr(self, k))
            for k in ("R", "v0", "eta", "amin", "amax", "kappa", "p", "gamma", "apmax")
        }

    def uniform_E_upper(self, Aend, t, left=F(0)):
        if not 0 <= left <= t or Aend < 0:
            raise ValueError("Envelope interval")
        with flint_runtime().ctx.workprec(256):
            ratio = self.b(self.v0) / (self.b(self.v0) + self.b(self.kappa) * self.b(left))
            upper = (ratio.log() * self.b(self.p) + self.b(self.gamma) * self.b(t)).exp() + self.b(
                self.eta
            )
            return self.interval(self.b(Aend) * upper).hi


def localized_coefficients(tr, env, p, native_cp, x0, x1, left, right, ta, tb, domain):
    """Whole local-coordinate rectangle z in [0,1], tau in [ta,tb]."""
    h = x1 - x0
    H = right - left
    rect = (F(0), F(1), ta, tb)
    U = p.bounds(rect)
    ps = p.derivative("y")
    pss = ps.derivative("y")
    if x0 == 0:
        ratio = ps.divide_y()
        lap = ((pss + ratio) * (1 / h**2)).bounds(rect)
    else:
        lap = pss.bounds(rect) / h**2 + ps.bounds(rect) / (Interval(x0, x1) * h)
    Ut = p.derivative("t").bounds(rect) / H
    cref = Interval(tr.C0) if not p.c else tr.inverse(U, domain)
    residual = Ut - tr.diffusion(cref) * lap / env.R**2
    physical_a = left + H * ta
    physical_b = left + H * tb
    psi = env.shape(x0, x1, physical_a, physical_b) if residual.absmax else Interval(env.eta)
    rho = residual.absmax
    L = env.apmax * max(F(0), lap.hi)
    source = rho / psi.lo
    initial = p.at("t", 0).bounds((0, 1, 0, 0)).absmax
    A0 = F(0) if initial == 0 else initial / env.shape(x0, x1, F(0), F(0)).lo
    if not (native_cp - Polynomial(tr.C0)).c and (not p.c):
        defect = F(0)
    else:
        C = native_cp.bounds(rect)
        midz = F(1, 2)
        midt = (ta + tb) / 2
        center = Interval(p.value(midz, midt)) - tr.primitive(native_cp.value(midz, midt))
        ds = ps.bounds(rect) - tr.slope(C) * native_cp.derivative("y").bounds(rect)
        dt = p.derivative("t").bounds(rect) - tr.slope(C) * native_cp.derivative("t").bounds(rect)
        defect = (
            center + Interval(-F(1, 2), F(1, 2)) * ds + Interval(-(tb - ta) / 2, (tb - ta) / 2) * dt
        ).absmax
    return {
        "U_min": U.lo,
        "U_max": U.hi,
        "rho": rho,
        "residual_lower": residual.lo,
        "residual_upper": residual.hi,
        "L": L,
        "source": source,
        "initial_A_required": A0,
        "potential_native_PL_defect": defect,
        "shape_lower": psi.lo,
        "positive_reference_C": cref.lo > 0,
    }


def inherit_trace(p, prior, left, prior_right, Ain, prior_A):
    if F(left) != F(prior_right):
        raise ValueError("Exact physical time gap/overlap")
    if F(Ain) < F(prior_A):
        raise ValueError("Inward amplitude reset")
    if (p.at("t", 0) - prior).c:
        raise ValueError("Entire spatial reference trace changed")


def query_at(tr, env, U, x, left, right, Ain, S, L, target, domain):
    target = F(target)
    if not left <= target <= right or right <= left:
        raise ValueError("Query outside original native step")
    tau = (target - left) / (right - left)
    amp = env.amplitude(Ain, S, L, target - left)
    rows = []
    N = len(x) - 1
    if N % 20:
        raise ValueError("Expected complete 21 uniform radius queries")
    for j in range(0, N + 1, N // 20):
        u = U[j].value(0, tau)
        E = (Interval(amp.hi) * env.shape(x[j], x[j], target, target)).hi
        C = tr.inverse(Interval(u - E, u + E), domain)
        kl = (10000 * C.lo + F(1, 2)).numerator // (10000 * C.lo + F(1, 2)).denominator
        ku = (10000 * C.hi + F(1, 2)).numerator // (10000 * C.hi + F(1, 2)).denominator
        rows.append(
            {
                "node": j,
                "x": str(x[j]),
                "t": str(target),
                "local_tau": str(tau),
                "C_exact": [str(C.lo), str(C.hi)],
                "unique_bin": kl == ku,
                "lower_bin": kl,
                "upper_bin": ku,
            }
        )
    return rows


class MoistureProblem:
    """Q1水分方程；径向网格、插值与有限体积算子直接来自原公共依赖。"""

    def __init__(self, inputs, contract, N):
        if N < 20 or N % 20:
            raise ValueError("N必须为不小于20的20倍数")
        self.config = read_json(contract)
        self.parameters = self.config["property_sets"]["P1"]
        initial = self.config["initial_state"]
        self.radius, self.C0 = initial["radius_m"], initial["moisture_C"]
        self.hm = self.config["fixed_domain_model"]["h_m_m_s"]
        if (
            self.parameters["D_temperature_exponent_K"] != 0
            or self.config["fixed_domain_model"]["moisture_boundary"]
            != "-D(C_s,T_s)*C_r = h_m*(C_s-C_eq)"
        ):
            raise ValueError("此水分推进要求P1扩散系数与温度无关")
        self.core = section("radial_core")
        self.grid = self.core.NodeRadialGrid(self.radius, N)
        self.environment = self.core.Environment(inputs, self.config)
        self.pattern = self.core.tridiagonal_sparsity(N + 1)
        self.first_step = min(
            0.01,
            0.1 * (self.radius / N) ** 2 / float(self.diffusivity(np.array([self.C0]))[0]),
        )

    def diffusivity(self, concentration):
        if np.any(concentration <= 0) or not np.isfinite(concentration).all():
            raise ValueError("试探含水率必须有限且为正，不能裁剪")
        return self.parameters["D_prefactor"] * np.exp(
            -self.parameters["D_moisture_exponent"] / concentration
        )

    def rhs(self, time, concentration):
        flux = self.hm * (concentration[-1] - self.environment(time)[1])
        return self.core.diffusion_rhs(
            self.grid,
            concentration,
            self.core.face_arithmetic(self.diffusivity(concentration)),
            flux,
        )

    def integrate(self, left, right, initial):
        result = solve_ivp(
            self.rhs,
            (left, right),
            initial,
            method="Radau",
            jac_sparsity=self.pattern,
            rtol=2e-12,
            atol=2e-14,
            dense_output=True,
            first_step=self.first_step if left == 0 else None,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result


def produce_reference(out, *, inputs=None, contract=None, N=20, end_s=1):
    """计算并永久保存原生Radau稠密多项式；不把数值轨迹标为PDE误差证明。"""
    from numpy.polynomial.legendre import leggauss

    defaults = paths()
    inputs, contract = Path(inputs or defaults["inputs"]), Path(contract or defaults["contract"])
    if int(end_s) != end_s or not 1 <= end_s <= 1800:
        raise ValueError("终止时间必须为1..1800整数秒")
    model = MoistureProblem(inputs, contract, N)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "dense").mkdir()
    (out / "full_integer_fields").mkdir()
    bindings = {
        str(p.resolve()): sha256(p)
        for p in [
            contract,
            inputs / "附件1.xlsx",
            inputs / "附件2.xlsx",
            Path(__file__),
        ]
    }
    bindings.update({v["path"]: v["sha256"] for v in source_identity()["files"].values()})
    times = np.arange(0.0, end_s + 1.0)
    indices = np.arange(0, N + 1, N // 20)
    concentration = np.full(N + 1, model.C0)
    output = np.full((len(times), 21), model.C0)
    mean = np.full(len(times), model.C0)
    records, fields, balances = [], [], []
    integrals = np.zeros(2)
    rules = [leggauss(q) for q in (8, 16)]
    np.savez(
        out / "grid.npz",
        r_m=model.grid.r_m,
        weights_m2=model.grid.weights_m2,
        faces_m=model.grid.faces_m,
    )
    cuts = model.environment.breakpoints(0, end_s)
    try:
        for span, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
            result = model.integrate(left, right, concentration)
            folder = out / "dense" / f"span_{span:03d}"
            record = DenseTrajectory.save(folder, result, left, right)
            stored = DenseTrajectory(folder)
            query = np.unique(np.r_[result.t, (result.t[:-1] + result.t[1:]) / 2])
            if not np.array_equal(result.sol(query), stored(query)):
                raise ValueError("重新读取的原生稠密轨迹与积分器不一致")
            records.append(
                {
                    "path": folder.relative_to(out).as_posix(),
                    "manifest_sha256": sha256(folder / "manifest.json"),
                    "native": record["native"],
                }
            )
            rows = np.flatnonzero((times > left) & (times <= right))
            values = result.sol(times[rows])
            output[rows] = values[indices].T
            mean[rows] = model.grid.weights_m2 @ values / (model.radius**2 / 2)
            field_time, field_value = times[rows], values.T
            if span == 0:
                field_time = np.r_[0.0, field_time]
                field_value = np.vstack([np.full((1, N + 1), model.C0), field_value])
            field = out / "full_integer_fields" / f"chunk_{span:03d}.npz"
            np.savez(field, t_s=field_time, r_m=model.grid.r_m, C=field_value)
            fields.append({"path": field.relative_to(out).as_posix(), "sha256": sha256(field)})
            for dense in result.sol.interpolants:
                for q, (nodes, weights) in enumerate(rules):
                    t = (dense.t_old + dense.t) / 2 + (dense.t - dense.t_old) / 2 * nodes
                    rate = -2 * model.hm / model.radius * (dense(t)[-1] - model.environment(t)[1])
                    integrals[q] += (dense.t - dense.t_old) / 2 * (weights @ rate)
            concentration = result.y[:, -1].copy()
            change = model.grid.weights_m2 @ concentration / (model.radius**2 / 2) - model.C0
            balances.append({"t_s": float(right), "residual_C": (change - integrals).tolist()})
        np.savez(
            out / "solution.npz",
            t_s=times,
            r_m=model.grid.r_m[indices],
            C=output,
            mean_C=mean,
        )
        write_json(out / "dense/index.json", {"completed": True, "spans": records})
        write_json(
            out / "full_integer_fields/manifest.json",
            {"completed": True, "chunks": fields},
        )
        if any(sha256(path) != digest for path, digest in bindings.items()):
            raise ValueError("计算期间实际源码或输入已改变")
        summary = {
            "schema": "merged-native-reference-v1",
            "completed": True,
            "N": N,
            "end_s": end_s,
            "settings": {"method": "Radau", "rtol": 2e-12, "atol_C": 2e-14},
            "input_paths": {
                "inputs": str(inputs.resolve()),
                "contract": str(contract.resolve()),
            },
            "bindings": bindings,
            "grid_sha256": sha256(out / "grid.npz"),
            "dense_index_sha256": sha256(out / "dense/index.json"),
            "accepted_steps": sum(r["native"]["steps"] for r in records),
            "water_balance": balances,
            "balance_passed": max(abs(v) for r in balances for v in r["residual_C"]) < 1e-7,
            "continuous_PDE_certified": False,
            "historical_acceptance_claimed": False,
        }
        write_json(out / "summary.json", summary)
        return summary
    except Exception as error:
        write_json(
            out / "summary.json",
            {
                "completed": False,
                "error": repr(error),
                "partial_outputs_preserved": True,
            },
        )
        raise


class NativeReference:
    """读取本实现生成的原生轨迹，严格检查覆盖、节点、输入与文件摘要。"""

    def __init__(self, folder, *, inputs=None, contract=None):
        self.folder = Path(folder)
        self.summary = read_json(self.folder / "summary.json")
        if self.summary.get("completed") is not True:
            raise ValueError("参考轨迹未完成")
        if self.summary.get("schema") != "merged-native-reference-v1":
            raise ValueError("请先produce生成本实现的轨迹；旧证明档案不能自动视为已重新验证")
        self._load_inputs_and_properties(inputs, contract)
        self._load_grid_and_validate_sources()
        self._load_and_validate_spans()
        self._load_environment()

    def _load_inputs_and_properties(self, inputs, contract):
        """绑定原题输入，提取固定域水分方程的精确参数。"""
        defaults = paths()
        source = self.summary.get("input_paths", {})
        self.inputs = Path(inputs or source.get("inputs") or defaults["inputs"])
        self.contract = Path(contract or source.get("contract") or defaults["contract"])
        self.config = read_json(self.contract)
        bindings = self.summary.get("bindings", {})
        for required in (
            self.contract,
            self.inputs / "附件1.xlsx",
            self.inputs / "附件2.xlsx",
        ):
            if bindings.get(str(required.resolve())) != sha256(required):
                raise ValueError("实际输入与参考轨迹未绑定：" + str(required))
        self.radius = F.from_float(float(self.config["initial_state"]["radius_m"]))
        self.C0 = F.from_float(float(self.config["initial_state"]["moisture_C"]))
        self.hm = F.from_float(float(self.config["fixed_domain_model"]["h_m_m_s"]))
        if self.config["property_sets"]["P1"]["D_temperature_exponent_K"] != 0:
            raise ValueError("当前证明仅覆盖温度无关的Q1水分方程")

    def _load_grid_and_validate_sources(self):
        """核对输入、源码和索引摘要，以及轴到表面的完整网格。"""
        for name, key in [
            ("grid.npz", "grid_sha256"),
            ("dense/index.json", "dense_index_sha256"),
        ]:
            if sha256(self.folder / name) != self.summary[key]:
                raise ValueError("轨迹索引或网格已改变：" + name)
        for path, digest in self.summary.get("bindings", {}).items():
            if sha256(path) != digest:
                raise ValueError("实际参考源码/输入已改变：" + path)
        with np.load(self.folder / "grid.npz", allow_pickle=False) as grid:
            self.radii = grid["r_m"].copy()
        if (
            len(self.radii) != self.summary["N"] + 1
            or self.radii[0] != 0
            or F.from_float(float(self.radii[-1])) != self.radius
            or not np.all(np.diff(self.radii) > 0)
        ):
            raise ValueError("参考网格必须完整覆盖轴到表面")

    def _load_and_validate_spans(self):
        """逐段检查原生稠密数据和完整节点初态、时间覆盖及状态继承。"""
        index = read_json(self.folder / "dense/index.json")
        if index.get("completed") is not True:
            raise ValueError("稠密轨迹索引未完成")
        self.spans = index["spans"]
        initial = None
        last = 0.0
        for row in self.spans:
            folder = (self.folder / row["path"]).resolve()
            if (
                not folder.is_relative_to(self.folder.resolve())
                or sha256(folder / "manifest.json") != row["manifest_sha256"]
            ):
                raise ValueError("稠密轨迹路径或摘要无效")
            span = DenseTrajectory(folder)
            if span.record["nodes"] != len(self.radii):
                raise ValueError("稠密轨迹节点数必须与完整径向网格一致")
            if span.arrays["accepted_t"][0] != last:
                raise ValueError("参考轨迹时间覆盖有间断")
            first = span.arrays["accepted_C"][0]
            if initial is None and not np.all(first == float(self.C0)):
                raise ValueError("必须从题定均匀初值开始，不能假定检查点无误差")
            if initial is not None and not np.array_equal(first, initial):
                raise ValueError("分段参考初态不继承前一段")
            initial = span.arrays["accepted_C"][-1].copy()
            last = float(span.arrays["accepted_t"][-1])
        if last != self.summary["end_s"]:
            raise ValueError("最终参考时间与完成记录不一致")

    def _load_environment(self):
        """保留附件中的 binary64 值并转换成精确有理数。"""
        import openpyxl

        book = openpyxl.load_workbook(self.inputs / "附件1.xlsx", read_only=True, data_only=True)
        try:
            self.ambient = [
                (F.from_float(float(t)), F.from_float(float(c)))
                for t, _, c in book["Sheet1"].iter_rows(min_row=2, max_col=3, values_only=True)
            ]
        finally:
            book.close()

    def environment(self, left, right):
        for (a, c0), (b, c1) in zip(self.ambient[:-1], self.ambient[1:]):
            if a <= left < right <= b:
                return Polynomial(c0) + (c1 - c0) / (b - a) * (TIME - a)
        raise ValueError("参考时间步跨过了原环境插值节点")

    def steps(self):
        for row in self.spans:
            folder = self.folder / row["path"]
            if sha256(folder / "manifest.json") != row["manifest_sha256"]:
                raise ValueError("推进期间稠密轨迹摘要已改变")
            stored = DenseTrajectory(folder)
            for step in range(stored.record["native"]["steps"]):
                coefficients, correction = stored.polynomials(step)
                yield coefficients, correction


def _serial(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: _serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(item) for item in value]
    return value


def _proof_sources(reference):
    files = [
        Path(__file__),
        reference.contract,
        reference.inputs / "附件1.xlsx",
        reference.folder / "summary.json",
        reference.folder / "grid.npz",
        reference.folder / "dense/index.json",
    ]
    files.extend(reference.folder / row["path"] / "manifest.json" for row in reference.spans)
    return {str(path.resolve()): sha256(path) for path in files}


def native_reference(reference):
    """接受已校验的参考对象，或按相同规则读取参考目录。"""
    return reference if isinstance(reference, NativeReference) else NativeReference(reference)


def write_step_record(folder, index, record):
    """保存一步证据并返回可绑定的相对路径和文件摘要。"""
    name = f"step_{index}.json"
    write_json(folder / name, record)
    return {"path": name, "sha256": sha256(folder / name)}


def verify_proof_sources(sources):
    """确认区间推进期间实际源码、输入与参考数据的身份保持一致。"""
    if any(sha256(path) != digest for path, digest in sources.items()):
        raise ValueError("证明期间参考数据或实际源码已改变")


def advance_mass(reference, out, *, target=None, cap="0.1", shape="0.5"):
    """从原始初值开始逐原生步推进耗散质量误差管；整段超限立即失败。"""
    ref = native_reference(reference)
    target = F(str(ref.summary["end_s"] if target is None else target))
    if not 0 < target <= F(str(ref.summary["end_s"])):
        raise ValueError("目标必须位于实际参考轨迹内")
    cap, shape, error = F(cap), F(shape), F(0)
    p = ref.config["property_sets"]["P1"]
    diffusion = DiffusionBounds(
        Interval(F.from_float(p["D_prefactor"])),
        Interval(F.from_float(p["D_moisture_exponent"])),
        Interval(0),
        "Q1 binary64 coefficients",
    )
    source = MassSource.zero(diffusion)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = _proof_sources(ref)
    previous, records, closed, covered = None, [], True, F(0)
    try:
        for index, (coefficients, correction) in enumerate(ref.steps()):
            left, right = map(
                F,
                (correction["physical_left_exact"], correction["physical_right_exact"]),
            )
            if left >= target:
                break
            cells, _ = cumulative_cells(coefficients, ref.radii, float(ref.radius))
            u = (TIME - left) * (1 / (right - left))
            patches = []
            for cell in cells:

                def temporal(values):
                    return sum((value * u**i for i, value in enumerate(values)), Polynomial())

                mass = temporal(cell["M_prefix_u_coefficients"])
                for degree, values in enumerate(cell["C_local_y_power_coefficients_each_in_u"]):
                    mass += temporal(values) * (Y - cell["y0"]) ** (degree + 1) * F(1, degree + 1)
                patches.append({"y0": cell["y0"], "y1": cell["y1"], "M": mass})
            if left != covered:
                raise ValueError("质量误差管存在时间间断")
            if previous is None:
                if any((cell["M"].at("t", 0) - ref.C0 * Y).c for cell in patches):
                    raise ValueError("参考初值未精确继承原始均匀场")
            else:
                validate_time_seam(previous, patches, left)
            error, record = advance_dissipative(
                patches,
                diffusion,
                Polynomial(ref.radius),
                Interval(ref.hm),
                ref.environment(left, right),
                source,
                left,
                right,
                error,
                cap,
                lambda rectangle: None,
                shape,
            )
            record.update(step=index, reference_correction=correction)
            records.append(write_step_record(out, index, record))
            previous, covered = patches, right
            if covered >= target:
                break
    except (ValueError, ArithmeticError, AssertionError) as error_message:
        closed = False
        failure = str(error_message)
    result = {
        "schema": "merged-mass-certificate-v1",
        "closed_to_target": closed and covered >= target,
        "certified_through_exact": str(covered),
        "target_exact": str(target),
        "error_upper_exact": str(error),
        "cap_exact": str(cap),
        "shape_exact": str(shape),
        "records": records,
        "sources": sources,
        "first_failure": None if closed else failure,
        "PDE_started": False,
        "scope": "Conditional cumulative-mass error tube; does not imply pointwise concentration or correct rounding.",
    }
    verify_proof_sources(sources)
    write_json(out / "summary.json", result)
    return result


def load_checkpoint(folder, reference_sources, envelope, domain, cap, cells, seen=None):
    """沿父链核对全空间参考与幅度；SHA只绑定文件，不替代区间计算。"""
    folder = Path(folder).resolve()
    seen = set() if seen is None else seen
    if folder in seen:
        raise ValueError("检查点父链存在循环")
    seen.add(folder)
    summary = read_json(folder / "summary.json")
    if summary.get("schema") != "merged-localized-certificate-v1" or not summary.get(
        "closed_to_target"
    ):
        raise ValueError("继承只接受本实现完成的条件证明；旧档案需独立重新验证")
    if (
        summary["reference_sources"] != reference_sources
        or summary["envelope"] != envelope.parameters()
        or summary["C_domain"] != [str(domain.lo), str(domain.hi)]
        or summary["cap_exact"] != str(cap)
    ):
        raise ValueError("继承的参考、源码、比较域或包络参数不同")
    for path, digest in summary["sources"].items():
        if sha256(path) != digest:
            raise ValueError("继承来源已改变：" + path)
    prior, amplitude, covered, next_step = None, F(0), F(0), 0
    parent = summary.get("parent")
    if parent:
        parent_path = Path(parent["path"])
        if sha256(parent_path / "summary.json") != parent["sha256"]:
            raise ValueError("父证明摘要已改变")
        prior, amplitude, covered, next_step = load_checkpoint(
            parent_path, reference_sources, envelope, domain, cap, cells, seen
        )
    if not summary["records"]:
        raise ValueError("证明没有新增的已闭合步骤")
    for row in summary["records"]:
        record_path = (folder / row["path"]).resolve()
        if not record_path.is_relative_to(folder) or sha256(record_path) != row["sha256"]:
            raise ValueError("继承步骤记录已改变或路径越界")
        record = read_json(record_path)
        left, right = map(F, record["native_time"])
        if (
            not record["closed"]
            or not record["guards"]
            or not all(record["guards"].values())
            or record["step"] != next_step
            or left != covered
            or right <= left
            or F(record["A_in"]) < amplitude
            or F(record["A_out"]) < F(record["A_in"])
        ):
            raise ValueError("继承步骤没有连续继承时间和误差幅度")
        for suffix, key in [
            ("reference.json.gz", "reference_sha256"),
            ("rectangles.jsonl.gz", "rectangles_sha256"),
        ]:
            if sha256(folder / f"step_{next_step}_{suffix}") != record[key]:
                raise ValueError("继承的全空间参考或区间记录已改变")
        with gzip.open(
            folder / f"step_{next_step}_reference.json.gz", "rt", encoding="utf-8"
        ) as stream:
            rows = json.load(stream)
        if len(rows) != cells or [i for i, _ in rows] != list(range(cells)):
            raise ValueError("参考记录没有覆盖全部空间单元")
        following = {}
        for i, coefficients in rows:
            polynomial = Polynomial({(a, b): F(c) for a, b, c in coefficients})
            if prior is not None:
                inherit_trace(polynomial, prior[i], left, covered, F(record["A_in"]), amplitude)
            following[i] = polynomial.at("t", 1)
        prior, amplitude, covered, next_step = (
            following,
            F(record["A_out"]),
            right,
            next_step + 1,
        )
    if (
        sha256(folder / "checkpoint.json") != summary["checkpoint_sha256"]
        or F(summary["certified_through_exact"]) != covered
        or F(summary["target_exact"]) > covered
    ):
        raise ValueError("最终覆盖范围或检查点摘要无效")
    checkpoint = read_json(folder / "checkpoint.json")
    if (
        F(checkpoint["time"]) != covered
        or checkpoint["next_step"] != next_step
        or F(checkpoint["amplitude"]) != amplitude
        or checkpoint["traces"] != [[i, poly.json()] for i, poly in prior.items()]
    ):
        raise ValueError("检查点没有继承已验证的末步全空间轨迹与幅度")
    return prior, amplitude, covered, next_step


def advance_localized(
    reference,
    out,
    *,
    target="1",
    cap=".01",
    C_min=".01",
    C_max="3",
    width_factor="16",
    floor="1e-12",
    inherit=None,
):
    """融合初始步与后续步：严格继承全空间势函数与误差幅度，不重置检查点。"""
    ref = native_reference(reference)
    target, cap = F(target), F(cap)
    if not 0 < target <= min(F(1), F(str(ref.summary["end_s"]))):
        raise ValueError("当前局部研究入口仅覆盖原研究的0..1秒目标")
    p = ref.config["property_sets"]["P1"]
    transform = KirchhoffTransform(
        ref.C0, F.from_float(p["D_moisture_exponent"]), F.from_float(p["D_prefactor"])
    )
    x = [F.from_float(float(r)) / ref.radius for r in ref.radii]
    domain = Interval(F(C_min), F(C_max))
    envelope = GaussianEnvelope(
        transform, ref.radius, domain, F(width_factor) * (x[-1] - x[-2]) ** 2, F(floor)
    )
    N = len(x) - 1
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    reference_sources = _proof_sources(ref)
    sources = dict(reference_sources)
    parent = None
    prior = None
    amplitude = None
    covered = F(0)
    start_step = 0
    records = []
    failure = None
    target_queries = []
    if inherit is not None:
        inherited = Path(inherit).resolve()
        prior, amplitude, covered, start_step = load_checkpoint(
            inherited, reference_sources, envelope, domain, cap, N
        )
        if target <= covered:
            raise ValueError("续推目标必须晚于已证明覆盖时间")
        parent = {"path": str(inherited), "sha256": sha256(inherited / "summary.json")}
        sources[str(inherited / "summary.json")] = parent["sha256"]
    try:
        for step, (coefficients, correction) in enumerate(ref.steps()):
            if step < start_step:
                continue
            left, right = map(
                F,
                (correction["physical_left_exact"], correction["physical_right_exact"]),
            )
            if left >= target:
                break
            if left != covered:
                raise ValueError("全空间证明的物理时间不连续")
            native = [
                sum((v * TIME**j for j, v in enumerate(values)), Polynomial())
                for values in coefficients
            ]
            knots = [F(0), F(1, 3), F(2, 3), F(1)]
            potential = []
            for concentration in native:
                values = [transform.primitive(concentration.value(0, t)) for t in knots]
                potential.append(time_interpolant([(v.lo + v.hi) / 2 for v in values], knots))
            ce = ref.environment(left, right)
            local_ce = (
                Polynomial(ce.value(0, left)) + (ce.value(0, right) - ce.value(0, left)) * TIME
            )
            rim = -ref.radius * ref.hm / transform.prefactor * (native[-1] - local_ce)
            slopes, curves = nodal_derivatives(potential, x, rim)
            coefficients_max = {
                key: F(0)
                for key in (
                    "rho",
                    "L",
                    "source",
                    "initial_A_required",
                    "potential_native_PL_defect",
                    "U_min",
                    "U_max",
                )
            }
            traces = {}
            reference_cells = []
            rows = 0
            with gzip.open(
                out / f"step_{step}_rectangles.jsonl.gz", "wt", encoding="utf-8"
            ) as stream:
                for i in range(N):
                    h = x[i + 1] - x[i]
                    poly = quintic(
                        potential[i],
                        potential[i + 1],
                        slopes[i],
                        slopes[i + 1],
                        curves[i],
                        curves[i + 1],
                        h,
                    )
                    validate_cell(
                        poly,
                        potential[i],
                        potential[i + 1],
                        slopes[i],
                        slopes[i + 1],
                        curves[i],
                        curves[i + 1],
                        h,
                    )
                    if prior is not None:
                        inherit_trace(poly, prior[i], left, covered, amplitude, amplitude)
                    traces[i] = poly.at("t", 1)
                    reference_cells.append([i, poly.json()])
                    native_cell = native[i] + (native[i + 1] - native[i]) * Y
                    for part in range(4):
                        bound = localized_coefficients(
                            transform,
                            envelope,
                            poly,
                            native_cell,
                            x[i],
                            x[i + 1],
                            left,
                            right,
                            F(part, 4),
                            F(part + 1, 4),
                            domain,
                        )
                        for key in coefficients_max:
                            coefficients_max[key] = (min if key == "U_min" else max)(
                                coefficients_max[key], bound[key]
                            )
                        stream.write(
                            json.dumps(
                                _serial({"cell": i, "part": part, **bound}),
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        rows += 1
            if amplitude is None:
                amplitude = coefficients_max["initial_A_required"]
            incoming = amplitude
            end_amplitude = envelope.amplitude(
                incoming,
                coefficients_max["source"],
                coefficients_max["L"],
                right - left,
            )
            error = envelope.uniform_E_upper(end_amplitude.hi, right, left)
            robin_defect = F(0)
            for part in range(4):
                a, b = F(part, 4), F(part + 1, 4)
                mid = (a + b) / 2
                cbox = native[-1].bounds((0, 0, a, b))
                center = Interval(potential[-1].value(0, mid)) - transform.primitive(
                    native[-1].value(0, mid)
                )
                derivative = potential[-1].derivative("t").bounds((0, 0, a, b)) - transform.slope(
                    cbox
                ) * native[-1].derivative("t").bounds((0, 0, a, b))
                robin_defect = max(
                    robin_defect,
                    (center + Interval(-(b - a) / 2, (b - a) / 2) * derivative).absmax,
                )
            guards = {
                "complete_rectangles": rows == 4 * N,
                "Hermite_C2": True,
                "whole_trace_inherited": prior is not None or left == 0,
                "positive_Gaussian_supersolution": envelope.kappa > 4 * envelope.amax
                and envelope.amin > 0,
                "initial_trace_enclosed": (
                    incoming >= coefficients_max["initial_A_required"] if left == 0 else True
                ),
                "nonlinear_Robin_floor": incoming * envelope.shape(F(1), F(1), left, right).lo
                >= robin_defect,
                "positive_barrier_domain": transform.primitive(domain.lo).hi
                < coefficients_max["U_min"] - error
                and coefficients_max["U_max"] + error < transform.primitive(domain.hi).lo,
                "comparison_cap": error <= cap,
            }
            closed = all(guards.values())
            with gzip.open(
                out / f"step_{step}_reference.json.gz", "wt", encoding="utf-8"
            ) as stream:
                json.dump(reference_cells, stream, separators=(",", ":"))
            record = {
                "step": step,
                "native_time": [str(left), str(right)],
                "guards": guards,
                "closed": closed,
                "A_in": str(incoming),
                "A_out": str(end_amplitude.hi),
                "error_upper": str(error),
                "coefficients": _serial(coefficients_max),
                "reference_sha256": sha256(out / f"step_{step}_reference.json.gz"),
                "rectangles_sha256": sha256(out / f"step_{step}_rectangles.jsonl.gz"),
            }
            records.append(write_step_record(out, step, record))
            if not closed:
                failure = {
                    "step": step,
                    "failed_guards": [k for k, v in guards.items() if not v],
                }
                break
            if left <= target <= right:
                target_queries = query_at(
                    transform,
                    envelope,
                    potential,
                    x,
                    left,
                    right,
                    incoming,
                    coefficients_max["source"],
                    coefficients_max["L"],
                    target,
                    domain,
                )
            amplitude, covered, prior = end_amplitude.hi, right, traces
            write_json(
                out / "checkpoint.json",
                {
                    "next_step": step + 1,
                    "time": str(covered),
                    "amplitude": str(amplitude),
                    "traces": [[i, p.json()] for i, p in traces.items()],
                },
            )
            envelope.shape.cache_clear()
            if covered >= target:
                break
    except (ValueError, ArithmeticError, AssertionError) as error_message:
        failure = {"error": str(error_message)}
    verify_proof_sources(sources)
    result = {
        "schema": "merged-localized-certificate-v1",
        "closed_to_target": failure is None and covered >= target,
        "target_exact": str(target),
        "certified_through_exact": str(covered),
        "first_failure": failure,
        "records": records,
        "envelope": envelope.parameters(),
        "C_domain": [str(domain.lo), str(domain.hi)],
        "cap_exact": str(cap),
        "target_queries": target_queries,
        "sources": sources,
        "reference_sources": reference_sources,
        "parent": parent,
        "checkpoint_sha256": (
            sha256(out / "checkpoint.json") if (out / "checkpoint.json").exists() else None
        ),
        "PDE_started": False,
        "all_required_C_bins_certified": False,
        "historical_acceptance_claimed": False,
    }
    write_json(out / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path)
    parser.add_argument("--research-root", type=Path)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("produce", help="生成永久原生水分稠密轨迹")
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--end", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    for name in ("mass", "localized"):
        p = sub.add_parser(
            name,
            help="连续质量误差管" if name == "mass" else "全空间局部屏障与严格继承",
        )
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--target", default="1")
        p.add_argument("--cap", default=".01")
        if name == "localized":
            p.add_argument("--inherit", type=Path)
            p.add_argument("--C-min", default=".01")
            p.add_argument("--C-max", default="3")
    sub.add_parser("check", help="无需Arb的精确指数/对数自检")
    args = parser.parse_args(argv)
    configure(args.code_dir, args.research_root)
    if args.mode == "produce":
        result = produce_reference(args.out, N=args.N, end_s=args.end)
    elif args.mode == "mass":
        result = advance_mass(args.data, args.out, target=args.target, cap=args.cap)
    elif args.mode == "localized":
        result = advance_localized(
            args.data,
            args.out,
            target=args.target,
            cap=args.cap,
            C_min=args.C_min,
            C_max=args.C_max,
            inherit=args.inherit,
        )
    else:
        result = {"passed": all(row["passed"] for row in self_checks())}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("closed_to_target") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
