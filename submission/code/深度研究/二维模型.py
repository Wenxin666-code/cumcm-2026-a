"""二维有限体积/有限元研究：同一配置和输出接口，两个独立空间离散。

公开接口：Settings、CylinderFV、TensorFE、CheckedBDF、HeatAudit、
PolynomialBounds、build_model、solve_fv、solve_fe、run、main。
FE 的连续极值/事件界只包围计算所得插值多项式，不是连续 PDE 误差界。
"""

from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_CEILING
from fractions import Fraction as F
from functools import partial
from math import comb
from pathlib import Path
import argparse
import gc
import hashlib
import heapq
import json
import struct
import time
import numpy as np
import scipy
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from openpyxl import load_workbook
from scipy.integrate import BDF
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, bmat, diags, eye, kron
from scipy.sparse.linalg import splu
from 公共接口 import write_json as write, sha256 as digest, paths


@dataclass
class Settings:
    rtol: float = 2e-10
    atol_t: float = 1e-09
    atol_c: float = 2e-12
    max_step: float = 60.0

    def validate(self):
        if any((not np.isfinite(v) or v <= 0 for v in asdict(self).values())):
            raise ValueError("所有积分容差和最大步长必须为正的有限值。")
        return self


def prepare_run(out, stop, settings):
    """两种离散使用相同的时间、精度和输出目录校验。"""
    if not np.isfinite(stop) or stop <= 0:
        raise ValueError("计算终点必须为正的有限时间。")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("输出目录非空，请为本次计算选择新目录。")
    out.mkdir(parents=True, exist_ok=True)
    return out, (settings or Settings()).validate()


class CheckedBDF(BDF):
    """显式选择检查式 LU 和段间 GC；可选记录实际接受步，绝不修改全局 BDF。"""

    def __init__(self, *args, checked=True, collect=True, statistics=None, trace=None, **kwargs):
        if collect:
            gc.collect()
        self.statistics = statistics if statistics is not None else {}
        defaults = {
            "factorizations": 0,
            "linear_solves": 0,
            "refinement_steps": 0,
            "maximum_componentwise_backward_error": 0.0,
            "maximum_factor_nnz": 0,
            "minimum_abs_U_pivot": None,
            "residual_threshold": 256 * np.finfo(float).eps,
        }
        for key, value in defaults.items():
            self.statistics.setdefault(key, value)
        super().__init__(*args, **kwargs)
        if checked:
            self.lu = self.factor
            self.solve_lu = self.checked_solve
        self.trace = None
        if trace is not None:
            self.trace = {
                "initial_t_s": float(self.t),
                "t_bound_s": float(self.t_bound),
                "state_size": len(self.y),
                "accepted_steps": 0,
                "times_hash": hashlib.sha256(),
                "states_hash": hashlib.sha256(),
                "dense_hash": hashlib.sha256(),
            }
            self.trace["times_hash"].update(np.float64(self.t).tobytes())
            self.trace["states_hash"].update(np.asarray(self.y, dtype="<f8").tobytes())
            trace.append(self.trace)

    def factor(self, matrix):
        self.nlu += 1
        matrix = matrix.tocsc(copy=True)
        matrix.eliminate_zeros()
        lu = splu(
            matrix,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"Equil": True},
        )
        stats = self.statistics
        stats["factorizations"] += 1
        stats["maximum_factor_nnz"] = max(stats["maximum_factor_nnz"], int(lu.L.nnz + lu.U.nnz))
        pivot = float(np.min(abs(lu.U.diagonal())))
        old = stats["minimum_abs_U_pivot"]
        stats["minimum_abs_U_pivot"] = pivot if old is None else min(old, pivot)
        if not np.isfinite(pivot) or pivot == 0.0:
            raise FloatingPointError("Invalid LU pivot")
        return (lu, matrix, abs(matrix))

    def checked_solve(self, factors, rhs):
        lu, matrix, absolute = factors
        result = lu.solve(rhs)
        self.statistics["linear_solves"] += 1
        for iteration in range(4):
            residual = rhs - matrix @ result
            scale = absolute @ abs(result) + abs(rhs)
            error = float(np.max(abs(residual) / np.maximum(scale, np.finfo(float).tiny)))
            if not np.isfinite(result).all() or not np.isfinite(error):
                raise FloatingPointError("Nonfinite linear solve")
            if error <= self.statistics["residual_threshold"]:
                self.statistics["maximum_componentwise_backward_error"] = max(
                    self.statistics["maximum_componentwise_backward_error"], error
                )
                return result
            if iteration < 3:
                result += lu.solve(residual)
                self.statistics["refinement_steps"] += 1
        raise FloatingPointError("Ordered LU failed its componentwise residual check")

    def step(self):
        message = super().step()
        if self.status != "failed" and self.trace is not None:
            dense, row = (super().dense_output(), self.trace)
            row["accepted_steps"] += 1
            row["times_hash"].update(np.float64(self.t).tobytes())
            row["states_hash"].update(np.asarray(self.y, dtype="<f8").tobytes())
            row["dense_hash"].update(struct.pack("<II", int(dense.order), len(self.y)))
            for value in [
                np.array([dense.t_old, dense.t]),
                dense.t_shift,
                dense.denom,
                dense.D,
            ]:
                row["dense_hash"].update(np.asarray(value, dtype="<f8").tobytes(order="C"))
            row.update(last_t_s=float(self.t), nfev=self.nfev, njev=self.njev, nlu=self.nlu)
        return message


class FieldWriter:
    """两个离散共享无损分块输出；各自保留真实网格与连续场的解释。"""

    def __init__(self, out, model, capacity=None):
        self.out, self.model = (Path(out), model)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "fields").mkdir(exist_ok=True)
        self.capacity = capacity or (16 if isinstance(model, TensorFE) else 64)
        self.times, self.values, self.records = ([], [], [])
        self.rows = self.count = 0
        self.last = None
        np.savez_compressed(self.out / "coordinates.npz", **model.coordinates())

    def add(self, t, state):
        if self.last is not None and abs(t - self.last) < 1e-09:
            return
        if self.last is not None and t <= self.last:
            raise ValueError("采样时间必须严格增加。")
        self.last = float(t)
        self.times.append(float(t))
        self.values.append(np.asarray(state).copy())
        self.rows += 1
        self.count += 1
        if len(self.times) >= self.capacity:
            self.flush()

    def flush(self):
        if not self.times:
            return
        y, n = (np.stack(self.values), self.model.size)
        filename = f"fields/chunk_{len(self.records):05d}.npz"
        geometry = np.asarray([self.model.dimensions(t) for t in self.times])
        path = self.out / filename
        np.savez_compressed(
            path,
            t_s=np.array(self.times),
            R_m=geometry[:, 0],
            ell=geometry[:, 1],
            half_length_m=self.model.H * geometry[:, 1],
            T_K=y[:, :n].reshape(-1, *self.model.shape),
            C=y[:, n:].reshape(-1, *self.model.shape),
        )
        self.records.append(
            {
                "file": filename,
                "path": filename,
                "rows": len(self.times),
                "first_s": self.times[0],
                "last_s": self.times[-1],
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
        )
        self.times.clear()
        self.values.clear()

    def finish(self):
        self.flush()
        write(
            self.out / "field_index.json",
            {
                "completed": True,
                "rows": self.rows,
                "field_shape": self.model.shape,
                "dtype": "float64",
                "chunks": self.records,
                "coordinates": "coordinates.npz",
                "discretization": type(self.model).__name__,
            },
        )


@dataclass
class CylinderFV:
    """柱体有限体积：径向控制体积与轴向运动在同一个守恒算子中计算。"""

    grid: object
    z_m: np.ndarray
    properties: object
    ambient: object
    geometry: object
    h: float = 25.0
    hm: float = 8e-07
    end_factor: float = 1.0
    axial_power: float = 0.0

    def __post_init__(self):
        self.nz = len(self.z_m)
        self.nr = len(self.grid.r_m)
        self.size = self.nr * self.nz
        self.dz = np.diff(self.z_m)
        faces = (self.z_m[:-1] + self.z_m[1:]) / 2
        self.wz = np.diff(np.r_[0.0, faces, self.z_m[-1]])
        self.H = float(self.z_m[-1])
        self.wr = self.grid.volume_weights_m2
        self.radial_probability = self.wr / self.wr.sum()
        self.axial_probability = self.wz / self.H
        self.weights = self.radial_probability[:, None] * self.axial_probability[None, :]
        pr = diags(
            [np.ones(self.nr - 1), np.ones(self.nr), np.ones(self.nr - 1)],
            [-1, 0, 1],
            format="csr",
        )
        pz = diags(
            [np.ones(self.nz - 1), np.ones(self.nz), np.ones(self.nz - 1)],
            [-1, 0, 1],
            format="csr",
        )
        p = (kron(pr, eye(self.nz, format="csr")) + kron(eye(self.nr, format="csr"), pz)).astype(
            bool
        )
        self.sparsity = bmat([[p, p], [p, p]], format="csr")
        self.shape = (self.nr, self.nz)

    def unpack(self, state):
        return (
            state[: self.size].reshape(self.nr, self.nz),
            state[self.size :].reshape(self.nr, self.nz),
        )

    def state(self, T=301.15, C=2.55):
        return np.r_[
            np.broadcast_to(T, (self.nr, self.nz)).ravel(),
            np.broadcast_to(C, (self.nr, self.nz)).ravel(),
        ]

    def divergence(self, field, coefficient, transfer, ambient, radius):
        """径向与轴向通量共用一式；轴向梯度缩放 ell^-2，端面换热缩放 ell^-1。"""
        ell = (radius / self.grid.radius_m) ** self.axial_power
        radial_flux = (
            self.grid.face_r_m[:, None]
            * 0.5
            * (coefficient[:-1] + coefficient[1:])
            * np.diff(field, axis=0)
            / self.grid.node_spacing_m[:, None]
        )
        radial = np.empty_like(field)
        radial[0] = radial_flux[0]
        radial[1:-1] = np.diff(radial_flux, axis=0)
        radial[-1] = -radius * transfer * (field[-1] - ambient) - radial_flux[-1]
        radial *= (self.grid.radius_m / radius) ** 2 / self.wr[:, None]
        axial_flux = (
            0.5
            * (coefficient[:, :-1] + coefficient[:, 1:])
            * np.diff(field, axis=1)
            / self.dz[None, :]
        )
        axial = np.empty_like(field)
        axial[:, 0] = axial_flux[:, 0]
        axial[:, 1:-1] = np.diff(axial_flux, axis=1)
        axial[:, -1] = (
            -ell * self.end_factor * transfer * (field[:, -1] - ambient) - axial_flux[:, -1]
        )
        return radial + axial / (ell**2 * self.wz[None, :])

    def rhs(self, t, state, right=False):
        T, C = self.unpack(state)
        if np.any(T <= 0):
            raise FloatingPointError("Absolute temperature must be positive")
        rho, cp, k, D = self.properties.evaluate(C, T)
        env = self.ambient.values(t, right_of_breakpoint=right)
        R = self.geometry.radius(t)
        dT = self.divergence(T, k, self.h, env.temperature_K, R) / (rho * cp)
        dC = self.divergence(C, D, self.hm, env.moisture_eq, R)
        return np.r_[dT.ravel(), dC.ravel()]

    def fluxes(self, t, state, right=False):
        T, C = self.unpack(state)
        env = self.ambient.values(t, right_of_breakpoint=right)
        R = self.geometry.radius(t)
        result = np.array(
            [
                2 * self.hm / R * np.dot(self.axial_probability, C[-1] - env.moisture_eq),
                self.end_factor
                * self.hm
                / self.H
                * np.dot(self.radial_probability, C[:, -1] - env.moisture_eq),
                2 * self.h / R * np.dot(self.axial_probability, T[-1] - env.temperature_K),
                self.end_factor
                * self.h
                / self.H
                * np.dot(self.radial_probability, T[:, -1] - env.temperature_K),
            ]
        )
        result[[1, 3]] /= self.axial_scale(t)
        return result

    def audit(self, t, state, right=False):
        T, C = self.unpack(state)
        dT, dC = self.unpack(self.rhs(t, state, right))
        rho, cp, _, _ = self.properties.evaluate(C, T)
        flux = self.fluxes(t, state, right)
        residual_C = np.sum(self.weights * dC) + flux[0] + flux[1]
        residual_T = np.sum(self.weights * rho * cp * dT) + flux[2] + flux[3]
        return {
            "moisture_rate_abs": float(abs(residual_C)),
            "heat_rate_abs": float(abs(residual_T)),
            "moisture_rate_scaled": float(
                abs(residual_C) / max(abs(flux[0]) + abs(flux[1]), 1e-14)
            ),
            "heat_rate_scaled": float(abs(residual_T) / max(abs(flux[2]) + abs(flux[3]), 1e-08)),
        }

    def statistics(self, t, state):
        T, C = self.unpack(state)
        maxloc = np.unravel_index(np.argmax(C), C.shape)
        row = [
            t,
            np.sum(self.weights * C),
            C.max(),
            C[0, 0],
            C[-1, 0],
            C[0, -1],
            C[-1, -1],
            T[0, 0],
            T[-1, 0],
            T[0, -1],
            T[-1, -1],
            self.grid.r_m[maxloc[0]] / self.grid.radius_m,
            self.z_m[maxloc[1]],
            self.geometry.radius(t),
        ]
        ell = self.axial_scale(t)
        return row + [2 * self.H * ell, ell, row[12] * ell]

    def axial_scale(self, t):
        return (float(self.geometry.radius(t)) / self.grid.radius_m) ** self.axial_power

    def dimensions(self, t):
        return (float(self.geometry.radius(t)), self.axial_scale(t))

    def coordinates(self):
        return {
            "a_m": self.grid.r_m,
            "z_reference_m": self.z_m,
            "weights": self.weights,
            "xi": self.grid.r_m / self.grid.radius_m,
            "zeta": self.z_m / self.H,
        }


class GLLAxis:
    """两坐标方向共享的三次 GLL 单元、质量权重和局部多项式。"""

    def __init__(self, elements, power=2.0, radial=False):
        if elements < 1 or power <= 0:
            raise ValueError("Positive mesh size and power required")
        self.elements = elements
        self.z = np.array([-1.0, -1 / np.sqrt(5), 1 / np.sqrt(5), 1.0])
        self.weights = np.array([1.0, 5.0, 5.0, 1.0]) / 6
        self.bases = [
            Polynomial.fromroots(np.delete(self.z, j)) / np.prod(self.z[j] - np.delete(self.z, j))
            for j in range(4)
        ]
        self.D = np.column_stack([b.deriv()(self.z) for b in self.bases])
        edges = 1 - (1 - np.linspace(0, 1, elements + 1)) ** power
        self.edges = edges**2 if radial else edges
        self.width = np.diff(self.edges)
        self.local = (self.edges[:-1, None] + self.edges[1:, None]) / 2 + self.width[
            :, None
        ] * self.z / 2
        self.connectivity = 3 * np.arange(elements)[:, None] + np.arange(4)[None, :]
        self.nodes = 3 * elements + 1
        self.x = np.zeros(self.nodes)
        self.x[self.connectivity] = self.local
        self.qweights = self.width[:, None] / 2 * self.weights
        self.mass = np.bincount(
            self.connectivity.ravel(),
            weights=self.qweights.ravel(),
            minlength=self.nodes,
        )


class TensorFE:
    """独立的张量有限元：网格、弱式组装和物理系数统一为一个模型。"""

    def __init__(
        self,
        radial_elements,
        axial_elements,
        radial_power=2.0,
        axial_power=2.0,
        *,
        properties=None,
        ambient=None,
        radius=None,
        tails=None,
        R0=0.02,
        H=0.125,
        moving=False,
        gamma=0.0,
        h=25.0,
        hm=8e-07,
        end_factor=1.0,
        material="P4",
    ):
        self._assemble_grid(radial_elements, axial_elements, radial_power, axial_power)
        self.p = properties
        self.ambient = np.asarray(ambient)
        self.radius = np.asarray(radius)
        self.tails = np.asarray(tails)
        self.R0 = R0
        self.H = H
        self.moving = moving
        self.gamma = gamma
        self.h = h
        self.hm = hm
        self.end_factor = end_factor
        self.material = material

    def _assemble_grid(self, radial_elements, axial_elements, radial_power, axial_power):
        """构造张量连接、质量与稀疏结构，供物理求解和独立场重构共同使用。"""
        self.radial = GLLAxis(radial_elements, radial_power, True)
        self.axial = GLLAxis(axial_elements, axial_power, False)
        a, b = (self.radial, self.axial)
        self.shape = (a.nodes, b.nodes)
        self.size = a.nodes * b.nodes
        self.connectivity = (
            a.connectivity[:, None, :, None] * b.nodes + b.connectivity[None, :, None, :]
        )
        self.local_mass = a.qweights[:, None, :, None] * b.qweights[None, :, None, :]
        self.mass = a.mass[:, None] * b.mass[None, :]
        flat = self.connectivity.reshape(-1, 16)
        ii = np.broadcast_to(flat[:, :, None], (len(flat), 16, 16))
        jj = np.broadcast_to(flat[:, None, :], (len(flat), 16, 16))
        self.sparsity = coo_matrix(
            (np.ones(ii.size), (ii.ravel(), jj.ravel())), shape=(self.size, self.size)
        ).tocsr()
        self.coupled_sparsity = bmat(
            [[self.sparsity, self.sparsity], [self.sparsity, self.sparsity]],
            format="csr",
        )

    def stiffness(self, u, coefficient):
        a, b = (self.radial, self.axial)
        v = np.asarray(u).reshape(-1)[self.connectivity]
        k = np.broadcast_to(coefficient, self.shape).reshape(-1)[self.connectivity]
        gs = np.einsum("ik,abkj->abij", a.D, v - v[:, :, :1, :]) * (
            2 / a.width[:, None, None, None]
        )
        gz = np.einsum("jk,abik->abij", b.D, v - v[:, :, :, :1]) * (
            2 / b.width[None, :, None, None]
        )
        fs = self.local_mass * a.local[:, None, :, None] * k * gs
        fz = self.local_mass * k * gz
        rs = np.einsum("ik,abij->abkj", a.D, fs) * (2 / a.width[:, None, None, None])
        rz = np.einsum("jk,abij->abik", b.D, fz) * (2 / b.width[None, :, None, None])
        scatter = lambda v: np.bincount(
            self.connectivity.ravel(), weights=v.ravel(), minlength=self.size
        ).reshape(self.shape)
        return (scatter(rs), scatter(rz))

    def weak_rhs(self, u, coefficient, R, H, ell, transfer, ambient, end_factor=1.0):
        u = np.asarray(u).reshape(self.shape)
        ks, kz = self.stiffness(u, coefficient)
        f = -4 / R**2 * ks - 1 / (H * ell) ** 2 * kz
        f[-1, :] -= 2 * transfer / R * self.axial.mass * (u[-1, :] - ambient)
        f[:, -1] -= end_factor * transfer / (H * ell) * self.radial.mass * (u[:, -1] - ambient)
        return f

    def evaluate(self, u, s, zeta):
        """Tensor polynomial evaluation on paired reference coordinates."""
        s, zeta = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(zeta, dtype=float))
        if np.any(s < 0) or np.any(s > 1) or np.any(zeta < 0) or np.any(zeta > 1):
            raise ValueError("Outside reference domain")
        weights = []
        indices = []
        for axis, x in ((self.radial, s), (self.axial, zeta)):
            e = np.clip(np.searchsorted(axis.edges, x, side="right") - 1, 0, axis.elements - 1)
            q = 2 * (x - axis.edges[e]) / axis.width[e] - 1
            weights.append(np.stack([p(q) for p in axis.bases], axis=-1))
            indices.append(axis.connectivity[e])
        local = np.asarray(u).reshape(self.shape)[
            indices[0][..., :, None], indices[1][..., None, :]
        ]
        return np.einsum("...i,...ij,...j->...", weights[0], local, weights[1])

    def geometry(self, t):
        R = float(np.interp(t, self.radius[:, 0], self.radius[:, 1])) if self.moving else self.R0
        return (R, (R / self.R0) ** self.gamma)

    def environment(self, t, right_tail=False):
        if t > 14400 or (right_tail and t >= 14400):
            return (self.tails[0] + 273.15, self.tails[1])
        return (
            np.interp(t, self.ambient[:, 0], self.ambient[:, 1]) + 273.15,
            np.interp(t, self.ambient[:, 0], self.ambient[:, 2]),
        )

    def coefficients(self, T, C):
        if np.any(C <= 0) or np.any(T <= 0):
            raise FloatingPointError("Nonpositive nodal state, no clipping")
        p = self.p
        rho = p["rho_offset"] + p["rho_C_coefficient"] * C
        cp = p["cp_offset"] + p["cp_fraction_coefficient"] * C / (1 + C)
        k = p["k_offset"] + p["k_fraction_coefficient"] * C / (1 + C)
        D = p["D_prefactor"] * np.exp(
            -p["D_moisture_exponent"] / C - p["D_temperature_exponent_K"] / T
        )
        return (rho * cp, k, D)

    def rhs(self, t, y, right_tail=False):
        m = self
        n = m.size
        T = y[:n].reshape(m.shape)
        C = y[n:].reshape(m.shape)
        b, k, D = self.coefficients(T, C)
        R, ell = self.geometry(t)
        Ta, Ce = self.environment(t, right_tail)
        ht = m.weak_rhs(T, k, R, self.H, ell, self.h, Ta, self.end_factor) / (m.mass * b)
        ct = m.weak_rhs(C, D, R, self.H, ell, self.hm, Ce, self.end_factor) / m.mass
        return np.r_[ht.ravel(), ct.ravel()]

    def knots(self, stop):
        arrays = [np.array([0.0, stop]), self.ambient[:, 0]]
        if self.moving:
            arrays.append(self.radius[:, 0])
        values = np.unique(np.concatenate(arrays))
        return values[(values >= 0) & (values <= stop)]

    def dimensions(self, t):
        return self.geometry(t)

    def coordinates(self):
        return {
            "s": self.radial.x,
            "zeta": self.axial.x,
            "radial_edges": self.radial.edges,
            "axial_edges": self.axial.edges,
            "local_gll_nodes": self.radial.z,
            "mass": self.mass,
            "radial_connectivity": self.radial.connectivity,
            "axial_connectivity": self.axial.connectivity,
        }


def downward(x):
    return np.nextafter(x, -np.inf)


def upward(x):
    return np.nextafter(x, np.inf)


def iadd(a, b):
    return (downward(a[0] + b[0]), upward(a[1] + b[1]))


def isub(a, b):
    return (downward(a[0] - b[1]), upward(a[1] - b[0]))


def imul(a, b):
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    lo = np.minimum(np.minimum(products[0], products[1]), np.minimum(products[2], products[3]))
    hi = np.maximum(np.maximum(products[0], products[1]), np.maximum(products[2], products[3]))
    return (downward(lo), upward(hi))


def fbound(x):
    f = float(x)
    lo = f
    hi = f
    while F.from_float(lo) > x:
        lo = float(np.nextafter(lo, -np.inf))
    while F.from_float(hi) < x:
        hi = float(np.nextafter(hi, np.inf))
    return (lo, hi)


def lagrange_power(z):
    nodes = [(F.from_float(float(q)) + 1) / 2 for q in z]
    result = []
    for j in range(4):
        p = [F(1)]
        den = F(1)
        for k in range(4):
            if j == k:
                continue
            q = [F(0)] * (len(p) + 1)
            for i, v in enumerate(p):
                q[i] -= nodes[k] * v
                q[i + 1] += v
            p = q
            den *= nodes[j] - nodes[k]
        result.append([v / den for v in p])
    return (nodes, result)


def restricted_transform(z, alpha, beta):
    nodes, bases = lagrange_power(z)
    matrix = [[F(0)] * 4 for _ in range(4)]
    for j, p in enumerate(bases):
        coefficients = [
            sum(
                (p[m] * comb(m, k) * alpha ** (m - k) * beta**k for m in range(k, 4)),
                F(0),
            )
            for k in range(4)
        ]
        for i in range(4):
            matrix[i][j] = sum(
                (F(comb(i, k), comb(3, k)) * coefficients[k] for k in range(i + 1)),
                F(0),
            )
    return matrix


def restriction_set(coarse_edges, fine_edges, z):
    lower = []
    upper = []
    exact = []
    parents = []
    for left, right in zip(fine_edges[:-1], fine_edges[1:]):
        parent = int(np.searchsorted(coarse_edges, left, side="right") - 1)
        if parent < 0 or parent >= len(coarse_edges) - 1 or right > coarse_edges[parent + 1]:
            raise ValueError("Fine cells must lie entirely in a coarse cell; no extrapolation")
        cl, cr = map(lambda x: F.from_float(float(x)), coarse_edges[parent : parent + 2])
        fl, fr = (F.from_float(float(left)), F.from_float(float(right)))
        matrix = restricted_transform(z, (fl - cl) / (cr - cl), (fr - fl) / (cr - cl))
        bounds = [[fbound(v) for v in row] for row in matrix]
        lower.append([[v[0] for v in row] for row in bounds])
        upper.append([[v[1] for v in row] for row in bounds])
        exact.append(matrix)
        parents.append(parent)
    return (np.array(lower), np.array(upper), np.array(parents), exact)


def tensor_transform(local, radial, axial):
    """Interval Kronecker action with no unbounded BLAS reductions/FMA."""
    shape = local[0].shape
    tmp = (np.zeros(shape), np.zeros(shape))
    result = (np.zeros(shape), np.zeros(shape))
    for i in range(4):
        total = (np.zeros(shape[:2] + (4,)), np.zeros(shape[:2] + (4,)))
        for k in range(4):
            coef = (radial[0][:, None, i, k, None], radial[1][:, None, i, k, None])
            value = (local[0][:, :, k, :], local[1][:, :, k, :])
            total = iadd(total, imul(coef, value))
        tmp[0][:, :, i, :] = total[0]
        tmp[1][:, :, i, :] = total[1]
    for j in range(4):
        total = (np.zeros(shape[:2] + (4,)), np.zeros(shape[:2] + (4,)))
        for k in range(4):
            coef = (axial[0][None, :, j, k, None], axial[1][None, :, j, k, None])
            value = (tmp[0][:, :, :, k], tmp[1][:, :, :, k])
            total = iadd(total, imul(coef, value))
        result[0][:, :, :, j] = total[0]
        result[1][:, :, :, j] = total[1]
    return result


def polynomial_bases(dense, left, right):
    """Exact powers in u for the original stored BDF Newton basis on [left,right]."""
    l = F.from_float(float(left))
    h = F.from_float(float(right)) - l
    result = [[F(1)]]
    for shift, denom in zip(dense.t_shift, dense.denom):
        d = F.from_float(float(denom))
        a = (l - F.from_float(float(shift))) / d
        b = h / d
        q = [F(0)] * (len(result[-1]) + 1)
        for i, v in enumerate(result[-1]):
            q[i] += a * v
            q[i + 1] += b * v
        result.append(q)
    return result


def time_bernstein_matrix(dense, left, right, derivative=False):
    bases = polynomial_bases(dense, left, right)
    degree = dense.order - int(derivative)
    dt = F.from_float(float(right)) - F.from_float(float(left))
    if derivative:
        bases = [
            [F(0)] if len(p) == 1 else [i * p[i] / dt for i in range(1, len(p))] for p in bases
        ]
    matrix = []
    for j in range(degree + 1):
        matrix.append(
            [
                sum(
                    (F(comb(j, k), comb(degree, k)) * p[k] for k in range(min(j, len(p) - 1) + 1)),
                    F(0),
                )
                for p in bases
            ]
        )
    return matrix


def scalar_time_bound(dense, index, left=None, right=None):
    left = dense.t_old if left is None else left
    right = dense.t if right is None else right
    matrix = time_bernstein_matrix(dense, left, right)
    values = [
        sum(
            (v * F.from_float(float(dense.D[k, index])) for k, v in enumerate(row)),
            F(0),
        )
        for row in matrix
    ]
    lo, hi = (min(values), max(values))
    return {
        "lower": fbound(lo)[0],
        "upper": fbound(hi)[1],
        "strictly_above_threshold": lo > F(3, 20),
    }


def dense_derivative(dense, t):
    """Differentiate the actual BDF dense polynomial; no RHS substitution."""
    product = 1.0
    derivative = 0.0
    answer = np.zeros(dense.D.shape[1])
    for j in range(dense.order):
        x = (t - dense.t_shift[j]) / dense.denom[j]
        derivative = derivative * x + product / dense.denom[j]
        product *= x
        answer += dense.D[j + 1] * derivative
    return answer


def dense_interval(dense, t, indices):
    """Outward evaluation of the exact stored polynomial at exact binary64 t."""
    tt = F.from_float(float(t))
    y = dense.D[0, indices]
    result = (np.asarray(y).copy(), np.asarray(y).copy())
    product = (1.0, 1.0)
    for j in range(dense.order):
        q = (tt - F.from_float(float(dense.t_shift[j]))) / F.from_float(float(dense.denom[j]))
        product = imul(product, fbound(q))
        v = dense.D[j + 1, indices]
        result = iadd(result, imul((v, v), product))
    return result


def split_curve(lo, hi):
    levels = [(lo, hi)]
    for _ in range(3):
        a, b = levels[-1]
        levels.append(imul(iadd((a[:-1], b[:-1]), (a[1:], b[1:])), (0.5, 0.5)))
    left = (np.array([q[0][0] for q in levels]), np.array([q[1][0] for q in levels]))
    right = (
        np.array([q[0][-1] for q in levels[::-1]]),
        np.array([q[1][-1] for q in levels[::-1]]),
    )
    return (left, right)


def split_patch(lo, hi):
    first = [(np.empty((4, 4)), np.empty((4, 4))) for _ in range(2)]
    for j in range(4):
        halves = split_curve(lo[:, j], hi[:, j])
        for k in range(2):
            first[k][0][:, j] = halves[k][0]
            first[k][1][:, j] = halves[k][1]
    result = []
    for a, b in first:
        half = [(np.empty((4, 4)), np.empty((4, 4))) for _ in range(2)]
        for i in range(4):
            q = split_curve(a[i], b[i])
            for k in range(2):
                half[k][0][i] = q[k][0]
                half[k][1][i] = q[k][1]
        result.extend(half)
    return result


def refine_maximum(lo, hi, lower, tolerance=1e-12, threshold=None):
    """Branch-and-bound of continuous tensor polynomial on every element."""
    queue = []
    serial = 0
    splits = 0
    for a, b in zip(lo.reshape(-1, 4, 4), hi.reshape(-1, 4, 4)):
        upper = float(b.max())
        if upper >= lower:
            heapq.heappush(queue, (-upper, serial, 0, a, b))
            serial += 1
    threshold_bounds = None if threshold is None else fbound(F(str(threshold)))
    roundoff_limited = False
    while queue:
        upper = max(lower, -queue[0][0])
        if threshold_bounds and (lower > threshold_bounds[1] or upper < threshold_bounds[0]):
            break
        if upper - lower <= tolerance:
            break
        negative, _, depth, a, b = heapq.heappop(queue)
        if -negative < lower:
            continue
        if depth >= 60:
            heapq.heappush(queue, (negative, serial, depth, a, b))
            roundoff_limited = True
            break
        for ca, cb in split_patch(a, b):
            lower = max(lower, float(ca[np.ix_([0, 3], [0, 3])].max()))
            upper = float(cb.max())
            if upper >= lower:
                heapq.heappush(queue, (-upper, serial, depth + 1, ca, cb))
                serial += 1
        splits += 1
    upper = max(lower, -queue[0][0]) if queue else lower
    classification = "unclassified"
    if threshold_bounds:
        if lower > threshold_bounds[1]:
            classification = "wet"
        elif upper < threshold_bounds[0]:
            classification = "dry"
        else:
            classification = "unresolved"
    return {
        "lower": lower,
        "upper": upper,
        "width": float(np.nextafter(upper - lower, np.inf)),
        "subdivisions": splits,
        "converged": upper - lower <= tolerance,
        "roundoff_limited": roundoff_limited,
        "classification": classification,
    }


class PolynomialBounds:
    """外向舍入 Bernstein 界：统一处理单场极值、双场差和连续事件。"""

    def __init__(self, mesh, coarse=None):
        self.mesh = mesh
        self.br = restriction_set(mesh.radial.edges, mesh.radial.edges, mesh.radial.z)
        self.bz = restriction_set(mesh.axial.edges, mesh.axial.edges, mesh.axial.z)
        self.ir = mesh.radial.connectivity
        self.iz = mesh.axial.connectivity
        if coarse is not None:
            self.prepare_difference(coarse.coordinates(), mesh.coordinates())

    def coefficients(self, interval):
        lo, hi = [np.asarray(x).reshape(self.mesh.shape) for x in interval]
        local = tuple((x[self.ir[:, None, :, None], self.iz[None, :, None, :]] for x in [lo, hi]))
        return tensor_transform(local, self.br, self.bz)

    def bound(self, interval, tolerance=1e-12, threshold=None):
        lower = float(np.max(interval[0]))
        if threshold is not None and lower > fbound(F(str(threshold)))[1]:
            return {
                "lower": lower,
                "upper": None,
                "classification": "wet",
                "subdivisions": 0,
                "reason": "A rigorously enclosed nodal value is a valid lower bound on the spatial maximum.",
            }
        lo, hi = self.coefficients(interval)
        return refine_maximum(lo, hi, lower, tolerance, threshold)

    def time_derivative_upper(self, dense, left, right, offset):
        matrix = time_bernstein_matrix(dense, left, right, True)
        upper = -np.inf
        for row in matrix:
            interval = (np.zeros(self.mesh.size), np.zeros(self.mesh.size))
            for k, v in enumerate(row):
                data = dense.D[k, offset : offset + self.mesh.size]
                interval = iadd(interval, imul((data, data), fbound(v)))
            _, hi = self.coefficients(interval)
            upper = max(upper, float(hi.max()))
        return upper

    def prepare_difference(self, coarse, fine):
        self.c = coarse
        self.f = fine
        z = coarse["local_gll_nodes"]
        if not np.array_equal(z, fine["local_gll_nodes"]):
            raise ValueError("Different polynomial nodes")
        self.cr = restriction_set(coarse["radial_edges"], fine["radial_edges"], z)
        self.cz = restriction_set(coarse["axial_edges"], fine["axial_edges"], z)
        self.fr = restriction_set(fine["radial_edges"], fine["radial_edges"], z)
        self.fz = restriction_set(fine["axial_edges"], fine["axial_edges"], z)
        self.cir = coarse["radial_connectivity"][self.cr[2]]
        self.ciz = coarse["axial_connectivity"][self.cz[2]]
        self.fir = fine["radial_connectivity"]
        self.fiz = fine["axial_connectivity"]

    def difference_coefficients(self, coarse, fine):
        shift = float(coarse[0, 0])
        x = coarse[self.cir[:, None, :, None], self.ciz[None, :, None, :]]
        y = fine[self.fir[:, None, :, None], self.fiz[None, :, None, :]]
        c = tensor_transform((downward(x - shift), upward(x - shift)), self.cr, self.cz)
        f = tensor_transform((downward(y - shift), upward(y - shift)), self.fr, self.fz)
        return isub(f, c)

    def difference(self, coarse, fine):
        if not hasattr(self, "cr"):
            raise ValueError("双场对照须在创建 PolynomialBounds 时指定 coarse 网格。")
        lo, hi = self.difference_coefficients(coarse, fine)
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise FloatingPointError("Nonfinite interval bound")
        upper = float(max(np.max(hi), -np.min(lo)))
        corner_lo = lo[:, :, [0, 3], :][:, :, :, [0, 3]]
        corner_hi = hi[:, :, [0, 3], :][:, :, :, [0, 3]]
        corner_abs_lower = np.where(
            (corner_lo <= 0) & (corner_hi >= 0),
            0.0,
            np.minimum(abs(corner_lo), abs(corner_hi)),
        )
        lower = float(np.max(corner_abs_lower))
        ix = np.unravel_index(np.argmax(np.maximum(hi, -lo)), lo.shape)
        return {
            "continuous_absolute_difference_lower": lower,
            "continuous_absolute_difference_upper": upper,
            "upper_bound_cell_radial": int(ix[0]),
            "upper_bound_cell_axial": int(ix[1]),
            "signed_coefficient_lower": float(lo.min()),
            "signed_coefficient_upper": float(hi.max()),
        }


def bracket_event(dense, maximum, offset, width=0.0001):
    left = float(dense.t_old)
    right = float(dense.t)
    records = []

    def classify(t):
        b = maximum.bound(
            dense_interval(dense, t, slice(offset, offset + maximum.mesh.size)),
            threshold=".15",
            tolerance=1e-13,
        )
        records.append({"t_s": t, **b})
        return b["classification"]

    if classify(left) != "wet" or classify(right) != "dry":
        raise RuntimeError("Event endpoints are not rigorously wet/dry for computed interpolant")
    while right - left > width:
        mid = (left + right) / 2
        state = classify(mid)
        if state == "wet":
            left = mid
        elif state == "dry":
            right = mid
        else:
            lq = (left + mid) / 2
            rq = (mid + right) / 2
            sl = classify(lq)
            sr = classify(rq)
            old = (left, right)
            if sl == "dry":
                right = lq
            elif sr == "wet":
                left = rq
            else:
                if sl == "wet":
                    left = lq
                if sr == "dry":
                    right = rq
            if old == (left, right):
                raise RuntimeError(
                    "Spatial polynomial interval cannot distinguish event sides at target time scale; preserve evidence, no false root"
                )
    derivative = maximum.time_derivative_upper(dense, left, right, offset)
    full_derivative = maximum.time_derivative_upper(
        dense, float(dense.t_old), float(dense.t), offset
    )
    return {
        "root_lower_s": left,
        "root_upper_s": right,
        "root_width_s": right - left,
        "whole_space_time_C_derivative_upper": derivative,
        "local_strict_decrease_verified": derivative < 0,
        "whole_event_step_C_derivative_upper": full_derivative,
        "whole_event_step_strict_decrease_verified": full_derivative < 0,
        "queries": records,
        "scope": "Only the stored BDF dense polynomial composed with tensor cubic FE interpolation, not the IVP/PDE exact trajectory.",
    }


def simpson_pair(samples, duration):
    """同一组九点值计算四分段与八分段 Simpson 积分，保留原求和顺序。"""
    four = (
        duration / 12 * (samples[0] + 4 * samples[2] + 2 * samples[4] + 4 * samples[6] + samples[8])
    )
    eight = (
        duration
        / 24
        * (samples[0] + 4 * sum(samples[1:8:2]) + 2 * sum(samples[2:8:2]) + samples[8])
    )
    return four, eight


class HeatAudit:
    """统一时间/空间守恒求积；j_heat 只给热通道加 J 权，不改变水分通道。"""

    def __init__(self, model, j_heat=True):
        self.mesh = self.model = model
        self.j_heat = j_heat
        self.simpson = {4: np.zeros(4), 8: np.zeros(4)}
        self.gauss = {8: np.zeros(6), 16: np.zeros(6)}
        self.nodes = {q: leggauss(q) for q in (8, 16)}
        self.steps = 0
        self.weighted = {q: np.zeros(3) for q in [8, 16]}
        self.weighted_simpson = {q: np.zeros(2) for q in [4, 8]}
        self.maximum = PolynomialBounds(model)
        self.tables = {}
        for order in [8, 16]:
            x, w = leggauss(order)
            L = np.stack([p(x) for p in model.radial.bases], axis=1)
            wr = model.radial.width[:, None] / 2 * w
            wz = model.axial.width[:, None] / 2 * w
            self.tables[order] = (L, wr, wz)

    def flux(self, t, y, tail):
        m = self.mesh
        p = self.model
        n = m.size
        T = y[:n].reshape(m.shape)
        C = y[n:].reshape(m.shape)
        R, ell = p.geometry(t)
        Ta, Ce = p.environment(t, tail)
        return np.array(
            [
                2 * p.hm / R * np.dot(m.axial.mass, C[-1] - Ce),
                p.end_factor * p.hm / (p.H * ell) * np.dot(m.radial.mass, C[:, -1] - Ce),
                2 * p.h / R * np.dot(m.axial.mass, T[-1] - Ta),
                p.end_factor * p.h / (p.H * ell) * np.dot(m.radial.mass, T[:, -1] - Ta),
            ]
        )

    def integrate(self, dense, left, right, tail):
        """每个求积点只读取一次状态；普通通道与 J 加权热通道分别累计。"""
        duration = right - left
        if duration <= 0:
            return
        times = np.linspace(left, right, 9)
        flux_samples = [self.flux(t, dense(t), tail) for t in times]
        for order, value in zip((4, 8), simpson_pair(flux_samples, duration)):
            self.simpson[order] += value
        if self.j_heat:
            weighted = [self.J(t) * flux[2:] for t, flux in zip(times, flux_samples)]
            for order, value in zip((4, 8), simpson_pair(weighted, duration)):
                self.weighted_simpson[order] += value
        m, n = self.mesh, self.mesh.size
        for order, (nodes, weights) in self.nodes.items():
            ordinary = np.zeros(6)
            weighted_heat = np.zeros(3)
            for node, weight in zip(nodes, weights):
                t = (left + right) / 2 + duration / 2 * node
                state, derivative = dense(t), dense_derivative(dense, t)
                T, C = state[:n].reshape(m.shape), state[n:].reshape(m.shape)
                capacity, _, _ = self.model.coefficients(T, C)
                volumes = np.array(
                    [
                        np.sum(m.mass * derivative[n:].reshape(m.shape)),
                        np.sum(m.mass * capacity * derivative[:n].reshape(m.shape)),
                    ]
                )
                flux = self.flux(t, state, tail)
                ordinary += weight * np.r_[flux, volumes]
                if self.j_heat:
                    weighted_heat += weight * self.J(t) * np.r_[flux[2:], volumes[1]]
            self.gauss[order] += duration / 2 * ordinary
            if self.j_heat:
                self.weighted[order] += duration / 2 * weighted_heat
        self.steps += 1

    def result(self, initial, final):
        m = self.mesh
        n = m.size
        C0 = initial[n:].reshape(m.shape)
        Cf = final[n:].reshape(m.shape)
        change = float(np.sum(m.mass * (Cf - C0)))
        q = self.gauss[16]
        heat_scale = max(abs(float(q[5])), abs(float(q[2] + q[3])), 1.0)
        result = {
            "gll_mean_C_change": change,
            "time_gauss8": self.gauss[8].tolist(),
            "time_gauss16": q.tolist(),
            "component_order": [
                "side_moisture",
                "end_moisture",
                "side_heat",
                "end_heat",
                "volume_Cdot",
                "volume_bC_Tdot",
            ],
            "gauss16_minus_gauss8": (q - self.gauss[8]).tolist(),
            "simpson4_boundary": self.simpson[4].tolist(),
            "simpson8_boundary": self.simpson[8].tolist(),
            "simpson8_minus_simpson4": (self.simpson[8] - self.simpson[4]).tolist(),
            "gauss16_boundary_minus_simpson8": (q[:4] - self.simpson[8]).tolist(),
            "moisture_balance_abs": abs(float(q[4] + q[0] + q[1])),
            "moisture_balance_normalized": abs(float(q[4] + q[0] + q[1])) / float(np.max(C0)),
            "mean_C_change_minus_integrated_Cdot": change - float(q[4]),
            "heat_balance_abs": abs(float(q[5] + q[2] + q[3])),
            "heat_balance_normalized": abs(float(q[5] + q[2] + q[3])) / heat_scale,
            "accepted_steps_or_final_step_restriction": self.steps,
            "scope": "All-step dense-polynomial temporal quadrature of the GLL lumped semidiscrete balance; heat volume term is b(C)Tdot, not derivative of b(C)T.",
        }
        if self.j_heat:
            v = self.weighted[16]
            result["initial_volume_heat_audit"] = {
                "gauss8": self.weighted[8].tolist(),
                "gauss16": v.tolist(),
                "components": ["J_side_heat", "J_end_heat", "J_volume_bC_Tdot"],
                "gauss16_minus_gauss8": (v - self.weighted[8]).tolist(),
                "simpson4_heat_boundary": self.weighted_simpson[4].tolist(),
                "simpson8_heat_boundary": self.weighted_simpson[8].tolist(),
                "simpson8_minus_simpson4": (
                    self.weighted_simpson[8] - self.weighted_simpson[4]
                ).tolist(),
                "gauss16_heat_boundary_minus_simpson8": (v[:2] - self.weighted_simpson[8]).tolist(),
                "heat_balance_abs_J_per_initial_m3": abs(float(v.sum())),
                "heat_balance_normalized": abs(float(v.sum()))
                / max(abs(float(v[2])), abs(float(v[0] + v[1])), 1.0),
                "J_weighting_effect_on_integrals": (v - self.gauss[16][[2, 3, 5]]).tolist(),
                "J_formula": "(R(t)/R0)^2 * ell(t)",
                "scope": "Every accepted-step dense polynomial, time Gauss8/16. Integral J*b(C)*Tdot and J times heat face rates, per initial volume; not d/dt integral J*b(C)*T. Moisture channels remain unchanged fixed dry-reference mass.",
            }
        return result

    def J(self, t):
        R, ell = self.model.geometry(t)
        return (R / self.model.R0) ** 2 * ell

    def local(self, u):
        m = self.mesh
        u = np.asarray(u).reshape(m.shape)
        return u[
            m.radial.connectivity[:, None, :, None],
            m.axial.connectivity[None, :, None, :],
        ]

    def field(self, u, L):
        return np.einsum("pi,abij,qj->abpq", L, self.local(u), L, optimize=True)

    def audit(self, t, state, derivative, right_tail=False):
        m = self.mesh
        n = m.size
        model = self.model
        T = state[:n].reshape(m.shape)
        C = state[n:].reshape(m.shape)
        Td = derivative[:n].reshape(m.shape)
        Cd = derivative[n:].reshape(m.shape)
        b, _, _ = model.coefficients(T, C)
        R, ell = model.geometry(t)
        Ta, Ce = model.environment(t, right_tail)
        gll = {
            "mean_C": float(np.sum(m.mass * C)),
            "mean_T_K": float(np.sum(m.mass * T)),
            "volume_moisture_rate": float(np.sum(m.mass * Cd)),
            "volume_heat_rate": float(np.sum(m.mass * b * Td)),
            "side_moisture_rate": float(2 * model.hm / R * np.dot(m.axial.mass, C[-1] - Ce)),
            "end_moisture_rate": float(
                model.end_factor * model.hm / (model.H * ell) * np.dot(m.radial.mass, C[:, -1] - Ce)
            ),
            "side_heat_rate": float(2 * model.h / R * np.dot(m.axial.mass, T[-1] - Ta)),
            "end_heat_rate": float(
                model.end_factor * model.h / (model.H * ell) * np.dot(m.radial.mass, T[:, -1] - Ta)
            ),
        }
        rows = {}
        for order, (L, wr, wz) in self.tables.items():
            Tg = self.field(T, L)
            Cg = self.field(C, L)
            Tdg = self.field(Td, L)
            Cdg = self.field(Cd, L)
            if np.any(Cg <= 0) or np.any(Tg <= 0):
                return {
                    "t_s": float(t),
                    "valid": False,
                    "reason": "Nonpositive continuous polynomial value at independent Gauss points; no clipping",
                    "order": order,
                    "minimum_C": float(Cg.min()),
                    "minimum_T_K": float(Tg.min()),
                }
            bg, _, _ = model.coefficients(Tg, Cg)
            integrate = lambda x: float(np.einsum("ap,bq,abpq->", wr, wz, x, optimize=True))
            side_T = np.einsum("qj,bj->bq", L, T[-1, m.axial.connectivity])
            side_C = np.einsum("qj,bj->bq", L, C[-1, m.axial.connectivity])
            end_T = np.einsum("pi,ai->ap", L, T[m.radial.connectivity, -1])
            end_C = np.einsum("pi,ai->ap", L, C[m.radial.connectivity, -1])
            q = {
                "mean_C": integrate(Cg),
                "mean_T_K": integrate(Tg),
                "volume_moisture_rate": integrate(Cdg),
                "volume_heat_rate": integrate(bg * Tdg),
                "side_moisture_rate": float(2 * model.hm / R * np.sum(wz * (side_C - Ce))),
                "end_moisture_rate": float(
                    model.end_factor * model.hm / (model.H * ell) * np.sum(wr * (end_C - Ce))
                ),
                "side_heat_rate": float(2 * model.h / R * np.sum(wz * (side_T - Ta))),
                "end_heat_rate": float(
                    model.end_factor * model.h / (model.H * ell) * np.sum(wr * (end_T - Ta))
                ),
                "minimum_C_at_Gauss_points": float(Cg.min()),
                "maximum_C_at_Gauss_points": float(Cg.max()),
            }
            q["instant_moisture_balance"] = (
                q["volume_moisture_rate"] + q["side_moisture_rate"] + q["end_moisture_rate"]
            )
            q["instant_heat_balance"] = (
                q["volume_heat_rate"] + q["side_heat_rate"] + q["end_heat_rate"]
            )
            rows[order] = q
        negative = self.maximum.bound((-C, -C), threshold="0", tolerance=1e-12)
        valid = negative["classification"] == "dry"
        result = {
            "t_s": float(t),
            "valid": valid,
            "gll": gll,
            "gauss8": rows[8],
            "gauss16": rows[16],
            "gauss16_minus_gll": {k: rows[16][k] - v for k, v in gll.items()},
            "gauss16_minus_gauss8": {k: rows[16][k] - rows[8][k] for k in rows[8]},
            "continuous_C_positivity_certificate": negative,
            "scope": "Independent continuous spatial quadrature at this saved time; no complete continuous space-time/PDE certification.",
        }
        if self.j_heat:
            R, ell = self.model.geometry(t)
            J = (R / self.model.R0) ** 2 * ell
            keys = ["volume_heat_rate", "side_heat_rate", "end_heat_rate"]
            result["current_to_initial_volume_ratio_J"] = J
            weighted = {}
            for section in ["gll", "gauss8", "gauss16"]:
                if section not in result:
                    continue
                weighted[section] = {key: J * result[section][key] for key in keys}
                weighted[section]["instant_heat_balance"] = sum(weighted[section].values())
            for section in ["gauss16_minus_gll", "gauss16_minus_gauss8"]:
                if section in result:
                    weighted[section] = {key: J * result[section][key] for key in keys}
            result["initial_volume_heat_rates"] = weighted
            result["initial_volume_heat_scope"] = (
                "Same exact saved physical time and same model geometry evaluation. Only heat rates multiplied by J; no replacement of all-step time Gauss audit and no multiplication of dry-reference C."
            )
        return result


def read_inputs(directory):
    wb = load_workbook(directory / "附件1.xlsx", read_only=True, data_only=True)
    try:
        ambient = np.array(
            list(wb["Sheet1"].iter_rows(min_row=2, max_col=3, values_only=True)),
            dtype=float,
        )
    finally:
        wb.close()
    wb = load_workbook(directory / "附件2.xlsx", read_only=True, data_only=True)
    try:
        raw = np.array(
            list(wb.active.iter_rows(min_row=2, max_col=2, values_only=True)),
            dtype=float,
        )
    finally:
        wb.close()
    radius = raw.copy()
    radius[:, 1] /= 100
    if ambient.shape != (241, 3) or ambient[-1, 0] != 14400 or radius[-1, 0] != 259200:
        raise ValueError("Unexpected input layout or time units")
    mask = ambient[:, 0] >= 10800
    tails = np.trapezoid(ambient[mask, 1:], ambient[mask, 0], axis=0) / 3600
    return (ambient, radius, tails)


def build_model(
    engine,
    inputs,
    contract,
    material="P4",
    nr=96,
    nz=48,
    elements=32,
    axial_power=0.0,
    core=None,
):
    """FV 复用既有公共物性/半径接口；FE 自行装配并独立读取原始观测。"""
    inputs, contract = (Path(inputs), Path(contract))
    cfg = json.loads(contract.read_text(encoding="utf-8-sig"))
    initial, fixed = (cfg["initial_state"], cfg["fixed_domain_model"])
    if material not in ("P1", "P23", "P4") or axial_power not in (0.0, 0.5, 1.0):
        raise ValueError("未知物性或轴向运动场景。")
    if engine == "fe":
        if elements < 2 or elements % 2:
            raise ValueError("FE 径向单元数须为不小于 2 的偶数。")
        ambient, radius, tails = read_inputs(inputs)
        return TensorFE(
            elements,
            elements // 2,
            properties=cfg["property_sets"][material],
            ambient=ambient,
            radius=radius,
            tails=tails,
            R0=initial["radius_m"],
            H=initial["length_m"] / 2,
            moving=material == "P4",
            gamma=axial_power,
            h=fixed["h_W_m2_K"],
            hm=fixed["h_m_m_s"],
            material=material,
        )
    if engine != "fv" or nr < 2 or nz < 1:
        raise ValueError("FV 网格要求 nr>=2、nz>=1；engine 只能是 fv 或 fe。")
    if core is None:
        from 公共接口 import section

        core = section("moving_model")
    ambient = core.AmbientProfile.from_xlsx(inputs / "附件1.xlsx")
    geometry = (
        core.RadiusProfile.from_xlsx(inputs / "附件2.xlsx")
        if material == "P4"
        else core.FixedRadius(initial["radius_m"])
    )
    grid = core.RadialGrid.surface_stretched(nr, initial["radius_m"], power=2)
    u = np.linspace(0, 1, nz + 1)
    z = initial["length_m"] / 2 * (1 - (1 - u) ** 2)
    model = CylinderFV(
        grid,
        z,
        core.MaterialProperties(**cfg["property_sets"][material]),
        ambient,
        geometry,
        fixed["h_W_m2_K"],
        fixed["h_m_m_s"],
        axial_power=axial_power,
    )
    model.material = material
    return model


def checked_backend(checked, collect, trace):
    """每次求解持有独立的线性求解统计与跟踪器。"""
    statistics = {}
    solver = partial(
        CheckedBDF, checked=checked, collect=collect, statistics=statistics, trace=trace
    )
    return solver, statistics


def solve_fv(
    model,
    out,
    stop,
    settings=None,
    event=True,
    initial=None,
    sample_times=None,
    checked=True,
    collect=True,
    trace=None,
):
    """按有限体积节点最大值检测事件，保留全部输入分段和边界通量求积。"""
    out, settings = prepare_run(out, stop, settings)
    rtol, atol_t, atol_c, max_step = (
        settings.rtol,
        settings.atol_t,
        settings.atol_c,
        settings.max_step,
    )
    solver_class, backend_statistics = checked_backend(checked, collect, trace)
    started = time.perf_counter()
    y = model.state() if initial is None else np.asarray(initial).copy()
    writer = FieldWriter(out, model)
    writer.add(0.0, y)
    if sample_times is None:
        sample_times = np.unique(
            np.r_[np.arange(1.0, min(1800.0, stop) + 1), np.arange(1860.0, stop + 1, 60.0)]
        )
    wanted = np.asarray(sorted((t for t in sample_times if 0 < t <= stop)), dtype=float)
    next_sample = 0
    cuts = np.unique(
        np.r_[model.ambient.breakpoints(0.0, stop), model.geometry.breakpoints(0.0, stop)]
    )
    if cuts[0] != 0 or cuts[-1] != stop or np.any(np.diff(cuts) <= 0):
        raise RuntimeError("Invalid complete input breakpoints")
    statistics = [model.statistics(0.0, y)]
    flux4 = np.zeros(4)
    flux8 = np.zeros(4)
    audit_max = {}
    root = None
    root_state = None
    bracket = None
    finish = stop
    steps = nfev = njev = nlu = 0
    segments = []
    next_progress = 0.0
    min_C = float(model.unpack(y)[1].min())
    max_C = float(model.unpack(y)[1].max())
    min_T = float(model.unpack(y)[0].min())
    boundary = float(model.ambient.breakpoint_s)
    for left, right_end in zip(cuts[:-1], cuts[1:]):
        right = bool(left >= boundary and np.isfinite(boundary))
        if left >= finish:
            break
        T0, C0 = model.unpack(y)
        *_, D0 = model.properties.evaluate(C0, T0)
        first = min(
            0.01,
            0.1
            * min(model.grid.minimum_spacing_m, model.dz.min()) ** 2
            / max(float(D0.max()), 1e-30),
            right_end - left,
        )
        solver = solver_class(
            lambda t, v: model.rhs(t, v, right),
            left,
            y,
            right_end,
            rtol=rtol,
            atol=np.r_[np.full(model.size, atol_t), np.full(model.size, atol_c)],
            first_step=first if left == 0 else None,
            max_step=max_step,
            jac_sparsity=model.sparsity,
        )
        segment_steps = 0
        while solver.status == "running":
            old_t = solver.t
            old_y = solver.y.copy()
            message = solver.step()
            if solver.status == "failed":
                raise RuntimeError(message)
            steps += 1
            segment_steps += 1
            dense = solver.dense_output()
            new_t = solver.t
            if root is None and event and (float(model.unpack(solver.y)[1].max()) <= 0.15):
                old_max = float(model.unpack(old_y)[1].max())
                if old_max <= 0.15:
                    raise RuntimeError("Event bracket begins below threshold")
                root = float(
                    brentq(
                        lambda t: float(model.unpack(dense(t))[1].max() - 0.15),
                        old_t,
                        new_t,
                        xtol=1e-08,
                    )
                )
                root_state = np.asarray(dense(root)).copy()
                bracket = [float(old_t), float(new_t)]
                finish = min(stop, root + 2.0)
                np.savez_compressed(out / "event_state.npz", t_s=root, state=root_state)
            new_t = min(new_t, finish)
            new_y = np.asarray(dense(new_t))
            dt = new_t - old_t
            values = [model.fluxes(t, dense(t), right) for t in np.linspace(old_t, new_t, 9)]
            four, eight = simpson_pair(values, dt)
            flux4 += four
            flux8 += eight
            while next_sample < len(wanted) and wanted[next_sample] <= new_t + 1e-09:
                t = float(wanted[next_sample])
                if t > old_t:
                    state = np.asarray(dense(t))
                    writer.add(t, state)
                    statistics.append(model.statistics(t, state))
                next_sample += 1
            if steps % 100 == 0 or new_t == finish:
                audit = model.audit(new_t, new_y, right)
                for key, value in audit.items():
                    audit_max[key] = max(audit_max.get(key, 0.0), float(value))
            T, C = model.unpack(new_y)
            min_C = min(min_C, float(C.min()))
            max_C = max(max_C, float(C.max()))
            min_T = min(min_T, float(T.min()))
            elapsed = time.perf_counter() - started
            if elapsed >= next_progress:
                progress = {
                    "status": "running",
                    "t_s": float(new_t),
                    "max_C": float(C.max()),
                    "steps": steps,
                    "elapsed_s": elapsed,
                    "interval_s": [float(left), float(right_end)],
                }
                write(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
                next_progress = elapsed + 45.0
            if new_t >= finish:
                y = new_y.copy()
                break
        nfev += solver.nfev
        njev += solver.njev
        nlu += solver.nlu
        segments.append(
            {
                "start_s": float(left),
                "stop_s": float(min(right_end, finish)),
                "steps": segment_steps,
                "nfev": solver.nfev,
                "njev": solver.njev,
                "nlu": solver.nlu,
            }
        )
        if new_t >= finish:
            break
        y = solver.y.copy()
    if event and root is None:
        raise RuntimeError("Threshold not reached within the requested physical horizon")
    writer.add(finish, y)
    writer.finish()
    if abs(statistics[-1][0] - finish) > 1e-09:
        statistics.append(model.statistics(finish, y))
    stats = np.asarray(statistics)
    np.savez_compressed(out / "statistics.npz", statistics=stats)
    initial_y = model.state() if initial is None else initial
    initial_mean = float(np.sum(model.weights * model.unpack(initial_y)[1]))
    final_mean = float(np.sum(model.weights * model.unpack(y)[1]))
    info = {
        "completed": True,
        "event_time_s": root,
        "event_bracket_s": bracket,
        "final_time_s": float(finish),
        "event_max_C": (None if root_state is None else float(model.unpack(root_state)[1].max())),
        "post_event_max_C": float(model.unpack(y)[1].max()),
        "steps": steps,
        "nfev": nfev,
        "njev": njev,
        "nlu": nlu,
        "segments": segments,
        "rtol": rtol,
        "atol_T_K": atol_t,
        "atol_C": atol_c,
        "max_step_s": max_step,
        "elapsed_s": time.perf_counter() - started,
        "saved_rows": writer.count,
        "all_input_knots_are_segment_boundaries": True,
        "initial_mean_C": initial_mean,
        "final_mean_C": final_mean,
        "cumulative_moisture_balance_abs": float(abs(final_mean - initial_mean + sum(flux8[:2]))),
        "boundary_integral_4_vs_8_panel_abs": abs(flux8 - flux4).tolist(),
        "integrated_fluxes": flux8.tolist(),
        "maximum_rate_audit": audit_max,
        "minimum_C_at_accepted_steps": min_C,
        "maximum_C_at_accepted_steps": max_C,
        "minimum_T_at_accepted_steps_K": min_T,
        "no_clipping": True,
        "axial_power": model.axial_power,
        "final_length_m": 2 * model.H * model.axial_scale(finish),
        "not_claimed": "Numerical full-field comparisons under stated physical closure; no continuous-PDE certification or experimental validation.",
    }
    info["backend_statistics"] = backend_statistics if checked else None
    return (info, y)


def solve_fe(
    model,
    out,
    stop,
    settings=None,
    event=True,
    j_heat=True,
    checked=True,
    collect=True,
    trace=None,
    initial=None,
):
    """按连续张量多项式检测事件，同时计算逐步时间与保存时刻空间守恒。"""
    out, setting = prepare_run(out, stop, settings)
    detect_event = bool(event)
    stop_at_s = None if detect_event else stop
    settings = asdict(setting)
    settings.update(
        method="BDF",
        material=model.material,
        gamma=model.gamma,
        radial_elements=model.radial.elements,
        axial_elements=model.axial.elements,
        field_shape=model.shape,
        end_factor=model.end_factor,
        moving=model.moving,
        stop_at_s=stop_at_s,
        event_polynomial_bracket_width_s=0.0001,
        report_tick_s=0.36,
        all_input_knots=True,
        no_clipping=True,
    )
    rtol, atolT, atolC = (setting.rtol, setting.atol_t, setting.atol_c)
    m, n = (model, model.size)
    initial = initial or {"temperature_K": 301.15, "moisture_C": 2.55}
    solver_class, backend_statistics = checked_backend(checked, collect, trace)
    status = {"status": "running", "completed": False, "settings": settings}
    write(out / "status.json", status)
    y = np.r_[np.full(n, initial["temperature_K"]), np.full(n, initial["moisture_C"])]
    y0 = y.copy()
    writer = FieldWriter(out, model)
    writer.add(0.0, y)
    ta = HeatAudit(model, j_heat=j_heat)
    sa = ta
    maximum = sa.maximum
    event_record = None
    report = None
    finish = None
    next_time = 1.0
    steps = 0
    segments = []
    started = time.perf_counter()
    next_progress = 0.0
    prior_witness_failures = []
    space_rows = 0
    space_failures = []
    space_peaks = {}
    initial_audited = False
    special = set()
    min_C = initial["moisture_C"]
    max_C = min_C
    min_T = initial["temperature_K"]
    max_T = min_T
    space_stream = (out / "spatial_audit.jsonl").open("x", encoding="utf-8")

    def space(t, state, dense, tail, kind):
        nonlocal space_rows
        row = sa.audit(t, state, dense_derivative(dense, t), tail)
        row["kind"] = kind
        row["environment_side"] = "right_tail" if tail and t >= 14400 else "observed_or_continuous"
        space_stream.write(json.dumps(row, allow_nan=False) + "\n")
        space_stream.flush()
        space_rows += 1
        if not row["valid"]:
            space_failures.append(
                {
                    "t_s": float(t),
                    "kind": kind,
                    "reason": row.get("reason", "continuous C positivity not proven"),
                }
            )
        for section in ("gauss16_minus_gll", "gauss16_minus_gauss8"):
            for key, value in row.get(section, {}).items():
                tag = section + "/" + key
                if tag not in space_peaks or abs(value) > space_peaks[tag]["absolute"]:
                    space_peaks[tag] = {
                        "absolute": abs(value),
                        "signed": value,
                        "t_s": float(t),
                    }

    def save_special(name, t, dense, tail):
        state = dense(t)
        R, ell = model.geometry(t)
        np.savez_compressed(
            out / (name + ".npz"),
            t_s=t,
            R_m=R,
            ell=ell,
            half_length_m=model.H * ell,
            T_K=state[:n].reshape(m.shape),
            C=state[n:].reshape(m.shape),
        )
        space(t, state, dense, tail, name)
        special.add(name)
        return state

    try:
        cuts = model.knots(stop)
        for left, right in zip(cuts[:-1], cuts[1:]):
            gc.collect()
            tail = left >= 14400.0
            first = None
            if left == 0:
                b, k, D = model.coefficients(y[:n].reshape(m.shape), y[n:].reshape(m.shape))
                shortest = min(
                    np.diff(model.R0 * np.sqrt(m.radial.x)).min(),
                    np.diff(model.H * m.axial.x).min(),
                )
                first = min(
                    0.01,
                    0.1 * shortest**2 / max(float(np.max(k / b)), float(D.max())),
                    float(right - left),
                )
                settings["first_step_s"] = first
            solver = solver_class(
                lambda t, v: model.rhs(t, v, tail),
                float(left),
                y,
                float(right),
                rtol=rtol,
                atol=np.r_[np.full(n, atolT), np.full(n, atolC)],
                first_step=first,
                max_step=setting.max_step,
                jac_sparsity=m.coupled_sparsity,
            )
            count = 0
            while solver.status == "running":
                old_t = float(solver.t)
                old_witness = n + int(np.argmax(solver.y[n:]))
                message = solver.step()
                if solver.status == "failed":
                    raise RuntimeError(message)
                dense = solver.dense_output()
                count += 1
                steps += 1
                if not initial_audited:
                    space(0.0, y0, dense, tail, "initial")
                    initial_audited = True
                if detect_event and event_record is None:
                    boundary = maximum.bound(
                        dense_interval(dense, float(solver.t), slice(n, 2 * n)),
                        threshold=".15",
                        tolerance=1e-13,
                    )
                    if boundary["classification"] == "dry":
                        event_record = bracket_event(dense, maximum, n, 0.0001)
                        event_record["prior_step_witness_failures"] = prior_witness_failures
                        event_record["all_earlier_steps_have_wet_nodal_witness"] = (
                            not prior_witness_failures
                        )
                        event_record["first_computed_crossing_supported"] = (
                            not prior_witness_failures
                            and event_record["whole_event_step_strict_decrease_verified"]
                        )
                        tick = Decimal("0.36")
                        report = float(
                            (
                                Decimal.from_float(event_record["root_upper_s"]) / tick
                            ).to_integral_value(rounding=ROUND_CEILING)
                            * tick
                        )
                        finish = max(report, event_record["root_upper_s"] + 2.0)
                        event_record.update(
                            report_time_s=report,
                            retained_until_s=finish,
                            no_earliest_strict_inequality_time_claim=True,
                            report_margin_scope="Tick above computed polynomial root upper bound; no grid/time-discretization error allowance included yet.",
                        )
                        for name, t in [
                            ("event_lower", event_record["root_lower_s"]),
                            ("event_upper", event_record["root_upper_s"]),
                        ]:
                            save_special(name, t, dense, tail)
                        write(out / "event.json", event_record)
                        print(
                            json.dumps(
                                {
                                    "event_bracket": event_record["root_width_s"],
                                    "root_lower_s": event_record["root_lower_s"],
                                    "root_upper_s": event_record["root_upper_s"],
                                }
                            ),
                            flush=True,
                        )
                    else:
                        witness = scalar_time_bound(dense, old_witness)
                        if not witness["strictly_above_threshold"]:
                            prior_witness_failures.append(
                                {
                                    "left_s": old_t,
                                    "right_s": float(solver.t),
                                    "node_index": old_witness - n,
                                    **witness,
                                }
                            )
                end = float(solver.t) if finish is None else min(float(solver.t), finish)
                ta.integrate(dense, old_t, end, tail)
                while next_time <= end:
                    state = dense(next_time)
                    writer.add(next_time, state)
                    space(next_time, state, dense, tail, "regular")
                    next_time = next_time + 1 if next_time < 1800 else next_time + 60
                if (
                    report is not None
                    and "report_state" not in special
                    and (old_t <= report <= end)
                ):
                    save_special("report_state", report, dense, tail)
                    strict = maximum.bound(
                        dense_interval(dense, report, slice(n, 2 * n)),
                        threshold=".15",
                        tolerance=1e-13,
                    )
                    event_record["report_continuous_C_bound"] = strict
                    event_record["report_strict_for_computed_interpolant"] = (
                        strict["classification"] == "dry"
                    )
                    if strict["classification"] != "dry":
                        raise RuntimeError(
                            "Report tick not strictly dry for the enclosed computed polynomial"
                        )
                    write(out / "event.json", event_record)
                T = solver.y[:n]
                C = solver.y[n:]
                min_C = min(min_C, float(C.min()))
                max_C = max(max_C, float(C.max()))
                min_T = min(min_T, float(T.min()))
                max_T = max(max_T, float(T.max()))
                elapsed = time.perf_counter() - started
                if elapsed >= next_progress:
                    p = {
                        "status": "running",
                        "t_s": float(solver.t),
                        "steps": steps,
                        "segment_steps": count,
                        "step_size_s": float(solver.step_size),
                        "nfev_segment": solver.nfev,
                        "nlu_segment": solver.nlu,
                        "elapsed_seconds": elapsed,
                        "Er": m.radial.elements,
                        "material": model.material,
                        "gamma": model.gamma,
                        "max_nodal_C": float(C.max()),
                        "saved_rows": writer.rows,
                        "spatial_audit_rows": space_rows,
                    }
                    write(out / "progress.json", p)
                    print(json.dumps(p), flush=True)
                    next_progress = elapsed + 45.0
                if finish is not None and solver.t >= finish:
                    y = save_special("final_state", finish, dense, tail)
                    break
            segments.append(
                {
                    "left_s": float(left),
                    "input_right_s": float(right),
                    "last_accepted_t_s": float(solver.t),
                    "steps": count,
                    "nfev": solver.nfev,
                    "njev": solver.njev,
                    "nlu": solver.nlu,
                }
            )
            if finish is not None and solver.t >= finish:
                del solver
                break
            y = solver.y.copy()
            if float(right) == float(cuts[-1]) and stop_at_s is not None:
                save_special("final_state", float(right), dense, tail)
            del solver
        if detect_event and event_record is None:
            raise RuntimeError(f"在指定的 {stop:g} 秒内未检测到干燥事件。")
        if detect_event and ("report_state" not in special or "final_state" not in special):
            raise RuntimeError(
                f"已检测到干燥事件，但 {stop:g} 秒的计算上限不足以完成报告与事件后保留时段。"
            )
        writer.finish()
        space_stream.close()
        unchanged = True
        balance = ta.result(y0, y)
        valid = (
            unchanged
            and (not space_failures)
            and (stop_at_s is not None or event_record["report_strict_for_computed_interpolant"])
        )
        record = {
            "completed": True,
            "numerical_checks_passed": valid,
            "settings": settings,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "segments": segments,
            "accepted_steps": steps,
            "elapsed_seconds": time.perf_counter() - started,
            "saved_regular_times": writer.rows,
            "field_index_sha256": digest(out / "field_index.json"),
            "time_balance": balance,
            "spatial_audit": {
                "rows": space_rows,
                "all_valid": not space_failures,
                "failures": space_failures,
                "peak_absolute_differences": space_peaks,
                "sha256": digest(out / "spatial_audit.jsonl"),
                "scope": "Independent Gauss8/16 continuous spatial integrals at every saved time. Not full continuous space-time certification, and not an additive PDE error bound.",
            },
            "event": event_record,
            "radius_extended_beyond_observed_72h": bool(
                model.moving
                and (finish if detect_event else stop_at_s) > float(model.radius[-1, 0])
            ),
            "nodal_extrema_at_accepted_steps": {
                "min_C": min_C,
                "max_C": max_C,
                "min_T_K": min_T,
                "max_T_K": max_T,
            },
            "linear_backend_statistics": backend_statistics if checked else None,
            "no_general_positivity_or_maximum_principle_claim": True,
            "no_PDE_error_certification": True,
        }
        write(out / "record.json", record)
        status.update(status="completed", completed=True, numerical_checks_passed=valid)
        write(out / "status.json", status)
        print(
            json.dumps(
                {
                    "completed": True,
                    "checks_passed": valid,
                    "material": model.material,
                    "gamma": model.gamma,
                    "Er": m.radial.elements,
                    "elapsed_s": record["elapsed_seconds"],
                    "time_balance": balance,
                }
            ),
            flush=True,
        )
        if not valid:
            raise RuntimeError(
                "Completed trajectory has a failed source/space/report check; preserved without clipping"
            )
        return (record, y)
    except Exception as exc:
        if not space_stream.closed:
            space_stream.close()
        status.update(
            status="failed", completed=False, exception=str(exc), exception_type=type(exc).__name__
        )
        write(out / "status.json", status)
        raise


def run(args):
    """统一入口：生产默认保留原容差；--end 是有界计算，--event 是首次事件。"""
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("请使用新的输出目录。")
    stop = args.horizon if args.event else args.end
    if stop is None or not np.isfinite(stop) or stop <= 0:
        raise ValueError("需要正的 --end 或事件计算的正 --horizon。")
    if args.material != "P4" and args.axial_power != 0:
        raise ValueError("轴向收缩仅适用于 P4。")
    settings = Settings()
    if args.tight:
        settings = (
            Settings(2e-10 / 100, 1e-9 / 100, 2e-12 / 100, 60.0)
            if args.engine == "fe"
            else Settings(2e-11, 1e-10, 2e-13, 30.0)
        )
    model = build_model(
        args.engine,
        args.inputs,
        args.contract,
        args.material,
        args.nr,
        args.nz,
        args.elements,
        args.axial_power,
    )
    out.mkdir(parents=True)
    hashes = {
        p.name: digest(p)
        for p in [
            Path(args.contract),
            Path(args.inputs) / "附件1.xlsx",
            Path(args.inputs) / "附件2.xlsx",
        ]
    }
    options = dict(settings=settings, event=args.event, checked=not args.default_lu, collect=True)
    try:
        result, final = (
            solve_fe(model, out, stop, j_heat=not args.unweighted, **options)
            if args.engine == "fe"
            else solve_fv(model, out, stop, **options)
        )
        result.update(
            engine=args.engine,
            material=args.material,
            input_sha256=hashes,
            refactored_source_sha256=digest(__file__),
        )
        write(out / "summary.json", result)
        return result
    except Exception as error:
        write(out / "status.json", {"status": "failed", "error": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["fv", "fe"], required=True)
    defaults = paths()
    parser.add_argument("--inputs", type=Path, default=defaults["inputs"])
    parser.add_argument("--contract", type=Path, default=defaults["contract"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--material", choices=["P1", "P23", "P4"], default="P4")
    parser.add_argument("--nr", type=int, default=96)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--elements", type=int, default=32)
    parser.add_argument("--axial-power", type=float, choices=[0.0, 0.5, 1.0], default=0.0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--end", type=float)
    mode.add_argument("--event", action="store_true")
    parser.add_argument("--horizon", type=float, default=360000.0)
    parser.add_argument("--tight", action="store_true")
    parser.add_argument(
        "--default-lu",
        action="store_true",
        help="使用 SciPy 默认 LU；默认使用残差检查式 LU。",
    )
    parser.add_argument(
        "--unweighted", action="store_true", help="仅FE：只保留未加J权的基准热量通道。"
    )
    result = run(parser.parse_args(argv))
    print(
        json.dumps(
            {"completed": result["completed"], "engine": result["engine"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
