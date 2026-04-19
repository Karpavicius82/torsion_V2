from __future__ import annotations

"""
Write a custom 1D torsion PDE dataset in The Well HDF5 format.

This file is self-contained:
- simulates trajectories for a 1D scalar torsion-like field S(x,t)
- writes HDF5 files compatible with the_well.data.WellDataset
- optionally creates train/valid/test split folders

References used for format:
- the_well/data/datasets.py
- the_well/data/miniwell.py

Tested assumptions from the_well loader:
- root attrs: dataset_name, grid_type, n_spatial_dims, n_trajectories
- groups: dimensions, t0_fields, t1_fields, t2_fields, scalars, boundary_conditions
- dimensions.attrs['spatial_dims'] defines spatial order
- each field/scalar dataset carries attrs sample_varying/time_varying
- t{i}_fields.attrs['field_names'] and scalars.attrs['field_names'] are required
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math

import h5py
import numpy as np


# -------------------------------
# PDE generator
# -------------------------------

@dataclass
class TorsionParams:
    nx: int = 256
    lx: float = 40.0
    c: float = 1.0
    mass: float = 0.5
    g_nl: float = 0.0
    source_amp: float = 0.5
    source_freq: float = 0.3
    source_sigma: float = 1.5
    source_fraction: float = 0.33
    bc: str = "wall"  # wall | open | periodic
    dt_cfl: float = 0.35


def _step_field(
    s_prev: np.ndarray,
    s_curr: np.ndarray,
    *,
    dt: float,
    dx: float,
    c: float,
    mass: float,
    g_nl: float,
    bc: str,
) -> np.ndarray:
    """One explicit time step for: d_tt S = c^2 d_xx S - m^2 S + g S^3."""
    nx = s_curr.shape[0]
    s_next = np.zeros_like(s_curr)

    lap = (s_curr[2:] - 2.0 * s_curr[1:-1] + s_curr[:-2]) / dx**2
    nonlinear = g_nl * s_curr[1:-1] ** 3
    s_next[1:-1] = (
        2.0 * s_curr[1:-1]
        - s_prev[1:-1]
        + dt**2 * (c**2 * lap - mass**2 * s_curr[1:-1] + nonlinear)
    )

    if bc == "wall":
        s_next[0] = 0.0
        s_next[-1] = 0.0
    elif bc == "periodic":
        lap0 = (s_curr[1] - 2.0 * s_curr[0] + s_curr[-2]) / dx**2
        lapN = (s_curr[1] - 2.0 * s_curr[-1] + s_curr[-2]) / dx**2
        s_next[0] = 2.0 * s_curr[0] - s_prev[0] + dt**2 * (
            c**2 * lap0 - mass**2 * s_curr[0] + g_nl * s_curr[0] ** 3
        )
        s_next[-1] = 2.0 * s_curr[-1] - s_prev[-1] + dt**2 * (
            c**2 * lapN - mass**2 * s_curr[-1] + g_nl * s_curr[-1] ** 3
        )
    elif bc == "open":
        s_next[0] = s_curr[1] + (c * dt - dx) / (c * dt + dx) * (s_next[1] - s_curr[0])
        s_next[-1] = s_curr[-2] + (c * dt - dx) / (c * dt + dx) * (s_next[-2] - s_curr[-1])
    else:
        raise ValueError(f"Unknown boundary condition: {bc}")

    return np.clip(s_next, -10.0, 10.0)


def simulate_trajectory(params: TorsionParams, n_steps: int, seed: int | None = None) -> dict[str, np.ndarray]:
    """Return one trajectory and metadata arrays."""
    rng = np.random.default_rng(seed)

    dx = params.lx / params.nx
    dt = params.dt_cfl * dx / max(params.c, 1e-8)

    x = np.linspace(0.0, params.lx, params.nx, dtype=np.float32)
    t = np.arange(n_steps, dtype=np.float32) * dt

    x0 = params.lx * (0.35 + 0.30 * rng.random())
    sigma = params.source_sigma * (0.75 + 0.50 * rng.random())
    source_steps = int(params.source_fraction * n_steps)

    profile = np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)).astype(np.float32)

    s_prev = np.zeros(params.nx, dtype=np.float32)
    s_curr = np.zeros(params.nx, dtype=np.float32)

    S = np.zeros((n_steps, params.nx), dtype=np.float32)
    dSdt = np.zeros((n_steps, params.nx), dtype=np.float32)
    source_on = np.zeros((n_steps,), dtype=np.float32)
    energy = np.zeros((n_steps,), dtype=np.float32)

    for n in range(n_steps):
        s_next = _step_field(
            s_prev,
            s_curr,
            dt=dt,
            dx=dx,
            c=params.c,
            mass=params.mass,
            g_nl=params.g_nl,
            bc=params.bc,
        )

        if n < source_steps:
            drive = params.source_amp * math.sin(2.0 * math.pi * params.source_freq * t[n]) * profile
            s_next += (dt**2 * drive).astype(np.float32)
            source_on[n] = 1.0

        dsdt = (s_next - s_prev) / (2.0 * dt)
        dsdx = np.gradient(s_curr, dx)
        e = 0.5 * np.trapz(dsdt**2 + params.c**2 * dsdx**2, dx=dx)

        S[n] = s_next
        dSdt[n] = dsdt
        energy[n] = float(e)

        s_prev, s_curr = s_curr, s_next

    return {
        "x": x,
        "time": t,
        "S": S,                 # variable t0 field
        "dSdt": dSdt,           # variable t0 field
        "source_on": source_on, # variable scalar
        "energy": energy,       # variable scalar
        "params": np.array(
            [params.c, params.mass, params.g_nl, dx, dt, x0, sigma, params.source_amp, params.source_freq],
            dtype=np.float32,
        ),
    }


# -------------------------------
# Well-format writer
# -------------------------------

BC_ENUM = {"wall": "WALL", "open": "OPEN", "periodic": "PERIODIC"}


def _string_array(values: Iterable[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=h5py.string_dtype(encoding="utf-8"))


def _write_dimensions_group(
    h5: h5py.File,
    *,
    x: np.ndarray,
    time: np.ndarray,
    n_trajectories: int,
    sample_varying_time: bool = False,
) -> None:
    g = h5.create_group("dimensions")
    g.attrs["spatial_dims"] = _string_array(["x"])

    x_ds = g.create_dataset("x", data=x)
    x_ds.attrs["sample_varying"] = False

    if sample_varying_time:
        time_data = np.broadcast_to(time[None, :], (n_trajectories, time.shape[0])).copy()
        t_ds = g.create_dataset("time", data=time_data)
        t_ds.attrs["sample_varying"] = True
    else:
        t_ds = g.create_dataset("time", data=time)
        t_ds.attrs["sample_varying"] = False


def _write_scalar_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    sample_varying: bool,
    time_varying: bool,
) -> None:
    ds = group.create_dataset(name, data=data)
    ds.attrs["sample_varying"] = sample_varying
    ds.attrs["time_varying"] = time_varying


def _write_field_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    sample_varying: bool,
    time_varying: bool,
    dim_varying: Iterable[str],
) -> None:
    ds = group.create_dataset(name, data=data)
    ds.attrs["sample_varying"] = sample_varying
    ds.attrs["time_varying"] = time_varying
    ds.attrs["dim_varying"] = _string_array(dim_varying)


def _write_boundary_conditions(h5: h5py.File, *, nx: int, bc: str) -> None:
    g = h5.create_group("boundary_conditions")
    bc_name = f"x_{bc.lower()}"
    sub = g.create_group(bc_name)
    sub.attrs["bc_type"] = BC_ENUM[bc]
    sub.attrs["associated_dims"] = _string_array(["x"])

    mask = np.zeros((nx,), dtype=np.bool_)
    mask[0] = True
    mask[-1] = True
    sub.create_dataset("mask", data=mask)


def write_well_hdf5(
    out_path: str | Path,
    trajectories: list[dict[str, np.ndarray]],
    *,
    dataset_name: str = "torsion_1d",
    grid_type: str = "cartesian",
    bc: str = "wall",
) -> Path:
    """
    Write one HDF5 file in The Well format.

    The file is designed to be loadable by the_well.data.WellDataset with
    boundary_return_type='padding'.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_trajectories = len(trajectories)
    if n_trajectories == 0:
        raise ValueError("Need at least one trajectory")

    first = trajectories[0]
    x = first["x"]
    time = first["time"]
    nx = x.shape[0]
    nt = time.shape[0]

    # Stack all samples
    S = np.stack([tr["S"] for tr in trajectories], axis=0)                # [N, T, X]
    dSdt = np.stack([tr["dSdt"] for tr in trajectories], axis=0)          # [N, T, X]
    source_on = np.stack([tr["source_on"] for tr in trajectories], axis=0) # [N, T]
    energy = np.stack([tr["energy"] for tr in trajectories], axis=0)       # [N, T]
    params = np.stack([tr["params"] for tr in trajectories], axis=0)       # [N, K]

    with h5py.File(out_path, "w") as h5:
        # Root attrs used by WellDataset metadata scan
        h5.attrs["dataset_name"] = dataset_name
        h5.attrs["grid_type"] = grid_type
        h5.attrs["n_spatial_dims"] = 1
        h5.attrs["n_trajectories"] = n_trajectories

        _write_dimensions_group(h5, x=x, time=time, n_trajectories=n_trajectories)

        # Scalar / vector / tensor field groups expected by the_well
        t0 = h5.create_group("t0_fields")
        t0.attrs["field_names"] = _string_array(["S", "dSdt"])
        _write_field_dataset(t0, "S", S, sample_varying=True, time_varying=True, dim_varying=["x"])
        _write_field_dataset(t0, "dSdt", dSdt, sample_varying=True, time_varying=True, dim_varying=["x"])

        t1 = h5.create_group("t1_fields")
        t1.attrs["field_names"] = _string_array([])

        t2 = h5.create_group("t2_fields")
        t2.attrs["field_names"] = _string_array([])

        scalars = h5.create_group("scalars")
        scalar_names = [
            "source_on",
            "energy",
            "c",
            "mass",
            "g_nl",
            "dx",
            "dt",
            "source_x0",
            "source_sigma",
            "source_amp",
            "source_freq",
        ]
        scalars.attrs["field_names"] = _string_array(scalar_names)

        _write_scalar_dataset(scalars, "source_on", source_on, sample_varying=True, time_varying=True)
        _write_scalar_dataset(scalars, "energy", energy, sample_varying=True, time_varying=True)
        _write_scalar_dataset(scalars, "c", params[:, 0], sample_varying=True, time_varying=False)
        _write_scalar_dataset(scalars, "mass", params[:, 1], sample_varying=True, time_varying=False)
        _write_scalar_dataset(scalars, "g_nl", params[:, 2], sample_varying=True, time_varying=False)
        _write_scalar_dataset(scalars, "dx", params[:, 3], sample_varying=True, time_varying=False)
        _write_scalar_dataset(scalars, "dt", params[:, 4], sample_varying=True, time_varying=False)
        _write_scalar_dataset(scalars, "source_x0", params[:, 5], sample_varying=True, time_varying=False)
        _write_scalar_dataset(scalars, "source_sigma", params[:, 6], sample_varying=True, time_varying=False)
        _write_scalar_dataset(scalars, "source_amp", params[:, 7], sample_varying=True, time_varying=False)
        _write_scalar_dataset(scalars, "source_freq", params[:, 8], sample_varying=True, time_varying=False)

        _write_boundary_conditions(h5, nx=nx, bc=bc)

    return out_path


# -------------------------------
# Split builder + smoke test
# -------------------------------

def sample_params(rng: np.random.Generator) -> TorsionParams:
    return TorsionParams(
        nx=256,
        lx=40.0,
        c=1.0,
        mass=float(rng.choice([0.0, 0.1, 0.3, 0.5, 1.0])),
        g_nl=float(rng.choice([0.0, 0.2, 0.5, 1.0])),
        source_amp=float(rng.uniform(0.2, 0.8)),
        source_freq=float(rng.uniform(0.1, 0.8)),
        source_sigma=float(rng.uniform(0.8, 2.2)),
        source_fraction=float(rng.uniform(0.15, 0.45)),
        bc=str(rng.choice(["wall", "open", "periodic"])),
        dt_cfl=0.35,
    )


def build_split_file(
    out_path: str | Path,
    *,
    n_trajectories: int,
    n_steps: int,
    dataset_name: str,
    seed: int,
) -> Path:
    rng = np.random.default_rng(seed)
    trajectories: list[dict[str, np.ndarray]] = []
    bcs: list[str] = []

    for i in range(n_trajectories):
        p = sample_params(rng)
        tr = simulate_trajectory(p, n_steps=n_steps, seed=seed + i)
        trajectories.append(tr)
        bcs.append(p.bc)

    # Well files can mix BC groups, but boundary decoding assumes masks grouped by type.
    # For simplicity and clean metadata, write one file per single BC type.
    if len(set(bcs)) != 1:
        # Re-simulate with a fixed BC chosen from the first sample.
        fixed_bc = bcs[0]
        trajectories = []
        for i in range(n_trajectories):
            p = sample_params(rng)
            p.bc = fixed_bc
            trajectories.append(simulate_trajectory(p, n_steps=n_steps, seed=seed + 10_000 + i))
        bc = fixed_bc
    else:
        bc = bcs[0]

    return write_well_hdf5(out_path, trajectories, dataset_name=dataset_name, bc=bc)


def build_dataset_tree(
    base_dir: str | Path,
    *,
    dataset_name: str = "torsion_1d",
    n_train: int = 128,
    n_valid: int = 16,
    n_test: int = 16,
    n_steps: int = 64,
    seed: int = 123,
) -> None:
    """
    Create a minimal The Well-style folder tree:
        base_dir/
          datasets/
            torsion_1d/
              data/
                train/torsion_1d_train.hdf5
                valid/torsion_1d_valid.hdf5
                test/torsion_1d_test.hdf5
    """
    base_dir = Path(base_dir)
    root = base_dir / "datasets" / dataset_name / "data"
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "valid").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)

    build_split_file(root / "train" / f"{dataset_name}_train.hdf5", n_trajectories=n_train, n_steps=n_steps, dataset_name=dataset_name, seed=seed)
    build_split_file(root / "valid" / f"{dataset_name}_valid.hdf5", n_trajectories=n_valid, n_steps=n_steps, dataset_name=dataset_name, seed=seed + 1_000)
    build_split_file(root / "test" / f"{dataset_name}_test.hdf5", n_trajectories=n_test, n_steps=n_steps, dataset_name=dataset_name, seed=seed + 2_000)


def smoke_test_hdf5(path: str | Path) -> None:
    """Quick structural verification without importing the_well."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        assert h5.attrs["n_spatial_dims"] == 1
        assert "dimensions" in h5
        assert "t0_fields" in h5 and "t1_fields" in h5 and "t2_fields" in h5
        assert "scalars" in h5
        assert "boundary_conditions" in h5
        assert "time" in h5["dimensions"] and "x" in h5["dimensions"]
        assert list(h5["t0_fields"].attrs["field_names"]) is not None
        assert "S" in h5["t0_fields"] and "dSdt" in h5["t0_fields"]
        assert "source_on" in h5["scalars"] and "energy" in h5["scalars"]
        print(f"Smoke test OK: {path}")


if __name__ == "__main__":
    out_root = Path(r"d:\SISTEMOS\7. Antigravity2025\Old\torsinonao laukai\torsion_well_data")
    build_dataset_tree(
        out_root,
        dataset_name="torsion_1d",
        n_train=256,
        n_valid=32,
        n_test=32,
        n_steps=64,
        seed=123,
    )
    for split in ["train", "valid", "test"]:
        smoke_test_hdf5(out_root / "datasets" / "torsion_1d" / "data" / split / f"torsion_1d_{split}.hdf5")
    print("Dataset sugeneruotas sekmingai!")
