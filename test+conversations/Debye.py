from dataclasses import dataclass
from pathlib import Path

import numpy as np


SIZE = 200
MAX_TIME = 450
IMP0 = 377.0
COURANT = 1.0
SNAPSHOT_INTERVAL = 10
WATERFALL_PATH = Path("debye_waterfall.png")
WATERFALL_OFFSET = 1.0
WATERFALL_SCALE = 1.9

SOURCE_POSITION = 40
PROBE_POSITION = 160
SLAB_START = 100
SLAB_END = 150


@dataclass
class DebyeGrid:
    ez: np.ndarray
    hy: np.ndarray
    jp: np.ndarray
    ca: np.ndarray
    cb: np.ndarray
    cjj: np.ndarray
    cje: np.ndarray
    slab: slice
    left_abc: float = 0.0
    right_abc: float = 0.0


@dataclass
class SimulationResult:
    grid: DebyeGrid
    source_history: np.ndarray
    probe_history: np.ndarray
    snapshots: np.ndarray
    snapshot_times: np.ndarray


def init_debye_grid(
    size: int,
    slab: tuple[int, int],
    eps_inf: float,
    eps_static: float,
    relaxation_time_steps: float,
    sigma: float = 0.0,
    dt: float = 1.0,
    eps0: float = 1.0,
    imp0: float = IMP0,
    courant: float = COURANT,
) -> DebyeGrid:
    """Create a 1D FDTD grid with a Debye slab and free space elsewhere."""
    if size < 3:
        raise ValueError("size must be at least 3")

    slab_start, slab_end = slab
    if not 1 <= slab_start < slab_end <= size - 1:
        raise ValueError("slab must be inside the non-boundary grid cells")
    if eps_static < eps_inf:
        raise ValueError("eps_static must be greater than or equal to eps_inf")
    if relaxation_time_steps <= 0:
        raise ValueError("relaxation_time_steps must be positive")

    ez = np.zeros(size)
    hy = np.zeros(size - 1)
    jp = np.zeros(size)

    eps_inf_grid = np.ones(size)
    eps_inf_grid[slab_start:slab_end] = eps_inf

    delta_eps = np.zeros(size)
    delta_eps[slab_start:slab_end] = eps_static - eps_inf

    sigma_grid = np.zeros(size)
    sigma_grid[slab_start:slab_end] = sigma

    nt = relaxation_time_steps
    cjj = np.zeros(size)
    cje = np.zeros(size)

    cjj[slab_start:slab_end] = (1.0 - 0.5 / nt) / (1.0 + 0.5 / nt)
    cje[slab_start:slab_end] = (
        (delta_eps[slab_start:slab_end] / (imp0 * courant))
        * (1.0 / nt)
        / (1.0 + 0.5 / nt)
    )

    loss = sigma_grid * dt / (2.0 * eps_inf_grid * eps0)
    debye_coupling = cje * imp0 * courant / (2.0 * eps_inf_grid)
    denom = 1.0 + loss + debye_coupling

    ca = (1.0 - loss + debye_coupling) / denom
    cb = (imp0 * courant / eps_inf_grid) / denom

    return DebyeGrid(
        ez=ez,
        hy=hy,
        jp=jp,
        ca=ca,
        cb=cb,
        cjj=cjj,
        cje=cje,
        slab=slice(slab_start, slab_end),
    )


def gaussian_pulse(time_step: int, delay: float = 35.0, width: float = 12.0) -> float:
    return float(np.exp(-((time_step - delay) / width) ** 2))


def update_h(grid: DebyeGrid, imp0: float = IMP0, courant: float = COURANT) -> None:
    grid.hy += (courant / imp0) * (grid.ez[1:] - grid.ez[:-1])


def update_e(grid: DebyeGrid) -> None:
    old_ez = grid.ez.copy()

    curl_h = grid.hy[1:] - grid.hy[:-1]
    interior = slice(1, -1)
    grid.ez[interior] = grid.ca[interior] * old_ez[interior] + grid.cb[interior] * (
        curl_h - 0.5 * (1.0 + grid.cjj[interior]) * grid.jp[interior]
    )

    grid.jp[interior] = (
        grid.cjj[interior] * grid.jp[interior]
        + grid.cje[interior] * (grid.ez[interior] - old_ez[interior])
    )

    grid.ez[0] = grid.left_abc
    grid.left_abc = grid.ez[1]
    grid.ez[-1] = grid.right_abc
    grid.right_abc = grid.ez[-2]


def run_simulation(
    max_time: int = MAX_TIME,
    snapshot_interval: int = SNAPSHOT_INTERVAL,
) -> SimulationResult:
    grid = init_debye_grid(
        size=SIZE,
        slab=(SLAB_START, SLAB_END),
        eps_inf=2.0,
        eps_static=5.0,
        relaxation_time_steps=50.0,
    )

    probe_history = np.zeros(max_time)
    source_history = np.zeros(max_time)
    snapshots: list[np.ndarray] = []
    snapshot_times: list[int] = []

    for time_step in range(max_time):
        update_h(grid)
        update_e(grid)

        source = gaussian_pulse(time_step)
        grid.ez[SOURCE_POSITION] += source

        source_history[time_step] = source
        probe_history[time_step] = grid.ez[PROBE_POSITION]

        if snapshot_interval > 0 and time_step % snapshot_interval == 0:
            snapshots.append(grid.ez.copy())
            snapshot_times.append(time_step)

    return SimulationResult(
        grid=grid,
        source_history=source_history,
        probe_history=probe_history,
        snapshots=np.array(snapshots),
        snapshot_times=np.array(snapshot_times, dtype=int),
    )


def run(max_time: int = MAX_TIME) -> tuple[DebyeGrid, np.ndarray, np.ndarray]:
    result = run_simulation(max_time=max_time, snapshot_interval=0)
    return result.grid, result.source_history, result.probe_history


def write_waterfall_plot(
    snapshots: np.ndarray,
    snapshot_times: np.ndarray,
    slab: slice,
    output_path: Path = WATERFALL_PATH,
    offset: float = WATERFALL_OFFSET,
    scale: float = WATERFALL_SCALE,
    show: bool = False,
) -> Path:
    """Write a Schneider-style 2D waterfall plot of offset 1D Ez snapshots."""
    if snapshots.size == 0:
        raise ValueError("no snapshots were recorded")
    if offset <= 0.0:
        raise ValueError("offset must be positive")

    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for the waterfall plot. "
            "Activate the venv that has matplotlib installed, then run `python Debye.py`."
        ) from exc

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    trace_count, cell_count = snapshots.shape
    cell_indices = np.arange(cell_count)
    peak_field = float(np.max(np.abs(snapshots)))
    if peak_field == 0.0:
        peak_field = 1.0

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.axvspan(
        slab.start,
        slab.stop - 1,
        color="#d8dde6",
        alpha=0.55,
        label=f"Debye slab {slab.start}-{slab.stop - 1}",
        zorder=0,
    )

    for trace_index, snapshot in enumerate(snapshots):
        baseline = offset * trace_index
        ax.plot(cell_indices, scale * snapshot + baseline, "k", linewidth=0.65)

    ax.axvline(SOURCE_POSITION, color="#2f6f9f", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.axvline(PROBE_POSITION, color="#a14f38", linewidth=0.9, linestyle="--", alpha=0.75)

    tick_count = min(10, trace_count)
    tick_indices = np.linspace(0, trace_count - 1, tick_count, dtype=int)
    ax.set_yticks(offset * tick_indices)
    ax.set_yticklabels(snapshot_times[tick_indices])

    ax.set_title("Debye 1D FDTD Waterfall")
    ax.set_xlabel("Space [spatial index]")
    ax.set_ylabel("Time [time step]")
    ax.set_xlim(0, cell_count - 1)
    ax.set_ylim(-scale * peak_field, offset * (trace_count - 1) + scale * peak_field)
    ax.text(SOURCE_POSITION + 2, 0.2, "source", color="#2f6f9f", fontsize=9)
    ax.text(PROBE_POSITION + 2, 0.2, "probe", color="#a14f38", fontsize=9)
    ax.text(
        slab.start + 2,
        offset * (trace_count - 1) + 0.15,
        f"Debye slab {slab.start}-{slab.stop - 1}",
        color="#59677a",
        fontsize=9,
    )
    ax.grid(False)

    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main() -> None:
    result = run_simulation()
    waterfall_path = write_waterfall_plot(
        result.snapshots,
        result.snapshot_times,
        result.grid.slab,
    )

    print("Debye 1D FDTD run complete")
    print(f"grid cells: {SIZE}")
    print(f"time steps: {MAX_TIME}")
    print(f"Debye slab: cells {SLAB_START}..{SLAB_END - 1}")
    print(f"source cell: {SOURCE_POSITION}")
    print(f"probe cell: {PROBE_POSITION}")
    print(f"waterfall snapshots: {len(result.snapshot_times)}")
    print(f"waterfall plot: {waterfall_path}")
    print(f"peak source Ez: {np.max(np.abs(result.source_history)):.6e}")
    print(f"peak probe Ez: {np.max(np.abs(result.probe_history)):.6e}")
    print(f"final max |Ez|: {np.max(np.abs(result.grid.ez)):.6e}")
    print(f"final max |Jp| in slab: {np.max(np.abs(result.grid.jp[result.grid.slab])):.6e}")


if __name__ == "__main__":
    main()
