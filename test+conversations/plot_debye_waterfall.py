from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a Schneider-style waterfall from Debye.c sim.* snapshots."
    )
    parser.add_argument("--source-file", type=Path, default=Path("Debye.c"))
    parser.add_argument("--basename", default="sim", help="snapshot basename")
    parser.add_argument("--directory", type=Path, help="snapshot directory")
    parser.add_argument("--output", type=Path, default=Path("debye_c_waterfall.png"))
    parser.add_argument("--interval", type=int, help="time steps between snapshots")
    parser.add_argument("--offset", type=float, default=1.3, help="vertical offset per trace")
    parser.add_argument("--scale", type=float, default=3.5, help="field scale factor")
    parser.add_argument("--gain", type=float, default=2.0, help="extra field gain")
    parser.add_argument("--source-cell", type=int)
    parser.add_argument("--slab-start", type=int)
    parser.add_argument("--slab-end", type=int, help="exclusive Debye slab end cell")
    parser.add_argument("--xlim", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--frames", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--every", type=int, default=1, help="plot every Nth snapshot")
    parser.add_argument(
        "--normalize-traces",
        action="store_true",
        help="normalize each trace before applying scale/gain",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="plot all snapshots across the full grid instead of the zoomed default",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--show", action="store_true", help="show an interactive plot window")
    return parser.parse_args()


def read_c_int_define(source: str, name: str) -> int | None:
    match = re.search(rf"^\s*#define\s+{re.escape(name)}\s+(\d+)\b", source, re.MULTILINE)
    if match is None:
        return None

    return int(match.group(1))


def read_c_string_define(source: str, name: str) -> str | None:
    match = re.search(rf'^\s*#define\s+{re.escape(name)}\s+"([^"]+)"', source, re.MULTILINE)
    if match is None:
        return None

    return match.group(1)


def read_source_cell(source: str) -> int | None:
    match = re.search(r"\bez\s*\[\s*(\d+)\s*\]\s*\+=", source)
    if match is None:
        return None

    return int(match.group(1))


def apply_source_defaults(args: argparse.Namespace) -> None:
    grid_size = None

    if args.source_file.exists():
        source = args.source_file.read_text()
        grid_size = read_c_int_define(source, "SIZE")
        if args.source_cell is None:
            args.source_cell = read_source_cell(source)
        if args.slab_start is None:
            args.slab_start = read_c_int_define(source, "DISPERSION_LAYER")
        if args.slab_end is None:
            args.slab_end = read_c_int_define(source, "DISPERSION_LAYER_END")
        if args.interval is None:
            args.interval = read_c_int_define(source, "SNAPSHOT_INTERVAL")
        if args.directory is None:
            snapshot_dir = read_c_string_define(source, "SNAPSHOT_DIR")
            if snapshot_dir is not None:
                args.directory = Path(snapshot_dir)

    if args.source_cell is None:
        args.source_cell = 50
    if args.slab_start is None:
        args.slab_start = 170
    if args.slab_end is None:
        args.slab_end = 190
    if args.interval is None:
        args.interval = 10
    if args.directory is None:
        args.directory = Path("snapshots")

    if not args.full:
        if args.frames is None:
            args.frames = [7, 35]
        if args.xlim is None:
            x_start = max(0, min(args.source_cell, args.slab_start) - 15)
            x_end = args.slab_end + 25
            if grid_size is not None:
                x_end = min(grid_size - 1, x_end)
            args.xlim = [x_start, x_end]


def select_snapshots(
    snapshots: list[tuple[int, list[float]]],
    frames: list[int] | None,
    every: int,
) -> list[tuple[int, list[float]]]:
    if every < 1:
        raise ValueError("--every must be at least 1")

    if frames is not None:
        first, last = frames
        if first > last:
            raise ValueError("--frames FIRST must be less than or equal to LAST")
        snapshots = [
            (index, values)
            for index, values in snapshots
            if first <= index <= last
        ]

    snapshots = snapshots[::every]
    if not snapshots:
        raise ValueError("no snapshots remain after applying --frames/--every")

    return snapshots


def snapshot_index(path: Path, basename: str) -> int | None:
    prefix = f"{basename}."
    if not path.name.startswith(prefix):
        return None

    suffix = path.name[len(prefix) :]
    if not suffix.isdigit():
        return None

    return int(suffix)


def read_snapshots(directory: Path, basename: str) -> list[tuple[int, list[float]]]:
    snapshots: list[tuple[int, list[float]]] = []

    for path in directory.glob(f"{basename}.*"):
        index = snapshot_index(path, basename)
        if index is None:
            continue

        with path.open() as snapshot_file:
            values = [float(line) for line in snapshot_file if line.strip()]

        snapshots.append((index, values))

    snapshots.sort(key=lambda item: item[0])

    if not snapshots:
        raise FileNotFoundError(
            f"no {basename}.* snapshots found in {directory}. "
            "Run Debye.c first to create them."
        )

    cell_count = len(snapshots[0][1])
    for index, values in snapshots:
        if len(values) != cell_count:
            raise ValueError(
                f"{basename}.{index} has {len(values)} cells, expected {cell_count}"
            )

    return snapshots


def spread_ticks(trace_count: int, tick_count: int = 10) -> list[int]:
    tick_count = min(tick_count, trace_count)
    if tick_count <= 1:
        return [0]

    return sorted(
        {
            round(i * (trace_count - 1) / (tick_count - 1))
            for i in range(tick_count)
        }
    )


def write_waterfall(args: argparse.Namespace) -> Path:
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    snapshots = read_snapshots(args.directory, args.basename)
    snapshots = select_snapshots(snapshots, args.frames, args.every)
    trace_count = len(snapshots)
    cell_count = len(snapshots[0][1])
    cells = list(range(cell_count))
    peak_field = max(abs(value) for _, values in snapshots for value in values)
    if peak_field == 0.0:
        peak_field = 1.0
    plot_peak = 1.0 if args.normalize_traces else peak_field

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.axvspan(
        args.slab_start,
        args.slab_end - 1,
        color="#d8dde6",
        alpha=0.55,
        label=f"Debye slab {args.slab_start}-{args.slab_end - 1}",
        zorder=0,
    )

    for trace_number, (_, values) in enumerate(snapshots):
        baseline = args.offset * trace_number
        trace_peak = max(abs(value) for value in values)
        if args.normalize_traces and trace_peak > 0.0:
            trace_values = [value / trace_peak for value in values]
        else:
            trace_values = values

        trace = [args.scale * args.gain * value + baseline for value in trace_values]
        ax.plot(cells, trace, "k", linewidth=0.65)

    ax.axvline(args.source_cell, color="#2f6f9f", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.axvline(args.slab_start, color="#59677a", linewidth=0.8, linestyle=":", alpha=0.75)
    ax.axvline(args.slab_end - 1, color="#59677a", linewidth=0.8, linestyle=":", alpha=0.75)

    tick_indices = spread_ticks(trace_count)
    ax.set_yticks([args.offset * tick_index for tick_index in tick_indices])
    ax.set_yticklabels(
        [str(snapshots[tick_index][0] * args.interval) for tick_index in tick_indices]
    )

    title = "Debye 1D FDTD Waterfall"
    if args.normalize_traces:
        title += " (trace-normalized)"
    if args.gain != 1.0:
        title += f" (gain {args.gain:g})"

    ax.set_title(title)
    ax.set_xlabel("Space [cell index]")
    ax.set_ylabel("Time [time step]")
    if args.xlim is None:
        ax.set_xlim(0, cell_count - 1)
    else:
        x_min, x_max = args.xlim
        if x_min >= x_max:
            raise ValueError("--xlim START must be less than END")
        ax.set_xlim(x_min, x_max)

    ax.set_ylim(
        -args.scale * args.gain * plot_peak,
        args.offset * (trace_count - 1) + args.scale * args.gain * plot_peak,
    )
    ax.text(args.source_cell + 2, 0.2, "source", color="#2f6f9f", fontsize=9)
    ax.text(
        args.slab_start + 2,
        args.offset * (trace_count - 1) + 0.15,
        f"Debye slab {args.slab_start}-{args.slab_end - 1}",
        color="#59677a",
        fontsize=9,
    )
    ax.grid(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return args.output


def main() -> None:
    args = parse_args()
    apply_source_defaults(args)
    output = write_waterfall(args)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
