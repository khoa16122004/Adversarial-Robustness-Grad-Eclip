import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


SOURCES = ["GradECLIP", "GAME", "GradCAM", "MaskCLIP", "RISE"]
RED_COLOR = "red"
EMPTY_DELTA_PHANTOM = "+.000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add transfer deltas vs X-Shift baseline to transfer_result table. "
            "Only off-diagonal cells are annotated; better deltas are colored red."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="script_eval/transfer_result.text",
        help="Input transfer table file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="script_eval/transfer_result_colored.text",
        help="Output LaTeX table file",
    )
    parser.add_argument(
        "--red-color",
        type=str,
        default=RED_COLOR,
        help="LaTeX color name for better deltas",
    )
    return parser.parse_args()


def _split_data_line(line: str) -> Tuple[str, List[str], str]:
    raw = line.rstrip("\n")
    indent_match = re.match(r"^(\s*)", raw)
    indent = indent_match.group(1) if indent_match else ""

    suffix = ""
    suffix_match = re.search(r"(\\\\\s*)$", raw)
    core = raw
    if suffix_match:
        suffix = suffix_match.group(1)
        core = raw[: suffix_match.start()].rstrip()

    if "&" not in core:
        raise ValueError(f"Invalid metric line (missing '&'): {line}")

    parts = [p.strip() for p in core.split("&")]
    # First segment is empty due to leading '&'.
    cells = parts[1:]
    return indent, cells, suffix


def _rebuild_data_line(indent: str, cells: Sequence[str], suffix: str) -> str:
    out = f"{indent}& " + " & ".join(cells)
    if suffix:
        out += f" {suffix}"
    return out + "\n"


def _strip_wrappers(cell: str) -> str:
    s = cell.strip()
    s = re.sub(r"\\textbf\{", "", s)
    s = re.sub(r"\\textcolor\{[^}]*\}\{", "", s)
    s = s.replace("{", "").replace("}", "")
    return s.strip()


def _to_float(cell: str) -> Optional[float]:
    plain = _strip_wrappers(cell)
    if plain == "--":
        return None
    if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", plain):
        return float(plain)
    return None


def _format_delta(delta: Optional[float]) -> Optional[str]:
    if delta is None:
        return None
    text = f"{delta:+.3f}"
    text = text.replace("+0.", "+.")
    text = text.replace("-0.", "-.")
    return text


def _format_cell_with_delta(cell: str, delta: float, better: bool, red_color: str) -> str:
    delta_text = _format_delta(delta)
    if delta_text is None:
        return cell

    delta_render = f"\\textcolor{{{red_color}}}{{{delta_text}}}" if better else delta_text
    return f"{cell}{{\\raisebox{{-0.65ex}}{{\\hspace{{0.12em}}\\tiny {delta_render}}}}}"


def _has_delta_suffix(cell: str) -> bool:
    return "\\raisebox{-0.65ex}" in cell


def _format_cell_with_empty_delta(cell: str) -> str:
    if _has_delta_suffix(cell):
        return cell
    return (
        f"{cell}"
        f"{{\\raisebox{{-0.65ex}}{{\\hspace{{0.12em}}\\tiny \\phantom{{{EMPTY_DELTA_PHANTOM}}}}}}}"
    )


def _collect_xshift_baseline(lines: List[str]) -> List[List[float]]:
    metric_rows: List[List[float]] = []

    for i, line in enumerate(lines):
        if line.strip() != "X-Shift":
            continue

        idx = i + 1
        while idx < len(lines) and len(metric_rows) < 3:
            stripped = lines[idx].strip()
            if not stripped.startswith("&"):
                idx += 1
                continue

            _, cells, _ = _split_data_line(lines[idx])
            # Source row marker line '& --' should be skipped.
            if len(cells) == 1:
                idx += 1
                continue

            if len(cells) != len(SOURCES):
                raise ValueError(
                    f"Unexpected X-Shift metric width at line {idx + 1}: {len(cells)}"
                )

            row_vals = []
            for cell in cells:
                val = _to_float(cell)
                if val is None:
                    raise ValueError(
                        f"Non-numeric X-Shift baseline at line {idx + 1}: {cell}"
                    )
                row_vals.append(val)
            metric_rows.append(row_vals)
            idx += 1
        break

    if len(metric_rows) != 3:
        raise ValueError("Could not parse full X-Shift baseline (IMD/SRC/ROAD)")

    return metric_rows


def _process_block(lines: List[str], block_start: int, baseline: List[List[float]], red_color: str) -> None:
    idx = block_start + 1
    source_pos = 0

    while idx < len(lines) and source_pos < len(SOURCES):
        stripped = lines[idx].strip()

        if stripped.startswith("\\midrule") or stripped.startswith("\\bottomrule"):
            break

        if not stripped.startswith("&"):
            idx += 1
            continue

        _, source_cells, _ = _split_data_line(lines[idx])
        if len(source_cells) != 1:
            idx += 1
            continue

        source_name = _strip_wrappers(source_cells[0])
        if source_name not in SOURCES:
            idx += 1
            continue

        source_idx = SOURCES.index(source_name)

        # Next 3 lines are IMD, SRC, ROAD rows.
        for metric_idx in range(3):
            line_idx = idx + 1 + metric_idx
            if line_idx >= len(lines):
                raise ValueError("Unexpected EOF while processing transfer block")

            indent, cells, suffix = _split_data_line(lines[line_idx])
            if len(cells) != len(SOURCES):
                raise ValueError(
                    f"Unexpected metric width at line {line_idx + 1}: {len(cells)}"
                )

            for target_idx, cell in enumerate(cells):
                # Keep diagonal and already-bold cells unchanged.
                if target_idx == source_idx:
                    continue
                if "\\textbf" in cell:
                    continue

                value = _to_float(cell)
                if value is None:
                    continue

                base = baseline[metric_idx][target_idx]
                delta = value - base

                if metric_idx in (0, 1):
                    better = delta < 0
                else:
                    better = delta > 0

                cells[target_idx] = _format_cell_with_delta(cell, delta, better, red_color)

            lines[line_idx] = _rebuild_data_line(indent, cells, suffix)

        source_pos += 1
        idx += 1


def rewrite_transfer_table(input_path: Path, output_path: Path, red_color: str) -> None:
    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    baseline = _collect_xshift_baseline(lines)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("\\multirow") and "{IOA}" in stripped:
            _process_block(lines, i, baseline, red_color)
        elif stripped.startswith("\\multirow") and "{DOA}" in stripped:
            _process_block(lines, i, baseline, red_color)
        elif stripped.startswith("\\multirow") and "{IOA+DOA}" in stripped:
            _process_block(lines, i, baseline, red_color)

    # Keep table rows visually aligned by adding empty delta boxes to remaining numeric cells
    # (e.g., diagonal bold cells and all X-Shift baseline cells).
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("&"):
            continue
        try:
            indent, cells, suffix = _split_data_line(line)
        except ValueError:
            continue
        if len(cells) != len(SOURCES):
            continue

        changed = False
        for j, cell in enumerate(cells):
            if _to_float(cell) is None:
                continue
            filled = _format_cell_with_empty_delta(cell)
            if filled != cell:
                cells[j] = filled
                changed = True

        if changed:
            lines[i] = _rebuild_data_line(indent, cells, suffix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rewrite_transfer_table(input_path, output_path, args.red_color)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
