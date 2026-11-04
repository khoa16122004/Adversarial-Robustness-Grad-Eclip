import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class MetricCell:
	line_idx: int
	col_idx: int
	raw: str
	plain: str
	value: Optional[float]


@dataclass
class AttackRow:
	attack_name: str
	metric_cells: List[List[MetricCell]]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Rewrite LaTeX faithfulness table and highlight better attack results in red. "
			"Rules: IMD/ROAD compared to Without Attack, Spearman compared to X-Shift."
		)
	)
	parser.add_argument(
		"--input",
		type=str,
		default="script_eval/result.txt",
		help="Input LaTeX table path",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="script_eval/result_red.txt",
		help="Output LaTeX table path",
	)
	parser.add_argument(
		"--red-color",
		type=str,
		default="red",
		help="LaTeX color name for highlighted values (default: red)",
	)
	return parser.parse_args()


def _normalize_text(text: str) -> str:
	s = text.strip()
	s = re.sub(r"\\textbf\{", "", s)
	s = re.sub(r"\\textcolor\{[^}]*\}\{", "", s)
	s = s.replace("{", "").replace("}", "")
	s = re.sub(r"\\", "", s)
	s = re.sub(r"\s+", " ", s)
	return s.strip().lower()


def _strip_numeric_wrappers(cell: str) -> str:
	s = cell.strip()
	s = re.sub(r"\\textbf\{", "", s)
	s = re.sub(r"\\textcolor\{[^}]*\}\{", "", s)
	s = s.replace("{", "").replace("}", "")
	return s.strip()


def _parse_value(cell: str) -> Tuple[str, Optional[float]]:
	plain = _strip_numeric_wrappers(cell)
	if plain == "--":
		return plain, None
	try:
		return plain, float(plain)
	except ValueError:
		return plain, None


def _split_ampersand_line(line: str) -> List[str]:
	stripped = line.strip()
	if not stripped.startswith("&"):
		return []
	parts = [p.strip() for p in stripped.split("&")[1:]]
	return parts


def _is_attack_header_line(parts: List[str]) -> bool:
	if not parts:
		return False
	first = _normalize_text(parts[0])
	if not first:
		return False
	# Header line should be attack label (e.g., Without Attack, X-Shift, IOA).
	return first not in {"--"} and not re.fullmatch(r"[-+]?\d+(\.\d+)?", first)


def _rebuild_metric_line(cells: List[str]) -> str:
	return f"& {cells[0]} & {cells[1]} & {cells[2]}"


def _format_delta(delta: Optional[float]) -> Optional[str]:
	if delta is None:
		return None
	rounded = float(f"{delta:+.3f}")
	text = f"{rounded:+.3f}"
	text = text.replace("+0.", "+.")
	text = text.replace("-0.", "-.")
	text = text.replace("0.", ".")
	return text


def _format_cell_with_delta(
	value_text: str,
	value_rendered: str,
	delta: Optional[float],
	highlight_delta: bool,
	red_color: str,
) -> str:
	delta_text = _format_delta(delta)
	if value_text == "--" or delta_text is None:
		return value_rendered

	if highlight_delta:
		delta_part = f"\\textcolor{{{red_color}}}{{{delta_text}}}"
	else:
		delta_part = delta_text

	# Render base value smaller and show a slightly larger corner delta.
	return (
		f"{{\\scriptsize {value_rendered}}}"
		f"{{\\raisebox{{-0.65ex}}{{\\hspace{{0.12em}}\\tiny {delta_part}}}}}"
	)


def _format_base_only(value_text: str, value_rendered: str) -> str:
	if value_text == "--":
		return value_rendered
	return f"{{\\scriptsize {value_rendered}}}"


def _find_attack_row_start(lines: List[str], start_idx: int) -> Optional[int]:
	for i in range(start_idx, len(lines)):
		parts = _split_ampersand_line(lines[i])
		if _is_attack_header_line(parts):
			return i
	return None


def _parse_attack_row(lines: List[str], row_start: int, dataset_count: int) -> Optional[Tuple[AttackRow, int]]:
	header_parts = _split_ampersand_line(lines[row_start])
	if not _is_attack_header_line(header_parts):
		return None

	attack_name = _strip_numeric_wrappers(header_parts[0])
	metric_cells: List[List[MetricCell]] = []
	idx = row_start + 1

	for _ in range(dataset_count):
		if idx >= len(lines):
			return None
		parts = _split_ampersand_line(lines[idx])
		if len(parts) < 3:
			return None

		row_cells: List[MetricCell] = []
		for col_idx in range(3):
			plain, value = _parse_value(parts[col_idx])
			row_cells.append(
				MetricCell(
					line_idx=idx,
					col_idx=col_idx,
					raw=parts[col_idx],
					plain=plain,
					value=value,
				)
			)
		metric_cells.append(row_cells)
		idx += 1

	# Move to the row-ending line containing "\\" if present.
	while idx < len(lines) and "\\" not in lines[idx]:
		idx += 1
	if idx < len(lines):
		idx += 1

	return AttackRow(attack_name=attack_name, metric_cells=metric_cells), idx


def _find_block_end(lines: List[str], block_start: int) -> int:
	for i in range(block_start + 1, len(lines)):
		if "\\multirow" in lines[i] or lines[i].strip() == "\\bottomrule":
			return i
	return len(lines)


def _attack_key(name: str) -> str:
	key = _normalize_text(name)
	key = key.replace("+", "plus")
	return key


def _apply_highlight_for_block(lines: List[str], block_start: int, block_end: int, red_color: str) -> None:
	dataset_count = 3
	rows: List[AttackRow] = []
	idx = block_start

	while idx < block_end:
		row_start = _find_attack_row_start(lines, idx)
		if row_start is None or row_start >= block_end:
			break
		parsed = _parse_attack_row(lines, row_start, dataset_count)
		if parsed is None:
			idx = row_start + 1
			continue
		row, next_idx = parsed
		rows.append(row)
		idx = next_idx

	if not rows:
		return

	by_key = {_attack_key(r.attack_name): r for r in rows}
	clean = by_key.get("without attack")
	xshift = by_key.get("x_shift") or by_key.get("x-shift") or by_key.get("xshift")
	if clean is None:
		return

	for row in rows:
		row_key = _attack_key(row.attack_name)
		for ds_idx in range(dataset_count):
			imd_cell = row.metric_cells[ds_idx][0]
			spear_cell = row.metric_cells[ds_idx][1]
			road_cell = row.metric_cells[ds_idx][2]

			clean_imd = clean.metric_cells[ds_idx][0].value
			clean_road = clean.metric_cells[ds_idx][2].value

			# Do not annotate baseline row.
			if row_key == "without attack":
				parts = _split_ampersand_line(lines[imd_cell.line_idx])
				if len(parts) >= 3:
					parts[0] = _format_base_only(imd_cell.plain, imd_cell.raw)
					parts[1] = _format_base_only(spear_cell.plain, spear_cell.raw)
					parts[2] = _format_base_only(road_cell.plain, road_cell.raw)
					lines[imd_cell.line_idx] = _rebuild_metric_line(parts)
				continue

			imd_delta = None
			if imd_cell.value is not None and clean_imd is not None:
				imd_delta = imd_cell.value - clean_imd
			better_imd = imd_delta is not None and imd_delta < 0

			road_delta = None
			if road_cell.value is not None and clean_road is not None:
				road_delta = road_cell.value - clean_road
			better_road = road_delta is not None and road_delta > 0

			xshift_key = row_key in {"x_shift", "x-shift", "xshift"}
			spear_delta = None
			better_spear = False
			if (not xshift_key) and xshift is not None:
				xshift_spear = xshift.metric_cells[ds_idx][1].value
				if spear_cell.value is not None and xshift_spear is not None:
					spear_delta = spear_cell.value - xshift_spear
					better_spear = spear_delta < 0

			parts = _split_ampersand_line(lines[imd_cell.line_idx])
			if len(parts) >= 3:
				parts[0] = _format_cell_with_delta(
					value_text=imd_cell.plain,
					value_rendered=imd_cell.raw,
					delta=imd_delta,
					highlight_delta=better_imd,
					red_color=red_color,
				)
				parts[1] = _format_cell_with_delta(
					value_text=spear_cell.plain,
					value_rendered=spear_cell.raw,
					delta=spear_delta,
					highlight_delta=better_spear,
					red_color=red_color,
				)
				parts[2] = _format_cell_with_delta(
					value_text=road_cell.plain,
					value_rendered=road_cell.raw,
					delta=road_delta,
					highlight_delta=better_road,
					red_color=red_color,
				)
				lines[imd_cell.line_idx] = _rebuild_metric_line(parts)


def rewrite_latex_table(input_path: Path, output_path: Path, red_color: str) -> None:
	text = input_path.read_text(encoding="utf-8")
	lines = text.splitlines()

	block_starts = [i for i, ln in enumerate(lines) if "\\multirow" in ln]
	for start in block_starts:
		end = _find_block_end(lines, start)
		_apply_highlight_for_block(lines, start, end, red_color)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
	args = parse_args()
	input_path = Path(args.input).expanduser().resolve()
	output_path = Path(args.output).expanduser().resolve()

	if not input_path.exists() or not input_path.is_file():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	rewrite_latex_table(
		input_path=input_path,
		output_path=output_path,
		red_color=args.red_color,
	)

	print(f"Saved rewritten LaTeX table to: {output_path}")


if __name__ == "__main__":
	main()
