import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


HIGHLIGHT_COLOR = "blue!60"


@dataclass
class DatasetLine:
	line_idx: int
	indent: str
	suffix: str
	cells: List[str]


@dataclass
class AttackRow:
	dataset_lines: List[DatasetLine]


def parse_dataset_line(line: str, line_idx: int, metrics_per_dataset: int) -> DatasetLine:
	raw = line.rstrip("\n")
	indent_match = re.match(r"^(\s*)", raw)
	indent = indent_match.group(1) if indent_match else ""

	suffix = ""
	suffix_match = re.search(r"(\\\\\s*)$", raw)
	core = raw
	if suffix_match:
		suffix = suffix_match.group(1)
		core = raw[: suffix_match.start()].rstrip()

	parts = core.split("&")
	if len(parts) < metrics_per_dataset + 1:
		raise ValueError(f"Unexpected dataset line format at line {line_idx + 1}: {line}")

	cells = [p.strip() for p in parts[1 : metrics_per_dataset + 1]]
	return DatasetLine(line_idx=line_idx, indent=indent, suffix=suffix, cells=cells)


def rebuild_dataset_line(ds_line: DatasetLine) -> str:
	content = f"{ds_line.indent}& " + " & ".join(ds_line.cells)
	if ds_line.suffix:
		content += f" {ds_line.suffix}"
	return content + "\n"


def strip_highlight(cell: str) -> str:
	text = cell
	text = re.sub(r"\\cellcolor\{[^}]+\}\s*", "", text)
	text = re.sub(
		r"\\textbf\s*\{\s*\\textcolor\{[^}]+\}\{\s*([^{}]+?)\s*\}\s*\}",
		r"\1",
		text,
	)
	return text.strip()


def format_highlight(value: str) -> str:
	return f"\\textbf{{\\textcolor{{{HIGHLIGHT_COLOR}}}{{{value}}}}}"


def to_float(cell: str) -> Optional[float]:
	text = strip_highlight(cell)
	if text == "--":
		return None
	if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text):
		return float(text)
	return None


def get_table_layout(lines: List[str]) -> Tuple[int, int, List[str]]:
	joined = "".join(lines)
	dataset_count = len(re.findall(r"\\multicolumn\{\d+\}\{c\}\{[^}]+\}", joined))
	if dataset_count == 0:
		raise ValueError("Cannot detect dataset groups from \\multicolumn headers")

	attack_header_idx = -1
	for idx, line in enumerate(lines):
		if "\\textbf{Attack}" in line:
			attack_header_idx = idx
			break
	if attack_header_idx < 0:
		raise ValueError("Cannot find attack header line")

	arrows: List[str] = []
	j = attack_header_idx + 1
	while j < len(lines):
		line = lines[j]
		if "\\midrule" in line:
			break
		if "\\uparrow" in line:
			arrows.append("up")
		elif "\\downarrow" in line:
			arrows.append("down")
		j += 1

	if not arrows:
		raise ValueError("Cannot detect metric arrows from header")
	if len(arrows) % dataset_count != 0:
		raise ValueError(
			f"Metric columns ({len(arrows)}) not divisible by dataset count ({dataset_count})"
		)

	metrics_per_dataset = len(arrows) // dataset_count
	return dataset_count, metrics_per_dataset, arrows


def highlight_best_in_block(
	rows: List[AttackRow], dataset_count: int, metrics_per_dataset: int, arrows: List[str]
) -> None:
	# Always clear old highlights so reruns follow current values only.
	for row in rows:
		for ds_line in row.dataset_lines:
			ds_line.cells = [strip_highlight(c) for c in ds_line.cells]

	for dataset_idx in range(dataset_count):
		for metric_idx in range(metrics_per_dataset):
			direction = arrows[dataset_idx * metrics_per_dataset + metric_idx]
			values = []
			for row_idx, row in enumerate(rows):
				value = to_float(row.dataset_lines[dataset_idx].cells[metric_idx])
				if value is not None:
					values.append((row_idx, value))

			if not values:
				continue

			if direction == "up":
				best_val = max(v for _, v in values)
			else:
				best_val = min(v for _, v in values)

			for row_idx, value in values:
				if value == best_val:
					current = rows[row_idx].dataset_lines[dataset_idx].cells[metric_idx]
					rows[row_idx].dataset_lines[dataset_idx].cells[metric_idx] = (
						format_highlight(current)
					)


def process_table(lines: List[str]) -> List[str]:
	dataset_count, metrics_per_dataset, arrows = get_table_layout(lines)

	i = 0
	while i < len(lines):
		if "\\multirow" not in lines[i]:
			i += 1
			continue

		multirow_match = re.search(r"\\multirow\[c\]\{(\d+)\}\{\*\}\{", lines[i])
		if not multirow_match:
			i += 1
			continue
		attack_count = int(multirow_match.group(1))

		rows: List[AttackRow] = []
		j = i + 1
		for _ in range(attack_count):
			while j < len(lines) and not lines[j].strip():
				j += 1

			if j >= len(lines):
				raise ValueError("Unexpected end of file while parsing attack labels")

			# Attack label line, e.g. '& X-Shift'
			j += 1

			if j + dataset_count - 1 >= len(lines):
				raise ValueError("Unexpected end of file while parsing attack rows")

			dataset_lines = []
			for ds_idx in range(dataset_count):
				line_idx = j + ds_idx
				dataset_lines.append(
					parse_dataset_line(lines[line_idx], line_idx, metrics_per_dataset)
				)
			rows.append(AttackRow(dataset_lines=dataset_lines))
			j = j + dataset_count

			# Optional row terminator line containing only '\\'.
			while j < len(lines) and re.fullmatch(r"\s*", lines[j]):
				j += 1
			if j < len(lines) and re.fullmatch(r"\s*\\\\\s*", lines[j]):
				j += 1

		highlight_best_in_block(rows, dataset_count, metrics_per_dataset, arrows)

		for row in rows:
			for ds_line in row.dataset_lines:
				lines[ds_line.line_idx] = rebuild_dataset_line(ds_line)

		i = j

	return lines


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Highlight best values per explainer/dataset/metric in a LaTeX table "
			"using \\textbf{\\textcolor{blue!90}{value}}."
		)
	)
	parser.add_argument(
		"--input",
		"-i",
		default="script_eval/result.txt",
		help="Path to input LaTeX table text file",
	)
	parser.add_argument(
		"--output",
		"-o",
		default=None,
		help="Path to output file. If omitted with --inplace disabled, writes to <input>_colored.txt",
	)
	parser.add_argument(
		"--inplace",
		action="store_true",
		help="Overwrite the input file in place",
	)

	args = parser.parse_args()
	input_path = Path(args.input)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	if args.inplace:
		output_path = input_path
	elif args.output:
		output_path = Path(args.output)
	else:
		output_path = input_path.with_name(f"{input_path.stem}_colored{input_path.suffix}")

	lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
	processed = process_table(lines)
	output_path.write_text("".join(processed), encoding="utf-8")

	print(f"Done. Wrote highlighted table to: {output_path}")


if __name__ == "__main__":
	main()
