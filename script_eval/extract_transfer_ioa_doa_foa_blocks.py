from __future__ import annotations

import argparse
from pathlib import Path


METHOD_NAME_MAP = {
    "Ours (Insertion-Oriented)": "IOA",
    "Ours (Deletion-Oriented)": "DOA",
    "Ours (Final-Oriented)": "FOA",
}


def extract_transfer_name(line: str) -> str:
    text = line.strip()
    if text.startswith("% Transfer:"):
        return text.split(":", 1)[1].strip()
    return ""


def clean_dataset_row(line: str) -> tuple[str, bool]:
    raw = line.strip()
    has_break = raw.endswith("\\")
    if has_break:
        raw = raw[:-2].rstrip()

    if not raw.startswith("&"):
        return "", has_break

    return raw, has_break


def method_key_from_line(line: str) -> str | None:
    for key in METHOD_NAME_MAP:
        if key in line:
            return key
    return None


def parse_blocks(lines: list[str]) -> list[str]:
    out: list[str] = []
    transfer_name = ""
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("% Transfer:"):
            transfer_name = extract_transfer_name(line)
            out.append(line.rstrip("\n"))
            i += 1
            continue

        method_key = method_key_from_line(stripped)
        if method_key is None:
            i += 1
            continue

        method_short = METHOD_NAME_MAP[method_key]
        out.append(f"& {method_short} (Transfer: {transfer_name})")

        dataset_rows: list[str] = []
        j = i + 1
        while j < len(lines) and len(dataset_rows) < 3:
            candidate = lines[j].strip()
            if not candidate.startswith("&"):
                break
            if "(Transfer:" in candidate:
                break
            row_text, _ = clean_dataset_row(lines[j])
            if row_text:
                dataset_rows.append(row_text)
            j += 1

        if not dataset_rows:
            out.append("& --")
            out.append("& --")
            out.append("& -- \\")
        else:
            for idx, row in enumerate(dataset_rows):
                if idx == len(dataset_rows) - 1:
                    out.append(f"{row} \\")
                else:
                    out.append(row)

        out.append("")
        i = j

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract IOA/DOA/FOA transfer blocks from combined transfer LaTeX rows file."
    )
    parser.add_argument("-i", "--input", type=Path, help="Path to input txt file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path. Default: next to input with suffix _ioa_doa_foa.txt",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_path = args.output
    if output_path is None:
        output_path = args.input.with_name(args.input.stem + "_ioa_doa_foa.txt")

    lines = args.input.read_text(encoding="utf-8").splitlines()
    blocks = parse_blocks(lines)

    header = [
        "% Dataset order: ImageNet -> StandfordPet -> CUB",
        "% Rows include only IOA / DOA / FOA (X-Shift removed)",
        "",
    ]
    output_text = "\n".join(header + blocks).rstrip() + "\n"
    output_path.write_text(output_text, encoding="utf-8")

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
