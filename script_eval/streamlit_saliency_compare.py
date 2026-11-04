import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image


ATTACK_FOLDERS = [
    "Xshift_score",
    "ins_optimize_output",
    "del_optimize_output",
    "ins_del_optimized_output",
]

ATTACK_FOLDER_ALIASES = {
    "Xshift_score": ["Xshift_score"],
    "ins_optimize_output": ["ins_optimize_output", "ins_optimized_output"],
    "del_optimize_output": ["del_optimize_output", "del_optimized_output"],
    "ins_del_optimized_output": [
        "ins_del_optimized_output",
        "ins_del_optimize_output",
        "ins_del_optimization_output",
        "insdel_optimized_output",
        "insdel_optimize_output",
        "ins_del_output",
        "ins_del_score",
        "final_output",
    ],
}

ATTACK_LABELS = {
    "clean_causual_score": "Clean",
    "Xshift_score": "X-Shift",
    "ins_optimize_output": "Ours (Ins-Oriented)",
    "del_optimize_output": "Ours (Del-Oriented)",
    "ins_del_optimized_output": "Ours (IOA+DOA)",
}

FRAME_ATTACK_LABELS = {
    "clean_causual_score": "Clean",
    "Xshift_score": "X-Shift",
    "ins_optimize_output": "IOA",
    "del_optimize_output": "DOA",
    "ins_del_optimized_output": "IOA+DOA",
}

METHOD_COLORS = {
    "clean_causual_score": "#2ca02c",
    "Xshift_score": "#1f77b4",
}
DEFAULT_ATTACK_COLOR = "#d62728"


def method_color(method_key: str) -> str:
    return METHOD_COLORS.get(method_key, DEFAULT_ATTACK_COLOR)


def _resolve_explainer_dir(attack_root: Path, explainer: str) -> Optional[Path]:
    def _contains_saliency(root: Path) -> bool:
        for pat in ("*saliency.npy", "*saliency.png"):
            if any(root.rglob(pat)):
                return True
        return False

    def _pick_deepest_with_saliency(candidates: List[Path]) -> Optional[Path]:
        if not candidates:
            return None
        ranked: List[Tuple[int, int, Path]] = []
        for p in candidates:
            has_sal = 1 if _contains_saliency(p) else 0
            ranked.append((has_sal, len(p.parts), p))
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return ranked[0][2]

    direct = attack_root / explainer
    if direct.exists() and direct.is_dir():
        # Prefer del+ins subtree when present for joint attack outputs.
        for child_name in ["del+ins", "del_ins", "del-ins"]:
            child = direct / child_name
            if child.exists() and child.is_dir() and _contains_saliency(child):
                return child
        return direct

    norm_exp = normalize_key(explainer)
    children = [p for p in attack_root.iterdir() if p.is_dir()]

    for child in children:
        if normalize_key(child.name) == norm_exp:
            return child

    for child in children:
        if norm_exp in normalize_key(child.name):
            return child

    # Deep recursive search for nested layouts such as
    # ins_del_optimized_output/imagenet/eclip/eclip/del+ins
    recursive_exact = [
        p for p in attack_root.rglob("*") if p.is_dir() and normalize_key(p.name) == norm_exp
    ]
    picked_exact = _pick_deepest_with_saliency(recursive_exact)
    if picked_exact is not None:
        for child_name in ["del+ins", "del_ins", "del-ins"]:
            child = picked_exact / child_name
            if child.exists() and child.is_dir() and _contains_saliency(child):
                return child
        return picked_exact

    recursive_fuzzy = [
        p for p in attack_root.rglob("*") if p.is_dir() and norm_exp in normalize_key(p.name)
    ]
    picked_fuzzy = _pick_deepest_with_saliency(recursive_fuzzy)
    if picked_fuzzy is not None:
        for child_name in ["del+ins", "del_ins", "del-ins"]:
            child = picked_fuzzy / child_name
            if child.exists() and child.is_dir() and _contains_saliency(child):
                return child
        return picked_fuzzy

    # Final ins_del fallback: use any nested del+ins-like directory with saliency files.
    if "ins_del" in normalize_key(attack_root.name) or "final" in normalize_key(attack_root.name):
        delins_dirs = []
        for p in attack_root.rglob("*"):
            if not p.is_dir():
                continue
            pname = normalize_key(p.name)
            if ("del+ins" in p.name.lower()) or ("del" in pname and "ins" in pname):
                if _contains_saliency(p):
                    delins_dirs.append(p)
        picked_delins = _pick_deepest_with_saliency(delins_dirs)
        if picked_delins is not None:
            return picked_delins

    return None


def _candidate_attack_roots(base_root: Path, attack_key: str) -> List[Path]:
    roots: List[Path] = []

    aliases = ATTACK_FOLDER_ALIASES.get(attack_key, [attack_key])
    alias_norms = {normalize_key(a) for a in aliases}
    key_norm = normalize_key(attack_key)

    for folder_name in aliases:
        p = base_root / folder_name
        if p.exists() and p.is_dir():
            roots.append(p)

    if not base_root.exists() or not base_root.is_dir():
        return roots

    for child in sorted(base_root.iterdir()):
        if not child.is_dir():
            continue
        child_norm = normalize_key(child.name)
        if child_norm in alias_norms or child_norm == key_norm:
            roots.append(child)

    # Fallback for common ins_del naming drifts in ImageNet runs.
    if key_norm == "ins_del_optimized_output":
        for child in sorted(base_root.iterdir()):
            if not child.is_dir():
                continue
            name = normalize_key(child.name)
            if "ins" in name and "del" in name and ("opt" in name or "final" in name):
                roots.append(child)

    dedup: List[Path] = []
    seen = set()
    for p in roots:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def resolve_attack_root(base_root: Path, attack_key: str, explainer: str) -> Optional[Path]:
    for attack_root in _candidate_attack_roots(base_root, attack_key):
        resolved = _resolve_explainer_dir(attack_root, explainer)
        if resolved is not None:
            return resolved
    return None


def normalize_key(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def sample_id_from_saliency_path(path: Path) -> str:
    parent = path.parent
    if parent.name.lower() == "saliency":
        return parent.parent.name
    return parent.name


def score_attack_map(path: Path, explainer: str) -> Tuple[int, str]:
    name = path.name.lower()
    score = 0
    if explainer.lower() in name:
        score += 20
    if "del_" in name:
        score += 3
    if "ins_" in name:
        score += 2
    return -score, name


def find_adv_image_near(sample_dir: Path) -> Optional[Path]:
    for p in sorted(sample_dir.glob("adversarial_image*.png")):
        return p
    for p in sorted(sample_dir.glob("*adv*.png")):
        return p
    return None


def find_clean_image_near(sample_dir: Path) -> Optional[Path]:
    for p in sorted(sample_dir.glob("clean_image*.png")):
        return p
    for p in sorted(sample_dir.glob("original*.png")):
        return p
    return None


def sample_dir_from_map_path(path: Path) -> Path:
    sample_dir = path.parent
    if sample_dir.name.lower() == "saliency":
        sample_dir = sample_dir.parent
    return sample_dir


def find_summary_image_near(sample_dir: Path, mode: str) -> Optional[Path]:
    mode_key = normalize_key(mode)
    if mode_key not in {"del", "deletion", "ins", "insertion"}:
        return None

    if mode_key in {"del", "deletion"}:
        patterns = [
            "del_summary.png",
            "deletion_summary.png",
            "*del*summary*.png",
            "*deletion*summary*.png",
        ]
    else:
        patterns = [
            "ins_summary.png",
            "insertion_summary.png",
            "*ins*summary*.png",
            "*insertion*summary*.png",
        ]

    for pattern in patterns:
        for p in sorted(sample_dir.glob(pattern)):
            return p
    return None


def find_overlay_image_for_map(map_path: Path) -> Optional[Path]:
    sample_dir = sample_dir_from_map_path(map_path)
    direct = map_path.with_name(f"{map_path.stem}_overlay.png")
    if direct.exists():
        return direct

    stem_lower = map_path.stem.lower()
    ranked = sorted(
        sample_dir.glob("*overlay*.png"),
        key=lambda p: (
            "summary" in p.name.lower(),
            0 if stem_lower in p.stem.lower() else 1,
            p.name.lower(),
        ),
    )
    for p in ranked:
        return p
    return None


@st.cache_data(show_spinner=False)
def list_datasets(result_root: str) -> List[str]:
    root = Path(result_root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


@st.cache_data(show_spinner=False)
def list_models(result_root: str, dataset: str) -> List[str]:
    droot = Path(result_root) / dataset
    if not droot.exists() or not droot.is_dir():
        return []
    return sorted([p.name for p in droot.iterdir() if p.is_dir()])


@st.cache_data(show_spinner=False)
def list_explainers(result_root: str, dataset: str, model_name: str, clean_folder: str) -> List[str]:
    root = Path(result_root) / dataset / model_name / clean_folder
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def collect_clean_maps(clean_explainer_root: Path, explainer: str, map_ext: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}

    if map_ext == "npy":
        candidates = sorted(clean_explainer_root.rglob(f"*{explainer}*saliency.npy"))
        if not candidates:
            candidates = sorted(clean_explainer_root.rglob("*saliency.npy"))
        if not candidates:
            candidates = sorted(clean_explainer_root.rglob(f"*{explainer}*saliency.png"))
        if not candidates:
            candidates = sorted(clean_explainer_root.rglob("*saliency.png"))
    else:
        candidates = sorted(clean_explainer_root.rglob(f"*{explainer}*saliency.png"))
        if not candidates:
            candidates = sorted(clean_explainer_root.rglob("*saliency.png"))
        if not candidates:
            candidates = sorted(clean_explainer_root.rglob(f"*{explainer}*saliency.npy"))
        if not candidates:
            candidates = sorted(clean_explainer_root.rglob("*saliency.npy"))

    for p in candidates:
        sample_id = sample_id_from_saliency_path(p)
        if sample_id.startswith("eval_curve_summary"):
            continue
        mapping[sample_id] = p
    return mapping


def collect_attack_maps(attack_explainer_root: Path, explainer: str, map_ext: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    groups: Dict[str, List[Path]] = {}

    pattern = "*saliency.npy" if map_ext == "npy" else "*saliency.png"
    candidates = sorted(attack_explainer_root.rglob(pattern))
    if not candidates:
        alt_pattern = "*saliency.png" if pattern.endswith(".npy") else "*saliency.npy"
        candidates = sorted(attack_explainer_root.rglob(alt_pattern))

    for p in candidates:
        if p.name.lower().endswith("overlay.png"):
            continue
        sample_id = sample_id_from_saliency_path(p)
        if sample_id.startswith("eval_curve_summary"):
            continue
        groups.setdefault(sample_id, []).append(p)

    for sample_id, candidates in groups.items():
        if map_ext == "npy":
            chosen = sorted(candidates, key=lambda x: score_attack_map(x, explainer))[0]
        else:
            # Prefer raw saliency image over summary/overlay.
            ranked = sorted(
                candidates,
                key=lambda x: (
                    "summary" in x.name.lower(),
                    "overlay" in x.name.lower(),
                    score_attack_map(x, explainer),
                ),
            )
            chosen = ranked[0]
        mapping[sample_id] = chosen
    return mapping


@st.cache_data(show_spinner=False)
def collect_slot_maps_for_explainer(
    result_root: str,
    dataset: str,
    model_name: str,
    clean_folder: str,
    explainer: str,
    map_ext: str,
) -> Dict[str, Dict[str, Path]]:
    base_root = Path(result_root) / dataset / model_name
    clean_explainer_root = base_root / clean_folder / explainer

    clean_maps: Dict[str, Path] = {}
    if clean_explainer_root.exists():
        clean_maps = collect_clean_maps(clean_explainer_root, explainer, map_ext)

    attack_maps_all: Dict[str, Dict[str, Path]] = {}
    for attack in ATTACK_FOLDERS:
        attack_root = resolve_attack_root(base_root, attack, explainer)
        if attack_root is not None:
            attack_maps_all[attack] = collect_attack_maps(attack_root, explainer, map_ext)
        else:
            attack_maps_all[attack] = {}

    slot_maps: Dict[str, Dict[str, Path]] = {"clean_causual_score": clean_maps}
    for attack in ATTACK_FOLDERS:
        slot_maps[attack] = attack_maps_all.get(attack, {})
    return slot_maps


def load_overlay_or_map_image(map_path: Path, map_ext: str, frame_source: str) -> Optional[np.ndarray]:
    if frame_source == "overlay":
        overlay_img = find_overlay_image_for_map(map_path)
        if overlay_img and overlay_img.exists():
            return np.asarray(Image.open(overlay_img).convert("RGB"), dtype=np.uint8)

    try:
        arr = load_map(map_path, map_ext)
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None


def load_rgb_image(path: Path) -> Optional[np.ndarray]:
    try:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arrf = np.asarray(arr, dtype=np.float32)
        amin = float(np.min(arrf))
        amax = float(np.max(arrf))
        if np.isclose(amin, amax):
            arrn = np.zeros_like(arrf, dtype=np.float32)
        else:
            arrn = (arrf - amin) / (amax - amin)
        rgb = (plt.get_cmap("jet")(arrn)[..., :3] * 255.0).astype(np.uint8)
        return rgb

    if arr.ndim == 3 and arr.shape[2] == 3:
        if arr.dtype == np.uint8:
            return arr
        arrf = np.asarray(arr, dtype=np.float32)
        if arrf.max() <= 1.0:
            arrf = arrf * 255.0
        return np.clip(arrf, 0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[2] == 4:
        rgba = arr[..., :3]
        if rgba.dtype == np.uint8:
            return rgba
        arrf = np.asarray(rgba, dtype=np.float32)
        if arrf.max() <= 1.0:
            arrf = arrf * 255.0
        return np.clip(arrf, 0, 255).astype(np.uint8)

    raise ValueError("Unsupported image array shape")


def resize_rgb_uint8(img: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray(img)
    pil = pil.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def stitch_rgb_grid(grid_rows: List[List[Optional[np.ndarray]]]) -> Optional[np.ndarray]:
    all_cells: List[np.ndarray] = []
    for row in grid_rows:
        for cell in row:
            if cell is not None:
                all_cells.append(to_rgb_uint8(cell))

    if not all_cells:
        return None

    target_h = min(img.shape[0] for img in all_cells)
    target_w = min(img.shape[1] for img in all_cells)
    blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    stitched_rows: List[np.ndarray] = []
    for row in grid_rows:
        row_tiles: List[np.ndarray] = []
        for cell in row:
            if cell is None:
                row_tiles.append(blank)
            else:
                row_tiles.append(resize_rgb_uint8(to_rgb_uint8(cell), target_w, target_h))
        stitched_rows.append(np.concatenate(row_tiles, axis=1))

    return np.concatenate(stitched_rows, axis=0)


def load_map(path: Path, map_ext: str) -> np.ndarray:
    # Infer by suffix so map collection can fallback across extensions.
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze()
        return arr

    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def compute_auc_from_curve(curve: np.ndarray) -> float:
    if curve.size == 0:
        return float("nan")
    x = np.linspace(0.0, 1.0, curve.size, dtype=np.float64)
    return float(np.trapezoid(curve.astype(np.float64), x))


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metric_curve_and_auc(data: dict, metric: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
    rerun = data.get("rerun")
    if isinstance(rerun, dict):
        metric_obj = rerun.get(metric)
        if isinstance(metric_obj, dict):
            curve = metric_obj.get("curve")
            auc = metric_obj.get("auc")
            curve_arr: Optional[np.ndarray] = None
            if isinstance(curve, list) and len(curve) > 0:
                curve_arr = np.asarray(curve, dtype=np.float32)
            auc_val = _safe_float(auc)
            if auc_val is None and curve_arr is not None:
                auc_val = compute_auc_from_curve(curve_arr)
            if curve_arr is not None:
                return curve_arr, auc_val

    key_map = {
        "del": ("deletion_curve", "deletion_auc"),
        "ins": ("insertion_curve", "insertion_auc"),
    }
    curve_key, auc_key = key_map[metric]
    curve = data.get(curve_key)
    auc = data.get(auc_key)
    curve_arr = None
    if isinstance(curve, list) and len(curve) > 0:
        curve_arr = np.asarray(curve, dtype=np.float32)
    auc_val = _safe_float(auc)
    if auc_val is None and curve_arr is not None:
        auc_val = compute_auc_from_curve(curve_arr)
    return curve_arr, auc_val


def normalize_curve_by_max(curve: np.ndarray) -> Optional[np.ndarray]:
    if curve.size == 0:
        return None
    max_val = float(np.max(curve))
    if not np.isfinite(max_val) or max_val <= 0.0:
        return None
    return curve.astype(np.float64) / max_val


def find_normalized_curve_json_near(sample_dir: Path, metric: str) -> Optional[Path]:
    preferred = [sample_dir / f"{metric}_information_normalized.json"]
    for p in preferred:
        if p.exists():
            return p

    patterns = [f"*{metric}*normalized*.json", "*normalized*.json"]
    for pattern in patterns:
        for p in sorted(sample_dir.glob(pattern)):
            return p
    return None


def find_curve_json_near(sample_dir: Path, metric: str) -> Optional[Path]:
    preferred = [sample_dir / "curve_information.json"]
    if metric == "del":
        preferred.extend(
            [
                sample_dir / "deletion_information.json",
                sample_dir / "del_information.json",
            ]
        )
    else:
        preferred.extend(
            [
                sample_dir / "insertion_information.json",
                sample_dir / "ins_information.json",
            ]
        )

    for p in preferred:
        if p.exists():
            return p

    if metric == "del":
        patterns = ["*deletion*information*.json", "*del*information*.json", "*information*.json"]
    else:
        patterns = ["*insertion*information*.json", "*ins*information*.json", "*information*.json"]

    for pattern in patterns:
        for p in sorted(sample_dir.glob(pattern)):
            return p
    return None


@st.cache_data(show_spinner=False)
def load_curve_for_sample(sample_dir_str: str, metric: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
    sample_dir = Path(sample_dir_str)

    normalized_json = find_normalized_curve_json_near(sample_dir, metric)
    if normalized_json is not None:
        try:
            with normalized_json.open("r", encoding="utf-8") as f:
                normalized_data = json.load(f)
            normalized_curve = normalized_data.get("normalized_curve")
            if isinstance(normalized_curve, list) and len(normalized_curve) > 0:
                normalized_arr = np.asarray(normalized_curve, dtype=np.float32)
                fair_auc = _safe_float(normalized_data.get("fair_auc"))
                if fair_auc is None:
                    fair_auc = compute_auc_from_curve(normalized_arr)
                return normalized_arr, fair_auc
        except Exception:
            pass

    curve_json = find_curve_json_near(sample_dir, metric)
    if curve_json is None:
        return None, None

    try:
        with curve_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None

    raw_curve, raw_auc = extract_metric_curve_and_auc(data, metric)
    if raw_curve is None:
        return None, None

    # If normalized artifact does not exist, normalize on-the-fly (same idea as fair-AUC script).
    normalized_curve = normalize_curve_by_max(raw_curve)
    if normalized_curve is not None:
        return normalized_curve.astype(np.float32), compute_auc_from_curve(normalized_curve)

    return raw_curve, raw_auc


def pretty_sample_options(sample_ids: List[str]) -> List[str]:
    def _sort_key(s: str) -> Tuple[int, str, int]:
        m = re.match(r"^([a-zA-Z]+)(\d+)$", s)
        if m:
            return (0, m.group(1), int(m.group(2)))
        return (1, s, 0)

    return sorted(sample_ids, key=_sort_key)


def build_dual_improvement_rows(
    sample_ids: List[str],
    clean_maps: Dict[str, Path],
    attack_maps: Dict[str, Path],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for sample_id in sample_ids:
        clean_map_path = clean_maps.get(sample_id)
        adv_map_path = attack_maps.get(sample_id)
        if clean_map_path is None or adv_map_path is None:
            continue

        clean_sample_dir = sample_dir_from_map_path(clean_map_path)
        adv_sample_dir = sample_dir_from_map_path(adv_map_path)

        clean_del_curve, clean_del_auc = load_curve_for_sample(str(clean_sample_dir), "del")
        clean_ins_curve, clean_ins_auc = load_curve_for_sample(str(clean_sample_dir), "ins")
        adv_del_curve, adv_del_auc = load_curve_for_sample(str(adv_sample_dir), "del")
        adv_ins_curve, adv_ins_auc = load_curve_for_sample(str(adv_sample_dir), "ins")

        if clean_del_auc is None and clean_del_curve is not None:
            clean_del_auc = compute_auc_from_curve(clean_del_curve)
        if clean_ins_auc is None and clean_ins_curve is not None:
            clean_ins_auc = compute_auc_from_curve(clean_ins_curve)
        if adv_del_auc is None and adv_del_curve is not None:
            adv_del_auc = compute_auc_from_curve(adv_del_curve)
        if adv_ins_auc is None and adv_ins_curve is not None:
            adv_ins_auc = compute_auc_from_curve(adv_ins_curve)

        if any(x is None for x in [clean_del_auc, clean_ins_auc, adv_del_auc, adv_ins_auc]):
            continue

        delta_del = float(adv_del_auc - clean_del_auc)
        delta_ins = float(adv_ins_auc - clean_ins_auc)
        if delta_del <= 0.0 or delta_ins <= 0.0:
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "clean_del_auc": float(clean_del_auc),
                "adv_del_auc": float(adv_del_auc),
                "delta_del": delta_del,
                "clean_ins_auc": float(clean_ins_auc),
                "adv_ins_auc": float(adv_ins_auc),
                "delta_ins": delta_ins,
                "dual_gain": float(delta_del + delta_ins),
                "min_gain": float(min(delta_del, delta_ins)),
            }
        )

    rows.sort(
        key=lambda x: (
            x["min_gain"],
            x["dual_gain"],
            x["delta_ins"],
            x["delta_del"],
        ),
        reverse=True,
    )
    return rows


def _sample_attack_auc_row(
    sample_id: str,
    attack_maps: Dict[str, Path],
) -> Optional[Dict[str, float]]:
    adv_map_path = attack_maps.get(sample_id)
    if adv_map_path is None:
        return None

    adv_sample_dir = sample_dir_from_map_path(adv_map_path)

    adv_del_curve, adv_del_auc = load_curve_for_sample(str(adv_sample_dir), "del")
    adv_ins_curve, adv_ins_auc = load_curve_for_sample(str(adv_sample_dir), "ins")

    if adv_del_auc is None and adv_del_curve is not None:
        adv_del_auc = compute_auc_from_curve(adv_del_curve)
    if adv_ins_auc is None and adv_ins_curve is not None:
        adv_ins_auc = compute_auc_from_curve(adv_ins_curve)

    if any(x is None for x in [adv_del_auc, adv_ins_auc]):
        return None

    return {
        "del_auc": float(adv_del_auc),
        "ins_auc": float(adv_ins_auc),
        "balance": float(adv_del_auc - adv_ins_auc),
        "badness": float(adv_ins_auc - adv_del_auc),
    }


def _sample_gain_row(
    sample_id: str,
    clean_maps: Dict[str, Path],
    attack_maps: Dict[str, Path],
) -> Optional[Dict[str, float]]:
    clean_map_path = clean_maps.get(sample_id)
    adv_map_path = attack_maps.get(sample_id)
    if clean_map_path is None or adv_map_path is None:
        return None

    clean_sample_dir = sample_dir_from_map_path(clean_map_path)
    adv_sample_dir = sample_dir_from_map_path(adv_map_path)

    clean_del_curve, clean_del_auc = load_curve_for_sample(str(clean_sample_dir), "del")
    clean_ins_curve, clean_ins_auc = load_curve_for_sample(str(clean_sample_dir), "ins")
    adv_del_curve, adv_del_auc = load_curve_for_sample(str(adv_sample_dir), "del")
    adv_ins_curve, adv_ins_auc = load_curve_for_sample(str(adv_sample_dir), "ins")

    if clean_del_auc is None and clean_del_curve is not None:
        clean_del_auc = compute_auc_from_curve(clean_del_curve)
    if clean_ins_auc is None and clean_ins_curve is not None:
        clean_ins_auc = compute_auc_from_curve(clean_ins_curve)
    if adv_del_auc is None and adv_del_curve is not None:
        adv_del_auc = compute_auc_from_curve(adv_del_curve)
    if adv_ins_auc is None and adv_ins_curve is not None:
        adv_ins_auc = compute_auc_from_curve(adv_ins_curve)

    if any(x is None for x in [clean_del_auc, clean_ins_auc, adv_del_auc, adv_ins_auc]):
        return None

    delta_del = float(adv_del_auc - clean_del_auc)
    delta_ins = float(adv_ins_auc - clean_ins_auc)

    return {
        "clean_del_auc": float(clean_del_auc),
        "clean_ins_auc": float(clean_ins_auc),
        "adv_del_auc": float(adv_del_auc),
        "adv_ins_auc": float(adv_ins_auc),
        "delta_del": delta_del,
        "delta_ins": delta_ins,
        "dual_gain": float(delta_del + delta_ins),
        "min_gain": float(min(delta_del, delta_ins)),
    }


def build_ours_vs_xshift_rows(
    sample_ids: List[str],
    attack_maps_all: Dict[str, Dict[str, Path]],
    ours_attacks: List[str],
    xshift_key: str = "Xshift_score",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    valid_ours = [a for a in ours_attacks if a != xshift_key]
    xshift_maps = attack_maps_all.get(xshift_key, {})

    for sample_id in sample_ids:
        xshift_auc = _sample_attack_auc_row(sample_id, xshift_maps)
        if xshift_auc is None:
            continue

        ours_candidates: List[Tuple[str, Dict[str, float]]] = []
        for attack_key in valid_ours:
            attack_auc = _sample_attack_auc_row(sample_id, attack_maps_all.get(attack_key, {}))
            if attack_auc is None:
                continue
            ours_candidates.append((attack_key, attack_auc))

        if not ours_candidates:
            continue

        ours_best_key, ours_best = max(ours_candidates, key=lambda item: item[1]["balance"])

        rows.append(
            {
                "sample_id": sample_id,
                "ours_best_attack": ours_best_key,
                "ours_del_auc": float(ours_best["del_auc"]),
                "ours_ins_auc": float(ours_best["ins_auc"]),
                "ours_imd": float(ours_best["ins_auc"] - ours_best["del_auc"]),
                "ours_balance": float(ours_best["balance"]),
                "xshift_del_auc": float(xshift_auc["del_auc"]),
                "xshift_ins_auc": float(xshift_auc["ins_auc"]),
                "xshift_imd": float(xshift_auc["ins_auc"] - xshift_auc["del_auc"]),
                "xshift_badness": float(xshift_auc["badness"]),
                "contrast_margin": float(ours_best["balance"] + xshift_auc["badness"]),
            }
        )

    rows.sort(
        key=lambda x: (
            x["contrast_margin"],
            x["ours_balance"],
            x["xshift_badness"],
        ),
        reverse=True,
    )
    return rows


def main() -> None:
    st.set_page_config(page_title="Saliency Qualitative Comparison", layout="wide")
    st.title("Saliency Qualitative Comparison")

    with st.sidebar:
        st.header("Settings")
        result_root = st.text_input("Result root", value=r"E:\Khoatn_ICLR_MMM_Neurips\iclr_neurips_result\result")
        clean_folder = st.text_input("Clean folder", value="clean_causual_score")
        map_ext = st.selectbox("Map source", options=["png", "npy"], index=0)
        only_complete = st.checkbox("Only samples with all 4 maps", value=True)
        show_figure_panel = st.checkbox("Show figure-style compare panel", value=True)
        default_selected_approaches = [
            a
            for a in [
                "ins_optimize_output",
                "del_optimize_output",
                "ins_del_optimized_output",
                "Xshift_score",
            ]
            if a in ATTACK_FOLDERS
        ]
        selected_approaches = st.multiselect(
            "Compare approaches (figure panel)",
            options=ATTACK_FOLDERS,
            default=default_selected_approaches,
            format_func=lambda x: ATTACK_LABELS.get(x, x),
        )
        show_images = st.checkbox("Show clean/adv images", value=True)
        show_overlays = st.checkbox("Show saliency overlays", value=True)
        show_summaries = st.checkbox("Show deletion/insertion summaries", value=True)
        show_composite_frame = st.checkbox("Show merged comparison frame", value=True)

    datasets = list_datasets(result_root)
    if not datasets:
        st.error("No dataset folders found. Check result root.")
        return

    dataset = st.selectbox("Dataset", options=datasets)

    models = list_models(result_root, dataset)
    if not models:
        st.error("No model folders found for selected dataset.")
        return

    default_model_idx = models.index("CLIP_ViTB16") if "CLIP_ViTB16" in models else 0
    model_name = st.selectbox("Model", options=models, index=default_model_idx)

    explainers = list_explainers(result_root, dataset, model_name, clean_folder)
    if not explainers:
        st.error("No explainer folders found under clean folder.")
        return

    default_exp_idx = explainers.index("eclip") if "eclip" in explainers else 0
    explainer = st.selectbox("Explainer", options=explainers, index=default_exp_idx)

    default_frame_rows = [
        e
        for e in ["grad_eclip", "game", "gradcam", "maskclip", "rise"]
        if e in explainers
    ]
    if not default_frame_rows:
        default_frame_rows = [explainer]

    frame_explainers = st.multiselect(
        "Frame rows (explain methods)",
        options=explainers,
        default=default_frame_rows,
    )

    frame_source = st.selectbox(
        "Frame image source",
        options=["overlay", "saliency_map"],
        index=0,
        help="overlay: use *_overlay.png when available; saliency_map: use raw saliency maps.",
    )

    base_root = Path(result_root) / dataset / model_name
    clean_explainer_root = base_root / clean_folder / explainer

    if not clean_explainer_root.exists():
        st.error(f"Missing clean explainer path: {clean_explainer_root}")
        return

    clean_maps = collect_clean_maps(clean_explainer_root, explainer, map_ext)

    attack_maps_all: Dict[str, Dict[str, Path]] = {}
    for attack in ATTACK_FOLDERS:
        attack_root = resolve_attack_root(base_root, attack, explainer)
        if attack_root is not None:
            attack_maps_all[attack] = collect_attack_maps(attack_root, explainer, map_ext)
        else:
            attack_maps_all[attack] = {}

    clean_ids = set(clean_maps.keys())
    attack_counts = {k: len(v) for k, v in attack_maps_all.items()}

    # Only intersect with attack branches that actually contain samples.
    active_attack_sets = [set(v.keys()) for v in attack_maps_all.values() if len(v) > 0]

    if only_complete:
        if active_attack_sets:
            sample_ids = sorted(set.intersection(clean_ids, *active_attack_sets))
        else:
            sample_ids = sorted(clean_ids)
    else:
        sample_ids = sorted(clean_ids)

    sample_ids = pretty_sample_options(sample_ids)

    if active_attack_sets:
        complete_count = len(set.intersection(clean_ids, *active_attack_sets))
    else:
        complete_count = len(clean_ids)

    st.caption(
        f"Found clean samples: {len(clean_maps)} | "
        f"complete samples (active attacks): {complete_count}"
    )

    missing_attack_names = [ATTACK_LABELS.get(k, k) for k, c in attack_counts.items() if c == 0]
    if missing_attack_names:
        st.warning(
            "Some attack branches have 0 samples for current selection: "
            + ", ".join(missing_attack_names)
            + ". They are ignored in 'Only complete' filtering."
        )

    with st.expander("Coverage details", expanded=False):
        coverage_rows = [
            {"Branch": ATTACK_LABELS.get("clean_causual_score", "clean_causual_score"), "Samples": len(clean_maps)}
        ]
        for attack in ATTACK_FOLDERS:
            coverage_rows.append({"Branch": ATTACK_LABELS.get(attack, attack), "Samples": attack_counts.get(attack, 0)})
        st.dataframe(coverage_rows, use_container_width=True, hide_index=True)

    if not sample_ids:
        st.warning("No sample IDs found for current filters.")
        return

    st.markdown("---")
    st.subheader("Recommended Samples (Ours Good, X-Shift Weak)")

    default_ours_attacks = [
        a
        for a in [
            "ins_optimize_output",
            "del_optimize_output",
            "ins_del_optimized_output",
        ]
        if a in ATTACK_FOLDERS
    ]
    ours_attacks = st.multiselect(
        "Ours approaches",
        options=[a for a in ATTACK_FOLDERS if a != "Xshift_score"],
        default=default_ours_attacks,
        format_func=lambda x: ATTACK_LABELS.get(x, x),
    )
    top_k = st.slider("Top-K recommendations", min_value=5, max_value=100, value=20, step=5)

    contrast_rows = build_ours_vs_xshift_rows(
        sample_ids=sample_ids,
        attack_maps_all=attack_maps_all,
        ours_attacks=ours_attacks,
        xshift_key="Xshift_score",
    )

    filtered_rows = [
        row
        for row in contrast_rows
        if float(row["ours_imd"]) < float(row["xshift_imd"])
    ]

    sample_id: str
    if not ours_attacks:
        st.info("Pick at least one 'Ours approach' to build recommendations.")
        sample_id = st.selectbox("Sample", options=sample_ids, key="sample_fallback_none_ours")
    elif not filtered_rows:
        st.info("No sample matches current thresholds. Showing all samples.")
        sample_id = st.selectbox("Sample", options=sample_ids, key="sample_fallback_no_rows")
    else:
        table_rows: List[Dict[str, object]] = []
        for row in filtered_rows[:top_k]:
            table_rows.append(
                {
                    "Sample": row["sample_id"],
                    "Best Ours": ATTACK_LABELS.get(str(row["ours_best_attack"]), str(row["ours_best_attack"])),
                    "Ours Del AUC": f"{row['ours_del_auc']:.4f}",
                    "Ours Ins AUC": f"{row['ours_ins_auc']:.4f}",
                    "Ours IMD (Ins-Del)": f"{row['ours_imd']:+.4f}",
                    "X-Shift Del AUC": f"{row['xshift_del_auc']:.4f}",
                    "X-Shift Ins AUC": f"{row['xshift_ins_auc']:.4f}",
                    "X-Shift IMD (Ins-Del)": f"{row['xshift_imd']:+.4f}",
                    "IMD Margin (X-Shift - Ours)": f"{row['contrast_margin']:+.4f}",
                }
            )
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.caption(
            "Auto filter: Ours IMD < X-Shift IMD. Ranking: IMD Margin (X-Shift - Ours) desc."
        )
        recommended_ids = [row["sample_id"] for row in filtered_rows[:top_k]]
        sample_id = st.selectbox(
            "Sample (recommended first)",
            options=recommended_ids,
            index=0,
            key="sample_recommended_primary",
        )
        if st.checkbox("Show all available samples instead", value=False, key="show_all_samples_toggle"):
            sample_id = st.selectbox("Sample", options=sample_ids, key="sample_full_list")

    map_paths: Dict[str, Optional[Path]] = {"clean_causual_score": clean_maps.get(sample_id)}
    for attack in ATTACK_FOLDERS:
        map_paths[attack] = attack_maps_all.get(attack, {}).get(sample_id)

    if show_figure_panel:
        st.markdown("---")
        st.subheader("Figure-Style Compare")

        clean_map_path = map_paths.get("clean_causual_score")
        if clean_map_path is None:
            st.info("Missing clean sample for this ID.")
        elif not selected_approaches:
            st.info("Pick at least one approach in sidebar to render the compare panel.")
        else:
            panel_keys = ["clean_causual_score"] + selected_approaches

            left_col, right_col = st.columns([1.15, 1.0])

            clean_sample_dir = sample_dir_from_map_path(clean_map_path)
            clean_img_path = find_clean_image_near(clean_sample_dir)
            clean_overlay_path = find_overlay_image_for_map(clean_map_path)

            with left_col:
                st.markdown("#### A. Image and Overlay")
                header_cols = st.columns(len(panel_keys))
                image_row_cells: List[Optional[np.ndarray]] = []
                overlay_row_cells: List[Optional[np.ndarray]] = []

                for idx, key in enumerate(panel_keys):
                    header_cols[idx].markdown(f"**{ATTACK_LABELS.get(key, key)}**")

                    if key == "clean_causual_score":
                        clean_img_arr = (
                            load_rgb_image(clean_img_path)
                            if clean_img_path and clean_img_path.exists()
                            else None
                        )
                        clean_overlay_arr = (
                            load_rgb_image(clean_overlay_path)
                            if clean_overlay_path and clean_overlay_path.exists()
                            else None
                        )
                        image_row_cells.append(clean_img_arr)
                        overlay_row_cells.append(clean_overlay_arr)
                        continue

                    attack_map_path = map_paths.get(key)
                    if attack_map_path is None:
                        image_row_cells.append(None)
                        overlay_row_cells.append(None)
                        continue

                    attack_sample_dir = sample_dir_from_map_path(attack_map_path)
                    adv_img_path = find_adv_image_near(attack_sample_dir)
                    adv_overlay_path = find_overlay_image_for_map(attack_map_path)

                    adv_img_arr = load_rgb_image(adv_img_path) if adv_img_path and adv_img_path.exists() else None
                    adv_overlay_arr = (
                        load_rgb_image(adv_overlay_path)
                        if adv_overlay_path and adv_overlay_path.exists()
                        else None
                    )
                    image_row_cells.append(adv_img_arr)
                    overlay_row_cells.append(adv_overlay_arr)

                merged_left_frame = stitch_rgb_grid([image_row_cells, overlay_row_cells])
                if merged_left_frame is None:
                    st.info("No image/overlay found for selected sample.")
                else:
                    st.image(merged_left_frame, use_container_width=True)

                st.caption("Top row: image | Bottom row: saliency overlay")

            with right_col:
                st.markdown("#### B. Deletion / Insertion Curves")
                curve_fig, axes = plt.subplots(
                    2,
                    len(selected_approaches),
                    figsize=(5.2 * len(selected_approaches), 7.0),
                    squeeze=False,
                )

                clean_del_curve, clean_del_auc = load_curve_for_sample(str(clean_sample_dir), "del")
                clean_ins_curve, clean_ins_auc = load_curve_for_sample(str(clean_sample_dir), "ins")

                metric_rows = [
                    ("del", "Deletion", clean_del_curve, clean_del_auc),
                    ("ins", "Insertion", clean_ins_curve, clean_ins_auc),
                ]

                for col_idx, attack_key in enumerate(selected_approaches):
                    attack_map_path = map_paths.get(attack_key)
                    attack_sample_dir = sample_dir_from_map_path(attack_map_path) if attack_map_path else None
                    clean_color = method_color("clean_causual_score")
                    adv_color = method_color(attack_key)

                    for row_idx, (metric_key, metric_title, clean_curve, clean_auc) in enumerate(metric_rows):
                        ax = axes[row_idx, col_idx]
                        ax.set_xlim(0.0, 1.0)
                        ax.set_ylim(0.0, 1.02)
                        ax.grid(True, alpha=0.25)

                        if clean_curve is not None and clean_curve.size > 0:
                            x_clean = np.linspace(0.0, 1.0, clean_curve.size, dtype=np.float64)
                            clean_curve_f = clean_curve.astype(np.float64)
                            ax.plot(x_clean, clean_curve_f, color=clean_color, linewidth=1.8, label="Clean")
                            ax.fill_between(x_clean, 0.0, clean_curve_f, color=clean_color, alpha=0.20)

                        adv_curve: Optional[np.ndarray] = None
                        adv_auc: Optional[float] = None
                        if attack_sample_dir is not None:
                            adv_curve, adv_auc = load_curve_for_sample(str(attack_sample_dir), metric_key)

                        if adv_curve is not None and adv_curve.size > 0:
                            x_adv = np.linspace(0.0, 1.0, adv_curve.size, dtype=np.float64)
                            adv_curve_f = adv_curve.astype(np.float64)
                            ax.plot(x_adv, adv_curve_f, color=adv_color, linewidth=1.5, label="Adv")
                            ax.fill_between(x_adv, 0.0, adv_curve_f, color=adv_color, alpha=0.20)
                        elif attack_sample_dir is None:
                            ax.text(0.5, 0.5, "No attack sample", ha="center", va="center", transform=ax.transAxes)

                        clean_auc_text = "N/A" if clean_auc is None else f"{clean_auc:.4f}"
                        adv_auc_text = "N/A" if adv_auc is None else f"{adv_auc:.4f}"
                        ax.text(
                            0.98,
                            0.95,
                            f"Clean-AUC={clean_auc_text}",
                            ha="right",
                            va="top",
                            transform=ax.transAxes,
                            fontsize=13,
                            fontweight="bold",
                            color=clean_color,
                            bbox={
                                "boxstyle": "round,pad=0.25",
                                "facecolor": "#eaf7ea",
                                "alpha": 0.95,
                                "edgecolor": "none",
                            },
                        )
                        ax.text(
                            0.98,
                            0.82,
                            f"Adv.-AUC={adv_auc_text}",
                            ha="right",
                            va="top",
                            transform=ax.transAxes,
                            fontsize=13,
                            fontweight="bold",
                            color=adv_color,
                            bbox={
                                "boxstyle": "round,pad=0.25",
                                "facecolor": "#e8f4ff" if attack_key == "Xshift_score" else "#ffe9eb",
                                "alpha": 0.95,
                                "edgecolor": "none",
                            },
                        )

                        ax.set_xlabel("")
                        ax.set_ylabel("")

                curve_fig.tight_layout()
                st.pyplot(curve_fig)
                plt.close(curve_fig)

    slot_keys = ["clean_causual_score"] + ATTACK_FOLDERS
    cols = st.columns(len(slot_keys))

    maps_loaded: Dict[str, Optional[np.ndarray]] = {}
    for key in slot_keys:
        p = map_paths.get(key)
        if p is None:
            maps_loaded[key] = None
            continue
        try:
            maps_loaded[key] = load_map(p, map_ext)
        except Exception:
            maps_loaded[key] = None

    valid_arrays = [a for a in maps_loaded.values() if a is not None and a.size > 0]
    if valid_arrays:
        global_min = float(min(np.min(a) for a in valid_arrays))
        global_max = float(max(np.max(a) for a in valid_arrays))
        if np.isclose(global_min, global_max):
            global_min, global_max = 0.0, 1.0
    else:
        global_min, global_max = 0.0, 1.0

    for col, key in zip(cols, slot_keys):
        label = ATTACK_LABELS.get(key, key)
        col.markdown(f"**{label}**")
        p = map_paths.get(key)
        arr = maps_loaded.get(key)

        if p is None or arr is None:
            col.warning("Not found")
            continue

        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(arr, cmap="jet", vmin=global_min, vmax=global_max)
        ax.axis("off")
        fig.tight_layout(pad=0.0)
        col.pyplot(fig)
        plt.close(fig)

        col.caption(str(p))

    if show_images:
        st.markdown("---")
        st.subheader("Reference Images")

        clean_sample_dir = sample_dir_from_map_path(clean_maps[sample_id])

        clean_img = find_clean_image_near(clean_sample_dir)
        img_cols = st.columns(1 + len(ATTACK_FOLDERS))

        img_cols[0].markdown("**Clean Image**")
        if clean_img and clean_img.exists():
            img_cols[0].image(str(clean_img), use_container_width=True)
        else:
            img_cols[0].info("No clean image found")

        for idx, attack in enumerate(ATTACK_FOLDERS, start=1):
            img_cols[idx].markdown(f"**{ATTACK_LABELS.get(attack, attack)} Adv Image**")
            ap = map_paths.get(attack)
            if ap is None:
                img_cols[idx].info("No attack map")
                continue
            sample_dir = sample_dir_from_map_path(ap)
            adv_img = find_adv_image_near(sample_dir)
            if adv_img and adv_img.exists():
                img_cols[idx].image(str(adv_img), use_container_width=True)
            else:
                img_cols[idx].info("No adv image found")

    if show_overlays:
        st.markdown("---")
        st.subheader("Saliency Overlays")
        overlay_cols = st.columns(len(slot_keys))

        for idx, key in enumerate(slot_keys):
            overlay_cols[idx].markdown(f"**{ATTACK_LABELS.get(key, key)}**")
            map_path = map_paths.get(key)
            if map_path is None:
                overlay_cols[idx].info("No attack map")
                continue

            overlay_img = find_overlay_image_for_map(map_path)
            if overlay_img and overlay_img.exists():
                overlay_cols[idx].image(str(overlay_img), use_container_width=True)
                overlay_cols[idx].caption(str(overlay_img))
            else:
                overlay_cols[idx].info("No overlay image found")

    if show_summaries:
        st.markdown("---")
        st.subheader("Raw Deletion / Insertion Summaries")

        for mode_key, mode_title in [("del", "Deletion"), ("ins", "Insertion")]:
            st.markdown(f"**{mode_title} Summary**")
            summary_cols = st.columns(len(slot_keys))

            for idx, key in enumerate(slot_keys):
                summary_cols[idx].markdown(f"**{ATTACK_LABELS.get(key, key)}**")

                map_path = map_paths.get(key)
                if map_path is None:
                    summary_cols[idx].info("No attack map")
                    continue

                sample_dir = sample_dir_from_map_path(map_path)
                summary_img = find_summary_image_near(sample_dir, mode_key)
                if summary_img and summary_img.exists():
                    summary_cols[idx].image(str(summary_img), use_container_width=True)
                    summary_cols[idx].caption(str(summary_img))
                else:
                    summary_cols[idx].info("No summary image found")

    if show_composite_frame:
        st.markdown("---")
        st.subheader("Merged Frame")

        if not frame_explainers:
            st.info("Pick at least one explainer in 'Frame rows (explain methods)'.")
            return

        frame_order = [
            ("clean_causual_score", "image"),
            ("clean_causual_score", "overlay"),
            ("ins_optimize_output", "image"),
            ("ins_optimize_output", "overlay"),
            ("del_optimize_output", "image"),
            ("del_optimize_output", "overlay"),
            ("ins_del_optimized_output", "image"),
            ("ins_del_optimized_output", "overlay"),
            ("Xshift_score", "image"),
            ("Xshift_score", "overlay"),
        ]

        n_cols = len(frame_order)
        grid_cells: List[List[Optional[np.ndarray]]] = []
        all_cells: List[np.ndarray] = []

        for row_idx, row_explainer in enumerate(frame_explainers):
            row_slot_maps = collect_slot_maps_for_explainer(
                result_root=result_root,
                dataset=dataset,
                model_name=model_name,
                clean_folder=clean_folder,
                explainer=row_explainer,
                map_ext=map_ext,
            )

            row_cells: List[Optional[np.ndarray]] = []

            for col_idx, (key, cell_kind) in enumerate(frame_order):
                row_map_path = row_slot_maps.get(key, {}).get(sample_id)
                if row_map_path is None:
                    row_cells.append(None)
                    continue

                sample_dir = sample_dir_from_map_path(row_map_path)

                if cell_kind == "image":
                    if key == "clean_causual_score":
                        base_img_path = find_clean_image_near(sample_dir)
                    else:
                        base_img_path = find_adv_image_near(sample_dir)
                    cell_img = load_rgb_image(base_img_path) if base_img_path else None
                else:
                    cell_img = load_overlay_or_map_image(row_map_path, map_ext, frame_source)

                if cell_img is None:
                    row_cells.append(None)
                    continue

                rgb = to_rgb_uint8(cell_img)
                row_cells.append(rgb)
                all_cells.append(rgb)

            # Keep a strict frame-order column count per row.
            while len(row_cells) < n_cols:
                row_cells.append(None)
            grid_cells.append(row_cells)

        if not all_cells:
            st.info("No images found for current selection.")
        else:
            target_h = min(img.shape[0] for img in all_cells)
            target_w = min(img.shape[1] for img in all_cells)
            blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            stitched_rows: List[np.ndarray] = []
            for row_cells in grid_cells:
                row_tiles: List[np.ndarray] = []
                for cell in row_cells[:n_cols]:
                    if cell is None:
                        row_tiles.append(blank)
                    else:
                        row_tiles.append(resize_rgb_uint8(cell, target_w, target_h))
                stitched_rows.append(np.concatenate(row_tiles, axis=1))

            merged = np.concatenate(stitched_rows, axis=0)
            st.image(merged, use_container_width=True)

        col_order: List[str] = []
        for key, cell_kind in frame_order:
            base_label = FRAME_ATTACK_LABELS.get(key, key)
            if cell_kind == "image":
                col_order.append(f"{base_label} image")
            else:
                col_order.append(f"{base_label} overlay")
        row_order = frame_explainers
        st.caption("Order info")
        st.caption("Cols (left -> right): " + " | ".join([f"{i + 1}:{name}" for i, name in enumerate(col_order)]))
        st.caption("Rows (top -> bottom): " + " | ".join([f"{i + 1}:{name}" for i, name in enumerate(row_order)]))


if __name__ == "__main__":
    main()
