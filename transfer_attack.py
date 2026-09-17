import argparse
import json
import os
import shutil
import unicodedata
from pathlib import Path

from tqdm import tqdm
import numpy as np

try:
    from CLIP import clip
except ImportError:
    import clip

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize

from util import (
    DEFAULT_PROMPT_TEMPLATE,
    build_blur_substrate,
    build_causal_metric_model,
    build_zero_shot_clip_classifier,
    generate_hm,
    load_classnames,
    predict_zero_shot_clip,
    save_causal_metric_summary,
    save_saliency_outputs,
)
from RISE.evaluation import CausalMetric, auc
from road_evaluation.road.imputed_dataset import ImputedDataset
from road_evaluation.road.imputations import ChannelMeanImputer, NoisyLinearImputer, ZeroImputer


DEFAULT_APPROACHES = ["ins_optimized_output", "del_optimized_output", "ins_del_optimized_output"]
DEFAULT_EXPLAINERS = ["eclip", "game", "gradcam", "maskclip", "rise"]
APPROACH_ALIASES = {
    "del_optimized_output": ["del_optimized_output", "del_optimize_output"],
    "ins_optimized_output": ["ins_optimized_output", "ins_optimize_output"],
    "ins_del_optimized_output": ["ins_del_optimized_output", "ins_del_optimize_output", "final_output"],
}
DEFAULT_ROAD_PERCENTAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_tensor: torch.Tensor, label: int):
        self.image_tensor = image_tensor.detach().cpu()
        self.label = int(label)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self.image_tensor.clone(), self.label


def normalize_imagenet1k_clip(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std


def choose_road_imputer(kind: str, linear_noise: float):
    if kind == "linear":
        return NoisyLinearImputer(noise=linear_noise)
    if kind == "zero":
        return ZeroImputer()
    return ChannelMeanImputer()


def saliency_to_road_mask(saliency: np.ndarray) -> np.ndarray:
    sal = np.asarray(saliency, dtype=np.float32)
    if sal.ndim == 3 and sal.shape[0] in (1, 3) and sal.shape[2] not in (1, 3):
        sal = np.transpose(sal, (1, 2, 0))
    if sal.ndim == 2:
        sal = np.repeat(sal[:, :, None], 3, axis=2)
    elif sal.ndim == 3 and sal.shape[2] == 1:
        sal = np.repeat(sal, 3, axis=2)
    elif sal.ndim == 3 and sal.shape[2] > 3:
        sal = sal[:, :, :3]
    elif sal.ndim != 3:
        raise ValueError(f"Unsupported saliency shape for ROAD: {sal.shape}")
    return sal


def _compute_auc(curve: np.ndarray, x: np.ndarray) -> float:
    if curve.size == 0:
        return float("nan")
    return float(np.trapezoid(curve.astype(np.float64), x.astype(np.float64)))


def _compute_fair_auc(curve: np.ndarray, x: np.ndarray):
    if curve.size == 0:
        return None
    max_val = float(np.max(curve))
    if not np.isfinite(max_val) or max_val <= 0.0:
        return None
    norm = curve.astype(np.float64) / max_val
    val = float(np.trapezoid(norm, x.astype(np.float64)))
    if not np.isfinite(val):
        return None
    return val


def _to_float_list(values):
    return [float(v) for v in values]


def evaluate_road_deletion(
    metric_model,
    image_raw: torch.Tensor,
    road_mask: np.ndarray,
    pred_label: int,
    target_label: int,
    percentages,
    imputer,
    device,
):
    base_dataset = SingleImageDataset(image_raw, pred_label)
    deletion_curve = []
    deletion_acc_curve = []

    metric_model.eval()
    metric_model.to(device)

    for p in percentages:
        ds_imputed = ImputedDataset(
            base_dataset=base_dataset,
            mask=[road_mask],
            th_p=float(p),
            remove=True,
            imputation=imputer,
            transform=normalize_imagenet1k_clip,
            prediction=[target_label],
            use_cache=False,
        )

        img_imp, label_imp, _ = ds_imputed[0]
        inputs = img_imp.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = metric_model(inputs)
            if outputs.ndim != 2:
                raise ValueError("Unexpected model output shape")
            if target_label < 0 or target_label >= outputs.shape[1]:
                raise ValueError(
                    f"target_label out of range: {target_label}, num_classes={outputs.shape[1]}"
                )
            top = int(torch.argmax(outputs, dim=1).item())
            target_prob = float(outputs[0, target_label].item())
            acc = float(top == int(label_imp))

        deletion_curve.append(target_prob)
        deletion_acc_curve.append(acc)

    x = np.asarray(percentages, dtype=np.float64)
    curve_arr = np.asarray(deletion_curve, dtype=np.float64)
    acc_arr = np.asarray(deletion_acc_curve, dtype=np.float64)
    fair_auc = _compute_fair_auc(curve_arr, x)

    return {
        "metric": "road_deletion",
        "percentages": _to_float_list(percentages),
        "deletion_curve": _to_float_list(deletion_curve),
        "deletion_accuracy_curve": _to_float_list(deletion_acc_curve),
        "deletion_auc": float(_compute_auc(curve_arr, x)),
        "deletion_fair_auc": (float(fair_auc) if fair_auc is not None else None),
        "deletion_accuracy_auc": float(_compute_auc(acc_arr, x)),
        "morf": True,
    }


def parse_csv_or_space_list(raw_items):
    if not raw_items:
        return []
    out = []
    seen = set()
    for item in raw_items:
        for token in str(item).split(","):
            token = token.strip()
            if token and token not in seen:
                seen.add(token)
                out.append(token)
    return out


def _normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return ascii_only.casefold().strip()


def resolve_existing_names(existing_names, requested_names, kind: str):
    if not requested_names:
        return sorted(existing_names)

    existing_list = sorted(set(existing_names))
    normalized_to_existing = {}
    for name in existing_list:
        normalized_to_existing.setdefault(_normalize_name(name), []).append(name)

    resolved = []
    unresolved = []
    for requested in requested_names:
        direct_match = requested if requested in existing_list else None
        if direct_match is not None:
            if direct_match not in resolved:
                resolved.append(direct_match)
            continue

        candidates = normalized_to_existing.get(_normalize_name(requested), [])
        if candidates:
            picked = candidates[0]
            if picked not in resolved:
                resolved.append(picked)
            print(f"[map] {kind}: '{requested}' -> '{picked}'")
            continue

        unresolved.append(requested)

    if unresolved:
        print(f"[warn] unresolved {kind}: {unresolved}")

    return resolved


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Transfer evaluation: rerun target explainer on adversarial images generated by source explainer "
            "and save saliency + deletion/insertion curves."
        )
    )
    parser.add_argument("--root-dir", default=r"D:\neurips_result", help="Root directory containing result/ and transfer/")
    parser.add_argument("--direct-subdir", default="result", help="Subdirectory under root-dir for direct attack outputs")
    parser.add_argument("--transfer-subdir", default="transfer", help="Subdirectory under root-dir for transfer outputs")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional absolute/relative path for transfer output root. "
            "If set, this path is used directly instead of <root-dir>/<transfer-subdir>."
        ),
    )

    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name")
    parser.add_argument(
        "--clip-checkpoint",
        default=None,
        help="Optional local checkpoint path passed to clip.load instead of --clip-model",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument(
        "--classnames-path",
        default=None,
        help="Optional path to class names (.txt or .json). Default: ImageNet class names",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template for zero-shot text, use {} as class placeholder",
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names. Default: auto-discover under direct root",
    )
    parser.add_argument(
        "--model-subdir",
        default=None,
        help=(
            "Optional model folder under each dataset (e.g., CLIP_ViTB16). "
            "If omitted, script auto-detects whether a model layer exists."
        ),
    )
    parser.add_argument(
        "--approaches",
        nargs="*",
        default=None,
        help="Approach folders to process (space/comma separated)",
    )
    parser.add_argument(
        "--explainers",
        nargs="*",
        default=None,
        help="Explainer folders to process (space/comma separated)",
    )

    parser.add_argument("--step", type=int, default=224, help="Pixels modified per causal-metric step")
    parser.add_argument("--kernel-size", type=int, default=11, help="Gaussian blur kernel size for insertion")
    parser.add_argument("--kernel-sigma", type=int, default=5, help="Gaussian blur sigma for insertion")
    parser.add_argument(
        "--save-process",
        action="store_true",
        help="Save every deletion/insertion step image",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="CausalMetric verbosity: 0 no plot, 1 final step only, 2 show every step",
    )
    parser.add_argument(
        "--pair-folder-style",
        choices=["underscore", "arrow"],
        default="underscore",
        help="Folder name style for transfer pair: source_target or source_to_target",
    )
    parser.add_argument(
        "--strict-size",
        action="store_true",
        help="If set, fail when adversarial image size differs from model input resolution",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["traditional", "road", "both"],
        default="both",
        help="Evaluation outputs to run on transfer samples.",
    )
    parser.add_argument(
        "--road-imputer",
        choices=["linear", "zero", "fixed"],
        default="linear",
        help="ROAD imputer used when eval-mode includes road.",
    )
    parser.add_argument(
        "--road-linear-noise",
        type=float,
        default=0.01,
        help="Noise term for ROAD NoisyLinearImputer (linear mode only).",
    )
    parser.add_argument(
        "--road-percentages",
        nargs="*",
        type=float,
        default=DEFAULT_ROAD_PERCENTAGES,
        help="ROAD deletion percentages in [0,1].",
    )
    parser.add_argument(
        "--road-json-name",
        default="road_deletion_information.json",
        help="Output filename for ROAD result per sample.",
    )
    return parser.parse_args()


def resolve_attack_mode(approach: str) -> str:
    lower = approach.lower()
    if "ins_del" in lower or "final" in lower:
        return "final"
    if "del" in lower:
        return "del"
    if "ins" in lower:
        return "ins"
    raise ValueError(f"Cannot infer attack mode from approach folder: {approach}")


def expand_approach_candidates(approach_name: str):
    if approach_name in APPROACH_ALIASES:
        return list(APPROACH_ALIASES[approach_name])

    for _, aliases in APPROACH_ALIASES.items():
        if approach_name in aliases:
            return list(aliases)

    return [approach_name]


def resolve_approach_for_dataset(base_dir: Path, requested_approach: str):
    if not base_dir.exists() or not base_dir.is_dir():
        return None

    existing = [p.name for p in base_dir.iterdir() if p.is_dir()]
    if not existing:
        return None

    candidates = []
    candidates.extend(expand_approach_candidates(requested_approach))

    for name in existing:
        if _normalize_name(name) == _normalize_name(requested_approach):
            candidates.append(name)

    for candidate in candidates:
        if candidate in existing:
            if candidate != requested_approach:
                print(f"[map] approach: '{requested_approach}' -> '{candidate}'")
            return candidate

    return None


def resolve_adv_image_names(attack_mode: str):
    if attack_mode == "del":
        return ["adversarial_image_del.png"]
    if attack_mode == "ins":
        return ["adversarial_image_ins.png"]
    if attack_mode == "final":
        return [
            "adversarial_image_del_ins.png",
            "adversarial_image_final.png",
            "adversarial_image_del.png",
            "adversarial_image_ins.png",
            "adversarial_image.png",
        ]
    raise ValueError(f"Unknown attack mode: {attack_mode}")


def load_json_if_exists(path: Path):
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_target_label(source_info, pred_label):
    if isinstance(source_info, dict):
        candidate = source_info.get("retain_class_label")
        if candidate is None:
            candidate = source_info.get("target_label")
        if candidate is not None:
            try:
                return int(candidate)
            except (TypeError, ValueError):
                pass
    return int(pred_label)


def make_pair_folder_name(source_explainer: str, target_explainer: str, style: str) -> str:
    if style == "arrow":
        return f"{source_explainer}_to_{target_explainer}"
    return f"{source_explainer}_{target_explainer}"


def has_approach_like_subdir(base_dir: Path):
    if not base_dir.exists() or not base_dir.is_dir():
        return False
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            resolve_attack_mode(child.name)
            return True
        except ValueError:
            continue
    return False


def resolve_dataset_model_dirs(dataset_dir: Path, requested_model_subdir: str):
    if requested_model_subdir:
        model_dir = dataset_dir / requested_model_subdir
        if model_dir.exists() and model_dir.is_dir():
            return [(requested_model_subdir, model_dir)]
        print(f"[warn] model-subdir not found in dataset '{dataset_dir.name}': {requested_model_subdir}")
        return []

    if has_approach_like_subdir(dataset_dir):
        return [(None, dataset_dir)]

    model_dirs = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir() and has_approach_like_subdir(p)]
    return [(p.name, p) for p in model_dirs]


def discover_explainers(direct_dir: Path, datasets):
    discovered = set()
    for dataset_name in datasets:
        dataset_dir = direct_dir / dataset_name
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            continue

        for _, base_dir in resolve_dataset_model_dirs(dataset_dir, requested_model_subdir=None):
            for approach_dir in base_dir.iterdir():
                if not approach_dir.is_dir():
                    continue
                for explainer_dir in approach_dir.iterdir():
                    if explainer_dir.is_dir():
                        discovered.add(explainer_dir.name)
    return sorted(discovered)


def discover_adv_samples(source_root: Path, adv_image_names):
    if isinstance(adv_image_names, str):
        names = [adv_image_names]
    else:
        names = list(adv_image_names)

    candidates_by_sample = {}
    for name in names:
        for adv_path in sorted(source_root.rglob(name)):
            if not adv_path.is_file():
                continue
            sample_dir = adv_path.parent
            rel_sample_path = sample_dir.relative_to(source_root)
            key = str(rel_sample_path).replace("\\", "/")
            if key not in candidates_by_sample:
                candidates_by_sample[key] = (rel_sample_path, sample_dir, adv_path)

    return [candidates_by_sample[k] for k in sorted(candidates_by_sample.keys())]


def main():
    args = parse_args()
    for p in args.road_percentages:
        if p < 0.0 or p > 1.0:
            raise ValueError("All --road-percentages values must be within [0, 1]")

    root_dir = Path(args.root_dir).expanduser().resolve()
    direct_dir = root_dir / args.direct_subdir
    transfer_dir = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (root_dir / args.transfer_subdir)
    )

    if not direct_dir.exists():
        raise FileNotFoundError(f"Direct result directory not found: {direct_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    classnames = load_classnames(args.classnames_path)
    blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)

    clip_source = args.clip_checkpoint or args.clip_model
    if args.clip_checkpoint and not os.path.isfile(args.clip_checkpoint):
        raise FileNotFoundError(f"--clip-checkpoint not found: {args.clip_checkpoint}")

    clip_model, preprocess = clip.load(clip_source, device=device)
    clip_model.eval()

    classifier, _ = build_zero_shot_clip_classifier(
        clip_model,
        device=device,
        classnames=classnames,
        prompt_template=args.prompt_template,
        num_classes_per_batch=10,
        use_tqdm=True,
    )
    metric_model = build_causal_metric_model(classifier)
    road_imputer = choose_road_imputer(args.road_imputer, args.road_linear_noise)
    run_traditional = args.eval_mode in ("traditional", "both")
    run_road = args.eval_mode in ("road", "both")

    discovered_datasets = sorted([p.name for p in direct_dir.iterdir() if p.is_dir()])
    datasets_requested = parse_csv_or_space_list(args.datasets)
    datasets = resolve_existing_names(discovered_datasets, datasets_requested, "datasets")
    if not datasets:
        datasets = discovered_datasets

    approaches_requested = parse_csv_or_space_list(args.approaches) or list(DEFAULT_APPROACHES)

    discovered_explainers = discover_explainers(direct_dir, datasets)
    explainers_requested = parse_csv_or_space_list(args.explainers) or list(DEFAULT_EXPLAINERS)
    explainers = resolve_existing_names(discovered_explainers, explainers_requested, "explainers")
    if not explainers:
        explainers = list(DEFAULT_EXPLAINERS)

    total_samples = 0
    processed_samples = 0
    skipped_samples = 0
    skipped_pairs_no_source = 0

    for dataset in datasets:
        dataset_dir = direct_dir / dataset
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            print(f"[skip] dataset directory missing: {dataset_dir}")
            continue

        dataset_model_dirs = resolve_dataset_model_dirs(dataset_dir, args.model_subdir)
        if not dataset_model_dirs:
            print(f"[skip] no valid model/approach directories found under dataset: {dataset_dir}")
            continue

        for model_name, dataset_base_dir in dataset_model_dirs:
            for requested_approach in approaches_requested:
                approach = resolve_approach_for_dataset(dataset_base_dir, requested_approach)
                if approach is None:
                    model_tag = f"/{model_name}" if model_name else ""
                    print(f"[skip] approach not found in dataset '{dataset}{model_tag}': {requested_approach}")
                    continue

                try:
                    attack_mode = resolve_attack_mode(approach)
                except ValueError:
                    print(f"[skip] unknown approach naming for mode inference: {dataset}/{approach}")
                    continue

                adv_image_names = resolve_adv_image_names(attack_mode)

                for source_explainer in explainers:
                    source_root = dataset_base_dir / approach / source_explainer
                    if not source_root.exists() or not source_root.is_dir():
                        skipped_pairs_no_source += 1
                        continue

                    adv_samples = discover_adv_samples(source_root, adv_image_names)
                    if not adv_samples:
                        continue

                    for target_explainer in explainers:
                        if target_explainer == source_explainer:
                            continue

                        pair_name = make_pair_folder_name(source_explainer, target_explainer, args.pair_folder_style)
                        if model_name:
                            pair_root = transfer_dir / dataset / model_name / approach / pair_name
                        else:
                            pair_root = transfer_dir / dataset / approach / pair_name
                        pair_root.mkdir(parents=True, exist_ok=True)

                        model_prefix = f"{model_name}/" if model_name else ""
                        progress_desc = f"{dataset}/{model_prefix}{approach}: {source_explainer}->{target_explainer}"

                        for rel_sample_path, source_sample_dir, adv_path in tqdm(adv_samples, desc=progress_desc, leave=False):
                            total_samples += 1

                            target_sample_dir = pair_root / rel_sample_path
                            target_sample_dir.mkdir(parents=True, exist_ok=True)

                            source_info = load_json_if_exists(source_sample_dir / "curve_information.json")

                            image = Image.open(adv_path).convert("RGB")
                            expected_res = int(clip_model.visual.input_resolution)
                            image_was_resized = False
                            if image.size != (expected_res, expected_res):
                                if args.strict_size:
                                    raise ValueError(
                                        f"Unexpected image size for {adv_path}: got {image.size}, expected {(expected_res, expected_res)}"
                                    )
                                image = image.resize((expected_res, expected_res), Image.BICUBIC)
                                image_was_resized = True

                            image_normalized = preprocess(image).unsqueeze(0)
                            image_raw = transforms.ToTensor()(image)
                            metric_resize = Resize(tuple(image_normalized.shape[-2:]))

                            _, probs, pred_label, pred_confidence = predict_zero_shot_clip(
                                classifier,
                                image_normalized,
                                device,
                            )

                            target_label = get_target_label(source_info, pred_label)
                            if not (0 <= target_label < len(classnames)):
                                target_label = int(pred_label)

                            target_texts = [classnames[target_label]]
                            with torch.no_grad():
                                text_tokens = clip.tokenize(target_texts).to(device)
                                text_embedding = clip_model.encode_text(text_tokens)
                                text_embedding = F.normalize(text_embedding, dim=-1)

                            heatmap = generate_hm(
                                clip_model,
                                target_explainer,
                                image_normalized,
                                text_embedding,
                                target_texts,
                                metric_resize,
                                preprocess,
                            )
                            saliency_np = heatmap.detach().cpu().numpy()

                            save_saliency_outputs(
                                saliency_np,
                                image,
                                str(target_sample_dir / "saliency"),
                                stem=f"{target_explainer}_saliency",
                            )

                            rerun_results = {}
                            if run_traditional:
                                for rerun_mode in ["del", "ins"]:
                                    rerun_step_function = (lambda x: torch.zeros_like(x)) if rerun_mode == "del" else blur_fn
                                    metric = CausalMetric(metric_model, rerun_mode, args.step, rerun_step_function)

                                    rerun_process_dir = target_sample_dir / f"steps_{rerun_mode}"
                                    if args.save_process:
                                        rerun_process_dir.mkdir(parents=True, exist_ok=True)

                                    curve = metric.single_run(
                                        image_normalized,
                                        saliency_np,
                                        verbose=args.verbose,
                                        save_to=str(rerun_process_dir) if args.save_process else None,
                                    )

                                    save_causal_metric_summary(
                                        image_tensor=image_normalized,
                                        final_tensor=torch.zeros_like(image_normalized) if rerun_mode == "del" else image_normalized,
                                        scores=curve,
                                        output_path=str(target_sample_dir / f"{rerun_mode}_summary.png"),
                                        mode=rerun_mode,
                                        class_name=classnames[target_label],
                                        preprocess=preprocess,
                                    )

                                    rerun_results[rerun_mode] = {
                                        "curve": curve.tolist(),
                                        "auc": float(auc(curve)),
                                    }

                            if run_road:
                                road_mask = saliency_to_road_mask(saliency_np)
                                road_information = evaluate_road_deletion(
                                    metric_model=metric_model,
                                    image_raw=image_raw,
                                    road_mask=road_mask,
                                    pred_label=int(pred_label),
                                    target_label=int(target_label),
                                    percentages=args.road_percentages,
                                    imputer=road_imputer,
                                    device=device,
                                )
                                road_information.update(
                                    {
                                        "dataset": dataset,
                                        "model_name": model_name,
                                        "approach": approach,
                                        "source_explainer": source_explainer,
                                        "target_explainer": target_explainer,
                                        "source_adv_image": str(adv_path),
                                        "pred_label": int(pred_label),
                                        "pred_confidence": float(pred_confidence),
                                        "target_label": int(target_label),
                                        "image_path_used": str(target_sample_dir / adv_path.name),
                                        "saliency_path_used": str(target_sample_dir / "saliency" / f"{target_explainer}_saliency.npy"),
                                    }
                                )
                                with (target_sample_dir / args.road_json_name).open("w", encoding="utf-8") as f:
                                    json.dump(road_information, f, ensure_ascii=False, indent=2)

                            adv_prob_target = float(probs[0, target_label].item())
                            curve_information = {
                                "attack_mode": attack_mode,
                                "eval_mode": args.eval_mode,
                                "dataset": dataset,
                                "model_name": model_name,
                                "approach": approach,
                                "source_explainer": source_explainer,
                                "target_explainer": target_explainer,
                                "source_adv_image": str(adv_path),
                                "image_was_resized": image_was_resized,
                                "retain_class_label": int(target_label),
                                "retain_classname": classnames[target_label],
                                "clean_prob": source_info.get("clean_prob") if isinstance(source_info, dict) else None,
                                "adv_prob": adv_prob_target,
                                "clean_score": source_info.get("clean_score") if isinstance(source_info, dict) else None,
                                "adv_score": adv_prob_target,
                                "pred_label_adv": int(pred_label),
                                "pred_classname_adv": classnames[pred_label],
                                "pred_confidence_adv": float(pred_confidence),
                                "adv_pred_label": int(pred_label),
                                "adv_pred_classname": classnames[pred_label],
                                "adv_pred_confidence": float(pred_confidence),
                                "adv_pred_retain": bool(pred_label == target_label),
                                "rerun": rerun_results,
                            }

                            with (target_sample_dir / "curve_information.json").open("w", encoding="utf-8") as f:
                                json.dump(curve_information, f, ensure_ascii=False, indent=2)

                            # Keep the attacked image in transfer folder for easier downstream auditing.
                            shutil.copy2(adv_path, target_sample_dir / adv_path.name)
                            processed_samples += 1

    print(
        "Transfer rerun finished | "
        f"processed={processed_samples}, skipped={skipped_samples}, total={total_samples}, "
        f"missing_source_pairs={skipped_pairs_no_source}, output={transfer_dir}"
    )

    if processed_samples == 0:
        print("[hint] No sample was processed. Check dataset/model/approach/explainer names under direct root.")
        print(f"[hint] direct root: {direct_dir}")
        if discovered_datasets:
            print(f"[hint] available datasets: {discovered_datasets}")


if __name__ == "__main__":
    main()
