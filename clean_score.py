import argparse
import json
import os
import numpy as np
from tqdm import tqdm
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


def choose_imputer(kind: str, linear_noise: float):
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


def compute_auc(curve: np.ndarray, x: np.ndarray) -> float:
    if curve.size == 0:
        return float("nan")
    return float(np.trapezoid(curve.astype(np.float64), x.astype(np.float64)))


def compute_fair_auc(curve: np.ndarray, x: np.ndarray):
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


def to_float_list(values):
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

    return {
        "percentages": to_float_list(percentages),
        "deletion_curve": to_float_list(deletion_curve),
        "deletion_accuracy_curve": to_float_list(deletion_acc_curve),
        "deletion_auc": float(compute_auc(curve_arr, x)),
        "deletion_fair_auc": compute_fair_auc(curve_arr, x),
        "deletion_accuracy_auc": float(compute_auc(acc_arr, x)),
        "morf": True,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype deletion/insertion evaluation for CLIP zero-shot explanations"
    )
    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name")
    parser.add_argument(
        "--clip-checkpoint",
        default=None,
        help="Optional local checkpoint path passed to clip.load instead of --clip-model",
    )
    parser.add_argument(
        "--hm-type",
        default="eclip",
        choices=["selfattn", "gradcam", "maskclip", "eclip", "eclip-wo-ksim", "game", "rollout", "surgery", "m2ib", "rise"],
        help="Explanation method passed to CLIPExplainRunner.generate_hm",
    )
    parser.add_argument(
        "--target-source",
        default="pred",
        choices=["pred", "gt"],
        help="Use predicted label or ground-truth label prompt for the saliency map",
    )
    parser.add_argument("--gt-label", type=int, default=None, help="Optional ImageNet class index for GT prompt")
    parser.add_argument("--step", type=int, default=224, help="Pixels modified per causal-metric step")
    parser.add_argument("--kernel-size", type=int, default=11, help="Gaussian blur kernel size for insertion")
    parser.add_argument("--kernel-sigma", type=int, default=5, help="Gaussian blur sigma for insertion")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--output-dir", default="test_eval_outputs", help="Where to save generated images")
    parser.add_argument("--img-dir", default="Imagenet/val", help="Directory containing input images")
    parser.add_argument("--sample-path", default="vit_b_16_1k.json", help="Path to JSON file containing image samples")
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
    parser.add_argument("--save-process", action="store_true", help="Save every deletion/insertion step image")
    parser.add_argument(
        "--road-imputer",
        choices=["linear", "zero", "fixed"],
        default="linear",
        help="ROAD imputer: linear (NoisyLinear), zero, fixed(channel mean)",
    )
    parser.add_argument(
        "--road-linear-noise",
        type=float,
        default=0.01,
        help="Noise term for NoisyLinearImputer (used when --road-imputer linear)",
    )
    parser.add_argument(
        "--road-percentages",
        nargs="*",
        type=float,
        default=DEFAULT_ROAD_PERCENTAGES,
        help="ROAD deletion percentages in [0,1], e.g. --road-percentages 0.1 0.2 ...",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="CausalMetric verbosity: 0 no plot, 1 final step only, 2 show every step",
    )
    return parser.parse_args()


def resolve_target_label(args, pred_label, num_classes):
    if args.target_source == "pred":
        return pred_label
    if args.gt_label is None:
        raise ValueError("--gt-label is required when --target-source gt")
    if not (0 <= args.gt_label < num_classes):
        raise ValueError("--gt-label must be a valid class index for the loaded class names")
    return int(args.gt_label)


def save_outputs(output_json, output_txt, payload):
    output_json_dir = os.path.dirname(output_json)
    output_txt_dir = os.path.dirname(output_txt)
    if output_json_dir:
        os.makedirs(output_json_dir, exist_ok=True)
    if output_txt_dir:
        os.makedirs(output_txt_dir, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        f"image_path: {payload['image_path']}",
        f"clip_model: {payload['clip_model']}",
        f"hm_type: {payload['hm_type']}",
        f"pred_label: {payload['pred_label']}",
        f"pred_classname: {payload['pred_classname']}",
        f"pred_confidence: {payload['pred_confidence']:.6f}",
        f"target_source: {payload['target_source']}",
        f"target_label: {payload['target_label']}",
        f"target_classname: {payload['target_classname']}",
        f"deletion_auc: {payload['deletion_auc']:.6f}",
        f"insertion_auc: {payload['insertion_auc']:.6f}",
    ]
    if payload["gt_label"] is not None:
        lines.append(f"gt_label: {payload['gt_label']}")
        lines.append(f"gt_classname: {payload['gt_classname']}")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    for p in args.road_percentages:
        if p < 0.0 or p > 1.0:
            raise ValueError("All --road-percentages values must be within [0, 1]")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    classnames = load_classnames(args.classnames_path)
    road_imputer = choose_imputer(args.road_imputer, args.road_linear_noise)

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
    output_dir = os.path.join(args.output_dir, f"{args.hm_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(args.sample_path, "r", encoding="utf-8") as f:
        sample_list = json.load(f)
    # evaluate loop
    for folder_name, image_name in tqdm(sample_list.items()):
        # sample_dir
        sample_dir = os.path.join(output_dir, folder_name)
        os.makedirs(sample_dir, exist_ok=True)
        
        
        # prepare input
        img_path = os.path.join(args.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        input_resolution = clip_model.visual.input_resolution
        resized_image = image.resize((input_resolution, input_resolution), Image.BICUBIC)
        image_tensor = preprocess(resized_image).unsqueeze(0)
        image_raw = transforms.ToTensor()(resized_image)
        metric_resize = Resize(tuple(image_tensor.shape[-2:]))
        
        _, _, pred_label, pred_confidence = predict_zero_shot_clip(classifier, image_tensor, device)
        target_label = resolve_target_label(args, pred_label, len(classnames))
        target_texts = [classnames[target_label]]

        with torch.no_grad():
            text_tokens = clip.tokenize(target_texts).to(device)
            text_embedding = clip_model.encode_text(text_tokens)
            text_embedding = F.normalize(text_embedding, dim=-1)
            
        heatmap = generate_hm(
            clip_model,
            args.hm_type,
            image_tensor, # normalized image
            text_embedding,
            target_texts,
            metric_resize,
            preprocess,
        )

        clean_resized_path = os.path.join(sample_dir, "clean_image_resized.png")
        resized_image.save(clean_resized_path)
        
        saliency = heatmap.detach().cpu().numpy()
        save_saliency_outputs(
            heatmap,
            resized_image,
            os.path.join(sample_dir, "saliency"),
            stem=f"{args.hm_type}_saliency",
        )
        
        blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)
        insertion = CausalMetric(metric_model, "ins", args.step, substrate_fn=blur_fn)
        deletion = CausalMetric(metric_model, "del", args.step, substrate_fn=lambda x: torch.zeros_like(x))  
        
        deletion_process_dir = os.path.join(sample_dir, "deletion_steps")
        insertion_process_dir = os.path.join(sample_dir, "insertion_steps")
        if args.save_process:
            os.makedirs(deletion_process_dir, exist_ok=True)
            os.makedirs(insertion_process_dir, exist_ok=True)
            
        deletion_curve = deletion.single_run(
            image_tensor,
            saliency,
            verbose=args.verbose,
            save_to=deletion_process_dir if args.save_process else None,
        )
        insertion_curve = insertion.single_run(
            image_tensor,
            saliency,
            verbose=args.verbose,
            save_to=insertion_process_dir if args.save_process else None,
        )
        
        deletion_summary_path = os.path.join(sample_dir, "deletion_summary.png")
        insertion_summary_path = os.path.join(sample_dir, "insertion_summary.png")
        
        save_causal_metric_summary(
            image_tensor=image_tensor,
            final_tensor=torch.zeros_like(image_tensor),
            scores=deletion_curve,
            output_path=deletion_summary_path,
            mode="del",
            class_name=classnames[pred_label],
            preprocess=preprocess,
        )
        save_causal_metric_summary(
            image_tensor=image_tensor,
            final_tensor=image_tensor,
            scores=insertion_curve,
            output_path=insertion_summary_path,
            mode="ins",
            class_name=classnames[pred_label],
            preprocess=preprocess,
        )
        
        insertion_information = {
            'insertion_curve': insertion_curve.tolist(),
            'insertion_auc': float(auc(insertion_curve)),
        }
        deletion_information = {
            'deletion_curve': deletion_curve.tolist(),
            'deletion_auc': float(auc(deletion_curve)),
        }

        road_mask = saliency_to_road_mask(saliency)
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
                "metric": "road_deletion",
                "pred_label": int(pred_label),
                "pred_confidence": float(pred_confidence),
                "target_label": int(target_label),
                "saliency_path_used": os.path.join(sample_dir, "saliency", f"{args.hm_type}_saliency.npy"),
                "image_path_used": clean_resized_path,
            }
        )
        
        with open(os.path.join(sample_dir, "insertion_information.json"), "w", encoding="utf-8") as f:
            json.dump(insertion_information, f, ensure_ascii=False, indent=2)
        with open(os.path.join(sample_dir, "deletion_information.json"), "w", encoding="utf-8") as f:
            json.dump(deletion_information, f, ensure_ascii=False, indent=2)
        with open(os.path.join(sample_dir, "road_deletion_information.json"), "w", encoding="utf-8") as f:
            json.dump(road_information, f, ensure_ascii=False, indent=2)
    

   

if __name__ == "__main__":
    main()
