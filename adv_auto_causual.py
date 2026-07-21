import argparse
import json
import os

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize
from torchvision.utils import save_image

from imagenet_metadata import IMAGENET_CLASSNAMES
from util import (
    build_blur_substrate,
    build_causal_metric_model,
    build_zero_shot_clip_classifier,
    generate_hm,
    normalize_ImageNet1k,
    normalize_ImageNet1k,
    predict_zero_shot_clip,
    save_causal_metric_summary,
    save_saliency_outputs,
    denorm_ImageNet1k,
    normalize_ImageNet1k,
)
from RISE.evaluation import AdversarialCausalMetric, CausalMetric, auc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype deletion/insertion evaluation for CLIP zero-shot explanations"
    )
    parser.add_argument("--image-path", required=True, help="Path to one RGB image")
    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name")
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
    parser.add_argument("--output-json", default="test_eval_result.json", help="Where to save numeric results as JSON")
    parser.add_argument("--output-txt", default="test_eval_result.txt", help="Where to save a short text summary")
    parser.add_argument("--output-dir", default="test_eval_outputs", help="Where to save generated images")
    parser.add_argument("--save-process", action="store_true", help="Save every deletion/insertion step image")
    parser.add_argument("--eps", type=float, default=32.0, help="Maximum perturbation for adversarial attack (in pixel values)")
    parser.add_argument("--alpha", type=float, default=8.0, help="Step size for adversarial attack (in pixel values)")
    parser.add_argument("--pgd-steps", type=int, default=50, help="Number of PGD steps for adversarial attack")
    parser.add_argument("--mode", default="del", choices=["del", "ins"], help="Causal metric mode: deletion or insertion")
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="CausalMetric verbosity: 0 no plot, 1 final step only, 2 show every step",
    )
    return parser.parse_args()


def resolve_target_label(args, pred_label):
    if args.target_source == "pred":
        return pred_label
    if args.gt_label is None:
        raise ValueError("--gt-label is required when --target-source gt")
    if not (0 <= args.gt_label < len(IMAGENET_CLASSNAMES)):
        raise ValueError("--gt-label must be a valid ImageNet class index")
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
    ]
    if payload["gt_label"] is not None:
        lines.append(f"gt_label: {payload['gt_label']}")
        lines.append(f"gt_classname: {payload['gt_classname']}")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


    

def main():
    args = parse_args()
    blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()

    classifier, _ = build_zero_shot_clip_classifier(
        clip_model,
        device=device,
        num_classes_per_batch=10,
        use_tqdm=True,
    )
    metric_model = build_causal_metric_model(classifier) # clip classifier with sofmax probabilitys

    image = Image.open(args.image_path).convert("RGB")
    input_resolution = clip_model.visual.input_resolution
    resized_image = image.resize((input_resolution, input_resolution), Image.BICUBIC) # Pillow_image
    image_normalized = preprocess(resized_image).unsqueeze(0) # normalized tensor
    image_raw = denorm_ImageNet1k(image_normalized).to(device) # raw tensor [0, 1]
    metric_resize = Resize(tuple(image_normalized.shape[-2:]))

    _, _, pred_label, pred_confidence = predict_zero_shot_clip(classifier, image_normalized, device)

    target_label = resolve_target_label(args, pred_label)
    target_texts = [IMAGENET_CLASSNAMES[target_label]]

    with torch.no_grad():
        text_tokens = clip.tokenize(target_texts).to(device)
        text_embedding = clip_model.encode_text(text_tokens)
        text_embedding = F.normalize(text_embedding, dim=-1)

    heatmap = generate_hm(
        clip_model, # raw models
        args.hm_type, # eclips
        image_normalized, # normalzied image
        text_embedding,
        target_texts,
        metric_resize,
        preprocess,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_saliency_outputs(
        heatmap,
        resized_image,
        args.output_dir,
        stem=f"{args.hm_type}_saliency",
    )
    
    step_function = lambda x: torch.zeros_like(x) if args.mode == "del" else blur_fn


    # ================================================ adversarial attack ================

    adv_causualmetric = AdversarialCausalMetric(
        metric_model, # softmax head model
        clip_model, # raw model
        args.mode,
        args.step,
        step_function,
        args.hm_type,
        text_embedding,
        target_texts,
        metric_resize,
        preprocess
    )
    
    process_dir = os.path.join(args.output_dir, "deletion_steps")
    if args.save_process:
        os.makedirs(process_dir, exist_ok=True)

    x_adv, details = adv_causualmetric.single_run( # are not normalzied
        image_raw,
        generate_hm, # explain function
        eps=args.eps / 255.0,
        alpha=args.alpha / 255.0, 
        pgd_steps=args.pgd_steps,
    )
    
    x_adv = x_adv.detach().cpu()    

    
    
    save_image(x_adv, os.path.join(args.output_dir, f"adversarial_image_{args.mode}.png"))
    
    
    x_adv_normalize = normalize_ImageNet1k(x_adv)
    
    # ================================================ rerun ================
    clean_causualmetric = CausalMetric(metric_model, args.mode, args.step, step_function)
    
 
    heatmap = generate_hm(
        clip_model,
        args.hm_type,
        # resized_image,
        x_adv_normalize, # replace resized_image
        text_embedding,
        target_texts,
        metric_resize,
        preprocess,
    )
    saliency = heatmap.detach().cpu().numpy()

    save_saliency_outputs(
        saliency,
        resized_image,
        args.output_dir,
        stem=f"{args.mode}_adv_{args.hm_type}_saliency",
    )
    curve = clean_causualmetric.single_run(
        x_adv_normalize,
        saliency,
        verbose=args.verbose,
        save_to=process_dir if args.save_process else None,
    )

    save_causal_metric_summary(
        image_tensor=x_adv,
        final_tensor=torch.zeros_like(x_adv) if args.mode == "del" else x_adv,
        scores=curve,
        output_path=os.path.join(args.output_dir, f"{args.mode}_summary.png"),
        mode=args.mode,
        class_name=IMAGENET_CLASSNAMES[pred_label],
        preprocess=preprocess,
    )
    
    print("Auc: ", auc(curve))



if __name__ == "__main__":
    main()
