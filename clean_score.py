import argparse
import json
import os
from tqdm import tqdm
import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize

from imagenet_metadata import IMAGENET_CLASSNAMES
from util import (
    build_blur_substrate,
    build_causal_metric_model,
    build_zero_shot_clip_classifier,
    generate_hm,
    predict_zero_shot_clip,
    save_causal_metric_summary,
    save_saliency_outputs,
)
from RISE.evaluation import CausalMetric, auc


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
    parser.add_argument("--output-dir", default="test_eval_outputs", help="Where to save generated images")
    parser.add_argument("--img-dir", default="magenet/valimages", help="Directory containing input images")
    parser.add_argument("--save-process", action="store_true", help="Save every deletion/insertion step image")
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
        f"insertion_auc: {payload['insertion_auc']:.6f}",
    ]
    if payload["gt_label"] is not None:
        lines.append(f"gt_label: {payload['gt_label']}")
        lines.append(f"gt_classname: {payload['gt_classname']}")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()

    classifier, _ = build_zero_shot_clip_classifier(
        clip_model,
        device=device,
        num_classes_per_batch=10,
        use_tqdm=True,
    )
    metric_model = build_causal_metric_model(classifier)
    output_dir = os.path.join(args.output_dir, f"{args.mode}_{args.hm_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(args.sample_path, "r", encoding="utf-8") as f:
        sample_list = json.load(f)
    # evaluate loop
    for folder_name, image_name in sample_list.items():
        # sample_dir
        sample_dir = os.path.join(output_dir, folder_name)
        os.makedirs(sample_dir, exist_ok=True)
        
        
        # prepare input
        img_path = os.path.join(args.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        input_resolution = clip_model.visual.input_resolution
        resized_image = image.resize((input_resolution, input_resolution), Image.BICUBIC)
        image_tensor = preprocess(resized_image).unsqueeze(0)
        metric_resize = Resize(tuple(image_tensor.shape[-2:]))
        
        _, _, pred_label, pred_confidence = predict_zero_shot_clip(classifier, image_tensor, device)
        target_label = resolve_target_label(args, pred_label)
        target_texts = [IMAGENET_CLASSNAMES[target_label]]

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
            class_name=IMAGENET_CLASSNAMES[pred_label],
            preprocess=preprocess,
        )
        save_causal_metric_summary(
            image_tensor=image_tensor,
            final_tensor=image_tensor,
            scores=insertion_curve,
            output_path=insertion_summary_path,
            mode="ins",
            class_name=IMAGENET_CLASSNAMES[pred_label],
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
        
        with open(os.path.join(sample_dir, "insertion_information.json"), "w", encoding="utf-8") as f:
            json.dump(insertion_information, f, ensure_ascii=False, indent=2)
        with open(os.path.join(sample_dir, "deletion_information.json"), "w", encoding="utf-8") as f:
            json.dump(deletion_information, f, ensure_ascii=False, indent=2)
    


   

if __name__ == "__main__":
    main()
