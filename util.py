import time
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import Game_MM_CLIP.clip as mm_clip
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import torch.nn.functional as F
from clip_utils import build_zero_shot_classifier
from generate_emap import CLIPExplainRunner
from imagenet_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES


_EXPLAINER_CACHE = {}
DEFAULT_PROMPT_TEMPLATE = "a photo of {}"


def load_classnames(classnames_path=None):
    """Load class names from txt/json file, fallback to ImageNet class names."""
    if classnames_path is None:
        return list(IMAGENET_CLASSNAMES)

    path = Path(classnames_path)
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        classnames = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    classnames.append(name)
        if not classnames:
            raise ValueError(f"No class names found in txt file: {path}")
        return classnames

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            classnames = [str(x).strip() for x in data if str(x).strip()]
            if not classnames:
                raise ValueError(f"No class names found in json list: {path}")
            return classnames

        if isinstance(data, dict):
            keys = list(data.keys())

            def _value_to_name(value):
                if isinstance(value, str):
                    return value.strip()
                if isinstance(value, list) and len(value) > 0:
                    if len(value) >= 2 and isinstance(value[1], str):
                        return value[1].strip()
                    if isinstance(value[0], str):
                        return value[0].strip()
                return None

            if keys and all(str(k).isdigit() for k in keys):
                sorted_items = sorted(data.items(), key=lambda kv: int(kv[0]))
                classnames = []
                for _, value in sorted_items:
                    name = _value_to_name(value)
                    if name:
                        classnames.append(name)
                if classnames:
                    return classnames

            classnames = []
            for _, value in data.items():
                name = _value_to_name(value)
                if name:
                    classnames.append(name)
            if classnames:
                return classnames

        raise ValueError(f"Unsupported json format for class names: {path}")

    raise ValueError(f"Unsupported class names file extension: {path.suffix}")


class ZeroShotClipClassifier(nn.Module):
    def __init__(self, clip_model, zero_shot_weights, logit_scale=100.0):
        super().__init__()
        self.clip_model = clip_model
        self.zero_shot_weights = zero_shot_weights
        self.logit_scale = logit_scale

    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        return self.logit_scale * image_features @ self.zero_shot_weights


class SoftmaxModel(nn.Module):
    def __init__(self, model, dim=1):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, inputs):
        return self.softmax(self.model(inputs))


def build_zero_shot_clip_classifier(
    clip_model,
    device,
    classnames=None,
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
    templates=None,
    num_classes_per_batch=10,
    use_tqdm=True,
):
    if classnames is None:
        classnames = IMAGENET_CLASSNAMES

    if templates is None:
        templates = (prompt_template,)

    zero_shot_weights = build_zero_shot_classifier(
        clip_model,
        classnames=classnames,
        templates=templates,
        num_classes_per_batch=num_classes_per_batch,
        device=device,
        use_tqdm=use_tqdm,
    )
    classifier = ZeroShotClipClassifier(clip_model=clip_model, zero_shot_weights=zero_shot_weights)
    classifier.eval()
    return classifier, zero_shot_weights


def predict_zero_shot_clip(classifier, image_tensor, device):
    with torch.no_grad():
        logits = classifier(image_tensor.to(device))
        probs = logits.softmax(dim=-1)
        pred_label = int(torch.argmax(probs, dim=-1).item())
        pred_confidence = float(probs[0, pred_label].item())
    return logits, probs, pred_label, pred_confidence


def build_causal_metric_model(classifier):
    metric_model = SoftmaxModel(classifier, dim=1)
    metric_model.eval()
    return metric_model


def build_blur_substrate(gkern_or_kernel_size=11, kernel_size=11, kernel_sigma=5):
    if callable(gkern_or_kernel_size):
        gkern_fn = gkern_or_kernel_size
    else:
        from RISE.evaluation import gkern as gkern_fn

        kernel_sigma = kernel_size
        kernel_size = gkern_or_kernel_size

    kernel = gkern_fn(kernel_size, kernel_sigma)

    def blur_fn(x):
        kernel_on_device = kernel.to(device=x.device, dtype=x.dtype)
        return nn.functional.conv2d(x, kernel_on_device, padding=kernel_size // 2)

    return blur_fn


def _get_explainer(clipmodel, preprocess):
    cache_key = id(clipmodel)
    if cache_key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[cache_key] = CLIPExplainRunner(
            clipmodel=clipmodel,
            preprocess=preprocess,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
    return _EXPLAINER_CACHE[cache_key]

def generate_hm(clipmodel, hm_type, img, txt_embedding, txts, resize, preprocess):
    explainer = _get_explainer(clipmodel, preprocess)
    return explainer.generate_hm(hm_type, img, txt_embedding, txts, resize)


def visualize(hmap, raw_image, resize):
    image = np.asarray(raw_image.copy())
    hmap = resize(hmap.unsqueeze(0))[0].cpu().numpy()
    color = cv2.applyColorMap((hmap*255).astype(np.uint8), cv2.COLORMAP_JET) # cv2 to plt
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    c_ret = np.clip(image * (1 - 0.5) + color * 0.5, 0, 255).astype(np.uint8)
    return c_ret


def save_saliency_outputs(hmap, raw_image, output_dir, stem="saliency_map"):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(hmap, torch.Tensor):
        saliency = hmap.detach().cpu().numpy()
    else:
        saliency = np.asarray(hmap)

    saliency = saliency.astype(np.float32)
    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()

    raw_path = os.path.join(output_dir, f"{stem}.npy")
    heatmap_path = os.path.join(output_dir, f"{stem}.png")
    overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")

    np.save(raw_path, saliency)

    heatmap_uint8 = (saliency * 255).astype(np.uint8)
    color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path, color)

    image = np.asarray(raw_image.copy())
    overlay = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = np.clip(image * 0.5 + overlay * 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(overlay_path, overlay[:, :, ::-1])

    return raw_path, heatmap_path, overlay_path


def get_preprocess_normalization_stats(preprocess):
    transforms = getattr(preprocess, "transforms", [])
    for transform in transforms:
        if hasattr(transform, "mean") and hasattr(transform, "std"):
            mean = np.asarray(transform.mean, dtype=np.float32)
            std = np.asarray(transform.std, dtype=np.float32)
            return mean, std
    raise ValueError("Could not infer normalization mean/std from preprocess")


def denormalize_image_tensor(image_tensor, preprocess):
    mean, std = get_preprocess_normalization_stats(preprocess)
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    image = image_tensor.detach().cpu().numpy().transpose((1, 2, 0))
    image = std * image + mean
    return np.clip(image, 0, 1)


def save_causal_metric_summary(image_tensor, final_tensor, scores, output_path, mode, class_name, preprocess):
    if mode == "del":
        title = "Deletion game"
        ylabel = "Pixels deleted"
    elif mode == "ins":
        title = "Insertion game"
        ylabel = "Pixels inserted"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    n_steps = len(scores) - 1
    preview_image = denormalize_image_tensor(final_tensor, preprocess)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title(f"{ylabel} 100.0%, P={scores[-1]:.4f}")
    plt.axis("off")
    plt.imshow(preview_image)

    plt.subplot(122)
    plt.plot(np.arange(n_steps + 1) / n_steps, scores)
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 1.05)
    plt.fill_between(np.arange(n_steps + 1) / n_steps, 0, scores, alpha=0.4)
    plt.title(title)
    plt.xlabel(ylabel)
    plt.ylabel(class_name)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def load_imagenet_label_map(index_json):
    folder_to_label, _ = load_dataset_label_map(index_json)
    return folder_to_label


def _normalize_label_entry(entry, default_folder=None):
    if isinstance(entry, dict):
        folder = (
            entry.get("folder")
            or entry.get("folder_name")
            or entry.get("wnid")
            or entry.get("id")
            or default_folder
        )
        label = entry.get("label")
        if label is None:
            label = entry.get("class_index")
        if label is None:
            label = entry.get("target")
        class_name = (
            entry.get("class_name")
            or entry.get("classname")
            or entry.get("name")
            or entry.get("class")
        )
        if folder is None or label is None:
            return None
        return str(folder), int(label), (str(class_name) if class_name is not None else None)

    if isinstance(entry, list):
        # [label, class_name] when key is folder OR [folder, label, class_name]
        if default_folder is not None and len(entry) >= 1:
            label = entry[0]
            class_name = entry[1] if len(entry) >= 2 else None
            return str(default_folder), int(label), (str(class_name) if class_name is not None else None)
        if len(entry) >= 2:
            folder = entry[0]
            label = entry[1]
            class_name = entry[2] if len(entry) >= 3 else None
            return str(folder), int(label), (str(class_name) if class_name is not None else None)

    if isinstance(entry, int) and default_folder is not None:
        return str(default_folder), int(entry), None

    return None


def load_dataset_label_map(index_json):
    with open(index_json, "r", encoding="utf-8") as f:
        class_dict = json.load(f)

    if not isinstance(class_dict, (dict, list)) or len(class_dict) == 0:
        raise ValueError(f"Invalid label json format: {index_json}")

    folder_to_label = {}
    label_to_classname = {}

    if isinstance(class_dict, dict):
        sample_key = next(iter(class_dict.keys()))

        if str(sample_key).isdigit():
            # Format: {"0": ["n01440764", "tench"], ...}
            for label_str, values in class_dict.items():
                if not isinstance(values, list) or len(values) < 1:
                    continue
                folder = str(values[0])
                label = int(label_str)
                class_name = str(values[1]) if len(values) >= 2 else None
                folder_to_label[folder] = label
                if class_name:
                    label_to_classname[label] = class_name
            return folder_to_label, label_to_classname

        # Generic dict formats
        # 1) {"n01440764": [0, "tench"], ...}
        # 2) {"n01440764": {"label":0, "class_name":"tench"}, ...}
        # 3) {"items": [{"folder":"...","label":0,"class_name":"..."}, ...]}
        if "items" in class_dict and isinstance(class_dict["items"], list):
            entries = class_dict["items"]
            for entry in entries:
                parsed = _normalize_label_entry(entry)
                if parsed is None:
                    continue
                folder, label, class_name = parsed
                folder_to_label[folder] = label
                if class_name:
                    label_to_classname[label] = class_name
            if folder_to_label:
                return folder_to_label, label_to_classname

        for folder_key, values in class_dict.items():
            parsed = _normalize_label_entry(values, default_folder=folder_key)
            if parsed is None:
                continue
            folder, label, class_name = parsed
            folder_to_label[folder] = label
            if class_name:
                label_to_classname[label] = class_name
        if folder_to_label:
            return folder_to_label, label_to_classname

    if isinstance(class_dict, list):
        # Generic list format: [{"folder":"...","label":0,"class_name":"..."}, ...]
        for entry in class_dict:
            parsed = _normalize_label_entry(entry)
            if parsed is None:
                continue
            folder, label, class_name = parsed
            folder_to_label[folder] = label
            if class_name:
                label_to_classname[label] = class_name
        if folder_to_label:
            return folder_to_label, label_to_classname

    raise ValueError(f"Could not parse label mapping from: {index_json}")


def build_classnames_from_label_map(label_to_classname):
    if not label_to_classname:
        return []
    max_label = max(label_to_classname.keys())
    classnames = []
    for label in range(max_label + 1):
        if label not in label_to_classname:
            return []
        classnames.append(label_to_classname[label])
    return classnames


def collect_image_items(data_path, folder_to_label, max_images=None):
    items = []
    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder not in folder_to_label:
            continue

        gt_label = folder_to_label[folder]
        for name in sorted(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, name)
            if os.path.isfile(image_path):
                rel_path = os.path.relpath(image_path, data_path).replace("\\", "/")
                items.append((image_path, rel_path, folder, gt_label))
                if max_images is not None and len(items) >= max_images:
                    return items
    return items


def batched(sequence, batch_size):
    for start in range(0, len(sequence), batch_size):
        yield sequence[start : start + batch_size]
        
        
def denorm_ImageNet1k(x):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=x.device).view(1,3,1,1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=x.device).view(1,3,1,1)
    return (x * std + mean).clamp(0,1)

def normalize_ImageNet1k(x):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=x.device).view(1,3,1,1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=x.device).view(1,3,1,1)
    return (x - mean) / std
        
        
        

 

