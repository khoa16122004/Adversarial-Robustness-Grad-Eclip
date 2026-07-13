import time
import torch
import Game_MM_CLIP.clip as mm_clip
import cv2
import numpy as np
from PIL import Image

import torch.nn.functional as F
from generate_emap import CLIPExplainRunner


_EXPLAINER_CACHE = {}


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