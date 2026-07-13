import time
import torch
import Game_MM_CLIP.clip as mm_clip
import cv2
import numpy as np
from PIL import Image

import torch.nn.functional as F

from generate_emap import preprocess, imgprocess_keepsize, mm_clipmodel, mm_interpret, \
        clip_encode_dense, grad_eclip, grad_cam, mask_clip, compute_rollout_attention, \
        surgery_model, clip_surgery_map, m2ib_model, m2ib_clip_map, \
        generate_masks, rise
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
    img_keepsized = imgprocess_keepsize(img).to(device).unsqueeze(0)
def generate_hm(clipmodel, hm_type, img, txt_embedding, txts, resize, preprocess):
    explainer = _get_explainer(clipmodel, preprocess)
    return explainer.generate_hm(hm_type, img, txt_embedding, txts, resize)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    c_ret = np.clip(image * (1 - 0.5) + color * 0.5, 0, 255).astype(np.uint8)
    return c_ret