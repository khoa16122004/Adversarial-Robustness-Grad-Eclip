import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from RISE.evaluation import CausalMetric


class MeanScore(nn.Module):
    def forward(self, image):
        score = image.mean(dim=(1, 2, 3))
        return torch.stack((score, 1 - score), dim=1)


class CausalMetricPixelUpdateTests(unittest.TestCase):
    def check_curves(self, device):
        image = torch.ones(1, 3, 224, 224, device=device)
        saliency = np.arange(224 * 224).reshape(224, 224)
        for mode, expected in (("del", [1., .5, 0.]), ("ins", [0., .5, 1.])):
            with self.subTest(device=device, mode=mode):
                metric = CausalMetric(MeanScore(), mode, 224 * 112, torch.zeros_like)
                with torch.no_grad():
                    curve = metric.single_run(image, saliency)
                np.testing.assert_allclose(curve, expected)
                torch.testing.assert_close(image, torch.ones_like(image))

    def test_cpu_pixel_updates(self):
        with patch.object(torch.Tensor, "cuda", lambda tensor: tensor):
            self.check_curves("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_pixel_updates(self):
        self.check_curves("cuda")


if __name__ == "__main__":
    unittest.main()
