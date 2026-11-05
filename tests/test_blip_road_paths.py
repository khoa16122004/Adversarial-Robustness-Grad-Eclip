import unittest
from pathlib import Path

from road_score_BLIP import attack_artifacts


class BLIPRoadPathTests(unittest.TestCase):
    def test_post_attack_artifacts(self):
        sample = Path("sample")
        for mode, image, saliency in (
            ("ins", "adversarial_image_ins.png", "ins_del_adv_eclip_saliency.npy"),
            ("del", "adversarial_image_del.png", "del_del_adv_eclip_saliency.npy"),
            ("del+ins", "adversarial_image_del_ins.png", "joint_del_adv_eclip_saliency.npy"),
            ("xshift", "adversarial_image_xshift.png", "del_xshift_eclip_saliency.npy"),
        ):
            with self.subTest(mode=mode):
                self.assertEqual(
                    attack_artifacts(sample, mode, "eclip"),
                    (sample / image, sample / saliency),
                )

    def test_unknown_mode_fails(self):
        with self.assertRaises(ValueError):
            attack_artifacts(Path("sample"), "clean", "eclip")
