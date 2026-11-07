import unittest

import torch

from models import mobilenet_v2, shufflenet05, shufflenet10


class ModelLoadersTest(unittest.TestCase):
    def test_winners_have_food11_outputs(self):
        for module, params in ((shufflenet05, 353067), (shufflenet10, 1264879),
                               (mobilenet_v2, 2237963)):
            with self.subTest(model=module.__name__):
                model = module.load_model(11, pretrained=False).eval()
                self.assertEqual(sum(p.numel() for p in model.parameters()), params)
                with torch.no_grad():
                    self.assertEqual(model(torch.randn(1, 3, 128, 128)).shape, (1, 11))


if __name__ == "__main__":
    unittest.main()
