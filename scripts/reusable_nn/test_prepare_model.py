# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import unittest
import torch
from torch import nn
from prepare_model import normalize_conv1d


class SharedLayer(nn.Module):
    def __init__(self,groups=1):
        super().__init__()
        self.conv=nn.Conv1d(2,2,3,padding=2,dilation=2,groups=groups)
    def forward(self,x):
        return self.conv(x),self.conv(x*2)


class FrontendTests(unittest.TestCase):
    def test_shared_module_and_original_preserved(self):
        torch.manual_seed(31)
        model=SharedLayer().eval()
        x=torch.randn(1,2,11)
        normalized,records=normalize_conv1d(model)
        self.assertEqual(len(records),2)
        self.assertIsInstance(model.conv,nn.Conv1d)
        self.assertIsInstance(normalized.conv,nn.Conv2d)
        with torch.no_grad():
            for actual,expected in zip(normalized(x),model(x)):
                torch.testing.assert_close(actual,expected)

    def test_grouped_convolution_rejected(self):
        with self.assertRaisesRegex(ValueError,"dense convolution"):
            normalize_conv1d(SharedLayer(groups=2))


if __name__ == "__main__":
    unittest.main()
