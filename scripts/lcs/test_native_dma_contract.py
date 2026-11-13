#!/usr/bin/env python3
"""A stale HLS input must not be silently paired with the thin adapter."""
import tempfile
import unittest
from pathlib import Path
from native_dma_contract import record, select

class NativeDMAContractTest(unittest.TestCase):
    def test_stale_hardware_and_stride(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'bambu_syn').mkdir()
            for name in ('05_hw_acc.ll', '05_kernel_riscv.ll', 'architecture.xml'):
                (root/name).write_text('qualified input\n')
            (root/'layout.txt').write_text('dma_word_stride=2\n')
            (root/'bambu_syn/accelerator.ll').write_text('qualified input\n')
            (root/'bambu_syn/architecture.xml').write_text('qualified input\n')
            record(root)
            self.assertTrue(select(root, 'thin').is_file())
            (root/'bambu_syn/accelerator.ll').write_text('stale input\n')
            with self.assertRaisesRegex(ValueError, 'Stale Bambu'):
                select(root, 'thin')
            (root/'layout.txt').write_text('dma_word_stride=1\n')
            record(root)
            with self.assertRaisesRegex(ValueError, 'stride=2'):
                select(root, 'thin')
            self.assertTrue(select(root, 'buffered').is_file())

if __name__ == '__main__':
    unittest.main()
