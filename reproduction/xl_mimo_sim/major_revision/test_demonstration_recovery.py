import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from major_revision.demonstration_recovery import convert_local


class CheckpointRecoveryTests(unittest.TestCase):
    def test_only_metadata_changes_and_strict_reload_works(self):
        with tempfile.TemporaryDirectory(prefix='local-checkpoint-test-') as directory:
            source,destination=Path(directory)/'original.pt',Path(directory)/'portable.pt'
            torch.save(dict(online={'weight':torch.arange(6.)},validation_key=(np.float64(.03),12.)),source)
            original=source.read_bytes()
            receipt=convert_local(source,destination)
            loaded=torch.load(destination,weights_only=True)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(type(loaded['validation_key'][0]),float)
            self.assertTrue(torch.equal(loaded['online']['weight'],torch.arange(6.)))
            self.assertTrue(receipt['tensor_identity'])


if __name__=='__main__':
    unittest.main()
