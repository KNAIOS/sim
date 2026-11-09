"""Read-only isolation/import check; stdout is also the reproducibility record."""
import importlib.metadata as metadata
import json
from pathlib import Path
import site
import sys
import torch
from sionna.phy.channel.tr38901 import UMi, PanelArray

root = Path(__file__).resolve().parent
assert Path(sys.prefix).resolve() == root/'.venv'
assert not site.ENABLE_USER_SITE
assert 'include-system-site-packages = false' in (root/'.venv/pyvenv.cfg').read_text()
torch.set_num_threads(1)
print(json.dumps(dict(python=sys.version, prefix=sys.prefix,
                     packages={p:metadata.version(p) for p in
                               ('torch','sionna-no-rt','numpy','scipy','matplotlib','h5py')},
                     torch_file=torch.__file__, cuda=torch.version.cuda,
                     user_site_enabled=site.ENABLE_USER_SITE,
                     import_test='UMi and PanelArray OK'), indent=2))
