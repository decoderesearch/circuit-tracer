import atexit
import os
import tempfile
from typing import Literal

from safetensors.torch import load_file, save_file

_offload_files = set()

_TEMP_PREFIX = "safetensors-offload-YqKRr8m3-"


@atexit.register
def cleanup_offload_files():
    for f in _offload_files:
        os.remove(f)


def cleanup_all_offload_files():
    temp_dir = tempfile.gettempdir()
    n_removed = 0
    for f in os.listdir(temp_dir):
        if f.startswith(_TEMP_PREFIX):
            os.remove(os.path.join(temp_dir, f))
            n_removed += 1
    return n_removed


def disk_offload_module(module):
    org_device = next(module.parameters()).device
    with tempfile.NamedTemporaryFile(prefix=_TEMP_PREFIX, delete=False) as f:
        save_file(module.state_dict(), f.name)
        _offload_files.add(f.name)

    module.to(device="meta")

    def reload_handle(device=None):
        module.load_state_dict(load_file(f.name, device=(device or str(org_device))), assign=True)
        os.remove(f.name)
        _offload_files.remove(f.name)

    return reload_handle


def cpu_offload_module(module):
    org_device = next(module.parameters()).device
    module.to(device='cpu')

    def reload_handle():
        module.to(device=org_device)

    return reload_handle


def offload_modules(modules, offload_type: Literal["cpu", "disk"]):
    offload_fn = disk_offload_module if offload_type == "disk" else cpu_offload_module
    return [offload_fn(module) for module in modules]
