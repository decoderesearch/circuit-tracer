import subprocess
import sys

import pytest
import torch

from circuit_tracer.utils import get_default_device


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_get_default_device_prefers_cuda_then_mps(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    assert get_default_device() == torch.device(expected)


@pytest.mark.parametrize(
    "module",
    ["circuit_tracer.utils", "circuit_tracer.utils.create_graph_files", "circuit_tracer.graph"],
)
def test_utils_import_has_no_cycle(module):
    # Each import runs in a fresh interpreter, so import order can't hide a cycle.
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
