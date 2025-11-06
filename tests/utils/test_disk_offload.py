"""Tests for disk_offload module functions."""

import pytest
import torch

from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from circuit_tracer.utils.disk_offload import (
    cleanup_all_offload_files,
    cpu_offload_module,
    disk_offload_module,
    offload_modules,
)


@pytest.fixture
def clt_module():
    """Create a small CLT."""
    return CrossLayerTranscoder(
        n_layers=2,
        d_transcoder=16,
        d_model=8,
        lazy_decoder=False,
        lazy_encoder=False,
        device=torch.device("cpu"),
    )


@pytest.fixture
def plt_module():
    """Create a small PLT."""
    return SingleLayerTranscoder(
        d_model=8,
        d_transcoder=16,
        activation_function=torch.nn.functional.relu,
        layer_idx=0,
        lazy_decoder=False,
        lazy_encoder=False,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("module_fixture", ["clt_module", "plt_module"])
@pytest.mark.parametrize("explicit_device", [True, False])
def test_disk_offload_module(module_fixture, explicit_device, request):
    """Test disk offload with CLT and PLT architectures."""
    module = request.getfixturevalue(module_fixture)

    # Store original state
    orig_param = next(module.parameters()).data.clone()
    orig_device = next(module.parameters()).device

    # Offload to disk
    reload_handle = disk_offload_module(module)

    # Verify module is on meta device
    assert next(module.parameters()).device.type == "meta"

    # Reload with or without explicit device
    if explicit_device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reload_handle(device=device)
        # Should be on the explicitly requested device
        assert next(module.parameters()).device.type == device.type
        assert torch.allclose(next(module.parameters()).data, orig_param.to(device))
    else:
        reload_handle()
        # Should be restored to original device
        assert next(module.parameters()).device.type == orig_device.type
        assert torch.allclose(next(module.parameters()).data, orig_param)


@pytest.mark.parametrize("module_fixture", ["clt_module", "plt_module"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_offload_module_cuda(module_fixture, request):
    """Test CPU offload with CLT and PLT on CUDA."""
    module = request.getfixturevalue(module_fixture)

    # Move to CUDA
    module.to("cuda")
    orig_param = next(module.parameters()).data.clone()

    # Offload to CPU
    reload_handle = cpu_offload_module(module)
    assert next(module.parameters()).device.type == "cpu"

    # Reload to CUDA
    reload_handle()
    assert next(module.parameters()).device.type == "cuda"
    assert torch.allclose(next(module.parameters()).data, orig_param.to("cuda"))


def test_cpu_offload_module_cpu(clt_module):
    """Test CPU offload when already on CPU."""
    orig_device = next(clt_module.parameters()).device

    reload_handle = cpu_offload_module(clt_module)
    assert next(clt_module.parameters()).device.type == "cpu"

    reload_handle()
    assert next(clt_module.parameters()).device == orig_device


@pytest.mark.parametrize(
    "modules_factory,expected_count",
    [
        # Single module
        (
            lambda: CrossLayerTranscoder(
                n_layers=2, d_transcoder=16, d_model=8, lazy_decoder=False, lazy_encoder=False
            ),
            1,
        ),
        # List of CLTs
        (
            lambda: [
                CrossLayerTranscoder(
                    n_layers=2, d_transcoder=16, d_model=8, lazy_decoder=False, lazy_encoder=False
                ),
                CrossLayerTranscoder(
                    n_layers=2, d_transcoder=16, d_model=8, lazy_decoder=False, lazy_encoder=False
                ),
            ],
            2,
        ),
        # ModuleDict with CLTs
        (
            lambda: torch.nn.ModuleDict(
                {
                    "clt1": CrossLayerTranscoder(
                        n_layers=2,
                        d_transcoder=16,
                        d_model=8,
                        lazy_decoder=False,
                        lazy_encoder=False,
                    ),
                    "clt2": CrossLayerTranscoder(
                        n_layers=2,
                        d_transcoder=16,
                        d_model=8,
                        lazy_decoder=False,
                        lazy_encoder=False,
                    ),
                }
            ),
            2,
        ),
    ],
    ids=["single_clt", "list_clt", "moduledict_clt"],
)
@pytest.mark.parametrize("offload_type", ["cpu", "disk"])
def test_offload_modules(modules_factory, expected_count, offload_type):
    """Test offload_modules with various container types using CLT architecture."""
    modules = modules_factory()
    expected_device = "cpu" if offload_type == "cpu" else "meta"

    handles = offload_modules(modules, offload_type=offload_type)

    # Verify handles
    assert isinstance(handles, list)
    assert len(handles) == expected_count
    for handle in handles:
        assert callable(handle)

    # Verify modules are offloaded
    if isinstance(modules, torch.nn.Module) and not isinstance(
        modules, (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)
    ):
        assert next(modules.parameters()).device.type == expected_device
    else:
        module_iter = modules.values() if isinstance(modules, torch.nn.ModuleDict) else modules
        for module in module_iter:
            assert next(module.parameters()).device.type == expected_device

    # Cleanup disk offloads
    if offload_type == "disk":
        for handle in handles:
            handle()


def test_cleanup_offload_files(clt_module):
    """Test cleanup removes offload files."""
    # Create some offload files
    modules = [clt_module]
    offload_modules(modules, offload_type="disk")

    # Cleanup
    n_removed = cleanup_all_offload_files()

    # Should have removed files
    assert n_removed >= 1


def test_cleanup_when_no_files():
    """Test cleanup when no offload files exist."""
    # First cleanup any existing files
    cleanup_all_offload_files()

    # Second cleanup should find nothing
    n_removed = cleanup_all_offload_files()
    assert n_removed == 0
