import pytest
import torch

# Check for 32GB+ VRAM once at module load time
has_32gb = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 32e9


@pytest.fixture(autouse=True)
def set_torch_seed() -> None:
    torch.manual_seed(42)
