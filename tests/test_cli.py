import sys
from unittest import mock

import pytest
import torch

from circuit_tracer import __main__ as cli


@pytest.fixture
def mocks():
    """Mock out model loading, attribution and graph file creation for the CLI."""
    mocked = {
        "load": mock.Mock(return_value=(mock.Mock(), {})),
        "from_pretrained": mock.Mock(),
        "create_graph_files": mock.Mock(),
    }
    with (
        mock.patch("circuit_tracer.utils.hf_utils.load_transcoder_from_hub", mocked["load"]),
        mock.patch(
            "circuit_tracer.ReplacementModel.from_pretrained_and_transcoders",
            mocked["from_pretrained"],
        ),
        mock.patch("circuit_tracer.attribute", mock.Mock()),
        mock.patch(
            "circuit_tracer.utils.create_graph_files.create_graph_files",
            mocked["create_graph_files"],
        ),
        mock.patch("os.makedirs"),
    ):
        yield mocked


def run_attribute(monkeypatch, *extra_args, outputs=("-o", "graph.pt")):
    argv = ["circuit-tracer", "attribute", "-t", "some/transcoders", "-m", "some/model"]
    monkeypatch.setattr(sys, "argv", [*argv, "-p", "hello", *outputs, *extra_args])
    cli.main()


def test_device_defaults_to_none(monkeypatch, mocks):
    run_attribute(monkeypatch)
    assert mocks["load"].call_args.kwargs["device"] is None
    assert mocks["from_pretrained"].call_args.kwargs["device"] is None


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_device_is_passed_to_model_and_transcoders(monkeypatch, mocks, device):
    run_attribute(monkeypatch, "--device", device)
    assert mocks["load"].call_args.kwargs["device"] == torch.device(device)
    assert mocks["from_pretrained"].call_args.kwargs["device"] == torch.device(device)


def test_device_is_passed_to_graph_file_creation(monkeypatch, mocks):
    run_attribute(monkeypatch, "--device", "cpu", outputs=("--slug", "s", "--graph_file_dir", "d"))
    assert mocks["create_graph_files"].call_args.kwargs["device"] == torch.device("cpu")


def test_invalid_device_is_a_usage_error(monkeypatch, mocks, capsys):
    with pytest.raises(SystemExit) as exc:
        run_attribute(monkeypatch, "--device", "not-a-device")
    assert exc.value.code == 2
    assert "invalid device string" in capsys.readouterr().err
    mocks["load"].assert_not_called()
