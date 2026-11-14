import sys

import pytest

from circuit_tracer import __main__ as cli


def parse_attribute_args(monkeypatch, *extra_args):
    """Parse `circuit-tracer attribute` arguments without loading anything."""
    argv = ["circuit-tracer", "attribute", "-t", "some/transcoders", "-p", "hi", "-o", "g.pt"]
    monkeypatch.setattr(sys, "argv", [*argv, *extra_args])
    captured = {}
    monkeypatch.setattr(cli, "run_attribution", lambda args, parser: captured.update(vars(args)))
    cli.main()
    return captured


@pytest.mark.parametrize(
    "flags, expected",
    [((), True), (("--lazy-decoder",), True), (("--no-lazy-decoder",), False)],
)
def test_lazy_decoder_flag(monkeypatch, flags, expected):
    assert parse_attribute_args(monkeypatch, *flags)["lazy_decoder"] is expected
