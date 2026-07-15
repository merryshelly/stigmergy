from stigmergy import __version__
from stigmergy.cli import main


def test_version_present():
    assert __version__ == "0.0.0"


def test_cli_stub_returns_zero():
    assert main([]) == 0
