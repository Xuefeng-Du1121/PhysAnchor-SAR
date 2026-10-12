from pathlib import Path

import pytest

from peva_sim.protocol import resolve_field_path


def test_recorded_path_is_used_when_present(tmp_path):
    f = tmp_path / "a.nc"
    f.write_bytes(b"x")
    assert resolve_field_path(f, tmp_path / "elsewhere") == f.resolve()


def test_recorded_path_is_relocated_below_data_root(tmp_path):
    root = tmp_path / "bundle" / "data"
    f = root / "raw" / "environment" / "copernicus" / "field.nc"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    recorded = "/home/someone/old/data_platform/data/raw/environment/copernicus/field.nc"
    assert resolve_field_path(recorded, root) == f.resolve()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_field_path("/nowhere/raw/field.nc", tmp_path)
