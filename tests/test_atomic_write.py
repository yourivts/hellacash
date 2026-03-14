# tests/test_atomic_write.py
"""Tests for atomic paper state persistence."""
from __future__ import annotations

import json
import os
import pytest


class TestAtomicWrite:
    def test_save_creates_valid_json(self, tmp_path, monkeypatch):
        state_file = str(tmp_path / "paper_state.json")
        monkeypatch.setattr("bot.exchange.bitvavo_client._PAPER_STATE_FILE", state_file)

        from bot.exchange.bitvavo_client import BitvavoClient
        client = BitvavoClient("", "", paper_trading=True)
        client._paper_balance = {"EUR": 9500.0, "BTC": 0.5}
        client._save_paper_state()

        with open(state_file) as f:
            data = json.load(f)
        assert data["EUR"] == 9500.0
        assert data["BTC"] == 0.5

    def test_no_tmp_file_left_after_save(self, tmp_path, monkeypatch):
        state_file = str(tmp_path / "paper_state.json")
        monkeypatch.setattr("bot.exchange.bitvavo_client._PAPER_STATE_FILE", state_file)

        from bot.exchange.bitvavo_client import BitvavoClient
        client = BitvavoClient("", "", paper_trading=True)
        client._save_paper_state()

        tmp_file = state_file + ".tmp"
        assert not os.path.exists(tmp_file)
