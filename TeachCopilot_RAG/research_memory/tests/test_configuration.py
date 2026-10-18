import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_memory.__main__ import _provider_client


class ProviderConfigurationTests(unittest.TestCase):
    def test_model_override_cannot_reuse_another_models_price(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provider.json"
            path.write_text(json.dumps({"model": "priced-model"}), encoding="utf-8")
            with patch("research_memory.spiderssense.SpiderssenseClient") as client:
                with self.assertRaisesRegex(ValueError, "verified pricing"):
                    _provider_client(path, "another-model")
                client.assert_not_called()

    def test_credentials_are_not_accepted_in_provider_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provider.json"
            path.write_text(json.dumps({"model": "priced-model", "api_key": "FAKE_SECRET"}), encoding="utf-8")
            with patch("research_memory.spiderssense.SpiderssenseClient") as client:
                with self.assertRaises(ValueError) as error:
                    _provider_client(path)
                self.assertNotIn("FAKE_SECRET", str(error.exception))
                client.assert_not_called()
