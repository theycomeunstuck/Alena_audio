"""Product-side provider client builder (ported from research_memory.__main__)."""

import json
from pathlib import Path

from .spiderssense import SpiderssenseClient


def _provider_client(path, model=None):
    from .spiderssense import SpiderssenseClient

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"model", "input_usd_per_million", "output_usd_per_million", "pricing_verified",
               "pricing_source", "ledger_path", "max_cost_usd", "max_calls", "max_output_tokens",
               "token_field", "key_file", "use_json_schema", "use_json_object", "timeout", "reasoning_effort"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unexpected provider configuration fields; credentials belong in the local key file")
    if model is not None and model != config.get("model"):
        raise ValueError("A different provider model requires its own verified pricing configuration")
    return SpiderssenseClient(**config)
