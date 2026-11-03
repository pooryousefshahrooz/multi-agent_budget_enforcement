"""Which users' signatures a framework-level enforcement point accepts as the root of a budget."""
import json
import os
from typing import Optional


def trusted_roots(token: Optional[dict] = None) -> set:
    """Public keys of the users the operator serves (JSON list in $VECTURAE_TRUST_FILE).

    Without a trust file, a framework can still check the chain's integrity and that no block loosens a limit,
    so it trusts the token's own root; authenticating the user stays with the operator's enforcement points."""
    path = os.getenv("VECTURAE_TRUST_FILE")
    if path and os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    if token and token.get("blocks"):
        return {token["blocks"][0]["payload"].get("root")}
    return set()
