"""Attenuable capability tokens for a request's budget.

The originating user signs a budget for the request (delegation depth, end-to-end deadline, cost, and finer cost
caveats); every enforcement point verifies it; each hop may only tighten a limit, and all later hops inherit the
tightened value. Construction follows Biscuit tokens (offline attenuation with ephemeral keys):

  block 0   signed by the user's key             caveats: max_depth=8, deadline=t, max_cost_usd=0.01, ...
  block 1   signed with block 0's next-key       caveats: max_depth=7, max_cost_usd=0.0093       (an A2A point)
  block 2   signed with block 1's next-key       caveats: max_llm_calls=3                         (an agent)
  ...
  proof     the private half of the last block's next-key

Whoever holds the token can append a block (it holds the proof), so attenuation needs no cooperation and no
identity: an enforcement point, an agent, or nobody. Nobody can remove a block, edit one, or loosen a limit:
verification recomputes the effective budget as the minimum over the chain and rejects any block that tries to raise
a limit. Blocks are identified by hash, so a metered resource can charge a call to every block on the chain
(hierarchical counters): a tightened limit then bounds the whole subtree below the block that set it, and the user's
block 0 bounds the whole request.
"""
import base64, hashlib, json, time
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

HEADER = "X-Capability-Token"               # model API calls
META_KEY = "vecturae/budget-token"          # A2A message metadata and MCP tools/call params._meta
# numeric limits: a block may set any of them to a value <= the inherited one
LIMITS = ("max_depth", "max_cost_usd", "max_llm_calls", "max_model_tokens", "max_tool_calls")
TIME_LIMITS = ("deadline",)   # absolute unix time; tightening = earlier


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _pub(key: Ed25519PrivateKey) -> str:
    return _b64(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def _priv_bytes(key: Ed25519PrivateKey) -> str:
    return _b64(key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()))


def block_id(block: dict) -> str:
    return hashlib.sha256(_canonical(block["payload"]) + block["sig"].encode()).hexdigest()[:24]


class Identity:
    """A long-term Ed25519 key, e.g. the originating user's. Its public key is what enforcement points trust."""

    def __init__(self):
        self._key = Ed25519PrivateKey.generate()
        self.public_key = _pub(self._key)

    def sign(self, payload: dict) -> str:
        return _b64(self._key.sign(_canonical(payload)))


def mint(user: Identity, request_id: str, caveats: dict, by: str = "user") -> dict:
    """The user's signed budget for a request (block 0)."""
    nxt = Ed25519PrivateKey.generate()
    payload = {"request_id": request_id, "root": user.public_key, "caveats": _clean(caveats), "by": by,
               "parent": None, "next_key": _pub(nxt)}
    return {"blocks": [{"payload": payload, "sig": user.sign(payload)}], "proof": _priv_bytes(nxt)}


def attenuate(token: dict, caveats: dict, by: str) -> dict:
    """Append a block that tightens the budget. Values looser than the inherited ones are clamped to them."""
    eff = effective(token)
    tight = {}
    for k, v in _clean(caveats).items():
        tight[k] = min(v, eff[k]) if k in eff else v
    signer = Ed25519PrivateKey.from_private_bytes(_unb64(token["proof"]))
    nxt = Ed25519PrivateKey.generate()
    payload = {"request_id": token["blocks"][0]["payload"]["request_id"], "caveats": tight, "by": by,
               "parent": block_id(token["blocks"][-1]), "next_key": _pub(nxt)}
    sig = _b64(signer.sign(_canonical(payload)))
    return {"blocks": token["blocks"] + [{"payload": payload, "sig": sig}], "proof": _priv_bytes(nxt)}


def effective(token: dict) -> dict:
    """The budget in force at the end of the chain: the tightest value of every limit (no verification)."""
    out = {}
    for b in token["blocks"]:
        for k, v in b["payload"]["caveats"].items():
            out[k] = min(v, out[k]) if k in out else v
    return out


def verify(token: dict, trusted_roots, request_id: Optional[str] = None, now: Optional[float] = None):
    """Verify the whole chain. Returns (True, {"caveats": effective budget, "chain": [block ids], "root": key})
    or (False, reason). Expired deadlines are reported as a reason too."""
    try:
        blocks = token["blocks"]
        b0 = blocks[0]["payload"]
    except (KeyError, IndexError, TypeError):
        return False, "malformed token"
    if b0.get("root") not in trusted_roots:
        return False, "block 0 is not signed by a user this enforcement point serves"
    if request_id is not None and b0["request_id"] != request_id:
        return False, f"token is for request {b0['request_id']}, not {request_id}"
    signer, eff, chain = b0["root"], {}, []
    for i, block in enumerate(blocks):
        p = block["payload"]
        try:
            Ed25519PublicKey.from_public_bytes(_unb64(signer)).verify(_unb64(block["sig"]), _canonical(p))
        except (InvalidSignature, ValueError):
            return False, f"block {i}: bad signature"
        if i and (p["parent"] != chain[-1] or p["request_id"] != b0["request_id"]):
            return False, f"block {i}: does not extend block {i - 1} (spliced, reordered or from another request)"
        for k, v in p["caveats"].items():
            if k not in LIMITS + TIME_LIMITS:
                return False, f"block {i}: unknown caveat {k!r}"
            if k in eff and v > eff[k]:
                return False, f"block {i}: loosens {k} from {eff[k]} to {v} (limits may only tighten)"
            eff[k] = min(v, eff[k]) if k in eff else v
        chain.append(block_id(block))
        signer = p["next_key"]
    try:
        if _pub(Ed25519PrivateKey.from_private_bytes(_unb64(token["proof"]))) != signer:
            return False, "proof does not match the last block"
    except (KeyError, ValueError):
        return False, "missing or bad proof"
    if "deadline" in eff and (now or time.time()) > eff["deadline"]:
        return False, "deadline passed"
    return True, {"caveats": eff, "chain": chain, "root": b0["root"]}


def encode(token: dict) -> str:
    """Compact form for an HTTP header."""
    return _b64(json.dumps(token, separators=(",", ":")).encode())


def decode(text: str) -> dict:
    return json.loads(_unb64(text))


def describe(token: dict) -> list:
    """One row per block: who added it and what it tightened."""
    return [{"block": i, "id": block_id(b), "by": b["payload"]["by"], **b["payload"]["caveats"]}
            for i, b in enumerate(token["blocks"])]


def _clean(caveats: dict) -> dict:
    return {k: v for k, v in (caveats or {}).items() if v is not None}
