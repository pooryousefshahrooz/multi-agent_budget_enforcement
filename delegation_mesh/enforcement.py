"""Shared logic of the enforcement points: whom to trust, how to charge, how to decide.

An enforcement point is a component the operator controls and the agents do not: a metered resource (the model API
and paid tools behind gateway.py, which hold the user's credentials) or a protocol boundary (the A2A point in
front of an agent, a2a_guard.py). Each one verifies the request's capability token, charges the request, and refuses
or tightens.

Hierarchical counters: a charge made under a token is added to *every* block of its chain. A limit set in block k
therefore bounds everything charged below block k (the subtree that inherited it), and the user's block 0 bounds
the whole request, however the work fans out.
"""
import json, os, threading, time
from collections import defaultdict
from typing import Optional

import budget_token as bt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ORG = "default"


def trust_file(org: str = DEFAULT_ORG) -> str:
    """The list of users an organization serves (its accounts). One file per organization."""
    if os.getenv("MESH_TRUST_FILE"):
        return os.environ["MESH_TRUST_FILE"]
    return os.path.join(HERE, ".trusted_users.json" if org == DEFAULT_ORG else f".trusted_users.{org}.json")


def known_orgs() -> list:
    """Organizations that have a trust file in this testbed (the default one included)."""
    orgs = [DEFAULT_ORG]
    for name in sorted(os.listdir(HERE)):
        if name.startswith(".trusted_users.") and name.endswith(".json") and name.count(".") == 3:
            orgs.append(name.split(".")[2])
    return orgs


TRUST_FILE = trust_file(os.getenv("MESH_ORG", DEFAULT_ORG))
os.environ.setdefault("VECTURAE_TRUST_FILE", TRUST_FILE)   # patched frameworks read their organization's users
COST_LIMITS = ("max_cost_usd", "max_llm_calls", "max_model_tokens", "max_tool_calls")
CHARGE_KEY = {"max_cost_usd": "cost_usd", "max_llm_calls": "llm_calls", "max_model_tokens": "model_tokens",
              "max_tool_calls": "tool_calls"}


class TrustStore:
    """Public keys of the users whose credentials this operator holds (stand-in for account registration)."""

    def __init__(self, path: str = TRUST_FILE):
        self.path, self._keys, self._mtime = path, set(), None

    def keys(self) -> set:
        try:
            m = os.path.getmtime(self.path)
            if m != self._mtime:
                with open(self.path) as f:
                    self._keys, self._mtime = set(json.load(f)), m
        except (FileNotFoundError, ValueError):
            pass
        return self._keys

    def register(self, public_key: str):
        keys = self.keys() | {public_key}
        with open(self.path, "w") as f:
            json.dump(sorted(keys), f)


class Pricing:
    """What the user's credentials are charged (nominal prices: the local model is free)."""

    def __init__(self, usd_per_1m_input=0.15, usd_per_1m_output=0.60, usd_per_tool_call=0.0005):
        self.inp, self.out, self.tool = usd_per_1m_input / 1e6, usd_per_1m_output / 1e6, usd_per_tool_call

    def model_call(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * self.inp + completion_tokens * self.out


class HierarchicalLedger:
    """Charges per block id; a block's usage includes everything charged below it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.used = defaultdict(lambda: defaultdict(float))       # block id -> charge key -> amount
        self.limits = {}                                          # block id -> its own caveats
        self.by = {}                                              # block id -> who added it
        self.requests = defaultdict(list)                         # request id -> block ids seen, in order
        self.events = defaultdict(list)                           # request id -> refusals and charges

    def _learn(self, token: dict, chain: list):
        rid = token["blocks"][0]["payload"]["request_id"]
        for b, bid in zip(token["blocks"], chain):
            if bid not in self.limits:
                self.limits[bid], self.by[bid] = b["payload"]["caveats"], b["payload"]["by"]
                self.requests[rid].append(bid)
        return rid

    def remaining(self, chain: list) -> dict:
        """Tightest remaining amount of every cost limit along the chain."""
        rem = {}
        for bid in chain:
            for k, lim in self.limits.get(bid, {}).items():
                if k in COST_LIMITS:
                    left = lim - self.used[bid][CHARGE_KEY[k]]
                    rem[k] = min(left, rem[k]) if k in rem else left
        return rem

    def reserve(self, token: dict, chain: list, fixed: dict, per_unit: Optional[dict] = None,
                max_units: int = 0) -> tuple:
        """Atomically check and reserve, so concurrent calls cannot both pass the check (no check-then-act race).

        Reserves `fixed` (e.g. one LLM call and the prompt's tokens) on every block of the chain, plus as many
        `units` as fit (each costing `per_unit`, e.g. one completion token), at most `max_units`. Only limits the
        call would consume can refuse it. Returns (ok, reason, units, clamped): `clamped` says whether some limit
        actually bounded the units. Settle afterwards with settle(chain, reserved, actual)."""
        per_unit = per_unit or {}
        with self._lock:
            self._learn(token, chain)
            rem, units, clamped = self.remaining(chain), max_units, False
            for k, left in rem.items():
                key = CHARGE_KEY[k]
                if key in fixed and (left <= 0 or left < fixed[key]):
                    return False, f"{k} exhausted (remaining {round(left, 6)})", 0, False
                if per_unit.get(key):
                    units, clamped = min(units, int((left - fixed.get(key, 0)) / per_unit[key])), True
            if per_unit and clamped and units < 1:
                return False, "budget exhausted: no room for a completion", 0, True
            if not clamped:
                units = 0   # nothing to bound: reserve only the fixed part
            self._add(chain, {k: fixed.get(k, 0) + units * per_unit.get(k, 0) for k in set(fixed) | set(per_unit)})
            return True, "", units, clamped

    def settle(self, chain: list, reserved: dict, actual: dict, who: str = ""):
        """Replace a reservation by what was actually used (refunds the difference)."""
        with self._lock:
            self._add(chain, {k: actual.get(k, 0) - reserved.get(k, 0) for k in set(reserved) | set(actual)})
            self.events[self._rid_of(chain[0])].append({"t": time.time(), "who": who, **actual})

    def _add(self, chain: list, amounts: dict):
        for bid in chain:
            for k, v in amounts.items():
                self.used[bid][k] += v

    def refuse(self, chain: list, who: str, reason: str):
        with self._lock:
            self.events[self._rid_of(chain[0])].append({"t": time.time(), "who": who, "refused": reason})

    def _rid_of(self, bid):
        return next((rid for rid, ids in self.requests.items() if bid in ids), "?")

    def report(self, rid: str) -> list:
        """Per block: who set it, its limits, and what was charged at or below it."""
        with self._lock:
            return [{"block": i, "id": bid, "by": self.by[bid], "limits": self.limits[bid],
                     "charged": {k: round(v, 6) for k, v in self.used[bid].items()}}
                    for i, bid in enumerate(self.requests.get(rid, []))]


def check_token(header_value: str, trust: TrustStore):
    """Decode and verify a token from a request header. Returns (token, info) or (None, reason)."""
    if not header_value:
        return None, "no capability token"
    try:
        token = bt.decode(header_value)
    except Exception:
        return None, "undecodable capability token"
    ok, info = bt.verify(token, trust.keys())
    return (token, info) if ok else (None, info)
