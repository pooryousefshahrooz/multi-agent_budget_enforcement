"""vecturae-core: the shared budget-token logic used by every enforcement point and every patched framework."""
from vecturae.token import (HEADER, META_KEY, LIMITS, TIME_LIMITS, Identity, attenuate, block_id, decode, describe,
                            effective, encode, mint, verify)
from vecturae.trust import trusted_roots

# the key under which a run's config carries the token in patched frameworks
CONFIG_KEY = "vecturae_token"


class BudgetLimitError(RuntimeError):
    """A request's budget forbids continuing: `kind` is depth, deadline or rejected (invalid token)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
