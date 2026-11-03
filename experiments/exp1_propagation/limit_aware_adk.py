"""Honour the request's limit in upstream ADK at run time, per A2A request (ADK 2.x).

ADK 2.x already hands each A2A request's metadata to the run (``RunConfig.custom_metadata["a2a_metadata"]``), but
nothing enforces it: ``max_llm_calls`` stays at the process-wide default and there is no deadline. A developer needs
only one before-model callback:

* on the run's first model call it sets this run's ``max_llm_calls`` from the request's limit (ADK re-reads it before
  every model call, so the run is then bounded by the user's number, smaller or larger than the default);
* before every model call it refuses calls made after the request's deadline.

Without a limit in the request's metadata, ADK behaves as upstream.
"""
import time


class DeadlineExceeded(RuntimeError):
    pass


def request_limit(run_config) -> dict:
    custom = getattr(run_config, "custom_metadata", None) or {}
    a2a = custom.get("a2a_metadata") or {}
    return dict(a2a.get("limit") or {})


def callback(callback_context, llm_request):
    run_config = callback_context._invocation_context.run_config
    limit = request_limit(run_config)
    if "max_llm_calls" in limit and run_config.max_llm_calls != int(limit["max_llm_calls"]):
        run_config.max_llm_calls = int(limit["max_llm_calls"])
    if limit.get("deadline") and time.time() > float(limit["deadline"]):
        raise DeadlineExceeded("the request's end-to-end deadline passed")
    return None
