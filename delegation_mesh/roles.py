"""Roles an agent can play, independent of the framework that runs it.

A role is what an agent is asked to do: its sub-task (shown to the next agent's model and in its Agent Card), its system
prompt, and, for the reviewer, how its answer routes the request (approve, or send it back for a revision). Every
framework runner (planner_langgraph.py, researcher_adk.py, ...) can play every role; a deployment config picks the
pairing (see deploy.py). Without a config, each runner plays its default role in the original chain:

    request → plan → research → analysis → report → checked_report → review   (REVISE → revision → planner)
"""
import re

# ---- the original chain's roles (default accepts/produces are used only without a deployment config) ----
ROLES = {
    "planner": {
        "name": "Planner", "accepts": ["request", "revision"], "produces": "plan",
        "subtask": "Break the request into an outline and the key technical questions to research.",
        "system": ("You are the Planner. Produce a short outline (max 120 words) and 3-5 key technical "
                   "questions for the researcher. Do not write the report."),
    },
    "researcher": {
        "name": "Researcher", "accepts": ["plan"], "produces": "research",
        "subtask": "Answer the planner's questions: how Reno and CUBIC grow and cut their window under packet loss.",
        "system": ("You are the Researcher. Explain the mechanics precisely (max 200 words): Reno's AIMD, fast "
                   "retransmit/fast recovery and halving on loss; CUBIC's cubic window growth, beta=0.7 "
                   "reduction and RTT-independence."),
    },
    "analyst": {
        "name": "Analyst", "accepts": ["research"], "produces": "analysis",
        "subtask": "Analyse what the research means for a high-bandwidth, high-latency network and recommend one.",
        "system": ("Analyse throughput recovery after loss on high bandwidth-delay-product links, fairness and "
                   "loss sensitivity, then give a clear, justified recommendation (max 180 words)."),
    },
    "writer": {
        "name": "Writer", "accepts": ["analysis"], "produces": "report",
        "subtask": "Write the final short technical report from the plan, research and analysis.",
        "system": ("You are the Writer. Write a concise technical report (max 350 words, markdown headings): "
                   "overview, Reno under loss, CUBIC under loss, key differences, recommendation for a "
                   "high-bandwidth network with justification."),
    },
    "factchecker": {
        "name": "Fact-checker", "accepts": ["report"], "produces": "checked_report",
        "subtask": "Check the technical claims in the Writer's report and list any that are wrong or unsupported.",
        "system": ("You are the Fact-checker. Check the report's technical claims about TCP Reno and TCP CUBIC "
                   "(window halving on loss, CUBIC's beta=0.7 reduction and cubic growth, RTT-independence, fast "
                   "retransmit/recovery). Reply in max 120 words: 'VERIFIED:' followed by the claims that are "
                   "correct, then 'ISSUES:' followed by any wrong or unsupported claims (or 'none')."),
    },
    "reviewer": {
        "name": "Reviewer", "accepts": ["checked_report"], "produces": "review",
        "subtask": "Review the report against the checklist and approve it or send it back.",
        "system": None,     # set below: depends on the request (noncompliant reviewer experiments)
    },
    "worker": {
        "name": "Worker", "accepts": [], "produces": "work",
        "subtask": "Contribute your part to the request, building on the work you received.",
        "system": ("You are a specialist on a team answering the request. Add what is missing from the work you "
                   "received, concisely (max 150 words). Do not repeat earlier work."),
    },
}

# ---- the reviewer: its answer decides whether the request ends or goes round again ----
SYS_REVIEWER = ("You are the Reviewer. Check the report against this checklist:\n"
                "1. both_algorithms: TCP Reno AND TCP CUBIC are both described.\n"
                "2. packet_loss: each algorithm's behavior under packet loss is explained.\n"
                "3. recommendation: a recommendation for a high-bandwidth network is given AND justified.\n"
                "Reply EXACTLY in this format:\n"
                "both_algorithms: YES|NO\npacket_loss: YES|NO\nrecommendation: YES|NO\n"
                "DECISION: APPROVE|REVISE\nFEEDBACK: <one short paragraph>")
# Experiment D3: a reviewer that never approves (switched on per request by the requester)
SYS_REVIEWER_NONCOMPLIANT = ("You are the Reviewer and you are never satisfied. Always reply with "
                             "DECISION: REVISE and list 3 additional analyses the team must add.")
CHECK_KEYS = ["both_algorithms", "packet_loss", "recommendation"]


def parse_review(text: str):
    checks = {k: bool(re.search(rf"{k}\W*\s*YES", text, re.I)) for k in CHECK_KEYS}
    decision = re.search(r"DECISION\W*\s*(APPROVE|REVISE)", text, re.I)
    approved = bool(decision and decision.group(1).upper() == "APPROVE" and all(checks.values()))
    fb = re.search(r"FEEDBACK\W*\s*(.*)", text, re.I | re.S)
    return approved, (fb.group(1).strip() if fb else text.strip())


def reviewer_system(env: dict) -> str:
    return SYS_REVIEWER_NONCOMPLIANT if env.get("flags", {}).get("noncompliant_reviewer") else SYS_REVIEWER


def reviewer_route(env: dict, output: str):
    approved, feedback = parse_review(output)
    if env.get("flags", {}).get("noncompliant_reviewer"):
        approved = False
    if approved:
        return "done", "completed", env
    if env["round"] >= env["max_rounds"]:
        return "done", "max_rounds", env
    report = next((w["output"] for w in reversed(env["work"]) if w["kind"] in ("report", "writer")
                   or w.get("role") == "writer"), "")
    return "forward", "revision", dict(env, round=env["round"] + 1, work=[], feedback=feedback,
                                       last_report=report)


ROLES["reviewer"]["system"] = reviewer_system
ROLES["reviewer"]["route"] = reviewer_route


def role(name: str) -> dict:
    if name not in ROLES:
        raise SystemExit(f"unknown role {name!r}; known roles: {', '.join(ROLES)}")
    return ROLES[name]
