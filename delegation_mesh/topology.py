"""Deployment configs: organizations, their agents, frameworks and roles, and the edges between agents.

A config describes the whole testbed (see deployments/*.yaml):

    organizations:
      orgA:
        gateway: {tools: [tcp-reference]}         # the org's metered resources (its credentials)
        agents:
          planner:    {framework: langgraph, role: planner}
          researcher: {framework: adk, role: researcher, tools: [tcp-reference]}
    edges:                                          # who hands work to whom (within and across orgs)
      - planner -> researcher
      - researcher -> [analyst, critic]             # fan-out: both, in parallel
      - [analyst, critic] -> writer                 # join: the writer waits for both
    entry: planner                                  # receives the user's request
    revise: reviewer -> planner                     # optional: where a reviewer sends work back

Every agent reads its own part of the config: the kinds of work it accepts (one per predecessor), the agents it hands
its output to (its successors; more than one = fan-out) and the inputs it must wait for (more than one = join).
Agents still find each other's addresses over mDNS; the config only says who the targets are.
"""
import os
from typing import Optional

import yaml

FRAMEWORKS = {   # framework -> runner script, framework name, slug, native limit
    "langgraph":     {"script": "planner_langgraph.py", "framework": "LangGraph", "slug": "langgraph"},
    "adk":           {"script": "researcher_adk.py", "framework": "Google ADK", "slug": "adk"},
    "crewai":        {"script": "analyst_crewai.py", "framework": "CrewAI", "slug": "crewai"},
    "openai-agents": {"script": "writer_openai_agents.py", "framework": "OpenAI Agents SDK", "slug": "openai-agents"},
    "autogen":       {"script": "reviewer_autogen.py", "framework": "AutoGen", "slug": "autogen"},
}
REQUEST, REVISION = "request", "revision"


def _edge(spec) -> tuple:
    """'a -> b', 'a -> [b, c]', '[a, b] -> c', or [a, b]  ->  (sources, targets)."""
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        src, dst = spec
    else:
        src, dst = (s.strip() for s in str(spec).split("->"))
    as_list = lambda x: [y.strip() for y in x.strip("[] ").split(",")] if isinstance(x, str) else list(x)
    return as_list(src), as_list(dst)


class Topology:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        with open(path) as f:
            self.raw = yaml.safe_load(f)
        self.orgs = self.raw.get("organizations") or {}
        self.agents = {}                                  # agent id -> its spec, with "org"
        for org, o in self.orgs.items():
            for aid, a in (o.get("agents") or {}).items():
                if aid in self.agents:
                    raise SystemExit(f"agent id {aid!r} used twice (ids must be unique across organizations)")
                self.agents[aid] = dict(a, id=aid, org=org)
        self.successors = {a: [] for a in self.agents}
        self.predecessors = {a: [] for a in self.agents}
        for spec in self.raw.get("edges") or []:
            src, dst = _edge(spec)
            for s in src:
                for d in dst:
                    self._check(s), self._check(d)
                    self.successors[s].append(d)
                    self.predecessors[d].append(s)
        self.entry = self.raw.get("entry")
        self._check(self.entry)
        self.revise = _edge(self.raw["revise"]) if self.raw.get("revise") else None
        if self.revise:
            for a in self.revise[0] + self.revise[1]:
                self._check(a)
        for aid, a in self.agents.items():
            if a.get("framework") not in FRAMEWORKS:
                raise SystemExit(f"agent {aid!r}: framework must be one of {', '.join(FRAMEWORKS)}")

    def _check(self, aid):
        if aid not in self.agents:
            raise SystemExit(f"{self.path}: unknown agent {aid!r}")

    def agent(self, aid: str) -> dict:
        """Everything one agent needs to know about its place in the deployment."""
        self._check(aid)
        a = self.agents[aid]
        accepts = list(self.predecessors[aid])            # kind of work = the id of the agent that produced it
        if aid == self.entry:
            accepts.append(REQUEST)
        if self.revise and aid in self.revise[1]:
            accepts.append(REVISION)
        return {**a, "accepts": accepts, "produces": aid,
                "successors": list(self.successors[aid]),
                "join": list(self.predecessors[aid]) if len(self.predecessors[aid]) > 1 else [],
                "revise_to": list(self.revise[1]) if self.revise and aid in self.revise[0] else [],
                "framework_info": FRAMEWORKS[a["framework"]]}

    def gateway_tools(self, org: str) -> list:
        return list(((self.orgs.get(org) or {}).get("gateway") or {}).get("tools", []))

    def tools(self) -> dict:
        return self.raw.get("tools") or {}

    def describe(self) -> str:
        lines = []
        for org, o in self.orgs.items():
            lines.append(f"{org}: gateway tools={self.gateway_tools(org) or '-'}")
            for aid in o.get("agents") or {}:
                a = self.agent(aid)
                shape = (" [fan-out]" if len(a["successors"]) > 1 else "") + (" [join]" if a["join"] else "")
                lines.append(f"  {aid:<14} {a['framework']:<14} role={a.get('role', 'worker'):<12} "
                             f"→ {', '.join(a['successors']) or '(end)'}{shape}"
                             + ("" if a.get("guarded", o.get("guarded", True)) else "  (no enforcement point)"))
        lines.append(f"entry: {self.entry}" + (f"; revise: {self.revise[0]} -> {self.revise[1]}" if self.revise else ""))
        return "\n".join(lines)


def load(path: Optional[str]) -> Optional[Topology]:
    return Topology(path) if path else None
