"""Testbed configuration: organizations, their framework and agent graph, cross-organization links, MCP servers.

    name: example
    model: {kind: fake}                                  # or {kind: openai-compatible, url: ..., model: ...}
    prices: {usd_per_1m_input: 0.15, usd_per_1m_output: 0.60, usd_per_tool_call: 0.0005}
    mcp_servers:
      tcp-reference: {script: mcp_servers/tcp_reference.py}     # started over HTTP; or {url: http://...}
    organizations:
      orgA:
        framework: langgraph                             # one framework per organization
        agent_limit: 10                                  # operator's per-agent bound (model calls), mapped natively
        agents:
          planner:    {role: "You plan ..."}
          researcher: {role: "You research ...", mcp: [tcp-reference]}
          critic:     {role: "You critique ..."}
          summarizer: {role: "You summarize ..."}
        edges:                                           # any directed acyclic graph inside the organization
          - planner -> [researcher, critic]              # fan-out
          - "[researcher, critic] -> summarizer"         # join
      orgB:                                              # an organization can be a single agent (no edges)
        framework: adk
        agents:
          lead: {role: "You ..."}
    links:                                               # delegation across organizations, over A2A
      - orgA.researcher -> orgB.lead
    entry: orgA.planner                                  # receives the user's request
    request_limit: {max_depth: 3, deadline_s: 600, max_cost_usd: 0.01}   # the user's limit, attached at the origin

Every agent is its own A2A server, and every edge, inside an organization or across, is an A2A message. The agent
that receives a request from outside its organization is that request's first agent: the work flows along the edges
reachable from it, and the last agents report their results back to it, which returns the organization's answer to
the caller. A link from agent X to another organization's agent Y gives X a delegate tool: X's model calls it with a
task, the tool sends the task to Y over A2A, and Y's organization's answer comes back as the tool's result.
"""
import os
from collections import defaultdict

import yaml

FRAMEWORKS = ("langgraph", "adk", "crewai", "openai_agents", "autogen",
              "crewai_a2a", "crewai_a2a_patched",   # CrewAI on its own A2A server: stock, and patched (testbed/patches)
              "openai_agents_patched", "autogen_patched")   # with an A2A server added to the library (testbed/patches)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _side(text: str) -> list:
    text = text.strip()
    return [x.strip() for x in text.strip("[]").split(",")] if text.startswith("[") else [text]


def parse_edge(spec) -> tuple:
    """'a -> b', 'a -> [b, c]', '[a, b] -> c', or [a, b]  ->  ([sources], [targets])."""
    src, dst = spec if isinstance(spec, (list, tuple)) else str(spec).split("->")
    norm = lambda x: [str(y).strip() for y in x] if isinstance(x, (list, tuple)) else _side(str(x))
    return norm(src), norm(dst)


class Config:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        with open(path) as f:
            self.raw = yaml.safe_load(f)
        self.name = self.raw.get("name") or os.path.splitext(os.path.basename(path))[0]
        self.orgs = self.raw["organizations"]
        for org, spec in self.orgs.items():
            if spec.get("framework") not in FRAMEWORKS:
                raise SystemExit(f"{org}: framework must be one of {FRAMEWORKS}")
        self.links = []                                     # (from_org, from_agent, to_org, to_agent)
        for spec in self.raw.get("links") or []:
            src, dst = parse_edge(spec)
            for s in src:
                for d in dst:
                    fo, fa = s.split(".")
                    to, ta = d.split(".")
                    self._check(fo, fa), self._check(to, ta)
                    if fo == to:
                        raise SystemExit(f"link {s} -> {d} stays inside {fo}; put it under that org's edges")
                    self.links.append((fo, fa, to, ta))
        self.entry = tuple(self.raw["entry"].split("."))
        self._check(*self.entry)

    def _check(self, org, agent):
        if org not in self.orgs or agent not in (self.orgs[org].get("agents") or {}):
            raise SystemExit(f"{self.path}: unknown agent {org}.{agent}")

    # ---- one organization's view -------------------------------------------------------------------------
    def edges(self, org: str) -> list:
        """(from, to) inside the org."""
        out = []
        for spec in self.orgs[org].get("edges") or []:
            src, dst = parse_edge(spec)
            for s in src:
                for d in dst:
                    if "." in s or "." in d:
                        raise SystemExit(f"{org}: {s} -> {d} leaves the organization; use a link")
                    self._check(org, s), self._check(org, d)
                    out.append((s, d))
        return out

    def delegates(self, org: str, agent: str) -> list:
        """The other organizations' agents this agent can delegate to: [(to_org, to_agent)]."""
        return [(to, ta) for fo, fa, to, ta in self.links if (fo, fa) == (org, agent)]

    def ingress(self, org: str) -> list:
        """Agents of this org that receive work from outside: the entry, and targets of incoming links."""
        names = [self.entry[1]] if self.entry[0] == org else []
        names += [ta for fo, fa, to, ta in self.links if to == org]
        return list(dict.fromkeys(names))

    def chain(self, org: str, start: str) -> list:
        """The organization's agents in order from `start`, when its edges form a chain."""
        order, n = [], start
        while n is not None:
            order.append(n)
            nxt = [b for a, b in self.edges(org) if a == n]
            if len(nxt) > 1:
                raise SystemExit(f"{org}: {n} has several successors; this mechanism needs a chain")
            n = nxt[0] if nxt else None
        return order

    def reachable(self, org: str, start: str) -> tuple:
        """The agents reachable from `start` inside the org, and the edges among them."""
        succ = defaultdict(list)
        for a, b in self.edges(org):
            succ[a].append(b)
        seen, stack = [], [start]
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.append(n)
                stack.extend(succ[n])
        edges = [(a, b) for a, b in self.edges(org) if a in seen and b in seen]
        return seen, edges

    def agent(self, org: str, name: str) -> dict:
        return dict(self.orgs[org]["agents"][name] or {})

    def framework(self, org: str) -> str:
        return self.orgs[org]["framework"]

    def mcp_servers(self) -> dict:
        return self.raw.get("mcp_servers") or {}

    @property
    def marker(self) -> str:
        """A unique string inside the user's limit, so the proxy and the observers can tell where it appears."""
        return self.raw.get("probe_marker") or f"LIMIT-{self.name}"

    def request_limit(self, **override) -> dict:
        """The user's request limit (delegation depth, end-to-end deadline, cost budget) as attached at the origin."""
        import time
        spec = {"max_depth": 3, "deadline_s": 600, "max_cost_usd": 0.01, **(self.raw.get("request_limit") or {}),
                **{k: v for k, v in override.items() if v is not None}}
        return {"id": self.marker, "max_depth": spec["max_depth"], "deadline": round(time.time() + spec["deadline_s"], 1),
                "max_cost_usd": spec["max_cost_usd"]}


# ---- ports: fixed layout, so every process can compute every address --------------------------------------------
PROXY_PORT = int(os.getenv("TESTBED_PROXY_PORT", "9300"))
FAKE_PORT = PROXY_PORT + 1
PROXY = f"http://127.0.0.1:{PROXY_PORT}"


def agent_port(cfg: Config, org: str, agent: str) -> int:
    """Every agent's own A2A server."""
    pairs = [(o, a) for o in cfg.orgs for a in cfg.orgs[o]["agents"]]
    return PROXY_PORT + 100 + pairs.index((org, agent))


def upstreams_dir(cfg: Config) -> str:
    """Where each organization's process writes its agents' A2A endpoints ({"org.agent": url}), for the proxy."""
    return os.path.join(ROOT, "runs", cfg.name, "upstreams")


def mcp_port(cfg: Config, server: str) -> int:
    return PROXY_PORT + 50 + list(cfg.mcp_servers()).index(server)


# ---- addresses agents use: everything goes through the instrumentation proxy -------------------------------------
def llm_url(org, agent):
    return f"{PROXY}/llm/{org}/{agent}/v1"


def mcp_url(org, agent, server):
    return f"{PROXY}/mcp/{org}/{agent}/{server}/mcp"


def a2a_url(from_label, to_org, to_agent):
    return f"{PROXY}/a2a/{from_label}/{to_org}/{to_agent}"
