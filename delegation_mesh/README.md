# delegation_mesh: bounding a request's work across a decentralized agent mesh

Five agents, each built with a different framework and each running as its own process, find each other over mDNS and build the delegation chain themselves. No process runs the chain. The originating user signs a **budget** for the request: delegation depth, an end-to-end deadline, and cost. **Enforcement points** controlled by the operator hold the request's work within that budget, and the agents themselves enforce nothing.

| Script | Agent | Framework | Accepts | Produces | Native limit |
|---|---|---|---|---|---|
| `planner_langgraph.py` | Planner | LangGraph | `request`, `revision` | `plan` | `recursion_limit` |
| `researcher_adk.py` | Researcher | Google ADK | `plan` | `research` | `RunConfig.max_llm_calls` |
| `analyst_crewai.py` | Analyst | CrewAI | `research` | `analysis` | `Agent.max_iter` |
| `writer_openai_agents.py` | Writer | OpenAI Agents SDK | `analysis` | `report` | `Runner.run(max_turns)` |
| `reviewer_autogen.py` | Reviewer | AutoGen | `report` | `review` (or a `revision`) | `RoundRobinGroupChat(max_turns)` |

| File | Role |
|---|---|
| `mesh.py` | Peer SDK for the agents: each agent is an **A2A server** (Agent Card + JSON-RPC `SendMessage`, `a2a-sdk` 1.2.2), mDNS advertise/browse, delegation as A2A messages, hop reports, and the MCP hook that puts the token in each `tools/call`'s `_meta`. |
| `budget_token.py` | Attenuable capability tokens (Biscuit-style). |
| `enforcement.py` | Shared enforcement-point logic: trusted users, pricing, hierarchical ledger with atomic reservation. |
| `meter.py`, `gateway.py` | **Metered-resource enforcement point:** the model API and paid MCP tools, holding the user's credentials. |
| `a2a_guard.py` | **Protocol-boundary enforcement point:** a proxy for A2A JSON-RPC in front of one agent. |
| `client.py` | The requester: signs the budget, submits the request, and collects hop reports and the per-block ledger. |
| `mcp_servers/tcp_reference.py` | An example MCP tool (stdio, or HTTP behind the gateway). |

## Deployments from a config file

A YAML config describes the whole testbed: how many organizations, which agents each has (framework + role), the edges between agents (within and across organizations, including fan-out and join), the entry agent, the revision loop, and each organization's gateway tools. See `deployments/three_orgs.yaml` (3 orgs × 2 agents, one chain) and `deployments/fanout_join.yaml` (fan-out to two researchers, join at the analyst).

```bash
python delegation_mesh/deploy.py show   deployments/fanout_join.yaml   # the topology and every process's command
python delegation_mesh/deploy.py up     deployments/fanout_join.yaml   # start everything (logs in runs/<config>/)
python delegation_mesh/deploy.py status deployments/fanout_join.yaml
python delegation_mesh/deploy.py down   deployments/fanout_join.yaml
```

`deploy.py` only launches processes, like docker-compose. Every gateway, guard and agent is its own process, and they find each other over mDNS. Each agent reads its own part of the config (`--config FILE --agent ID`):

- **Kinds of work.** An agent accepts the work of its predecessors (the kind of work is the producer's id).
- **Fan-out.** An agent hands its output to all of its successors in parallel.
- **Join.** An agent with several predecessors waits for all of them, merges their work and runs once, under the tightest of the budget tokens it received.
- **Completion.** The requester counts open branches. A request is done when none is left.

**Organizations.** Each organization has its own gateway (its credentials and its cost counter), its own trust file (`.trusted_users.<org>.json`) and its own guards. An agent uses its organization's gateway. Depth and deadline hold across organizations. The cost budget does not: each organization's gateway sees only its own charges, so three organizations together can spend up to about three times the budget. This is the shared-balance problem; we measured 2.1× over a $0.004 budget.

**Roles** (`roles.py`): planner, researcher, analyst, writer, factchecker, reviewer, worker. Any framework runner can play any role; a config may override a role's `subtask` and `system` prompt.

## Run

One terminal per process, started in any order:

```bash
source .venv/bin/activate
python delegation_mesh/mcp_servers/tcp_reference.py --http 8765                      # a paid tool
python delegation_mesh/gateway.py --mcp tcp-reference=http://127.0.0.1:8765/mcp      # model API + tools, user's credentials
python delegation_mesh/a2a_guard.py delegation_mesh/planner_langgraph.py
python delegation_mesh/a2a_guard.py delegation_mesh/researcher_adk.py --gateway-tool tcp-reference
python delegation_mesh/a2a_guard.py delegation_mesh/analyst_crewai.py
python delegation_mesh/a2a_guard.py delegation_mesh/writer_openai_agents.py
python delegation_mesh/a2a_guard.py delegation_mesh/reviewer_autogen.py

python delegation_mesh/client.py --enforcement token --max-depth 8 --deadline-s 300 --max-cost-usd 0.01
```

Agent options: `--limit N` (native limit), `--host IP` (bind and advertise; use the LAN IP to spread over machines), `--gateway-tool NAME` (give the agent a paid tool behind the gateway), and `--own-model URL` (use a model outside the user's credentials). An agent started without `a2a_guard.py` has no enforcement point at its boundary.

## The budget and its token

The user signs block 0: `{"max_depth": 8, "deadline": t, "max_cost_usd": 0.01, "max_llm_calls": ..., "max_model_tokens": ..., "max_tool_calls": ...}`.

```
block 0   signed by the user's key              max_depth 8, deadline t, max_cost_usd 0.01
block 1   signed with block 0's next-key        max_depth 7, max_cost_usd 0.0098     (A2A point: Planner)
block 2   signed with block 1's next-key        max_cost_usd 0.0008                  (the Researcher tightens for the Analyst)
block 3   ...                                   max_depth 5, ...                     (A2A point: Analyst)
proof     private half of the last next-key
```

- **Construction:** like Biscuit tokens. Each block names a fresh public key, and the token carries the matching private key (the proof). Anyone holding the token can append a block without coordinating with anyone, which is what lets a partial deployment work. Nobody can edit, drop or reorder a block.
- **Tighten only:** verification recomputes the effective budget as the minimum over the chain and rejects any block that raises a limit. All later hops inherit the tightened value.
- **Trust:** an enforcement point accepts only tokens whose block 0 is signed by a user it serves (`.trusted_users.json`, standing in for account registration).

## Messages on the wire

- **Agent to agent (A2A 1.0, JSON-RPC):** a delegation is one `SendMessage` with `returnImmediately`. The text part is the delegation the model reads. `message.metadata` carries `vecturae/delegation` (the envelope as a JSON string) and `vecturae/budget-token` (the capability token). The receiver answers with an A2A Task: `TASK_STATE_WORKING` if it takes the work, or `TASK_STATE_REJECTED` with the reason. The task completes when the hop's work is done, with the output as an artifact.
- **Agent Card:** standard A2A card at `/.well-known/agent-card.json`, with one skill per accepted kind of work. Chain fields (framework, accepts/produces, native limit, model, tools, enforcement point) are params of the extension `urn:vecturae:ext:delegation-chain:v1`. The extension `urn:vecturae:ext:budget-token:v1` declares the token in metadata.
- **Agent to tool (MCP):** every `tools/call` carries the token in `params._meta["vecturae/budget-token"]`. One hook on `mcp.ClientSession.call_tool` adds it for all five frameworks.
- **Agent to model API:** the token goes in the `X-Capability-Token` HTTP header.
- **Unchanged:** discovery (mDNS) and the hop reports to the requester's reply-to address (testbed instrumentation).

## Enforcement points

**Metered resources (`gateway.py`).** Only this process holds the provider credentials, so every charge to the user passes through it, whatever the agents do. For each model call or MCP `tools/call` it does the following:
1. **Verify** the token (model calls: `X-Capability-Token`; tool calls: the MCP request's `_meta`).
2. **Refuse** after the deadline.
3. **Reserve atomically** the call's worst case: the prompt (estimated) plus the largest completion that fits, or the tool calls. A reservation is placed on **every block of the chain** (hierarchical counters), and the call is refused if any block lacks room. A completion is clamped to what was reserved for it.
4. **Settle** against the provider's reported usage. A refused tool call returns an MCP tool error (`isError`), so the agent's model can go on without the tool.

**A2A boundaries (`a2a_guard.py`).** This process starts its agent on a private port and owns the agent's public endpoint and mDNS advertisement. For each A2A `SendMessage` it does the following, answering a refusal with an A2A Task in state `REJECTED`:
1. **Verify** the token.
2. **Check** the deadline, the remaining depth, and the remaining cost (asked of the gateway).
3. **Charge one level of depth** and append a block with `max_depth − 1` and the remaining cost limits. The agent and everything it delegates to inherit those values.

**Hierarchical counters.** A limit set in block k bounds everything charged below block k, including siblings that fan out from it. Block 0 bounds the whole request.

## What is guaranteed, and under which deployment

- **Cost charged to the user's credentials** stays within the signed budget. This holds for any set of agents, compliant or not, because the metered resources enforce it on every call. Reservations are atomic, so concurrent calls cannot jointly overspend. The one approximation is that a call's prompt size is estimated before the call; the estimate is conservative but not exact.
- **Depth and deadline** are enforced at every A2A point the request crosses. The more of the path runs through enforcement points, the tighter the bound.
- **Outside work:** an agent with no enforcement point, on its own model, can do unbounded work itself. But it cannot send work across the next enforcement point without a valid, tightened token, and it cannot loosen or strip one. Its work is confined to the region of the graph it occupies.

## Adding MCP servers

Each agent script has an `MCP_SERVERS` list:

```python
MCP_SERVERS = [
    gateway_mcp("tcp-reference"),                     # paid tool behind the gateway: metered, uses the request's token
    local_mcp("tcp_reference.py"),                    # stdio script in mcp_servers/ (not metered)
    {"name": "search", "transport": "http", "url": "http://localhost:8000/mcp", "headers": {...}},
]
```

Each script turns these specs into its framework's native MCP integration:

| Framework | MCP integration |
|---|---|
| LangGraph | `langchain-mcp-adapters` → `ToolNode` loop |
| Google ADK | `McpToolset` |
| CrewAI | native `MCPServerStdio` / `MCPServerHTTP` |
| OpenAI Agents SDK | `MCPServerStdio` / `MCPServerStreamableHttp` |
| AutoGen | `McpWorkbench` |
