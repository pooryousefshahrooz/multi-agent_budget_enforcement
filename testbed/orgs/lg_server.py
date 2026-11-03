"""The graphs LangGraph's Agent Server serves for one organization (TESTBED_CONFIG, TESTBED_ORG).

  delegation: a2a (default)  graph_<agent> for every agent: one node, the agent's behaviour on a message
                             (common.agent_logic), whose work is the agent's ReAct agent (org_langgraph.make_work)
  delegation: native         graph_<agent> for every agent that receives work from outside (the entry, link
                             targets): the organization's agents joined by LangGraph's own mechanism
                             (org_langgraph.native_graph), built per run by an async graph factory
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from common.agent_logic import AgentLogic
from common.config import Config
from orgs.org_langgraph import make_work, native_graph

CFG = Config(os.environ["TESTBED_CONFIG"])
ORG = os.environ["TESTBED_ORG"]


def build(name):
    logic = AgentLogic(CFG, ORG, name, make_work(CFG, ORG, name))

    async def agent(state, config):
        text = next(m.content for m in reversed(state["messages"]) if m.type == "human")
        pad = (config.get("configurable") or {}).get("__pregel_scratchpad")
        # everything the Agent Server handed this run: the run config, and the graph state (the incoming messages with
        # all their fields), so a limit placed anywhere by the server is seen
        received = {"metadata": config.get("metadata"),
                    "configurable": {k: v for k, v in (config.get("configurable") or {}).items() if not k.startswith("__")},
                    "state": [m.model_dump() for m in state["messages"]],
                    "state_other": {k: v for k, v in state.items() if k != "messages"}}
        counter = {"steps_left": pad.stop - pad.step} if pad else None
        return {"messages": [AIMessage(content=await logic.handle(text, received, counter))]}
    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    return g.compile()


def native_factory(ingress):
    async def factory(config):
        return await native_graph(CFG, ORG, ingress)
    return factory


if CFG.orgs[ORG].get("delegation", "a2a") == "native":
    for _name in CFG.ingress(ORG):
        globals()[f"graph_{_name}"] = native_factory(_name)
else:
    for _name in CFG.orgs[ORG]["agents"]:
        globals()[f"graph_{_name}"] = build(_name)
