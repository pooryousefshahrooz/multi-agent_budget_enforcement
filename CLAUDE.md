# Project rule: everything goes through the testbed

The goal of this project is to build and complete the testbed in `testbed/`, and to run every experiment of the
paper (`main_delegation.pdf`, "Time-to-Live for Agents") with it.

- Every experiment runs ON the testbed: organizations, agents, topology, links, MCP servers and model come from a
  testbed config (`testbed/configs/*.yaml`), started with `testbed/launch.py`, driven with the testbed's runners
  (`run.py`, `bench.py`, ...), measured by the testbed's instrumentation proxy (`testbed/common/instrument.py`).
- Do NOT build separate scripts, probes or harnesses that create their own agents or servers outside the testbed.
- If an experiment needs something the testbed cannot do yet (a new kind of edge, an observer, a metric, a
  framework mechanism), ADD IT TO THE TESTBED (config option, `common/`, `orgs/`), then run the experiment through it.
- Every organization runs as its OWN process (`launch.py` -> `orgs/run_org.py`, its framework's own environment), as if
  the organizations were on different continents. No script or process ever holds the code of two organizations;
  organizations interact only over the network (A2A through their own endpoints, via the proxy). A framework's native
  delegation (sub-agents, handoffs, subgraphs) stays inside one organization's process.
- Before starting any experiment, check: does this run on the testbed? If not, stop and extend the testbed instead.
- Do only what the user asks; confirm before large or ambiguous changes.
