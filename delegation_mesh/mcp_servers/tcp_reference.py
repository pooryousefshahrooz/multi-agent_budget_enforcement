"""Example stdio MCP server: reference facts about TCP congestion control.

Two ways to use it:
  * as a local tool: add `local_mcp("tcp_reference.py")` to an agent's MCP_SERVERS (stdio, not metered);
  * as a paid tool behind the metered-resource gateway (charged to the request's budget):
        python delegation_mesh/mcp_servers/tcp_reference.py --http 8765
        python delegation_mesh/gateway.py --mcp tcp-reference=http://127.0.0.1:8765/mcp
    and add `gateway_mcp("tcp-reference")` to the agent's MCP_SERVERS.
"""
import sys

from mcp.server.fastmcp import FastMCP

PORT = int(sys.argv[sys.argv.index("--http") + 1]) if "--http" in sys.argv else 8765
mcp = FastMCP("tcp-reference", log_level="WARNING", host="127.0.0.1", port=PORT)

FACTS = {
    "reno": ("TCP Reno (RFC 5681): slow start, then AIMD congestion avoidance (+1 MSS per RTT). "
             "On 3 duplicate ACKs: fast retransmit + fast recovery, cwnd halved (beta=0.5). "
             "On timeout: cwnd reset to 1 MSS. Growth is RTT-dependent, so it recovers slowly on "
             "high bandwidth-delay-product paths."),
    "cubic": ("TCP CUBIC (RFC 9438): window grows as a cubic function of time since the last loss, "
              "W(t) = C(t-K)^3 + W_max with C=0.4, plateauing near W_max. On loss cwnd is reduced to "
              "beta=0.7 of its value. Growth depends on elapsed time, not RTT, so it is RTT-fair and "
              "recovers quickly on long fat networks. Default in Linux, Windows and macOS."),
}


@mcp.tool()
def congestion_control_facts(algorithm: str) -> str:
    """Return reference facts for a TCP congestion control algorithm ('reno' or 'cubic')."""
    return FACTS.get(algorithm.strip().lower().replace("tcp ", ""),
                     f"No entry for {algorithm!r}; known: {', '.join(FACTS)}")


if __name__ == "__main__":
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")
