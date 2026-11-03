"""Metered-resource enforcement point (model API + paid MCP tools), as its own process.

Holds the provider credentials that pay for the user's requests, so every charge to the user passes through it.
See meter.py for the two modes (capability tokens, legacy per-request call budget) and the endpoints.
Advertised on mDNS as _llm-gateway._tcp, so agents and A2A enforcement points find it without configuration.

    python delegation_mesh/gateway.py [--upstream http://localhost:11434/v1] [--api-key KEY]
                                      [--mcp tcp-reference=http://127.0.0.1:8765/mcp] ...
"""
import argparse, os, signal, threading

from mesh import GATEWAY_SERVICE, Mesh, log
from enforcement import DEFAULT_ORG, Pricing, TrustStore, trust_file
from meter import start_meter


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Metering LLM gateway")
    p.add_argument("--upstream", default=os.getenv("OLLAMA_V1", "http://localhost:11434/v1"))
    p.add_argument("--host", default=os.getenv("MESH_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--org", default=os.getenv("MESH_ORG", DEFAULT_ORG),
                   help="the organization this gateway belongs to: its own users (trust file) and credentials")
    p.add_argument("--api-key", default=os.getenv("UPSTREAM_API_KEY"), help="provider credential (never given to agents)")
    p.add_argument("--mcp", action="append", default=[], metavar="NAME=URL",
                   help="a streamable-HTTP MCP server to meter and expose as /<rid>/<agent>/mcp/NAME")
    p.add_argument("--usd-per-1m-input", type=float, default=0.15)
    p.add_argument("--usd-per-1m-output", type=float, default=0.60)
    p.add_argument("--usd-per-tool-call", type=float, default=0.0005)
    args = p.parse_args()

    mcp = dict(m.split("=", 1) for m in args.mcp)
    _, server, url = start_meter(args.upstream, args.host, args.port, mcp_upstreams=mcp, api_key=args.api_key,
                                 trust=TrustStore(trust_file(args.org)),
                                 pricing=Pricing(args.usd_per_1m_input, args.usd_per_1m_output, args.usd_per_tool_call))
    port = server.server_address[1]
    mesh = Mesh(args.host)
    mesh.advertise(GATEWAY_SERVICE, f"llm-gateway-{args.org}-{port}", port,
                   {"upstream": args.upstream, "mcp": ",".join(mcp), "org": args.org})
    log(f"[org {args.org}] metered-resource gateway up at {url} → {args.upstream}"
        + (f", MCP tools: {', '.join(mcp)}" if mcp else "") + "; advertising on mDNS")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    mesh.close()
    server.shutdown()
