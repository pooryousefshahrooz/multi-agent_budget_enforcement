"""FanOutQA's tool as an MCP server: ReDel's `search` (redel/tools/fanoutqa/impl.py, branch demo/emnlp), unchanged
in name, docstring and behaviour: search Wikipedia (the FanOutQA snapshot) for an article by title; return it whole if
it fits, else the best fragments that fit (BM25, ReDel's retrieval_type="bm25").

Two differences, both forced by MCP: the size budget is fixed (--max-chars) instead of being derived from the calling
engine's context window, and fragments are ranked by the task's question (set by bench.py through POST /task) because
a tool server cannot see the calling agent's last message.

    python testbed/mcp_servers/fanoutqa_search.py --http 9350 [--max-chars 16000]
"""
import sys

import fanoutqa
from fanoutqa.retrieval import Corpus
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

PORT = int(sys.argv[sys.argv.index("--http") + 1]) if "--http" in sys.argv else 9350
MAX_CHARS = int(sys.argv[sys.argv.index("--max-chars") + 1]) if "--max-chars" in sys.argv else 16000
mcp = FastMCP("fanoutqa", log_level="WARNING", host="127.0.0.1", port=PORT)
TASK = {"question": ""}


@mcp.custom_route("/task", methods=["POST"])
async def set_task(request: Request):
    TASK["question"] = (await request.json()).get("question", "")
    return JSONResponse({"ok": True})


@mcp.tool()
def search(query: str) -> str:
    """Search Wikipedia for an article with the given title, and get its content. If no such article is found, return similar article names."""
    query = query.strip()
    matches = fanoutqa.wiki_search(query)
    for match in matches:
        if match.title.lower() == query.lower():
            found_article = match
            break
    else:
        similar_searches = "\n".join(f"- {m.title}" for m in matches)
        return (
            "No page with that title exists. Try searching for one of these similar articles"
            f" instead:\n{similar_searches}"
        )

    prompt = f"<document>\n<title>{found_article.title}</title>\n{{}}</document>"
    wiki_content = fanoutqa.wiki_content(found_article)
    full_content = prompt.format(f"<content>\n{wiki_content}\n</content>\n")
    if len(full_content) <= MAX_CHARS:
        return full_content

    corpus = Corpus([found_article], doc_len=1024)
    retrieved_docs = []
    for doc in corpus.best(TASK["question"] or query):
        formatted = f"<fragment>\n{doc.content}\n</fragment>\n"
        if len(prompt.format("".join(retrieved_docs) + formatted)) > MAX_CHARS:
            break
        retrieved_docs.append(formatted)
    return prompt.format("".join(retrieved_docs))


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
