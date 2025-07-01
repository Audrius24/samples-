from duckduckgo_search import DDGS

async def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo and return summarized results.
    """
    with DDGS() as ddgs:
        results = ddgs.text(query)
        docs = []
        for i, r in enumerate(results):
            if i > 4:
                break
            docs.append(f"{r['title']}: {r['body']}")
        summary = "\n\n".join(docs) if docs else "No results found."
    return summary