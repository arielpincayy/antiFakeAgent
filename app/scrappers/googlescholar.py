from serpapi import GoogleSearch
import os
from typing import List, Dict, Any


SERP_API_KEY: str | None = os.getenv("GOOGLESCHOLAR")


def search_scholar(query: str = "climate misinformation",
                   num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for scientific papers on Google Scholar using SerpApi.

    Parameters
    ----------
    query : str
        Search query string.
        Example: "climate misinformation"

    num_results : int
        Number of results to retrieve from Google Scholar.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries where each dictionary contains:
            - "title": str | None
            - "link": str | None
            - "snippet": str | None
            - "publication_info": dict
            - "cited_by": int

        Returns an empty list if:
            - API key is missing
            - Request fails
            - No results are found
    """

    if SERP_API_KEY is None:
        print("Error: SERP API key not found in environment variables.")
        return []

    params: Dict[str, Any] = {
        "engine": "google_scholar",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": num_results
    }

    try:
        search: GoogleSearch = GoogleSearch(params)
        results: Dict[str, Any] = search.get_dict()
    except Exception as e:
        print(f"Error during SerpApi request: {e}")
        return []

    papers: List[Dict[str, Any]] = []

    for result in results.get("organic_results", []):

        title: str | None = result.get("title")
        link: str | None = result.get("link")
        snippet: str | None = result.get("snippet")

        publication_info: Dict[str, Any] = result.get("publication_info", {})

        cited_by: int = (
            result.get("inline_links", {})
                  .get("cited_by", {})
                  .get("total", 0)
        )

        papers.append({
            "title": title,
            "link": link,
            "snippet": snippet,
            "publication_info": publication_info,
            "cited_by": cited_by
        })

    return papers