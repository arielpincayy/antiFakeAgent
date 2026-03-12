import os
import requests
from typing import List, Dict, Any, Optional

API_KEY: Optional[str] = os.getenv("OPENALEX")


def search_openalex(query: str = "climate misinformation",
                    per_page: int = 5) -> List[Dict[str, Any]]:
    """
    Search for scientific papers using the OpenAlex API.

    Parameters
    ----------
    query : str
        Search query string.
        Example: "climate misinformation"

    per_page : int
        Number of results to retrieve per request.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries where each dictionary contains:
            - "title": str | None
            - "doi": str | None
            - "year": int | None
            - "citations": int | None
            - "type": str | None
            - "abstract": str | None

        Returns an empty list if:
            - The request fails
            - No results are found
    """

    url: str = "https://api.openalex.org/works"

    params: Dict[str, Any] = {
        "search": query,
        "per-page": per_page,
        "mailto": "ariel.pincay@yachaytech.edu.ec"
    }

    # Add API key only if available
    if API_KEY:
        params["api_key"] = API_KEY

    try:
        response: requests.Response = requests.get(url, params=params, timeout=10)
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return []

    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        return []

    data: Dict[str, Any] = response.json()
    results: List[Dict[str, Any]] = []

    for work in data.get("results", []):

        abstract: Optional[str] = reconstruct_abstract(
            work.get("abstract_inverted_index")
        )

        results.append({
            "title": work.get("title"),
            "doi": work.get("doi"),
            "year": work.get("publication_year"),
            "citations": work.get("cited_by_count"),
            "type": work.get("type"),
            "abstract": abstract
        })

    return results


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """
    Reconstruct the abstract text from OpenAlex's inverted index format.

    Parameters
    ----------
    inverted_index : dict | None
        Dictionary where:
            - key = word (str)
            - value = list of integer positions

    Returns
    -------
    str | None
        The reconstructed abstract text in correct word order,
        or None if no abstract is available.
    """

    if not inverted_index:
        return None

    word_positions: Dict[int, str] = {}

    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions[pos] = word

    ordered_words: List[str] = [
        word_positions[i] for i in sorted(word_positions)
    ]

    return " ".join(ordered_words)