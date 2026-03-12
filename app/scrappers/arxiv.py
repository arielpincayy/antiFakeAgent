import requests
import xml.etree.ElementTree as ET
from typing import List, Dict


def search_arxiv(query: str = "climate change", max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for scientific papers on arXiv using the official API.

    Parameters
    ----------
    query : str
        Search query string. It will be used to match papers in all fields.
        Example: "climate misinformation"

    max_results : int
        Maximum number of results to retrieve from the API.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries where each dictionary contains:
            - "title": str      → Paper title
            - "summary": str    → Abstract of the paper
            - "link": str       → URL to the paper (arXiv abstract page)

        Returns an empty list if the request fails or no results are found.
    """

    base_url: str = "http://export.arxiv.org/api/query"

    # Query parameters sent to arXiv API
    params: Dict[str, str | int] = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }

    response: requests.Response = requests.get(base_url, params=params)

    # If request fails, return empty list
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        return []

    # Parse XML response
    root: ET.Element = ET.fromstring(response.text)

    namespace: Dict[str, str] = {
        "atom": "http://www.w3.org/2005/Atom"
    }

    papers: List[Dict[str, str]] = []

    # Iterate over each paper entry
    for entry in root.findall("atom:entry", namespace):

        title_element = entry.find("atom:title", namespace)
        summary_element = entry.find("atom:summary", namespace)
        link_element = entry.find("atom:id", namespace)

        # Defensive check in case any field is missing
        if title_element is None or summary_element is None or link_element is None:
            continue

        title: str = title_element.text.strip()
        summary: str = summary_element.text.strip()
        link: str = link_element.text.strip()

        papers.append({
            "title": title,
            "summary": summary,
            "link": link
        })

    return papers