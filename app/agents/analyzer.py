from typing import Dict, Any, List
import re
import ast

def filter_papers_by_title(query: str, papers: List[Dict[str, Any]], agent) -> List[Dict[str, Any]]:
    """
    Filter papers using only one LLM.
    """

    titles_context = ""
    for i, p in enumerate(papers):
        titles_context += f"[{i}] {p['title']}\n"

    prompt = f"""
    User Research Topic: "{query}"

    PAPER TITLES:
    {titles_context}

    TASK:
    Select the papers that are clearly relevant to the research topic based ONLY on their titles.

    Rules:
    - Be strict: discard vague or unrelated titles.
    - Keep only papers that likely contribute to the research.
    - Return ONLY a Python list of selected IDs (integers).
    - No explanations.

    Example output:
    [0, 2, 5, 7]
    """

    response = agent.chat(
        messages=[
            {
                "role": "system",
                "content": "You are a strict academic filter. You select only relevant papers based on titles."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    clean_response = re.sub(r"```[a-z]*\n|```", "", response).strip()

    try:
        selected_ids = ast.literal_eval(clean_response)
    except:
        print("⚠️ Error parsing filter response, fallback: keep all papers")
        return papers

    # Filtrar papers
    filtered = [papers[i] for i in selected_ids if i < len(papers)]

    print(f"Filtered {len(filtered)}/{len(papers)} papers based on title\n")

    return filtered

def analyze_single_paper(query: str, paper: Dict[str, Any], agent) -> str:
    summary = paper['summary'] if paper['summary'] else "Not available"

    prompt = f"""
    User Research Topic: "{query}"

    PAPER:
    TITLE: {paper['title']}
    SOURCE: {paper['source']}
    CITATIONS: {paper['citations']}
    SUMMARY: {summary}
    LINK: {paper['link']}

    TASK:
    Analyze this paper individually as an expert researcher.

    OUTPUT FORMAT (Markdown):
    - Title:
    - Methodology:
    - Key Results:
    - Strengths:
    - Weaknesses:
    - Relevance to Research Topic:
    - Key Concepts / Keywords:
    """

    return agent.chat(
        messages=[
            {
                "role": "system",
                "content": "You are a Senior Research Analyst. You analyze one paper at a time with deep academic rigor."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

def analyzer(query: str, papers: List[Dict[str, Any]], agent) -> str:
    print("Filtering papers by title...\n")
    
    papers = filter_papers_by_title(query, papers, agent)
    
    print("Analyzing papers one by one...\n")

    individual_analyses = []

    for i, paper in enumerate(papers):
        print(f"Analyzing paper {i+1}/{len(papers)}...")
        analysis = analyze_single_paper(query, paper, agent)
        individual_analyses.append(f"--- PAPER {i} ---\n{analysis}")

    combined_context = "\n\n".join(individual_analyses)

    final_prompt = f"""
    User Research Topic: "{query}"

    INDIVIDUAL PAPER ANALYSES:
    {combined_context}

    TASK:
    Based on the individual analyses, create a high-quality literature review.

    OUTPUT:
    1. A Markdown table with the most important papers (3–10 max)
       Columns:
       - Paper
       - Methodology
       - Key Results
       - Relevance
       - Link

    2. A 2-paragraph "State of the Art" synthesis:
       - Trends
       - Common methods
       - Research gaps
    """

    print("\nGenerating final synthesis...\n")

    return agent.chat(
        messages=[
            {
                "role": "system",
                "content": "You are a Senior Research Consultant synthesizing multiple paper analyses into a structured literature review."
            },
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.2
    )