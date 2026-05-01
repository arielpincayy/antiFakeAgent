from .agents import Agents
from typing import List, Dict, Any


def agent_synthetizer(agent: Agents, analysis: List[Dict[str, Any]], news: str) -> str:

    if not analysis:
        return "# Resultado\n\nNo se encontró evidencia suficiente para evaluar la noticia."

    # Compactar resultados (clave para el LLM)
    formatted_results = "\n\n".join([
        f"""
        Title: {r.get('title')}
        Summary: {r.get('summary')}
        Key Points: {r.get('key_points')}
        Relevance: {r.get('relevance')}
        Credibility: {r.get('credibility')}
        Source: {r.get('source')}
        """
        for r in analysis
    ])

    prompt = f"""
    You are a professional fact-checker.
    
    Your task is to synthesize the analyzed evidence and determine whether the claim is true or false.
    
    CLAIM:
    {news}
    
    EVIDENCE:
    {formatted_results}
    
    INSTRUCTIONS:
    - Determine if the claim is:
        TRUE
        FALSE
        or UNCERTAIN
    - Base your decision ONLY on the evidence
    - Consider:
        - Agreement between sources
        - Credibility of sources
        - Strength of evidence
    - Do NOT hallucinate information
    
    OUTPUT FORMAT (Markdown):
    
    # 🧠 Fact-Check Result
    
    ## Verdict
    (TRUE / FALSE / UNCERTAIN)
    
    ## Explanation
    (Short paragraph explaining why)
    
    ## Key Evidence
    - Bullet points summarizing strongest evidence
    
    ## Source Quality
    (Brief evaluation of reliability of sources)
    
    ## Final Confidence
    (High / Medium / Low)
    
    DO NOT write anything outside this format.
    """

    response = agent.invoke([
        {"role": "system", "content": "You are an expert fact-checker and scientific analyst."},
        {"role": "user", "content": prompt}
    ])

    return response.strip()