from .agents import Agents
from typing import List, Dict, Any

def router_enough(agent: Agents, results: List[Dict[str, Any]], news: str, iteration: int, max_iter: int = 3) -> bool:
    
    if iteration >= max_iter:
        return True

    if len(results) < 1:
        return False

    # Compactar resultados (MUY importante para el LLM)
    formatted_results = "\n\n".join([
        f"""
        Title: {r.get('title')}
        Summary: {r.get('summary')}
        Key Points: {r.get('key_points')}
        Relevance: {r.get('relevance')}
        Credibility: {r.get('credibility')}
        Source: {r.get('source')}
        """
                for r in results
            ])
        
    prompt = f"""
        You are a fact-checking evaluator.
        
        CLAIM TO VERIFY:
        {news}
        
        You are given analyzed research results.
        
        Your job is to decide if the evidence is sufficient to verify or refute the claim.
        
        Criteria:
        - High or Medium relevance only
        - Prefer High credibility sources
        - Evidence should clearly support or refute the claim
        - Avoid redundancy (same idea repeated ≠ more evidence)
        - There should be some diversity of sources
        
        IMPORTANT:
        If results are vague, weak, or mostly irrelevant → return FALSE.
        
        Respond ONLY with:
        TRUE
        FALSE
        """

    decision = agent.invoke(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Analyzed Results:\n{formatted_results}"}
        ]
    )

    decision_text = decision.strip().upper()

    return decision_text.startswith("TRUE")