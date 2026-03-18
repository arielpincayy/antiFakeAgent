import re
import ast

from .agents import Agents


def generate_search_queries(query:str, agents: Agents):
    prompt = f"""
            User Research Question:
            "{query}"
        
            Task:
            Understand the user's research intent and generate 3 high-quality academic search queries.
        
            Instructions:
            1. Infer the main topic, subtopics, and research goal.
            2. Expand with synonyms, related terms, and technical vocabulary.
            3. If applicable, include:
               - methods (e.g., "deep learning", "survey", "case study")
               - context (e.g., "social media", "healthcare", "federated learning")
               - outcomes (e.g., "detection", "impact", "performance")
        
            Rules:
            - Use Boolean operators (AND, OR).
            - Each query must reflect a DIFFERENT angle of the research.
            - Be specific and academic (like Google Scholar queries).
            - Return ONLY a Python list of strings.
            - No markdown, no explanation.
        
            Example:
            Input: "fake news detection"
            Output: [
                "fake news detection AND deep learning AND social media",
                "misinformation classification OR disinformation detection AND NLP",
                "machine learning approaches AND fake news detection AND dataset analysis"
            ]
            """
        
    response = agents.chat(
        messages=[
            {"role": "system", "content": "You are an API that generates academic search queries. You deeply understand research intent and return only raw Python lists. No prose, no markdown."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    # Limpieza robusta: eliminar bloques de código markdown y espacios extra
    clean_response = re.sub(r"```[a-z]*\n|```", "", response).strip()

    try:
        search_queries = ast.literal_eval(clean_response)
        if isinstance(search_queries, list):
            return search_queries[:3] 
        return [query]
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing search queries: {e}. Falling back to original query.")
        return [query]