from typing import Any, List, Dict

from app.agents.agents import Agents

def router_create_queries(agent: Agents, news: str, query: str, entities: List[Dict[str, Any]]) -> str:

    if len(entities) >= 1:
        return query
    
    prompt = f"""
    USER INPUT: "{news}"

    Genera UNA sola query de búsqueda optimizada.

    REGLAS:
    - Clara, específica y concisa
    - Incluir términos clave del claim
    - Puede usar AND y OR si mejora la búsqueda
    - Evitar complejidad innecesaria
    - Debe funcionar en Google y bases científicas

    NO expliques nada.

    FORMATO:
    query
    """

    response = agent.invoke([
        {
            "role": "system",
            "content": "Eres un experto en generación de queries de búsqueda."
        },
        {"role": "user", "content": prompt}
    ])

    query_result = response.strip().replace("\n", "")

    if not query_result or len(query_result) < 10:
        query_result = news

    return query_result