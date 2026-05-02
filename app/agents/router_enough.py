from .agents import Agents
from typing import List, Dict, Any


def router_enough(agent: Agents, results: List[Dict[str, Any]], news: str) -> bool:
    if len(results) < 1:
        return False

    formatted_results = "\n\n".join([
        f"""Título: {r.get('title')}
        Resumen: {r.get('summary')}
        Relevancia: {r.get('relevance')}
        Credibilidad: {r.get('credibility')}
        Fuente: {r.get('source')}"""
        for r in results
    ])

    prompt = f"""Eres un evaluador experto en verificación de hechos.

    NOTICIA A VERIFICAR:
    {news}
    
    Se te proporcionan fuentes de información ya analizadas.
    Tu única tarea es decidir si ya tenemos suficiente material para investigar
    la veracidad de la noticia, NO si la noticia es verdadera o falsa.
    
    Para responder VERDADERO (tenemos suficiente material) se deben cumplir AMBAS condiciones:
    1. Al menos una fuente trata DIRECTAMENTE el tema de la noticia (relevancia Alta o Media)
    2. Al menos una fuente tiene credibilidad Alta o Media
    3. Las fuentes contienen la suficiente información para validar o desmentir la NOTICIA A VERIFICAR
    
    Para responder FALSO (necesitamos buscar más):
    - Las fuentes son tangenciales o no abordan el tema central
    - Las fuentes disponibles tienen credibilidad Baja o son poco confiables
    - Las fuentes no contienen la suficiente información para dar un veredicto concreto a favor o en contra
    
    IGNORA completamente si la postura de la fuente apoya o refuta la noticia.
    Un artículo científico neutral que estudia el tema ES evidencia útil.
    
    Responde con:
    VERDADERO
    FALSO"""

    decision = agent.invoke(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Fuentes analizadas:\n{formatted_results}"}
        ]
    )
    
    decision_text = decision.strip().upper()
    return decision_text.startswith("VERDADERO")