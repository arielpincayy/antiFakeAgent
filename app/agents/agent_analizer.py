from .agents import Agents

from typing import Dict, List, Any

def agent_analizer(agent: Agents, retrieval_data: List[Dict[str, Any]], news: str) -> List[Dict[str, Any]]:

    """
    Analiza los resultados obtenidos de las diferentes fuentes de información y los resume para evaluar su relevancia y confiabilidad en relación al claim o noticia a verificar.
    Para cada resultado, el agente debe:
    - Resumir la información clave en 2-3 líneas
    - Extraer los puntos clave que apoyan o refutan el claim
    - Evaluar si el contenido APOYA, REFUTA o es NEUTRAL respecto al claim
    - Indicar la relevancia para verificar el claim (Alta, Media, Baja)
    - Evaluar la confiabilidad de la fuente (Alta, Media, Baja)
    El agente debe procesar cada resultado de manera independiente y luego devolver un resumen estructurado que permita al agente de decisión final evaluar la evidencia recopilada.
    
    input:
    - agent: El agente de análisis que procesará la información
    - retrieval_data: Lista de resultados obtenidos de las fuentes, cada uno con título, resumen, fuente, link, citas, etc.
    - news: El claim o noticia a verificar

    output:
    Un diccionario con un resumen estructurado de los resultados analizados, incluyendo título, resumen, puntos clave, relevancia, confiabilidad, fuente y link para cada resultado relevante.
    """

    summarized_results = []

    for item in retrieval_data:

        title = item.get("title", "")
        summary = item.get("summary", "")
        source = item.get("source", "")

        if not summary or len(summary) < 20:
            continue

        prompt = f"""
        CLAIM / NEWS TO VERIFY:
        {news}
        
        TITLE: {title}
        
        CONTENT:
        {summary}
        
        SOURCE TYPE: {source}
        
        TASK:
        - Resume la información en 2-3 líneas
        - Extrae los puntos clave
        - Evalúa si el contenido APOYA, REFUTA o es NEUTRAL respecto al claim
        - Indica relevancia para verificar el claim
        - Evalúa la confiabilidad de la fuente
        
        FORMATO:
        Summary: ...
        Key Points: ...
        Stance: Supports / Refutes / Neutral
        Relevance: High / Medium / Low
        Credibility: High / Medium / Low
        
        NO EXPLIQUES NADA MÁS.
        """

        response = agent.invoke([
            {"role": "system", "content": "Eres un experto en análisis de información y verificación de hechos."},
            {"role": "user", "content": prompt}
        ])

        try:
            parts = response.split("\n")

            summary_out = ""
            key_points = ""
            relevance = ""
            credibility = ""

            for p in parts:
                if "Summary:" in p:
                    summary_out = p.replace("Summary:", "").strip()
                elif "Key Points:" in p:
                    key_points = p.replace("Key Points:", "").strip()
                elif "Relevance:" in p:
                    relevance = p.replace("Relevance:", "").strip()
                elif "Credibility:" in p:
                    credibility = p.replace("Credibility:", "").strip()

            # filtro opcional
            if relevance.lower() == "low":
                continue

            summarized_results.append({
                "title": title,
                "summary": summary_out,
                "key_points": key_points,
                "relevance": relevance,
                "credibility": credibility,
                "source": source,
                "link": item.get("link"),
                "citations": item.get("citations")
            })

        except:
            continue

    return summarized_results