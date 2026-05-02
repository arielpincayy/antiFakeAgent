from .agents import Agents
from typing import List, Dict, Any


def agent_synthetizer(agent: Agents, analysis: List[Dict[str, Any]], news: str) -> str:

    if not analysis:
        return "# Resultado\n\nNo se encontró evidencia suficiente para evaluar la noticia."

    # Compactar resultados (clave para el LLM)
    formatted_results = "\n\n".join([
        f"""
        Título: {r.get('title')}
        Resumen: {r.get('summary')}
        Puntos Clave: {r.get('key_points')}
        Relevancia: {r.get('relevance')}
        Credibilidad: {r.get('credibility')}
        Fuente: {r.get('source')}
        """
        for r in analysis
    ])

    prompt = f"""
    Eres un verificador de hechos profesional.
    
    Tu tarea es sintetizar la evidencia analizada y determinar si la afirmación es verdadera o falsa.
    
    AFIRMACIÓN:
    {news}
    
    EVIDENCIA:
    {formatted_results}
    
    INSTRUCCIONES:
    - Determina si la afirmación es:
        VERDADERA
        FALSA
        o INCIERTA
    - Basa tu decisión ÚNICAMENTE en la evidencia proporcionada
    - Considera:
        - El grado de acuerdo entre las fuentes
        - La credibilidad de las fuentes
        - La solidez de la evidencia
    - NO inventes información
    
    FORMATO DE SALIDA (Markdown):
    
    # 🧠 Resultado de Verificación
    
    ## Veredicto
    (VERDADERA / FALSA / INCIERTA)
    
    ## Explicación
    (Párrafo corto explicando el porqué)
    
    ## Evidencia Clave
    - Puntos clave más relevantes
    
    ## Calidad de las Fuentes
    (Breve evaluación de la confiabilidad de las fuentes)
    
    ## Nivel de Confianza
    (Alto / Medio / Bajo)
    
    NO escribas nada fuera de este formato.
    """

    response = agent.invoke([
        {"role": "system", "content": "Eres un experto en verificación de hechos y análisis científico."},
        {"role": "user", "content": prompt}
    ])

    return response.strip()