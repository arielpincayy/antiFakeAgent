from .agents import Agents

def agent_summarizer(agent: Agents, news: str) -> str:

    """
    Resume la noticia en un párrafo de máximo 100 palabras, destacando los puntos clave y el contexto general.
    input:
        - agent: El agente de resumen que procesará la información
        - news: El claim o noticia a resumir
    output:
        Un diccionario con el resumen generado por el agente, destacando los puntos clave y el contexto general de la noticia.
    """

    prompt = f"""
    USER INPUT:
    {news}
    
    TAREA:
    Resume la noticia en un máximo de 100 palabras SIN perder información verificable.
    
    REGLAS:
    - Mantén entidades clave (personas, organizaciones, lugares)
    - Mantén la afirmación principal EXACTA
    - Elimina redundancia y ruido
    - No agregues información nueva
    - No cambies el significado
    
    SALIDA:
    Un solo párrafo.
    """

    response = agent.invoke([
        {
            "role": "system",
            "content": "Eres un experto en resumen de noticias."
        },
        {"role": "user", "content": prompt}
    ])

    return response.strip()