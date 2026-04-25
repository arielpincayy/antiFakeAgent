from .agents import Agents


def router_bases(query: str, agents: Agents) -> str:
    prompt = f"""
    USER QUERY: "{query}"
    
    Tu tarea es clasificar la consulta en UNA de estas dos categorías:
    
    1. 'web search'
    2. 'scientific'
    
    CRITERIOS:
    
    Responde 'web search' si la consulta:
    - Se refiere a noticias recientes, eventos actuales o declaraciones públicas
    - Incluye personas, instituciones o eventos ("X dijo", "nuevo anuncio", "últimas noticias")
    - Es sobre política, sociedad, economía, tecnología actual o tendencias
    - Depende de información reciente o en tiempo real
    
    Responde 'scientific' si la consulta:
    - Trata sobre conocimiento establecido o verificable científicamente
    - Involucra estudios, evidencia, experimentos o teorías
    - Es sobre temas como medicina, biología, física, química, IA, etc.
    - No depende de eventos recientes sino de conocimiento acumulado
    
    EJEMPLOS:
    
    "CNN reporta nuevo brote de COVID" → web search  
    "Las vacunas COVID alteran el ADN?" → scientific  
    "Elon Musk anunció nueva IA" → web search  
    "Cómo funcionan los modelos transformers?" → scientific  
    
    IMPORTANTE:
    - Si hay duda, prioriza 'scientific' SOLO si el claim requiere evidencia científica
    - Si depende del tiempo o actualidad → 'web search'
    
    RESPONDE SOLO CON:
    'web search' o 'scientific'
    """

    decision = agents.invoke(
        messages=[
            {
                "role": "system", 
                "content": "Eres un agente de enrutamiento que decide si una consulta del usuario debe ser respondida por un agente de búsqueda web o por un agente de investigación científica. Tu tarea es analizar la consulta y determinar si es más adecuada para una búsqueda general en la web o si requiere una investigación más profunda y técnica. Responde solo con 'web search' o 'scientific' según corresponda, sin proporcionar explicaciones adicionales."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0 # Crucial para que no haya variaciones
    )

    return "web search" if "web search" in decision.strip().lower() else "scientific"