from .agents import Agents
from typing import Dict, List, Any


def _filter_by_title(agent: Agents, items: List[Dict[str, Any]], news: str) -> List[Dict[str, Any]]:
    """
    Fase 1: Descarte rápido por título.
    Usa el agente para evaluar si el título es relevante al claim.
    Retorna solo los items cuyo título parece relevante.
    """
    if not items:
        return []

    # Construimos un batch prompt para evaluar todos los títulos de una vez
    titles_block = "\n".join(
        f"{i}. {item.get('title', '(sin título)')}"
        for i, item in enumerate(items)
    )

    prompt = f"""
    CLAIM / NEWS TO VERIFY:
    {news}
    
    TITLES TO EVALUATE:
    {titles_block}
    
    TASK:
    Para cada título, indica únicamente si es RELEVANT o IRRELEVANT respecto al claim.
    No expliques nada. Responde SOLO en este formato exacto, una línea por título:
    0: RELEVANT
    1: IRRELEVANT
    2: RELEVANT
    ...
    """

    response = agent.invoke([
        {"role": "system", "content": "Eres un experto en verificación de hechos. Evalúa relevancia de títulos de forma rápida y precisa."},
        {"role": "user", "content": prompt}
    ])

    # Parsear respuesta
    relevant_indices = set()
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            idx_str, verdict = line.split(":", 1)
            if verdict.strip().upper() == "RELEVANT":
                relevant_indices.add(int(idx_str.strip()))
        except (ValueError, IndexError):
            continue

    return [item for i, item in enumerate(items) if i in relevant_indices]


def _analyze_content(agent: Agents, item: Dict[str, Any], news: str) -> Dict[str, Any] | None:
    """
    Fase 2: Análisis profundo del contenido.
    Analiza el contenido del item y lo descarta si no es realmente relevante.
    Retorna un dict con el análisis estructurado, o None si debe descartarse.
    """
    title = item.get("title", "")
    summary = item.get("summary", "")
    source = item.get("source", "")

    if not summary or len(summary) < 20:
        return None

    prompt = f"""
    CLAIM / NEWS TO VERIFY:
    {news}
    
    TITLE: {title}
    CONTENT:
    {summary}
    
    SOURCE TYPE: {source}
    
    TASK:
    1. Determina si el CONTENIDO (no solo el título) realmente habla sobre el claim.
       Si el contenido NO está relacionado con el claim, responde únicamente: DISCARD
    2. Si sí está relacionado:
       - Resume la información en 2-3 líneas
       - Extrae los puntos clave
       - Evalúa si el contenido APOYA, REFUTA o es NEUTRAL respecto al claim
       - Indica relevancia para verificar el claim (High / Medium / Low)
       - Evalúa la confiabilidad de la fuente (High / Medium / Low)
    
    FORMATO (si no se descarta):
    Verdict: KEEP
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

    if "DISCARD" in response.upper() and "VERDICT: KEEP" not in response.upper():
        return None

    try:
        summary_out = ""
        key_points = ""
        stance = ""
        relevance = ""
        credibility = ""

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("Summary:"):
                summary_out = line.replace("Summary:", "").strip()
            elif line.startswith("Key Points:"):
                key_points = line.replace("Key Points:", "").strip()
            elif line.startswith("Stance:"):
                stance = line.replace("Stance:", "").strip()
            elif line.startswith("Relevance:"):
                relevance = line.replace("Relevance:", "").strip()
            elif line.startswith("Credibility:"):
                credibility = line.replace("Credibility:", "").strip()

        if relevance.lower() == "low":
            return None

        return {
            "title": title,
            "summary": summary_out,
            "key_points": key_points,
            "stance": stance,
            "relevance": relevance,
            "credibility": credibility,
            "source": source,
            "link": item.get("link"),
            "citations": item.get("citations"),
        }

    except Exception:
        return None


def agent_analizer(agent: Agents, retrieval_data: List[Dict[str, Any]], news: str) -> List[Dict[str, Any]]:
    """
    Analiza los resultados en dos fases:
      - Fase 1: Descarte rápido por título (batch, una sola llamada al agente)
      - Fase 2: Análisis profundo del contenido (una llamada por item superviviente)

    Descarta items ya analizados, sin summary, con relevancia Low,
    o cuyo contenido no esté realmente relacionado con el claim.
    """
    # Excluir items ya analizados
    pending = [item for item in retrieval_data if not item.get("analyzed")]
    for item in pending:
        item["analyzed"] = True

    if not pending:
        return []

    # — Fase 1: filtro rápido por título —
    title_filtered = _filter_by_title(agent, pending, news)

    # — Fase 2: análisis de contenido —
    summarized_results = []
    for item in title_filtered:
        result = _analyze_content(agent, item, news)
        if result is not None:
            summarized_results.append(result)

    return summarized_results