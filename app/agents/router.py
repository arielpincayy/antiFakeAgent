from .agents import Agents


def router(query: str, memory: list, agents: Agents) -> str:
    if not memory:
        return "research"

    memory_summary = ""
    for i, m in enumerate(memory):
        memory_summary += f"- Topic/Paper {i+1}: {m[0]}\n"

    prompt = f"""
    USER QUERY: "{query}"

    AVAILABLE DATA IN MEMORY:
    {memory_summary}

    DECISION CRITERIA:
    - Respond 'memory' if the user is asking for:
        * Specific details of the papers above (citations, links, methodology, authors).
        * A summary, comparison, or formatting change of the existing data.
        * Any question where the answer is likely contained in the topics listed.
    - Respond 'research' if the user is asking for:
        * A completely new topic not mentioned above.
        * Information that requires updated real-time data or a broader search.
        * A deep dive into a concept that was only mentioned briefly without details.

    RESPONSE (ONLY 'memory' OR 'research'):
    """

    decision = agents.chat(
        messages=[
            {
                "role": "system", 
                "content": "You are a precise Intent Classifier for a Research Assistant. Your goal is to minimize redundant web searches by identifying if the data is already in the system memory."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0 # Crucial para que no haya variaciones
    )

    return "memory" if "memory" in decision.strip().lower() else "research"