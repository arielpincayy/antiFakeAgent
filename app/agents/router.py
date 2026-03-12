from .agents import Agents


def router(query: str, memory: list, agents: Agents) -> str:
        if not memory:
            return "research"

        memory_topics = "\n".join([m[0] for m in memory])
        prompt = f"""
        The user asked the following question:

        "{query}"

        These are the previously researched topics:

        {memory_topics}

        Decide whether the question:
        1. Can be answered using the already researched topics.
        2. Requires a new academic search.

        Respond with ONLY one word:

        memory
        research
        """

        decision = agents.chat(
            messages=[
                {"role": "system", "content": "You are an intent classifier."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return "memory" if "memory" in decision.strip().lower() else "research"