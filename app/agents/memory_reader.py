from typing import List

from .agents import Agents


def memory_readery(query: str, memory: List, agent:Agents) -> str:
        context = ""
        for topic, _, analysis in memory:
            context += f"RESEARCHED TOPIC: {topic}\n"
            context += f"ANALYSIS:\n{analysis}\n\n"

        prompt = f"""
        The user asks:

        {query}

        Use ONLY the following previously researched information to answer.

        {context}

        If the information is not sufficient, say so.
        """

        return agent.chat(
            messages=[
                {"role": "system", "content": "You are an academic research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )