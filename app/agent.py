from typing import List, Dict, Any

from .agents import router
from .agents import summarizer
from .scrapping import Scrapping
from .agents import Agents


class ResearchAgent:
    def __init__(self):
        self.agent = Agents(provider="groq", model="openai/gpt-oss-20b")
        self.router_agent = Agents(provider="ollama", model="qwen3.5:0.8b")
        self.scrapping = Scrapping()
        self.memory = []

    def collect_results(self, query: str, limit_per_source: int = 5) -> List[Dict[str, Any]]:
        return self.scrapping.collect_results(query, limit_per_source)

    def analyze_relevance(self, query: str, papers: List[Dict[str, Any]]) -> str:
        context = ""
        for i, p in enumerate(papers):
            summary = p['summary'] if p['summary'] else "Not available"
            context += f"[{i}] TITLE: {p['title']}\n"
            context += f"SOURCE: {p['source']} | CITATIONS: {p['citations']}\n"
            context += f"SUMMARY: {summary}...\n\n"

        prompt = f"""
        Act as an expert academic librarian. The user is researching: "{query}".
        Below is a list of papers retrieved from several databases.
        
        Your task:
        1. Select the 3-5 most relevant papers for this topic.
        2. Briefly explain WHY they are significant.
        3. Provide a concise synthetic summary of the findings.

        SEARCH RESULTS:
        {context}
        """

        print("Analyzing relevance with AI...")
        return self.agent.chat(
            messages=[
                {"role": "system", "content": "You are a high level research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

    def handle_memory(self, max_entries: int = 5):
        mem = summarizer(self.memory, self.agent, max_entries)
        self.memory = mem

    def route_query(self, query: str) -> str:
        """
        Decides whether to use memory or conduct new research.
        Returns: 'memory' or 'research'
        """
        return router(query, self.memory, self.router_agent)

    def answer_from_memory(self, query: str) -> str:
        context = ""
        for topic, _, analysis in self.memory:
            context += f"RESEARCHED TOPIC: {topic}\n"
            context += f"ANALYSIS:\n{analysis}\n\n"

        prompt = f"""
        The user asks:

        {query}

        Use ONLY the following previously researched information to answer.

        {context}

        If the information is not sufficient, say so.
        """

        return self.agent.chat(
            messages=[
                {"role": "system", "content": "You are an academic research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

    def run(self, user_topic: str):
        mode = self.route_query(user_topic)

        if mode == "memory":
            print("Using previous knowledge...")
            return self.answer_from_memory(user_topic)

        print("Starting deep research...")
        raw_data = self.collect_results(user_topic)

        if not raw_data:
            return "No results found on any platform."

        response = self.analyze_relevance(user_topic, raw_data)
        self.memory.append((user_topic, raw_data, response))
        self.handle_memory()

        return response