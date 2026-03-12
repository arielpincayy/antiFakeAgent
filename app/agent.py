from typing import List, Dict, Any
import ast
import re

from .agents import router
from .agents import summarizer
from .scrapping import Scrapping
from .agents import Agents

OPENAI = ["gpt-4-turbo-preview"]
GROQ = ["openai/gpt-oss-20b"]
OLLAMA = ["qwen3.5:0.8b", "gemma3:1b"]


class ResearchAgent:
    def __init__(self):
        self.agent = Agents(provider="openai", model=OPENAI[0])
        self.router_agent = Agents(provider="groq", model=GROQ[0])
        self.scrapping = Scrapping()
        self.memory = []

    def generate_search_queries(self, query: str) -> List[str]:
        """
        Transforma la duda del usuario en términos de búsqueda optimizados,
        limpiando el formato Markdown si el modelo lo incluye.
        """
        prompt = f"""
        User Research Topic: "{query}"
    
        Task: Generate exactly 3 optimized search strings for academic databases.
        Rules:
        - Use Boolean operators (AND, OR).
        - Return ONLY a Python list of strings.
        - DO NOT include markdown code blocks, explanations, or any other text.
        - Example: ["term1 AND term2", "term3 OR term4"]
        """
        
        response = self.agent.chat(
            messages=[
                {"role": "system", "content": "You are a specialized API that returns only raw Python lists. No prose, no markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        # Limpieza robusta: eliminar bloques de código markdown y espacios extra
        clean_response = re.sub(r"```[a-z]*\n|```", "", response).strip()
    
        try:
            search_queries = ast.literal_eval(clean_response)
            if isinstance(search_queries, list):
                return search_queries[:3] 
            return [query]
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing search queries: {e}. Falling back to original query.")
            return [query]

    def collect_results(self, query: str, limit_per_source: int = 5) -> List[Dict[str, Any]]:
        print(f"Refining search terms for: {query}...")
        search_terms = self.generate_search_queries(query)
        
        all_results = []
        for term in search_terms:
            print(f"Searching for: {term}")
            results = self.scrapping.collect_results(term, limit_per_source)
            all_results.extend(results)

        unique_results = {p['link']: p for p in all_results}.values()
        return list(unique_results)

    def analyze_relevance(self, query: str, papers: List[Dict[str, Any]]) -> str:
        context = ""
        for i, p in enumerate(papers):
            summary = p['summary'] if p['summary'] else "Not available"
            context += f"ID: [{i}]\nTITLE: {p['title']}\n"
            context += f"SOURCE: {p['source']} | CITATIONS: {p['citations']}\n"
            context += f"SUMMARY: {summary}\n"
            context += f"LINK: {p['link']}\n---\n"

        prompt = f"""
        User Research Topic: "{query}"

        CONTEXT (Search Results):
        {context}

        TASK:
        As an expert Academic Librarian, evaluate the results above and select the 3-10 most impactful papers. 
        Present your analysis in a Markdown table for a Literature Review, followed by a brief synthesis.

        TABLE REQUIREMENTS:
        Include the following columns:
        1. **Paper ID & Title**: Short title and its reference ID.
        2. **Methodology**: Type of study (e.g., Qualitative, Experimental, Meta-analysis, etc.).
        3. **Key Results**: A concise summary of the most relevant findings.
        4. **Research Relevance**: Explain specifically why this paper is essential for the user's research topic.
        5. **Link**: The direct URL.

        SYNTHESIS:
        After the table, provide a 2-paragraph "State of the Art" summary that identifies common trends or gaps found in these selected papers.
        """

        print("Analyzing relevance with AI...")
        return self.agent.chat(
            messages=[
                {
                    "role": "system", 
                    "content": "You are a Senior Research Consultant. Your goal is to extract deep methodological insights and evaluate the academic rigor of provided sources. Output must be professional, objective, and formatted in clean Markdown."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 # Lower temperature for more consistent, analytical output
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