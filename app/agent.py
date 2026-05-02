from typing import List, Dict, TypedDict, Any

from langgraph.graph import StateGraph, END

from .agents import Agents, router_bases, router_create_queries, agent_analizer, router_enough, agent_summarizer, agent_synthetizer
from .scrapping import collect_results
from .ner import NERModel

OPENAI = ["gpt-4-turbo", "gpt-4o-mini", "gpt-3.5-turbo"]
GROQ = ["openai/gpt-oss-20b"]
OLLAMA = ["qwen3.5:0.8b", "gemma3:1b"]

class AgentState(TypedDict):
    claim: str
    queries: List[str]        # lista de queries generadas por el NER
    query_index: int          # índice de la query actual
    results: List[Dict[str, Any]]
    iteration: int            # índice de la base actual (se reinicia al cambiar de query)
    route: str
    enough: bool
    analysis: List[Dict[str, Any]]
    final_answer: str

class ResearchAgent:
    def __init__(self):
        self.agent = Agents(provider="openai", model=OPENAI[0])
        self.router_agent = Agents(provider="groq", model=GROQ[0])
        self.ner_model = NERModel()
        self.web_search = ["google_scholar", "openalex", "arxiv", "google_web"]
        self.scientific = ["google_scholar", "openalex", "arxiv", "google_web"]

    def router_bases(self, state: AgentState):
        new_route = router_bases(state['claim'], self.router_agent)
        return {'route': new_route}

    def generate_queries(self, state: AgentState):
        queries, entities = self.ner_model.get_queries_and_entities(state["claim"], state["route"])
        return {'queries': queries, 'query_index': 0}

    def summarizer(self, state: AgentState):
        new_claim = agent_summarizer(self.agent, state["claim"])
        return {'claim': new_claim}

    def analyzer(self, state: AgentState):
        new_analysis = agent_analizer(self.agent, state['results'], state['claim'])
        return {'analysis': [*state['analysis'], *new_analysis]}

    def router_enough(self, state: AgentState):
        bases = self.web_search if state['route'] else self.scientific
        new_iteration = state['iteration'] + 1
        new_query_index = state['query_index']

        # Si ya se recorrieron todas las bases, avanzar a la siguiente query
        if new_iteration >= len(bases):
            new_iteration = 0
            new_query_index += 1

        new_enough = router_enough(
            self.router_agent, state['analysis'], state['claim']
        )

        return {
            'enough': new_enough,
            'iteration': new_iteration,
            'query_index': new_query_index,
        }

    def retrieve_information(self, state: AgentState):
        bases = self.web_search if state['route'] else self.scientific
        base = bases[state['iteration']]
        query = state['queries'][state['query_index']]
        new_results = collect_results(query, base)
        return {'results': [*state['results'], *new_results]}

    def synthetizer(self, state: AgentState):
        final_answer = agent_synthetizer(self.agent, state['analysis'], state['claim'])
        return {'final_answer': final_answer}

    def _should_continue(self, state: AgentState) -> str:
        if state['enough']:
            return "end"
        # Parar si ya se agotaron todas las queries
        if state['query_index'] >= len(state['queries']):
            return "end"
        return "continue"

    def run(self, news: str) -> str:
        workflow = StateGraph(AgentState)

        workflow.add_node("summarizer", self.summarizer)
        workflow.add_node("router_bases", self.router_bases)
        workflow.add_node("generate_queries", self.generate_queries)
        workflow.add_node("analyzer", self.analyzer)
        workflow.add_node("check_enough", self.router_enough)
        workflow.add_node("retrieve_information", self.retrieve_information)
        workflow.add_node("synthetizer", self.synthetizer)

        workflow.set_entry_point("summarizer")
        workflow.add_edge("summarizer", "router_bases")
        workflow.add_edge("router_bases", "generate_queries")
        workflow.add_edge("generate_queries", "retrieve_information")
        workflow.add_edge("retrieve_information", "analyzer")
        workflow.add_edge("analyzer", "check_enough")

        workflow.add_conditional_edges(
            "check_enough",
            self._should_continue,
            {
                "end": "synthetizer",
                "continue": "retrieve_information"
            }
        )

        workflow.add_edge("synthetizer", END)

        initial_state: AgentState = {
            "claim": news,
            "queries": [],
            "query_index": 0,
            "results": [],
            "iteration": 0,
            "route": "",
            "enough": False,
            "analysis": [],
            "final_answer": ""
        }

        app = workflow.compile()

        final_state: AgentState = app.invoke(initial_state)
        return final_state['final_answer'], final_state['analysis']