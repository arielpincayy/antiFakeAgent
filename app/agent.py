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
    query: str
    results: List[Dict[str, Any]]
    iteration: int
    route: str
    enough: bool
    analysis: List[Dict[str, Any]]
    final_answer: str

class ResearchAgent:
    def __init__(self):
        self.agent = Agents(provider="openai", model=OPENAI[0])
        self.router_agent = Agents(provider="groq", model=GROQ[0])
        self.ner_model = NERModel()
        self.web_search = ["arxiv", "openalex", "google_scholar", "google_web"]
        self.scientific = ["google_web", "arxiv", "google_scholar", "openalex"]

    def router_bases(self, state: AgentState):
        new_route = router_bases(state['claim'], self.router_agent)
        return {'route': new_route}
    
    def generate_queries(self, state: AgentState):
        query, entities = self.ner_model.get_query_and_entities(state["claim"])
        new_query = router_create_queries(self.agent, state['claim'], query, entities)
        return {'query': new_query}

    def summarizer(self, state: AgentState):
        new_claim = agent_summarizer(self.agent, state["claim"])
        return {'claim': new_claim}

    def analyzer(self, state: AgentState):
        new_analysis = agent_analizer(self.agent, state['results'], state['claim'])
        return {'analysis': new_analysis}
    
    def router_enough(self, state: AgentState):
        max_iter = len(self.web_search) - 1
        new_enough = router_enough(self.router_agent, state['analysis'], state['claim'], state["iteration"], max_iter)
        return {'enough': new_enough, 'iteration': state['iteration'] + 1}
    
    def retrieve_information(self, state: AgentState):
        base = self.web_search if state['route'] else self.scientific
        new_results = collect_results(state['claim'], base[state["iteration"]])
        return {'results': [*state['results'], *new_results]}
    
    def sythetyzer(self, state: AgentState):
        final_answer = agent_synthetizer(self.agent, state['analysis'], state['claim'])
        return {'final_answer': final_answer}
    

    def run(self, news: str) -> str:
        workflow = StateGraph(AgentState)

        workflow.add_node("summarizer", self.summarizer)
        workflow.add_node("router_bases", self.router_bases)
        workflow.add_node("generate_queries", self.generate_queries)
        workflow.add_node("analyzer", self.analyzer)
        workflow.add_node("check_enough", self.router_enough)
        workflow.add_node("retrieve_information", self.retrieve_information)
        workflow.add_node("synthetizer", self.sythetyzer)

        ######################
        ### INIT WORKFLOW ####
        ######################

        workflow.set_entry_point("summarizer")


        workflow.add_edge("summarizer", "router_bases")
        workflow.add_edge("router_bases", "generate_queries")
        workflow.add_edge("generate_queries", "retrieve_information")
        workflow.add_edge("retrieve_information", "analyzer")
        workflow.add_edge("analyzer", "check_enough")

        ######################
        ##### START LOOP #####
        ######################

        workflow.add_conditional_edges(
            "check_enough",
            lambda state: "end" if state['enough'] else "continue",
            {   
                "end":"synthetizer",
                "continue":"retrieve_information"
            }
        )

        workflow.add_edge("synthetizer", END)

    
        initial_state: AgentState = {
            "claim": news,
            "query": "",
            "results": [],
            "iteration": 0,
            "route": "",
            "enough": False,
            "analysis": [],
            "final_answer": ""
        }

        app = workflow.compile()

        for step in app.stream(initial_state):
            print("STEP:", step)

        final_state:AgentState = app.invoke(initial_state)

        return final_state['final_answer']