# FactCheck AI — Project Documentation

> An autonomous multi-source fact-checking agent built with LangGraph, NER-powered query generation, and a RAG-based chat interface.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Core Components](#core-components)
   - [Entry Points](#entry-points)
   - [Agent Workflow](#agent-workflow)
   - [Agents Module](#agents-module)
   - [NER Module](#ner-module)
   - [Scrapers Module](#scrapers-module)
   - [RAG Module](#rag-module)
5. [Data Flow](#data-flow)
6. [Agent Graph](#agent-graph)
7. [Configuration & Environment Variables](#configuration--environment-variables)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Design Decisions](#design-decisions)

---

## Overview

FactCheck AI is an agentic pipeline that takes a news claim as input and autonomously determines whether it is **true**, **false**, or **uncertain**. The system:

- Summarizes and normalizes the input claim
- Routes it to the appropriate search strategy (scientific databases or general web)
- Extracts named entities with a custom NER model to generate structured search queries
- Iterates over multiple sources (arXiv, Google Scholar, OpenAlex, Google Web) until enough evidence is collected
- Produces a structured fact-check report in Markdown
- Activates a RAG-based chat interface so users can ask follow-up questions grounded in the retrieved evidence

---

## Architecture

```
User Input (claim)
       │
       ▼
┌─────────────────────────────────────────────────┐
│                  LangGraph Workflow              │
│                                                 │
│  Summarizer → Router → NER Query Gen            │
│       ↓                                         │
│  Retrieve (arXiv / Scholar / OpenAlex / Web)    │
│       ↓                                         │
│  Analyzer → Enough? ──No──► Retrieve (loop)     │
│                 │                               │
│                Yes                              │
│                 ↓                               │
│           Synthetizer                           │
└─────────────────────────────────────────────────┘
       │
       ▼
 Markdown Report + Analysis list
       │
       ▼
  RAG Index built (Ollama embeddings)
       │
       ▼
  Chat interface (context-grounded Q&A)
```

---

## Project Structure

```
project/
│
├── run.py                      # CLI entry point
├── streamlit_app.py            # Web UI entry point
│
├── app/
│   ├── __init__.py             # Exports ResearchAgent, collect_results
│   ├── agent.py                # LangGraph workflow + AgentState
│   ├── ner.py                  # Custom NER model + query generation
│   ├── rag.py                  # Vector store + RAG chat
│   ├── scrapping.py            # Dispatcher for all scrapers
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agents.py           # LangChain LLM wrapper (Agents class)
│   │   ├── agent_analizer.py   # Two-phase relevance filter + content analysis
│   │   ├── agent_summarizer.py # Claim normalization
│   │   ├── agent_synthetizer.py# Final verdict + report generation
│   │   ├── router_bases.py     # web_search vs scientific router
│   │   ├── router_create_queries.py  # Fallback query generator
│   │   └── router_enough.py   # Stopping criterion evaluator
│   │
│   └── scrappers/
│       ├── __init__.py
│       ├── arxiv.py            # arXiv API client
│       ├── googlescholar.py    # Google Scholar + Google Web (SerpApi)
│       └── openalex.py         # OpenAlex API client
```

---

## Core Components

### Entry Points

#### `run.py` — CLI Interface

The command-line interface. Starts an interactive loop where the user enters a claim and receives a printed Markdown report. After the initial analysis, a RAG index is built and the session switches to chat mode.

**Special commands:**
- `/bye` — exit the program
- `/new` — clear state and verify a new claim

#### `streamlit_app.py` — Web UI

A full Streamlit-based web interface with the same logic as `run.py`. Organized into three tabs after verification:

- **Report** — rendered Markdown report with download button
- **Sources** — all collected sources with credibility scores
- **Ask AI** — RAG-powered chat grounded in the retrieved evidence

---

### Agent Workflow

**File:** `app/agent.py`

Defines the `AgentState` (a `TypedDict`) and the `ResearchAgent` class, which compiles and runs a LangGraph `StateGraph`.

#### `AgentState` fields

| Field | Type | Description |
|---|---|---|
| `claim` | `str` | The current (possibly summarized) claim |
| `queries` | `List[str]` | All NER-generated queries |
| `query_index` | `int` | Index of the active query |
| `results` | `List[Dict]` | Accumulated raw retrieval results |
| `iteration` | `int` | Index of the active source within the current query |
| `route` | `str` | `"web_search"` or `"scientific"` |
| `enough` | `bool` | Whether the stopping criterion has been met |
| `analysis` | `List[Dict]` | Filtered, analyzed evidence items |
| `final_answer` | `str` | The final Markdown report |

#### LLM Providers used

| Role | Provider | Model |
|---|---|---|
| Main analysis & synthesis | OpenAI | `gpt-4-turbo` |
| Routing & stopping checks | Groq | `openai/gpt-oss-20b` |

The router agent uses Groq for fast, cheap binary decisions; the heavier analysis is delegated to the OpenAI model.

---

### Agents Module

#### `agents.py` — LLM Wrapper

`Agents` wraps LangChain's `init_chat_model` to provide a simple `.invoke(messages)` interface that returns a plain string. Accepts any provider and model supported by LangChain.

#### `agent_summarizer.py` — Claim Normalization

Condenses the raw user input to ≤100 words, preserving all verifiable entities (people, organizations, locations) and the core claim. This reduces noise in downstream NER and retrieval steps.

#### `agent_analizer.py` — Two-Phase Evidence Filter

The most complex agent. Processes raw retrieval results in two sequential phases to avoid wasting tokens:

**Phase 1 — Batch title filter**
Sends all titles in a single prompt and asks the LLM to mark each as `RELEVANT` or `IRRELEVANT`. Only relevant titles advance to Phase 2.

**Phase 2 — Deep content analysis**
For each surviving item, asks the LLM to:
- Confirm the content is actually about the claim (not just the title)
- Summarize the content in 2–3 lines
- Extract key points
- Determine the stance: `Supports / Refutes / Neutral`
- Rate relevance (`High / Medium / Low`) and source credibility (`High / Medium / Low`)

Items rated `Low` relevance are discarded. The output is a structured dict per item, which becomes a node in the RAG index.

#### `agent_synthetizer.py` — Verdict & Report

Receives the filtered analysis list and generates the final Markdown report with these sections:

- **Veredicto** — `VERDADERA / FALSA / INCIERTA`
- **Explicación** — reasoning paragraph
- **Evidencia Clave** — bullet list of the most relevant points
- **Calidad de las Fuentes** — source reliability assessment
- **Nivel de Confianza** — `Alto / Medio / Bajo`

#### `router_bases.py` — Search Strategy Router

Classifies the claim as either `"web_search"` (recent events, news, public statements) or `"scientific"` (established knowledge, studies, empirical claims). Determines which source list is used during retrieval.

#### `router_enough.py` — Stopping Criterion

After each analysis step, evaluates whether the accumulated evidence is sufficient to issue a verdict. Returns `True` if **both** of these conditions hold:
1. At least one source directly addresses the claim (relevance High or Medium)
2. At least one source has credibility High or Medium

Importantly, it does **not** judge whether the claim is true — only whether there is enough material to make that judgment.

#### `router_create_queries.py` — Fallback Query Generator

Used when the NER model extracts no entities. Generates a single clean search query from the claim text using the LLM.

---

### NER Module

**File:** `app/ner.py`

Uses a fine-tuned token classification model (checkpoint at `app/modelo_ner_metricas/checkpoint-759`) loaded via HuggingFace `transformers`.

#### Entity types recognized

| Mode | Entity types |
|---|---|
| `scientific` | `TECHNOLOGY`, `METHOD`, `ORGANIZATION` |
| `web_search` | + `PERSON`, `LOCATION` |

#### Query generation strategy

1. Extracts entities from the summarized claim and groups them by type.
2. Builds OR-groups per type: `("transformer" OR "BERT")`
3. Generates every combination of AND-joined groups across types, from least to most specific:

```
("transformer")
("encoder")
("transformer") AND ("encoder")
("transformer") AND ("Google")
("encoder") AND ("Google")
("transformer") AND ("encoder") AND ("Google")
```

This produces a query list ordered from broad to specific, which the workflow iterates through until evidence is sufficient.

---

### Scrapers Module

**File:** `app/scrapping.py` + `app/scrappers/`

A dispatcher (`collect_results`) that routes to one of four sources based on the `base` parameter:

| Source | File | API | Returns |
|---|---|---|---|
| **arXiv** | `arxiv.py` | arXiv Atom API (free) | title, abstract, link |
| **Google Scholar** | `googlescholar.py` | SerpApi | title, snippet, cited_by, link |
| **OpenAlex** | `openalex.py` | OpenAlex REST API (free) | title, abstract (reconstructed), DOI, citation count |
| **Google Web** | `googlescholar.py` | SerpApi | title, snippet, URL |

OpenAlex abstracts are stored as inverted indexes (word → positions); `reconstruct_abstract()` reassembles them in correct word order.

All results are normalized to a common dict schema before being stored in `AgentState.results`:

```python
{
    "source": str,      # "arXiv" | "Google Scholar" | "OpenAlex" | "Web"
    "title": str,
    "summary": str,
    "link": str,
    "citations": int | "N/A",
    "analyzed": bool    # flag to avoid re-analyzing the same item
}
```

---

### RAG Module

**File:** `app/rag.py`

A lightweight in-memory vector store built after the workflow completes.

#### Build phase

Each analyzed evidence item is converted to a plain-text chunk:
```
Título: ...
Resumen: ...
Puntos clave: ...
Relevancia: ... | Credibilidad: ...
Fuente: ... — <link>
```

A special chunk containing the full final report is appended. All chunks are embedded using `OllamaEmbeddings` (`nomic-embed-text` model, must be running locally).

#### Retrieval

Cosine similarity over NumPy arrays. Returns the top-k most relevant chunks for a given query.

#### Chat

Retrieves the top-4 chunks, injects them as a system-prompt context, and calls the main `Agents` LLM with the full conversation history. Answers are strictly grounded in the retrieved context.

---

## Data Flow

```
User claim
    │
    ▼ agent_summarizer
Normalized claim (≤100 words)
    │
    ▼ router_bases
Route: "web_search" | "scientific"
    │
    ▼ NERModel.get_queries_and_entities
queries = [q1, q2, ..., qN]  (ordered broad → specific)
    │
    ▼ collect_results(queries[0], sources[0])
Raw results (title, summary, link, ...)
    │
    ▼ agent_analizer  (Phase 1: title filter → Phase 2: content analysis)
Filtered analysis items
    │
    ▼ router_enough
enough=True? ──No──► next source / next query ──► collect_results (loop)
    │
   Yes
    │
    ▼ agent_synthetizer
Final Markdown report
    │
    ▼ RAGStore.build(analysis, claim, report)
In-memory vector index
    │
    ▼ RAGStore.chat(history, question, claim)
Grounded Q&A responses
```

---

## Agent Graph

```
[summarizer]
     │
[router_bases]
     │
[generate_queries]
     │
[retrieve_information]  ◄──────────────────┐
     │                                     │
[analyzer]                                 │
     │                                     │
[check_enough]                             │
     │                                     │
  enough? ──── No ────────────────────────┘
     │
    Yes
     │
[synthetizer]
     │
    END
```

The loop increments `iteration` (cycling through sources) and `query_index` (cycling through queries) until either the stopping criterion is met or all queries are exhausted.

---

## Configuration & Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI — main analysis LLM
OPENAI_API_KEY=sk-...

# Groq — routing LLM
GROQ_API_KEY=gsk_...

# SerpApi — Google Scholar and Google Web search
GOOGLESCHOLAR=your-serpapi-key

# OpenAlex (optional — increases rate limits)
OPENALEX=your-openalex-key
```

Ollama must be running locally with `nomic-embed-text` pulled for the RAG embeddings:

```bash
ollama pull nomic-embed-text
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd factcheck-ai

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Pull the Ollama embedding model
ollama pull nomic-embed-text
```

Key dependencies: `langgraph`, `langchain`, `transformers`, `torch`, `langchain-ollama`, `streamlit`, `rich`, `serpapi`, `requests`, `python-dotenv`.

---

## Usage

### CLI

```bash
python run.py
```

Enter a claim at the `User:` prompt. After the report is printed, the session enters RAG chat mode. Type `/new` to verify another claim or `/bye` to exit.

### Web UI

```bash
streamlit run streamlit_app.py
```

Opens in the browser. Enter a claim or select an example, click **Verify Claim**, and explore the results across the Report, Sources, and Ask AI tabs.

---

## Design Decisions

**Two LLMs for different tasks** — Groq handles fast binary routing decisions (web vs scientific, enough vs not enough) at low cost and latency. OpenAI GPT-4 handles the deeper analysis and synthesis where quality matters most.

**Two-phase analysis filter** — Phase 1 eliminates irrelevant items with a single batch LLM call (cheap). Phase 2 does deep analysis only on survivors (expensive). This avoids sending full article content for obviously unrelated results.

**NER-driven query combinatorics** — Rather than generating one query with an LLM, the NER model extracts typed entities and generates every logical AND/OR combination. This covers more of the search space systematically and avoids hallucinated queries.

**Iteration order: sources before queries** — The loop exhausts all sources for a given query before moving to the next query. This means the broadest query is tried against all sources first, and only if evidence is still insufficient does the system move to more specific queries.

**RAG over the analysis, not the raw results** — The vector index is built from the filtered, analyzed items (not raw scraped text). This means retrieval in chat mode is over curated, claim-relevant content, improving answer quality.

**`analyzed` flag on results** — Each raw result carries an `analyzed: bool` flag. The analyzer skips already-processed items, so appending new results across loop iterations never causes duplicate work.