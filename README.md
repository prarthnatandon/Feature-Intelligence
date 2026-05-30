# Feature Intelligence

Retrieval-augmented multi-agent system that scrapes real user feedback, runs a 4-agent AI pipeline with semantic retrieval, and produces PM-ready feature opportunity briefs — with built-in evaluation framework and cost observability.

**Not a wrapper. Not a web search tool.** This is a production-grade agentic system with RAG, structured tool use, extended thinking, SSE streaming, and automated quality measurement.

![Python](https://img.shields.io/badge/python-3.11+-blue) ![Claude API](https://img.shields.io/badge/Claude-Sonnet%204-blueviolet) ![License](https://img.shields.io/badge/license-MIT-green)

---

## What It Does

Enter any company name. The system:

1. **Scrapes** real feedback from Reddit, App Store, Google Play, and Hacker News (no auth required)
2. **Indexes** all feedback into a TF-IDF vector store for semantic retrieval
3. **Dispatches** 4 specialist AI agents in a 2-wave parallel architecture
4. **Synthesizes** results via an Orchestrator with extended thinking (5K token reasoning budget)
5. **Evaluates** output quality with a 4-dimension eval framework
6. **Tracks** API cost per agent with dollar-level granularity

Output: a ranked Feature Opportunity Brief grounded in user evidence — the kind of document a VP of Product would read in a meeting.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │   5 Data Sources         │
                    │  Reddit · App Store      │
                    │  Google Play · HN · Seed │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │  TF-IDF Vector Store     │
                    │  (RAG Retrieval Layer)   │
                    │  8K features · bigrams   │
                    │  cosine similarity       │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              │           WAVE 1 (parallel)     │
              │                                 │
     ┌────────▼────────┐            ┌───────────▼───────┐
     │   ThemeAgent    │            │    QuoteAgent      │
     │   RAG: diverse  │            │    RAG: emotional  │
     │   cluster query │            │    resonance query │
     └────────┬────────┘            └───────────┬───────┘
              │                                 │
              ├─────────────────────────────────┤
              │           WAVE 2 (parallel)     │
              │         (depends on themes)     │
              │                                 │
     ┌────────▼────────┐            ┌───────────▼───────┐
     │ FeasibilityAgent│            │    GapAgent        │
     │ RAG: per-theme  │            │    Product feature │
     │ evidence lookup │            │    cross-reference │
     └────────┬────────┘            └───────────┬───────┘
              │                                 │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │     Orchestrator         │
                    │  Extended Thinking (5K)  │
                    │  Structured tool use     │
                    │  6 brief sections        │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼──────┐ ┌──────▼──────┐ ┌───────▼───────┐
     │ Feature Brief │ │ Cost Report │ │  Eval Suite   │
     │ (Markdown)    │ │ (per-agent) │ │ (4 dimensions)│
     └───────────────┘ └─────────────┘ └───────────────┘
```

### Why This Architecture Matters

| Design Decision | What It Solves |
|---|---|
| **RAG retrieval instead of context-stuffing** | Original approach truncated at 120 items. RAG uses semantic search to give each agent the *most relevant* feedback, scaling to thousands of items |
| **Per-agent retrieval strategies** | ThemeAgent gets diverse clusters. FeasibilityAgent gets per-theme evidence. QuoteAgent gets emotionally resonant text. Same store, different queries |
| **2-wave parallelism** | Wave 1 agents are independent → run simultaneously. Wave 2 agents depend on themes → wait for Wave 1, then parallelize |
| **Eval framework** | Quote grounding detects hallucinated evidence. LLM-as-judge scores brief quality. Theme stability measures consistency across runs |
| **Cost tracking** | Per-agent token usage and dollar costs. Agentic systems with 4+ agents can burn $5-10/run — making cost visible is operational necessity |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Claude Sonnet 4 (tool use + extended thinking + streaming) |
| **RAG** | scikit-learn TF-IDF vectorizer + cosine similarity |
| **Backend** | FastAPI, asyncio, SSE streaming |
| **Data** | aiohttp (Reddit JSON, App Store RSS, HN Algolia), google-play-scraper |
| **Models** | Pydantic v2 (typed data flow between all pipeline stages) |
| **Evals** | LLM-as-judge, Jaccard similarity, quote grounding verification |
| **Frontend** | Vanilla JS + D3.js (SSE client, opportunity matrix, theme charts) |

---

## Quickstart

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/feature-intelligence.git
cd feature-intelligence

# Set up
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# Install
pip install -r requirements.txt

# Run
uvicorn backend.main:app --reload --port 8000

# Open http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Start analysis for any company |
| `GET` | `/stream/{run_id}` | SSE stream of real-time agent progress |
| `GET` | `/brief/{run_id}` | Get completed feature brief (JSON) |
| `GET` | `/brief/{run_id}/markdown` | Get brief as downloadable Markdown |
| `GET` | `/cost/{run_id}` | Per-agent cost breakdown |
| `POST` | `/eval/{run_id}` | Run evaluation suite on completed analysis |
| `GET` | `/eval/history` | Historical eval results for trend analysis |

---

## Evaluation Framework

The eval suite measures pipeline quality across four dimensions:

| Dimension | Method | What It Catches |
|---|---|---|
| **Quote Grounding** | TF-IDF similarity search against source data | Hallucinated quotes — claims not backed by real feedback |
| **Brief Quality** | LLM-as-judge with structured rubric | Vague recommendations, weak evidence, poor coherence |
| **Retrieval Quality** | Relevance scores, coverage, source diversity | RAG returning irrelevant or biased results |
| **Theme Stability** | Jaccard + fuzzy overlap across N runs | Inconsistent theme identification between runs |

Run evals via API:
```bash
# After completing an analysis:
curl -X POST http://localhost:8000/eval/{run_id}

# View historical results:
curl http://localhost:8000/eval/history
```

---

## RAG Layer Details

The retrieval layer (`backend/rag.py`) replaces naive context-stuffing with semantic retrieval:

**Why TF-IDF over dense embeddings?**
- Zero additional API cost (no embedding model calls)
- Sub-second indexing for <10K documents
- Bigram features capture product-specific phrases ("streak anxiety", "AI tutor")
- For this corpus size, TF-IDF with cosine similarity rivals dense retrieval quality

**Per-agent retrieval strategies:**
- `for_theme_agent()` — 8 diverse seed queries + upvote-weighted sampling → broad cluster discovery
- `for_feasibility_agent()` — per-theme retrieval → concrete evidence for each feasibility rating
- `for_quote_agent()` — emotion-targeted queries ("I wish I could", "frustrating because") → compelling evidence

**Production upgrade path:** swap TF-IDF for Voyage AI embeddings + Supabase pgvector by implementing the `VectorStore` protocol with a different backend.

---

## Cost Tracking

Every API call is recorded with input/output token counts and estimated dollar costs:

```json
{
  "total_cost_usd": 0.3510,
  "total_api_calls": 6,
  "by_agent": {
    "ThemeAgent": {"cost_usd": 0.0765, "total_tokens": 11500, "api_calls": 2},
    "QuoteAgent": {"cost_usd": 0.0345, "total_tokens": 5500, "api_calls": 1},
    "Orchestrator": {"cost_usd": 0.1440, "total_tokens": 16000, "api_calls": 2}
  }
}
```

---

## Project Structure

```
├── backend/
│   ├── agents.py          # 4 specialist agents + orchestrator + pipeline
│   ├── rag.py             # TF-IDF vector store + per-agent retrieval
│   ├── evals.py           # 4-dimension evaluation framework
│   ├── cost_tracker.py    # Per-agent token + dollar cost tracking
│   ├── scraper.py         # Reddit, App Store, Google Play, HN scrapers
│   ├── models.py          # Pydantic v2 models for all pipeline stages
│   ├── prompts.py         # Agent system prompts (company-agnostic)
│   └── main.py            # FastAPI server + SSE + eval/cost endpoints
├── frontend/
│   ├── index.html         # Landing + running + results states
│   ├── dashboard.js       # SSE client, state machine, rendering
│   ├── charts.js          # CSS bar chart animations
│   └── visualization.js   # D3 skill dependency graph
├── requirements.txt
├── Procfile               # Heroku/Render deployment
└── runtime.txt
```

---

## Built With

[Anthropic Claude API](https://docs.anthropic.com) — tool use, extended thinking, streaming

---

## License

MIT
