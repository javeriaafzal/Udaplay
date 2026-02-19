# UdaPlay

UdaPlay is an AI research agent focused on video games. It answers questions about:

- game titles and details
- release dates and platforms
- genres and descriptions
- publisher information

  Part of the Udacity coursework 

## Architecture

UdaPlay uses a two-tier retrieval strategy:

1. **Primary retrieval (RAG)** using ChromaDB over local game JSON data.
2. **Secondary fallback (Web)** using Tavily search when local retrieval confidence is low.

The workflow is implemented as a state machine:

1. `retrieve_game`
2. `evaluate_retrieval`
3. `game_web_search` (conditional fallback)
4. response generation + report formatting

## Install

```bash
pip install -e .
```

## Environment

Create a `.env` file (or export variables):

```bash
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

> OpenAI is used for final natural-language response generation.
> Tavily is used only when retrieval quality is insufficient.

## Usage

First ingest local data and ask a question:

```bash
python -m udaplay.main "Who published Elden Ring and what platforms is it on?" --rebuild
```

Run without `--rebuild` to query an already persisted ChromaDB index:

```bash
python -m udaplay.main "When did Hades release?"
```

Run an interactive multi-query session (keeps conversation context in-memory):

```bash
python -m udaplay.main --session
```

## Project Structure

- `udaplay/vector_store.py`: ChromaDB manager and semantic search.
- `udaplay/agent.py`: agent tools + state machine.
- `udaplay/reporting.py`: markdown reporting output.
- `udaplay/models.py`: pydantic contracts.
- `data/games.json`: sample game dataset.
