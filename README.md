# agentic_emo

Agentic AI simulating human consciousness, unconsciousness, emotions and instincts.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     HumanMind                           │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  SUPERVISOR-1 (Unconscious)                       │  │
│  │  Fast/cheap model (gpt-4.1-mini, qwen-3b, etc.)  │  │
│  │                                                   │  │
│  │  Sub-agents (run in parallel):                    │  │
│  │   • fight_or_flight  • startle                    │  │
│  │   • disgust_withdrawal  • social_bonding          │  │
│  │   • curiosity_orienting                           │  │
│  │                                                   │  │
│  │  Output: emotion spikes (short + powerful)        │  │
│  └──────────────┬────────────────────────────────────┘  │
│                 │ emotion spikes                         │
│                 ▼                                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  EMOTION ENGINE (dynamic prompt variables)        │  │
│  │                                                   │  │
│  │  joy · fear · anger · sadness · surprise          │  │
│  │  disgust · trust · anticipation                   │  │
│  │                                                   │  │
│  │  Each = prompt fragment + intensity (0–1)         │  │
│  │  Decays over time; spiked by reflexes,            │  │
│  │  gently shifted by consciousness                  │  │
│  └──────────────┬────────────────────────────────────┘  │
│                 │ injected into system prompt            │
│                 ▼                                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  SUPERVISOR-2 (Conscious)                         │  │
│  │  Powerful model (gpt-4.1, claude, etc.)           │  │
│  │                                                   │  │
│  │  Sex-specific personality (male / female)         │  │
│  │  Tools: STM, LTM, emotion_adjust, think          │  │
│  │  Agentic loop: reason → use tools → respond       │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  MEMORY                                           │  │
│  │  • Short-term: sliding window (in-memory deque)   │  │
│  │  • Long-term: persistent JSON store with tags     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## How it works

1. A **stimulus** enters the mind (e.g. "A stranger shouts at you aggressively")
2. **Supervisor-1** dispatches it to all reflex sub-agents in parallel (fast model)
3. Reflexes that fire produce **emotion spikes** (short-duration, high-intensity)
4. The **Emotion Engine** updates — emotions are dynamic prompt fragments with intensity
5. **Supervisor-2** receives the stimulus + reflex results + current emotional state
6. It reasons using an agentic tool loop (memory recall/store, emotion adjustment, internal thought)
7. It produces a conscious **response** — what the human says or does

Emotions naturally **decay over time**. Reflex spikes decay fast; conscious shifts decay slowly.

## Setup

```bash
pip install -e .
```

Set your API key:
```bash
export OPENAI_API_KEY="sk-..."
# Optional: use a different provider
export OPENAI_BASE_URL="https://api.together.xyz/v1"
```

## Usage

```bash
# Male mind (default)
python -m agentic_emo.main --sex male

# Female mind
python -m agentic_emo.main --sex female

# Custom models
python -m agentic_emo.main --sex female \
  --fast-model qwen3-8b \
  --strong-model gpt-4.1 \
  --base-url http://localhost:11434/v1

# Verbose logging
python -m agentic_emo.main -v
```

### Interactive commands

| Command | Description |
|---|---|
| `/state` | Show full mind state |
| `/emotions` | Show current emotional state |
| `/stm` | Show short-term memory |
| `/ltm <query>` | Search long-term memory |
| `/quit` | Exit |

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | API key |
| `OPENAI_BASE_URL` | OpenAI default | API base URL |
| `AGENTIC_FAST_MODEL` | `gpt-4.1-mini` | Model for reflexes |
| `AGENTIC_STRONG_MODEL` | `gpt-4.1` | Model for consciousness |

## Project structure

```
agentic_emo/
├── __init__.py
├── main.py            # CLI entry point
├── mind.py            # HumanMind orchestrator
├── emotions.py        # Emotion Engine (dynamic prompt vars)
├── llm.py             # OpenAI-compatible LLM client
├── supervisor1.py     # Unconscious reflex coordinator
├── supervisor2.py     # Conscious reasoning agent
├── prompts/
│   ├── __init__.py
│   ├── profiles.py    # Male/female personality prompts
│   └── reflexes.py    # Reflex sub-agent prompts
└── memory/
    ├── __init__.py
    ├── short_term.py  # Sliding window STM
    └── long_term.py   # JSON-backed LTM
```
