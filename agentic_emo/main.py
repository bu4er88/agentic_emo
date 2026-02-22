"""
CLI entry point — interactive conversation with a simulated human mind.

Usage:
    python -m agentic_emo.main --sex male
    python -m agentic_emo.main --sex female --fast-model gpt-4.1-mini --strong-model gpt-4.1

Environment variables:
    OPENAI_API_KEY      — API key for the LLM provider
    OPENAI_BASE_URL     — Base URL (for non-OpenAI providers)
    AGENTIC_FAST_MODEL  — Override fast model name
    AGENTIC_STRONG_MODEL — Override strong model name
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from agentic_emo.llm import LLMConfig
from agentic_emo.mind import HumanMind, MindConfig


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic AI simulating human consciousness and emotions",
    )
    parser.add_argument(
        "--sex", choices=["male", "female"], default="male",
        help="Biological sex affecting personality prompts (default: male)",
    )
    parser.add_argument(
        "--fast-model", default=None,
        help="Model for reflexes/unconscious (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--strong-model", default=None,
        help="Model for consciousness (default: gpt-4.1)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--ltm-path", default="data/long_term_memory.json",
        help="Path to long-term memory JSON file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Build config
    fast_cfg = LLMConfig(
        model=args.fast_model or "gpt-4.1-mini",
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0.3,
        max_tokens=256,
    )
    strong_cfg = LLMConfig(
        model=args.strong_model or "gpt-4.1",
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0.7,
        max_tokens=2048,
    )
    mind_cfg = MindConfig(
        sex=args.sex,
        fast_llm=fast_cfg,
        strong_llm=strong_cfg,
        ltm_path=args.ltm_path,
    )

    mind = HumanMind(mind_cfg)

    print(f"\n{'='*60}")
    print(f"  Agentic Emo — Simulated {args.sex.title()} Human Mind")
    print(f"  Reflexes: {fast_cfg.model} | Consciousness: {strong_cfg.model}")
    print(f"{'='*60}")
    print("  Type a stimulus (what happens to this person).")
    print("  Commands: /state  /emotions  /stm  /ltm <query>  /quit")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("stimulus> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # Meta-commands
        if user_input.startswith("/"):
            _handle_command(user_input, mind)
            continue

        # Process stimulus through the full mind pipeline
        try:
            result = mind.perceive(user_input)
        except Exception as exc:
            print(f"\n[ERROR] {exc}\n")
            continue

        # Display results
        _display_result(result)


def _handle_command(cmd: str, mind: HumanMind) -> None:
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()

    if command == "/quit":
        print("Goodbye.")
        sys.exit(0)

    elif command == "/state":
        print(f"\n{mind.introspect()}\n")

    elif command == "/emotions":
        print(f"\n{mind.emotions.render_prompt()}\n")

    elif command == "/stm":
        print(f"\n{mind.stm.render()}\n")

    elif command == "/ltm":
        query = parts[1] if len(parts) > 1 else ""
        if query:
            print(f"\n{mind.ltm.render_search(query)}\n")
        else:
            recent = mind.ltm.recent(5)
            if recent:
                print("\nRecent long-term memories:")
                for e in recent:
                    print(f"  - [{', '.join(e.tags) or 'no tags'}] {e.content}")
            else:
                print("\nLong-term memory is empty.")
            print()

    else:
        print(f"Unknown command: {command}")
        print("Commands: /state  /emotions  /stm  /ltm <query>  /quit")


def _display_result(result: dict) -> None:
    reflex = result["reflex_result"]
    conscious = result["conscious_result"]
    emotions = result["emotion_snapshot"]

    print()

    # Reflex output
    fired = reflex.get("fired_reflexes", [])
    if fired:
        names = [f"{r['reflex']}" for r in fired]
        print(f"  [REFLEXES] {', '.join(names)}")
        motor = reflex.get("overall_motor")
        if motor:
            print(f"  [BODY]     {motor}")

    # Emotions
    if emotions:
        emo_str = ", ".join(f"{k}={v:.0%}" for k, v in emotions.items())
        print(f"  [EMOTIONS] {emo_str}")

    # Conscious response
    print(f"\n  {conscious['response']}")

    # Tool usage summary
    tools_used = conscious.get("tool_log", [])
    if tools_used:
        tool_names = [t["call"].get("tool") for t in tools_used]
        print(f"\n  [tools used: {', '.join(tool_names)}]")

    print()


if __name__ == "__main__":
    cli()
