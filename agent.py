#!/usr/bin/env python3
"""
Boilerplate LangChain agent with selectable LLM provider.

Usage:
    python agent.py --provider bedrock "What is 123 * 456?"
    python agent.py --provider gemini "Add bacon and spaghetti to my list"

Environment variables:
    MODEL_PROVIDER=gemini|bedrock
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    GOOGLE_CLOUD_PROJECT=my-gcp-project
    GOOGLE_CLOUD_LOCATION=europe-west1
"""

import argparse
import json
import os
from datetime import datetime, timezone

import google.auth
from langchain.agents import create_agent
from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import calculator, add_item_to_list, get_common_items


def get_google_cloud_project() -> str | None:

    explicit_project = os.environ.get("GCP_PID")
    if explicit_project:
        return explicit_project

    try:
        _, detected_project = google.auth.default()
        return detected_project
    except Exception:
        return None


def create_llm(model_provider: str):

    selected_provider = model_provider.lower()

    if selected_provider == "gemini":
        project = get_google_cloud_project()
        location = os.environ.get("GCP_REGION", "europe-west1")
        model = os.environ.get("GEMINI_MODEL",  "gemini-2.5-flash")

        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS is not set. "
                "Set it to your GCP service account JSON file."
            )

        if not project:
            raise RuntimeError(
                "Could not determine GOOGLE_CLOUD_PROJECT. "
                "Set GOOGLE_CLOUD_PROJECT or GCP_PROJECT."
            )

        return ChatGoogleGenerativeAI(
            model=model,
            project=project,
            location=location,
            temperature=0,
            thinking_budget=-1,
            include_thoughts=True,
        )

    if selected_provider == "bedrock":
        model_id = os.environ.get("BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
        aws_profile = os.environ.get("AWS_PROFILE", "nimat")
        aws_region = os.environ.get("AWS_REGION", "eu-north-1")

        return ChatBedrock(
            model_id=model_id,
            region_name=aws_region,
            credentials_profile_name=aws_profile,
            model_kwargs={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 4096,
                },
            },
        )

    raise ValueError("Unsupported MODEL_PROVIDER. Use 'gemini' or 'bedrock'.")


# ── Agent ────────────────────────────────────────────────────────────────────

tools = [calculator, add_item_to_list, get_common_items]

SYSTEM_PROMPT = """
    You are an agent that helps the user manage their shopping list (supermarket list).

    Important rules to follow:
    1.  When a user wants to add items to the shopping list, double check with the list of most common items used by the user.
        If some terms in the items that the user wants to add are mispelled or look weird, double check the common items list and pick the one that has the highest potential of fitting what the user meant (e.g. closest matching).

    2.  Always avoid adding multiple times an item to the shopping list. Make sure that you are not creating duplicates before adding items to the shopping list.
"""


def create_shopping_agent(model_provider: str):

    llm = create_llm(model_provider)
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


# ── Trace ─────────────────────────────────────────────────────────────────────

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")


def _format_message(msg) -> str:
    """Return a Markdown section for a single LangChain message."""

    if isinstance(msg, HumanMessage):
        return f"## 🧑 Human\n\n{msg.content}\n"

    if isinstance(msg, AIMessage):
        blocks = msg.content_blocks
        reasoning_blocks = [b for b in blocks if b["type"] == "reasoning"]
        text_blocks = [b for b in blocks if b["type"] == "text"]

        parts: list[str] = ["## 🤖 AI\n"]

        if reasoning_blocks:
            reasoning = "".join(b["reasoning"] for b in reasoning_blocks)
            quoted = "\n".join(f"> {line}" for line in reasoning.splitlines())
            parts.append(f"### 💭 Thinking\n\n{quoted}\n")

        if text_blocks:
            text = "".join(b["text"] for b in text_blocks)
            parts.append(f"{text}\n")

        if msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = json.dumps(tc["args"], indent=2, ensure_ascii=False)
                parts.append(
                    f"### 🔧 Tool Call → `{tc['name']}`\n\n"
                    f"```json\n{args_str}\n```\n"
                )
        return "\n".join(parts)

    if isinstance(msg, ToolMessage):
        return (
            f"## 🛠️ Tool Result — `{msg.name}`\n\n"
            f"```\n{msg.content}\n```\n"
        )

    return f"## ❓ {type(msg).__name__}\n\n{getattr(msg, 'content', str(msg))}\n"


def write_trace(messages: list, query: str) -> str:
    """Write a Markdown trace of the agent run and return the file path."""
    os.makedirs(TRACES_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(TRACES_DIR, f"trace_{timestamp}.md")

    lines = [
        f"# Agent Trace — {timestamp}\n",
        f"**Query:** {query}\n",
        "---\n",
    ]
    for msg in messages:
        lines.append(_format_message(msg))
        lines.append("---\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


# ── Run ───────────────────────────────────────────────────────────────────────

def run(query: str, provider: str) -> str:

    agent = create_shopping_agent(provider)
    result = agent.invoke({"messages": [("human", query)]})

    trace_path = write_trace(result["messages"], query)
    print(f"Trace   : {trace_path}")

    return result["messages"][-1].content


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Shopping-list agent with Gemini or Bedrock")
    parser.add_argument(
        "--provider",
        "-p",
        choices=["gemini", "bedrock"],
        default=os.environ.get("MODEL_PROVIDER", "bedrock").lower(),
        help="LLM provider to use",
    )
    parser.add_argument("query", nargs="*", help="Question or shopping-list request")
    return parser.parse_args()


def main():

    args = parse_args()
    query = " ".join(args.query) if args.query else "What is (123 * 456) + 789?"
    print(f"Provider: {args.provider}")
    print(f"Query   : {query}")
    print(f"Answer  : {run(query, args.provider)}")


if __name__ == "__main__":
    main()
