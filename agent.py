#!/usr/bin/env python3
"""
Boilerplate LangChain agent using AWS Bedrock (Claude Sonnet 4.5).

Usage:
    python agent.py
    python agent.py "What is 123 * 456?"

Requires AWS credentials configured (via env vars, ~/.aws/credentials, or IAM role).
"""

import os
import sys

from langchain_aws import ChatBedrock
from langchain.agents import create_agent

from tools import calculator, add_item_to_list, get_common_items

# ── Model ────────────────────────────────────────────────────────────────────

MODEL_ID = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"

# AWS profile: defaults to "nimat", override with AWS_PROFILE env var.
AWS_PROFILE = os.environ.get("AWS_PROFILE", "nimat")

llm = ChatBedrock(
    model_id=MODEL_ID,
    region_name="eu-north-1",  # change to your preferred region
    credentials_profile_name=AWS_PROFILE,
)

# ── Agent ────────────────────────────────────────────────────────────────────

tools = [calculator, add_item_to_list, get_common_items]

agent = create_agent(llm, tools)

# ── Run ───────────────────────────────────────────────────────────────────────

def run(query: str) -> str:
    result = agent.invoke({"messages": [("human", query)]})
    return result["messages"][-1].content


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is (123 * 456) + 789?"
    print(f"Query : {query}")
    print(f"Answer: {run(query)}")


if __name__ == "__main__":
    main()
