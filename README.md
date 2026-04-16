# LangChain Agent Experiment

This project is a small experiment in building **LangChain agents** with interchangeable LLM backends. It wires a shopping-list assistant to a few simple tools, so you can try the same agent flow against either **AWS Bedrock** or **Google Cloud / Gemini** from the CLI.

## What it does

- Creates a LangChain agent in `agent.py`
- Exposes a few tools from `tools.py`
- Lets you pick the model provider with `--provider bedrock` or `--provider gemini`
- Runs a prompt from the command line and prints the answer

## Install

```bash
pip install -r requirements.txt
```

## Test it from the CLI

The entrypoint is:

```bash
python agent.py --provider <bedrock|gemini> "<your prompt>"
```

If you omit the prompt, the script uses a default math question.

### Using AWS Bedrock

Set these environment variables before running the agent:

- `AWS_PROFILE`: AWS profile used for credentials
- `AWS_REGION`: AWS region for Bedrock
- `BEDROCK_MODEL_ID`: Bedrock model ID to use
- `MODEL_PROVIDER`: optional default provider if you do not pass `--provider`

Example:

```bash
export AWS_PROFILE=your-profile
export AWS_REGION=eu-north-1
export BEDROCK_MODEL_ID=eu.anthropic.claude-sonnet-4-5-20250929-v1:0

python agent.py --provider bedrock "Add bacon and spaghetti to my list"
```

### Using Google Cloud / Gemini

Set these environment variables before running the agent:

- `GOOGLE_APPLICATION_CREDENTIALS`: path to a GCP service account JSON file
- `GCP_PID`: Google Cloud project ID used by the current implementation
- `GCP_REGION`: optional region, defaults to `europe-west1`
- `GEMINI_MODEL`: optional Gemini model name, defaults to `gemini-2.5-flash`
- `MODEL_PROVIDER`: optional default provider if you do not pass `--provider`

Example:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GCP_PID=your-gcp-project-id
export GCP_REGION=europe-west1
export GEMINI_MODEL=gemini-2.5-flash

python agent.py --provider gemini "What is (123 * 456) + 789?"
```

## Notes

- The provider can be selected explicitly with `--provider`, or implicitly with `MODEL_PROVIDER`.
- The current GCP implementation reads `GCP_PID` for the project ID.
