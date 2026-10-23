# TrialAtlas

**TrialAtlas: Multi-Agent Research Organization for Clinical Trial Design and Optimization**

TrialAtlas is a hierarchical multi-agent system that assesses clinical trial designs, proposes
protocol-level optimizations, and predicts regulatory success, grounding its reasoning in retrieved
evidence and regulatory precedent. This repository contains the TrialAtlas implementation and
**TrialAtlasBench**, a benchmark built from FDA Complete Response Letters (CRLs).

[Paper](https://arxiv.org/pdf/2609.21859) · [Dataset](https://huggingface.co/datasets/linjc16/TrialAtlasBench)

## Overview

Rather than acting as a single general-purpose agent, TrialAtlas is organized as a research
organization with explicit roles:

- **Director** — an orchestration controller that plans the analysis, assigns subtasks to the
  relevant departments, and synthesizes their findings into a final assessment.
- **Department Heads** — coordinate three lines of analysis, each carried out by their Working Agents:
  - **Competing Trial Analysis** — searches and compares related studies via the ClinicalTrials.gov API.
  - **Literature Evidence Synthesis** — retrieves and interprets biomedical evidence from PubMed and PubTator.
  - **Drug Benchmarking** — examines regulatory precedent via openFDA (Complete Response Letters, Drugs@FDA, product labeling).
- **Working Agents** — tool-using reasoning agents that iteratively call their domain tools and a
  sandboxed code-execution environment, returning structured, inspectable outputs.
- **Memory base** — a structured, human-readable record of regulatory experience (recurring
  deficiency patterns, regulatory red flags, and reusable reasoning strategies) that the organization
  draws on at inference.

The system is built on [LangGraph](https://github.com/langchain-ai/langgraph) with an LLM backbone
(GPT-5 by default; OpenAI, Azure OpenAI, Anthropic, Google, and AWS Bedrock backends are supported).

## Repository structure

```
trialatlas/
├── biodsa/
│   ├── agents/
│   │   ├── base_agent.py            # shared LLM/agent base
│   │   └── trialatlas/              # Director orchestrator + subagents
│   │       ├── agent.py
│   │       ├── prompt.py
│   │       └── subagents/           # competing-trial, literature, drug-benchmarking, ...
│   ├── tools/                       # clinical_trials, pubmed, openfda, code_exec_tool
│   ├── memory/                      # long-/short-term memory
│   ├── sandbox/                     # sandboxed code execution
│   └── utils/
├── benchmarks/
│   └── TrialOutcomePrediction/      # TrialAtlasBench task files (.jsonl)
├── scripts/
│   └── run_trialatlas_agent.py      # single-trial demo
├── Pipfile                          # dependencies
└── .env.example                     # API keys
```

## Installation

Requires **Python 3.11+** and **Docker** (used for the code-execution sandbox).

```bash
git clone https://github.com/linjc16/trialatlas.git
cd trialatlas

# with pipenv
pip install pipenv
pipenv install
pipenv shell

# or with pip
pip install -r <(pipenv requirements)   # or install the packages listed in Pipfile
```

## Configuration

Copy the example env file and fill in the credentials for the backend(s) you use:

```bash
cp .env.example .env
```

TrialAtlas defaults to Azure OpenAI:

```
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
```

Other supported backends (OpenAI, Anthropic, Google Vertex AI, AWS Bedrock) can be configured with
their respective keys in `.env`.

## Quickstart

Run TrialAtlas on a single trial design:

```bash
python scripts/run_trialatlas_agent.py
```

Or use it programmatically:

```python
from biodsa.agents import TrialAtlasAgent

agent = TrialAtlasAgent(
    model_name="gpt-5",
    api_type="azure",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    small_model_name="gpt-5",
    small_model_api_type="azure",
    small_model_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    small_model_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
agent.register_workspace()

results = agent.go("Assess the design of the following trial:\n\n<structured trial description>")
```

## TrialAtlasBench

The benchmark comprises three tasks over 150 unique clinical trials, provided under
`benchmarks/TrialOutcomePrediction/`:

| File | Task | Format | Size |
|---|---|---|---|
| `task1_deficiency_detection.jsonl` | Trial design deficiency detection | Multi-label multiple choice | 36 |
| `task2_optimization.jsonl` | Actionable trial design optimization | Multi-label multiple choice | 35 |
| `task3_ptrs_prediction.jsonl` | PTRS (regulatory approval) prediction | Binary classification | 104 |

Each record has fields `_id`, `nctid`, `prompt`, and `answer`. The dataset is also available on the
Hugging Face Hub: `https://huggingface.co/datasets/linjc16/trialatlasbench`.

**Evaluation.** For Tasks 1 and 2, score each question by the overlap between the selected option set
`S` and the correct set `G`: precision = |S ∩ G| / |S|, recall = |S ∩ G| / |G|, F1 = harmonic mean;
report the mean over questions. For Task 3, report balanced accuracy, F1, and Cohen's kappa.

## Citation

```bibtex
@article{trialatlas,
  title   = {TrialAtlas: Multi-Agent Research Organization for Clinical Trial Design and Optimization},
  author  = {Lin, Jiacheng and Wang, Zifeng and Chen, Zheng and Scott, Erick and Yang, Ziwei and Yu, Fanyang and Zhong, Sheng and Sun, Jimeng},
  journal = {arXiv preprint arXiv:2609.21859},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.21859}
}
```

## License

<!-- TODO: choose a license (e.g., MIT) and add a LICENSE file. -->
