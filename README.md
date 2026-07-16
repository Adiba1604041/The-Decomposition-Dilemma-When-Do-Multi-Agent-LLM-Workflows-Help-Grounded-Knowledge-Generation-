This repository contains the datasets, generation pipelines, evaluation code, and analysis notebook for the paper **“The Decomposition Dilemma: When Do Multi-Agent LLM Workflows Help Grounded Knowledge Generation?”**

The study compares two LLM-based knowledge-generation workflows:

- **Direct Generation Workflow:** one generation agent defines the expected milestones and produces the final output.
- **Decomposed Generation Workflow:** a planner decomposes the task, an agent architect assigns executor roles, executor agents complete subtasks, a synthesizer produces the final output, and a judge evaluates the result. Shared memory stores intermediate workflow artifacts.

- ## Tasks and Datasets
- ### Public-Event Narrative Synthesis

The task generates an event-centered narrative from public-event metadata and multiple cited news articles.

- Source: English Wikipedia annual United States chronology pages from 2020–2025 and their cited news articles
- Dataset size: 55 event instances
- Each instance contains at least two usable source articles
- Evaluation uses a manually validated constraint-based reference describing required factual and structural elements

### Project-Grounded Research Ideation

The task generates a plausible future research direction grounded in an existing funded project.

- Source: National Science Foundation award records
- Dataset size: 100 projects
- Research areas: artificial intelligence, machine learning, cybersecurity, data science, and human–computer interaction
- Evaluation uses manually validated constraint-based references that preserve project metadata and source anchors

### Scientific Related-Work Generation

The task generates a citation-aware related-work paragraph from a query-paper abstract and cited-reference abstracts.

- Source: Multi-XScience
- Dataset size: 100 instances
- Sampling: 25 instances from each cited-reference-count bin: 1–5, 6–10, 11–15, and 16–20
- Evaluation uses the cited abstracts for factual grounding and the provided gold related-work paragraph as an evaluation reference

## Models and Experimental Settings

The experiments use four model settings:

- Gemma3:4B
- Qwen3:14B
- DeepSeek-R1:32B
- GPT-5-mini

Within each run, the same underlying model is used for every agent role.

For the three open-weight models, generation is evaluated at:

```text
T = 0.0, 0.5, and 1.0
```
