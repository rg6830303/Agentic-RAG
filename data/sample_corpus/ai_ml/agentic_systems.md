# Agentic AI Systems

An **AI agent** is a system that uses an LLM (or other model) as a controller to decide on actions, invoke tools, and pursue a goal across multiple steps. Unlike a simple chat completion, an agent maintains state, plans, observes results, and revises its approach.

## Core building blocks
- **Planner / reasoner**: an LLM, often with chain-of-thought prompting, that decides what to do next.
- **Tools**: well-typed functions the agent can call (web search, code execution, database queries, file I/O, calculators, APIs).
- **Memory**: short-term scratchpad context, plus optional long-term store (vector DB, key-value store, structured DB).
- **Feedback loop**: the agent observes tool results, reflects, and either continues, replans, or returns a final answer.

## Agentic RAG specifically
Agentic RAG layers planning and self-correction on top of standard retrieval-augmented generation:
1. **Query understanding**: rewrite the user question into a precise retrieval query.
2. **Retrieval**: pull candidates from a vector index, BM25, or hybrid; possibly multiple passes.
3. **Reflection**: critique the draft answer against the retrieved evidence; decide if more retrieval is needed.
4. **External augmentation**: fall back to web search (Wikipedia, DuckDuckGo, Google) when local corpus is insufficient.
5. **Guardrails**: confidence scoring, citation coverage checks, retrieval score floors, hallucination detection.
6. **Synthesis**: final answer with inline citations and structured metadata.

## Popular agent frameworks
- **LangChain / LangGraph**: graph-based agent orchestration with first-class state.
- **LlamaIndex**: retrieval-focused; recently added agent and workflow primitives.
- **Anthropic's Claude Agent SDK**: native tool use, computer use, and managed agents.
- **OpenAI Assistants / Agents SDK**: function calling, code interpreter, file search built in.
- **CrewAI, AutoGen**: multi-agent collaboration patterns.

## Common evaluation metrics
- **Token F1 / exact match** for QA tasks.
- **Context recall**: did retrieval surface the right evidence?
- **Citation hit rate**: did the answer actually cite the supporting source?
- **Faithfulness**: does the answer stay grounded in the cited context?
- **Confidence / self-rated calibration**.
- **RAGAS**: an open-source library that operationalizes the above for RAG pipelines.

## Challenges
- **Hallucination** still occurs when retrieval misses or the model over-generalizes from weak evidence.
- **Tool reliability**: agents are brittle to flaky APIs, rate limits, and schema drift.
- **Long-horizon planning**: many tasks degrade as the agent's trajectory grows; compaction and persistent memory help.
- **Safety**: agents that execute code or call APIs need sandboxing and policy guardrails.
