# Large Language Models (LLMs)

Large language models are deep neural networks, almost always built on the Transformer architecture, trained on massive amounts of text to predict the next token given prior context. After pre-training, they are typically fine-tuned with supervised learning and reinforcement learning from human feedback (RLHF) or constitutional methods to produce instruction-following assistants.

## Transformer architecture
The Transformer was introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. at Google. It replaced recurrence with self-attention, allowing every token in a sequence to directly attend to every other token. Key components: multi-head self-attention, positional encodings, feed-forward sublayers, residual connections, and layer normalization.

## Scaling laws
Empirical scaling laws (Kaplan et al. 2020, Chinchilla 2022) showed that LLM performance follows predictable power-law improvements with more parameters, more tokens, and more compute. Chinchilla showed that earlier large models were under-trained; for a given compute budget, training a smaller model on more tokens gives better loss.

## Training stages
1. **Pretraining** — autoregressive next-token prediction on broad web/code corpora.
2. **Supervised fine-tuning (SFT)** — fine-tune on human-written instruction/response pairs.
3. **Preference optimization** — RLHF, DPO, or constitutional AI to align with helpfulness, honesty, and harmlessness.

## Inference techniques
- **Greedy and temperature sampling** trade off determinism vs. diversity.
- **Top-p (nucleus) sampling** truncates the distribution to the smallest set summing to probability p.
- **Speculative decoding** uses a small draft model to propose tokens that a larger model verifies in parallel.
- **KV-cache** stores per-token attention key/value tensors so each new token costs only one forward pass through the new token.

## Retrieval-augmented generation (RAG)
RAG augments an LLM at inference time by retrieving relevant document chunks from an external corpus and inserting them into the prompt. This lets the model answer questions over private or up-to-date data without fine-tuning. Typical stack: dense embeddings + vector store (FAISS, pgvector), often combined with BM25 sparse retrieval for hybrid search, then a reranker, then the LLM with cited chunks in context.

## Agentic RAG
Agentic RAG adds planning, self-reflection, multi-step retrieval, and tool use on top of standard RAG. The system may rewrite the query, retrieve again after critiquing its draft answer, fetch external sources like Wikipedia for general-knowledge gaps, and check confidence before returning a final response.
