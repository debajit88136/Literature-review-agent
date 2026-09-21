# Literature Review: retrieval augmented generation

_Generated on 2026-09-21 from 3 arXiv papers using gemini-3.5-flash-lite._

## Overview

This review analyses 3 papers (3 from full text, long papers trimmed to their start and end; 0 from the abstract only), grouped into 2 themes:

- **Evaluation and Privacy in RAG** (2 papers): These papers focus on assessing the performance and security of retrieval-augmented generation systems, covering automated reference-free evaluation metrics and stealthy membership inference attacks.
- **Advanced Retrieval for Code Generation** (1 paper): This paper explores specialized retrieval-augmented generation pipelines tailored for code generation by dynamically evolving queries and knowledge bases through multi-round interactions.

## Evaluation and Privacy in RAG

These papers focus on assessing the performance and security of retrieval-augmented generation systems, covering automated reference-free evaluation metrics and stealthy membership inference attacks.

### Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation

Ali Naseh et al. (2025). *Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation*. arXiv:2502.00306v2. https://arxiv.org/abs/2502.00306v2

- **Type:** new method
- **Approach:** The paper introduces the Interrogation Attack (IA), a black-box membership inference attack targeting documents in a RAG datastore. It crafts natural-text queries using few-shot prompting and an auxiliary LLM that are uniquely answerable by a target document, then probes the RAG system to verify membership based on answer correctness.
- **Key findings:**
  - IA achieves a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations.
  - The attack uses as few as 30 natural-text queries while remaining stealthy, with straightforward detectors identifying adversarial prompts from existing methods up to 76x more frequently.
  - The total financial cost to execute the attack is less than $0.02 per document inference using OpenAI APIs.
  - Failure cases are primarily linked to RAG generator limitations, such as paraphrasing failures or the ability of the underlying LLM to answer questions without retrieval context.
- **Datasets:** NFCorpus, SCIDOCS, TREC-COVID
- **Relation to the topic:** The paper investigates privacy risks in retrieval-augmented generation (RAG) systems by proposing a novel membership inference attack to determine if specific documents are present in a RAG datastore.
- **Stated limitations:**
  - In some settings, the true positive rate at low false positive rates leaves room for improvement.
  - Evaluating the attack against RAG variations involving different forms of input modification beyond query rewriting remains unexplored.

### Ragas: Automated Evaluation of Retrieval Augmented Generation

Shahul Es et al. (2023). *Ragas: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217v2. https://arxiv.org/abs/2309.15217v2

- **Type:** system or tool
- **Approach:** The paper introduces Ragas (Retrieval Augmented Generation Assessment), a framework for reference-free evaluation of RAG pipelines. It provides a suite of automated metrics—specifically faithfulness, answer relevance, and context relevance—computed by prompting an LLM to decompose answers into statements, verify them against context, and leverage embeddings without relying on ground truth human annotations.
- **Key findings:**
  - Ragas achieves high agreement with human annotators in pairwise comparisons, with accuracies of 0.95 for faithfulness, 0.78 for answer relevance, and 0.70 for context relevance.
  - Ragas outperforms baseline approaches such as GPT Score and GPT Ranking across all evaluated quality dimensions.
  - Human annotators agreed in around 95% of cases for faithfulness and context relevance, and around 90% for answer relevance on the WikiEval dataset.
- **Datasets:** WikiEval
- **Relation to the topic:** This paper directly contributes to retrieval-augmented generation (RAG) by proposing a novel, automated, reference-free evaluation framework to assess RAG pipelines.
- **Stated limitations:**
  - Agreement for answer relevance is lower because differences between candidate answers are often very subtle.
  - Context relevance is found to be the hardest quality dimension to evaluate.
  - ChatGPT often struggles with the task of selecting the crucial sentences from the context, especially for longer contexts.

## Advanced Retrieval for Code Generation

This paper explores specialized retrieval-augmented generation pipelines tailored for code generation by dynamically evolving queries and knowledge bases through multi-round interactions.

### EVOR: Evolving Retrieval for Code Generation

Hongjin Su et al. (2024). *EVOR: Evolving Retrieval for Code Generation*. arXiv:2402.12317v2. https://arxiv.org/abs/2402.12317v2

- **Type:** new method
- **Approach:** The paper introduces EVOR, a retrieval-augmented code generation (RACG) pipeline that dynamically and synchronously evolves both queries and a diverse knowledge base (incorporating web search, documentation, execution feedback, and code snippets). Through multi-round interactions among retrievers, LLMs, and executors, queries and knowledge bases are continuously updated based on execution feedback to extract the most pertinent information.
- **Key findings:**
  - EVOR achieves two to four times the execution accuracy compared to existing methods like Reflexion and DocPrompting.
  - EVOR's performance benefits significantly from the synchronous evolution of queries and documents combined with diverse knowledge sources.
  - EVOR can be easily integrated with existing code generation and agent-based approaches, such as SWE-agent, yielding further performance improvements.
  - EVOR demonstrates more effective token usage and higher pass rates across various token consumption levels ranging from 4k to 24k.
- **Datasets:** EVOR-BENCH, Scipy-M, Tensorflow-M, Ring, Pony, DS-1000, LeetCode, SWE-bench-Lite
- **Relation to the topic:** This paper is directly about retrieval-augmented generation (RAG), specifically proposing an advanced retrieval-augmented code generation (RACG) pipeline that employs synchronous evolution of queries and diverse knowledge bases.
- **Stated limitations:**
  - The iterative process of multiple rounds of interactions among retrievers, LLMs, and executors leads to longer latency.
  - The iterative pipeline increases energy consumption, posing concerns for real-time applications and energy-constrained environments.

## Recurring open problems

No problem shared by two or more papers was identified in this set.

## References

- Shahul Es et al. (2023). *Ragas: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217v2. https://arxiv.org/abs/2309.15217v2
- Ali Naseh et al. (2025). *Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation*. arXiv:2502.00306v2. https://arxiv.org/abs/2502.00306v2
- Hongjin Su et al. (2024). *EVOR: Evolving Retrieval for Code Generation*. arXiv:2402.12317v2. https://arxiv.org/abs/2402.12317v2

---
_This review was generated automatically by an LLM pipeline. Titles, authors, years and links come directly from arXiv, but the summaries are model-written and may contain errors. Verify important claims against the original papers._
