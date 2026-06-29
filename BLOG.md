# When Your RAG Agent Goes Wrong, Where Did It Actually Break?

*A plain-language summary of the AgenticRAG failure-propagation research project.*

## The problem in one sentence

When an AI agent retrieves documents, calls tools, and generates an answer across multiple steps ("hops"), and the final answer is wrong, **today there's no good way to tell which step actually caused the failure.**

## Why this matters

Retrieval-Augmented Generation (RAG) systems are increasingly "agentic": instead of doing one retrieval pass and answering, they loop -- retrieve, check, maybe retrieve again, call a tool, then answer. This is great for accuracy on hard multi-step questions, but it also means there are more places for things to silently break:

- The retriever might fetch nothing useful.
- The agent might skip a tool call it needed.
- The model might hallucinate an answer that sounds confident but isn't grounded in anything it retrieved.

If all you measure is "was the final answer right or wrong," you can't tell these apart. A team trying to improve their RAG system is left guessing: do we need a better retriever? Better tool-use prompting? Better answer grounding? Without knowing *where* failures originate, you can spend months fixing the wrong thing.

## What this project does

This project builds a **diagnostic framework** for agentic RAG pipelines, with two key pieces:

1. **Controlled failure injection.** Instead of waiting for failures to happen naturally (which makes it hard to know the true cause), we deliberately break a pipeline in a known way -- wipe out retrieved documents at a chosen hop, swap in irrelevant documents, skip tool calls, or inject a fabricated ("hallucinated") answer. Because we know exactly what we broke and where, we can check whether our diagnostic tool correctly figures it out.

2. **A diagnostic benchmark.** Given a pipeline trace (the full record of what was retrieved, what tools were called, and what was answered at each hop), `DiagnosticBenchmark` tries to identify the *root cause* stage of any failure -- not just "it failed" but "it failed because retrieval came back empty at hop 1," for example.

On top of this, the framework computes research-relevant metrics that end-to-end accuracy can't give you:

- **Root-cause accuracy** -- how often the diagnostic correctly names the true failing stage.
- **Failure amplification** -- do early-hop failures cause more downstream damage than late-hop failures, or does the pipeline absorb them?
- **Recovery rate** -- how often does the agent self-correct a mid-pipeline failure by the time it reaches the final hop (e.g., by reformulating its query)?

## What we expect to find (and haven't fully confirmed yet)

The design hypothesizes that early-stage retrieval failures cascade into answer failures far more often than they get caught early, and that agentic pipelines have a real -- but limited -- ability to self-correct via iterative re-querying (somewhere in the 20-35% range, by our working hypothesis). Answer-stage failures, like hallucination, are expected to be effectively unrecoverable since there's no later step to catch them.

**Important caveat:** as of this writing, these are *hypotheses from the design document*, not confirmed experimental results. The code to run these experiments (`scripts/run_baseline.py`, `scripts/run_ablation.py`, `scripts/plot_amplification.py`) exists and is implemented, but the experiments have not yet been executed end-to-end and the results have not been collected into the `results/` directory. See `PAPER_DRAFT.md` and `results/README.md` for exactly what's measured vs. projected.

## Why we think this is a useful contribution

Most RAG evaluation today still reports a single end-to-end accuracy number. This project's contribution is methodological: a way to *causally* attribute failures to pipeline stages via controlled injection, rather than just observing natural failures and guessing. If the diagnostic benchmark proves reliable (i.e., it scores well on root-cause accuracy in the planned experiments), it gives teams building agentic RAG systems a concrete answer to "which part of my pipeline should I harden first?" -- a retrieval fallback, a tool-call guard, or an answer-validation gate.

## What's next

The remaining work is mostly about running the experiments that are already coded up: baseline failure rates on HotpotQA/MuSiQue, the full injection-sensitivity and root-cause-accuracy sweeps, and the amplification/recovery-rate analysis across hops and retrievers. Once those numbers come in, this blog post and the paper draft will be updated with real, measured results.
