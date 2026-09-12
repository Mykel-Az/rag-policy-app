"""
Evaluation harness for the RAG policy assistant.

Measures (per project rubric):
  - Groundedness (LLM-as-judge: is the answer fully supported by retrieved context?)
  - Citation accuracy (does the returned citation match the expected source doc?)
  - Latency (p50/p95 across all questions, warm-state)

Run with:
    python -m eval.run_eval
"""
import json, re
import statistics
from pathlib import Path

from langchain_groq import ChatGroq

from backend import rag_chain, config

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
EVAL_RESULTS_PATH = Path(__file__).parent / "eval_results.json"


def load_questions():
    return json.loads(EVAL_QUESTIONS_PATH.read_text())


def judge_groundedness(question: str, answer: str, chunks: list) -> bool:
    """
    Ask a separate, stronger model whether the answer is fully supported by
    the retrieved context. Using a different model than the one that
    generated the answer avoids self-preference bias in the judgment.
    """
    context = "\n\n".join(c.page_content for c in chunks)
    prompt = f"""You are a strict evaluator. Given the CONTEXT and ANSWER below,
determine whether every factual claim in the ANSWER is supported by the
CONTEXT. You may reason briefly, but you MUST end your response with exactly
one line in this exact format, and nothing after it:

Final verdict: YES
or
Final verdict: NO

CONTEXT:
{context}

ANSWER:
{answer}"""

    judge = ChatGroq(
        model=config.GROQ_JUDGE_MODEL,
        api_key=config.GROQ_API_KEY,
        max_tokens=500,
        temperature=0,
    )
    response = judge.invoke(prompt)
    content = response.content.strip()

    match = re.search(r"final verdict:\s*(yes|no)", content, re.IGNORECASE)
    if match is None:
        print(f"  [judge warning] no parseable verdict, raw: {content[:200]!r}")
        return False  # fail closed: treat unparseable judgments as ungrounded

    return match.group(1).lower() == "yes"


def citation_correct(expected_source, answer: str, citations: list) -> bool:
    """
    For answerable questions: does the expected source appear among the
    returned citations?
    For out-of-scope questions (expected_source is None): "correct" means the
    app returned the exact refusal message, regardless of what got retrieved.
    """
    if expected_source is None:
        return answer.strip() == config.OUT_OF_SCOPE_MESSAGE
    cited_sources = {c["source"] for c in citations}
    return expected_source in cited_sources



def run_eval():
    questions = load_questions()
    retriever = rag_chain.get_retriever()
    llm = rag_chain.get_llm()
    results = []

    for q in questions:
        print(f"[{q['id']}/{len(questions)}] {q['question']}")

        chunks = retriever.invoke(q["question"])
        result = rag_chain.answer_question(q["question"])

        is_refusal = result["answer"].strip() == config.OUT_OF_SCOPE_MESSAGE
        if is_refusal:
            grounded = True
        else:
            grounded = judge_groundedness(q["question"], result["answer"], chunks)

        cited_ok = citation_correct(
            q["expected_source"], result["answer"], result["citations"]
        )

        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": result["answer"],
            "expected_source": q["expected_source"],
            "cited_sources": [c["source"] for c in result["citations"]],
            "citation_correct": cited_ok,
            "grounded": grounded,
            "latency_ms": result["latency_ms"],
        })

    return results


def summarize(results):
    n = len(results)
    groundedness_pct = 100 * sum(r["grounded"] for r in results) / n
    citation_accuracy_pct = 100 * sum(r["citation_correct"] for r in results) / n

    latencies = sorted(r["latency_ms"] for r in results)
    p50 = statistics.median(latencies)
    p95_index = min(int(len(latencies) * 0.95), len(latencies) - 1)
    p95 = latencies[p95_index]

    return {
        "num_questions": n,
        "groundedness_pct": round(groundedness_pct, 1),
        "citation_accuracy_pct": round(citation_accuracy_pct, 1),
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
    }


if __name__ == "__main__":
    results = run_eval()
    summary = summarize(results)

    EVAL_RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2)
    )

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nFull results written to {EVAL_RESULTS_PATH}")