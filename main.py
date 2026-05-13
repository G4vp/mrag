from embeddings import make_mpnet_embedder
from manifolds import make_pca_projector
from distances import cosine_distance, make_mahalanobis_distance
from traversals import semantic_decay_traversal
from visualizations import plot_manifolds
import re
import string
from collections import Counter
from dotenv import load_dotenv
from google import genai
from datasets import load_dataset

dataset = load_dataset("hotpot_qa", "distractor", split="validation[:50]")

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def score_response(llm_response: str, correct_answer: str) -> dict:
    pred = normalize_text(llm_response)
    gold = normalize_text(correct_answer)

    exact_match = int(pred == gold or gold in pred)

    pred_tokens = Counter(pred.split())
    gold_tokens = Counter(gold.split())
    common = sum((pred_tokens & gold_tokens).values())

    if common == 0:
        f1 = 0.0
    else:
        precision = common / sum(pred_tokens.values())
        recall = common / sum(gold_tokens.values())
        f1 = 2 * precision * recall / (precision + recall)

    return {"exact_match": exact_match, "f1": f1}

def build_prompt(query: str, corpus: list, hops: dict) -> str:
    primary = [corpus[i] for i in hops.get(0, frozenset())]
    structural = [corpus[i] for t, docs in hops.items() if t > 0 for i in docs]

    return f"""<primary_semantic_context>
{chr(10).join(f"- {doc}" for doc in primary)}
</primary_semantic_context>

<structurally_linked_context>
{chr(10).join(f"- {doc}" for doc in structural)}
</structurally_linked_context>

System: Synthesize the primary context to answer the user's query: "{query}".
Use the structurally linked context to draw broader multi-hop connections if the primary context is missing direct evidence."""


def main():
    load_dotenv()
    client = genai.Client()

    scores = []

    for i, item in enumerate(dataset):
        corpus = []
        for sentence_list in item["context"]["sentences"]:
            corpus.append(" ".join(sentence_list))

        query = item["question"]
        correct_answer = item["answer"]

        embedder = make_mpnet_embedder()
        X = embedder(corpus)
        q = embedder([query])[0]

        projector = make_pca_projector(d=2)
        Z, inv_cov = projector(X)

        dist_X = cosine_distance
        dist_Z = make_mahalanobis_distance(inv_cov)

        hops = semantic_decay_traversal(
            q=q,
            X=X,
            Z=Z,
            dist_X_fn=dist_X,
            dist_Z_fn=dist_Z,
            tau=0.55,
            epsilon=1.5,
            gamma=1.5,
            max_hops=3,
        )

        print(f"\n=== Item {i+1} ===")
        print(f"Question: {query}")

        print("\n--- Multi-hop Traversal ---")
        for t in sorted(hops.keys()):
            print(f"Hop {t}:")
            for d in hops[t]:
                print(f"  [D{d}] {corpus[d]}")

        plot_manifolds(X, Z, hops, corpus)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=build_prompt(query, corpus, hops),
        )

        llm_response = response.text
        result = score_response(llm_response, correct_answer)
        scores.append(result)

        print(f"\nLLM Response: {llm_response}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Exact Match: {result['exact_match']}  |  F1: {result['f1']:.3f}")

    print("\n" + "=" * 50)
    print("=== BENCHMARK RESULTS ===")
    print("=" * 50)
    n = len(scores)
    em_rate = sum(s["exact_match"] for s in scores) / n * 100
    avg_f1 = sum(s["f1"] for s in scores) / n * 100
    print(f"Items evaluated : {n}")
    print(f"Exact Match     : {em_rate:.1f}%")
    print(f"Average F1      : {avg_f1:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
