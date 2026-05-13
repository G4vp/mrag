import time
import numpy as np

from distances import cosine_distance, make_mahalanobis_distance
from embeddings import make_mpnet_embedder, make_tfidf_embedder
from manifolds import make_pca_projector
from traversals import semantic_decay_traversal
from google.genai.local_tokenizer import LocalTokenizer
from datasets import load_dataset

ds = load_dataset("hotpotqa/hotpot_qa", "distractor")

CORPUS = [
    "The latest iPhone battery uses lithium and advanced chemistry.",
    "Lithium is a key component of the global battery supply chain.",
    "Cobalt mining is essential for the battery supply chain.",
    "Electric vehicles rely on extensive battery supply chains.",
    "Apples and bananas are yellow fruits.",
    "The weather in London is rainy today.",
]
QUERY = "iPhone battery materials"

TRAVERSAL_PARAMS = {"tau": 0.35, "epsilon": 1.8, "gamma": 1.5, "max_hops": 3}

def make_euclidean_distance(_inv_cov):
    return lambda z1, z2: float(np.linalg.norm(z1 - z2))

def make_manhattan_distance(_inv_cov):
    return lambda z1, z2: float(np.sum(np.abs(z1 - z2)))

def make_dot_product(_inv_cov):
    return lambda z1, z2: float(np.dot(z1, z2))

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
Answer the question as directly and briefly as possible.
Use the structurally linked context to draw broader multi-hop connections if the primary context is missing direct evidence."""

CONFIGS = [
    ("Cosine + Mahalanobis", cosine_distance, make_mahalanobis_distance, None),
    ("Cosine + Euclidean",   cosine_distance, make_euclidean_distance,   1.5),
    ("Cosine + Manhattan",   cosine_distance, make_manhattan_distance,   2.2),
    ("Cosine + MakeDotProduct",   cosine_distance, make_dot_product,   1.5),
]

def run(name, dist_X_fn, make_dist_Z_fn, epsilon_override):
    t0 = time.monotonic_ns()

    embedder = make_mpnet_embedder() 
    X = embedder(CORPUS)
    q = embedder([QUERY])[0]
    projector = make_pca_projector(d=2)
    Z, inv_cov = projector(X)

    params = dict(TRAVERSAL_PARAMS)
    if epsilon_override is not None:
        params["epsilon"] = epsilon_override

    hops = semantic_decay_traversal(
        q=q,
        X=X,
        Z=Z,
        dist_X_fn=dist_X_fn,
        dist_Z_fn=make_dist_Z_fn(inv_cov),
        **params,
    )

    elapsed_ms = (time.monotonic_ns() - t0) / 1e6
    print(f"  {'Config':<30}  {'Time':>8}")
    print(f"  {name:<30}  {elapsed_ms:>8.3f} ms")
    print("  Hops:")
    for t in sorted(hops.keys()):
        print(f"    Hop {t}:")
        for d in hops[t]:
            print(f"     [D{d}] {CORPUS[d]}")
    print("\nPrompt:")
    prompt = build_prompt(QUERY, CORPUS, hops)
    print(prompt)
    tokenizer = LocalTokenizer(model_name="gemini-2.0-flash")
    result = tokenizer.count_tokens(prompt)
    print(f"\nTokens: {result.total_tokens}")
    print("-" * 42) 

def main():
    print(f"\nQuery: {QUERY!r}\n")
    print("  " + "-" * 42)
    for cfg in CONFIGS:
        run(*cfg)
    print()


if __name__ == "__main__":
    main()
