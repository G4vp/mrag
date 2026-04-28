import time
import numpy as np

from distances import cosine_distance, make_mahalanobis_distance
from embeddings import make_tfidf_embedder
from manifolds import make_pca_projector
from traversals import semantic_decay_traversal

CORPUS = [
    "The latest iPhone battery uses lithium and advanced chemistry.",
    "Lithium is a key component of the global battery supply chain.",
    "Cobalt mining is essential for the battery supply chain.",
    "Electric vehicles rely on extensive battery supply chains.",
    "Apples and bananas are yellow fruits.",
    "The weather in London is rainy today.",
]
QUERY = "iPhone battery materials"
TRAVERSAL_PARAMS = {"tau": 0.75, "epsilon": 1.8, "gamma": 1.5, "max_hops": 3}

def make_euclidean_distance(_inv_cov):
    return lambda z1, z2: float(np.linalg.norm(z1 - z2))

def make_manhattan_distance(_inv_cov):
    return lambda z1, z2: float(np.sum(np.abs(z1 - z2)))

def make_dot_product(_inv_cov):
    return lambda z1, z2: float(np.dot(z1, z2))

CONFIGS = [
    ("Cosine + Mahalanobis", cosine_distance, make_mahalanobis_distance, None),
    ("Cosine + Euclidean",   cosine_distance, make_euclidean_distance,   1.5),
    ("Cosine + Manhattan",   cosine_distance, make_manhattan_distance,   2.2),
    ("Cosine + MakeDotProduct",   cosine_distance, make_dot_product,   1.5),
]

def run(name, dist_X_fn, make_dist_Z_fn, epsilon_override):
    t0 = time.monotonic_ns()

    embedder = make_tfidf_embedder(CORPUS + [QUERY])
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
    print(f"  {name:<30}  {elapsed_ms:>8.3f} ms")
    print("  Hops:")
    for t in sorted(hops.keys()):
        print(f"    Hop {t}:")
        for d in hops[t]:
            print(f"     [D{d}] {CORPUS[d]}")
    print("-" * 42) 

def main():
    print(f"\nQuery: {QUERY!r}\n")
    print(f"  {'Config':<30}  {'Time':>8}")
    print("  " + "-" * 42)
    for cfg in CONFIGS:
        run(*cfg)
    print()


if __name__ == "__main__":
    main()
