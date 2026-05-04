import time
import numpy as np

from distances import cosine_distance, make_mahalanobis_distance
from embeddings import make_mpnet_embedder
from manifolds import make_pca_projector
from traversals import semantic_decay_traversal

#CORPUS = [
#    "The latest iPhone battery uses lithium and advanced chemistry.",
#    "Lithium is a key component of the global battery supply chain.",
#    "Cobalt mining is essential for the battery supply chain.",
#    "Electric vehicles rely on extensive battery supply chains.",
#    "Apples and bananas are yellow fruits.",
#    "The weather in London is rainy today.",
#]
#QUERY = "iPhone battery materials"

CORPUS = [
    # Path 1: The Product to the Component
    "The iPhone Pro features a high-density power cell designed for longevity.",
    "Modern smartphone power cells are primarily composed of lithium-ion technology.",
    # Path 2: The Component to the Material (The "Hop")
    "Lithium-ion technology requires high-purity anode materials like synthetic graphite.",
    "Synthetic graphite production is heavily concentrated in specific industrial hubs.",
    # Path 3: The Material to the Source/Impact
    "The manufacturing of synthetic graphite relies on petroleum coke as a primary feedstock.",
    "Petroleum coke is a byproduct of the oil refining process.",
    # Noise: Related Keywords but Irrelevant Context
    "The Apple Store in London is famous for its glass staircase architecture.",
    "Graphite pencils were first mass-produced in the 19th century.",
    "Refining sugar is a multi-stage process involving filtration and crystallization.",
    "The battery life of a MacBook is significantly longer than that of a standard laptop.",
    # Pure Noise: Completely Unrelated
    "Deep-sea squids have evolved unique bioluminescent organs for hunting.",
    "Standardized testing scores have fluctuated globally over the last decade.",
    "The recipe for a perfect sourdough requires a 70% hydration level."
]
QUERY = "What are the raw industrial byproducts used to create iPhone energy storage?"

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
