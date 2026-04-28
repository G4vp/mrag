from embeddings import make_tfidf_embedder
from manifolds import make_pca_projector
from distances import cosine_distance, make_mahalanobis_distance
from traversals import semantic_decay_traversal
from visualizations import plot_manifolds
import os
from dotenv import load_dotenv
import time

from google import genai


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


def test_main(corpus, query, iteration): 
    # Measure Time for Execution
    start_time = time.monotonic_ns()

    # 2. Dependency Injection: Embeddings & Projections
    embedder = make_tfidf_embedder(corpus + [query])
    X = embedder(corpus)
    q = embedder([query])[0]

    # Project to d=2 bottleneck
    projector = make_pca_projector(d=2)
    Z, inv_cov = projector(X)

    # 3. Distance Metrics passed as pure closures
    dist_X = cosine_distance
    dist_Z = make_mahalanobis_distance(inv_cov)

    # 4. Traversal
    hops = semantic_decay_traversal(
        q=q,
        X=X,
        Z=Z,
        dist_X_fn=dist_X,
        dist_Z_fn=dist_Z,
        tau=0.75,  # Semantic similarity threshold limits starting point to D0 only
        epsilon=1.8,  # Structural Mahalanobis jump limit
        gamma=1.5,  # Decay factor (drift multiplier > 1 allows thematic walking)
        max_hops=3,
    )

    print("=== Multi-hop Traversal Result ===")
    for t in sorted(hops.keys()):
        print(f"Hop {t}:")
        for d in hops[t]:
            print(f"  [D{d}] {corpus[d]}")

    end_time = time.monotonic_ns()
    print(f"Execution time for embedding and traversal: {end_time - start_time:.2f} nano seconds")

    print("\n=== Section 6 LLM Synthesis Prompt ===")
    print(build_prompt(query, corpus, hops))

    print("\n=== Section 7 LLM Response ===")
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model = "gemini-3-flash-preview",
        contents= build_prompt(query, corpus, hops),
    )
    print(response.text)


    # 5. Visualization Map
    plot_manifolds(X, Z, hops, corpus, iteration)
    
 
def tests():
    corpus = [
    # --- Chain A: The Battery Composition Path ---
    "The latest iPhone battery utilizes a high-density cathode made of Lithium Cobalt Oxide (LiCoO2).",
    "Lithium is primarily extracted from brine pools in the 'Lithium Triangle' of South America.",
    "Cobalt used in smartphone cathodes is a critical mineral often sourced from the Katanga Province.",
    "Katanga Province is a major copper and cobalt mining hub located in the Democratic Republic of the Congo.",
    "High-density cathodes require chemical stabilizers to prevent thermal runaway during rapid charging.",

    # --- Chain B: The Manufacturing/Hardware Path ---
    "iPhone assembly relies heavily on precision manufacturing facilities in Zhengzhou, China.",
    "Precision manufacturing of mobile devices requires high-purity Silicon for SoC (System on Chip) production.",
    "Modern SoCs, like the A-series chips, use a 3nm lithography process provided by TSMC.",
    "TSMC is headquartered in Hsinchu Science Park, Taiwan.",

    # --- Noise / Distractors (Related but Irrelevant) ---
    "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
    "Bananas contain potassium, which is chemically similar to lithium as an alkali metal.",
    "Most electric vehicles use Lithium Iron Phosphate (LFP) batteries rather than Cobalt-based ones.",
    "The weather in Cupertino, California, is currently sunny with a light breeze.",
    "Neovim is a hyperextensible Vim-based text editor often configured using Lua.",
    "Minecraft's 'Redstone' mechanics are often used to simulate basic logic gates and transistors.",
    "Arch Linux users often prefer the 'rolling release' model for the latest kernel updates.",
    "The periodic table organizes elements by atomic number; Lithium is atomic number 3.",
    ]
    query = "Does the iPhone battery use the same chemistry as most Electric Vehicles?"

    test_main(corpus, query, 1)

if __name__ == "__main__":
    tests()
