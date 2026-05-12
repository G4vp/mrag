import time
import numpy as np

from distances import cosine_distance, make_mahalanobis_distance
from embeddings import make_mpnet_embedder, make_tfidf_embedder
from manifolds import make_pca_projector
from traversals import semantic_decay_traversal
from google.genai.local_tokenizer import LocalTokenizer


#CORPUS = [
#    "The latest iPhone battery uses lithium and advanced chemistry.",
#    "Lithium is a key component of the global battery supply chain.",
#    "Cobalt mining is essential for the battery supply chain.",
#    "Electric vehicles rely on extensive battery supply chains.",
#    "Apples and bananas are yellow fruits.",
#    "The weather in London is rainy today.",
#]
QUERY = "iPhone battery materials"

CORPUS = [
    # --- PATH 1: The Device to the Energy Component ---
    "The flagship iPhone models utilize a custom-layered power cell to maximize internal space.",
    "Modern energy cells in mobile devices rely on high-capacity chemical electrolytes.",
    "Lithium-ion polymer technology is the standard for thin-profile smartphone power units.",
    "Smartphone energy storage efficiency is limited by the thermal stability of the anode.",

    # --- PATH 2: The Component to the Specific Mineral (The First Hop) ---
    "Lithium-ion anodes are frequently constructed using specialized carbon allotropes.",
    "Synthetic graphite is the preferred carbon allotrope for high-performance battery anodes.",
    "The crystalline structure of synthetic graphite allows for efficient lithium-ion intercalation.",
    "Anode quality is determined by the purity of the graphite used in the coating process.",

    # --- PATH 3: The Mineral to the Industrial Feedstock (The Second Hop) ---
    "Synthetic graphite is produced through the high-temperature graphitization of needle coke.",
    "Needle coke is a highly crystalline form of carbon used specifically for electrodes.",
    "The primary feedstock for high-grade needle coke is a substance known as decant oil.",
    "Decant oil, also called slurry oil, is the heavy fraction resulting from fluid catalytic cracking.",

    # --- PATH 4: The Feedstock to the Raw Byproduct (The Final Hop) ---
    "Fluid catalytic cracking is a central process in the conversion of crude oil into gasoline.",
    "Decant oil is an industrial byproduct generated during the petroleum refining process.",
    "The heavy residues of oil refineries are often repurposed for carbon-intensive manufacturing.",
    "Refining crude oil produces several secondary streams, including bitumen and decant oil.",

    # --- NOISE CATEGORY A: Related Brands/Tech (Semantic Distractors) ---
    "The Apple Watch uses a similar magnetic charging mechanism to the MagSafe connector.",
    "MacBook Pro enclosures are milled from a single block of recycled aluminum.",
    "The Apple Store Fifth Avenue features a prominent glass cube design.",
    "iOS power management software optimizes charging cycles to prevent battery swelling.",
    "Energy storage for electric vehicles often involves nickel-manganese-cobalt chemistries.",
    "Next-generation solid-state batteries may replace current liquid electrolytes.",
    "Graphene is a single layer of carbon atoms with extraordinary electrical conductivity.",
    "Wireless charging pads utilize induction coils to transfer energy without cables.",
    "The production of semiconductors requires ultra-pure silicon wafers and cleanroom environments.",
    "USB-C standards dictate the power delivery wattage for modern handheld electronics.",

    # --- NOISE CATEGORY B: Industrial & Chemical Red Herrings ---
    "Pencils use a mixture of clay and natural graphite, which differs from synthetic variants.",
    "Refining cane sugar involves the use of bone char or granular activated carbon.",
    "The steel industry uses electric arc furnaces to melt scrap metal into new beams.",
    "Petroleum jelly is a common byproduct used in the pharmaceutical and cosmetic industries.",
    "Lubricating oils are refined to reduce friction in internal combustion engines.",
    "The process of vulcanizing rubber requires sulfur and high pressure.",
    "Coal tar is a byproduct of coke production used in the paving of roads.",
    "Natural gas processing yields helium as a valuable secondary noble gas.",
    "Industrial filtration systems often use sand or anthracite to remove particulates.",
    "Coking coal is essential for the reduction of iron ore in blast furnaces.",

    # --- NOISE CATEGORY C: Abstract & Unrelated (Hard Noise) ---
    "The migration patterns of Arctic terns cover thousands of miles annually.",
    "Quantum entanglement suggests that particles can remain connected across vast distances.",
    "The Renaissance period saw a massive shift in European artistic techniques.",
    "Photosynthesis converts solar energy into chemical energy within plant chloroplasts.",
    "The Great Barrier Reef is the world's largest coral reef system.",
    "Deep-sea hydrothermal vents support ecosystems that do not rely on sunlight.",
    "The standard model of particle physics describes the fundamental forces of nature.",
    "Sourdough starters rely on a symbiotic culture of yeast and bacteria.",
    "Tectonic plate movements are responsible for the majority of volcanic activity.",
    "The Fibonacci sequence is frequently observed in the spiral patterns of seashells.",
    "Game theory is used to model strategic interactions between rational agents.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "Ancient Mesopotamians developed one of the earliest known writing systems, Cuneiform.",
    "Acoustic levitation uses sound waves to suspend physical matter in mid-air.",
    "The James Webb Space Telescope orbits the sun at the second Lagrange point.",
    "Organic chemistry focuses on the study of carbon-based compounds and structures.",
    "The Treaty of Westphalia in 1648 helped establish the modern concept of nation-states.",
    "Enzymes act as biological catalysts to speed up metabolic reactions in the body.",
    "Beethoven composed his Ninth Symphony while he was almost completely deaf.",
    "The Sahara Desert is the largest hot desert in the world, spanning North Africa.",
    "Blockchain technology uses a decentralized ledger to record transactions securely.",
    "The concept of 'Ikigai' in Japanese culture refers to one's reason for being.",
    "Marsupials, such as kangaroos, carry their young in a specialized pouch.",
    "The Hubble Constant is a unit used to describe the expansion rate of the universe.",
    "Pythagoras is best known for his theorem regarding right-angled triangles.",
    "The Turing Test measures a machine's ability to exhibit intelligent behavior.",
    "Volcanic obsidian was used by ancient civilizations to create sharp cutting tools.",
    "The Rosetta Stone provided the key to deciphering Egyptian hieroglyphs.",
    "Dark matter is hypothesized to account for approximately 85% of the matter in the universe.",
    "The Magna Carta, signed in 1215, limited the power of the English monarchy.",
    "Bioluminescence in fireflies is caused by a chemical reaction involving luciferin.",
    "The Roman Colosseum could hold an estimated 50,000 to 80,000 spectators.",
    "Glaciers store about 69% of the world's fresh water in the form of ice.",
    "The printing press was invented by Johannes Gutenberg in the mid-15th century.",
    "Standardized time zones were established to facilitate railway scheduling in the 1800s.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
    "A surrealist painting by Salvador Dalí often features melting clocks.",
    "The Periodic Table organizes elements by their atomic number and properties.",
    "Insects have a three-part body consisting of a head, thorax, and abdomen.",
    "The Great Wall of China was built over several centuries for border protection.",
    "Vinyl records produce sound via a stylus following grooves in the plastic.",
    "The Doppler Effect explains why the pitch of a siren changes as it passes by.",
    "Monarch butterflies migrate from North America to central Mexico every winter.",
    "The Industrial Revolution began in Great Britain during the late 18th century.",
    "Caffeine works by blocking adenosine receptors in the human brain.",
    "The Amazon Rainforest produces a significant portion of the Earth's oxygen.",
    "Space-time curvature is a core concept of Einstein's General Relativity.",
    "The binary system uses only two digits, 0 and 1, to represent data.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "Atmospheric pressure decreases as altitude increases above the surface.",
    "The heart is a muscular organ that pumps blood throughout the circulatory system.",
    "Ocean tides are primarily caused by the gravitational pull of the moon.",
    "The Eiffel Tower was originally built as a temporary entrance for the 1889 World's Fair.",
    "A light-year is the distance that light travels in a single Earth year.",
    "The Silk Road was an ancient network of trade routes connecting East and West.",
    "Vaccines stimulate the immune system to recognize and fight specific pathogens.",
    "The Sahara's dust can be carried by wind across the Atlantic to the Amazon.",
    "Photosensitive cells in the retina are known as rods and cones.",
    "The Wright brothers achieved the first powered airplane flight in 1903.",
    "Gold is a highly malleable and ductile metal used in jewelry and electronics.",
    "The ozone layer protects the Earth from harmful ultraviolet radiation.",
    "Newton's Third Law states that for every action, there is an equal and opposite reaction.",
    "The internet originated from a research project called ARPANET in the 1960s.",
    "Photosynthesis is the process by which green plants make their own food."
]

# QUERY = "What are the raw industrial byproducts used to create iPhone energy storage?"

TRAVERSAL_PARAMS = {"tau": 0.75, "epsilon": 1.8, "gamma": 1.5, "max_hops": 3}

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
