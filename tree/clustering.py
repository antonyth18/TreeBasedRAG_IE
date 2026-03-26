import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from umap import UMAP

from tree.node import RaptorNode

logger = logging.getLogger(__name__)

# UMAP 768-dim → low-dim
def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> np.ndarray:
    n = embeddings.shape[0]
    n_components = min(n_components, n - 2)
    n_neighbors  = min(n_neighbors, n - 1)

    if n_components < 1:
        logger.warning(f"Too few nodes ({n}) for UMAP. Returning original embeddings.")
        return embeddings

    return UMAP(
        n_components=n_components,      # target dimensionality for GMM clustering
        n_neighbors=n_neighbors,        # larger n_neighbors = broader global topics, smaller n_neighbors = finer local topics
        metric="cosine",                # use cosine similarity
        random_state=None,              # Set to None to enable Numba parallel threading in UMAP (n_jobs=-1)
        low_memory=False,               # UMAP's default is True for large datasets, but it can cause convergence issues. Setting to False can improve stability at the cost of higher memory usage.
        n_jobs=-1                       # Enforce all cores
    ).fit_transform(embeddings)


# Find optimal k via BIC (Bayesian Information Criterion)
def select_cluster_count_bic(
    reduced: np.ndarray,
    max_clusters: int = 50,
    random_state: int = 42,
) -> int:
    n     = reduced.shape[0]
    max_k = min(max_clusters, n - 1)

    if max_k < 2:
        return 1

    import concurrent.futures

    def fit_and_score(k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            try:
                gmm = GaussianMixture(n_components=k, covariance_type="full",
                                      random_state=random_state, n_init=1)
                gmm.fit(reduced)
                return k, gmm.bic(reduced)
            except Exception as e:
                logger.debug(f"GMM fit failed at k={k}: {e}.")
                return k, np.inf

    best_bic, best_k = np.inf, 1

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fit_and_score, k): k for k in range(1, max_k + 1)}
        for future in concurrent.futures.as_completed(futures):
            k, bic = future.result()
            if bic < best_bic:
                best_bic, best_k = bic, k

    logger.debug(f"BIC selected k={best_k}  (BIC={best_bic:.2f})")
    return best_k


# Fit GMM, get soft probability labels
def fit_gmm(
    reduced: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[GaussianMixture, np.ndarray]:
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_state, n_init=1)
    gmm.fit(reduced)
    return gmm, gmm.predict_proba(reduced)   # soft_labels shape (N, k)

# Threshold probabilities, node can belong to >1 cluster
def assign_clusters(
    nodes: List[RaptorNode],
    soft_labels: np.ndarray,
    threshold: float = 0.1,
) -> Dict[int, List[RaptorNode]]:
    clusters: Dict[int, List[RaptorNode]] = {k: [] for k in range(soft_labels.shape[1])}

    for node_idx, node in enumerate(nodes):
        for cluster_id in range(soft_labels.shape[1]):
            if soft_labels[node_idx, cluster_id] > threshold:
                clusters[cluster_id].append(node)

    return {k: v for k, v in clusters.items() if v}   # drop empty clusters

# Orchestrates 1-4 with two-pass UMAP + overflow handling
def cluster_nodes(
    nodes: List[RaptorNode],
    max_cluster_tokens: int = 8000,
    umap_n_components: int = 10,
    assignment_threshold: float = 0.1,
    random_state: int = 42,
) -> Dict[int, List[RaptorNode]]:
    if len(nodes) < 2:
        return {0: nodes}

    embeddings = np.stack([n.embedding for n in nodes])

    # Pass 1: Global clustering (large n_neighbors → broad topics)
    global_reduced  = reduce_dimensions(embeddings, umap_n_components, n_neighbors=15, random_state=random_state)
    global_k        = select_cluster_count_bic(global_reduced, random_state=random_state)

    if global_k <= 1:
        return {0: nodes}

    _, global_soft  = fit_gmm(global_reduced, global_k, random_state)
    global_clusters = assign_clusters(nodes, global_soft, assignment_threshold)

    # Pass 2: Local clustering within each global cluster (small n_neighbors)
    intermediate: Dict[int, List[RaptorNode]] = {}
    counter = 0

    for g_nodes in global_clusters.values():
        if len(g_nodes) < 2:
            intermediate[counter] = g_nodes
            counter += 1
            continue

        local_reduced = reduce_dimensions(
            np.stack([n.embedding for n in g_nodes]),
            umap_n_components, n_neighbors=5, random_state=random_state,
        )
        local_k = select_cluster_count_bic(local_reduced, random_state=random_state)

        if local_k <= 1:
            intermediate[counter] = g_nodes
            counter += 1
        else:
            _, local_soft  = fit_gmm(local_reduced, local_k, random_state)
            local_clusters = assign_clusters(g_nodes, local_soft, assignment_threshold)
            for l_nodes in local_clusters.values():
                intermediate[counter] = l_nodes
                counter += 1

    # Overflow: recursively split clusters that exceed token limit
    final: Dict[int, List[RaptorNode]] = {}
    final_counter = 0

    for c_nodes in intermediate.values():
        if sum(n.token_count for n in c_nodes) > max_cluster_tokens and len(c_nodes) > 1:
            logger.warning(f"Cluster too large ({sum(n.token_count for n in c_nodes)} tokens). Recursively splitting...")
            sub = cluster_nodes(c_nodes, max_cluster_tokens, umap_n_components, assignment_threshold, random_state)
            for sub_nodes in sub.values():
                final[final_counter] = sub_nodes
                final_counter += 1
        else:
            final[final_counter] = c_nodes
            final_counter += 1

    logger.info(f"cluster_nodes: {len(nodes)} nodes → {len(final)} clusters")
    return final