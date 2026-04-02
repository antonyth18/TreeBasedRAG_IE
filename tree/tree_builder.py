# REMINDER: Set OLLAMA_NUM_PARALLEL=2 as an environment variable before starting the Ollama server.
# Run: export OLLAMA_NUM_PARALLEL=2
# Remind to restart Ollama after setting it!

import logging
import os
import json
import asyncio
import httpx
from typing import List, Optional

import tiktoken

from tree.clustering import cluster_nodes
from tree.node import RaptorNode, RaptorTree
from tree.summarization import (
    LLMSummarizer,
    _SYSTEM_PROMPT,
    _USER_TEMPLATE,
    _VERIFICATION_SYSTEM_PROMPT,
    _VERIFICATION_USER_TEMPLATE
)
from tree.tree_serializer import save_tree

logger = logging.getLogger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

CHECKPOINT_PATH = "my_tree/checkpoints.json"


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


async def _summarize_cluster_async(client, semaphore, node_id, texts, summarizer_instance, checkpoints):
    # Check if this node_id is already in checkpoint
    if str(node_id) in checkpoints:
        logger.info(f"Skipping summarization for node {node_id} (found in checkpoint).")
        return node_id, checkpoints[str(node_id)], 0

    context = "\n\n".join(texts)
    model = summarizer_instance.model
    verify = summarizer_instance.verify_faithfulness
    max_verifications = summarizer_instance.max_verification_retries

    async with semaphore:
        last_summary = None
        total_tokens = 0
        
        # We loop max_verifications (plus the first attempt) if verify is enabled, otherwise just once
        attempts = max_verifications + 1 if verify else 1
        
        for verification_attempt in range(1, attempts + 1):
            summary = None
            
            # --- GENERATION ---
            for generate_attempt in range(1, summarizer_instance.max_retries + 1):
                try:
                    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                    response = await client.post(
                        f"{base_url}/api/generate",
                        json={
                            "model": model,
                            "system": _SYSTEM_PROMPT,
                            "prompt": _USER_TEMPLATE.format(context=context),
                            "stream": False
                        },
                        timeout=180.0
                    )
                    response.raise_for_status()
                    data = response.json()
                    summary = data.get("response", "").strip()
                    total_tokens += data.get("eval_count", 0)
                    if summary:
                        break
                except Exception as e:
                    logger.warning(f"Ollama API generation error for node {node_id} (attempt {generate_attempt}/{summarizer_instance.max_retries}): {e}")
                    await asyncio.sleep(2)
                    
            if not summary:
                logger.error(f"Failed to generate summary for node {node_id} after {summarizer_instance.max_retries} attempts.")
                fallback = " ".join(context.split())[:800] + "..."
                last_summary = fallback
                break # Cannot proceed to verify an empty/failed summary properly
                
            last_summary = summary
            
            # --- VERIFICATION ---
            if not verify:
                break
                
            passed = False
            failed_claim = ""
            try:
                base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                ver_response = await client.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model,
                        "system": _VERIFICATION_SYSTEM_PROMPT,
                        "prompt": _VERIFICATION_USER_TEMPLATE.format(source=context, summary=summary),
                        "stream": False
                    },
                    timeout=60.0
                )
                ver_response.raise_for_status()
                ver_data = ver_response.json()
                verdict = ver_data.get("response", "").strip()
                total_tokens += ver_data.get("eval_count", 0)
                
                if verdict.upper().startswith("PASS"):
                    passed = True
                else:
                    failed_claim = verdict[5:].strip() if verdict.upper().startswith("FAIL:") else verdict
            except Exception as e:
                # If verification itself fails, log and pass through rather than blocking tree construction
                logger.warning(f"Faithfulness check HTTP error for node {node_id}: {e}. Skipping check.")
                passed = True 
                
            if passed:
                if verification_attempt > 1:
                    logger.info(f"Node {node_id} summary passed faithfulness check on attempt {verification_attempt}.")
                break
            
            # Verification failed
            logger.warning(
                f"Faithfulness check failed for node {node_id} (attempt {verification_attempt}/{max_verifications}). "
                f"Hallucinated claim: '{failed_claim}'. Retrying summary..."
            )

        # Append to checkpoint file immediately
        checkpoints[str(node_id)] = last_summary
        
        # Write back checkpoint
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoints, f, indent=2)
            
        return node_id, last_summary, total_tokens


async def _process_layer_summaries(clusters, start_node_id, summarizer_instance, checkpoints):
    semaphore = asyncio.Semaphore(2)
    tasks = []
    
    # We need a stable mapping of node_id to c_nodes
    node_mapping = []
    current_id = start_node_id
    
    # One client for all concurrent tasks
    async with httpx.AsyncClient() as client:
        for c_nodes in clusters.values():
            texts = [n.text for n in c_nodes]
            node_mapping.append((current_id, c_nodes))
            tasks.append(_summarize_cluster_async(client, semaphore, current_id, texts, summarizer_instance, checkpoints))
            current_id += 1
            
        results = await asyncio.gather(*tasks)
        
    new_nodes = []
    total_layer_tokens = 0
    
    # gather preserves the order of the tasks list, so we can zip safely
    for (node_id, c_nodes), (res_id, summary_text, tokens) in zip(node_mapping, results):
        assert node_id == res_id
        
        total_layer_tokens += tokens
        
        parent = RaptorNode(
            index=node_id,
            text=summary_text,
            embedding=None,  # We chunk-encode later sequentially
            layer=c_nodes[0].layer + 1,
            children=[n.index for n in c_nodes],
            token_count=_count_tokens(summary_text),
        )
        new_nodes.append(parent)
        
    return new_nodes, total_layer_tokens


def build_tree(
    chunks: List[str],
    embedder,
    summarizer: Optional[LLMSummarizer] = None,
    min_cluster_size: int = 2,
    max_cluster_tokens: int = 8000,
    save_path: Optional[str] = None,
    source_pdf: str = "",
) -> RaptorTree:
    
    if summarizer is None:
        summarizer = LLMSummarizer()

    # Make sure parent directory exists if my_tree doesn't exist
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    
    checkpoints = {}
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                checkpoints = json.load(f)
            logger.info(f"Loaded {len(checkpoints)} checkpoints from {CHECKPOINT_PATH}. Resuming tree build!")
        except Exception as e:
            logger.warning(f"Could not load checkpoints: {e}")

    all_nodes = {}
    node_id   = 0

    # Leaf nodes
    embeddings = embedder.encode(chunks)

    for text, emb in zip(chunks, embeddings):
        all_nodes[node_id] = RaptorNode(
            index=node_id, text=text, embedding=emb,
            layer=0, children=[], token_count=_count_tokens(text),
        )
        node_id += 1

    logger.info(f"Leaf nodes: {len(all_nodes)}")

    current_layer_nodes = list(all_nodes.values())
    layer = 0
    total_llm_tokens = 0

    # Build loop
    while len(current_layer_nodes) > min_cluster_size:
        clusters = cluster_nodes(current_layer_nodes, max_cluster_tokens=max_cluster_tokens)

        if len(clusters) == 1:
            logger.info("Clustering produced 1 cluster — terminating build loop.")
            break

        logger.info(f"Layer {layer} → {layer + 1}: {len(clusters)} clusters, summarising...")

        # Run async summarization block containing httpx requests
        try:
            loop = asyncio.get_running_loop()
            is_running = True
        except RuntimeError:
            is_running = False

        if is_running:
            import threading
            
            result_container = {}
            def run_in_thread():
                try:
                    res = asyncio.run(_process_layer_summaries(clusters, node_id, summarizer, checkpoints))
                    result_container['result'] = res
                except Exception as inner_e:
                    result_container['error'] = inner_e
                    
            t = threading.Thread(target=run_in_thread)
            t.start()
            t.join()
            
            if 'error' in result_container:
                raise result_container['error']
            parent_nodes, tokens = result_container['result']
        else:
            parent_nodes, tokens = asyncio.run(_process_layer_summaries(clusters, node_id, summarizer, checkpoints))
        
        # We bulk-compute embeddings outside the fully async summarization phase
        texts_to_embed = [n.text for n in parent_nodes]
        parent_embeddings = embedder.encode(texts_to_embed)
        
        for parent, emb in zip(parent_nodes, parent_embeddings):
            parent.embedding = emb
            all_nodes[parent.index] = parent
            
        # Update node_id offset for next layer
        node_id += len(parent_nodes)
        total_llm_tokens += tokens

        layer += 1
        current_layer_nodes = parent_nodes
        logger.info(f"Layer {layer} done: {len(parent_nodes)} nodes | LLM tokens: {total_llm_tokens}")

    # Assemble + save
    root_ids = [n.index for n in current_layer_nodes]

    tree = RaptorTree(
        nodes      = all_nodes,
        root_ids   = root_ids,
        num_layers = layer + 1,
        embed_model= embedder.MODEL,
        source_pdf = source_pdf,
    )

    logger.info(f"Build complete: {len(all_nodes)} nodes, {tree.num_layers} layers, {len(root_ids)} roots")

    if save_path:
        save_tree(tree, save_path)
        
    # Clean up checkpoint on success
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logger.info(f"Tree build successful! Deleted checkpoint file: {CHECKPOINT_PATH}")

    return tree