import os, sys
for i, sys_path in enumerate(sys.path):
    if sys_path == os.path.dirname(__file__):
        sys.path[i] = os.path.dirname(os.path.dirname(__file__))

import json
import torch
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    set_seed,
    AutoTokenizer, 
    AutoConfig, 
)
from skip.skip_llama import SkipLlamaForCausalLM
from skip.skip_opt import SkipOPTForCausalLM
from utils.data_utils import get_trainloaders, chunk_by_sentence
from skip.algorithms import run_in_genetic_algorithm, run_in_beam_search
from utils.eval_utils import load_and_eval_ppl
sys.path.append(f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/lib/retriever-lib/src")
from faisslib.build_retriever import FaissRetrieverBuilder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--output_dir", type=str, default="motivation-2-5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--nclusters", type=int, default=10)
    parser.add_argument("--nsentences", type=int, default=12)
    parser.add_argument("--cluster_dir", type=str, default=None) # for cluster results of slebs
    args = parser.parse_args()
    
    logger.info(f"Arguments: {args}")
    set_seed(args.seed)

    encoder = SentenceTransformer('google-bert/bert-base-uncased').to('cuda')
    if args.dataset == 'wikitext2':
        train_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    elif args.dataset == 'c4':
        train_data = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    train_data = [data['text'] for data in train_data]

    logger.info("Chunking by sentence")
    train_data = chunk_by_sentence('\n'.join(train_data), args.nsentences)
    logger.info(f"Number of sentences: {len(train_data)}")

    train_data = list([data for data in filter(lambda x: len(x) > 0, train_data)])
    train_embs = encoder.encode(train_data)

    if args.nclusters != 0:
        kmeans = KMeans(n_clusters=args.nclusters, random_state=0).fit(train_embs)
        train_labels = kmeans.labels_
    else:
        train_labels = None

    # Reduce dimensions using UMAP or t-SNE
    tsne = TSNE(n_components=2, random_state=args.seed)
    train_embs_2d = tsne.fit_transform(train_embs)

    # Plot using matplotlib
    plt.figure(figsize=(10, 10))
    plt.scatter(train_embs_2d[:, 0], train_embs_2d[:, 1], c=train_labels, cmap='viridis')
    # plt.title('Sentence Embeddings Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(os.path.join(args.output_dir, 'embeddings_visualization.png'))
    plt.close()
    
    if args.cluster_dir is not None:
        rm_map = {}
        total_blocks = None
        for i in range(args.nclusters):
            sparsity = None
            with open(os.path.join(args.cluster_dir, f'cluster-{i}/sleb_results.txt'), 'r') as f:
                results = f.readlines()
                for line in results:
                    if "# Total Blocks:" in line:
                        sparsity = int(line.split(":")[-1].strip())
                        if total_blocks is None:
                            total_blocks = sparsity
                        else:
                            assert total_blocks == sparsity
                    elif "# Remove Blocks:" in line:
                        assert sparsity != None
                        rm_blocks = float(line.split(":")[-1].strip())
                        sparsity = rm_blocks / sparsity
                    elif "Block Removal Order:" in line:
                        rm_list = line.split(":")[-1].strip().strip("[]").split(", ")
                        rm_list = sorted([int(x) for x in rm_list])
                        if sparsity not in rm_map:
                            rm_map[sparsity] = []
                        rm_map[sparsity].append(rm_list)
                        sparsity = None
        
        # Plot heatmap for different sparsity levels
        for sparsity, rm_lists in rm_map.items():
            logger.info(f"Sparsity: {sparsity:.2f}")
            logger.info(f"Number of clusters: {len(rm_lists)}")
            logger.info(f"Map: {rm_lists}")
            heatmap_data = np.zeros((len(rm_lists), total_blocks))
            for idx, rm_list in enumerate(rm_lists):
                heatmap_data[idx, rm_list] = 1

            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, cmap="YlGnBu", cbar=True)
            plt.title(f'Heatmap for Sparsity Level: {sparsity:.2f}')
            plt.xlabel('Block Index')
            plt.ylabel('Cluster Index')
            plt.savefig(os.path.join(args.output_dir, f'heatmap_sparsity_{sparsity:.2f}.png'))
            plt.close()