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

from sentence_transformers import SentenceTransformer
from transformers import (
    set_seed,
    AutoTokenizer, 
    AutoConfig, 
)
from skip.skip_llama import SkipLlamaForCausalLM
from skip.skip_opt import SkipOPTForCausalLM
from utils.data_utils import get_trainloaders
from skip.algorithms import run_in_genetic_algorithm, run_in_beam_search
from utils.eval_utils import load_and_eval_ppl
sys.path.append(f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/lib/retriever-lib/src")
from faisslib.build_retriever import FaissRetrieverBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--model_name_or_path", type=str, default="llama-3-8b")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--data_log_path", type=str, default="logs/data_random.log")
    parser.add_argument("--algo_log_path", type=str, default="logs/algo_beam.log")
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--nrm_blocks", type=int, default=None)
    parser.add_argument("--eval_ppl", action="store_true")
    args = parser.parse_args()
    
    data_configs = json.load(open(args.data_log_path, "r"))
    for key, value in data_configs.items():
        setattr(args, key, value)
    algo_configs = json.load(open(args.algo_log_path, "r"))
    for key, value in algo_configs.items():
        setattr(args, key, value)
    del algo_configs['search_strategy']
    logger.info(f"Data configs: {data_configs}")
    logger.info(f"Algo configs: {algo_configs}")
    logger.info(f"Arguments: {args}")
    set_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if "llama" in args.model_name_or_path.lower():
        model = SkipLlamaForCausalLM.from_pretrained(args.model_name_or_path, config=config, torch_dtype=torch.bfloat16, device_map="auto",)
    elif "opt" in args.model_name_or_path.lower():
        model = SkipOPTForCausalLM.from_pretrained(args.model_name_or_path, config=config, torch_dtype=torch.bfloat16, device_map="auto",)
    else:
        raise ValueError("Model name should be either llama or opt.")
    model.eval()
    model.to(args.device)

    num_layers = config.num_hidden_layers
    assert not (args.sparsity == None and args.nrm_blocks == None), "Either sparsity or nrm_blocks should be provided."
    if args.sparsity is not None:
        nrm_blocks = int(np.ceil(num_layers * args.sparsity))
    elif args.nrm_blocks is not None:
        nrm_blocks = args.nrm_blocks
    else:
        nrm_blocks = None
    logger.info(f"Number of removed blocks: {nrm_blocks}")

    model_name = args.model_name_or_path.split("/")[-1]
    model.name = model_name
    model.num_layers = num_layers
    model.seqlen = config.max_position_embeddings
    model.tokenizer = tokenizer
    logger.info(f"Model: {model_name}, SeqLen: {model.seqlen}, NumLayers: {num_layers}")
    logger.info(f"Config: {config}")

    train_ids_cls = get_trainloaders(
        args.dataset, 
        seed=args.seed, 
        tokenizer=tokenizer, 
        **data_configs
    )
    logger.info(f"Dataloader ({args.dataset}) loaded, smapled strategy {args.sample_strategy}.")
    logger.info(f"Number of clusters: {len(train_ids_cls)}")
    total_tokens = 0
    for cid, train_ids in enumerate(train_ids_cls):
        total_tokens += np.sum([input_ids.numel() for input_ids in train_ids])
    logger.info(f"Total number of tokens: {total_tokens}")

    skip_table = {}
    for cid, train_ids in enumerate(train_ids_cls):
        nsamples = len(train_ids)
        num_tokens = [input_ids.numel() for input_ids in train_ids]
        logger.info(f"Number of samples: {nsamples}")
        logger.info(f"Number of tokens in {cid}-th cluster: {np.sum(num_tokens)}")
        logger.info(f"Avg number of tokens in {cid}-th cluster: {np.mean(num_tokens)}")

        if nsamples == 0:
            logger.info(f"Warning: Number of samples is zero. Skipping...")
            continue
        
        if args.search_strategy == "genetic":
            best_loss, skip_layers = run_in_genetic_algorithm(model, nrm_blocks, train_ids, algo_configs)
        elif args.search_strategy == "beam":
            best_loss, skip_layers = run_in_beam_search(model, nrm_blocks, train_ids, args.num_beam)
        logger.info(f"Best loss: {best_loss}")
        logger.info(f"Best skip layers: {sorted(skip_layers)}")
        skip_table[cid] = skip_layers

        if args.eval_ppl:
            model.set_skip_layers(skip_layers)
            logger.info(f"Starting PPL evaluation...")

            w2_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2', tokenizer=tokenizer)
            logger.info(f"WikiText-2 PPL = {w2_ppl:.2f}")

            c4_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='c4', tokenizer=tokenizer)
            logger.info(f"C4 PPL = {c4_ppl:.2f}")

    with open(f"{args.output_dir}/{model_name}_{args.dataset}_{nrm_blocks}.pkl", "wb") as f:
        pickle.dump(skip_table, f)

    if hasattr(args, "return_in_clusters") and args.return_in_clusters:
        data_configs['ndata'] = data_configs['nclusters'] * data_configs['nbuilddata']
        data_configs['return_in_text'] = True
        train_data = get_trainloaders(
            args.dataset, 
            seed=args.seed, 
            tokenizer=tokenizer, 
            max_seq_len=model.seqlen,
            **data_configs
        )
        encoder = SentenceTransformer('google-bert/bert-base-uncased').to('cuda')
        if not os.path.exists(f"{args.output_dir}/emb"):
            os.makedirs(f"{args.output_dir}/emb")
        if not os.path.exists(f"{args.output_dir}/value"):
            os.makedirs(f"{args.output_dir}/value")
        num_emb = 0
        emb_dim = 0
        values = []
        for cid, text_in_cluster in enumerate(train_data):
            text_in_cluster = tokenizer.batch_decode(text_in_cluster)
            embeddings = encoder.encode(text_in_cluster, show_progress_bar=False)
            num_emb += embeddings.shape[0]
            emb_dim = embeddings.shape[1]
            logger.info(f"Number of embeddings in {cid}-th cluster: {embeddings.shape}")
            values.extend([cid for _ in range(embeddings.shape[0])])
            df = pd.DataFrame({
                'embedding': embeddings.tolist(),
            })
            emb_path = f"{args.output_dir}/emb/{model_name}_{args.dataset}_{nrm_blocks}-{cid}.json.gz"
            df.to_json(emb_path, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})
        
        with open(f"{args.output_dir}/value/value.pkl", "wb") as f:
            pickle.dump(values, f)

        with open(f"{args.output_dir}/metadata.json", "w") as f:
            json.dump({
                'num_emb': num_emb,
                'emb_dim': emb_dim,
            }, f)

        builder = FaissRetrieverBuilder(
            'google-bert/bert-base-uncased',
            data_dir=None, 
            output_dir=args.output_dir, 
            batch_size=32,
            device_id=-1,
            do_chunk=False,
            do_encode=False,
        )
        builder.build(
            build_index=True,
            build_db=True,
            index_type=args.index_type, 
            train_ratio=1,
            use_embeddings_as_values=False
        )

