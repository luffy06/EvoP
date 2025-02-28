import os, sys
for i, sys_path in enumerate(sys.path):
    if sys_path == os.path.dirname(__file__):
        sys.path[i] = os.path.dirname(os.path.dirname(__file__))

import json
import copy
import torch
import random
import pickle
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    set_seed,
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM
)
from skip.skip_llama import SkipLlamaForCausalLM
from skip.skip_opt import SkipOPTForCausalLM
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="llama-3-8b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--skip_layers", type=str, default=None)
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--eval_zeroshot", action="store_true")
    args = parser.parse_args()
    
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
    with open(args.skip_layers, "rb") as f:
        skip_layers = pickle.load(f)
    assert len(skip_layers.values()) == 1
    skip_layers = [val for val in skip_layers.values()]
    skip_layers = skip_layers[0]
    logger.info(f"Skip layers: {skip_layers}")

    model_name = args.model_name_or_path.split("/")[-1]
    model.name = model_name
    model.seqlen = config.max_position_embeddings
    model.num_layers = num_layers
    model.tokenizer = tokenizer
    model.set_skip_layers(skip_layers)
    logger.info(f"Model: {model_name}, SeqLen: {model.seqlen}, NumLayers: {num_layers}")
    logger.info(f"Config: {config}")
    
    if args.eval_ppl:
        logger.info(f"Starting PPL evaluation...")

        w2_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2', tokenizer=tokenizer)
        logger.info(f"WikiText-2 PPL = {w2_ppl:.2f}")

        c4_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='c4', tokenizer=tokenizer)
        logger.info(f"C4 PPL = {c4_ppl:.2f}")

    if args.eval_zeroshot:
        logger.info(f"Starting Zero-shot tasks evaluation...")
        if '30b' or '66b' or '70b' in args.model_name_or_path:
            parallelize = True
        else:
            parallelize = False

        tasks = ['arc_easy', 'arc_challenge', 'hellaswag', 
                 'piqa', 'boolq', 'winogrande']

        results = eval_zero_shot(args.model_name_or_path, model, tasks, parallelize=parallelize)
        results = results['results']

        for task in tasks:
            logger.info(f"{task}: {results[task]}")
