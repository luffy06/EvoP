import os, sys
for i, sys_path in enumerate(sys.path):
    if sys_path == os.path.dirname(__file__):
        sys.path[i] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

import argparse
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from importlib.metadata import version
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
from comparisons.wanda.prune import (
    prune_wanda, 
    prune_magnitude, 
    prune_sparsegpt, 
    prune_ablate, 
    check_sparsity, 
    find_layers
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_llm(model_name):
    logger.info(f"loading llm model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    model.name = model_name
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--dataset', type=str, choices=['wikitext2', 'c4'], help='Dataset for calibration.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--data_log_path', type=str, default=None, help='Path to save the calibration data.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--eval_ppl', action="store_true")
    parser.add_argument("--eval_zeroshot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    set_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    logger.info(f"loading llm model {model_name}")
    model = get_llm(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    # if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
    #     device = model.hf_device_map["lm_head"]
    logger.info(f"use device  {device}")

    if args.sparsity_ratio != 0:
        logger.info("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    sparsity_ratio = check_sparsity(model)
    logger.info(f"sparsity sanity check {sparsity_ratio:.4f}")
    ################################################################

    if args.eval_ppl:
        logger.info(f"Starting PPL evaluation...")
        
        w2_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2')
        logger.info(f"WikiText-2 PPL = {w2_ppl:.2f}")

        c4_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='c4')
        logger.info(f"C4 PPL = {c4_ppl:.2f}")

    if args.eval_zeroshot:
        logger.info(f"Starting Zero-shot tasks evaluation...")
        if '30b' or '66b' or '70b' in model_name:
            parallelize = True
        else:
            parallelize = False

        tasks = ['arc_easy', 'arc_challenge', 'hellaswag', 
                 'piqa', 'boolq', 'winogrande']

        results = eval_zero_shot(args.model, model, tasks, parallelize=parallelize)
        results = results['results']

        for task in tasks:
            logger.info(f"{task}: {results[task]}")

if __name__ == '__main__':
    main()