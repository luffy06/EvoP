import os, sys
for i, sys_path in enumerate(sys.path):
    if sys_path == os.path.dirname(__file__):
        sys.path[i] = os.path.dirname(os.path.dirname(__file__))

import torch
import pickle
import argparse
import numpy as np

from utils import latency_utils
from utils.block_remove import block_remove
from comparisons.sleb.utils import model_utils
from comparisons.wanda.prune import prune_wanda
from comparisons.slicegpt.slicegpt import hf_utils, data_utils, layernorm_fusion, rotate
from comparisons.slicegpt.slicegpt.config import config
from comparisons.slicegpt.slicegpt.slicing_scheduler import ConstSlicingScheduler
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed
)
from skip.skip_llama import SkipLlamaForCausalLM
from skip.skip_opt import SkipOPTForCausalLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Latency')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf', help='Model Name')
    parser.add_argument('--method', type=str, default='sleb', help='Method')
    parser.add_argument('--seed', type=int, default=31, help='Seed')
    parser.add_argument('--sparsity', type=float, default=0.2, help='Sparsity')
    parser.add_argument('--generation', action='store_true', help='Generation')
    parser.add_argument('--data_log_path', type=str, default=None, help='Path to save the calibration data.')
    parser.add_argument('--retriever_dir', type=str, default=None, help='Path to the retriever directory.')
    parser.add_argument('--skip_tables', type=str, default=None, help='Path to the skip tables.')
    args = parser.parse_args()
    set_seed(args.seed)

    gpu_num = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if args.method == 'dense':
        model = model_utils.get_llm(args.model_name)
        num_of_blocks = model.config.num_hidden_layers
    elif args.method == 'sleb':
        model = model_utils.get_llm(args.model_name)
        num_of_blocks = model.config.num_hidden_layers
        num_removal = int(np.ceil(num_of_blocks * args.sparsity))
        removal_list = [i+1 for i in range(num_removal)]
        model = block_remove(model, removal_list)
    elif args.method == 'wanda':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )
        model.seqlen = model.config.max_position_embeddings 
        model.name = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        model.eval()
        args.dataset = 'wikitext2'
        prune_wanda(args, model, tokenizer, torch.device("cuda"), prune_n=2, prune_m=4)
    elif args.method == 'slicegpt':
        # load one of the pre-trained models
        config.device = torch.device("cuda")
        config.dtype = torch.bfloat16
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            args.model_name, None, token=None, dtype=torch.bfloat16
        )
        model = model_adapter.model
        def reset_model_device() -> None:
            model.to(config.device)
        dataset = data_utils.get_dataset("wikitext2")
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        train_loader = data_utils.prepare_dataloader(
            dataset=train_dataset,
            tokenizer=tokenizer,
            max_seqlen=2048,
            batch_size=16,
            nsamples=5,
            varied_seqlen=False,
            seed=args.seed,
        )
        # replace modules with compressible equivalents
        reset_model_device()
        layernorm_fusion.replace_layers(model_adapter)
        # fuse layernorms and add rotations to skip connections
        reset_model_device()
        layernorm_fusion.fuse_modules(model_adapter)
        # compute new embedding dimension given the desired sparsity level
        new_embedding_dimension = int((1 - args.sparsity) * model_adapter.hidden_size)
        # round (down) to the nearest multiple of round_interval
        new_embedding_dimension -= new_embedding_dimension % 8
        reset_model_device()
        scheduler = ConstSlicingScheduler(new_embedding_dimension)
        rotate.rotate_and_slice(model_adapter, train_loader, scheduler, final_orientation="random")
        reset_model_device()
    elif args.method == 'skip-ret':
        model_config = AutoConfig.from_pretrained(args.model_name)
        model_config.use_cache = True
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        if "llama" in args.model_name.lower():
            model = SkipLlamaForCausalLM.from_pretrained(args.model_name, config=model_config, torch_dtype=torch.bfloat16, device_map="auto",)
        elif "opt" in args.model_name.lower():
            model = SkipOPTForCausalLM.from_pretrained(args.model_name, config=model_config, torch_dtype=torch.bfloat16, device_map="auto",)
        else:
            raise ValueError("Model name should be either llama or opt.")
        with open(args.skip_tables, "rb") as f:
            skip_tables = pickle.load(f)
        model_name = args.model_name.split("/")[-1]
        model.name = model_name
        model.num_layers = model_config.num_hidden_layers
        model.seqlen = model_config.max_position_embeddings
        model.set_retriever(args.retriever_dir, 8, skip_tables)
        model.tokenizer = tokenizer
        model.to("cuda")
        removal_list = set()
        for skip_layers in skip_tables.values():
            removal_list.update(skip_layers)
        removal_list = sorted(list(removal_list))
        print(f"all removed layers: {removal_list}")
        model = block_remove(model, removal_list)        
    else:
        raise ValueError(f"Invalid Method {args.method}")
    model.eval()
    
    print("==================================================")
    print("Experiment Environment")
    print(f"Current GPU: {gpu_name}")
    print(f"# GPU: {str(gpu_num)}")
    print(f"Model Name: {args.model_name}")
    print(f"Infernce type : {'Token Generation' if args.generation else 'Prompt Processing'}")
    print("==================================================")

    # latency for dense model
    latency = latency_utils.test_latency(model, args.generation)
    print(f"Method: {args.method}, Sparsity: {args.sparsity}, Latency: {latency:.2f}")