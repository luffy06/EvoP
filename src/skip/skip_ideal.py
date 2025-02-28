import os, sys
for i, sys_path in enumerate(sys.path):
    if sys_path == os.path.dirname(__file__):
        sys.path[i] = os.path.dirname(os.path.dirname(__file__))
import time
import json
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
from datasets import load_dataset
from tqdm import tqdm
from utils.data_utils import get_trainloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

def compute_similarity(a, b, cross_tokens=False):
    assert a.shape[-2] == b.shape[-2]
    if cross_tokens:
        seq_len = a.shape[-2]
        a = a.unsqueeze(-2).expand(-1, -1, -1, seq_len, -1).transpose(-2, -3)
        b = b.unsqueeze(-2).expand(-1, -1, -1, seq_len, -1)
    scores = cosine(a, b)
    return scores

def compute_lm_loss(lm_logits, input_ids, max_model_seq_length, batch_size):
    # Shift logits and labels for next token prediction
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Compute loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

    # Calculate negative log likelihood
    loss = loss.detach().cpu().float() * max_model_seq_length * batch_size

    return loss

def get_wikitext_enc(seed, nsamples, tokenizer, data_name='wikitext-2'):
    traindata = load_dataset('wikitext', data_name + '-raw-v1', split='train')
    testdata = load_dataset('wikitext', data_name + '-raw-v1', split='test')

    traindata = traindata.shuffle(seed=seed)
    trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    return trainenc, testenc

def generate_skip_layers(skip_code, num_layers):
    skip_layers = []
    for i in range(num_layers):
        if skip_code & (1 << i):
            skip_layers.append(i)
    return skip_layers

def compute_sparsity(skip_code, num_layers):
    return np.sum([1 if skip_code & (1 << i) else 0 for i in range(num_layers)]) / num_layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--model_name_or_path", type=str, default="llama-3-8b")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--data_log_path", type=str, default="logs/data_random.log")
    parser.add_argument("--ratio", type=float, default=0.01)
    args = parser.parse_args()

    data_configs = json.load(open(args.data_log_path, "r"))
    for key, value in data_configs.items():
        setattr(args, key, value)
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
    model_name = args.model_name_or_path.split("/")[-1]
    logger.info(config)

    train_ids = get_trainloaders(
        args.dataset, 
        seed=args.seed, 
        tokenizer=tokenizer, 
        **data_configs
    )
    assert len(train_ids) == 1
    train_ids = train_ids[0]
    logger.info(f"Dataloader({args.dataset}) loaded.")
    logger.info(f"Train data shape: {train_ids.shape}, number of tokens {train_ids.numel()}")

    start_time = time.time()
    max_model_seq_length = model.config.max_position_embeddings
    nsamples = train_ids.numel() // max_model_seq_length
    logger.info(f"Number of samples: {nsamples}")
    loss_list = {}
    codes = np.power(2, num_layers + 1)
    codes = [i for i in range(codes)]
    np.random.shuffle(codes)
    ncodes = int(args.ratio * len(codes))
    codes = codes[:ncodes]
    if 0 not in codes:
        codes.append(0)
    codes = sorted(codes)
    for i in range(0, nsamples, args.batch_size):
        j = min(i + args.batch_size, nsamples)
        input_ids = train_ids[:, (i*max_model_seq_length):(j*max_model_seq_length)].to(args.device)
        input_ids = input_ids.view(-1, max_model_seq_length)

        bar = tqdm(codes)
        for skip_code in bar:
            skip_layers = generate_skip_layers(skip_code, num_layers)
            with torch.no_grad():
                model.set_skip_layers(skip_layers)
                output = model(input_ids)
                lm_logits = output.logits
                loss = compute_lm_loss(lm_logits, input_ids, max_model_seq_length, j - i)
                loss = loss.detach().cpu().numpy().item()
                if skip_code not in loss_list:
                    loss_list[skip_code] = [loss]
                else:
                    loss_list[skip_code].append(loss)
                bar.set_description(f"[{j}/{nsamples}] Skip code: {skip_code}, loss: {loss}")
                del output, lm_logits, loss
        del input_ids
    end_time = time.time()
    logger.info("Loss computation done, elapsed time: {:.2f}s".format(end_time - start_time))

    start_time = time.time()
    sparsity_loss = {}
    best_skip_code = {}
    min_loss = {}
    for skip_code, losses in loss_list.items():
        sparsity = compute_sparsity(skip_code, num_layers)
        sparsity = int(round(sparsity, 2) * 100)
        loss = np.sum(losses)
        if sparsity not in sparsity_loss:
            sparsity_loss[sparsity] = [loss]
        else:
            sparsity_loss[sparsity].append(loss)
        if sparsity not in best_skip_code:
            best_skip_code[sparsity] = skip_code
            min_loss[sparsity] = loss
        elif loss < min_loss[sparsity]:
            best_skip_code[sparsity] = skip_code
            min_loss[sparsity] = loss
    end_time = time.time()
    logger.info("Sparsity and loss computed, elapsed time: {:.2f}s".format(end_time - start_time))
    
    start_time = time.time()
    for sparsity, losses in sorted(sparsity_loss.items(), key=lambda x: x[0]):
        mean_loss = np.mean(losses)
        max_loss = np.max(losses)
        min_loss = np.min(losses)
        logger.info(f"Sparsity: {sparsity}%")
        logger.info(f"Loss: {mean_loss} (max: {max_loss}, min: {min_loss})")
    end_time = time.time()
    logger.info("Results printed, elapsed time: {:.2f}s".format(end_time - start_time))

    start_time = time.time()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, f"sparsity_loss_{model_name}.json"), "w") as f:
        json.dump(sparsity_loss, f, indent=4)
    with open(os.path.join(args.output_dir, f"best_skip_code_{model_name}.json"), "w") as f:
        json.dump(best_skip_code, f, indent=4)
    end_time = time.time()
    logger.info(f"Results saved in {args.output_dir}, elapsed time: {end_time - start_time:.2f}s")  

    start_time = time.time()
    fig = plt.figure()
    fig_data = [(sparsity, losses) for sparsity, losses in sorted(sparsity_loss.items(), key=lambda x: x[0])]
    sparsities, losses = zip(*fig_data)
    if sparsities[0] == 0:
        sparsities = sparsities[1:]
        origin_loss = losses[0]
        if len(origin_loss) > 1:
            origin_loss = origin_loss[0]
        losses = losses[1:]
    plt.boxplot(losses, labels=[f"{s}%" for s in sparsities])
    plt.axhline(y=origin_loss, color='r', linestyle='--', label='Original Loss')
    plt.xlabel("Sparsity")
    plt.ylabel("Loss")
    plt.title("Loss Distribution by Sparsity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"sparsity_loss_boxplot_{model_name}.png"))
    plt.close()
    end_time = time.time()
    logger.info(f"Boxplot saved in {args.output_dir}, elapsed time: {end_time - start_time:.2f}s")
    