# Import necessary modules
import os
import torch
import logging
import torch.nn as nn

from tqdm import tqdm
from utils.data_utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def compute_sparsity(skip_code, num_layers):
    return np.sum([1 if skip_code & (1 << i) else 0 for i in range(num_layers)]) / num_layers

def compute_lm_loss(lm_logits, input_ids, max_model_seq_length, batch_size=1):
    # Shift logits and labels for next token prediction
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Compute loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

    # Calculate negative log likelihood
    loss = loss.detach().cpu().float() * max_model_seq_length * batch_size

    return loss

def compute_loss(skip_layers, model, train_ids, nrm_blocks):
    model.set_skip_layers(skip_layers)
    sum_loss = 0
    for input_ids in train_ids:
        input_ids = input_ids.view(1, -1).to(model.device)
        with torch.no_grad():
            model_outputs = model(input_ids)
            lm_logits = model_outputs.logits
            loss = compute_lm_loss(lm_logits, input_ids, model.seqlen)
            loss = loss.detach().cpu().numpy().item()
            sum_loss += loss
            del model_outputs, lm_logits, loss
    return sum_loss

def compute_loss_ga(skip_layers, model, train_ids, nrm_blocks):
    skip_layers = [i for i in range(model.num_layers) if skip_layers[i] == 1]
    return compute_loss(skip_layers, model, train_ids, nrm_blocks)

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
@torch.no_grad()
def load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2', testloader=None, tokenizer=None):
    logger.info(f"Evaluating on {dataset}")

    # Get the test loader
    if testloader is None:
        if tokenizer is None:
            tokenizer = get_tokenizer(model.name)

        _, testloader = get_loaders(
            dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
        )
        logger.info(f"Dataset Loaded.")

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl(model, testloader, 1, device)
    return ppl_test 

@torch.no_grad()
def eval_ppl(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    logger.info(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0,nsamples,bs)):

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()

torch.no_grad()
def eval_zero_shot(model_name, pruned_model, task_list=['piqa','winogrande','hellaswag','arc_challenge','arc_easy'], 
        num_fewshot=3, parallelize=False):
    
    from lm_eval import tasks, evaluator, utils
    task_manager = tasks.TaskManager(include_path='lm-evaluation-harness/lm_eval/tasks')
 
    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
    task_missing = [
        task
        for task in task_list
        if task not in task_names and "*" not in task
        ]  # we don't want errors if a wildcard ("*") task name was used
    
    
    model_args = f"pretrained={model_name},"
    if parallelize:
        model_args = f"pretrained={model_name},parallelize=True"

    results = evaluator.simple_evaluate(
        model='hf',
        model_args=model_args,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size='auto',
        max_batch_size=None,
        device='cuda:0',
        use_cache=None,
        limit=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=False,
        gen_kwargs=None,
        task_manager=task_manager,
        pruned_model=pruned_model,
    )

    return results 