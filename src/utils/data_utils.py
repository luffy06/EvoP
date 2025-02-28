import random
import logging
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def chunk_by_sentence(document, nsentences=1, end_words=['.', '! ']):
    sentences = []
    sentence = ''
    for i, w in enumerate(document):
        sentence += w
        if any([w.endswith(sw) for sw in end_words]):
            if i != len(document) - 1 \
                and (document[i + 1] == ' ' or document[i + 1] == '\n'):
                sentences.append(sentence.strip())
                sentence = ''
            elif i == len(document) -1:
                sentences.append(sentence.strip())
                sentence = ''
    if sentence != '':
        sentences.append(sentence.strip())
    sentences = list(filter(lambda x: x.strip() != '', sentences))
    merged_sentences = []
    for i in range(0, len(sentences), nsentences):
        merged_sentences.append(' '.join(sentences[i:i+nsentences]))
    return merged_sentences

class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    new_trainloader = []
    num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
    for i in range(0, num_batches):
        start =  i * batch_size
        end = min(start + batch_size, nsamples)
        batched_inp = []
        batched_tar = []
        for j in range(start, end):
            batched_inp.append(trainloader[j][0])
            batched_tar.append(trainloader[j][1])
        batched_inp = torch.cat(batched_inp)
        batched_tar = torch.cat(batched_tar)
        new_trainloader.append((batched_inp, batched_tar))
    del trainloader
    trainloader = new_trainloader
    del new_trainloader

    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer, batch_size):
   
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    new_trainloader = []
    num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
    for i in range(0, num_batches):
        start =  i * batch_size
        end = min(start + batch_size, nsamples)
        batched_inp = []
        batched_tar = []
        for j in range(start, end):
            batched_inp.append(trainloader[j][0])
            batched_tar.append(trainloader[j][1])
        batched_inp = torch.cat(batched_inp)
        batched_tar = torch.cat(batched_tar)
        new_trainloader.append((batched_inp, batched_tar))
    del trainloader
    trainloader = new_trainloader
    del new_trainloader

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, model='', batch_size=1):
    if tokenizer is None:
        tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer, batch_size)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer, batch_size)

def sample_data_from_clusters(train_data, seed, nclusters, ndata, data_max_seq_len, tokenizer, return_in_clusters=False):
    assert ndata % nclusters == 0, f"Number of samples {ndata} should be divisible by number of clusters {nclusters}"
    ndata_per_cluster = ndata // nclusters
    logger.info(f"Sampling {ndata_per_cluster} samples from each of {nclusters} clusters")
    from sklearn.cluster import KMeans
    from sentence_transformers import SentenceTransformer
    kmeans = KMeans(n_clusters=nclusters, random_state=seed)
    encoder = SentenceTransformer('google-bert/bert-base-uncased').to('cuda')
    train_data = list([data for data in filter(lambda x: len(x) > 0, train_data)])
    train_embs = encoder.encode(train_data, show_progress_bar=False)
    kmeans.fit(train_embs)
    cluster_idx = kmeans.predict(train_embs)
    train_idx = {}
    for i, cid in enumerate(cluster_idx):
        if cid not in train_idx:
            train_idx[cid] = []
        train_idx[cid].append(i)
    if return_in_clusters:
        train_data_cls = []
        for cid in sorted(train_idx.keys()):
            random.shuffle(train_idx[cid])
            train_ids = []
            tokens = []
            num_tokens = 0
            for i in train_idx[cid]:
                if num_tokens > data_max_seq_len:
                    tokens = torch.cat(tokens, dim=1).view(1, -1)
                    train_ids.append(tokens[:, :data_max_seq_len])
                    if len(train_ids) == ndata_per_cluster:
                        break
                    tokens = []
                    num_tokens = 0
                else:
                    tokens.append(tokenizer(train_data[i], return_tensors='pt').input_ids)
                    num_tokens += tokens[-1].numel()
            train_data_cls.append(torch.cat(train_ids))
        return train_data_cls
    else:
        cids = list(train_idx.keys())
        for cid in cids:
            random.shuffle(train_idx[cid])
        train_ids = []
        tokens = []
        num_tokens = 0
        i = 0
        while len(train_ids) < ndata and len(cids) > 0:
            sample_cid = np.random.choice(cids)
            if i >= len(train_idx[sample_cid]):
                del cids[np.where(cids == sample_cid)[0][0]]
                continue
            tokens.append(tokenizer(train_data[train_idx[sample_cid][i]], return_tensors='pt').input_ids)
            num_tokens += tokens[-1].numel()
            i += 1
            if num_tokens > data_max_seq_len:
                tokens = torch.cat(tokens, dim=1).view(1, -1)
                train_ids.append(tokens[:, :data_max_seq_len])
                if len(train_ids) == ndata:
                    break
                tokens = []
                num_tokens = 0
        train_ids = torch.cat(train_ids)
        train_ids = [train_ids]
        return train_ids

def get_wikitext2_trainenc(seed, ndata, tokenizer, data_max_seq_len, sample_strategy='random', data_grain='sample', **kwargs):
    train_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    train_data = [data['text'] for data in train_data]
    if 'semantic_chunk' in kwargs and kwargs['semantic_chunk']:
        logger.info("Chunking by sentence")
        train_data = chunk_by_sentence('\n'.join(train_data), kwargs['nsentences'] if 'nsentences' in kwargs else 1)
        logger.info(f"Number of sentences: {len(train_data)}")
    logger.info(f"Data grain: {data_grain}")
    if data_grain == 'sample':
        ndata = min(ndata, len(train_data))
    elif data_grain == 'token':
        raise ValueError("Not implemented")
        num_tokens = [len(tokenizer(i, return_tensors='pt').input_ids[0]) for i in train_data]
        num_tokens = np.cumsum(num_tokens)
        idx = np.argmax(num_tokens > ndata)
        ndata = idx
    else:
        raise ValueError(f"Unknown data grain: {data_grain}")
    logger.info(f"Number of samples: {ndata}")

    if sample_strategy == 'sleb':
        logger.info("Sampling using SLEB")
        random.shuffle(train_data)
        train_data = train_data[:kwargs['nsamples']]
        train_ids = tokenizer(kwargs['merge'].join(train_data), return_tensors='pt').input_ids
        train_ids = train_ids[:, :ndata*data_max_seq_len]
        train_ids = train_ids.view(ndata, data_max_seq_len)
        train_ids = [train_ids]
    elif sample_strategy == 'random':
        logger.info("Sampling randomly")
        random.shuffle(train_data)
        assert 'merge' in kwargs and type(kwargs['merge']) == str
        train_ids = []
        tokens = []
        num_tokens = 0
        for i in range(len(train_data)):
            tokens.append(tokenizer(train_data[i] + kwargs['merge'], return_tensors='pt').input_ids)
            num_tokens += tokens[-1].numel()
            if num_tokens > data_max_seq_len:
                tokens = torch.cat(tokens, dim=1).view(1, -1)
                train_ids.append(tokens[:, :data_max_seq_len])
                if len(train_ids) == ndata:
                    break
                tokens = []
                num_tokens = 0
        train_ids = torch.cat(train_ids)
        assert train_ids.shape[0] == ndata
        train_ids = [train_ids]
    elif sample_strategy == 'clustering':
        logger.info("Sampling from clusters")
        train_ids = sample_data_from_clusters(
            train_data, 
            seed, 
            kwargs['nclusters'], 
            ndata, 
            data_max_seq_len, 
            tokenizer,
            return_in_clusters=kwargs['return_in_clusters'] if 'return_in_clusters' in kwargs else False
        )
    else:
        raise ValueError(f"Unknown sample strategy: {sample_strategy}")
    return train_ids

def get_c4_trainenc(seed, ndata, tokenizer, data_max_seq_len, sample_strategy='random', data_grain='sample', **kwargs):
    train_data = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    train_data = [data['text'] for data in train_data]
    if 'semantic_chunk' in kwargs and kwargs['semantic_chunk']:
        logger.info("Chunking by sentence")
        train_data = chunk_by_sentence('\n'.join(train_data), kwargs['nsentences'] if 'nsentences' in kwargs else 1)
        logger.info(f"Number of sentences: {len(train_data)}")
    logger.info(f"Data grain: {data_grain}")
    if data_grain == 'sample':
        ndata = min(ndata, len(train_data))
    elif data_grain == 'token':
        raise ValueError("Not implemented")
        num_tokens = [len(tokenizer(i, return_tensors='pt').input_ids[0]) for i in train_data]
        num_tokens = np.cumsum(num_tokens)
        idx = np.argmax(num_tokens > ndata)
        ndata = idx
    else:
        raise ValueError(f"Unknown data grain: {data_grain}")
    logger.info(f"Number of samples: {ndata}")

    if sample_strategy == 'sleb':
        logger.info("Sampling using SLEB")
        random.shuffle(train_data)
        train_data = train_data[:kwargs['nsamples']]
        train_ids = tokenizer(kwargs['merge'].join(train_data), return_tensors='pt').input_ids
        train_ids = train_ids[:, :ndata*data_max_seq_len]
        train_ids = train_ids.view(ndata, data_max_seq_len)
    elif sample_strategy == 'random':
        logger.info("Sampling randomly")
        random.shuffle(train_data)
        assert 'merge' in kwargs and type(kwargs['merge']) == str
        train_ids = []
        tokens = []
        num_tokens = 0
        for i in range(len(train_data)):
            tokens.append(tokenizer(train_data[i] + kwargs['merge'], return_tensors='pt').input_ids)
            num_tokens += tokens[-1].numel()
            if num_tokens > data_max_seq_len:
                tokens = torch.cat(tokens, dim=1).view(1, -1)
                train_ids.append(tokens[:, :data_max_seq_len])
                if len(train_ids) == ndata:
                    break
                tokens = []
                num_tokens = 0
        train_ids = torch.cat(train_ids)
        assert train_ids.shape[0] == ndata
    elif sample_strategy == 'clustering':
        logger.info("Sampling from clusters")
        train_ids = sample_data_from_clusters(
            train_data, 
            seed, 
            kwargs['nclusters'], 
            ndata, 
            data_max_seq_len, 
            tokenizer,
            return_in_clusters=kwargs['return_in_clusters'] if 'return_in_clusters' in kwargs else False
        )
    else:
        raise ValueError(f"Unknown sample strategy: {sample_strategy}")
    return train_ids

def get_trainloaders(name, ndata=128, seed=0, tokenizer=None, data_max_seq_len=2048, sample_strategy='random', data_grain='sample', **kwargs):
    if 'wikitext2' in name:
        return get_wikitext2_trainenc(seed, ndata, tokenizer, data_max_seq_len, sample_strategy, data_grain, **kwargs)
    if 'c4' in name:
        return get_c4_trainenc(seed, ndata, tokenizer, data_max_seq_len, sample_strategy, data_grain, **kwargs)
    raise ValueError(f"Unknown dataset: {name}")