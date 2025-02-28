import os
import sys
import torch
import logging
import numpy as np
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from sentence_transformers import SentenceTransformer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import (
    LlamaConfig, 
    LlamaForCausalLM,
    Unpack,
    KwargsForCausalLM,
    LlamaModel,
    FlashAttentionKwargs,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
from dataclasses import dataclass

sys.path.append(f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/lib/retriever-lib/src")
from faisslib.retriever import FaissRetriever


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkipLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.skip_layers = []

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if decoder_layer.self_attn.layer_idx in self.skip_layers:
                continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class SkipLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = SkipLlamaModel(config)
        self.enable_retrieval = False

    def set_skip_layers(self, skip_layers: List[int]):
        self.model.skip_layers = skip_layers
    
    def set_retriever(self, retriever_path, topk, skip_tables):
        self.encoder = SentenceTransformer('google-bert/bert-base-uncased').to('cuda')
        self.retriever = FaissRetriever(retriever_path, nprobe=512, topk=topk)
        self.skip_tables = skip_tables
        self.enable_retrieval = True

    def compute_skip_layers(self, neighbors, skip_grain='model-wise'):
        neighbors = {k: np.array(v["results"]) for k, v in neighbors.items()}
        for key, values in neighbors.items():
            min_val = np.min(values[:, -1])
            max_val = np.max(values[:, -1])
            if max_val > min_val:
                neighbors[key][:, -1] = (values[:, -1] - min_val) / (max_val - min_val)
            else:
                neighbors[key][:, -1] = 0
        skip_pattern = {}
        for value in neighbors.values():
            cids = value[:, 0]
            weights = value[:, 1]
            for cid, weight in zip(cids, weights):
                skip_layers_i = self.skip_tables[cid] if cid in self.skip_tables else []
                if skip_grain == 'model-wise':
                    skip_layers_i = sorted(skip_layers_i)
                    skip_layers_i = ','.join([str(l) for l in skip_layers_i])
                    if skip_layers_i not in skip_pattern:
                        skip_pattern[skip_layers_i] = weight
                    else:
                        skip_pattern[skip_layers_i] += weight
                elif skip_grain == 'layer-wise':
                    for l in skip_layers_i:
                        if l not in skip_pattern:
                            skip_pattern[l] = weight
                        else:
                            skip_pattern[l] += weight
                else:
                    raise NotImplementedError(f"Skip grain {skip_grain} not implemented")
        if skip_grain == 'model-wise':
            skip_pattern = sorted(skip_pattern.items(), key=lambda x: x[1], reverse=True)
            skip_layers = skip_pattern[0][0].split(',')
            skip_layers = [int(l) for l in skip_layers]
        elif skip_grain == 'layer-wise':
            num_skipped_layers = [val for val in self.skip_tables.values()]
            num_skipped_layers = len(num_skipped_layers[0])
            skip_pattern = sorted(skip_pattern.items(), key=lambda x: x[1], reverse=True)
            skip_layers = [int(l) for l, _ in skip_pattern[:num_skipped_layers]]
        else:
            raise NotImplementedError(f"Skip grain {skip_grain} not implemented")
        skip_layers = sorted(skip_layers)
        return skip_layers

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.enable_retrieval:
            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            embeddings = self.encoder.encode(texts, show_progress_bar=False)
            def post_process_func(value, distance, neighbors):
                value = value['value']
                if 'results' not in neighbors:
                    neighbors['results'] = [(value, distance)]
                else:
                    neighbors['results'].append((value, distance))
                return neighbors
            neighbors = self.retriever.search(embeddings, post_process_func)
            skip_layers = self.compute_skip_layers(neighbors)
            self.set_skip_layers(skip_layers)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
