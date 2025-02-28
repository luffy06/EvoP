import os
import sys
import torch
import logging
import numpy as np
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from sentence_transformers import SentenceTransformer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.opt.modeling_opt import (
    OPTConfig, 
    OPTDecoderLayer,
    OPTDecoder,
    OPTModel,
    OPTForCausalLM,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
from dataclasses import dataclass

sys.path.append(f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/lib/retriever-lib/src")
from faisslib.retriever import FaissRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkipOPTDecoderLayer(OPTDecoderLayer):
    def __init__(self, config: OPTConfig, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx

class SkipOPTDecoder(OPTDecoder):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.skip_layers = []
        self.layers = nn.ModuleList([SkipOPTDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`. for padding use -1.

                [What are position IDs?](../glossary#position-ids)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        causal_attention_mask, attention_mask = self._update_causal_mask(
            inputs_embeds, input_shape, past_key_values_length, attention_mask, head_mask, output_attentions
        )
        # embed positions

        if position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            # cut positions if `past_key_values_length` is > 0
            position_ids = position_ids[:, past_key_values_length:]

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length, position_ids=position_ids)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if decoder_layer.layer_idx in self.skip_layers:
                continue

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    position_ids=position_ids,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class SkipOPTModel(OPTModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = SkipOPTDecoder(config)


class SkipOPTForCausalLM(OPTForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = SkipOPTModel(config)
        self.enable_retrieval = False

    def set_skip_layers(self, skip_layers: List[int]):
        self.model.decoder.skip_layers = skip_layers
    
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
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=position_ids,
        )
