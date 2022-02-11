__version__ = '0.0.0'

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM, \
    PreTrainedTokenizer, PreTrainedModel, BatchEncoding, BartForSequenceClassification
from typing import Any, Dict, Optional, List, Tuple
from types import MethodType
import torch.nn as nn
from ratransformers.t5 import T5RelationalAttention, T5Attention
from ratransformers.bart import BartRelationalAttention, BartAttention
import torch
import functools
import numpy as np


class RATransformer:

    def __init__(self, pretrained_model_name_or_path: str, relation_kinds: List[str],
                 alias_model_name: Optional[str] = '', tokenizer_cls: Optional[PreTrainedTokenizer] = None,
                 model_cls: Optional[PreTrainedTokenizer] = None):

        if tokenizer_cls is None:
            tokenizer_cls = AutoTokenizer

        if model_cls is None:
            if (alias_model_name or pretrained_model_name_or_path).startswith('t5'):
                model_cls = AutoModelForSeq2SeqLM
            elif (alias_model_name or pretrained_model_name_or_path).startswith('tapas'):
                model_cls = BartForSequenceClassification
            else:
                model_cls = AutoModel

        self.tokenizer = tokenizer_cls.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        self.model = model_cls.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)

        self.relational_kind_to_index = {t: i + 1 for i, t in enumerate(relation_kinds)}
        self.input_relation_kinds: List[torch.Tensor] = [] # will be used to pass variable by reference to attention layers

        # change attention layers with relational ones
        for module_name, module in self.model.named_modules():
            if self._change_this_module(module_name=module_name, module=module,
                                        model_name=alias_model_name or pretrained_model_name_or_path):
                self._change_attention_layer(attention_layer=module, num_relation_kinds=len(relation_kinds))

        def model_prefix_function(function):
            @functools.wraps(function)
            def run(*args, **kwargs):
                if 'input_relations' in kwargs:
                    if self.input_relation_kinds:
                        self.input_relation_kinds[0].cpu().detach()
                        self.input_relation_kinds.pop()
                    self.input_relation_kinds.append(kwargs['input_relations'])
                    del kwargs['input_relations']
                torch.cuda.empty_cache()
                return function(*args, **kwargs)
            return run

        def tokenizer_suffix_function(function):
            @functools.wraps(function)
            def run(*args, **kwargs):
                if 'input_relations' not in kwargs:
                    raise Exception("tokenizer expects 'input_relations' argument")
                input_relations = kwargs.pop('input_relations')
                kwargs['return_offsets_mapping'] = True
                out = function(*args, **kwargs)
                out['input_relations'] = self.get_new_input_relation_kinds(
                    tokenizer_outputs=out, input_relations=input_relations
                )
                return out
            return run

        # change model's call, forward and generate method
        self.model.__call__ = model_prefix_function(self.model.__call__)
        self.model.forward = model_prefix_function(self.model.forward)
        self.model.generate = model_prefix_function(self.model.generate)

        # change tokenizer's call and encode plus methods
        self.tokenizer.__call__ = tokenizer_suffix_function(self.tokenizer.__call__)
        self.tokenizer.batch_encode_plus = tokenizer_suffix_function(self.tokenizer.batch_encode_plus)
        self.tokenizer.encode_plus = tokenizer_suffix_function(self.tokenizer.encode_plus)

    def get_new_input_relation_kinds(
        self,
        tokenizer_outputs: BatchEncoding,
        input_relations: Optional[List[Dict[Tuple[int, int], Dict[Tuple[int, int], str]]]] = None
    ) -> torch.Tensor:

        assert 'offset_mapping' in tokenizer_outputs, "Run tokenizer with return_offsets_mapping=True"

        aux_input_relation_kinds = np.zeros(
            (len(tokenizer_outputs['input_ids']), len(tokenizer_outputs['input_ids'][0]), len(tokenizer_outputs['input_ids'][0])),
            dtype=np.int64
        )
        if input_relations is not None:
            if isinstance(input_relations, dict):
                input_relations = [input_relations]
            assert len(tokenizer_outputs['offset_mapping']) == len(input_relations)
            for batch_idx, (token_mappings, relations) in enumerate(zip(tokenizer_outputs['offset_mapping'], input_relations)):

                for word_i_span, word_relations in relations.items():
                    word_i_token_ids = [
                        token_idx for token_idx, token_span in enumerate(token_mappings)
                        if max(0, min(token_span[1], word_i_span[1]) - max(token_span[0], word_i_span[0])) > 0 # check for word/token overlaps
                    ]
                    for word_j_span, relation_kind in word_relations.items():
                        for token_j_idx, token_span in enumerate(token_mappings):
                            if max(0, min(token_span[1], word_j_span[1]) - max(token_span[0], word_j_span[0])) > 0: # check for word/token overlaps
                                for token_i_idx in word_i_token_ids:
                                    try:
                                        aux_input_relation_kinds[batch_idx, token_i_idx, token_j_idx] = \
                                            self.relational_kind_to_index[relation_kind]

                                    except IndexError:
                                        raise IndexError(f"Could not find relation kind '{relation_kind}'")

        # remove previous relational types
        if self.input_relation_kinds:
            self.input_relation_kinds[0].cpu().detach()
            self.input_relation_kinds.pop()
            torch.cuda.empty_cache()

        # add to global variable
        self.input_relation_kinds.append(
            torch.from_numpy(aux_input_relation_kinds).to(tokenizer_outputs['input_ids'].device)
        )

        return self.input_relation_kinds[0]

    def _change_this_module(self, module_name: str, module: nn.Module, model_name: str) -> bool:

        if model_name.startswith('t5'):
            return module_name.startswith('encoder') and isinstance(module, T5Attention)

        elif model_name.startswith('tapas'):
            return module_name.startswith('encoder') and isinstance(module, BartAttention)

        else:
            raise NotImplementedError(f"Could not find implementation for the model: '{model_name}'")

    def _change_attention_layer(self, attention_layer: nn.Module, num_relation_kinds: int, use_same_relation_kv_emb: bool = True) -> None:
        if type(attention_layer) == T5Attention:
            attention_layer.forward = MethodType(T5RelationalAttention.forward, attention_layer)

        elif type(attention_layer) == BartAttention:
            attention_layer.forward = MethodType(BartRelationalAttention.forward, attention_layer)

        else:
            raise NotImplementedError(f"Could not find implementation for the module: '{attention_layer}'")

        attention_layer.num_relation_kinds = num_relation_kinds
        attention_layer.relation_k_emb = nn.Embedding(num_relation_kinds + 1, attention_layer.inner_dim // attention_layer.n_heads, padding_idx=0)
        if use_same_relation_kv_emb:
            attention_layer.relation_v_emb = attention_layer.relation_k_emb
        else:
            attention_layer.relation_v_emb = nn.Embedding(num_relation_kinds + 1, attention_layer.inner_dim // attention_layer.n_heads, padding_idx=0)
        attention_layer.input_relation_kinds = self.input_relation_kinds # will hold (batch, seq_length, seq_length, num_relation_kinds)
