__version__ = '0.0.0'

from transformers import AutoTokenizer, AutoModel, BertPreTrainedModel, BartPretrainedModel, T5PreTrainedModel, \
     PreTrainedTokenizer, BatchEncoding, GPT2PreTrainedModel, PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from typing import Dict, Optional, List, Tuple, Type
from types import MethodType
import torch.nn as nn
from ratransformers.t5 import T5RelationalAttention, T5Attention
from ratransformers.bart import BartRelationalAttention, BartAttention
from ratransformers.bert import BertRelationalSelfAttention, BertSelfAttention
from ratransformers.roberta import RobertaRelationalSelfAttention, RobertaSelfAttention
from ratransformers.gpt2 import GPT2RelationalAttention, GPT2Attention
import torch
import functools
import numpy as np
import os


class RATransformer:

    def __init__(self, pretrained_model_name_or_path: str, relation_kinds: List[str],
                 tokenizer_cls: Type[PreTrainedTokenizer] = AutoTokenizer,
                 model_cls: Type[PreTrainedModel] = AutoModel,
                 pretrained_tokenizer_name_or_path: Optional[str] = None):
        """
        Returns an initialized and ready to test/train RATransformer
        Args:
            pretrained_model_name_or_path: model name or path to pass directly to Huggingface's `model_cls` class
            relation_kinds: list with all the possible relation kinds that can exist within the input
            tokenizer_cls: pass your own AutoTokenizer class to initialize the tokenizer
            model_cls: pass your own AutoModel class to initialize the model
            pretrained_tokenizer_name_or_path: Optional. Tokenizer name or path to pass directly
                to Huggingface's `tokenizer_cls` class. By default, will be equal to pretrained_model_name_or_path
        """

        pretrained_tokenizer_name_or_path = pretrained_tokenizer_name_or_path or pretrained_model_name_or_path
        self.tokenizer = tokenizer_cls.from_pretrained(pretrained_model_name_or_path=pretrained_tokenizer_name_or_path)

        from transformers.modeling_utils import logger
        has_pretrained_rat_model = os.path.isfile(f'{pretrained_model_name_or_path}/pytorch_model.bin')
        logger.disabled = has_pretrained_rat_model # disable logger, if hsa pretrained rat model
        self.model = model_cls.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        logger.disabled = False # enable logger

        self.relational_kind_to_index = {t: i + 1 for i, t in enumerate(relation_kinds)}
        self.input_relation_kinds: List[torch.Tensor] = [] # will be used to pass variable by reference to attention layers

        # change attention layers with relational ones
        for module_name, module in self.model.named_modules():
            self._change_this_module(module_name=module_name, module=module, num_relation_kinds=len(relation_kinds))

        # reload model weights if they exist
        if has_pretrained_rat_model:
            state_dict = torch.load(f'{pretrained_model_name_or_path}/pytorch_model.bin', map_location="cpu")
            self.model, _, _, _ = self.model._load_state_dict_into_model(
                self.model, state_dict, pretrained_model_name_or_path, _fast_init=True
            )

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
                if 'offset_mapping' in kwargs:
                    del kwargs['offset_mapping']
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

    def _change_this_module(self, module_name: str, module: nn.Module, num_relation_kinds: int, use_same_relation_kv_emb: bool = True) -> None:

        relational_embedding_dim = None
        if isinstance(self.model, T5PreTrainedModel):
            if 'encoder' in module_name and 'decoder' not in module_name and isinstance(module, T5Attention):
                module.forward = MethodType(T5RelationalAttention.forward, module)
                relational_embedding_dim = module.inner_dim // module.n_heads

        elif isinstance(self.model, BertPreTrainedModel):
            if 'encoder' in module_name and 'decoder' not in module_name and isinstance(module, BertSelfAttention):
                module.forward = MethodType(BertRelationalSelfAttention.forward, module)
                relational_embedding_dim = module.attention_head_size

        elif isinstance(self.model, BartPretrainedModel):
            if 'encoder' in module_name and 'decoder' not in module_name and isinstance(module, BartAttention):
                module.forward = MethodType(BartRelationalAttention.forward, module)
                relational_embedding_dim = module.head_dim

        elif isinstance(self.model, RobertaPreTrainedModel):
            if 'encoder' in module_name and 'decoder' not in module_name and isinstance(module, RobertaSelfAttention):
                module.forward = MethodType(RobertaRelationalSelfAttention.forward, module)
                relational_embedding_dim = module.attention_head_size

        elif isinstance(self.model, GPT2PreTrainedModel):
            if isinstance(module, GPT2Attention):
                module.forward = MethodType(GPT2RelationalAttention.forward, module)
                module._attn = MethodType(GPT2RelationalAttention._attn, module)
                relational_embedding_dim = module.head_dim

        else:
            raise NotImplementedError(f"Could not find implementation for the model type: '{type(self.model)}'. "
                                      f"Feel free to open an issue in GitHub to ask for its addition!")

        if relational_embedding_dim is None:
            return

        module.num_relation_kinds = num_relation_kinds
        module.relation_k_emb = nn.Embedding(num_relation_kinds + 1, relational_embedding_dim, padding_idx=0)
        if use_same_relation_kv_emb:
            module.relation_v_emb = module.relation_k_emb
        else:
            module.relation_v_emb = nn.Embedding(num_relation_kinds + 1, relational_embedding_dim, padding_idx=0)
        module.input_relation_kinds = self.input_relation_kinds # will hold (batch, seq_length, seq_length, num_relation_kinds)
