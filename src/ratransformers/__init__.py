__version__ = '1.1.0'

from transformers import AutoTokenizer, AutoModel, BertPreTrainedModel, BartPretrainedModel, T5PreTrainedModel, \
    PreTrainedTokenizer, BatchEncoding, GPT2PreTrainedModel, PreTrainedModel, LongT5PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from typing import Dict, Optional, List, Tuple, Type
import torch.nn as nn

from ratransformers.longt5 import change_longt5_module_and_get_relational_emb_dim
from ratransformers.t5 import change_t5_module_and_get_relational_emb_dim
import torch
import functools
import numpy as np
import os


def _change_this_module(model: PreTrainedModel, module_name: str, module: nn.Module, num_relation_kinds: int,
                        use_same_relation_kv_emb: bool = True) -> None:
    relational_embedding_dim = None
    if isinstance(model, T5PreTrainedModel):
        relational_embedding_dim = change_t5_module_and_get_relational_emb_dim(module_name=module_name, module=module)

    elif isinstance(model, LongT5PreTrainedModel):
        relational_embedding_dim = change_longt5_module_and_get_relational_emb_dim(module_name=module_name,
                                                                                   module=module)

    elif isinstance(model, BertPreTrainedModel):
        raise NotImplementedError(f"Not implemented for this version, downgrade to ratransformers==0.1.1")

    elif isinstance(model, BartPretrainedModel):
        raise NotImplementedError(f"Not implemented for this version, downgrade to ratransformers==0.1.1")

    elif isinstance(model, RobertaPreTrainedModel):
        raise NotImplementedError(f"Not implemented for this version, downgrade to ratransformers==0.1.1")

    elif isinstance(model, GPT2PreTrainedModel):
        raise NotImplementedError(f"Not implemented for this version, downgrade to ratransformers==0.1.1")

    else:
        raise NotImplementedError(f"Could not find implementation for the model type: '{type(model)}'. "
                                  f"Feel free to open an issue in GitHub to ask for its addition!")

    if relational_embedding_dim is None:
        return

    module.num_relation_kinds = num_relation_kinds
    module.relation_k_emb = nn.Embedding(num_relation_kinds + 1, relational_embedding_dim, padding_idx=0)
    if use_same_relation_kv_emb:
        module.relation_v_emb = module.relation_k_emb
    else:
        module.relation_v_emb = nn.Embedding(num_relation_kinds + 1, relational_embedding_dim, padding_idx=0)


class RATransformer:

    def __init__(self, pretrained_model_name_or_path: str, relation_kinds: List[str],
                 tokenizer_cls: Type[PreTrainedTokenizer] = AutoTokenizer,
                 model_cls: Type[PreTrainedModel] = AutoModel,
                 pretrained_tokenizer_name_or_path: Optional[str] = None, **model_kwargs):
        """
        Returns an initialized and ready to test/train RATransformer
        Args:
            pretrained_model_name_or_path: model name or path to pass directly to Huggingface's `model_cls` class
            relation_kinds: list with all the possible relation kinds that can exist within the input
            tokenizer_cls: pass your own AutoTokenizer class to initialize the tokenizer
            model_cls: pass your own AutoModel class to initialize the model
            pretrained_tokenizer_name_or_path: Optional. Tokenizer name or path to pass directly
                to Huggingface's `tokenizer_cls` class. By default, will be equal to pretrained_model_name_or_path
            model_kwargs: other arguments to be passed to model_cls.from_pretrained method
        """

        pretrained_tokenizer_name_or_path = pretrained_tokenizer_name_or_path or pretrained_model_name_or_path
        self.tokenizer = tokenizer_cls.from_pretrained(pretrained_model_name_or_path=pretrained_tokenizer_name_or_path)

        has_pretrained_rat_model = os.path.isfile(f'{pretrained_model_name_or_path}/pytorch_model.bin')
        if has_pretrained_rat_model:
            if model_cls.__name__.startswith('AutoModel'):
                raise NotImplementedError(
                    "`model_cls` cannot be an AutoModel class when loading a pretrained RATransformer. "
                    "Please use a specific `model_cls` class. "
                    "For example, for T5 with AutoModelForSeq2SeqLM, use `model_cls=T5ForConditionalGeneration`"
                )
            def model_cls_load_pretrained_model_prefix_function(function):
                @functools.wraps(function)
                def run(model, *args, **kwargs):
                    # change attention layers with relational ones, if not done before
                    for module_name, module in model.named_modules():
                        _change_this_module(
                            model=model, module_name=module_name, module=module,
                            num_relation_kinds=len(relation_kinds)
                        )
                    return function(model, *args, **kwargs)
                return run
            model_cls._load_pretrained_model = model_cls_load_pretrained_model_prefix_function(
                model_cls._load_pretrained_model
            )

        self.model = model_cls.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, **model_kwargs
        )

        self.relational_kind_to_index = {t: i + 1 for i, t in enumerate(relation_kinds)}

        if not has_pretrained_rat_model:
            # change attention layers with relational ones, if not done before
            for module_name, module in self.model.named_modules():
                _change_this_module(
                    model=self.model, module_name=module_name, module=module, num_relation_kinds=len(relation_kinds)
                )

        def model_prefix_function(function):
            @functools.wraps(function)
            def run(*args, **kwargs):
                if 'offset_mapping' in kwargs:
                    del kwargs['offset_mapping']
                return function(*args, **kwargs)
            return run

        def tokenizer_suffix_function(function):
            @functools.wraps(function)
            def run(*args, **kwargs):
                if 'return_tensors' not in kwargs or kwargs['return_tensors'] != 'pt':
                    raise Exception("RATransformer's tokenizer expects `return_tensors='pt'`")
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
                        if relation_kind not in self.relational_kind_to_index:
                            raise AttributeError(
                                f"relation of type '{relation_kind}' not found, "
                                f"RATransformer was initialized with these relation types: {list(self.relational_kind_to_index)}"
                            )
                        for token_j_idx, token_span in enumerate(token_mappings):
                            if max(0, min(token_span[1], word_j_span[1]) - max(token_span[0], word_j_span[0])) > 0: # check for word/token overlaps
                                for token_i_idx in word_i_token_ids:
                                    try:
                                        aux_input_relation_kinds[batch_idx, token_i_idx, token_j_idx] = \
                                            self.relational_kind_to_index[relation_kind]

                                    except IndexError:
                                        raise IndexError(f"Could not find relation kind '{relation_kind}'")

        return torch.from_numpy(aux_input_relation_kinds).to(tokenizer_outputs['input_ids'].device)
