import inspect
import warnings
from types import MethodType
from typing import Optional, Tuple, Union, Any

from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqModelOutput, \
    BaseModelOutput, Seq2SeqLMOutput
from transformers.models.longt5.modeling_longt5 import LongT5Attention, LongT5LayerSelfAttention, \
    LongT5LayerCrossAttention, LongT5Block, \
    logger, LongT5Stack, LongT5Model, _CONFIG_FOR_DOC, LONGT5_INPUTS_DOCSTRING, _get_local_attention_mask, \
    LongT5LocalAttention, _split_into_blocks, _concatenate_3_blocks, _pad_to_multiple, LongT5TransientGlobalAttention, \
    _make_global_fixed_block_ids, _create_global_aggregates, LongT5LayerTransientGlobalSelfAttention, \
    LongT5LayerLocalSelfAttention, LongT5PreTrainedModel, LongT5ForConditionalGeneration, LongT5EncoderModel, \
    LONGT5_ENCODER_INPUTS_DOCSTRING
import torch.nn as nn
import torch
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings


def change_longt5_module_and_get_relational_emb_dim(module_name: str, module: nn.Module) -> Optional[int]:
    relational_embedding_dim = None
    if 'encoder' in module_name and 'decoder' not in module_name and isinstance(module, LongT5Attention):
        module.forward = MethodType(RelationalLongT5Attention.forward, module)
        relational_embedding_dim = module.inner_dim // module.n_heads
    if 'encoder' in module_name and 'decoder' not in module_name and isinstance(module, LongT5LocalAttention):
        module.forward = MethodType(RelationalLongT5LocalAttention.forward, module)
        relational_embedding_dim = module.inner_dim // module.n_heads
    if 'encoder' in module_name and 'decoder' not in module_name and isinstance(module, LongT5TransientGlobalAttention):
        module.forward = MethodType(RelationalLongT5TransientGlobalAttention.forward, module)
        relational_embedding_dim = module.inner_dim // module.n_heads
    elif isinstance(module, LongT5Stack):
        module.forward = MethodType(RelationalLongT5Stack.forward, module)
    elif isinstance(module, LongT5ForConditionalGeneration):
        module.forward = MethodType(RelationalLongT5ForConditionalGeneration.forward, module)
    elif isinstance(module, LongT5Model):
        module.forward = MethodType(RelationalLongT5Model.forward, module)
    elif isinstance(module, LongT5EncoderModel):
        module.forward = MethodType(RelationalLongT5EncoderModel.forward, module)
    elif isinstance(module, LongT5Block):
        module.forward = MethodType(RelationalLongT5Block.forward, module)
    elif isinstance(module, LongT5LayerCrossAttention):
        module.forward = MethodType(RelationalLongT5LayerCrossAttention.forward, module)
    elif isinstance(module, LongT5LayerSelfAttention):
        module.forward = MethodType(RelationalLongT5LayerSelfAttention.forward, module)
    elif isinstance(module, LongT5LayerLocalSelfAttention):
        module.forward = MethodType(RelationalLongT5LayerLocalSelfAttention.forward, module)
    elif isinstance(module, LongT5LayerTransientGlobalSelfAttention):
        module.forward = MethodType(RelationalLongT5LayerTransientGlobalSelfAttention.forward, module)
    return relational_embedding_dim


class RelationalLongT5ForConditionalGeneration(LongT5ForConditionalGeneration):
    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            input_relations=None  # will hold (batch, seq_length, seq_length, num_relation_kinds)
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LongT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
        >>> model = LongT5ForConditionalGeneration.from_pretrained(
        ...     "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"
        ... )

        >>> # Let's try a very long input.
        >>> inputs = tokenizer(100 * "studies have shown that owning a dog is good for you ", return_tensors="pt")
        >>> input_ids = inputs.input_ids

        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        abstractthe aim of this article is to provide an overview of the literature on the role of dog
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(
                    """
                    The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
                    `decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
                    If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
                    num_heads)`.
                    """
                    , FutureWarning
                )
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                input_relations=input_relations
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            input_relations=input_relations
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class RelationalLongT5Model(LongT5Model):
    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_relations=None  # will hold (batch, seq_length, seq_length, num_relation_kinds)
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import T5Tokenizer, LongT5Model

        >>> tokenizer = T5Tokenizer.from_pretrained("google/long-t5-local-base")
        >>> model = LongT5Model.from_pretrained("google/long-t5-local-base")

        >>> # Let's try a very long encoder input.
        >>> input_ids = tokenizer(
        ...     100 * "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1

        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(
                    """
                    The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
                    `decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
                    If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
                    num_heads)`.
                    """
                    , FutureWarning
                )
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                input_relations=input_relations
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            input_relations=input_relations
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class RelationalLongT5EncoderModel(LongT5EncoderModel):
    @add_start_docstrings_to_model_forward(LONGT5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_relations=None  # will hold (batch, seq_length, seq_length, num_relation_kinds)
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LongT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
        >>> model = LongT5EncoderModel.from_pretrained("google/long-t5-local-base")
        >>> input_ids = tokenizer(
        ...     100 * "Studies have been shown that owning a dog is good for you ", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            input_relations=input_relations
        )

        return encoder_outputs

class RelationalLongT5Stack(LongT5Stack):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        input_relations=None  # will hold (batch, seq_length, seq_length, num_relation_kinds)
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # We use local attention in encoder self-attention, otherwise standard self & cross attentions are used
        if self.is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape, inputs_embeds.device
            )
        elif self.config.encoder_attention_type == "local":
            extended_attention_mask = _get_local_attention_mask(attention_mask, self.block_len, inputs_embeds.device)
        else:  # we need to use both local attention mask and standard extended mask for transient-global attention
            extended_attention_mask = attention_mask

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs, **kwd):
                        return tuple(module(*inputs, **kwd, use_cache=use_cache, output_attentions=output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    input_relations=input_relations
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    input_relations=input_relations
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class RelationalLongT5Block(LongT5Block):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        input_relations=None  # will hold (batch, seq_length, seq_length, num_relation_kinds)
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            input_relations=input_relations
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                input_relations=input_relations
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class RelationalLongT5LayerSelfAttention(LongT5LayerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        input_relations=None  # will hold (batch, seq_length, seq_length, num_relation_kinds)
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        kwargs = {'input_relations': input_relations} \
            if 'input_relations' in inspect.getfullargspec(self.SelfAttention.forward)[0] else {}
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class RelationalLongT5LayerCrossAttention(LongT5LayerCrossAttention):
    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        input_relations=None  # will hold (batch, seq_length, seq_length, num_relation_kinds)
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        kwargs = {'input_relations': input_relations} \
            if 'input_relations' in inspect.getfullargspec(self.EncDecAttention.forward)[0] else {}
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            **kwargs
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class RelationalLongT5LayerLocalSelfAttention(LongT5LayerLocalSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
        input_relations=None,  # will hold (batch, seq_length, seq_length, num_relation_kinds)
        **kwargs: Any,  # to accept past_key_value and use_cache kwargs
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        kwargs = {'input_relations': input_relations} \
            if 'input_relations' in inspect.getfullargspec(self.LocalSelfAttention.forward)[0] else {}
        attention_output = self.LocalSelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class RelationalLongT5LayerTransientGlobalSelfAttention(LongT5LayerTransientGlobalSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
        input_relations=None,  # will hold (batch, seq_length, seq_length, num_relation_kinds)
        **kwargs: Any,  # to accept past_key_value and use_cache kwargs
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        kwargs = {'input_relations': input_relations} \
            if 'input_relations' in inspect.getfullargspec(self.TransientGlobalSelfAttention.forward)[0] else {}
        attention_output = self.TransientGlobalSelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class RelationalLongT5Attention(LongT5Attention):
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        input_relations=None # will hold (batch, seq_length, seq_length, num_relation_kinds)
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """

        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        assert input_relations is not None
        assert input_relations.shape == (batch_size, seq_length, seq_length)

        # (batch_size, seq_length, seq_length, self.num_relation_kinds, self.inner_dim // num_relation_kinds)
        relation_k_embeds = self.relation_k_emb(input_relations)
        relation_v_embeds = self.relation_v_emb(input_relations)

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                    len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        # q_t is [batch, seq_length, n_heads, dim_per_head]
        q_t = query_states.permute(0, 2, 1, 3)

        # r_t is [batch, seq_length, dim_per_head, seq_length]
        r_t = relation_k_embeds.transpose(-2, -1)

        q_tr_t_matmul = torch.matmul(q_t, r_t) # [batch, seq_length, n_heads, seq_length]
        q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3) # [batch, n_heads, seq_length, seq_length]

        # Add to scores
        scores += q_tr_tmatmul_t

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # [batch, n_heads, seq_length, seq_length]
        wv_matmul = torch.matmul(attn_weights, value_states)

        # w_t is [batch, seq_length, n_heads, seq_length]
        w_t = attn_weights.permute(0, 2, 1, 3)

        # [batch, seq_length, n_heads, seq_length]
        w_tr_matmul = torch.matmul(w_t, relation_v_embeds)

        attn_output = unshape(wv_matmul + w_tr_matmul.permute(0, 2, 1, 3))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class RelationalLongT5LocalAttention(LongT5LocalAttention):
    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
        input_relations=None  # will hold (batch, seq_length, seq_length, num_relation_kinds)
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        assert input_relations is not None
        assert input_relations.shape == (batch_size, seq_length, seq_length)

        # (batch_size, seq_length, seq_length, dim_per_head)
        relation_k_embeds = self.relation_k_emb(input_relations)
        relation_v_embeds = self.relation_v_emb(input_relations)

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)

        def unshape(states):
            """reshape"""
            return states.contiguous().view(batch_size, -1, self.inner_dim)

        # get query/key/value states -> (batch_size, seq_length, n_heads, dim_per_head)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))

        # Split into blocks -> (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        query_states = _split_into_blocks(query_states, self.block_len, dim=1)
        key_states = _split_into_blocks(key_states, self.block_len, dim=1)
        value_states = _split_into_blocks(value_states, self.block_len, dim=1)

        # Split relation_k_embeds, relation_v_embeds into blocks -> (batch_size, num_blocks, block_len, seq_length, dim_per_head)
        relation_k_embeds = _split_into_blocks(relation_k_embeds, self.block_len, dim=1)
        relation_v_embeds = _split_into_blocks(relation_v_embeds, self.block_len, dim=1)

        # Resize relation_k_embeds and relation_v_embeds -> (batch_size, num_blocks, block_len, block_len, dim_per_head)
        # each block can only have relations with another block, not with the full length
        num_blocks = query_states.shape[1]
        def get_new_relation_embeds(relation_embeds):
            new_relation_embeds = []
            for block_i in range(num_blocks):
                new_relation_embeds.append(
                    relation_embeds[:, block_i, :, block_i * self.block_len: (block_i + 1) * self.block_len,
                    :].unsqueeze(1)
                )
                if new_relation_embeds[-1].shape[-2] % self.block_len != 0:
                    # pad tensor to multiple of block_len
                    new_relation_embeds[-1] = _pad_to_multiple(
                        new_relation_embeds[-1], self.block_len, dim=3, pad_value=0
                    )
            return torch.concatenate(new_relation_embeds)
        relation_k_embeds = get_new_relation_embeds(relation_k_embeds)
        relation_v_embeds = get_new_relation_embeds(relation_v_embeds)

        # Concatenate 3 blocks for keys and values -> (batch_size, num_blocks, 3 * block_len, n_heads, dim_per_head)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=2)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=2)

        # Concatenate 3 blocks -> (batch_size, num_blocks, 3 * block_len, block_len, dim_per_head)
        relation_k_embeds = _concatenate_3_blocks(relation_k_embeds, block_dim=1, sequence_dim=2)
        relation_v_embeds = _concatenate_3_blocks(relation_v_embeds, block_dim=1, sequence_dim=2)

        # Compute scores
        scores = torch.einsum(
            "...qhd,...khd->...hqk", query_states, key_states
        )  # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)

        # q_t is (batch_size, num_blocks, 3 * block_len, n_heads, dim_per_head)
        q_t = _concatenate_3_blocks(query_states, block_dim=1, sequence_dim=2)

        # r_t is (batch_size, num_blocks, 3 * block_len, dim_per_head, block_len)
        r_t = relation_k_embeds.transpose(-2, -1)

        # (batch_size, num_blocks, 3 * block_len, n_heads, block_len)
        q_tr_t_matmul = torch.matmul(q_t, r_t)

        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 1, 3, 4, 2)

        # Add to scores
        scores += q_tr_tmatmul_t

        if position_bias is None:
            # position_bias shape: # (1, 1, n_heads, block_len, 3 * block_len)
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, 1, self.n_heads, self.block_len, 3 * self.block_len), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(self.block_len)

            if mask is not None:
                # Replace masked positions with -1e10 (according to the original implementation)
                mask = torch.where(mask > 0, 0.0, -1e10)
                # We need to adjust position bias shape to be sum with mask
                position_bias = position_bias + mask.transpose(1, 2)

        scores += position_bias
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        attn_weights = attn_weights.type(value_states.dtype)
        wv_matmul = torch.einsum("...hqk,...khd->...qhd", attn_weights, value_states)

        # (batch_size, num_blocks, block_len, n_heads, 3 * block_len)
        w_t = attn_weights.permute(0, 1, 3, 2, 4)

        # (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        w_tr_matmul = torch.matmul(w_t, relation_v_embeds.transpose(-3, -2))

        # (batch_size, seq_length, dim_per_head)
        attn_output = unshape(wv_matmul + w_tr_matmul)[:, :seq_length, :]

        # (batch_size, seq_length, d_model)
        attn_output = self.o(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class RelationalLongT5TransientGlobalAttention(LongT5TransientGlobalAttention):
    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
        input_relations=None  # will hold (batch, seq_length, seq_length, num_relation_kinds)
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        assert input_relations is not None
        assert input_relations.shape == (batch_size, seq_length, seq_length)

        # (batch_size, seq_length, seq_length, dim_per_head)
        relation_k_embeds = self.relation_k_emb(input_relations)
        relation_v_embeds = self.relation_v_emb(input_relations)

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)

        def unshape(states):
            """reshape"""
            return states.contiguous().view(batch_size, -1, self.inner_dim)

        # Prepare components for transient-global attention
        # Obtain block_ids and global_segment_ids
        # global_seq_len := seq_len // self.global_block_size
        # shapes: (batch_size, seq_len) & (batch_size, global_seq_len)
        block_ids, global_segment_ids = _make_global_fixed_block_ids(
            mask if mask is not None else torch.ones(hidden_states.shape[:-1]),
            self.global_block_size,
        )
        # Create global inputs
        _global_seq_len = global_segment_ids.shape[-1]
        global_inputs = _create_global_aggregates(hidden_states, block_ids, _global_seq_len)
        global_inputs = self.global_input_layer_norm(global_inputs)

        # get query states -> (batch_size, seq_length, n_heads, dim_per_head)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))
        # Get global/side key/value states  shape: (batch_size, global_seq_len, n_heads, dim_per_head)
        side_key_states = shape(self.k(global_inputs))
        side_value_states = shape(self.v(global_inputs))

        # Split into blocks -> (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        query_states = _split_into_blocks(query_states, self.block_len, dim=1)
        key_states = _split_into_blocks(key_states, self.block_len, dim=1)
        value_states = _split_into_blocks(value_states, self.block_len, dim=1)

        # Split relation_k_embeds, relation_v_embeds into blocks -> (batch_size, num_blocks, block_len, seq_length, dim_per_head)
        relation_k_embeds = _split_into_blocks(relation_k_embeds, self.block_len, dim=1)
        relation_v_embeds = _split_into_blocks(relation_v_embeds, self.block_len, dim=1)

        # Resize relation_k_embeds and relation_v_embeds -> (batch_size, num_blocks, block_len, block_len, dim_per_head)
        # each block can only have relations with another block, not with the full length
        num_blocks = query_states.shape[1]
        def get_new_relation_embeds(relation_embeds):
            new_relation_embeds = []
            for block_i in range(num_blocks):
                new_relation_embeds.append(
                    relation_embeds[:, block_i, :, block_i * self.block_len: (block_i + 1) * self.block_len, :].unsqueeze(1)
                )
                if new_relation_embeds[-1].shape[-2] % self.block_len != 0:
                    # pad tensor to multiple of block_len
                    new_relation_embeds[-1] = _pad_to_multiple(
                        new_relation_embeds[-1], self.block_len, dim=3, pad_value=0
                    )
            return torch.concatenate(new_relation_embeds)
        relation_k_embeds = get_new_relation_embeds(relation_k_embeds)
        relation_v_embeds = get_new_relation_embeds(relation_v_embeds)

        # Concatenate 3 blocks for keys and values -> (batch_size, num_blocks, 3 * block_len, n_heads, dim_per_head)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=2)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=2)

        # Concatenate 3 blocks -> (batch_size, num_blocks, 3 * block_len, block_len, dim_per_head)
        relation_k_embeds = _concatenate_3_blocks(relation_k_embeds, block_dim=1, sequence_dim=2)
        relation_v_embeds = _concatenate_3_blocks(relation_v_embeds, block_dim=1, sequence_dim=2)

        # Tile side inputs across local key/value blocks
        # New shape: (batch_size, num_blocks, global_seq_len, n_heads, dim_per_head)
        reps = [1] * (side_key_states.ndim + 1)
        reps[1] = key_states.shape[1]
        side_key_states = side_key_states.unsqueeze(1).repeat(reps)
        side_value_states = side_value_states.unsqueeze(1).repeat(reps)

        # Concatenate "local" and "side"/"global" key/value states to allow each token to attend global aggregated ones
        # New shape: (batch_size, num_blocks, 3 * block_len + global_seq_len, n_heads, dim_per_head)
        key_states = torch.cat([key_states, side_key_states], dim=2)
        value_states = torch.cat([value_states, side_value_states], dim=2)

        # Add zeros to relation_k_embeds and relation_v_embeds
        # New shape: (batch_size, num_blocks, 3 * block_len + global_seq_len, n_heads, dim_per_head)
        zeros_shape_to_add = list(relation_k_embeds.shape[:2]) + [side_value_states.shape[2]] + list(relation_k_embeds.shape[3:])
        relation_k_embeds = torch.cat([relation_k_embeds, torch.zeros(zeros_shape_to_add)], dim=2)
        relation_v_embeds = torch.cat([relation_v_embeds, torch.zeros(zeros_shape_to_add)], dim=2)

        # Compute scores -> (batch_size, num_block, n_heads, block_len, 3 * block_len + global_seq_len)
        scores = torch.einsum("...qhd,...khd->...hqk", query_states, key_states)

        # q_t is (batch_size, num_blocks, 3 * block_len + global_seq_len, n_heads, dim_per_head)
        q_t = _concatenate_3_blocks(query_states, block_dim=1, sequence_dim=2)
        q_t = torch.cat([q_t, side_value_states], dim=2)

        # r_t is (batch_size, num_blocks, 3 * block_len + global_seq_len, dim_per_head, block_len)
        r_t = relation_k_embeds.transpose(-2, -1)

        # (batch_size, num_blocks, 3 * block_len + global_seq_len, n_heads, block_len)
        q_tr_t_matmul = torch.matmul(q_t, r_t)

        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len + global_seq_len)
        q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 1, 3, 4, 2)

        # Add to scores
        scores += q_tr_tmatmul_t

        if mask is not None:
            # We need to adjust position bias shape to be sum with mask
            local_attention_mask = _get_local_attention_mask(mask, self.block_len, hidden_states.device)
            # Replace masked positions with -10_000 (according to the original implementation)
            local_attention_mask = torch.where(local_attention_mask > 0, 0.0, -1e10)
        else:
            local_attention_mask = None

        if position_bias is None:
            # position_bias shape: # (1, 1, n_heads, block_len, 3 * block_len)
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, 1, self.n_heads, self.block_len, 3 * self.block_len),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(self.block_len)

            if local_attention_mask is not None:
                # (batch_size, 1, n_heads, block_len, 3 * block_len)
                position_bias = position_bias + local_attention_mask.transpose(1, 2)
            position_bias = position_bias.type(scores.dtype)

            # Calculate global/side bias - shape: # (batch_size, num_heads, seq_len, global_seq_len)
            if mask is None:
                mask = torch.ones(batch_size, seq_length)
            # (batch_size, num_heads, seq_len, global_seq_len)
            side_position_bias = self.compute_side_bias(mask, global_segment_ids)
            # (batch_size, num_blocks, num_heads, block_len, global_seq_len)
            side_position_bias = _split_into_blocks(side_position_bias, self.block_len, dim=-2).transpose(1, 2)
            side_position_bias = side_position_bias.type(scores.dtype).to(scores.device)
            # (batch_size, num_blocks, num_heads, block_len, 3 * block_len + global_seq_len)
            position_bias = torch.cat([position_bias, side_position_bias], dim=-1)

        scores += position_bias
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len + global_seq_len)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        attn_weights = attn_weights.type(value_states.dtype)
        wv_matmul = torch.einsum("...hqk,...khd->...qhd", attn_weights, value_states)

        # (batch_size, num_blocks, block_len, n_heads, 3 * block_len + global_seq_len)
        w_t = attn_weights.permute(0, 1, 3, 2, 4)

        # (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        w_tr_matmul = torch.matmul(w_t, relation_v_embeds.transpose(-3, -2))

        # (batch_size, seq_length, dim_per_head)
        attn_output = unshape(wv_matmul + w_tr_matmul)[:, :seq_length, :]

        # (batch_size, seq_length, d_model)
        attn_output = self.o(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

