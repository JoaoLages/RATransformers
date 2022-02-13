import math

from transformers.models.bert.modeling_bert import BertSelfAttention
import torch.nn as nn
import torch


class BertRelationalSelfAttention(BertSelfAttention):
    def __init__(self, *args, num_relation_kinds: int, use_same_relation_kv_emb: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_relation_kinds = num_relation_kinds
        self.relation_k_emb = nn.Embedding(num_relation_kinds + 1, self.attention_head_size, padding_idx=0)
        if use_same_relation_kv_emb:
            self.relation_v_emb = self.relation_k_emb
        else:
            self.relation_v_emb = nn.Embedding(num_relation_kinds + 1, self.attention_head_size, padding_idx=0)
        self.input_relation_kinds = [] # will hold (batch, seq_length, seq_length, num_relation_kinds)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):

        batch_size, seq_length = hidden_states.shape[:2]

        assert len(self.input_relation_kinds) == 1
        input_relation_kinds = self.input_relation_kinds[0]
        assert input_relation_kinds.shape == (batch_size, seq_length, seq_length)

        # (batch_size, seq_length, seq_length, self.num_relation_kinds, self.inner_dim // num_relation_kinds)
        relation_k_embeds = self.relation_k_emb(input_relation_kinds)
        relation_v_embeds = self.relation_v_emb(input_relation_kinds)

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # q_t is [batch, seq_length, n_heads, dim_per_head]
        q_t = query_layer.permute(0, 2, 1, 3)

        # r_t is [batch, seq_length, dim_per_head, seq_length]
        r_t = relation_k_embeds.transpose(-2, -1)

        q_tr_t_matmul = torch.matmul(q_t, r_t) # [batch, seq_length, n_heads, seq_length]
        q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3) # [batch, n_heads, seq_length, seq_length]

        # Add to scores
        attention_scores += q_tr_tmatmul_t

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        # w_t is [batch, seq_length, n_heads, seq_length]
        w_t = attention_probs.permute(0, 2, 1, 3)

        # [batch, seq_length, n_heads, seq_length]
        w_tr_matmul = torch.matmul(w_t, relation_v_embeds)

        context_layer += w_tr_matmul.permute(0, 2, 1, 3)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
