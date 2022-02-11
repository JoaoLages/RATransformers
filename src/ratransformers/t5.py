from transformers.models.t5.modeling_t5 import T5Attention
import torch.nn as nn
import torch


class T5RelationalAttention(T5Attention):
    def __init__(self, *args, num_relation_kinds: int, use_same_relation_kv_emb: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_relation_kinds = num_relation_kinds
        self.relation_k_emb = nn.Embedding(num_relation_kinds + 1, self.inner_dim // self.n_heads, padding_idx=0)
        if use_same_relation_kv_emb:
            self.relation_v_emb = self.relation_k_emb
        else:
            self.relation_v_emb = nn.Embedding(num_relation_kinds + 1, self.inner_dim // self.n_heads, padding_idx=0)
        self.input_relation_kinds = [] # will hold (batch, seq_length, seq_length, num_relation_kinds)

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
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """

        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        assert len(self.input_relation_kinds) == 1
        input_relation_kinds = self.input_relation_kinds[0]
        assert input_relation_kinds.shape == (batch_size, seq_length, seq_length)

        # (batch_size, seq_length, seq_length, self.num_relation_kinds, self.inner_dim // num_relation_kinds)
        relation_k_embeds = self.relation_k_emb(input_relation_kinds)
        relation_v_embeds = self.relation_v_emb(input_relation_kinds)

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

        scores += position_bias
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
