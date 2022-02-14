from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import torch.nn as nn
import torch


class GPT2RelationalAttention(GPT2Attention):
    def __init__(self, *args, num_relation_kinds: int, use_same_relation_kv_emb: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_relation_kinds = num_relation_kinds
        self.relation_k_emb = nn.Embedding(num_relation_kinds + 1, self.head_dim, padding_idx=0)
        if use_same_relation_kv_emb:
            self.relation_v_emb = self.relation_k_emb
        else:
            self.relation_v_emb = nn.Embedding(num_relation_kinds + 1, self.head_dim, padding_idx=0)
        self.input_relation_kinds = [] # will hold (batch, seq_length, seq_length, num_relation_kinds)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        batch_size, seq_length = hidden_states.shape[:2]

        assert len(self.input_relation_kinds) == 1
        input_relation_kinds = self.input_relation_kinds[0]
        assert input_relation_kinds.shape == (batch_size, seq_length, seq_length)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):

        input_relation_kinds = self.input_relation_kinds[0]
        relation_k_embeds = self.relation_k_emb(input_relation_kinds)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # q_t is [batch, seq_length, n_heads, dim_per_head]
        q_t = query.permute(0, 2, 1, 3)

        # r_t is [batch, seq_length, dim_per_head, seq_length]
        r_t = relation_k_embeds.transpose(-2, -1)

        q_tr_t_matmul = torch.matmul(q_t, r_t) # [batch, seq_length, n_heads, seq_length]
        q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3) # [batch, n_heads, seq_length, seq_length]

        # Add to scores
        attn_weights += q_tr_tmatmul_t

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        relation_v_embeds = self.relation_v_emb(input_relation_kinds)

        # w_t is [batch, seq_length, n_heads, seq_length]
        w_t = attn_weights.permute(0, 2, 1, 3)

        # [batch, seq_length, n_heads, seq_length]
        w_tr_matmul = torch.matmul(w_t, relation_v_embeds)

        attn_output += w_tr_matmul.permute(0, 2, 1, 3)

        return attn_output, attn_weights