from typing import Optional, Tuple
from transformers.models.bart.modeling_bart import BartAttention
import torch.nn as nn
import torch


class BartRelationalAttention(BartAttention):
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
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        assert len(self.input_relation_kinds) == 1
        input_relation_kinds = self.input_relation_kinds[0]
        assert input_relation_kinds.shape == (bsz, tgt_len, tgt_len)

        # (batch_size, seq_length, seq_length, self.num_relation_kinds, self.inner_dim // num_relation_kinds)
        relation_k_embeds = self.relation_k_emb(input_relation_kinds)
        relation_v_embeds = self.relation_v_emb(input_relation_kinds)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        src_len = key_states.size(2)

        # compute scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        # q_t is [batch, seq_length, n_heads, dim_per_head]
        q_t = query_states.permute(0, 2, 1, 3)

        # r_t is [batch, seq_length, dim_per_head, seq_length]
        r_t = relation_k_embeds.transpose(-2, -1)

        q_tr_t_matmul = torch.matmul(q_t, r_t) # [batch, seq_length, n_heads, seq_length]
        q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3) # [batch, n_heads, seq_length, seq_length]

        # Add to scores
        attn_weights += q_tr_tmatmul_t
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states.view(*proj_shape))

        # w_t is [batch, seq_length, n_heads, seq_length]
        w_t = attn_probs.view(bsz, self.num_heads, tgt_len, src_len).permute(0, 2, 1, 3)

        # [batch, seq_length, n_heads, seq_length]
        w_tr_matmul = torch.matmul(w_t, relation_v_embeds)

        attn_output += w_tr_matmul.permute(0, 2, 1, 3).view(bsz * self.num_heads, tgt_len, self.head_dim)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value