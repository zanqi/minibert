from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # this dropout is applied to normalized attention scores following the original implementation of transformer
        # although it is a bit unusual, we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # next, we need to produce multiple heads for the proj
        # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
        proj = proj.view(
            bs, seq_len, self.num_attention_heads, self.attention_head_size
        )
        # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
        # attention scores are calculated by multiply query and key
        # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
        # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
        # before normalizing the scores, use the attention mask to mask out the padding token scores
        # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number
        # its shape: [bs, 1, 1, seq_len], a row vector that contains -inf for padding tokens and 0 for non-padding tokens.
        # it can tell that for each sequence in the batch, which tokens (padding) to ignore in the attention calculation

        # normalize the scores
        # multiply the attention scores to the value and get back V'
        # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]

        ### TODO
        # key, query, value's shape: [bs, num_attention_heads, seq_len, attention_head_size]
        sqrt_dk = key.shape[-1] ** 0.5
        S = query @ key.transpose(-1, -2)  # [bs, num_attention_heads, seq_len, seq_len]
        attention_mask = attention_mask[:, None, None, :] # (bs, 1, 1, seq_len)
        S = S.masked_fill(attention_mask == 0, float("-inf"))
        attention_probs = self.dropout(F.softmax(S / sqrt_dk, dim=-1))
        multihead_result = (
            attention_probs @ value
        )  # [bs, num_attention_heads, seq_len, attention_head_size]
        return multihead_result.transpose(1, 2).reshape(
            key.shape[0], -1, self.all_head_size
        )  # [bs, seq_len, hidden_size]

    def forward(self, xAndMask: Tuple[torch.Tensor, torch.Tensor]):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
        # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
        hidden_states, attention_mask = xAndMask
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        # calculate the multi-head attention
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # multi-head attention
        self.self_attention = BertSelfAttention(config)
        # add-norm
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attn = nn.Sequential(
            self.self_attention, self.dropout, self.attention_dense
        )

        # feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # another add-norm
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.gelu = nn.GELU()
        self.mlp = nn.Sequential(
            self.interm_dense, self.gelu, self.dropout, self.out_dense
        )

    # def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    #     """
    #     this function is applied after the multi-head attention layer or the feed forward layer
    #     input: the input of the previous layer
    #     output: the output of the previous layer
    #     dense_layer: used to transform the output
    #     dropout: the dropout to be applied
    #     ln_layer: the layer norm to be applied
    #     """
    #     # Hint: Remember that BERT applies to the output of each sub-layer, before it is added to the sub-layer input and normalized
    #     ### TODO
    #     # output: [bs, seq_len, hidden_state]
    #     # input: [bs, seq_len, hidden_state = 768]
    #     return ln_layer(dense_layer(dropout(output)) + input)

    def forward(self, x, attention_mask):
        """
        x: hidden_states, either from the embedding layer (first bert layer) or from the previous bert layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf
        each block consists of
        1. a multi-head attention layer (BertSelfAttention)
        2. a add-norm that takes the input and output of the multi-head attention layer
        3. a feed forward layer
        4. a add-norm that takes the input and output of the feed forward layer
        """
        ### TODO
        x = self.attention_layer_norm(self.attn((x, attention_mask)) + x)
        x = self.out_layer_norm(self.mlp(x) + x)
        return x

        # handout backup
        # self_attention = self.add_norm(
        #     hidden_states,
        #     self_attention,
        #     self.attention_dense,
        #     self.attention_dropout,
        #     self.attention_layer_norm,
        # )

        # return self.add_norm(
        #     self_attention,
        #     interm_af,
        #     self.out_dense,
        #     self.out_dropout,
        #     self.out_layer_norm,
        # )


class BertModel(BertPreTrainedModel):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # embedding
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.tk_type_embedding = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.embed_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # for [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()
        self.cls_head = nn.Linear(config.hidden_size, 1)

        # for token predictions
        self.mlm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_af = nn.Tanh()
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlm_head.weight = self.word_embedding.weight

        self.init_weights()

    def embed(self, input_ids, input_type):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Get word embedding from self.word_embedding into input_embeds.
        inputs_embeds = None
        ### TODO
        inputs_embeds = self.word_embedding(input_ids)

        # Get position index and position embedding from self.pos_embedding into pos_embeds.
        pos_ids = self.position_ids[:, :seq_length]

        pos_embeds = None
        ### TODO
        pos_embeds = self.pos_embedding(pos_ids)

        # Get token type ids, since we are not consider token type, just a placeholder.
        # tk_type_ids = torch.zeros(
        #     input_shape, dtype=torch.long, device=input_ids.device
        # )
        tk_type_embeds = self.tk_type_embedding(input_type)

        # Add three embeddings together; then apply embed_layer_norm and dropout and return.
        ### TODO
        return self.embed_dropout(
            self.embed_layer_norm(inputs_embeds + pos_embeds + tk_type_embeds)
        )

    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # get the extended attention mask for self attention
        # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
        # non-padding tokens with 0 and padding tokens with a large negative number
        # extended_attention_mask: torch.Tensor = get_extended_attention_mask(
        #     attention_mask, self.dtype
        # )

        # pass the hidden states through the encoder layers
        for i, layer_module in enumerate(self.bert_layers):
            # feed the encoding from the last bert_layer to the next
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states

    def forward2(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_ids=input_ids)

        # feed to a transformer (a stack of BertLayers)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}

    def forward(self, input_ids, input_type, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_ids, input_type)

        # feed to a transformer (a stack of BertLayers), shape [batch_size, seq_len, hidden_size]
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)
        all_tk = self.mlm_dense(sequence_output)
        all_tk = self.mlm_af(all_tk)

        cls_logits = self.cls_head(first_tk)
        tk_logits = self.mlm_head(all_tk)

        return {
            "last_hidden_state": sequence_output,
            "pooler_output": first_tk,
            "cls_logits": cls_logits,
            "token_logits": tk_logits,
        }
