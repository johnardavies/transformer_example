import torch
import torch.nn.functional as Fun
from torch import nn
from config import *


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim_embedding % config.n_head == 0
        # key, query, value projections for all heads, but in a batc
        # nn.Linear applies a linear y=Ax +b transformation to the input
        # the input dimension is the first argument
        # the output dimension is the second argument
        # the last argument is the b (bias) term
        self.c_attn = nn.Linear(
            config.dim_embedding, 3 * config.dim_embedding, bias=config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.dim_embedding, config.dim_embedding, bias=config.bias
        )
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dim_embedding = config.dim_embedding
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )

    def forward(self, x):
        # Split the input tensor dimension down into the batch size B , the sequence length T and the emnedding dimensionality G
        (
            B,
            T,
            C,
        ) = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (dim_embedding)
        # calculate query, key, values
        # by splitting the output of the attention layer into tensors of dimensio dim_embedding on the 2nd dimension
        q, k, v = self.c_attn(x).split(self.dim_embedding, dim=2)
        # split k down by batch_size, sequence_length, number_heads, dimension_embedding/number_heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # self.training is set to true when model.train() is initiated
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )

        else:
            # manual implementation of attention
            # Calculate the inner product of the q and k vectors and normalise by square root of length of key vectors
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Apply the softmax layer so that everything sums to 1
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # Multiply the attention results by the value vectors
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            # Change the shape of the tensor back to B, T, C removing the heads
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        #  Creates the text embedding and the position embedding
        # The embedding automatically does the one hot encoding so this
        # does not need to be created directly
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.dim_embedding)
        self.wtp = nn.Embedding(config.block_size, config.dim_embedding)

    def forward(self, x):
        # Applies the word embedding and then adds it to the position embedding
        x = self.wte(x)
        # The position embedding is applied over a tensor that ranges 0 to 20
        # The embedding is torch.Size([20, 200]) and is broadcast onto the larger tensor
        position_ids = (
            torch.arange(self.config.block_size).unsqueeze(0).repeat(x.size(0), 1)
        )
        position_embeddings = self.wtp(position_ids)
        x = x + position_embeddings

        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return Fun.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Processing_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.denselayer1 = nn.Linear(config.dim_embedding, config.dim_inner_layer)
        self.denselayer2 = nn.Linear(config.dim_inner_layer, config.dim_embedding)
        self.layernorm1 = LayerNorm(config.dim_embedding, bias=config.bias)
        self.layernorm2 = LayerNorm(config.dim_embedding, bias=config.bias)

    def forward(self, x):
        # A layer norm and then two dense layers\n",
        x_in = self.layernorm1(x)
        x = self.denselayer1(x_in)
        x = self.denselayer2(x) + x_in
        x = self.layernorm2(x)
        return x


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim_embedding % config.n_head == 0
        # key, query, value projections for all heads, but in a batc
        # nn.Linear applies a linear y=Ax +b transformation to the input
        # the input dimension is the first argument
        # the output dimension is the second argument
        # the last argument is the b (bias) term
        self.c_attn = nn.Linear(
            config.dim_embedding, 3 * config.dim_embedding, bias=config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.dim_embedding, config.dim_embedding, bias=config.bias
        )
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dim_embedding = config.dim_embedding
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        # Split the input tensor dimension down into the batch size B , the sequence length T and the embedding dimensionality C
        (
            B,
            T,
            C,
        ) = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (dim_embedding)
        # calculate query, key, values
        # by splitting the output of the attention layer into tensors of dimensio dim_embedding on the 2nd dimension
        q, k, v = self.c_attn(x).split(self.dim_embedding, dim=2)
        # split k down by batch_size, sequence_length, number_heads, dimension_embedding/number_heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            # Calculate the inner product of the q and k vectors and normalise by square root of length of key vectors
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Calculate the masked attention
            att = att.masked_fill(attn_mask == 0, float("-inf"))
            # Apply the softmax layer so that everything sums to 1
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # Multiply the attention results by the value vectors
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            # Change the shape of the tensor back to B, T, C removing the heads

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


class EncoderDecoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim_embedding % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn_en = nn.Linear(
            config.dim_embedding, 3 * config.dim_embedding, bias=config.bias
        )

        self.c_attn = nn.Linear(
            config.dim_embedding, 3 * config.dim_embedding, bias=config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.dim_embedding, config.dim_embedding, bias=config.bias
        )
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dim_embedding = config.dim_embedding

        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x, e):

        # Generating the query and key vectors from the vectors from the encoder
        (
            B,
            T,
            C,
        ) = (
            e.size()
        )  # batch size, sequence length, embedding dimensionality (dim_embedding)
        # calculate query, key, values
        # by splitting the output of the attention layer into tensors of dimensio dim_embedding on the 2nd dimension
        _, k, v = self.c_attn_en(e).split(self.dim_embedding, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Generating the value vector from  the decoder input
        (
            B,
            T,
            C,
        ) = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (dim_embedding)

        # calculate values for all heads in batch and move head forward to be the batch dim
        q, _, _ = self.c_attn(x).split(self.dim_embedding, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


# The encoder class


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create embeddings for the encoder
        self.encoder_embed = Embedding(config)

        # Create an attention mechanism for the encoder
        self.attention_encoder = MultiHeadAttention(config)

        # Set up a processing layer
        self.encoder_processing_layer = Processing_layer(config)

    def forward(self, x):
        # Encode the language we want to translate from
        x = self.encoder_embed(x)

        # Apply the attention mechanism and add the input
        x = self.attention_encoder(x) + x

        # apply layer norm, two dense layers and a layer norm again
        x = self.encoder_processing_layer(x)

        return x


# The decoder class


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create an embedding layer for the decoder
        self.decoder_embed = Embedding(config)

        # Create an attention mechanism for the decoder
        self.attention_decoder = MaskedMultiHeadAttention(config)

        # Create a layernorm layer
        self.layernorm = LayerNorm(config.dim_embedding, bias=config.bias)

        # Create the encoder decoder attention
        self.decoder_attn = EncoderDecoderAttention(config)

        # Set up a processing layer for the decoder
        self.decoder_processing_layer = Processing_layer(config)

        # The final layer which maps the models embedding dimension back to the vocab size
        self.final_layer = nn.Linear(config.dim_embedding, config.vocab_size)

    def forward(self, x, y):

        # Encode the language we want to translate into
        y = self.decoder_embed(y)

        # Apply the attention mechanism and add the input
        y = self.attention_decoder(y) + y

        # Apply layer norm
        y = self.layernorm(y)

        # Take the output from the encoder and last layer of decoder and calculate attention again then add the input
        y = self.decoder_attn(y, x) + y

        # apply layer norm, two dense layers and a layer norm again
        y = self.decoder_processing_layer(y)

        # Map the embedding dimension back to the vocabularly size
        y = self.final_layer(y)

        return y


# The Transformers class


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Create both the encoder and the decoder layers
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x, y):
        # Encode the language we want to translate from
        encoder_out = self.encoder(x)

        # Take the output from the encoder and translated text and pass to the decoder
        y = self.decoder(encoder_out, y)
        return y
