import torch
import torch.nn.functional as Fun
from torch import nn
from config import TransformerConfig
import math

# Initialize configuration
config = TransformerConfig()


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return Fun.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Creates the text embedding and the position embedding
        # nn.Embedding automatically does the one hot encoding so this
        # does not need to be created directly
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.dim_embedding)
        self.wtp = nn.Embedding(config.block_size, config.dim_embedding)

    def forward(self, x):
        # Generates the word embedding from the text
        x = self.wte(x)
        # Generates the position embedding is applied over a tensor representing word position 
        position_ids = (
            torch.arange(self.config.block_size).unsqueeze(0).repeat(x.size(0), 1)
        )
        position_embeddings = self.wtp(position_ids)
        # Add the two embeddings
        x = x + position_embeddings

        return x


class Processing_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.denselayer1 = nn.Linear(config.dim_embedding, config.dim_inner_layer)
        self.denselayer2 = nn.Linear(config.dim_inner_layer, config.dim_embedding)
        self.layernorm1 = LayerNorm(config.dim_embedding, bias=config.bias)
        self.layernorm2 = LayerNorm(config.dim_embedding, bias=config.bias)

    def forward(self, x):
        # A layer norm and then two dense layers with a skip connection and then layer norm again
        x_in = self.layernorm1(x)
        x = self.denselayer1(x_in)
        x = self.denselayer2(x) + x_in
        x = self.layernorm2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Checks that the dimension of the embedding vector can be divided by the number of heads
        assert config.dim_embedding % config.n_head == 0

        # set embedding and head sizes
        self.n_head = config.n_head
        self.dim_embedding = config.dim_embedding

        # nn.Linear applies a linear y=Ax+b transformation to the input
        # the input dimension is the first argument
        # the output dimension is the second argument
        # the last argument is the b (bias) term

        # Sets up a layer that increases the dimensionality of the embedding 3x to calculate the query, key and value vectors
        self.c_attn = nn.Linear(
            config.dim_embedding, 3 * config.dim_embedding, bias=config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.dim_embedding, config.dim_embedding, bias=config.bias
        )

        # regularisation
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Uses a faster implementation of attention if scaled_dot_product_attention available in module torch.nn.functional
        # (which it is from PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )

    def forward(self, x):
        # Get the values of the batch size, block size and embedding dimensionality 
        (
            B,
            T,
            C,
        ) = (
            x.size()
        )  
        # calculate query, key, values vectors from the input embedding vectors
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

            # Apply dropout
            att = self.attn_dropout(att)

            # Multiply the attention results by the value vectors
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Change the shape of the tensor back to B, T, C removing the heads
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
      
        # output projection and droput
        y = self.resid_dropout(self.c_proj(y))
       
        return y


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Checks that the dimension of the embedding vector can be divided by the number of heads
        assert config.dim_embedding % config.n_head == 0
        
        # set embedding and head size
        self.n_head = config.n_head
        self.dim_embedding = config.dim_embedding

        self.c_attn = nn.Linear(
            config.dim_embedding, 3 * config.dim_embedding, bias=config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.dim_embedding, config.dim_embedding, bias=config.bias
        )

        # regularisation
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Uses a faster implementation of attention if scaled_dot_product_attention available in module torch.nn.functional
        # (which it is from PyTorch >= 2.0)
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
        # Get the values of the batch size, block size and embedding dimensionality 
        (
            B,
            T,
            C,
        ) = (
            x.size()
        )  

        # calculate query, key, values vectors from the input embedding vectors
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
            att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

            # Apply the softmax layer so that everything sums to 1
            att = F.softmax(att, dim=-1)
            
            # Apply dropout
            att = self.attn_dropout(att)

            # Multiply the attention results by the value vectors
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        # Change the shape of the tensor back to B, T, C removing the heads
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection and dropout
        y = self.resid_dropout(self.c_proj(y))

        return y


class EncoderDecoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Checks that the dimension of the embedding vector can be divided by the number of heads
        assert config.dim_embedding % config.n_head == 0

        # set embedding and head sizes
        self.n_head = config.n_head
        self.dim_embedding = config.dim_embedding
 
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
        self.dropout = config.dropout

        # Uses a faster implementation of attention if scaled_dot_product_attention available in module torch.nn.functional
        # (which it is from PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
     
    def forward(self, x, e):

        # Get the values of the batch size, block size and embedding dimensionality 
        (
            B,
            T,
            C,
        ) = (
            e.size()
        )  

        # calculate the key and value vectors from the output of the encoder
        _, k, v = self.c_attn_en(e).split(self.dim_embedding, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
       
        # calculate the query vectors from the output of the previous decoder layers
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

        # output projection and dropout
        y = self.resid_dropout(self.c_proj(y))

        return y

# The Encoder class

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create an attention mechanism layer for the encoder
        self.attention_encoder = MultiHeadAttention(config)

        # Set up a processing layer
        self.encoder_processing_layer = ProcessingLayer(config)

    def forward(self, x):

        # Apply the attention mechanism and add the input
        x = self.attention_encoder(x) + x

        # apply layer norm, two dense layers and a layer norm again
        x = self.encoder_processing_layer(x)

        return x

# The Decoder class

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create an attention mechanism layer for the decoder
        self.attention_decoder = MaskedMultiHeadAttention(config)

        # Create a layernorm layer
        self.layernorm = LayerNorm(config.dim_embedding, bias=config.bias)

        # Create the encoder decoder attention layer
        self.decoder_attn = EncoderDecoderAttention(config)

        # Set up a processing layer for the decoder
        self.decoder_processing_layer = ProcessingLayer(config)


    def forward(self, x, y):

        # Apply the attention mechanism and add the input
        y = self.attention_decoder(y) + y
        
        #  # Apply layer normalisation
        y = self.layernorm(y)

        # Take the output from the encoder and last layer of decoder and calculate attention again then add the input
        y = self.decoder_attn(y, x) + y

        # apply layer norm, two dense layers and a layer norm again
        y = self.decoder_processing_layer(y)

        return y

# The Transformers class

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Create an embedding layer for the encoder 
        self.encoder_embed = Embedding(config)

        # Create an embedding layer for the decoder 
        self.decoder_embed = Embedding(config)

        # Create both the encoder and the decoder layers
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Create the final layer which maps the model's embedding dimension back to the vocab size
        self.final_layer = nn.Linear(config.dim_embedding, config.vocab_size)

    def forward(self, x, y):

        # Embed the text in the language we want to translate from
        x = self.encoder_embed(x)

        # Embed the text in the language we want to translate into
        y = self.decoder_embed(y)

        # Pass the language we want to translate from into the encoder
        encoder_out = self.encoder(x)

        # Take the output from the encoder and translated text and pass to the decoder
        y = self.decoder(encoder_out, y)

        # Map the embedding dimension back to the vocabulary size
        y = self.final_layer(y)

        return y

