import torch
import torch.nn.functional as Fun
from torch import nn
import math

from config import TransformerConfig
import network_components as nc

# Initialize configuration
config = TransformerConfig()

# The Encoder class


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create an attention mechanism layer for the encoder
        self.attention_encoder = nc.MultiHeadAttention(config)

        # Set up a processing layer
        self.encoder_processing_layer = nc.ProcessingLayer(config)

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
        self.masked_attention = nc.MaskedMultiHeadAttention(config)

        # Create a layernorm layer
        self.layernorm = nc.LayerNorm(config.dim_embedding, bias=config.bias)

        # Create the encoder decoder attention layer
        self.encoder_decoder_attn = nc.EncoderDecoderAttention(config)

        # Set up a processing layer for the decoder
        self.decoder_processing_layer = nc.ProcessingLayer(config)

    def forward(self, x, y):

        # Apply the masked attention mechanism and add the input
        y = self.masked_attention(y) + y

        #  # Apply layer normalisation
        y = self.layernorm(y)

        # Take the output from the encoder and last layer of decoder and calculate attention again then add the input
        y = self.encoder_decoder_attn(y, x) + y

        # apply layer norm, two dense layers and a layer norm again
        y = self.decoder_processing_layer(y)

        return y


# The Transformer class


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # The final layer which maps the model's embedding dimension batch to the vocab size
        self.final_layer = nn.Linear(config.dim_embedding, config.vocab_size)

        # Create embeddings for both the encoder and the decoder
        self.encoder_embed = nc.Embedding(config)
        self.decoder_embed = nc.Embedding(config)

        # Create the 2 layers of encoders and decoders
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.encoder2 = Encoder(config)
        self.decoder2 = Decoder(config)

    def forward(self, x, y):

        # Embed the text in the language we want to translate from
        embedded_ger = self.encoder_embed(x)

        # Embed the text in the language we want to translate into
        embedded_en = self.decoder_embed(x)

        # Pass the language we want to translate from through two encoding layers
        encoder_out = self.encoder(embedded_ger)
        encoder_out_two = self.encoder2(encoder_out)

        # Pass the output from the last layer of the encoder into two decoding layers with the embedded language we want to translate into
        decoder_out = self.decoder(encoder_out_two, embedded_en)
        y = self.decoder2(encoder_out_two, decoder_out)

        y = self.final_layer(y)

        return y
