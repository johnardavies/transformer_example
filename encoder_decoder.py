import torch
import torch.nn.functional as Fun
from torch import nn
import math

import network_components as nc

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