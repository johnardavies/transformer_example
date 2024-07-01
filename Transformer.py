import torch
import torch.nn.functional as Fun
from torch import nn
import math

import encoder_decoder as ed
import network_components as nc
from config import TransformerConfig


# Initialize configuration
config = TransformerConfig()


# The Transformers class


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Create an embedding layer that embeds the text we want to translate from before it passes to the encoder
        self.encoder_embed = nc.Embedding(config)

        # Create an embedding layer that embeds the text we want to translate into before it passes to the decoder
        self.decoder_embed = nc.Embedding(config)

        # Create both the encoder and the decoder layers
        self.encoder = ed.Encoder(config)
        self.decoder = ed.Decoder(config)

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
