import torch
import torch.nn.functional as Fun
from torch import nn
import math

import network_components as nc
import encoder_decoder as ed



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
        self.encoder = ed.Encoder(config)
        self.decoder = ed.Decoder(config)

        self.encoder2 = ed.Encoder(config)
        self.decoder2 = ed.Decoder(config)

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
