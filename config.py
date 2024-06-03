from dataclasses import dataclass


@dataclass
class TransformerConfig:

    """With a dataclass we don't need to write an init function, just specify class attributes and their types"""

    block_size: int = 20 
    batch_size: int = 30 
    epoch: int = 30
    vocab_size: int = 15000  
    dim_embedding: int = 600
    dim_inner_layer: int =  2000
    n_head: int = 6
    dropout: float = 0.2
    bias: bool = False
   
config = TransformerConfig()
