"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.contextual_transformer import Cross_Transformer
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder

__all__ = ["EncoderBase", "Cross_Transformer", "RNNEncoder", "CNNEncoder",
           "MeanEncoder"]
