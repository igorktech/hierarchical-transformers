# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" RoHAT configuration"""
from collections import OrderedDict
from typing import Mapping

from transformers.onnx import OnnxConfig
from transformers.utils import logging
from transformers import PretrainedConfig

logger = logging.get_logger(__name__)

# RoHAT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
#     "kiddothe2b/hierarchical-transformer-base-4096": "https://huggingface.co/kiddothe2b/hierarchical-transformer-base-4096/resolve/main/config.json",
#     "kiddothe2b/adhoc-hierarchical-transformer-base-4096": "https://huggingface.co/kiddothe2b/adhoc-hierarchical-transformer-base-4096/resolve/main/config.json",
# }


class RoHATConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.RoHAT`.
    It is used to instantiate a RoHAT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the RoHAT `kiddothe2b/hierarchical-transformer-base-4096
    <https://huggingface.co/kiddothe2b/hierarchical-transformer-base-4096>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        max_sentences (:obj:`int`, `optional`, defaults to 64):
            The maximum number of sentences that this model might ever be used with.
        max_sentence_size (:obj:`int`, `optional`, defaults to 128):
            The maximum sentence length that this model might ever be used with.
        model_max_length (:obj:`int`, `optional`, defaults to 8192):
            The maximum  sequence length (max_sentences * max_sentence_size) that this model might ever be used with
        encoder_layout (:obj:`Dict`):
            The sentence/document encoder layout.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"silu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"silu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        classifier_dropout (:obj:`float`, `optional`):
            The dropout ratio for the classification head.
    """
    model_type = "rohat"

    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            max_sentences=64,
            max_sentence_size=128,
            model_max_length=8192,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="silu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            encoder_layout=None,
            rotary_value=False,
            use_cache=True,
            classifier_dropout=None,
            auto_map={
                "AutoConfig": "configuration_rohat.RoHATConfig",
                "AutoModel": "modelling_rohat.RoHATModel",
                "AutoModelForMaskedLM": "modelling_rohat.RoHATForMaskedLM",
                "AutoModelForSequenceClassification": "modelling_rohat.RoHATForSequenceClassification",
            },
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_sentences = max_sentences
        self.max_sentence_size = max_sentence_size
        self.model_max_length = model_max_length
        self.encoder_layout = encoder_layout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rotary_value = rotary_value
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.auto_map = auto_map

class RoHATOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )
