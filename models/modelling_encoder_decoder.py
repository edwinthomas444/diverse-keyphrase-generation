# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
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
""" Classes to support Encoder-Decoder architectures"""

from transformers.utils.generic import ModelOutput
import tempfile
import warnings
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, EncoderDecoderConfig
from losses.loss_utils import hungarian_realignment
from transformers.models.bert_generation.modeling_bert_generation import BertGenerationOnlyLMHead
from models.modeling_bert import BertForMaskedLM, BertLMHeadModel, BertModel, BertLayer
import torch.nn.functional as F
import copy

logger = logging.get_logger(__name__)

# overide seq2seqLMOutput
@dataclass
class Seq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_precision_head_mask: Optional[torch.FloatTensor] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# Custom class for precision head attention
class PrecisionHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # print('\n configs: ', config)
        self.layers = nn.ModuleList([BertLayer(config)
                                     for _ in range(config.precision_attn_layers)])
        # self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        for i, layer_module in enumerate(self.layers):
            # hidden states for next iteration is output from the current
            # iteration
            outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions
            )
            hidden_states = outputs[0]
            # print(f'\n hidden states from layer {i}: ', hidden_states.shape)
        # return final hidden states
        return (hidden_states,)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError(
            "Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError(
            "Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

@dataclass
class UnitGenerationOutput(ModelOutput):
    sequences: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    decoder_hidden_states: torch.FloatTensor = None
    decoder_precision_head_mask: torch.FloatTensor = None


class EncoderDecoderModel(PreTrainedModel):
    r"""
    [`EncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with one
    of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        # print('\nEntered Encoder Decoder Model init function: ')
        if config is None and (encoder is None or decoder is None):
            raise ValueError(
                "Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(
                encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(
                    f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if encoder is None:
            # from transformers import AutoModel

            # encoder = AutoModel.from_config(config.encoder)
            encoder = BertModel(config=config.encoder)
            # print('### Encoder: ', encoder)

        if decoder is None:
            # from transformers import AutoModelForCausalLM

            # decoder = AutoModelForCausalLM.from_config(config.decoder)

            # use the custom decoder definition if EncoderDecoderModel
            # is invoked with .from_pretrained(path of encoder decoder mdoel, saved config)
            decoder = BertLMHeadModel(
                config=config.decoder
            )
            # print('### Decoder: ', decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # encoder outputs might need to be projected to different dimension for decoder
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(
                self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        # precision heads
        if self.decoder.config.precision_heads:
            self.add_precision_heads()

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        # Commented out
        # decoder_signature = set(inspect.signature(self.decoder.forward).parameters.keys())
        # if "encoder_hidden_states" not in decoder_signature:
        #     raise ValueError(
        #         "The selected decoder is not prepared for the encoder hidden states to be passed. Please see the "
        #         "following discussion on GitHub: https://github.com/huggingface/transformers/issues/23350"
        #     )

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def _set_gradient_checkpointing(self, module, value=False):
        # call both encoder and decoder function on gradient checkpointing
        self.encoder._set_gradient_checkpointing(module, value=value)
        self.decoder._set_gradient_checkpointing(module, value=value)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import EncoderDecoderModel

        >>> model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")
        ```"""

        from_tf = kwargs.pop("from_tf", False)
        if from_tf:
            from transformers import TFEncoderDecoderModel

            # a workaround to load from tensorflow checkpoint
            # Using `_tf_model` won't work, because the weight names in the encoder/decoder of `_tf_model` get
            # extended before saving those components. For example, The name of `_tf_model.encoder.vit` is
            # `[top model name]/encoder/vit`, but the name of `tf_model.encoder.vit` is `[top model name]/vit`. The
            # [top model name] is handled (stripped) by the conversion method, and the former case gets extra `encoder`,
            # which should not occur when we want to save the components alone.
            # There was a (very) ugly potential fix, which wasn't integrated to `transformers`: see
            #   https://github.com/huggingface/transformers/pull/13222/commits/dbb3c9de76eee235791d2064094654637c99f36d#r697304245
            #   (the change in `src/transformers/modeling_tf_utils.py`)
            _tf_model = TFEncoderDecoderModel.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs)
            config = _tf_model.config

            # Using `tf_model` instead
            encoder = _tf_model.encoder.__class__(_tf_model.config.encoder)
            decoder = _tf_model.decoder.__class__(_tf_model.config.decoder)
            # Make sure models are built
            encoder(encoder.dummy_inputs)
            decoder(decoder.dummy_inputs)

            # Get the variable correspondence between `_tf_model` and `encoder` and `decoder`
            encoder_variables = {}
            for v in encoder.trainable_variables + encoder.non_trainable_variables:
                encoder_variables["/".join(v.name.split("/")[1:])] = v
            decoder_variables = {}
            for v in decoder.trainable_variables + decoder.non_trainable_variables:
                decoder_variables["/".join(v.name.split("/")[1:])] = v

            _encoder_variables = {}
            for v in _tf_model.encoder.trainable_variables + _tf_model.encoder.non_trainable_variables:
                _encoder_variables["/".join(v.name.split("/")[2:])] = v
            _decoder_variables = {}
            for v in _tf_model.decoder.trainable_variables + _tf_model.decoder.non_trainable_variables:
                _decoder_variables["/".join(v.name.split("/")[2:])] = v

            # assign weight values to `encoder` and `decoder` from `_tf_model`
            for name, v in encoder_variables.items():
                v.assign(_encoder_variables[name])
            for name, v in decoder_variables.items():
                v.assign(_decoder_variables[name])

            tf_model = TFEncoderDecoderModel(encoder=encoder, decoder=decoder)

            # Deal with `enc_to_dec_proj`
            if hasattr(_tf_model, "enc_to_dec_proj"):
                tf_model(tf_model.dummy_inputs)
                tf_model.enc_to_dec_proj.kernel.assign(
                    _tf_model.enc_to_dec_proj.kernel)
                tf_model.enc_to_dec_proj.bias.assign(
                    _tf_model.enc_to_dec_proj.bias)

            with tempfile.TemporaryDirectory() as tmpdirname:
                encoder_dir = os.path.join(tmpdirname, "encoder")
                decoder_dir = os.path.join(tmpdirname, "decoder")
                tf_model.encoder.save_pretrained(encoder_dir)
                tf_model.decoder.save_pretrained(decoder_dir)

                if hasattr(tf_model, "enc_to_dec_proj"):
                    enc_to_dec_proj_weight = torch.transpose(
                        torch.from_numpy(
                            tf_model.enc_to_dec_proj.kernel.numpy()), 1, 0
                    )
                    enc_to_dec_proj_bias = torch.from_numpy(
                        tf_model.enc_to_dec_proj.bias.numpy())

                del _tf_model
                del tf_model
                gc.collect()

                model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                    encoder_dir, decoder_dir, encoder_from_tf=True, decoder_from_tf=True
                )
                # This is only for copying some specific attributes of this particular model.
                model.config = config

                if hasattr(model, "enc_to_dec_proj"):
                    model.enc_to_dec_proj.weight.data = enc_to_dec_proj_weight
                    model.enc_to_dec_proj.bias.data = enc_to_dec_proj_bias

                return model

        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for EncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        # print(kwargs)

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import EncoderDecoderModel

        >>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./bert2bert")
        >>> # load fine-tuned model
        >>> model = EncoderDecoderModel.from_pretrained("./bert2bert")
        ```"""

        kwargs_encoder = {
            argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        kwargs_external = {
            argument[len("external_"):]: value for argument, value in kwargs.items() if argument.startswith("external_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # print('kwargs_decoder_extracted: ', kwargs_decoder)
        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            kwargs_encoder['config'].update(kwargs_external)
            encoder = BertModel.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)
            
            # loading without pretrained weights
            # encoder = BertModel(
            #     **kwargs_encoder
            # )

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            # decoder = AutoModelForCausalLM.from_pretrained(
            #     decoder_pretrained_model_name_or_path, **kwargs_decoder)
            kwargs_decoder['config'].update(kwargs_external)
            # print('\nUpdated Decoder kwargs: ', kwargs_decoder)
            # ignoring size mismatches for the token type embeddings (as pretrained has 2, which is overrided to max_kp_len)
            decoder = BertLMHeadModel.from_pretrained(
                decoder_pretrained_model_name_or_path, **kwargs_decoder, ignore_mismatched_sizes=True
            )

            # loading without pretrained weights
            # decoder = BertLMHeadModel(
            #     **kwargs_decoder
            # )

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config)

    def loss_mask_and_normalize(self, loss, mask):
        mask = mask.type_as(loss)

        # print(loss.shape, mask.shape)
        loss = loss * mask
        denominator = torch.sum(mask) + 1e-5
        return (loss / denominator).sum()

    def add_mlm_head(self):
        self.mlm_head = BertGenerationOnlyLMHead(self.encoder.config)

    # placeholder to add head for precision mask prediction
    def add_precision_heads(self):
        # add additional attention layer (BertLayer)
        self.precision_head_attention = BertLayer(self.decoder.config)
        # self.precision_head_attention = PrecisionHeadAttention(self.decoder.config)
        self.precision_heads = nn.Linear(768, 2)

    def get_seq2set_dec_attn_mask_from_seq2seq(self, attn_mask):
        # print('from get seq2set dec attn mask from seq2set: ', attn_mask.shape)
        mask = []
        kp_len = self.config.max_kp_len
        for i in range(self.config.max_kps):
            # extract diagonal squares as kps are stacked
            mask.append(attn_mask[:, i*kp_len:(i+1) *
                                  kp_len, i*kp_len:(i+1)*kp_len])
        mask = torch.stack(mask, dim=1)  # (bs, max_kps, kp_len, kp_len)
        mask = mask.contiguous().view(-1, kp_len, kp_len)
        # print('Mask seq2set shape: ', mask.shape)
        return mask

    # define a function for generation
    # ToDO: Change for unstacked implementation
    # currently supports stacked implementation
    def unit_generation(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_position_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            encoder_outputs: Optional[torch.LongTensor] = None
    ):
        # print('\n Inside unit generation...')

        max_kps = self.config.max_kps
        max_kp_len = self.config.max_kp_len

        # forward pass through encoder
        with torch.no_grad():
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=None
                )

            encoder_hidden_states = encoder_outputs[0]

            bs, _ = decoder_input_ids.size()

            # input_tokens = decoder_input_ids.new_zeros(bs, max_kps, max_kp_len)
            input_tokens = decoder_input_ids.contiguous().view(bs*max_kps, max_kp_len)
            decoder_position_ids = decoder_position_ids.contiguous().view(bs*max_kps, max_kp_len)

            # preparing encoder hidden states
            enc_hidden_states = encoder_hidden_states
            if cross_attention_mask.size(0) != encoder_hidden_states.size(0):
                cross_attention_mask = cross_attention_mask.repeat_interleave(
                    max_kps, dim=0)
            enc_cross_attention_mask = cross_attention_mask

            # print('cross_attention_mask: ', cross_attention_mask.shape)
            # print('Encoder hidden states shape: ', enc_hidden_states.shape)
            # print('enc cross attention mask shape: ',
            #       enc_cross_attention_mask.shape)

            # print('In Unit Generation: decoder attention mask shape',
            #       decoder_attention_mask.shape)

            dec_attention_mask = self.get_seq2set_dec_attn_mask_from_seq2seq(
                decoder_attention_mask)

            # print('\n from unit generation: ', decoder_position_ids.shape)
            decoder_outputs = self.decoder(
                input_ids=input_tokens,  # bs*max_kps, max_kp_len
                position_ids=decoder_position_ids,
                attention_mask=dec_attention_mask,  # bs*max_kps, max_kp_len, max_kp_len
                encoder_hidden_states=enc_hidden_states,
                encoder_attention_mask=enc_cross_attention_mask,
                past_key_values=None,
                output_attentions=None,
                use_cache=False,
                output_hidden_states=True
            )
            output = decoder_outputs[0]
            decoded_tokens = output.argmax(-1)

            #bs*max_kps, 2
            precision_head_mask = None
            if self.decoder.config.precision_heads:
                precision_head_mask = self.precision_heads(
                    decoder_outputs.hidden_states[-1].mean(dim=1))
        # print('unit generation prob shape: ', output.shape)
        # print('unit generation sequence shape: ', decoded_tokens.shape)
        # input_tokens stores the final greedy search output of the decoder using unit-wise decoding
        return UnitGenerationOutput(
            sequences=decoded_tokens,
            logits=output,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_precision_head_mask=precision_head_mask
        )

    def unit_generation_student_forcing(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_position_ids: Optional[torch.LongTensor] = None,
            decoder_token_type_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            encoder_outputs: Optional[torch.LongTensor] = None,
            cross_unit_attention=False,
            output_precision_head=False
    ):
        # print('\n Inside unit generation...')

        max_kps = self.config.max_kps
        max_kp_len = self.config.max_kp_len

        # forward pass through encoder
        # with torch.no_grad():

        # the below operations are computed with gradients, when precision head is enabled (for precision had train forward pass)
        # and disabled for hungarian loss forward pass (see call function)
        with torch.set_grad_enabled(output_precision_head):
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=None
                )

            encoder_hidden_states = encoder_outputs[0]

            bs, _ = decoder_input_ids.size()

            # input_tokens = decoder_input_ids.new_zeros(bs, max_kps, max_kp_len)
            input_tokens = decoder_input_ids.contiguous().view(bs, max_kps, max_kp_len)
            # print('\n input tokens shape: ', input_tokens.shape)
            # preparing encoder hidden states
            enc_hidden_states = encoder_hidden_states
            # if cross_attention_mask.size(0) != encoder_hidden_states.size(0):
            #     cross_attention_mask = cross_attention_mask.repeat_interleave(
            #         max_kps, dim=0)
            enc_cross_attention_mask = cross_attention_mask

            # print('\ncross_attention_mask: ', cross_attention_mask.shape)
            # print('Encoder hidden states shape: ', enc_hidden_states.shape)

            # print('In Unit Generation: decoder attention mask shape',
            #       decoder_attention_mask.shape)
            # print('\n INSIDE Unit Generation')

            dec_attention_mask = decoder_attention_mask.new_zeros(
                decoder_attention_mask.size())
            unit_causal_mask = decoder_attention_mask.new_ones(
                (max_kp_len, max_kp_len))

            # default for precision head pass
            if not cross_unit_attention:
                for i in range(max_kps):
                    dec_attention_mask[:, i*max_kp_len:(i+1)*max_kp_len, i*max_kp_len:(
                        i+1)*max_kp_len] = torch.tril(unit_causal_mask)
            else:
                # for i in range(max_kps):
                #     for j in range(max_kps):
                #         dec_attention_mask[:, i*max_kp_len:(i+1)*max_kp_len, j*max_kp_len:(
                #             j+1)*max_kp_len] = torch.tril(unit_causal_mask)

                # causal unit attention (unit k pays attention to all tokens so far frmo units 1..k-1)
                dec_attention_mask = decoder_attention_mask.new_zeros(
                    bs, max_kps, max_kp_len)

            # print('\n Decoder attention mask shape: ', dec_attention_mask.shape)
            # print('\n Decoder attention mask:  \n')

            # for x in dec_attention_mask[1].tolist():
            #     print(x)
            if output_precision_head:
                steps = 1 # using first output instead of average
            else:
                steps = max_kp_len
            # steps = max_kp_len

            for i in range(steps):
                # input_tokens_ = input_tokens[:, :i+1]

                # decoder attention mask for student forcing should be vanilla causal masks
                # extending till end of sequence or max_kp_len
                # print('\n Final decoder attention mask: ', dec_attention_mask_.shape)
                # position_ids_ = decoder_position_ids[:, :i+1]

                if cross_unit_attention:
                    dec_attention_mask[:, :, i] = 1

                decoder_outputs = self.decoder(
                    # bs*max_kps, max_kp_len
                    input_ids=input_tokens.view(-1, max_kps*max_kp_len),
                    position_ids=decoder_position_ids,
                    token_type_ids=decoder_token_type_ids,
                    attention_mask=dec_attention_mask if not cross_unit_attention else dec_attention_mask.contiguous().view(-1, max_kps*max_kp_len),
                    encoder_hidden_states=enc_hidden_states,
                    encoder_attention_mask=enc_cross_attention_mask,
                    past_key_values=None,
                    output_attentions=None,
                    use_cache=False,
                    # output hidden states only if precision heads are enabled
                    output_hidden_states=output_precision_head
                )
                # print('\nloop: decoder output logits shape: ', decoder_outputs.logits.shape)
                curr_output = decoder_outputs.logits.contiguous().view(-1, max_kps, max_kp_len,
                                                                       self.config.vocab_size).argmax(-1)[:, :, i]
                # print('\nloop: curr_output shape: ', curr_output.shape, curr_output)

                if i < max_kp_len-1:
                    input_tokens = input_tokens.clone()
                    input_tokens[:, :, i+1] = curr_output
                else:
                    input_tokens = torch.cat(
                        [input_tokens, curr_output.unsqueeze(-1)], dim=-1)
                # print('\n Input tokens: ', input_tokens.shape, input_tokens)

                if output_precision_head:
                    curr_hidden_states = decoder_outputs.hidden_states[-1]

            # reshape input tokens
            # print('\n input tokens shape: ', input_tokens.shape)
            if steps==1:
                decoded_tokens = None
            else:
                decoded_tokens = input_tokens[:, :, 1:].contiguous().view(-1, max_kps*max_kp_len)
            
            

            precision_head_mask = None
            if output_precision_head:
                # using mean
                # curr_hidden_states = curr_hidden_states.contiguous(
                # ).view(-1, max_kps, max_kp_len, 768).mean(dim=2)
                # print('\n curr_hidden_states', curr_hidden_states)

                # using only first hidden state
                curr_hidden_states = curr_hidden_states.contiguous(
                ).view(-1, max_kps, max_kp_len, 768)[:,:,0,:].clone()

                attn_mask_shape = (curr_hidden_states.size()[0], max_kps)
                attn_mask = decoder_input_ids.new_ones(attn_mask_shape)
                extended_attention_mask = attn_mask[:, None, None, :]
                curr_hidden_states = self.precision_head_attention(hidden_states=curr_hidden_states,
                                                                   attention_mask=extended_attention_mask,
                                                                   head_mask=None,
                                                                   encoder_hidden_states=None,
                                                                   encoder_attention_mask=None,
                                                                   past_key_value=None,
                                                                   output_attentions=False)
                # print('\n LOGITS SHAPE: ', logits[0].shape)
                curr_hidden_states = curr_hidden_states[0]
                precision_head_mask = self.precision_heads(curr_hidden_states)
            # print('\n precision head mask shape: ', precision_head_mask.shape)
        # print('unit generation prob shape: ', output.shape)
        # print('unit generation sequence shape: ', decoded_tokens.shape)
        # input_tokens stores the final greedy search output of the decoder using unit-wise decoding

        # print('\n final decoded tokens shape: ', decoded_tokens.shape)
        # print('\n final logits shape: ', decoder_outputs.logits.shape)
        # if output_precision_head:
        #     print('\n decoder hidden states shape: ', len(
        #         decoder_outputs), decoder_outputs.hidden_states[-1].shape)
        #     print('\n precision head mask shape: ', precision_head_mask.shape)

        return UnitGenerationOutput(
            sequences=decoded_tokens,  # (bs, max_kps*max_kp_len)
            # (bs, max_kps*max_kp_len, vocab_size)
            logits=decoder_outputs.logits,
            # (bs, max_kps*max_kp_len, 768)
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_precision_head_mask=precision_head_mask  # (bs, max_kps, 2)
        )

    def unit_generation_past_kv(self,
                                input_ids: Optional[torch.LongTensor] = None,
                                attention_mask: Optional[torch.FloatTensor] = None,
                                cross_attention_mask: Optional[torch.FloatTensor] = None,
                                decoder_input_ids: Optional[torch.LongTensor] = None,
                                encoder_outputs: Optional[torch.LongTensor] = None):

        # print('\n Inside unit generation pask kv...')

        max_kps = self.config.max_kps
        max_kp_len = self.config.max_kp_len

        # forward pass through encoder
        with torch.no_grad():
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=None
                )

            encoder_hidden_states = encoder_outputs[0]

            bs, _ = decoder_input_ids.size()

            # input_tokens = decoder_input_ids.new_zeros(bs, max_kps, max_kp_len)
            input_tokens = decoder_input_ids.new_zeros(bs*max_kps, 1)
            # 101 is token id for CLS token/ SOS token
            input_tokens[:, 0] = 101
            # preparing encoder hidden states
            enc_hidden_states = encoder_hidden_states
            if cross_attention_mask.size(0) != encoder_hidden_states.size(0):
                cross_attention_mask = cross_attention_mask.repeat_interleave(
                    max_kps, dim=0)
            enc_cross_attention_mask = cross_attention_mask

            dec_attention_mask = input_tokens.new_ones(input_tokens.size())

            # without output attentions, decoder_outputs is a tuple of length 2
            # first value is output hidden state
            # second is a tuple of 12 past-key-value tuples
            # each past key value tuple has following tuples
            # torch.Size([256, 12, 1, 64]) # self attention key
            # torch.Size([256, 12, 1, 64]) # self attention value
            # torch.Size([256, 12, 384, 64]) # cross attention key
            # torch.Size([256, 12, 384, 64]) # cross attention value

            past_key_values = None

            # decoded_tokens = input_tokens
            for t in range(1, self.config.max_kp_len+1):
                # dec mask for past key value decoding is irrelevant as
                # all tokens from left to right are attented equally
                decoder_outputs = self.decoder(
                    input_ids=input_tokens,  # bs*max_kps, 1
                    attention_mask=dec_attention_mask,  # bs*max_kps, 1, 1
                    encoder_hidden_states=enc_hidden_states,
                    encoder_attention_mask=enc_cross_attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=None,
                    use_cache=True
                )
                past_key_values = decoder_outputs[-1]
                output = decoder_outputs[0]
                # Note: as past key value is used, the input sequence is always of length 1
                # this implies the output sequence will also be of length 1
                # for t th step, past_kv will contain hidden states for seq
                # upto length t-1. The inp seq is of length 1
                # for the decoder, the key layer would be key on inp_seq of length 1 concat with past t-1 hidden states for key
                # which makes it of length t-1+ 1 = t.
                # the query on the other hand is cleverly only computed for inp_seq of length 1 instead of t
                # key: (bs, 12, t, 64) query: (bs, 12, 1, 64), value: (bs, 12, t, 64)
                # therefore, when we perform (Q*K Trasp), we get (bs, 12, t, 1) instead of (bs, 12, t, t) in the case when past key values not used
                # this trick is possible as attention for Q's (t-1) positions from the t th position is not required in causal decoding
                # so this computation of t-1 need not be done
                input_tokens = output.argmax(-1)
                decoded_tokens = input_tokens if t == 1 else torch.cat(
                    [decoded_tokens, input_tokens], dim=-1)
                decoded_token_probs = output if t == 1 else torch.cat(
                    [decoded_token_probs, output], dim=1)

        # flattening and reshaping so that all unit kps for batch element in single row
        decoded_tokens = decoded_tokens.contiguous(
        ).view(-1, max_kps, max_kp_len).view(-1, max_kps*max_kp_len)
        decoded_token_probs = decoded_token_probs.contiguous().view(-1, max_kps, max_kp_len,
                                                                    self.config.vocab_size).view(-1, max_kps*max_kp_len, self.config.vocab_size)
        # input_tokens stores the final greedy search output of the decoder using unit-wise decoding
        return UnitGenerationOutput(
            sequences=decoded_tokens,
            logits=decoded_token_probs
        )

    def prepare_student_forcing_input(self, max_kps, max_kp_len, decoder_input_ids):
        decoder_input_ids_stud = decoder_input_ids.contiguous().view(-1, max_kps, max_kp_len)
        decoder_input_ids_stud = decoder_input_ids_stud.new_ones(
            decoder_input_ids_stud.size())
        decoder_input_ids_stud[:, :, 0] = 101
        decoder_input_ids_stud = decoder_input_ids_stud.contiguous().view(-1,
                                                                          max_kps*max_kp_len)
        return decoder_input_ids_stud

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_labels_mask: Optional[torch.LongTensor] = None,
        # # encoder hidden states positions is a list of [start, end] positions
        # encoder_hidden_states_positions: Optional[List[List[int]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items(
        ) if (not argument.startswith("decoder_")) and (not argument.startswith("loss_weight_mask"))}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        isTrain = self.config.istrain

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

            # print('\n Encoder outputs shape: ', encoder_outputs[0].shape)

        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        # print('encoder outputs: ',encoder_outputs)
        encoder_hidden_states = encoder_outputs[0]
        # enc_bs, _, _ = encoder_hidden_states.size()

        # add mlm head at end of encoder
        if self.config.encoder_mlm and isTrain:
            mlm_outputs = self.mlm_head(encoder_hidden_states)

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        curr_device = decoder_input_ids.device

        if self.config.seq2set:
            max_kps = self.config.max_kps
            max_kp_len = self.config.max_kp_len
        else:
            max_len_d = self.config.max_len_d

        bs, _ = decoder_input_ids.size()

        if self.config.seq2set and self.config.hungarian_assign and isTrain:
            # in the first step use student enforced k-step generation
            # to obtain the label and input orderings

            # unit generation using teacher forcing
            # unit_generation_out = self.unit_generation(cross_attention_mask=cross_attention_mask,
            #                                            decoder_input_ids=decoder_input_ids,
            #                                            decoder_position_ids=kwargs_decoder['position_ids'].contiguous().view(-1, max_kp_len),
            #                                            encoder_outputs=encoder_outputs,
            #                                            decoder_attention_mask=decoder_attention_mask)

            # unit generation using student forcing instead of teacher forcing while performing hungarian align (default)

            decoder_input_ids_stud = self.prepare_student_forcing_input(max_kp_len=max_kp_len,
                                                                        max_kps=max_kps,
                                                                        decoder_input_ids=decoder_input_ids)

            # print('\n decoder input ids stud (before->after): ', decoder_input_ids.shape, decoder_input_ids_stud.shape)

            # Note: in student forcing generation, decoder attention mask of teacher cant by used
            # as this mask ignores the units which dont have a valid kp
            # However in student forcing, we have to generate kps for every unit before choosing the
            # realignment. So, the attention mask is recreated inside unit_generation_student_forcing
            # The passed decoder attention mask is only used to obtain the size for creation
            # by default a causal mask is created for every unit regardless of valid kp unit condition
            unit_generation_out = self.unit_generation_student_forcing(
                encoder_outputs=encoder_outputs,
                cross_attention_mask=cross_attention_mask,
                decoder_input_ids=decoder_input_ids_stud,
                # .contiguous().view(-1, max_kp_len),
                decoder_position_ids=kwargs_decoder['position_ids'],
                decoder_token_type_ids=kwargs_decoder['token_type_ids'],
                decoder_attention_mask=decoder_attention_mask,
                cross_unit_attention=False, # changed
                output_precision_head=False)

            # # efficient decoding
            # unit_generation_out = self.unit_generation_past_kv(cross_attention_mask=cross_attention_mask,
            #                                                    decoder_input_ids=decoder_input_ids,
            #                                                    encoder_outputs=encoder_outputs)

            # input tokens are populated in t steps by student forcing
            # use the outputs for hungarian loss computation
            # get decoder ouput based on input of all t steps

            with torch.no_grad():
                # decoder_attention_mask = self.get_seq2set_dec_attn_mask_from_seq2seq(
                #     decoder_attention_mask).view(-1, max_kps, max_kp_len, max_kp_len)

                dec_attention_mask = decoder_attention_mask.new_zeros(
                    decoder_attention_mask.size())
                unit_causal_mask = decoder_attention_mask.new_ones(
                    (max_kp_len, max_kp_len))
                for i in range(max_kps):
                    dec_attention_mask[:, i*max_kp_len:(i+1)*max_kp_len, i*max_kp_len:(
                        i+1)*max_kp_len] = torch.tril(unit_causal_mask.detach().clone().to(curr_device))
                decoder_attention_mask = dec_attention_mask

                # print('\n from hungarian alignment: ', dec_attention_mask.shape)
                # print('\n Decoder attention mask:  \n')

                # for x in dec_attention_mask[1].tolist():
                #     print(x)

                decoder_outputs = unit_generation_out.logits.contiguous(
                ).view(-1, max_kps, max_kp_len, self.config.vocab_size)
                # (bs*max_kps, max_kp_len, vocab_size) - > (bs, max_kps, max_kp_len, vocab_size)
                # print('Decoder outputs shape: ', decoder_outputs.shape)

                shifted_prediction_scores = decoder_outputs[:,
                                                            :, :-1, :].contiguous()
                # print('\n shifter tprediction scores: ', shifted_prediction_scores)

                # print('labels: ',labels)
                labels = labels.contiguous().view(-1, max_kps,
                                                  max_kp_len)[:, :, 1:]

                # now compute hungarian realignment
                # mask needs to be regenerated due to shuffling
                align_inds = hungarian_realignment(
                    dist=shifted_prediction_scores.detach().cpu(), target=labels.detach().cpu(),
                    ignore_idx=[0, 100, 101, 102, 3904])  # PAD, UNK, CLS, SEP, none (phi token)

                # print('\n aligned inds: ', align_inds)
                # print('\n labels: ', labels)
                # print('\n kwargs decoder position ids: ', kwargs_decoder['position_ids'])
                # prepare realligned labels and attention mask
                # position_ids_temp = kwargs_decoder['position_ids'].contiguous().view(-1, max_kps, max_kp_len)
                # print('\n align_inds: ', align_inds)

                labels_aligned = []  # , decoder_attention_mask_aligned = [], []
                # position_ids_aligned = []

                for b_i in range(bs):
                    # print(f'batch {b_i} label order: {align_inds[b_i]}')
                    labels_aligned.append(labels[b_i][align_inds[b_i]])
                    # decoder_attention_mask_aligned.append(
                    #     decoder_attention_mask[b_i][align_inds[b_i]])
                    # aligning position ids
                    # position_ids_aligned.append(position_ids_temp[b_i][align_inds[b_i]])

                labels = torch.stack(labels_aligned, dim=0)
                # decoder_attention_mask = torch.stack(
                #     decoder_attention_mask_aligned, dim=0)
                # kwargs_decoder['position_ids'] = torch.stack(position_ids_aligned, dim=0).view(-1, max_kp_len)

                # if seq2set and the hungarian assign then teh target positions are permuted by hungarian align
                # as valid units can get all PAD token units assigned to it, we would only want to backpropogate
                # on the units where non padd units are assigned. The avtive loss indices needs to be computed
                # based on these non pad assigned units from the labels

                # assign labels to realigned labels, create input from labels
                sos = torch.ones(bs, max_kps, 1).to(
                    curr_device).long() * 101  # 101 for CLS token
                # for PAD unit inputs, the sos should be 0 or PAD
                sos = torch.where(labels[:, :, :1] == 0, labels[:, :, :1], sos)

                # as labels was truncated as sos token
                labels = torch.cat([sos, labels], dim=-
                                   1).view(-1, max_kps*max_kp_len)

                decoder_input_ids = labels.detach().clone().to(
                    curr_device).view(-1, max_kps*max_kp_len)
                decoder_attention_mask = decoder_attention_mask.view(
                    -1, max_kps*max_kp_len, max_kps*max_kp_len)

                kwargs_decoder['position_ids'] = kwargs_decoder['position_ids']

        # The decoder outputs needs to be collected

        # Use student forcing forward pass for precision head loss computation

        ph_loss = None
        if self.config.seq2set and self.decoder.config.precision_heads and isTrain:
            # bug-fix new_zeros to new_ones
            decoder_input_ids_stud = self.prepare_student_forcing_input(max_kp_len=max_kp_len,
                                                                        max_kps=max_kps,
                                                                        decoder_input_ids=decoder_input_ids)
            # print('\nDec inp ids shape: ', dec_inp_ids_.shape)
            # print('\nencouder outputs shape: ', encoder_outputs[0].shape)
            # print('\ncross attention mask shape: ' ,cross_attention_mask.shape)
            # print('\ndecoder attention mask shape: ', orig_decoder_attention_mask.shape)
            # print('\n precision head student forcing..')
            generation_res = self.unit_generation_student_forcing(
                encoder_outputs=encoder_outputs,
                cross_attention_mask=cross_attention_mask,
                decoder_input_ids=decoder_input_ids_stud,
                decoder_position_ids=kwargs_decoder['position_ids'],
                decoder_token_type_ids=kwargs_decoder['token_type_ids'],
                decoder_attention_mask=decoder_attention_mask,
                cross_unit_attention=False,
                output_precision_head=True)
            # print('\n\n orig decoder attention mask: ', orig_decoder_attention_mask)

            # (bs, max_kps, 2)
            logits = generation_res.decoder_precision_head_mask
            # print('\n logits shape: ', logits.shape)
            # get valid kp positions from decoder input ids
            # bs, max_kps (boolean)
            dec_inp_ids = decoder_input_ids.detach().clone()
            # print('\n dec_inp_ids: ', dec_inp_ids.contiguous().view(-1, max_kps, max_kp_len))

            # as phi tokens are included check condition changes to find all units havig phi (token 3904) as the 2nd token after cls token
            valid_kp_pos = (dec_inp_ids.contiguous().view(-1, max_kps, max_kp_len)[:,:,1] == 3904).long().view(-1)

            # valid_kp_pos = dec_inp_ids.contiguous().view(-1, max_kps, max_kp_len).sum(dim=-1) != 0


            # print('\n dec_inp_ids: ', dec_inp_ids)
            # print('\n valid kp pos: ', valid_kp_pos)
            ph_gt = valid_kp_pos.long().view(-1)
            ph_pred = logits.view(-1, 2)

            # apply weights based on invert label count of 0s and 1s in flattened batch
            total_ones = (torch.sum((ph_gt == 1).long()).item())*1.0
            total_zeros = (len(ph_gt)-total_ones)*1.0
            eps = 1e-08
            weights = torch.tensor(
                [total_ones+eps, total_zeros+eps]).to(ph_gt.device)
            # print('\n total ones: ', total_ones)
            # print('\n total zeros: ', total_zeros)

            ph_loss_fn = nn.CrossEntropyLoss(weight=weights)

            # print('\n ph pred: ', ph_pred, ph_gt)
            ph_loss = ph_loss_fn(ph_pred, ph_gt)
            # print('\n ph loss item: ', ph_loss.item())

        # print('\n\n from encoder decoder model (position ids) : ', kwargs_decoder['position_ids'], decoder_input_ids.shape)
        # Actual Decode
        # not passing position_ids here explicitly as kwargs_decoder has position ids, same for token type ids
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            # position_ids=kwargs['decoder_position_ids'].contiguous().view(-1, max_kp_len),
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            # encoder attention mask while decoding is given by cross attention mask
            # cross attention mask is same as attention mask if no filter extractive tokens
            encoder_attention_mask=cross_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,  # output_hidden_states,
            use_cache=use_cache,
            # labels=labels, # Added labels for decoding loss from model
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder
        )
        precision_head_mask = None
        if self.config.seq2set and self.decoder.config.precision_heads and not isTrain:
            # for inference
            # wait until shape of decoder outputs (in kp_seq dim) is equal to max_kp_len
            # Note: usage of past kv and single token for each pass decoding not supported with precision heads
            # each decode pass requires all tokens from left to current token
            # print('\nEntered precision heads')
            curr_hidden_states = decoder_outputs.hidden_states[-1]
            # using mean
            # curr_hidden_states = curr_hidden_states.contiguous(
            # ).view(-1, max_kps, max_kp_len, 768).mean(dim=2)

            # using only first output embedding of sequence generation
            curr_hidden_states = curr_hidden_states.contiguous(
            ).view(-1, max_kps, max_kp_len, 768)[:,:,0,:]

            attn_mask_shape = (curr_hidden_states.size()[0], max_kps)
            attn_mask = decoder_input_ids.new_ones(attn_mask_shape)
            extended_attention_mask = attn_mask[:, None, None, :]
            curr_hidden_states = self.precision_head_attention(hidden_states=curr_hidden_states,
                                                               attention_mask=extended_attention_mask,
                                                               head_mask=None,
                                                               encoder_hidden_states=None,
                                                               encoder_attention_mask=None,
                                                               past_key_value=None,
                                                               output_attentions=False)
            # print('\n LOGITS SHAPE: ', logits[0].shape)
            curr_hidden_states = curr_hidden_states[0]
            precision_head_mask = self.precision_heads(curr_hidden_states)

        mlm_loss = None
        # mlm loss implementation
        if input_ids is not None and self.config.encoder_mlm and isTrain:
            mlm_loss = CrossEntropyLoss()
            encoder_labels_flat = input_ids.contiguous().view(-1)
            mlm_active_loss_indices = encoder_labels_mask.contiguous().view(-1) == 1
            mlm_active_logits = mlm_outputs.contiguous(
            ).view(-1, self.config.vocab_size)[mlm_active_loss_indices]
            mlm_active_labels = encoder_labels_flat[mlm_active_loss_indices]
            mlm_loss = mlm_loss(mlm_active_logits, mlm_active_labels)
        # modified implementation of loss (ignoring [PAD] token positions)

        loss = None
        # active_loss_indices = None
        if labels is not None:
            if self.config.seq2set:
                shifted_prediction_scores = decoder_outputs.logits.contiguous(
                ).view(-1, max_kps, max_kp_len, self.config.vocab_size)[:, :, :-1, :].contiguous()

                labels = labels.contiguous().view(-1, max_kps,
                                                  max_kp_len)[:, :, 1:]

                labels = labels.contiguous().view(-1, max_kps*(max_kp_len-1))

                # shifted_prediction_scores = shifted_prediction_scores.to(curr_device)
                shifted_decoder_input_ids = decoder_input_ids.contiguous().view(-1, max_kps,
                                                                                max_kp_len)[:, :, :-1]
                shifted_decoder_input_ids = shifted_decoder_input_ids.contiguous().view(-1,
                                                                                        max_kps*(max_kp_len-1))
            else:
                # we are doing next-token prediction; shift prediction scores and input ids by one
                shifted_prediction_scores = decoder_outputs.logits[:, :-1, :].contiguous(
                )
                # print('shifted pred scores shape: ', shifted_prediction_scores.shape)

                labels = labels[..., 1:].contiguous()
                # obtain mask from decoder_input_ids (non 0 ids)
                shifted_decoder_input_ids = decoder_input_ids[..., :-1]

            # cases where not seq2set with hungarian (or target not permuted)
            # if active_loss_indices is None:
            # find active logits
            # mask_temp = torch.ones(
            #     shifted_decoder_input_ids.shape).to(curr_device)
            # assuming token id of 0 corresponds to [CLS]
            # loss is computed by ignoring PAD tokens
            # active_loss_indices = torch.where(
            #     shifted_decoder_input_ids != 0, mask_temp, shifted_decoder_input_ids).contiguous().view(-1) == 1
            active_loss_indices = (shifted_decoder_input_ids!=0).contiguous().view(-1)
            # print('active loss indices shape: ', active_loss_indices.shape)

            # print('\nlabels shape: ', labels.shape)
            # print('\nshifted decoder input ids shape: ', shifted_decoder_input_ids.shape)
            active_logits = shifted_prediction_scores.view(
                -1, self.config.vocab_size)[active_loss_indices]
            active_labels = labels.view(-1)[active_loss_indices]

            # find number of None token weight
            # none_tokid = 3904
            # none_count = (active_labels == none_tokid).sum()
            active_count = len(active_labels)

            if self.config.seq2set:
                total_length = (
                    (active_count//(max_kps*max_kp_len)) + 1)*(max_kps*max_kp_len)
            else:
                total_length = ((active_count//max_len_d) + 1)*(max_len_d)

            norm_factor = total_length-active_count
            none_wt = min(1.0, 1.0/(norm_factor+1e-08))
            # print('\n none_wt: ', none_wt)
            # overriding none_wt to constant factor of 0.1
            # none_wt = 0.1
            none_wt = 0.1

            weight_list = [1.0]*3904 + [none_wt] + \
                [1.0]*(self.config.vocab_size-3905)

            assert len(
                weight_list) == self.config.vocab_size, 'weight list length != vocab size'

            class_weight = torch.tensor(
                weight_list, device=curr_device)

            # loss_fct = CrossEntropyLoss(weight=class_weight, label_smoothing=0.2)
            loss_fct = CrossEntropyLoss(weight=class_weight)

            loss = loss_fct(active_logits, active_labels)

        # for evaluation using Transformers generate libarary
        # the output should be of cur_seq_len, where cur_seq_len is equal to input seq_len (1->max_kp_len)
        if not isTrain and self.config.seq2set:
            # print('\n Reverting to stacked during testing end: ')
            num_beams = self.config.num_beams
            # print('\n numbeams from reverting: ', num_beams)
            # truncate to seq len and restack
            decoder_attention_mask = decoder_attention_mask[0][:max_kp_len, :max_kp_len]
            # sum the first column
            curr_seq_len = decoder_attention_mask[:, 0].sum().item()

            # print('\n decoder attentino mask shape: ',
            #       decoder_attention_mask.shape, "curr seq len: ", curr_seq_len)
            # use the curr seq len to truncate the output logits
            # as beam uses repeat interleave the order is (bs*max_kps*num_beams, seq_len)
            # but in prepare_inputs we change it to (bs*num_beams, max_kps*seq_len)
            # so we need to permute to bring max_kps afer bs like (bs, max_kps, num_beams, seq_len) -> (bs*max_kps*num_beams, seq_len)
            decoder_outputs.logits = decoder_outputs.logits.contiguous().view(-1, num_beams, max_kps, max_kp_len,
                                                                              self.config.vocab_size)[:, :, :, :curr_seq_len, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, curr_seq_len, self.config.vocab_size)

            # print('\n decoder output logits shape: ',
            #       decoder_outputs.logits.shape)
            # ToDO: Flatten extra return params
            decoder_outputs.past_key_values = None
            # decoder_outputs.hidden_states = None
            decoder_outputs.cross_attentions = None

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        # ToDO: to add precision head mask result to this output
        # also change the implementation of huggingface generation library
        # to output head mask for post processing results

        # print("Decoder hidden states: ", decoder_outputs.hidden_states)
        return Seq2SeqLMOutput(
            loss=(loss, mlm_loss, ph_loss),
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            decoder_precision_head_mask=precision_head_mask,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # called after preparing prepare_input_ids_for_generation (which creates [CLS] batch)
    # called after expand_inputs_for_generation in beam search
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # input_ids are decoder generation inputids, it is passed explicitly for seq2set to expand it to (bs*max_kps)
        # attention_mask is encoder attention mask, which by default (inside generation utils) is computed to mask pad tokens
        # The utils calls get_encoder() first and uses that to store model_outputs
        # followed by calling this model forward with encoder_outputs
        # therefore attention mask of the encoder has no role to play from this point and hence is not repeated to (bs*max_kps)
        # Flow: (pass to encoder(with regular bs))->encoder_outputs (bert imp extended to expand bs*max_kps for seq2set)
        # ->expand encoder ouputs to num_beams and all decoder params (decoder input_ids explicitly expanded and passed to .generate)
        # modified batch size = batch_size*max_kps before innitializing beam_scorer in utils
        # generation and decoding can now happen for seq2set for all stacked units in one forward pass

        # Note: use_cache is changed to default false so that past key values are not used
        # also the prepare inputs for generation in decoder of bert modeling, cuts input ids
        # to last index if past key values are used, passing them one at a time, leveraging the past kvs that
        # are incrementally populated. By default if use_cache not given its set as True in the function argument.
        # so its passed expclictly as False , use_cache is actually taken from model config use_cache parameter
        # if generation config is not passed in .generate(). by default its set as True
        # explictly changed to false here.
        use_cache = False
        decoder_inputs = self.decoder.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, use_cache=use_cache)

        if self.config.seq2set:
            max_kps = self.config.max_kps
            max_kp_len = self.config.max_kp_len
            bs_maxkps_num_beams, seq_len = input_ids.size()
            num_beams = self.config.num_beams
            bs = bs_maxkps_num_beams//(max_kps*num_beams)
            bs_num_beams = bs_maxkps_num_beams//max_kps
            # print('\n batch size: ', bs, 'num_beams: ', num_beams)

            ###### PREPARE INPUT IDS ##########
            # the inputs need to be flattened and shapped as max_kp_len as we are unstacking for forward pass
            # it should be truncated using information from decoder mask and stacked again for isTrain=False in forward() at the end
            # print('\n (prepare generation outputs) decoder inputs: ',
            #     decoder_inputs["input_ids"].shape)
            # decoder_inputs["input_ids"] = decoder_inputs["input_ids"].contiguous(
            # ).view(-1, max_kps, seq_len)

            # pad only last dim or seq_len dimension
            decoder_inputs["input_ids"] = F.pad(
                decoder_inputs["input_ids"], (0, max_kp_len-seq_len), "constant", 0)
            decoder_inputs["input_ids"] = decoder_inputs["input_ids"].contiguous(
            ).view(-1, max_kps, num_beams, max_kp_len).permute(0, 2, 1, 3).contiguous().view(-1, max_kps*max_kp_len)
            # print('\n (prepare generation outputs (next)) decoder inputs: ',
            #     decoder_inputs["input_ids"].shape)

            ###### PREPARE ATTENTION MASK ##########
            # Note: prepare 2d causal attention mask flattened version which contains info on length of current sequence
            # not permuted based on num_beams as 2d causal mask is constant for every unit, every beam and batch element
            # in this implementation
            decoder_attention_mask = input_ids.new_zeros(
                (bs_num_beams, max_kps*max_kp_len, max_kps*max_kp_len))
            unit_2d_mask = input_ids.new_zeros((max_kp_len, max_kp_len))
            unit_1d_pos = torch.tril(input_ids.new_ones((seq_len, seq_len)))
            unit_2d_mask[:seq_len, :seq_len] = unit_1d_pos

            for kp_i in range(self.config.max_kps):
                decoder_attention_mask[:, kp_i*max_kp_len: (
                    kp_i+1)*max_kp_len, kp_i*max_kp_len: (kp_i+1)*max_kp_len] = unit_2d_mask

            if 'decoder_position_ids' in kwargs:
                kwargs['decoder_position_ids'] = kwargs['decoder_position_ids'].contiguous(
                ).view(-1, max_kps, num_beams, max_kp_len).permute(0, 2, 1, 3).contiguous().view(-1, max_kps*max_kp_len)
            if 'decoder_token_type_ids' in kwargs:
                kwargs['decoder_token_type_ids'] = kwargs['decoder_token_type_ids'].contiguous(
                ).view(-1, max_kps, num_beams, max_kp_len).permute(0, 2, 1, 3).contiguous().view(-1, max_kps*max_kp_len)
            # print('\n decoder token type ids: ', kwargs['decoder_token_type_ids'])
            # print('\n position ids: ', kwargs['decoder_position_ids'].tolist())

        else:
            # for seq2seq, we use the default settings
            decoder_attention_mask = input_ids.new_ones(input_ids.shape)

        cross_attention_mask = kwargs['cross_attention_mask']
        # print('\n cross attention mask: ', cross_attention_mask.shape)
        # print('\n attention mask: ', attention_mask.shape)
        # print('\n decoder attention mask: ', decoder_attention_mask.shape)
        # print('\n input ids: ', decoder_inputs["input_ids"].shape)
        # print('\n encoder outputs: ', encoder_outputs[0].shape)

        input_dict = {
            "attention_mask": attention_mask,
            "cross_attention_mask": cross_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "decoder_position_ids": kwargs["decoder_position_ids"] if 'decoder_position_ids' in kwargs else None,
            "decoder_token_type_ids": kwargs["decoder_token_type_ids"] if 'decoder_token_type_ids' in kwargs else None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)

    # beam search implementation

    def beam_search_decode(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            encoder_outputs: Optional[torch.LongTensor] = None,
            max_beam_size=4,
            beam_width=4
    ):
        # print('\n Inside unit generation...')

        max_kps = self.config.max_kps
        max_kp_len = self.config.max_kp_len

        # forward pass through encoder
        with torch.no_grad():
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=None
                )

            encoder_hidden_states = encoder_outputs[0]
            # encoder_hidden_states = encoder_hidden_states.repeat_interleave(
            #     beam_width, dim=0)

            bs_maxkps, _ = decoder_input_ids.size()
            bs = bs_maxkps // max_kps

            # input_tokens = decoder_input_ids.new_zeros(bs, max_kps, max_kp_len)
            # input_tokens = decoder_input_ids.contiguous().view(bs*max_kps, max_kp_len)
            # preparing encoder hidden states
            enc_hidden_states = encoder_hidden_states
            cross_attention_mask = cross_attention_mask.repeat_interleave(
                max_kps, dim=0)  # .repeat_interleave(beam_width, dim=0)
            enc_cross_attention_mask = cross_attention_mask

            # innitialize first input
            input_tokens = decoder_input_ids  # .repeat_interleave(
            decoded_tokens = input_tokens.detach().clone().to(input_tokens.device)
            # beam_width, dim=0)  # * [101]
            dec_attention_mask = input_tokens.new_ones(input_tokens.size())

            # 0s where eos is encountered
            eos_mask = input_tokens.new_ones(input_tokens.size())
            cum_probs = input_tokens.new_zeros(input_tokens.size())
            # print('encoder hidden states shape: ', encoder_hidden_states.shape)
            # print('encoder cross attention mask: ',
            #       enc_cross_attention_mask.shape)
            # print('input_tokens seed shape: ', input_tokens.shape)
            # print('decoded tokens seed shape: ', decoded_tokens.shape)
            # print('dec attention mask seed shape: ', dec_attention_mask.shape)
            # print('eos mask seed shape: ', eos_mask.shape)
            # print('cum_prob seed shape: ', cum_probs.shape)

            # without output attentions, decoder_outputs is a tuple of length 2
            # first value is output hidden state
            # second is a tuple of 12 past-key-value tuples
            # each past key value tuple has following tuples
            # torch.Size([256, 12, 1, 64]) # self attention key
            # torch.Size([256, 12, 1, 64]) # self attention value
            # torch.Size([256, 12, 384, 64]) # cross attention key
            # torch.Size([256, 12, 384, 64]) # cross attention value

            past_key_values = None

            # decoded_tokens = input_tokens
            for t in range(1, self.config.max_kp_len+1):
                # dec mask for past key value decoding is irrelevant as
                # all tokens from left to right are attented equally
                decoder_outputs = self.decoder(
                    input_ids=input_tokens,  # bs*max_kps*bw, 1
                    attention_mask=dec_attention_mask,  # bs*max_kps*bw, 1, 1
                    encoder_hidden_states=enc_hidden_states,
                    encoder_attention_mask=enc_cross_attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=None,
                    use_cache=True
                )
                # print(f'time step: {t} decoder outputs:  ',
                #   decoder_outputs[0].shape)
                past_key_values = decoder_outputs[-1]
                output = decoder_outputs[0]

                # new expand size*bs
                # Assumption:  max_beam_size should be a multiple of beam width
                B = output.size()[0]*beam_width
                # print(f'time step: {t} B: ', B)

                # expand_by = B//output.size()[0]

                # Note: as past key value is used, the input sequence is always of length 1
                # this implies the output sequence will also be of length 1
                # for t th step, past_kv will contain hidden states for seq
                # upto length t-1. The inp seq is of length 1
                # for the decoder, the key layer would be key on inp_seq of length 1 concat with past t-1 hidden states for key
                # which makes it of length t-1+ 1 = t.
                # the query on the other hand is cleverly only computed for inp_seq of length 1 instead of t
                # key: (bs, 12, t, 64) query: (bs, 12, 1, 64), value: (bs, 12, t, 64)
                # therefore, when we perform (Q*K Trasp), we get (bs, 12, t, 1) instead of (bs, 12, t, t) in the case when past key values not used
                # this trick is possible as attention for Q's (t-1) positions from the t th position is not required in causal decoding
                # so this computation of t-1 need not be done

                # (bs*max_kps*bw^x, bw) for both values and indices->(bs*max_kps*bw^x+1, 1)
                values, indices = output.topk(
                    k=beam_width, largest=True, dim=-1)
                values = values.contiguous().view(-1, 1)
                indices = indices.contiguous().view(-1, 1)

                # print('values: ', values.shape)
                # print('indices: ', indices.shape)

                # expand input tokens (bs*max_kps*bw^x+1, 1)
                input_tokens = input_tokens.repeat_interleave(
                    beam_width, dim=0)

                # concatentate and store cumulitative outputs (bs*max_kps*bw^x+1, prev_len+1)
                # input_tokens received at this point is always sorted by beam indices

                # find cumulitative probabilities (bs*max_kps*bw^x*bw, 1)
                curr_eos_mask = torch.where(indices != 102, input_tokens.new_ones(
                    input_tokens.size()), input_tokens.new_zeros(input_tokens.size()))
                # broadcast previous eos mask
                prev_eos_mask = eos_mask.repeat_interleave(beam_width, dim=0)
                # set new eos mask (make all positions where curr_mask == eos to zeros and retain previous mask)
                eos_mask = curr_eos_mask*prev_eos_mask
                # print('curr mask: ', curr_eos_mask[0])
                # print('prev mask: ', prev_eos_mask[0])
                # print('final eos mask: ', eos_mask[0])
                # print('new eos mask shape: ', eos_mask.shape)

                curr_probs = torch.log(values+1e-08)
                # mask curr_probs correctly
                # include sep token if encountered newly
                # previous eos mask used instead of current
                # curr_probs = curr_probs*prev_eos_mask
                # print('curr probs shape: ', curr_probs.shape)
                # print('eos mask shape: ', prev_eos_mask.shape)
                # print('curr_probs shape: ', curr_probs.shape)
                cum_probs = cum_probs.repeat_interleave(beam_width, dim=0)
                # print('cum probs: ', cum_probs.shape,
                #       'curr_probs: ', curr_probs.shape)
                cum_probs = cum_probs + curr_probs
                # print('cum probs shape: ', cum_probs.shape)
                # sort the beams in decreasing order of their cum log probabilites (bs*max_kps*bw^x+1, 1)
                sorted_beam_indices = cum_probs.view(bs_maxkps, B//bs_maxkps, -1).argsort(
                    dim=1, descending=True)
                # rearrange indices across batch dimension
                indices = indices.view(
                    bs_maxkps, B//bs_maxkps, -1).gather(1, sorted_beam_indices)
                # print('indices: ', indices)

                # print('sorted beam indices shape: ', sorted_beam_indices.shape)
                # print('sorted indices shape: ', indices.shape)
                # also rearrange past kv self-attention values across batch dimension so that
                # past kv decoding occurs correctly
                # past_key_values = past_key_values.repeat_interleave(
                #     beam_width, dim=0)

                # tuple implementation
                new_past_key_values = ()
                for layer in past_key_values:
                    new_pkv = ()
                    # key and value tuple for self attention layer is stored in 0, 1 and for cross attention layer at 3,4
                    # but before calling cross attention layer, the 2,3 is sliced and passed as 0,1 to cross attentino layer
                    # for self attention layer, the 0,1 is sliced as is and passed
                    # the resultant kpv caches for self and cross attention layer are again concat to 0,1,2,3 psotions
                    # (which is from decoder for self attention and encoder for cross attention)
                    # the cross attention key and values are stored in postions 2 and 3
                    for pk_i in range(4):
                        # print('layer sizes: ', layer[pk_i].shape)
                        new_pkv = new_pkv + \
                            (layer[pk_i].repeat_interleave(beam_width, dim=0),)
                    # new_pkv = new_pkv + layer[2:]
                    new_past_key_values = new_past_key_values + (new_pkv,)
                past_key_values = new_past_key_values

                # list implementation
                # past_key_values = list(past_key_values)
                # for i in range(len(past_key_values)):
                #     past_key_values[i] = list(past_key_values[i])
                #     for pk_i in range(4):
                #         past_key_values[i][pk_i] = past_key_values[i][pk_i].repeat_interleave(beam_width, dim=0)
                #     past_key_values[i] = tuple(past_key_values[i])
                # past_key_values = list(past_key_values)

                # 1st and 2nd indices correspond to past key and value layers
                # iterate through all layers to access the past key and values
                new_past_key_values = ()
                for layer in past_key_values:
                    new_pkv = ()
                    for pk_i in range(4):
                        orig_size = layer[pk_i].size()
                        view_size = (bs_maxkps, B//bs_maxkps) + \
                            layer[pk_i].size()[1:]
                        new_pkv_ = layer[pk_i].view(view_size).gather(
                            1, sorted_beam_indices.view(bs_maxkps, B//bs_maxkps, 1, 1, 1).expand(-1, -1, view_size[2], view_size[3], view_size[4]))
                        # remerge batch dimension after rearranging
                        new_pkv = new_pkv + (new_pkv_.view(orig_size),)

                    new_past_key_values = new_past_key_values + (new_pkv,)

                # sort encoder cross attention mask according to sorted beam indices
                enc_cross_attention_mask = enc_cross_attention_mask.repeat_interleave(
                    beam_width, dim=0).view(bs_maxkps, B//bs_maxkps, -1)
                enc_cross_attention_mask = enc_cross_attention_mask.gather(
                    1, sorted_beam_indices.view(
                        bs_maxkps, B//bs_maxkps, 1).expand(-1, -1, enc_cross_attention_mask.size()[-1])
                )

                # new input tokens is rearranged beams
                # indices are sorted earlier
                input_tokens = indices.view(-1, 1)

                past_key_values = new_past_key_values
                enc_cross_attention_mask = enc_cross_attention_mask.view(
                    -1, enc_cross_attention_mask.size()[-1])
                # broad cast past keys and values to current input_tokens shape
                # past_key_values = past_key_values.repeat_interleave(beam_width, dim=0)
                # decoder attention mask need not be shuffled as by default is 1 always (when past kv is used has size 1 and value 1)
                dec_attention_mask = dec_attention_mask.repeat_interleave(
                    beam_width, dim=0)

                # decoded tokens so far should be sorted based on current output beam indices (topk)
                # then previous can be concatenated with current
                decoded_tokens = decoded_tokens.repeat_interleave(
                    beam_width, dim=0)
                decoded_tokens = decoded_tokens.view(bs_maxkps, B//bs_maxkps, decoded_tokens.size()[-1]).gather(
                    1, sorted_beam_indices.view(bs_maxkps, B//bs_maxkps, 1).expand(bs_maxkps, B//bs_maxkps, decoded_tokens.size()[-1]))
                decoded_tokens = decoded_tokens.view(-1,
                                                     decoded_tokens.size()[-1])
                decoded_tokens = torch.cat(
                    [decoded_tokens, input_tokens], dim=-1)

                # only use the top k beams once max_beam_size has exceeded
                # print('B//bs_maxkps', B, bs_maxkps, B//bs_maxkps, max_beam_size)
                if B//bs_maxkps > max_beam_size:
                    # print('\nReduction..')
                    # B_sel = B//beam_width

                    # pruned beam size B0
                    B0 = (B//bs_maxkps)//beam_width
                    pruned_bs = B0*bs_maxkps

                    decoded_tokens = decoded_tokens.contiguous().view(bs_maxkps, B//bs_maxkps, -
                                                                      1)[:, :B0, ...].contiguous().view(pruned_bs, -1)
                    input_tokens = input_tokens.contiguous().view(bs_maxkps, B//bs_maxkps, -
                                                                  1)[:, :B0, ...].contiguous().view(pruned_bs, -1)
                    dec_attention_mask = dec_attention_mask.contiguous().view(bs_maxkps, B //
                                                                              bs_maxkps, -1)[:, :B0, ...].contiguous().view(pruned_bs, -1)
                    enc_cross_attention_mask = enc_cross_attention_mask.contiguous().view(
                        bs_maxkps, B//bs_maxkps, -1)[:, :B0, ...].contiguous().view(pruned_bs, -1)

                    new_past_key_values = ()
                    for layer in past_key_values:
                        pk_new = ()
                        for pk_i in range(4):
                            view_size = layer[pk_i].size()
                            view_size_split = (
                                bs_maxkps, B//bs_maxkps)+view_size[1:]
                            new_view_size = (
                                pruned_bs,)+view_size[1:]
                            # print(view_size, view_size_split, new_view_size)

                            pk_new = pk_new + (layer[pk_i].contiguous().view(view_size_split)[
                                               :, :B0, ...].contiguous().view(new_view_size),)
                        new_past_key_values = new_past_key_values + \
                            (pk_new,)
                    past_key_values = new_past_key_values

                    eos_mask = eos_mask.contiguous().view(bs_maxkps, B//bs_maxkps, -
                                                          1)[:, :B0, ...].contiguous().view(bs_maxkps*beam_width, -1)
                    cum_probs = cum_probs.contiguous().view(bs_maxkps, B//bs_maxkps, -
                                                            1)[:, :B0, ...].contiguous().view(bs_maxkps*beam_width, -1)

                # print('decoded tokens: ', decoded_tokens)
                # print('\nnew input tokens shape: ', input_tokens.shape)
                # print('new past key values shape: ', len(past_key_values),
                #       past_key_values[-1][0].shape, past_key_values[-1][1].shape, past_key_values[-1][2].shape, past_key_values[-1][3].shape)
                # print('new decoder attention mask: ', dec_attention_mask.shape)
                # print('new encoder cross attention mask: ',
                #       enc_cross_attention_mask.shape)
                # print('new decoded tokens shape: ', decoded_tokens.shape)
                # print('new eos mask shape: ', eos_mask.shape)
                # print('new cum prob shape: ', cum_probs.shape)
                # Note: Encoder hidden states and encoder cross attention mask is not used once past-key values are obtained
                # after timestamp t. These hidden states (broadcasted to num_beams) is obtained directly from past key values

        # print('unit generation prob shape: ', output.shape)
        # print('unit generation sequence shape: ', decoded_tokens.shape)
        # input_tokens stores the final greedy search output of the decoder using unit-wise decoding
        return UnitGenerationOutput(
            sequences=decoded_tokens,
            logits=None
        )
