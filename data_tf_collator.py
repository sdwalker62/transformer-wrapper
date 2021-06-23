# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf

from ..tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from tensorflow.data import Dataset


@dataclass(frozen=True)
class TFDataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    @tf.function
    def pseudo_bernoulli(self, prob_matrix, labels):
        return tf.cast(prob_matrix - tf.random.uniform(tf.shape(labels), 0, 1) >= 0, tf.bool)

    @tf.function
    def mask_special_tokens(self, labels, special_tokens):
        # Finds all special tokens within labels
        x = tf.map_fn(lambda b: tf.cast(tf.math.equal(labels, b), tf.int32), special_tokens)
        return tf.math.greater(tf.reduce_sum(x, axis=0), 0)

    @tf.function()
    def __call__(self, examples: Union[List[int], tf.Tensor, Dict[str, tf.Tensor]]) -> Dataset:
        encoding_batch = {}
        # special_tokens_mask = examples0.pop("special_tokens_mask", None)
        encoding_batch["input_ids"], encoding_batch["labels"] = self.tf_mask_tokens(
            examples, special_tokens_mask=None
        )
        return encoding_batch

    @tf.function
    def tf_mask_tokens(
            self,
            inputs: tf.Tensor,
            special_tokens_mask: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """-
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = tf.identity(inputs)

        # # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = tf.fill(tf.shape(labels), self.mlm_probability)

        if special_tokens_mask is None:
            special_tokens_tensor = tf.constant(self.tokenizer.all_special_ids, dtype=tf.int32)
            special_tokens_mask = self.mask_special_tokens(labels, special_tokens_tensor)
        else:
            special_tokens_mask = tf.cast(special_tokens_mask, dtype=tf.bool)

        probability_matrix = tf.where(~special_tokens_mask, probability_matrix, 0)
        masked_indices = self.pseudo_bernoulli(probability_matrix, labels)

        labels = tf.where(masked_indices, labels, -100)  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
                self.pseudo_bernoulli(tf.fill(tf.shape(labels), 0.8), labels) & masked_indices
        )

        mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        inputs = tf.where(~indices_replaced, inputs, mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
                self.pseudo_bernoulli(tf.fill(tf.shape(labels), 0.5), labels)
                & masked_indices
                & ~indices_replaced
        )

        random_words = tf.random.uniform(tf.shape(labels), maxval=len(self.tokenizer), dtype=tf.int32)

        inputs = tf.where(indices_random, random_words, inputs)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
