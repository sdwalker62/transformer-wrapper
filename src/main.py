from transformers.data.data_tf_collator import TFDataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorForLanguageModeling

import tensorflow as tf
import tensorflow_probability as tfp
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

# -- Tokenizer -- #
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers

from transformers.tokenization_utils_base import BatchEncoding

from tokenizers.normalizers import (
    Lowercase,
    NFD,
    StripAccents
)

from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders


class PrimeTokenizer:
    def __init__(self, max_seq_length: int):
        self.prime_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

        self.prime_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

        self.prime_tokenizer.pre_tokenizer = Whitespace()

        self.prime_tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )

        self.trainer = WordPieceTrainer(
            vocab_size=153411, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )

        self.prime_tokenizer.decoder = decoders.WordPiece()
        # self.prime_tokenizer.enable_padding(length=max_seq_length)
        # self.prime_tokenizer.enable_truncation(max_seq_length)

    def text_to_sequence(self, input_):
        if type(input_) is list:
            return self.prime_tokenizer.encode_batch(input_)
        return self.prime_tokenizer.encode(input_)

    def sequence_to_text(self, input_):
        if type(input_) is list:
            return self.prime_tokenizer.decode_batch(batch)
        return self.prime_tokenizer.decode(input_)

    def train(self, data):
        log_itr = iter(data)
        #         tqdm_log_itr = tqdm(iterable=log_itr, total=len(data))
        # tqdm_log_itr.__iter__()
        self.prime_tokenizer.train_from_iterator(log_itr, self.trainer)
        self.save()

    def get_tokenizer(self):
        return self.prime_tokenizer

    def get_vocab(self):
        return self.prime_tokenizer.get_vocab()

    def get_vocab_size(self):
        return self.prime_tokenizer.get_vocab_size()

    def save(self):
        self.prime_tokenizer.save("prime_tokenizer.json")

    def load(self):
        self.prime_tokenizer = Tokenizer.from_file("prime_tokenizer.json")


if __name__ == "__main__":
    tokenizer = PrimeTokenizer(50)
    corpus = ["00000067:solr-1.clicls[0016:adfd]", "00000064:solr-1.srvcls[0020:adfd]"]
    tokenizer.train(corpus)

    the_tokenizer_obj = tokenizer.get_tokenizer()

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=the_tokenizer_obj)
    fast_tokenizer.unk_token = "[UNK]"
    fast_tokenizer.sep_token = "[SEP]"
    fast_tokenizer.pad_token = "[PAD]"
    fast_tokenizer.cls_token = "[CLS]"
    fast_tokenizer.mask_token = "[MASK]"
    batch_encodings = fast_tokenizer(corpus, truncation=True, padding=True)
    batch = tokenizer.text_to_sequence(corpus)
    batch_ids = [list(range(10)), list(range(10))]

    data_set_tf = tf.data.Dataset.from_tensor_slices(batch_ids)

    tf_data_collator = TFDataCollatorForLanguageModeling(fast_tokenizer)
    data_collator = DataCollatorForLanguageModeling(fast_tokenizer, pad_to_multiple_of=20)

    PAD_TOKEN = 3

    output_torch = data_collator(batch_ids)

    output_tf = (
        data_set_tf
        .padded_batch(batch_size=10, padded_shapes=20, padding_values=PAD_TOKEN)
        .map(tf_data_collator)
    )

    output_tf2 = (
        data_set_tf
        .map(lambda x: x+10)
    )

    # for step, batch in enumerate(output):
    #     print(batch['input_ids'].numpy())

    # print(output._input_dataset._input_dataset)
    print(list(output_tf.as_numpy_iterator()))
    print(output_torch)
