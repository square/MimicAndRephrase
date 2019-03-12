import os
import struct
import sys
import unicodedata
import props
from typing import List, Sequence

import numpy as np

from token_mapper import TokenMapper, default_token_mapper
from log import info, warn, track


class Glove:
    def __init__(
            self,
            token_mapper: TokenMapper,
            glove_name: str,
            embedding_dim: int,
            vocab: Sequence[str],
            numbers: np.ndarray,
            numbers_is_zero_padded=False,
    ):
        def create_vocab_dict(vocab_list):
            vocab_dict = dict.fromkeys(vocab_list)
            for i, word in enumerate(vocab_dict):
                vocab_dict[word] = i
            return vocab_dict

        self.token_mapper = token_mapper
        self._mapped_output_size = self.token_mapper.mapped_output_size()  # cache result for speed
        self.glove_name = glove_name
        self.embedding_dim = embedding_dim
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.vocab_dict = create_vocab_dict(self.vocab)
        self.numbers = numbers

        if not numbers_is_zero_padded:
            # Provide a 0 padding at the beginning because it makes it much easier on the pytorch end
            # The shape and layout of self.numbers is important for the pytorch GloveEmbedding layer to work correctly,
            # so be careful about changing / make sure TestGloveEmbedding passes.
            padded_numbers = np.zeros(shape=(self.numbers.shape[0] + 1, self.numbers.shape[1]), dtype=np.float32)
            padded_numbers[1:] = self.numbers
            self.numbers = padded_numbers

        assert self.numbers.shape[1] == self.embedding_dim, "Embedding dim must be same as the last dimension of numbers matrix"
        assert self.numbers.shape[0] == self.vocab_size + 1, "Numbers matrix is not the correct size."

    @classmethod
    def mock(cls):
        return Glove(
            token_mapper=default_token_mapper(),
            glove_name="mock_glove",
            embedding_dim=300,
            vocab=["the", "A", "B"],
            numbers=np.random.rand(3, 300),
            numbers_is_zero_padded=False
        )

    @classmethod
    def from_binary(
            cls,
            embedding_file:str = None,  # Default filled in below -- want to load it statically.
            token_mapper: TokenMapper=None
    ) -> 'Glove':
        """
        Just responsible for reading binary glove and storing basic parameters
        :param embedding_file: the raw, original format Glove embeddings
        :param token_mapper: the token mapper to use, if None, it defaults to nn.special_token_mapper
        """
        if not embedding_file:
            embedding_file = props.auto.GLOVE_PATH
        embedding_file = embedding_file
        token_mapper = default_token_mapper() if token_mapper is None else token_mapper
        glove_name = os.path.splitext(os.path.basename(embedding_file))[0]
        with track('Loading the embeddings from binary file ({})'.format(embedding_file)):
            def read_int(f):
                return struct.unpack('>i', f.read(4))[0]

            def read_bytes(f, num_bytes):
                chunks = []
                while num_bytes > 0:
                    to_read = min(10000000, num_bytes)
                    chunk = f.read(to_read)
                    if len(chunk) == 0:
                        raise Exception('Got an empty chunk back! File terminated before expected! Still need '+str(num_bytes)+' bytes')
                    num_bytes -= len(chunk)
                    chunks.append(chunk)
                return b''.join(chunks)

            def read_vocab(characters, word_begins, word_lengths):
                vocab = []
                for begin, length in zip(word_begins, word_lengths):
                    word = ''.join([chr(c) for c in characters[begin:begin+length]])
                    vocab.append(word)
                return vocab

            with open(embedding_file, 'rb') as f:
                embedding_dim = read_int(f)
                vocab_size = read_int(f)
                num_characters = read_int(f)
                characters = np.ndarray((num_characters,), '>i', read_bytes(f, num_characters * 4))
                num_word_begins = read_int(f)
                word_begins = np.ndarray((num_word_begins,), '>i', read_bytes(f, num_word_begins * 4))
                num_word_lengths = read_int(f)
                word_lengths = np.ndarray((num_word_lengths,), 'B', read_bytes(f, num_word_lengths))
                num_numbers = read_int(f)
                assert(num_numbers == num_word_lengths * 300)
                sys.stdout.flush()
                numbers = np.ndarray((num_numbers,), '>f', read_bytes(f, num_numbers * 4))
                unk = np.ndarray((embedding_dim,), '>f', read_bytes(f, embedding_dim * 4))
                numbers = np.reshape(numbers, [vocab_size, embedding_dim])
                numbers = numbers.astype(np.float32)
                vocab = read_vocab(characters, word_begins, word_lengths)

            info('Embedding dim: '+str(embedding_dim))
            info('Vocab size: '+str(vocab_size))
            # info('read start of characters array: '+str(characters))

        assert len(vocab) == vocab_size, "Length of vocabulary does not match state vocab size in {}".format(embedding_file)

        return cls(
            token_mapper=token_mapper,
            glove_name=glove_name,
            embedding_dim=embedding_dim,
            vocab=vocab,
            numbers=numbers,
        )

    def with_new_token_mapper(self, token_mapper: TokenMapper, new_name: str=None) -> 'Glove':
        """
        Returns a new Glove that shares memory with this Glove that uses a different token mapper.
        :param token_mapper: The new token mapper to use
        :param new_name: (Optional) If given, we will write this new name to this glove
        :return:
        """
        return Glove(
            token_mapper,
            new_name if new_name else self.glove_name,
            self.embedding_dim,
            self.vocab,
            self.numbers,
            numbers_is_zero_padded=True
        )

    def lookup_word(self, word: str) -> int:
        """
        This searches through word indices to arrive at the index for the correct word index quickly.

        :param word: the word to search for
        :return: the index of the word
        """
        mapped_index = self.token_mapper.map_token(word)
        if mapped_index >= 0:
            return mapped_index
        if word in self.vocab_dict:
            index = self.vocab_dict[word]
            return index + self.token_mapper.mapped_output_size()
        else:
            return self.token_mapper.map_unk(word)

    def lookup_unk(self, word: str) -> int:
        """
        Returns the unk embedding index for any word
        :param word: The word
        :return: An integer index
        """
        return self.token_mapper.map_unk(word)

    def get_word_at_index(self, index: int) -> str:
        """
        This retrieves the word at the given index in our database

        :param index:
        :return:
        """
        if index < self.token_mapper.mapped_output_size():
            return self.token_mapper.debug_token(index)
        index -= self.token_mapper.mapped_output_size()
        return self.vocab[index]

    def words_to_indices(self, words: Sequence[str]) -> List[int]:
        """
        Returns a list of integer indices corresponding to each word in words. Unknown words will be replaced with UNK
        :param words: (list(string)) words
        :return: (list(int)) indices
        """
        return [self.lookup_word(token) for token in words]

    def indices_to_words(self, indices: Sequence[int]) -> List[str]:
        """
        Converts a list of indices to their corresponding words in Glove
        :param indices: (list(int)) indices
        :return: (list(string)) words
        """
        return [self.get_word_at_index(index) for index in indices]

    def get_embedding_at_index(self, index: int) -> np.ndarray:
        """
        This gets an embedding at a given index in our structures.

        :param index: the word index to lookup
        :return: the embedding
        """
        if index < self.token_mapper.mapped_output_size():
            return np.zeros((self.embedding_dim,))
        else:
            return self.numbers[index - self.token_mapper.mapped_output_size() + 1]

    def __len__(self) -> int:
        return self.vocab_size + self.token_mapper.mapped_output_size()

    def __iter__(self):
        return zip(self.vocab, range(self.vocab_size))

    def __contains__(self, key) -> bool:
        if type(key) == int:
            return 0 <= key < len(self)
        elif type(key) == str:
            mapped_index = self.token_mapper.map_token(key)
            if mapped_index >= 0:
                return True
            return key in self.vocab_dict

    def __getitem__(self, key):
        if type(key) == int:
            return self.get_word_at_index(key)
        if type(key) == str:
            return self.lookup_word(self.normalize_word(key))

    def __setitem__(self, key, item):
        warn('Should not be setting item in embedding: %s' % key)

    def add(self, token):
        warn('Should not be adding item to embedding: %s' % token)

    def tokens(self) -> Sequence[str]:
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        return self.vocab

    @staticmethod
    def normalize_word(token: str):
        return unicodedata.normalize('NFD', token)


if __name__ == "__main__":
    g = Glove.from_binary()


class TunedBinaryGlove:
    """
    The point of this class is to wrap a Glove, and allow certain custom words to be overwritten with new 
    embeddings without any disruption downstream.
    """
    def __init__(self, glove: Glove):
        self.glove = glove
        self.custom_words_vectors = {}
        self.custom_words_indices = {}
        self.custom_words_back = {}
        self.custom_unk = None

    def set_custom_word(self, word: str, vector: np.ndarray):
        """
        This adds a custom word to the dataset, which will overwrite any words previously listed
        
        :param word: the word to add
        :param vector: the vector to associate with this word
        """
        if word not in self.custom_words_indices:
            custom_index = len(self.custom_words_vectors) + len(self.glove)
            self.custom_words_indices[word] = custom_index
        else:
            custom_index = self.custom_words_indices[word]
        self.custom_words_vectors[custom_index] = vector
        self.custom_words_back[custom_index] = word

    def set_custom_unk(self, vector: np.ndarray):
        """
        If we want to override the UNK behavior with a new value, this is it.
        
        :param vector: 
        """
        self.custom_unk = vector

    def lookup_word(self, word: str) -> int:
        """
        This searches through word indices to arrive at the index for the correct word index quickly.

        :param word: the word to search for
        :return: the index of the word
        """
        if word in self.custom_words_indices:
            return self.custom_words_indices[word]
        index = self.glove.lookup_word(word)
        return index

    def get_word_at_index(self, index: int) -> str:
        """
        This retrieves the word at the given index in our database

        :param index:
        :return:
        """
        if index in self.custom_words_back:
            return self.custom_words_back[index]
        return self.glove.get_word_at_index(index)

    def words_to_indices(self, words: Sequence[str]) -> List[int]:
        """
        Returns a list of integer indices corresponding to each word in words. Unknown words will be replaced with UNK
        :param words: (list(string)) words
        :return: (list(int)) indices
        """
        return [self.lookup_word(token) for token in words]

    def indices_to_words(self, indices: Sequence[int]) -> List[str]:
        """
        Converts a list of indices to their corresponding words in Glove
        :param indices: (list(int)) indices
        :return: (list(string)) words
        """
        return [self.get_word_at_index(index) for index in indices]

    def get_embedding_at_index(self, index: int) -> np.ndarray:
        """
        This gets an embedding at a given index in our structures.

        :param index: the word index to lookup
        :return: the embedding
        """
        if index in self.custom_words_vectors:
            return self.custom_words_vectors[index]
        if self.glove.token_mapper.is_unk(index) and self.custom_unk is not None:
            return self.custom_unk
        return self.glove.get_embedding_at_index(index)

    def __len__(self) -> int:
        return len(self.glove) + len(self.custom_words_indices)

    def __iter__(self):
        return self.glove.__iter__()

    def __contains__(self, key) -> bool:
        if key in self.custom_words_indices: return True
        return self.glove.__contains__(key)

    def __getitem__(self, key):
        if type(key) == int:
            return self.get_word_at_index(key)
        if type(key) == str:
            return self.lookup_word(self.glove.normalize_word(key))
        return self.glove.__getitem__(key)

    def __setitem__(self, key, item):
        warn('Should not be setting item in embedding: %s' % key)

    def add(self, token):
        warn('Should not be adding item to embedding: %s' % token)

    def tokens(self) -> Sequence[str]:
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        return list(self.glove.vocab) + list(self.custom_words_indices.keys())
