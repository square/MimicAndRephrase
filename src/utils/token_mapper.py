from typing import List, Sequence, Dict
from abc import ABC, abstractmethod
import re
import TensorflowModel_pb2 as proto


def simple_hash(token: str, output_size: int) -> int:
    """
    This runs a simple, deterministic hash function on the input token. This needs to be exactly duplicated in Java, and
    is very sensitive. In general, don't change this function.
    
    :param token: the token to hash
    :param output_size: the output size to mod into
    :return: a number in [0, output_size)
    """
    encoded = token.encode("utf-8")
    hash_sum = 0
    for letter in encoded:
        hash_sum = ((31 * hash_sum) + letter) % output_size
    return hash_sum


class TokenMapping(ABC):
    """
    This holds a single function to map tokens. Most commonly, this is a single regex pattern (which can hash to 
    different entries), though it's an abstract class so that we can implement future TokenMappings as we think of them.
    """

    @abstractmethod
    def match(self, token: str) -> bool:
        """
        :param token: the token we're interested in mapping
        :return: True if this TokenMapping matches this token. False otherwise.
        """
        pass

    @abstractmethod
    def map(self, token: str) -> int:
        """
        :param token: the token we're interested in mapping
        :return: the offset into the list of entries this TokenMapping contains, if any. Returns -1 when token doesn't
                 match this TokenMapping
        """
        pass

    @abstractmethod
    def output_size(self) -> int:
        """
        :return: the number of entries this mapping maps to
        """
        pass

    @abstractmethod
    def debug_token(self, offset: int) -> str:
        """
        :param offset: the offset into this TokenMapping we're interested in debugging
        :return: the name of this token mapping entry, for debugging and display.
        """
        pass

    @abstractmethod
    def serialize(self, serialized: proto.TokenMapping) -> proto.TokenMapping:
        """
        This returns the proto'd up version of this TokenMapping
        """
        pass


class RegexTokenMapping(TokenMapping):
    """
    This holds a mapping for a single regex pattern. It can optionally hash tokens that match the regex pattern out onto
    multiple buckets, so that attention mechanisms can distinguish between (for example) different numbers, emails, etc
    within the same class of tokens.
    """
    def __init__(self, regex: str, num_hash: int, debug_base: str):
        """
        :param regex: the regex pattern we're mapping
        :param num_hash: the number of outputs to hash onto if we match this regex
        :param debug_base: the base of the debug string, before we add the hash number to the token
        """
        self.regex = re.compile(regex)
        self.num_hash = num_hash
        self.debug_base = debug_base

    def match(self, token: str):
        return self.regex.match(token) is not None

    def map(self, token: str) -> int:
        return simple_hash(token, self.num_hash)

    def output_size(self) -> int:
        return self.num_hash

    def debug_token(self, offset: int):
        return self.debug_base+'$'+str(offset)

    def serialize(self, serialized: proto.TokenMapping):
        serialized.type = proto.REGEX
        serialized.regex = self.regex
        serialized.num_hash = self.num_hash
        serialized.debug_base = self.debug_base
        return serialized


class HashTokenMapping(TokenMapping):
    """
    This is a good thing to use with unks. It matches any input token, and hashes everything to the same output space.
    We lowercase all tokens before mapping to improve case insensitivity.
    """
    def __init__(self, num_hash: int):
        self.num_hash = num_hash

    def match(self, token: str):
        return True

    def map(self, token: str):
        return simple_hash(token.lower(), self.num_hash)

    def output_size(self):
        return self.num_hash

    def debug_token(self, offset: int):
        return "HASH$"+str(offset)

    def serialize(self, serialized: proto.TokenMapping):
        serialized.type = proto.HASH
        serialized.num_hash = self.num_hash
        return serialized


class ExactTokenMapping(TokenMapping):
    """
    Matches tokens exactly. This is good for training a subset of GloVe's vocabulary.
    """
    def __init__(self, vocab: Sequence[str]):
        self.tokens = list(vocab)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}

    def match(self, token: str):
        return token in self.token_to_idx

    def map(self, token: str):
        return self.token_to_idx[token]

    def output_size(self):
        return len(self.tokens)

    def debug_token(self, offset: int):
        return self.tokens[offset]

    def serialize(self, serialized: proto.TokenMapping):
        serialized.type = proto.TOKEN
        serialized.tokens.extend(self.tokens)


class TokenMapper:
    def __init__(self, mappings: List[TokenMapping], unk_mappings: List[TokenMapping]):
        """
        This constructs a TokenMapper, which takes a list of regexes that we'll use to parse incoming tokens, and a
        number of unks to allocate.

        :param mappings: If any of these match, they will override any underlying word with a vector for this pattern
        :param unk_mappings: These patterns are used to match anything that doesn't match the original mappings, and is
                             out of the regular vocabulary.
        """
        self.mappings = mappings
        self.unk_mappings = unk_mappings

    def mapped_output_size(self) -> int:
        """
        :return: the total size of the output space that we've mapped for this TokenMapper.
        """
        return sum([mapping.output_size() for mapping in self.unk_mappings + self.mappings])

    def map_token(self, token: str) -> int:
        """
        This attempts to map a token to one of the special mappings we have in this TokenMapper. The first mapping that
        triggers wins ties. If no mappings fire, this returns -1.

        :param token: the token to map
        :return: an offset into the output matrix, or -1 if no match
        """
        offset = sum([mapping.output_size() for mapping in self.unk_mappings])
        for mapping in self.mappings:
            if mapping.match(token):
                return offset + mapping.map(token)
            offset += mapping.output_size()
        return -1

    def map_unk(self, unk: str) -> int:
        """
        This attempts to map a token after it has failed to map to any previous mappings or to anything in the vocab.

        :param unk: the unk to map
        :return: an offset into the output matrix. Failing to match any mappings is actually an invalid state.
        """
        offset = 0
        for mapping in self.unk_mappings:
            if mapping.match(unk):
                return offset + mapping.map(unk)
            offset += mapping.output_size()
        # We should never reach here
        assert False

    def is_unk(self, mapped: int) -> bool:
        """
        This returns true if the integer we've mapped to is actually an UNK
        
        :param mapped: 
        :return: 
        """
        offset = sum([mapping.output_size() for mapping in self.unk_mappings])
        return mapped < offset

    def debug_token(self, token: int) -> str:
        """
        This takes a token offset into the token_mapper and returns a human-readable string for what that offset means.

        :param token: the offset we'd like to understand
        :return: a human-readable debug string
        """
        offset = token
        for mapping in self.unk_mappings + self.mappings:
            if offset < mapping.output_size():
                return mapping.debug_token(offset)
            offset -= mapping.output_size()
        return 'OOB'

    def state_dict(self) -> Dict:
        """
        Output a configuration dict (pickable) for this token mapper
        :return:
        """
        return {
            "mappings": self.mappings,
            "unk_mappings": self.unk_mappings,
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict) -> 'TokenMapper':
        """
        Load token mapper from a state dict.
        :param state_dict:
        :return:
        """
        return TokenMapper(state_dict["mappings"], state_dict["unk_mappings"])


def default_token_mapper(unk_classes: int = 10):
    return TokenMapper([
        RegexTokenMapping("^[0-9]$", 3, "NUM_1"),
        RegexTokenMapping("^[0-9]{2}$", 3, "NUM_2"),
        RegexTokenMapping("^[0-9]{3}$", 3, "NUM_3"),
        RegexTokenMapping("^[0-9]+$", 3, "NUM_MANY"),
        RegexTokenMapping("^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", 3, "EMAIL")
    ], [
        HashTokenMapping(unk_classes)
    ])
