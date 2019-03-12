import pdb

import CoreNLP_pb2
from typing import Sequence, List, Optional
from lazy import lazy
import re
import bisect

from log import warn


def _try_begin_chars(tokens, text, prev_end):
    best_token, best_offset = None, len(text) + 1
    for token in tokens:
        begin_char = text.find(token, prev_end)
        if begin_char != -1 and begin_char < best_offset:
            best_token, best_offset = token, begin_char
            # You can't get better than 0
            if best_offset == 0:
                break

    if best_token is not None:
        return best_token, best_offset
    else:
        return tokens[0], -1


class SimpleToken:
    def __init__(
            self,
            pos: str,
            original_text: str,
            lemma: str,
            ner: str = 'O',
            before: str = ' ',
            after: str = ' ',
            sentence_index: int = 0,
            token_index: int = 0,
            begin_char: int = 0,
    ):
        self.pos = pos
        self.original_text = original_text
        self.lemma = lemma
        self.ner = ner
        self.before = before
        self.after = after
        self.sentence_index = sentence_index
        self.token_index = token_index
        self.begin_char = begin_char

    def __str__(self: 'SimpleToken') -> str:
        return self.original_text

    def __repr__(self: 'SimpleToken') -> str:
        return f"<Token: {self.original_text}>"

    @classmethod
    def from_proto(cls, proto: CoreNLP_pb2.Token, sentence_index=0, token_index=0) -> 'SimpleToken':
        return cls(pos=proto.pos, original_text=proto.originalText, lemma=proto.lemma, ner=proto.ner,
                   before=proto.before, after=proto.after, sentence_index=sentence_index, token_index=token_index,
                   begin_char=proto.beginChar)

    def fill_proto(self, proto: CoreNLP_pb2.Token):
        proto.pos = self.pos
        proto.originalText = self.original_text
        proto.lemma = self.lemma
        proto.ner = self.ner
        proto.before = self.before
        proto.after = self.after
        proto.beginChar = self.begin_char
        return proto

    @classmethod
    def detokenize(cls, tokens: List['SimpleToken']) -> str:
        """
        Detokenize a list of tokens into the original sentence it came from.
        :param tokens: The tokens we are detokenizing
        :return: A plain-text string corresponding to the detokenized text
        """
        text = ''
        for i, token in enumerate(tokens):
            if i != 0:  # don't include unnecessary whitespace
                text += token.before
            text += token.original_text
        return text

    def with_fields(self,
                    pos=None,
                    original_text=None,
                    lemma=None,
                    ner=None,
                    before=None,
                    after=None):
        return SimpleToken(
            pos=pos or self.pos,
            original_text=original_text or self.original_text,
            lemma=lemma or self.lemma,
            ner=ner or self.ner,
            before=before or self.before,
            after=after or self.after,
        )


class SimpleSentence:
    def __init__(
            self,
            text: str,
            tokens: 'Sequence[SimpleToken]',
            character_offset_begin: int=0,
            token_offset_begin: int=0,
    ):
        """
        Create a simple sentence from text and tokens
        :param text: A string, representing the original text of the sentence
        :param tokens: A sequence of SimpleToken
        :param character_offset_begin: Index of the first character in context of the entire document.
        :param token_offset_begin: Index of the first token in context of the entire document.
        """
        self.text = text
        self.tokens = tokens
        self.character_offset_begin = character_offset_begin
        self.token_offset_begin = token_offset_begin

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"<SimpleSentence: {self.text}>"

    def __lt__(self, other):
        if not isinstance(other, SimpleSentence):
            raise ValueError(f"Cannot compare SimpleSentence with something not a SimpleSentence")
        return self.text < other.text

    def __len__(self):
        """
        :return: The number of tokens in the sentence
        """
        return len(self.tokens)

    def __eq__(self, other):
        """
        Compares only the stored text! Be careful.
        """
        if not isinstance(other, SimpleSentence):
            return False
        else:
            return self.text == other.text

    def __hash__(self):
        """
        Hashes only the stored text! Be careful.
        :return:
        """
        return self.text.__hash__()

    def fill_proto(self, proto: CoreNLP_pb2.Sentence) -> CoreNLP_pb2.Sentence:
        for token in self.tokens:
            token_proto = proto.token.add()
            token.fill_proto(token_proto)
        proto.tokenOffsetBegin = self.token_offset_begin
        proto.tokenOffsetEnd = self.token_offset_begin + len(self.tokens)
        proto.characterOffsetBegin = self.character_offset_begin
        return proto

    @lazy
    def original_texts(self) -> List[str]:
        """
        Returns the tokens of the sentence as a list of each of their original text string. Works the same as its
        java counterpart.
        :return:
        """
        return [token.original_text for token in self.tokens]

    @lazy
    def lemmas(self) -> List[str]:
        """
        A list of Lemmas corresponding to each token of the sentence. Defaults to the original text if no lemmas
        were originally specified.
        :return:
        """
        return [token.lemma for token in self.tokens]

    @lazy
    def pos(self) -> List[str]:
        """
        A list of POS corresponding to each token of the sentence. Defaults to '' if no pos tag was initially given.
        :return:
        """
        return [token.pos for token in self.tokens]

    @lazy
    def _token_begin_indices(self) -> List[int]:
        return [tok.begin_char - self.character_offset_begin for tok in self.tokens]

    def char_idx_to_token_idx(self, char_idx: int, whitespace_before: bool=False):
        """
        Given the index of a character of the original text of the sentence. Find the index of the closest token.
        :param char_idx: The character index
        :param whitespace_before: By default (False), white space after a token belongs to that token with exception of
                                 the first token, where white space before and after will map to the first token.
                                 Otherwise (True), whitespace before a token belongs to that token with the exception of
                                 the final token.
        :return: the index of the token the character belongs to.
        """
        begin_indices = self._token_begin_indices()
        ans = max(bisect.bisect_right(begin_indices, char_idx) - 1, 0)
        if whitespace_before:
            if 0 < ans < len(self) - 1 and self.text[char_idx].isspace():
                return ans + 1
        return ans

    @classmethod
    def from_proto(cls, proto: CoreNLP_pb2.Sentence) -> 'SimpleSentence':
        """
        Creates a SimpleSentence given a proto'd CoreNLP Sentence object.
        :param proto: The corresponding proto object. Note that this is a proto object not the raw proto bytes.
        :return: A SimpleSentence
        """
        tokens = []
        for i in range(len(proto.token)):
            tokens.append(SimpleToken.from_proto(proto.token[i], proto.sentenceIndex, i))
        text = proto.text
        return cls(
            text=text,
            tokens=tokens,
            character_offset_begin=proto.characterOffsetBegin,
            token_offset_begin=proto.tokenOffsetBegin
        )

    @classmethod
    def from_text(
            cls, text: str, sentence_index: int=0, token_offset: int=0, character_offset: int=0
    ) -> 'SimpleSentence':
        """
        Creates a SimpleSentence given a string of text where tokens are separated by whitespace. Lemmas default
        to the original text of the token. NER defaults to 'O'. And pos defaults to ''.
        :param text: A string where tokens are separated by whitespace.
        :param sentence_index: Index of the sentence in the document. Defaults to 0
        :param token_offset: Index of the first token in the sentence in terms of all the tokens in the document.
                             Defaults to 0
        :param character_offset: Index of the first character in the sentence in the overall document. Defaults to 0.
        :return: A SimpleSentence
        """
        tokens = []
        prev_end = 0
        matches = list(re.finditer(r"\S+", text))
        for i, match in enumerate(matches):
            curr_token = match.group()
            begin_char = match.start()
            before = text[prev_end:begin_char]
            after = text[match.end():matches[i+1].start()] if i + 1 < len(matches) else text[match.end():]
            prev_end = match.end()
            tokens.append(
                SimpleToken(
                    pos='', original_text=curr_token, lemma=curr_token, ner='O', before=before, after=after,
                    sentence_index=sentence_index, token_index=token_offset + i,
                    begin_char=begin_char + character_offset
                )
            )

        return SimpleSentence(
            text=text,
            tokens=tokens,
            character_offset_begin=character_offset,
            token_offset_begin=token_offset,
        )

    @classmethod
    def from_text_tokens(
            cls,
            text: str,
            token_strings: Sequence[str],
            sentence_index: int=0,
            token_offset: int=0,
            character_offset: int=0,
    ) -> 'SimpleSentence':
        """
        Creates a SimpleSentence from text and a list of tokens. Unlike SimpleSentence.from_text, this method does
        not guess the tokens based off of the input text.
        This tries to guess the begin_char, before, and, after token fields by searching through the text. So make sure
        that the text actually contains the tokens!
        :param text: The original text of the sentence.
        :param token_strings: A list of strings representing the original texts of each token.
        :param sentence_index: Index of the sentence in the document. Defaults to 0
        :param token_offset: Index of the first token in the sentence in terms of all the tokens in the document.
                             Defaults to 0
        :param character_offset: Index of the first character in the sentence in the overall document. Defaults to 0.
        :return: A SimpleSentence
        """
        SPECIAL_TOKENS = [
            ("-LRB-", ("(",)),
            ("-RRB-", (")",)),
            ("-LSB-", ("[",)),
            ("-RSB-", ("]",)),
            ("-LCB-", ("{",)),
            ("-RCB-", ("}",)),
            ("``", ["''", "\"", "\u201C", "\u201D", "\u201F"]),
            ("''", ["\"", "\u201C", "\u201D", "\u201F"]),
            ("'", ["\u2018", "\u2019", "\u201B", "\u00b4"]),
            ("`", ["'", "\u2018", "\u2019", "\u201B"]),
            ("\"", ["\u201C", "\u201D", "\u201F"]),
            ("\u00a0", [" "]),
            ("...", [".. .", ". ..", ". . .", "\u2026"]),
            ("--", "\u0096\u0097\u2013\u2014\u2015"),
        ]

        tokens = []
        prev_end = 0
        token_beginnings = []
        for token in token_strings:
            if token == "%%SOS%%" or token == "%%EOS%%":
                # These are zero-width characters, so move on.
                token_beginnings.append(prev_end)
                continue

            begin_char = text.find(token, prev_end)
            if begin_char < 0:
                warn(f"I could not find the exact token '{token}' so I am going to try some rewrites."
                     " If you value your life, please give me honest-to-god original_texts")

                # CoreNLP can add terminal '.' some times.
                if token == "." and prev_end == len(text):
                    token_beginnings.append(prev_end)
                    continue

                # Handle special characters
                # These are ambiguous maps so choose the closest one. GOD DAMN YOU.
                choices = [token] +\
                          [token.replace(char, choice)
                           for char, choices in SPECIAL_TOKENS for choice in choices if char in token]

                token, begin_char = _try_begin_chars(choices, text, prev_end)

                if begin_char == -1:
                    raise ValueError("Cannot find token in original text")

            token_beginnings.append(begin_char)
            prev_end = begin_char + len(token)

        prev_end = 0
        for idx, token in enumerate(token_strings):
            begin_char = token_beginnings[idx]
            before = text[prev_end: begin_char]
            prev_end = begin_char + len(token)
            after = text[prev_end:token_beginnings[idx + 1]] if idx + 1 < len(token_strings) else text[prev_end:]

            tokens.append(
                SimpleToken(
                    pos='', original_text=token, lemma=token, ner='O', before=before, after=after,
                    sentence_index=sentence_index, token_index=token_offset + idx,
                    begin_char=begin_char + character_offset
                )
            )
        return SimpleSentence(
            text=text,
            tokens=tokens,
            character_offset_begin=character_offset,
            token_offset_begin=token_offset,
        )

    @classmethod
    def from_annotated_text(
            cls,
            annotated_text: str,
    ) -> 'SimpleSentence':
        """
        Creates a SimpleSentence from annotated text that looks like: word::lemma::pos
        :return: A SimpleSentence
        """
        annotated_tokens = annotated_text.split()
        tokens = []
        for i, token_spec in enumerate(annotated_tokens):
            if token_spec == ":" * 9 + "O":  # Special case!
                original_text, lemma, pos, ner = ":", ":", ":", "O"
            else:
                original_text, lemma, pos, ner = token_spec.split("::")
            token = SimpleToken(
                pos,
                original_text,
                lemma,
                ner, # ner
                ' ', # space before
                ' ', # space after
                0,   # sentence_begin
                i,  # token_index
                -1, # begin_char
            )
            tokens.append(token)
        return SimpleSentence(" ".join([t.original_text for t in tokens]), tokens)

    @classmethod
    def from_json(
            cls,
            json: dict,
    ) -> 'SimpleSentence':
        """
        Creates a SimpleSentence from a json blob that's like:
         {"text": str,
         "tokens": List[str],
         "lemmas": List[str],
         "pos_tags": List[str],
         "ner": List[str],
         }
        :return: A SimpleSentence
        """
        tokens = [SimpleToken(
            original_text=token,
            pos=pos,
            lemma=lemma,
            ner=ner,
            token_index=i)
            for i, (token, pos, lemma, ner) in enumerate(zip(
                json["tokens"], json["lemmas"], json["pos_tags"], json["ner"]))]
        return SimpleSentence(json["text"], tokens)


class SimpleDocument:
    """
    A mirror of CoreNLP's document (which is basically a list of sentences!)
    """
    def __init__(
            self,
            text: str,
            sentences: List[SimpleSentence],
    ):
        self.text = text
        self.sentences = sentences

    def fill_proto(self, proto: CoreNLP_pb2.Document) -> CoreNLP_pb2.Document:
        proto.text = self.text
        for sent in self.sentences:
            sent_proto = proto.sentence.add()
            sent.fill_proto(sent_proto)
        return proto

    @classmethod
    def from_proto(cls, proto: CoreNLP_pb2.Document) -> 'SimpleDocument':
        """
        Creates a SimpleDocument from a proto definition
        """
        sentences = []
        for sentence_proto in proto.sentence:
            sentences.append(SimpleSentence.from_proto(sentence_proto))
        text = proto.text
        return SimpleDocument(text, sentences)

    @classmethod
    def from_text(cls, text: str):
        """
        Constructs a very minimal document consisting of one SimpleSentence which splits the input text by spaces.
        :param text: Input document text
        :return: A SimpleDocument object
        """
        sent = SimpleSentence.from_text(text)
        return SimpleDocument(text, [sent])
