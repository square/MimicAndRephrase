"""
A Vocabulary maps strings to integers, allowing for strings to be mapped to an
out-of-vocabulary token.
"""

import codecs
from log import info, error, warn
import os
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

from elmo.common.util import namespace_match
from elmo.common import Params, Registrable
from elmo.common.checks import ConfigurationError
from elmo.data import instance as adi  # pylint: disable=unused-import


DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'


class _NamespaceDependentDefaultDict(defaultdict):
    """
    This is a `defaultdict
    <https://docs.python.org/2/library/collections.html#collections.defaultdict>`_ where the
    default value is dependent on the key that is passed.

    We use "namespaces" in the :class:`Vocabulary` object to keep track of several different
    mappings from strings to integers, so that we have a consistent API for mapping words, tags,
    labels, characters, or whatever else you want, into integers.  The issue is that some of those
    namespaces (words and characters) should have integers reserved for padding and
    out-of-vocabulary tokens, while others (labels and tags) shouldn't.  This class allows you to
    specify filters on the namespace (the key used in the ``defaultdict``), and use different
    default values depending on whether the namespace passes the filter.

    To do filtering, we take a set of ``non_padded_namespaces``.  This is a set of strings
    that are either matched exactly against the keys, or treated as suffixes, if the
    string starts with ``*``.  In other words, if ``*tags`` is in ``non_padded_namespaces`` then
    ``passage_tags``, ``question_tags``, etc. (anything that ends with ``tags``) will have the
    ``non_padded`` default value.

    Parameters
    ----------
    non_padded_namespaces : ``Iterable[str]``
        A set / list / tuple of strings describing which namespaces are not padded.  If a namespace
        (key) is missing from this dictionary, we will use :func:`namespace_match` to see whether
        the namespace should be padded.  If the given namespace matches any of the strings in this
        list, we will use ``non_padded_function`` to initialize the value for that namespace, and
        we will use ``padded_function`` otherwise.
    padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a namespace that `should` be
        padded.
    non_padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a namespace that should `not` be
        padded.
    """
    def __init__(self,
                 non_padded_namespaces: Iterable[str],
                 padded_function: Callable[[], Any],
                 non_padded_function: Callable[[], Any]) -> None:
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super(_NamespaceDependentDefaultDict, self).__init__()

#    def __missing__(self, key: str):
#        if any(namespace_match(pattern, key) for pattern in self._non_padded_namespaces):
#            value = self._non_padded_function()
#        else:
#            value = self._padded_function()
#        dict.__setitem__(self, key, value)
#        print("_NamespaceDependentDefaultDict: __missing__", flush=True)
#        return value

    def add_non_padded_namespaces(self, non_padded_namespaces: Set[str]):
        # add non_padded_namespaces which weren't already present
        self._non_padded_namespaces.update(non_padded_namespaces)


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super(_TokenToIndexDefaultDict, self).__init__(non_padded_namespaces,
                                                       lambda: {padding_token: 0, oov_token: 1},
                                                       lambda: {})


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super(_IndexToTokenDefaultDict, self).__init__(non_padded_namespaces,
                                                       lambda: {0: padding_token, 1: oov_token},
                                                       lambda: {})


def pop_max_vocab_size(params: Params) -> Union[int, Dict[str, int]]:
    """
    max_vocab_size is allowed to be either an int or a Dict[str, int] (or nothing).
    but it could also be a string representing an int (in the case of environment variable
    substitution). So we need some complex logic to handle it.
    """
    size = params.pop("max_vocab_size", None)

    if isinstance(size, Params):
        # This is the Dict[str, int] case.
        return size.as_dict()
    elif size is not None:
        # This is the int / str case.
        return int(size)
    else:
        return None


class Vocabulary(Registrable):
    """
    a vocabulary maps strings to integers, allowing for strings to be mapped to an
    out-of-vocabulary token.

    vocabularies are fit to a particular dataset, which we use to decide which tokens are
    in-vocabulary.

    vocabularies also allow for several different namespaces, so you can have separate indices for
    'a' as a word, and 'a' as a character, for instance, and so we can use this object to also map
    tag and label strings to indices, for a unified :class:`~.fields.field.Field` API.  Most of the
    methods on this class allow you to pass in a namespace; by default we use the 'tokens'
    namespace, and you can omit the namespace argument everywhere and just use the default.

    parameters
    ----------
    counter : ``Dict[str, Dict[str, int]]``, optional (default=``None``)
        a collection of counts from which to initialize this vocabulary.  We will examine the
        counts and, together with the other parameters to this class, use them to decide which
        words are in-vocabulary.  If this is ``None``, we just won't initialize the vocabulary with
        anything.
    min_count : ``Dict[str, int]``, optional (default=None)
        when initializing the vocab from a counter, you can specify a minimum count, and every
        token with a count less than this will not be added to the dictionary.  These minimum
        counts are `namespace-specific`, so you can specify different minimums for labels versus
        words tokens, for example.  If a namespace does not have a key in the given dictionary, we
        will add all seen tokens to that namespace.
    max_vocab_size : ``Union[int, Dict[str, int]]``, optional (default=``None``)
        if you want to cap the number of tokens in your vocabulary, you can do so with this
        parameter.  If you specify a single integer, every namespace will have its vocabulary fixed
        to be no larger than this.  If you specify a dictionary, then each namespace in the
        ``counter`` can have a separate maximum vocabulary size.  Any missing key will have a value
        of ``None``, which means no cap on the vocabulary size.
    non_padded_namespaces : ``Iterable[str]``, optional
        by default, we assume you are mapping word / character tokens to integers, and so you want
        to reserve word indices for padding and out-of-vocabulary tokens.  However, if you are
        mapping NER or SRL tags, or class labels, to integers, you probably do not want to reserve
        indices for padding and out-of-vocabulary tokens.  Use this field to specify which
        namespaces should `not` have padding and OOV tokens added.

        the format of each element of this is either a string, which must match field names
        exactly,  or ``*`` followed by a string, which we match as a suffix against field names.

        we try to make the default here reasonable, so that you don't have to think about this.
        the default is ``("*tags", "*labels")``, so as long as your namespace ends in "tags" or
        "labels" (which is true by default for all tag and label fields in this code), you don't
        have to specify anything here.
    pretrained_files : ``Dict[str, str]``, optional
        if provided, this map specifies the path to optional pretrained embedding files for each
        namespace. This can be used to either restrict the vocabulary to only words which appear
        in this file, or to ensure that any words in this file are included in the vocabulary
        regardless of their count, depending on the value of ``only_include_pretrained_words``.
        words which appear in the pretrained embedding file but not in the data are NOT included
        in the Vocabulary.
    only_include_pretrained_words : ``bool``, optional (default=False)
        this defines the stategy for using any pretrained embedding files which may have been
        specified in ``pretrained_files``. If False, an inclusive stategy is used: and words
        which are in the ``counter`` and in the pretrained file are added to the ``Vocabulary``,
        regardless of whether their count exceeds ``min_count`` or not. If True, we use an
        exclusive strategy: words are only included in the Vocabulary if they are in the pretrained
        embedding file (their count must still be at least ``min_count``).
    tokens_to_add : ``Dict[str, List[str]]``, optional (default=None)
        if given, this is a list of tokens to add to the vocabulary, keyed by the namespace to add
        the tokens to.  This is a way to be sure that certain items appear in your vocabulary,
        regardless of any other vocabulary computation.
    """
    def __init__(self,
                 counter: Dict[str, Dict[str, int]] = None,
                 min_count: Dict[str, int] = None,
                 max_vocab_size: Union[int, Dict[str, int]] = None,
                 non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
                 pretrained_files: Optional[Dict[str, str]] = None,
                 only_include_pretrained_words: bool = False,
                 tokens_to_add: Dict[str, List[str]] = None) -> None:
        self._padding_token = DEFAULT_PADDING_TOKEN
        self._oov_token = DEFAULT_OOV_TOKEN
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._token_to_index = _TokenToIndexDefaultDict(self._non_padded_namespaces,
                                                        self._padding_token,
                                                        self._oov_token)
        self._index_to_token = _IndexToTokenDefaultDict(self._non_padded_namespaces,
                                                        self._padding_token,
                                                        self._oov_token)
        self._retained_counter: Optional[Dict[str, Dict[str, int]]] = None
        # Made an empty vocabulary, now extend it.
        self._extend(counter,
                     min_count,
                     max_vocab_size,
                     non_padded_namespaces,
                     pretrained_files,
                     only_include_pretrained_words,
                     tokens_to_add)

    def save_to_files(self, directory: str) -> None:
        """
        persist this Vocabulary to files so it can be reloaded later.
        each namespace corresponds to one file.

        parameters
        ----------
        directory : ``str``
            The directory where we save the serialized vocabulary.
        """
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            warn("vocabulary serialization directory %s is not empty", directory)

        with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'w', 'utf-8') as namespace_file:
            for namespace_str in self._non_padded_namespaces:
                print(namespace_str, file=namespace_file)

        for namespace, mapping in self._index_to_token.items():
            # Each namespace gets written to its own file, in index order.
            with codecs.open(os.path.join(directory, namespace + '.txt'), 'w', 'utf-8') as token_file:
                num_tokens = len(mapping)
                start_index = 1 if mapping[0] == self._padding_token else 0
                for i in range(start_index, num_tokens):
                    print(mapping[i].replace('\n', '@@NEWLINE@@'), file=token_file)

    def set_from_file(self,
                      filename: str,
                      is_padded: bool = True,
                      oov_token: str = DEFAULT_OOV_TOKEN,
                      namespace: str = "tokens"):
        """
        if you already have a vocabulary file for a trained model somewhere, and you really want to
        use that vocabulary file instead of just setting the vocabulary from a dataset, for
        whatever reason, you can do that with this method.  You must specify the namespace to use,
        and we assume that you want to use padding and OOV tokens for this.

        parameters
        ----------
        filename : ``str``
            The file containing the vocabulary to load.  It should be formatted as one token per
            line, with nothing else in the line.  The index we assign to the token is the line
            number in the file (1-indexed if ``is_padded``, 0-indexed otherwise).  Note that this
            file should contain the OOV token string!
        is_padded : ``bool``, optional (default=True)
            Is this vocabulary padded?  For token / word / character vocabularies, this should be
            ``True``; while for tag or label vocabularies, this should typically be ``False``.  If
            ``True``, we add a padding token with index 0, and we enforce that the ``oov_token`` is
            present in the file.
        oov_token : ``str``, optional (default=DEFAULT_OOV_TOKEN)
            What token does this vocabulary use to represent out-of-vocabulary characters?  This
            must show up as a line in the vocabulary file.  When we find it, we replace
            ``oov_token`` with ``self._oov_token``, because we only use one OOV token across
            namespaces.
        namespace : ``str``, optional (default="tokens")
            What namespace should we overwrite with this vocab file?
        """
        if is_padded:
            self._token_to_index[namespace] = {self._padding_token: 0}
            self._index_to_token[namespace] = {0: self._padding_token}
        else:
            self._token_to_index[namespace] = {}
            self._index_to_token[namespace] = {}
        with codecs.open(filename, 'r', 'utf-8') as input_file:
            lines = input_file.read().split('\n')
            # Be flexible about having final newline or not
            if lines and lines[-1] == '':
                lines = lines[:-1]
            for i, line in enumerate(lines):
                index = i + 1 if is_padded else i
                token = line.replace('@@NEWLINE@@', '\n')
                if token == oov_token:
                    token = self._oov_token
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
        if is_padded:
            assert self._oov_token in self._token_to_index[namespace], "OOV token not found!"

    # there's enough logic here to require a custom from_params.
    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):  # type: ignore
        """
        there are two possible ways to build a vocabulary; from a
        collection of instances, using :func:`Vocabulary.from_instances`, or
        from a pre-saved vocabulary, using :func:`Vocabulary.from_files`.
        you can also extend pre-saved vocabulary with collection of instances
        using this method. This method wraps these options, allowing their
        specification from a ``Params`` object, generated from a JSON
        configuration file.

        parameters
        ----------
        params: Params, required.
        instances: Iterable['adi.Instance'], optional
            If ``params`` doesn't contain a ``directory_path`` key,
            the ``Vocabulary`` can be built directly from a collection of
            instances (i.e. a dataset). If ``extend`` key is set False,
            dataset instances will be ignored and final vocabulary will be
            one loaded from ``directory_path``. If ``extend`` key is set True,
            dataset instances will be used to extend the vocabulary loaded
            from ``directory_path`` and that will be final vocabulary used.

        returns
        -------
        a ``Vocabulary``.
        """
        # pylint: disable=arguments-differ

        # Vocabulary is ``Registrable`` so that you can configure a custom subclass,
        # but (unlike most of our registrables) almost everyone will want to use the
        # base implementation. So instead of having an abstract ``VocabularyBase`` or
        # such, we just add the logic for instantiating a registered subclass here,
        # so that most users can continue doing what they were doing.
        vocab_type = params.pop("type", None)
        if vocab_type is not None:
            return cls.by_name(vocab_type).from_params(params=params, instances=instances)

        extend = params.pop("extend", False)
        vocabulary_directory = params.pop("directory_path", None)
        if not vocabulary_directory and not instances:
            raise ConfigurationError("You must provide either a Params object containing a "
                                     "vocab_directory key or a Dataset to build a vocabulary from.")
        if extend and not instances:
            raise ConfigurationError("'extend' is true but there are not instances passed to extend.")
        if extend and not vocabulary_directory:
            raise ConfigurationError("'extend' is true but there is not 'directory_path' to extend from.")

        if vocabulary_directory and instances:
            if extend:
                info("Loading Vocab from files and extending it with dataset.")
            else:
                info("Loading Vocab from files instead of dataset.")

        if vocabulary_directory:
            vocab = Vocabulary.from_files(vocabulary_directory)
            if not extend:
                params.assert_empty("Vocabulary - from files")
                return vocab
        if extend:
            vocab.extend_from_instances(params, instances=instances)
            return vocab
        min_count = params.pop("min_count", None)
        max_vocab_size = pop_max_vocab_size(params)
        non_padded_namespaces = params.pop("non_padded_namespaces", DEFAULT_NON_PADDED_NAMESPACES)
        pretrained_files = params.pop("pretrained_files", {})
        only_include_pretrained_words = params.pop_bool("only_include_pretrained_words", False)
        tokens_to_add = params.pop("tokens_to_add", None)
        params.assert_empty("Vocabulary - from dataset")
        return Vocabulary.from_instances(instances=instances,
                                         min_count=min_count,
                                         max_vocab_size=max_vocab_size,
                                         non_padded_namespaces=non_padded_namespaces,
                                         pretrained_files=pretrained_files,
                                         only_include_pretrained_words=only_include_pretrained_words,
                                         tokens_to_add=tokens_to_add)

    def _extend(self,
                counter: Dict[str, Dict[str, int]] = None,
                min_count: Dict[str, int] = None,
                max_vocab_size: Union[int, Dict[str, int]] = None,
                non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
                pretrained_files: Optional[Dict[str, str]] = None,
                only_include_pretrained_words: bool = False,
                tokens_to_add: Dict[str, List[str]] = None) -> None:
        """
        this method can be used for extending already generated vocabulary.
        it takes same parameters as Vocabulary initializer. The token2index
        and indextotoken mappings of calling vocabulary will be retained.
        it is an inplace operation so None will be returned.
        """
        if not isinstance(max_vocab_size, dict):
            int_max_vocab_size = max_vocab_size
            max_vocab_size = defaultdict(lambda: int_max_vocab_size)  # type: ignore
        min_count = min_count or {}
        pretrained_files = pretrained_files or {}
        non_padded_namespaces = set(non_padded_namespaces)
        counter = counter or {}
        tokens_to_add = tokens_to_add or {}

        self._retained_counter = counter
        # Make sure vocabulary extension is safe.
        current_namespaces = {*self._token_to_index}
        extension_namespaces = {*counter, *tokens_to_add}

        for namespace in current_namespaces & extension_namespaces:
            # if new namespace was already present
            # Either both should be padded or none should be.
            original_padded = not any(namespace_match(pattern, namespace)
                                      for pattern in self._non_padded_namespaces)
            extension_padded = not any(namespace_match(pattern, namespace)
                                       for pattern in non_padded_namespaces)
            if original_padded != extension_padded:
                raise ConfigurationError("Common namespace {} has conflicting ".format(namespace)+
                                         "setting of padded = True/False. "+
                                         "Hence extension cannot be done.")

        # Add new non-padded namespaces for extension
        self._token_to_index.add_non_padded_namespaces(non_padded_namespaces)
        self._index_to_token.add_non_padded_namespaces(non_padded_namespaces)
        self._non_padded_namespaces.update(non_padded_namespaces)

        for namespace in counter:
            if namespace in pretrained_files:
                raise NotImplementedError("Uh oh, I deleted this line and something bad happened : {}".format(
                    "pretrained_list = _read_pretrained_tokens(pretrained_files[namespace])"))
            else:
                pretrained_list = None
            token_counts = list(counter[namespace].items())
            token_counts.sort(key=lambda x: x[1], reverse=True)
            try:
                max_vocab = max_vocab_size[namespace]
            except KeyError:
                max_vocab = None
            if max_vocab:
                token_counts = token_counts[:max_vocab]
            for token, count in token_counts:
                if pretrained_list is not None:
                    if only_include_pretrained_words:
                        if token in pretrained_list and count >= min_count.get(namespace, 1):
                            self.add_token_to_namespace(token, namespace)
                    elif token in pretrained_list or count >= min_count.get(namespace, 1):
                        self.add_token_to_namespace(token, namespace)
                elif count >= min_count.get(namespace, 1):
                    self.add_token_to_namespace(token, namespace)

        for namespace, tokens in tokens_to_add.items():
            for token in tokens:
                self.add_token_to_namespace(token, namespace)

    def is_padded(self, namespace: str) -> bool:
        """
        returns whether or not there are padding and OOV tokens added to the given namepsace.
        """
        return self._index_to_token[namespace][0] == self._padding_token

    def add_token_to_namespace(self, token: str, namespace: str = 'tokens') -> int:
        """
        adds ``token`` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError("Vocabulary tokens must be strings, or saving and loading will break."
                             "  Got %s (with type %s)" % (repr(token), type(token)))
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def get_index_to_token_vocabulary(self, namespace: str = 'tokens') -> Dict[int, str]:
        return self._index_to_token[namespace]

    def get_token_to_index_vocabulary(self, namespace: str = 'tokens') -> Dict[str, int]:
        return self._token_to_index[namespace]

    def get_token_index(self, token: str, namespace: str = 'tokens') -> int:
        if token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][token]
        else:
            try:
                return self._token_to_index[namespace][self._oov_token]
            except KeyError:
                error('Namespace: %s', namespace)
                error('Token: %s', token)
                raise

    def get_token_from_index(self, index: int, namespace: str = 'tokens') -> str:
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str = 'tokens') -> int:
        return len(self._token_to_index[namespace])

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) -> str:
        base_string = f"Vocabulary with namespaces:\n"
        non_padded_namespaces = f"\tNon Padded Namespaces: {self._non_padded_namespaces}\n"
        namespaces = [f"\tNamespace: {name}, Size: {self.get_vocab_size(name)} \n"
                      for name in self._index_to_token]
        return " ".join([base_string, non_padded_namespaces] + namespaces)

    def print_statistics(self) -> None:
        if self._retained_counter:
            info("Printed vocabulary statistics are only for the part of the vocabulary generated "
                 "from instances. If vocabulary is constructed by extending saved vocabulary with "
                 "dataset instances, the directly loaded portion won't be considered here.")
            print("\n\n----Vocabulary Statistics----\n")
            # Since we don't saved counter info, it is impossible to consider pre-saved portion.
            for namespace in self._retained_counter:
                tokens_with_counts = list(self._retained_counter[namespace].items())
                tokens_with_counts.sort(key=lambda x: x[1], reverse=True)
                print(f"\nTop 10 most frequent tokens in namespace '{namespace}':")
                for token, freq in tokens_with_counts[:10]:
                    print(f"\tToken: {token}\t\tFrequency: {freq}")
                # Now sort by token length, not frequency
                tokens_with_counts.sort(key=lambda x: len(x[0]), reverse=True)

                print(f"\nTop 10 longest tokens in namespace '{namespace}':")
                for token, freq in tokens_with_counts[:10]:
                    print(f"\tToken: {token}\t\tlength: {len(token)}\tFrequency: {freq}")

                print(f"\nTop 10 shortest tokens in namespace '{namespace}':")
                for token, freq in reversed(tokens_with_counts[-10:]):
                    print(f"\tToken: {token}\t\tlength: {len(token)}\tFrequency: {freq}")
        else:
            # _retained_counter would be set only if instances were used for vocabulary construction.
            info("Vocabulary statistics cannot be printed since "
                 "dataset instances were not used for its construction.")