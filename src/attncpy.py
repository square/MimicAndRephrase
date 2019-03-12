import gzip
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List, Iterable, Dict, Optional, Callable, NamedTuple, Union, cast
import io
import os

import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np


from attention import LearnedAttention
from data import IDKRephraseDataset, default_data_loader
from utils.token_mapper import RegexTokenMapping, TokenMapping, ExactTokenMapping, TokenMapper, HashTokenMapping
from utils.maskedsequence import MaskedSequence
from utils.gloveembedding import GloveEmbedding
from utils.term_frequencies import TermFrequencies
from utils.elmo import Elmo, batch_to_ids
from utils.word_embedding import Glove

from utils.log import track, info, warn, error, init


class IDKRephraseModel:

    def __init__(self, model_params: Dict, glove: Glove=None, writer=None):
        self.lr = model_params['lr']
        self.hidden_size = model_params['hidden_size']
        self.n_layers = model_params['n_layers']
        self.attention_size = model_params.get("attention_size", 512)
        self.copy_attn_size = model_params.get("copy_attn_size", 512)
        self.bidirectional = model_params['bidirectional']
        if self.bidirectional:
            self.n_directions = 2
        else:
            self.n_directions = 1
        self.dropout = model_params['dropout']
        self.vocab = model_params['vocab']
        self.num_unks = model_params['num_unks']
        self.misc_tokens = model_params['misc_tokens']
        self.use_copy = model_params.get('use_copy', True)
        self.use_cuda = model_params.get('use_cuda', False)
        self.copy_extra_layer = model_params.get('copy_extra_layer', True)
        self.attn_extra_layer = model_params.get('attn_extra_layer', False)
        self.copy_layer_size = model_params.get('copy_extra_layer_size', self.copy_attn_size)
        self.attn_extra_layer_size = model_params.get('attn_extra_layer_size', self.attention_size)
        self.do_sentiment = model_params.get('do_sentiment', False)
        self.use_coverage = model_params.get('use_coverage', False)
        self.coverage_weight = model_params.get('coverage_weight', 1.0)
        self.model_params = model_params

        self.writer = writer

        # Create glove
        if glove is None:
            info('Creating duplicate detection model with new GloVE')
            glove = Glove.from_binary()

        self.elmo = Elmo.get_default(1)

        token_mappings: List[TokenMapping] = [
            RegexTokenMapping("^[0-9]$", 3, "NUM_1"),
            RegexTokenMapping("^[0-9]{2}$", 3, "NUM_2"),
            RegexTokenMapping("^[0-9]{3}$", 3, "NUM_3"),
            RegexTokenMapping("^[0-9]+$", 3, "NUM_MANY"),
            RegexTokenMapping("^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", 3, "EMAIL"),
            ExactTokenMapping(self.misc_tokens),
        ]

        info('Adding {} token mappings'.format(len(token_mappings)))
        self.glove = glove.with_new_token_mapper(
            token_mapper=TokenMapper(token_mappings, [HashTokenMapping(self.num_unks)])
        )

        self.glove_embedding: Optional[GloveEmbedding] = None
        self.glove_embedding = GloveEmbedding(self.glove)

        num_sentiment_flags = 0
        if self.do_sentiment:
            num_sentiment_flags = 2
        self.encoder = nn.LSTM(
            input_size=self.glove.embedding_dim + 1024 + num_sentiment_flags,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional
        )

        self.decoder_rnn = nn.LSTM(
            input_size=self.glove.embedding_dim + self.n_directions * self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
        )
        self.attn = LearnedAttention(self.n_directions * self.hidden_size, self.hidden_size, self.attention_size,
                                     self.attn_extra_layer, use_coverage=self.use_coverage, use_bias=self.do_sentiment)

        if self.use_copy:
            self.copy_attn = LearnedAttention(self.n_directions * self.hidden_size + self.glove.embedding_dim
                                              + num_sentiment_flags + 1024,
                                              self.hidden_size + self.n_directions * self.hidden_size +
                                              self.glove.embedding_dim, attn_dim=self.copy_attn_size,
                                              extra_layer=self.copy_extra_layer, dropout=self.dropout, use_bias=self.do_sentiment)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.softmax_size = len(self.vocab)
        self.vocab_logits = nn.Linear(self.hidden_size, self.softmax_size, bias=False)
        self.softmax = nn.LogSoftmax(dim=1)

        params = list(self.encoder.parameters()) + list(self.decoder_rnn.parameters()) \
                 + list(self.vocab_logits.parameters()) + list(self.attn.parameters()) \
                 + list(filter(lambda p: p.requires_grad, self.elmo.parameters()))
        if self.use_copy:
            params += list(self.copy_attn.parameters())

        self.optimizer = torch.optim.Adam(self.filter_params(params), lr=self.lr)

    @classmethod
    def get_vocab_from_list_and_files(cls, vocab_list: str, vocab_size: int, files: List[str], misc_tokens: List[str]) \
            -> List[str]:
        vocab_set = set()
        if vocab_list is not None:
            vocab_stats = TermFrequencies.from_file(vocab_list).most_common(vocab_size)
            vocab_set |= set([count[0] for count in vocab_stats])
        vocab_set |= set(misc_tokens)
        for file in files:
            vocab_set |= IDKRephraseDataset.get_vocab_from_TSV(file)
        return sorted(list(vocab_set))

    def set_dropout(self, dropout: float):
        self.encoder.dropout = dropout
        self.decoder_rnn.dropout = dropout

    def train_dataset(self, dataset: IDKRephraseDataset, devset: IDKRephraseDataset, batch_size: int, num_workers: int,
                      num_epochs: int, model_path: str, shuffle: bool = True, verbose: bool = False):
        data_loader = default_data_loader(dataset, batch_size, num_workers, shuffle)
        batches = 0
        score = 0
        loss_avg = 0
        epoch_loss_avg = 0
        successive_without_improving = 0
        for epoch in range(num_epochs):
            for batch in data_loader:
                loss = self.train_batch(batch.question_indices, batch.question_lengths,
                                        batch.response_indices, batch.response_lengths, batch.copy_indices,
                                        batch.raw_questions, batch.sentiment_flags)
                epoch_loss_avg += loss
                loss_avg += loss
                batches += 1
                if batches % 20 == 0:
                    if verbose:
                        info("Epoch: " + str(epoch) + " Batch: " + str(batches) + " " + str(loss_avg / 20))
                    loss_avg = 0
            info("Average loss over epoch " + str(epoch) + ": " + str(epoch_loss_avg / batches))
            batches = 0
            epoch_loss_avg = 0
            loss_avg = 0
            new_score = self.evaluate_bleu(devset, max_samples=300)
            print("Train value:")
            self.evaluate_bleu(dataset, max_samples=50)
            if new_score > (score + 0.0001):
                self.save(model_path)
                score = new_score
                successive_without_improving = 0
            else:
                successive_without_improving += 1
                #if successive_without_improving >= 10:
                #    break
            #if epoch >= 6 and score < 0.1:
                #break
        return score

    def train_batch(self, indices: torch.Tensor, lengths: torch.Tensor, target_seq: torch.Tensor,
                    target_lengths: torch.Tensor, copy_indices: torch.Tensor, raw_questions: List[List[str]],
                    sentiment_flags: torch.Tensor):
        elmo_indices = self.place(batch_to_ids(raw_questions))
        elmo_data = self.elmo(elmo_indices)["elmo_representations"][0]
        self.optimizer.zero_grad()
        self.set_dropout(self.dropout)
        target_seq = self.place(target_seq)
        target_lengths = self.place(target_lengths)
        copy_indices = self.place(copy_indices)
        sentiment_flags = self.place(sentiment_flags)
        embed: torch.Tensor = self.place(self.glove_embedding(indices))

        sentiment_flags = torch.unsqueeze(sentiment_flags, 1)
        sentiment_flags = sentiment_flags.expand(-1, embed.shape[1], -1)
        #Append sentiment flags to embedding
        if self.do_sentiment:
            embed = torch.cat([sentiment_flags, embed], dim=2)

        sequence: MaskedSequence = MaskedSequence(embed, lengths.numpy())
        elmo_seq: MaskedSequence = MaskedSequence(elmo_data, lengths.numpy())
        combined_data = torch.cat([sequence.get_data(), elmo_seq.get_data()], dim=2)
        combined_seq = sequence.with_new_data(combined_data)
        padded = combined_seq.as_packed_sequence()
        output_tensor, (h_n, c_n) = self.encoder(padded)
        output_seq = combined_seq.from_padded_sequence(pad_packed_sequence(output_tensor, batch_first=True)[0])

        tokens = indices.new(indices.size()[0]).fill_(self.glove.lookup_word("<SOS>"))  # batch_size
        tokens = torch.unsqueeze(tokens, 0)
        target_length = target_seq.size()[1]
        h_n = torch.t(sequence.unsort(torch.t(h_n)))

        h_n = self.dropout_layer(h_n)

        if self.bidirectional:
            c_0 = h_n.new_zeros(h_n.size()[0] // 2, h_n.size()[1], h_n.size()[2])
            h_0 = h_n.new_zeros(h_n.size()[0] // 2, h_n.size()[1], h_n.size()[2])
            for i in range(self.n_layers):
                h_0[i] = h_n[2 * i] + h_n[2 * i + 1]
            hiddens = (h_0, c_0)
        else:
            c_n = h_n.new_zeros(h_n.size())
            hiddens = (h_n.contiguous(), c_n)
        loss = 0
        curr_hidden = hiddens[0][self.n_layers - 1]
        coverage = None
        for i in range(target_length):
            embed_decode = self.place(self.glove_embedding(tokens))
            context, _, coverage, coverage_loss = self.attn(output_seq, curr_hidden, coverage)
            if self.use_coverage:
                loss += self.coverage_weight * coverage_loss
            context = torch.unsqueeze(context, 0)
            input = torch.cat([embed_decode, context], dim=2).contiguous()
            rnn_out, hiddens = self.decoder_rnn(input, hiddens)
            rnn_out = torch.squeeze(rnn_out, dim=0)
            curr_hidden = rnn_out
            correct_indices = torch.index_select(target_seq, 1, self.place(torch.LongTensor([i])))  # batch_size * 1
            tokens = torch.squeeze(correct_indices, 1)  # batch_size
            words = [self.vocab[index] for index in tokens]
            tokens = torch.LongTensor(self.glove.words_to_indices(words))
            self.place(tokens)
            tokens = torch.unsqueeze(tokens, 0)
            vocab_logits = self.vocab_logits(rnn_out)
            if self.use_copy:
                copy_input = torch.cat([curr_hidden, torch.squeeze(context, 0), torch.squeeze(embed_decode, 0)], dim=1)
                copy_seq_data = torch.cat([output_seq.get_data(), combined_seq.get_data()], dim=2)
                copy_seq = output_seq.with_new_data(copy_seq_data)
                _, copy_logits, _, _ = self.copy_attn(copy_seq, copy_input)
                copy_logits = torch.squeeze(copy_logits, 2)
                logits = torch.cat([vocab_logits, copy_logits], dim=1)
            else:
                logits = vocab_logits
            out = self.softmax(logits)
            word_loss = torch.squeeze(torch.gather(out, 1, correct_indices), 1)
            if self.use_copy:
                copy_logits_normalized = out[:, len(self.vocab):]
                copy_weight = 2
                word_loss += copy_weight * torch.sum(copy_logits_normalized * copy_indices[:, :, i], 1)
            mask_vec = torch.gt(target_lengths - i, 0)  # apply mask to word_loss
            word_loss = torch.sum(word_loss * mask_vec.float())
            loss -= word_loss

        batch_size = indices.size()[0]
        loss /= batch_size
        loss.backward()

        self.optimizer.step()

        loss_val = loss.data.cpu().numpy()
        return loss_val

    def rephrase(self, tokens: List[str]) -> List[str]:
        return self.predict(torch.LongTensor(self.glove.words_to_indices(tokens)), tokens, True)

    def predict(self, indices: torch.Tensor, str_tokens: List[str], use_threshold: bool = False,
                make_attn_graphics: bool = False, sentiment_tensor: torch.Tensor = None) -> List[str]:
        return self.predict_with_threshold(indices, str_tokens, use_threshold, make_attn_graphics, sentiment_tensor)[0]

    def predict_with_threshold(self, indices: torch.Tensor, str_tokens: List[str], use_threshold: bool = False,
                make_attn_graphics: bool = False, sentiment_tensor: torch.Tensor = None) -> (List[str], float):
        """
        Actually run the model, returning a list of tokens to show.
        If we should not show anything, this returns an empty list.

        :param indices: The indices into GloVE of the input sentence
        :param str_tokens:  The text tokens of the input sentence.
        :param use_threshold: If true, extractions under this threshold of confidence
                              should not be shown.

        :return: A list of output tokens. Returns an empty list if we don't want to generate anything
        """
        self.set_dropout(0)
        embed: torch.Tensor = self.place(self.glove_embedding(indices))
        if sentiment_tensor is not None:
            sentiment_tensor = self.place(sentiment_tensor)

        seq = torch.unsqueeze(embed, 1)
        #Append sentiment flags to embedding
        if self.do_sentiment:
            sentiment_flags = torch.unsqueeze(sentiment_tensor, 0)
            sentiment_flags = torch.unsqueeze(sentiment_flags, 0)
            sentiment_flags = sentiment_flags.expand(seq.shape[0], seq.shape[1], -1)
            seq = torch.cat([sentiment_flags, seq], dim=2)

        raw_questions = [str_tokens]
        elmo_indices = self.place(batch_to_ids(raw_questions))
        elmo_data = self.elmo(elmo_indices)["elmo_representations"][0]
        seq = torch.cat([seq, torch.t(elmo_data)], dim=2).contiguous()
        input_length = embed.size()[0]
        output_tensor, (h_n, c_n) = self.encoder(seq)
        output_seq = MaskedSequence(torch.t(output_tensor.data), [embed.size()[0]])
        token = indices.new(1).fill_(self.glove.lookup_word("<SOS>"))  # batch_size
        token = torch.unsqueeze(token, 0)
        if self.bidirectional:
            c_0 = h_n.new_zeros(h_n.size()[0] // 2, h_n.size()[1], h_n.size()[2])
            h_0 = h_n.new_zeros(h_n.size()[0] // 2, h_n.size()[1], h_n.size()[2])
            for i in range(self.n_layers):
                h_0[i] = h_n[2 * i] + h_n[2 * i + 1]
            hiddens = (h_0, c_0)
        else:
            c_n = h_n.new_zeros(h_n.size())
            hiddens = (h_n.contiguous(), c_n)
        curr_hidden = hiddens[0][self.n_layers - 1]
        out_tokens = ["<SOS>"]
        length = 0
        max_length = 40
        perplexity = 0
        logits_for_graphic = []
        copy_logits_for_graphic = []
        beam_size = 3
        coverage = None
        old_beams = [(perplexity, out_tokens, token, hiddens, curr_hidden,
                      [logits_for_graphic, copy_logits_for_graphic], length, coverage)]
        finished = False
        while not finished:
            new_beams = []
            for beam in old_beams:
                beam_perplexity = beam[0]
                if beam[1][-1] == "<EOS>":
                    new_beams.append(beam)
                    continue
                beam_out_tokens = beam[1].copy()
                beam_token = beam[2]
                beam_hiddens = (torch.tensor(beam[3][0]), torch.tensor(beam[3][1]))
                beam_curr_hidden = torch.tensor(beam[4])
                beam_logits_for_graphic = beam[5][0].copy()
                beam_copy_logits_for_graphic = beam[5][1].copy()
                beam_length = beam[6]
                embed_decode = self.place(self.glove_embedding(beam_token))
                context, attn_logits, new_coverage, _ = self.attn(output_seq, beam_curr_hidden, beam[7])
                if make_attn_graphics:
                    beam_logits_for_graphic.append(torch.squeeze(attn_logits).cpu().detach().numpy())
                context = torch.unsqueeze(context, 0)
                input = torch.cat([embed_decode, context], dim=2).contiguous()
                rnn_out, beam_hiddens = self.decoder_rnn(input, beam_hiddens)
                rnn_out = torch.squeeze(rnn_out, dim=0)
                beam_curr_hidden = rnn_out
                vocab_logits = self.vocab_logits(rnn_out)
                if self.use_copy:
                    copy_input = torch.cat(
                        [beam_curr_hidden, torch.squeeze(context, 0), torch.squeeze(embed_decode, 0)], dim=1)
                    copy_seq_data = torch.cat([output_seq.get_data(), torch.t(seq)], dim=2)
                    copy_seq = output_seq.with_new_data(copy_seq_data)
                    _, copy_logits, _, _ = self.copy_attn(copy_seq, copy_input)
                    copy_logits = torch.squeeze(copy_logits, 2)
                    logits = torch.cat([vocab_logits, copy_logits], dim=1)
                else:
                    logits = vocab_logits
                out = self.softmax(logits)
                if make_attn_graphics and self.use_copy:
                    beam_copy_logits_for_graphic.append(torch.squeeze(out[0, len(self.vocab):])
                                                        .cpu().detach().numpy())
                next_indices = torch.squeeze(out.topk(beam_size)[1], 0)
                beam_length += 1
                for i in range(beam_size):
                    new_beam_tokens = beam_out_tokens.copy()
                    if beam_length > max_length:
                        new_beam_perplexity = beam_perplexity
                        new_token = indices.new(1).fill_(self.glove.lookup_word("<EOS>"))
                        new_beam_tokens.append("<EOS>")
                    else:
                        new_beam_perplexity = beam_perplexity + out[0, next_indices[i]].cpu().detach().numpy()
                        if next_indices[i] < len(self.vocab):
                            new_beam_tokens.append(self.vocab[next_indices[i]])
                            new_token = indices.new(1).fill_(self.glove.lookup_word(self.vocab[next_indices[i]]))
                            new_token = torch.unsqueeze(new_token, 0)
                        else:
                            copy_token = str_tokens[next_indices[i] - len(self.vocab)]
                            if next_indices[i] - len(self.vocab) == 0 and beam_length != 0:
                                copy_token = copy_token.lower()
                            new_beam_tokens.append(copy_token)
                            new_token = indices.new(1).fill_(self.glove.lookup_word(copy_token))
                            new_token = torch.unsqueeze(new_token, 0)
                    new_beams.append((new_beam_perplexity, new_beam_tokens, new_token, beam_hiddens, beam_curr_hidden,
                                      [beam_logits_for_graphic, beam_copy_logits_for_graphic], beam_length, new_coverage))
            old_beams = sorted(new_beams, key=lambda sample: -(sample[0]
                            + self.scale_perplexity(sample[6], input_length)))[:beam_size]
            finished = True
            for beam in old_beams:
                if beam[1][-1] != "<EOS>":
                    finished = False
                    break
        perplexity, out_tokens, _, _, _, graph_data, length, _ = max(old_beams,
                        key=lambda beam: beam[0] + self.scale_perplexity(beam[6], input_length))
        logits_for_graphic = graph_data[0]
        copy_logits_for_graphic = graph_data[1]
        if make_attn_graphics:
            self.make_attn_graphic(np.array(logits_for_graphic), str_tokens, out_tokens[1:], "attn"
                                   + str(int(time.time())))
            if self.use_copy:
                self.make_attn_graphic(np.array(copy_logits_for_graphic), str_tokens, out_tokens[1:],
                                       "copy" + str(int(time.time())))

        normalized_perplexity = perplexity + self.scale_perplexity(length, input_length)
        if use_threshold:
            first_tokens = [token.lower() for token in out_tokens[:5]]
            if first_tokens != ["<sos>", "i", "do", "not", "know"]\
                    and first_tokens != ["<sos>", "i", "am", "not", "able"]:
                out_tokens = []
        if use_threshold and normalized_perplexity < -.51:
                out_tokens = []
        return out_tokens, normalized_perplexity

    def scale_perplexity(self, length, input_length):
        reward_for_length = 0.3758
        additive = reward_for_length * np.minimum(float(length), input_length*1.2951)
        return additive

    def make_attn_graphic(self, logits: np.array, input_strs: List[str], output_strs: List[str], name: str):
        from utils.viz import figure_to_image
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        if output_strs[0] == "i":
            output_strs[0] = "I"
        df = pd.DataFrame(logits[:-1, :], columns=input_strs, index=output_strs[:-1])
        fig, ax = plt.subplots()
        plt.gcf().subplots_adjust(bottom=0.3, left=0.3)
        sns.heatmap(df, annot=False, ax=ax, cmap="Blues")
        image = figure_to_image(fig)
        if self.writer:
            self.writer.add_image("attn_graph_" + name, image, 0)

        sns.set(style="dark")

    def filter_params(self, params):
        return filter(lambda p: p.requires_grad, params)

    def get_elmo_state_dict(self):
        all_dict = self.elmo.state_dict()
        new_dict = {}
        for name, weights in all_dict.items():
            name = cast(str, name)
            if not name.startswith('_elmo_lstm'):
                new_dict[name] = weights
        return new_dict

    def save(self, path: str, compress: bool = True):
        model_data = {
            "glove": self.glove_embedding.state_dict(),
            "decoder_rnn": self.decoder_rnn.state_dict(),
            "vocab_logits": self.vocab_logits.state_dict(),
            "attn": self.attn.state_dict(),
            "encoder": self.encoder.state_dict(),
            "model_params": self.model_params,
            "elmo": self.get_elmo_state_dict()
        }
        if self.use_copy:
            model_data["copy_attn"] = self.copy_attn.state_dict()
        bytes_io = io.BytesIO()
        torch.save(model_data, bytes_io)
        model_bytes = bytes_io.getvalue()
        if compress:
            with gzip.open(path, "wb+") as f:
                f.write(model_bytes)
        else:
            with open(path, "wb+") as f:
                f.write(model_bytes)

    @classmethod
    def from_file(cls, filename: str, glove: Glove = None, writer=None) -> 'IDKRephraseModel':
        """
        Load a model from file
        :param filename: Filename for the MCModel
        :param glove: Glove to use for the Model
        """
        info("Loading model from file: {}".format(filename))
        if filename.endswith(".pkl.gz"):
            with gzip.open(filename, "rb") as f:
                return cls.from_bytes(f.read(), glove=glove, writer=writer)
        elif filename.endswith(".pkl"):
            with open(filename, "rb") as f:
                return cls.from_bytes(f.read(), glove=glove, writer=writer)
        else:
            raise ValueError("Invalid file due to unsupported extension: {}".format(filename))
        pass

    @classmethod
    def from_bytes(cls, content: bytes, glove: Glove = None, writer=None) -> 'IDKRephraseModel':
        # Parse the bytes
        with io.BytesIO(content) as f:
            model_data = torch.load(f, map_location=lambda storage, loc: storage)

        # Load the config
        model_params = model_data["model_params"]
        model = IDKRephraseModel(model_params, glove, writer)

        model.encoder.load_state_dict(model_data["encoder"])
        model.decoder_rnn.load_state_dict(model_data["decoder_rnn"])
        model.vocab_logits.load_state_dict(model_data["vocab_logits"])
        model.attn.load_state_dict(model_data["attn"])
        elmo_state = model.elmo.state_dict()
        elmo_state.update(model_data["elmo"])
        model.elmo.load_state_dict(elmo_state)
        if model.use_copy:
            model.copy_attn.load_state_dict(model_data["copy_attn"])
        model.glove_embedding.load_state_dict(model_data["glove"])

        return model

    def set_cuda(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self.place_all()

    def place_all(self):
        self.place(self.encoder)
        self.place(self.attn)
        self.place(self.decoder_rnn)
        if self.use_copy:
            self.place(self.copy_attn)
        self.place(self.vocab_logits)
        self.place(self.elmo)

    def place(self, var):
        """
        Places a Tensor / Module in the right place according to whether or not this network has been placed
        on the GPU
        :return:
        """
        if self.use_cuda:
            if issubclass(type(var), nn.Module):
                return var.cuda()
            else:
                return var.cuda(non_blocking=True)
        else:
            return var.cpu()

    def evaluate_bleu(self, dataset: IDKRephraseDataset, print_predictions: bool = False, max_samples: int = -1,
                      condensed_only: bool = True):
        data_loader = default_data_loader(dataset, 1, 0, False)
        from nltk.translate.bleu_score import sentence_bleu
        total_bleu = 0
        samples = 0
        f = None
        if print_predictions:
            f = open("model_out.tsv", "w")
        for batch in data_loader:
            if condensed_only and self.do_sentiment:
                if batch.sentiment_flags[0].cpu().numpy()[1] != 1:
                    continue
            prediction, threshold = self.predict_with_threshold(batch.question_indices[0], batch.raw_questions[0],
                                                                sentiment_tensor=batch.sentiment_flags[0])
            correct = batch.raw_responses[0]
            correct = [token.lower() for token in correct]
            correct = correct[:-1]
            prediction = [token.lower() for token in prediction]
            prediction = prediction[1:-1]
            question = batch.raw_questions[0]
            if print_predictions:
                info("Question: " + ' '.join(question))
                info("Correct Response: " + ' '.join(correct))
                info("Response: " + ' '.join(prediction))
                info("Threshold: " + str(threshold))
                if self.do_sentiment:
                    f.write(' '.join(question[1:-1]) + "\t" + ' '.join(correct) + "\t" + ' '.join(prediction) + "\t"
                        + str(threshold) + "\n")
                else:
                    f.write(' '.join(question) + "\t" + ' '.join(correct) + "\t" + ' '.join(prediction) + "\t"
                            + str(threshold) + "\n")
            total_bleu += sentence_bleu([correct], prediction)
            samples += 1
            if samples >= max_samples and max_samples != -1:
                break

        avg_score = total_bleu / samples
        info("Average Bleu Score: " + str(avg_score))
        return avg_score

    def get_worst_bleu(self, dataset: IDKRephraseDataset):
        data_loader = default_data_loader(dataset, 1, 0, True)
        from nltk.translate.bleu_score import sentence_bleu
        result_tuples = []
        for batch in data_loader:
            prediction = self.predict(batch.question_indices[0], batch.raw_questions[0])
            prediction = [token.lower() for token in prediction]
            correct = batch.raw_responses[0]
            correct.insert(0, "<SOS>")
            correct = [token.lower() for token in correct]
            question = self.glove.indices_to_words(batch.question_indices[0])
            result_tuples.append((question, correct, prediction, sentence_bleu([correct], prediction)))

        result_tuples = sorted(result_tuples, key=lambda sample: sample[3])
        for sample in result_tuples:
            info("Question: " + ' '.join(sample[0]))
            info("Correct Response: " + ' '.join(sample[1]))
            info("Response: " + ' '.join(sample[2]))
            info("BLEU: " + str(sample[3]))


def main(train: bool, model_path: str, dev_tsv_file: str = None, tsv_file: str = None,
         vocab_list: str = None, vocab_size=30000, batch_size=64, epochs=1, use_cuda=True, lr=0.002, hidden_size=1024,
         n_layers=1, bidirectional=True, dropout=0.5, num_unks=10, use_copy=True, attention_size=512,
         copy_attn_size=512, copy_extra_layer=True, attn_extra_layer=True, copy_extra_layer_size=512,
         attn_extra_layer_size=512, continue_training=False, do_sentiment=False, use_coverage=False,
         coverage_weight = 1.0):
    init("log/out.txt")

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), "tensorboard"))
    dataset = None
    devset = None

    if train:

        model_param = {
            'lr': lr,
            'hidden_size': hidden_size,
            'n_layers': n_layers,
            'bidirectional': bidirectional,
            'dropout': dropout,
            'num_unks': num_unks,
            'use_copy': use_copy,
            'attention_size': attention_size,
            'copy_attn_size': copy_attn_size,
            'copy_extra_layer': copy_extra_layer,
            'attn_extra_layer': attn_extra_layer,
            'copy_extra_layer_size': copy_extra_layer_size,
            'attn_extra_layer_size': attn_extra_layer_size,
            'use_coverage': use_coverage,
            "do_sentiment": do_sentiment,
            "coverage_weight": coverage_weight
        }
        misc_tokens = ["<SOS>", "<EOS>"]
        model_param['misc_tokens'] = misc_tokens
        model_param['vocab'] = IDKRephraseModel.get_vocab_from_list_and_files(
            vocab_list, vocab_size, [tsv_file, dev_tsv_file], misc_tokens)
        model = IDKRephraseModel(model_param, writer=writer)
        dataset = IDKRephraseDataset.from_TSV(tsv_file, model.glove, "<SOS>", "<EOS>", model.vocab, do_sentiment)
    else:
        model = IDKRephraseModel.from_file(model_path, writer=writer)

    model.set_cuda(use_cuda)
    if dev_tsv_file is not None:
        devset = IDKRephraseDataset.from_TSV(dev_tsv_file, model.glove, "<SOS>", "<EOS>", model.vocab, do_sentiment)

    if train or continue_training:
        if dataset is None:
            dataset = IDKRephraseDataset.from_TSV(tsv_file, model.glove, "<SOS>", "<EOS>", model.vocab)
        model.train_dataset(dataset, devset, batch_size, 0, epochs, model_path)

    while True:
        question = input(": ")
        if question == "quit":
            break
        question = "<SOS> " + question + " <EOS>"
        sentiment = input("Positive (y/n):")
        sentiment_flag = 0
        if sentiment.lower() == "y":
            sentiment_flag = 1
        condensed = input("Condensed (y/n):")
        condensed_flag = 0
        if condensed.lower() == "y":
            condensed_flag = 1
        info(" ".join(model.predict(torch.LongTensor(model.glove.words_to_indices(question.split())), question.split(),
                                    make_attn_graphics=False, sentiment_tensor=torch.Tensor([sentiment_flag, condensed_flag]))))

    if dev_tsv_file is not None:
        #model.get_worst_bleu(devset)
        model.evaluate_bleu(devset, print_predictions=True)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
