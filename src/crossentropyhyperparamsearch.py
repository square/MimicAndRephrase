import csv

import io
import os

import torch
from typing import Dict, List

from attncpy import IDKRephraseModel
from data import IDKRephraseDataset
from utils.log import track, info, warn, error, init
import numpy as np
import gzip
import sys

from utils.word_embedding import Glove


class CrossEntropyHyperparamSearch:

    def __init__(self, run_params: Dict, previous_runs: Dict=None, writer=None):
        if previous_runs is not None:
            self.previous_runs = previous_runs
        else:
            self.previous_runs = []

        self.run_params = run_params
        self.dim_params = run_params["dim_params"]
        self.const_params = run_params["const_params"]
        self.glove = Glove.from_binary()
        misc_tokens = ["<SOS>", "<EOS>"]
        self.const_params['misc_tokens'] = misc_tokens
        self.const_params['vocab'] = IDKRephraseModel.get_vocab_from_list_and_files(
            None, 0, [run_params["train"], run_params["dev"]], misc_tokens)
        self.devset = run_params["dev"]
        self.dataset = run_params["train"]
        self.batch_size = run_params["batch_size"]
        self.n_samples = run_params["n_samples"]
        self.elite_samples = run_params["elite_samples"]
        self.run_directory = run_params["run_directory"]
        self.mean = run_params.get("mean", 0.5*np.ones(len(self.dim_params)))
        self.covariance = run_params.get("covariance", 0.3 * np.identity(len(self.dim_params)))
        self.best_loss = run_params.get("best_loss", None)
        self.use_cuda = run_params["use_cuda"]
        self.writer = writer

    def search_params(self, n_sets=7):
        for sample_it in range(n_sets):
            coordinate_samples = np.random.multivariate_normal(self.mean, self.covariance, self.n_samples)
            coordinate_samples = np.clip(coordinate_samples, 0, 0.999999)
            scores = np.zeros(self.n_samples)
            for i in range(len(coordinate_samples)):
                scores[i] = self.train_with_params(coordinate_samples[i])[0]
                self.save(save_directory=self.run_directory)
                if self.writer:
                    for j in range(len(self.dim_params)):
                        self.writer.add_scalar(self.dim_params[j]["name"],
                                               self.dim_params[j]["transform"](coordinate_samples[i][j]), i +
                                               sample_it*len(coordinate_samples))
                    self.writer.add_scalar("BLEU Score", scores[i], i + sample_it*len(coordinate_samples))
                info("Iteration: " + str(sample_it + 1))
                info("Sample: " + str(i + 1) + "/" + str(self.n_samples))
                info("Trained model that achieved BLEU score of " + str(scores[i]))
                best_run = self.get_best_run()
                info("Best run so far: " + str(best_run[0]) + " with score of " + str(best_run[1]))

            sample_tuples = [(coordinate_samples[i], scores[i]) for i in range(len(coordinate_samples))]
            sample_tuples = sorted(sample_tuples, key=lambda sample: -sample[1])
            elite_samples = sample_tuples[:self.elite_samples]
            elite_coordinates = np.array([elite_sample[0] for elite_sample in elite_samples])
            self.covariance = np.cov(elite_coordinates.T)
            self.mean = np.mean(elite_coordinates, axis=0)
            self.run_params["covariance"] = self.covariance
            self.run_params["mean"] = self.mean

    def get_best_run(self):
        index = max(enumerate(self.previous_runs), key=lambda x: x[1][1])[0]
        return (self.run_directory + "model" + str(index) + ".pkl.gz",
                self.previous_runs[index][1], self.previous_runs[index][0])

    def save(self, save_directory: str, compress: bool=True):
        save_data = {
            "params": self.run_params,
            "previous_runs": self.previous_runs
        }
        bytes_io = io.BytesIO()
        torch.save(save_data, bytes_io)
        model_bytes = bytes_io.getvalue()
        path = save_directory + "searchdata.pkl.gz"
        if compress:
            with gzip.open(path, "wb+") as f:
                f.write(model_bytes)
        else:
            with open(path, "wb+") as f:
                f.write(model_bytes)

    def train_with_params(self, coordinate: np.array):
        model_params = {}
        for i in range(len(self.dim_params)):
            model_params[self.dim_params[i]["name"]] = self.dim_params[i]["transform"](coordinate[i])
        model_params.update(self.const_params)
        model_params_for_printing = model_params.copy()
        model_params_for_printing["vocab"] = ["omitted"]
        info("Training model with model params " + str(model_params_for_printing))
        try:
            model = IDKRephraseModel(model_params, self.glove)
            model.set_cuda(self.use_cuda)
            devset = IDKRephraseDataset.from_TSV(self.devset, model.glove, "<SOS>", "<EOS>", model.vocab)
            dataset = IDKRephraseDataset.from_TSV(self.dataset, model.glove, "<SOS>", "<EOS>", model.vocab)
            score = model.train_dataset(dataset, devset, self.batch_size, 0, 70, self.run_directory + "model"
                                        + str(len(self.previous_runs)) + ".pkl.gz")
        except Exception as e:
            error("Model failed to train: ", e)
            score = 0

        if self.use_cuda:
            torch.cuda.empty_cache()

        self.previous_runs.append((coordinate, score))
        return score, len(self.previous_runs) - 1

    @classmethod
    def read_runs(cls, params_file: str, writer=None):
        info("Loading previous runs from file: {}".format(params_file))
        if params_file.endswith(".pkl.gz"):
            with gzip.open(params_file, "rb") as f:
                return CrossEntropyHyperparamSearch.load_from_bytes(f.read(), writer)
        elif params_file.endswith(".pkl"):
            with open(params_file, "rb") as f:
                return CrossEntropyHyperparamSearch.load_from_bytes(f.read(), writer)
        else:
            raise ValueError("Invalid file due to unsupported extension: {}".format(params_file))
        pass

    @classmethod
    def load_from_bytes(cls, content: bytes, writer=None):
        with io.BytesIO(content) as f:
            run_data = torch.load(f, map_location=lambda storage, loc: storage)

        return CrossEntropyHyperparamSearch(run_data["params"], run_data["previous_runs"], writer)


class HyperparamTransform:
    def __init__(self, min_val: float=0, max_val: float=1, make_int: bool=False, make_bool: bool=False,
                 log_space: bool=False, log_base: int=10):
        self.min_val = min_val
        self.max_val = max_val
        self.make_int = make_int
        self.make_bool = make_bool
        self.log_space = log_space
        self.log_base = log_base

    def __call__(self, value: float):
        if self.make_bool:
            return value > 0.5
        transformed = (1 - value)*self.min_val + value*self.max_val
        if self.log_space:
            transformed = self.log_base ** transformed
        if self.make_int:
            transformed = int(np.floor(transformed))
        return transformed


def main(run_directory: str, dev_tsv_file: str=None, tsv_file: str=None, reload=False, use_cuda=True):
    init(run_directory + "log/out.txt")

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), "tensorboard"))

    if reload:
        searcher = CrossEntropyHyperparamSearch.read_runs(run_directory + "searchdata.pkl.gz", writer=writer)
    else:
        search_params = {
            "batch_size": 8,
            "n_samples": 21,
            "elite_samples": 6,
            "run_directory": run_directory,
            "dev": dev_tsv_file,
            "train": tsv_file,
            "use_cuda": use_cuda
        }
        search_params["const_params"] = {
            "use_copy": True,
            "bidirectional": True
        }
        search_params["dim_params"] = [{
            "name": "num_unks",
            "transform": HyperparamTransform(min_val=2, max_val=6.5, make_int=True, log_space=True, log_base=2)
        }, {
            "name": "lr",
            "transform": HyperparamTransform(min_val=-3.15, max_val=-2.3, log_space=True)
        }, {
            "name": "hidden_size",
            "transform": HyperparamTransform(min_val=150, max_val=650, make_int=True)
        }, {
            "name": "n_layers",
            "transform": HyperparamTransform(min_val=1, max_val=3, make_int=True)
        }, {
            "name": "dropout",
            "transform": HyperparamTransform(min_val=0.1, max_val=0.8)
        }, {
            "name": "bidirectional",
            "transform": HyperparamTransform(make_bool=True)
        }, {
            "name": "attention_size",
            "transform": HyperparamTransform(min_val=100, max_val=650, make_int=True)
        }, {
            "name": "copy_attn_size",
            "transform": HyperparamTransform(min_val=100, max_val=650, make_int=True)
        }, {
            "name": "copy_extra_layer",
            "transform": HyperparamTransform(make_bool=True)
        }, {
            "name": "attn_extra_layer",
            "transform": HyperparamTransform(make_bool=True)
        }, {
            "name": "copy_extra_layer_size",
            "transform": HyperparamTransform(min_val=50, max_val=650, make_int=True)
        }, {
            "name": "attn_extra_layer_size",
            "transform": HyperparamTransform(min_val=50, max_val=650, make_int=True)
        }]
        searcher = CrossEntropyHyperparamSearch(run_params=search_params, writer=writer)
    searcher.search_params(n_sets=50)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
