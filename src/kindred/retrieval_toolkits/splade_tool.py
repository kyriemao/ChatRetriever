import os
import h5py
import json
import time
import numba
import array
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from IPython import embed

from kindred.utils import pload, pstore
import torch
from torch.utils.data import IterableDataset

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()

class IndexDictOfArray:
    def __init__(self, sample_ids_filename="sample_ids.pkl", array_filename="array_index.h5py"):
        self.sample_ids_filename = sample_ids_filename
        self.array_filename = array_filename
        
        self.new()
    
    def __len__(self):
        return len(self.index_doc_id)
    
    
    def new(self):
        self.n = 0
        self.sample_ids = []
        logger.info("initializing new index...")
        self.index_doc_id = defaultdict(lambda: array.array("I"))
        self.index_doc_value = defaultdict(lambda: array.array("f"))
        
        
    def load(self, index_dir):
        self.file = h5py.File(os.path.join(index_dir, self.array_filename), "r")
        dim = self.file["dim"][()]
        
        self.index_doc_id = dict()
        self.index_doc_value = dict()
        for key in tqdm(range(dim)):
            try:
                self.index_doc_id[key] = np.array(self.file["index_doc_id_{}".format(key)],
                                                    dtype=np.int32)
                # ideally we would not convert to np.array() but we cannot give pool an object with hdf5
                self.index_doc_value[key] = np.array(self.file["index_doc_value_{}".format(key)],
                                                        dtype=np.float32)
            except:
                self.index_doc_id[key] = np.array([], dtype=np.int32)
                self.index_doc_value[key] = np.array([], dtype=np.float32)
        self.file.close()
        del self.file
        
        self.sample_ids = pload(os.path.join(index_dir, self.sample_ids_filename)) # all passage ids
        self.n = len(self.sample_ids)
        
        logger.info("done loading index...")
        
                
    def add_psg_lexical_embs(self, psg_embs:np.ndarray, sample_ids:list):
        row, col = np.nonzero(psg_embs)
        val = psg_embs[row, col].tolist()
        
        start_psg_id = self.n
        psg_ids = row + start_psg_id
        for psg_id, dim_id, value in zip(psg_ids, col, val):
            self.index_doc_id[dim_id].append(psg_id)
            self.index_doc_value[dim_id].append(value)
        
        self.n += len(sample_ids)
        self.sample_ids += sample_ids 
    
    def merge(self, other_index):
        # Note that we merge for index_doc_id, index_doc_value in int32 ndarray. But if merge in unit8, it will fail.
        for dim_id in tqdm(other_index.index_doc_id, desc="Merging splade index with another splade index..."):
            doc_id_of_a_dim_of_other_index = other_index.index_doc_id[dim_id] + self.n
            if dim_id not in self.index_doc_id:
                self.index_doc_id[dim_id] = doc_id_of_a_dim_of_other_index
                self.index_doc_value[dim_id] = other_index.index_doc_value[dim_id]
            else:
                self.index_doc_id[dim_id] = np.concatenate([self.index_doc_id[dim_id], doc_id_of_a_dim_of_other_index])
                self.index_doc_value[dim_id] = np.concatenate([self.index_doc_value[dim_id], other_index.index_doc_value[dim_id]])

        self.n += other_index.n
        self.sample_ids += other_index.sample_ids
        
    def save(self, index_dir: str):
        logger.info("converting to numpy")
        for key in tqdm(list(self.index_doc_id.keys())):
            self.index_doc_id[key] = np.array(self.index_doc_id[key], dtype=np.int32)
            self.index_doc_value[key] = np.array(self.index_doc_value[key], dtype=np.float32)
        
        logger.info("save to disk...")
        with h5py.File(os.path.join(index_dir, self.array_filename), "w") as f:
            
            f.create_dataset("dim", data=len(self.index_doc_id.keys()))
            for key in tqdm(self.index_doc_id.keys()):
                f.create_dataset("index_doc_id_{}".format(key), data=self.index_doc_id[key])
                f.create_dataset("index_doc_value_{}".format(key), data=self.index_doc_value[key])
            f.close()
            
        logger.info("saving index distribution...")  # => size of each posting list in a dict
        index_dist = {}
        for k, v in self.index_doc_id.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(index_dir, "index_dist.json"), "w"))

        logger.info("saving sample ids...")
        pstore(self.sample_ids, os.path.join(index_dir, "sample_ids.pkl"), high_protocol=True)
        

class SpladeRetrievalTool:
    """retrieval from SparseIndexing
    """

    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,
                          inverted_index_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        n = len(indexes_to_retrieve)

        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            if local_idx in inverted_index_ids:
                retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
                retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
                for j in numba.prange(len(retrieved_indexes)):
                    scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]
    
    def __init__(self,
                 index_dir: str,
                 top_n: int):
        self.splade_index = IndexDictOfArray()
        self.splade_index.load(index_dir)
        self.top_n = top_n

        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.splade_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.splade_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value
        
    
    def retrieve(self, query_embs: np.ndarray):
        scores_mat, psg_ids_mat = [], []
        for query_emb in tqdm(query_embs, desc="Splade retrieving..."):
            query_emb = query_emb.reshape(1, -1)
            row, col = np.nonzero(query_emb)
            values = query_emb[row, col]
            threshold = 0
            
            filtered_indexes, scores = self.numba_score_float(self.numba_index_doc_ids,
                                                              self.numba_index_doc_values,
                                                              col,
                                                              values,
                                                              threshold=threshold,
                                                              size_collection=self.splade_index.n)
            
            
            # threshold set to 0 by default, could be better
            filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=self.top_n)
            indices_ = np.argsort(scores)[::-1]
            sorted_scores = scores[indices_]
            sorted_filtered_indexes = filtered_indexes[indices_]
 
            sample_scores, sample_psg_ids = [], []
            for id_, sc in zip(sorted_filtered_indexes, sorted_scores):
                sample_psg_ids.append(str(self.splade_index.sample_ids[id_]))
                sample_scores.append(float(sc))
            scores_mat.append(sample_scores)
            psg_ids_mat.append(sample_psg_ids)

        return np.array(scores_mat), np.array(psg_ids_mat)

    def get_psg_lexical_emb(self, raw_psg_idx):
        psg_idx = -1
        for i, x in enumerate(self.splade_index.sample_ids):
            if x == raw_psg_idx:
                psg_idx = i
                break
    
        psg_lexical_emb = np.zeros(30522)
        for dim_id in self.splade_index.index_doc_id:
            if psg_idx in self.splade_index.index_doc_id[dim_id]:
                idx = np.where(self.splade_index.index_doc_id[dim_id] == psg_idx)[0][0]
                psg_lexical_emb[dim_id] = self.splade_index.index_doc_value[dim_id][idx]

        return psg_lexical_emb
    
PyTorch_over_1_6 = float((torch.__version__.split('.')[1])) >= 6 and float((torch.__version__.split('.')[0])) >= 1

class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass

class MixedPrecisionManager:
    def __init__(self, activated, use_cuda):
        assert (not activated) or PyTorch_over_1_6, "Cannot use AMP for PyTorch version < 1.6"

        print("Using FP16:", activated)
        self.activated = activated and use_cuda
        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer):
        if self.activated:
            self.scaler.unscale_(optimizer)
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()