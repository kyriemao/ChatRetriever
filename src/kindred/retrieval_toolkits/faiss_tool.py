import numpy as np

import os
import copy
import time
import pickle
import faiss
import torch
from IPython import embed

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FaissTool:
    def __init__(self, 
                 embedding_size: int,
                 index_dir: str,
                 top_n: int,
                 num_split_block:int=1):
        self.n_gpu = torch.cuda.device_count()
        self.embedding_size = embedding_size
        self.index_dir = index_dir
        self.top_n = top_n
        self.num_split_block = num_split_block
        
        self.index = self.build_faiss_index()
        
        
    def build_faiss_index(self):
        logger.info("Building index...")
        ngpu = faiss.get_num_gpus()
        gpu_resources = []
        tempmem = -1

        for i in range(ngpu):
            res = faiss.StandardGpuResources()
            if tempmem >= 0:
                res.setTempMemory(tempmem)
            gpu_resources.append(res)

        cpu_index = faiss.IndexFlatIP(self.embedding_size)  
        index = None
        if ngpu > 0:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.usePrecomputed = False
            # gpu_vector_resources, gpu_devices_vector
            vres = faiss.GpuResourcesVector()
            vdev = faiss.Int32Vector()
            for i in range(0, ngpu):
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, cpu_index, co)
            index = gpu_index
        else:
            index = cpu_index
            logger.warning("Use cpu for faiss!")

        return index

    # faiss dense retrieval on each psg block and merge the results one by one
    def search_on_blocks(self, query_embs: np.ndarray):
        merged_candidate_matrix = None
        block_id = 1
        for emb_block_name in os.listdir(self.index_dir):
            if "emb_block" not in emb_block_name:
                continue  
            logger.info("Loading block {}, ".format(block_id) + emb_block_name)
            psg_emb = None
            psg_emb2id = None
            embid_block_name = emb_block_name.replace("emb_block", "embid_block")

            try:
                with open(os.path.join(self.index_dir, emb_block_name), 'rb') as f:
                    psg_emb = pickle.load(f)
                with open(os.path.join(self.index_dir, embid_block_name), 'rb') as f:
                    psg_emb2id = pickle.load(f)
                    if isinstance(psg_emb2id, list):
                        psg_emb2id = np.array(psg_emb2id)
            except Exception as e:
                logger.error("An unexpected error occurred while loading block " + emb_block_name + ": " + str(e))

            logger.info('passage embedding shape: ' + str(psg_emb.shape))
            logger.info("query embedding shape: " + str(query_embs.shape))
            
            psg_embs = np.array_split(psg_emb, self.num_split_block)
            psg_emb2ids = np.array_split(psg_emb2id, self.num_split_block)
            for split_idx in range(len(psg_embs)):
                psg_emb = psg_embs[split_idx]
                psg_emb2id = psg_emb2ids[split_idx]
                
                logger.info("Adding block {}: {}, split {} into index...".format(block_id, emb_block_name, split_idx))
                psg_emb = psg_emb.astype(np.float32) if psg_emb.dtype != np.float32 else psg_emb
                self.index.add(psg_emb)
                
                # search
                tb = time.time()
                D, I = self.index.search(query_embs, self.top_n)
                elapse = time.time() - tb
                logger.info({
                    'time cost': elapse,
                    'query num': query_embs.shape[0],
                    'time cost per query': elapse / query_embs.shape[0]
                })

                candidate_id_matrix = psg_emb2id[I] # passage_idx -> passage_id
                D = D.tolist()
                candidate_id_matrix = candidate_id_matrix.tolist()
                candidate_matrix = []

                for score_list, passage_list in zip(D, candidate_id_matrix):
                    candidate_matrix.append([])
                    for score, passage in zip(score_list, passage_list):
                        candidate_matrix[-1].append((score, passage))
                    assert len(candidate_matrix[-1]) == len(passage_list)
                assert len(candidate_matrix) == I.shape[0]

                self.index.reset()
                del psg_emb
                del psg_emb2id

                if merged_candidate_matrix == None:
                    merged_candidate_matrix = candidate_matrix
                    continue
                
                # merge with previous search results
                merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
                merged_candidate_matrix = []
                for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                                candidate_matrix):
                    p1, p2 = 0, 0
                    merged_candidate_matrix.append([])
                    while p1 < self.top_n and p2 < self.top_n:
                        if merged_list[p1][0] >= cur_list[p2][0]:
                            merged_candidate_matrix[-1].append(merged_list[p1])
                            p1 += 1
                        else:
                            merged_candidate_matrix[-1].append(cur_list[p2])
                            p2 += 1
                    while p1 < self.top_n:
                        merged_candidate_matrix[-1].append(merged_list[p1])
                        p1 += 1
                    while p2 < self.top_n:
                        merged_candidate_matrix[-1].append(cur_list[p2])
                        p2 += 1
                        
            block_id += 1
    
        merged_D, merged_I = [], []

        for merged_list in merged_candidate_matrix:
            merged_D.append([])
            merged_I.append([])
            for candidate in merged_list:
                merged_D[-1].append(candidate[0])
                merged_I[-1].append(candidate[1])
        merged_D, merged_I = np.array(merged_D), np.array(merged_I)

        logger.debug(merged_D.shape)
        logger.debug(merged_I.shape)

        return merged_D, merged_I

    