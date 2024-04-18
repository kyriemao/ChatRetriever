from IPython import embed

import os
import json
import time
import shutil
import pickle
import random
import numpy as np

import torch
import torch.distributed as dist
torch.multiprocessing.set_sharing_strategy('file_system')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def mkdirs(dir_list: list, force_emptying_dir:bool=False, allow_dir_exist=False):
    for x in dir_list:
        if not os.path.exists(x):
            os.makedirs(x)
        elif len(os.listdir(x)) > 0:    # not empty
            if force_emptying_dir:
                print("Forcing to erase all contens of {}".format(x))
                shutil.rmtree(x)
                os.makedirs(x)
            elif not allow_dir_exist:
                raise FileExistsError
        else:
            continue

def write_running_args(output_dir: str, arg_list: list):
    with open(os.path.join(output_dir, "running_args.txt"), "w") as f:
        f.write("start time: {}".format(time.asctime(time.localtime(time.time()))))
        f.write('\n\n')
        for args in arg_list:
            params = vars(args)
            # only store serializable args
            serializable_params = {}
            for key in params:
                try:
                    json.dumps(params[key])
                    serializable_params[key] = params[key]
                except:
                    continue
            f.write(json.dumps(serializable_params, indent=4))
            f.write("\n\n")
            


def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	logger.info('load path = {} object'.format(path))
	return res

def pstore(x, path, high_protocol = False):
    with open(path, 'wb') as f:
        if high_protocol:  
            pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(x, f)
    logger.info('store object in path = {} ok'.format(path))



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)



def get_qrel_sample_ids(qrel_path: str):
    sample_ids = set()
    with open(qrel_path, 'r') as f:
        for line in f:            
            if '\t' in line:
                sid, _, _, _= line.strip('\n').split('\t')
            else:
                sid, _, _, _= line.strip('\n').split(' ')
            sample_ids.add(sid)
    return sample_ids