# Data

We provide some basic datasets and data collators in `dataset.py`:
- MatchingDataset
- MatchingCollator
- IndexCollator: it is for indexing passages.
- json_psg_generator: to generate IterableDataset for loading large jsonl-format passage collections when indexing.


## Create your new dataset
1. Create `your_dataset.py` and implement your dataset class. 

2. Write your data loading logic in `load_data_from_file` function. 

3. Register your dataset into `load_dataset` function in `data_utils.py`
