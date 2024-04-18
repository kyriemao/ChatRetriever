from dataclasses import dataclass
from typing import List

@dataclass
class MatchingSample:
    sample_idx: str
    query: str=None
    rewrite: str = None
    history: List[str]=None
    
    pos_psg: str = None# can only have 1 positive passage per sample
    neg_psgs: List[str]=None
    
    # below is for input
    session: str = None