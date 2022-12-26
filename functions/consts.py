from pathlib import Path

MODEL_DICT_FN = 'rubert_tiny_state_dict.pkl'
TOKENIZER_FN = 'tokenizer_rubert.pkl'
IDX2ITEM_FN = 'idx2item.pkl'
TARGET2EMB_FN = 'target2emb.pkl'
EMBS_DST_FN = 'emb_distances.pkl'
RESP_FN = 'responsbl.csv'
HISTORY_FN = 'history.csv'

PROB_THR = 0.55
POSS_TYPE_THR = 0.15
N_PHRASES = 5

OUTP_DIM = 312
BATCH_SIZE = 6
N_CATS = 5
MAX_LEN = 100

MODEL_PATH = Path('models')
DATA_PATH = Path('data')
OUTPUT_PATH = Path('output')