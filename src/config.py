
import torch
import os
from transformers import BertTokenizer
from pathlib import Path


TASK_NAME = "your-task"  # insert task name here


# DATA CONFIGURATION ###################
TEXT_COLUMN = "<whatever>"  # specify text column here (input for model)
TARGET = "<whatever>"  # specify target here


# MODEL CONFIGURATION ###################
TRAIN_SIZE = -1  # -1 for using all train samples, else N
VALID_SIZE = -1  # -1 for using all valid samples, else N
TRAIN_BATCH_SIZE = 16  # if OOM during training, try lower values
VALID_BATCH_SIZE = 16   # if OOM during evaluation, try lower values

# IMPORTANT, if attempting to train again, check this by running
# > max(map(dataset_utils.get_description_length, df_train[config.TEXT_COLUMN]))
MAX_SEQ_LENGTH = 160

CLASSIFIER = "bert-base-uncased"  # specify here the classifier to use
DO_LOWER_CASE = True  # set to False when using case sensitive tokenizer
USE_CHECKPOINT = False  # use provided model checkpoint to continue training
MODEL_CKPT = "model-ckpt.bin"  # if finetuning from checkpoint, specify ckpt filename here
LEARNING_RATE = 3e-5
NUM_EPOCHS = 15
WARMUP_PROPORTION = 0.0  # proportion of num_train_steps to do warmup
MAX_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 1
USE_GPU = False  # set to true when GPUs available. Note: if True, and no GPU avail, it'll use cpu
SEED = 42
OPTIMIZE_ON_CPU = False
FP16 = False  # Not implemented. Left it as TODO to incorporate NVIDIA's apex library
LOSS_SCALE = 128
OPTIMIZE_LOGIT_THR_FOR = 'f1'  # either 'accuracy', 'precision', 'recall', 'f1', 'fbeta'. Recommended = f1
OPTIMIZE_LOGIT_THR_INDEP = True  # if True, logit thr will be optimized per each label independently. Recommended = True
FBETA_B = 2.0  # if any fbeta function used (incl OPTIMIZE_LOGIT_THR_FOR), use this value for Beta

TOKENIZER = BertTokenizer.from_pretrained(CLASSIFIER, do_lower_case=DO_LOWER_CASE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")


# FILE CONFIGURATION ###################
MODEL_VERSION = "1.0.0"
COMMENTS = ""  # suffix to be added on the model filename as extra comments

SOURCE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SOURCE_DIR.parent
DATA_PATH = PROJECT_DIR / 'data'
INPUT_DATA_PATH = DATA_PATH / 'input'
MODEL_PATH = DATA_PATH / 'model'
RESOURCES_PATH = DATA_PATH / 'resources'

TRAIN_SET_FILE = INPUT_DATA_PATH / 'train.csv'  # insert train set file here
VALID_SET_FILE = INPUT_DATA_PATH / 'valid.csv'  # insert valid set file here

OUTPUT_MODEL_PATH = MODEL_PATH / 'output'
OUTPUT_MODEL_PATH.mkdir(exist_ok=True)

LOGIT_THR_FILE = f'{TASK_NAME}_{CLASSIFIER}_{MODEL_VERSION}_logit_thresholds.json'

