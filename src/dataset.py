
import torch
import src.config as config


def process_data(description, target, tokenizer, max_len):

    tokens_desc = tokenizer.tokenize(description)
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_desc) > max_len - 2:
        tokens_desc = tokens_desc[:(max_len - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_desc + ["[SEP]"]
    segment_id = [0] * len(tokens)

    input_id = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_id)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_len - len(input_id))
    input_id += padding
    input_mask += padding
    segment_id += padding

    if target:
        label_ids = list(target.astype(float))
        data_out = {
            'input_id':    input_id,
            'input_mask':  input_mask,
            'segment_id':  segment_id,
            'label_ids':   label_ids
        }
    else:
        data_out = {
            'input_id':   input_id,
            'input_mask': input_mask,
            'segment_id': segment_id
        }

    return data_out


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, description, target=None):
        self.description = description
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_SEQ_LENGTH

    def __len__(self):
        return len(self.description)

    def __getitem__(self, item):
        if self.target:
            data = process_data(
                    self.description[item],
                    self.target[item],
                    self.tokenizer,
                    self.max_len
            )
            output = {
                'input_id':     torch.tensor(data["input_id"], dtype=torch.long),
                'input_mask':   torch.tensor(data["input_mask"], dtype=torch.long),
                'segment_id':   torch.tensor(data["segment_id"], dtype=torch.long),
                'label_ids':    torch.tensor(data["label_ids"], dtype=torch.long),
            }
        else:
            data = process_data(
                    self.description[item],
                    self.target,  # passing targets as None
                    self.tokenizer,
                    self.max_len
            )
            output = {
                'input_id':   torch.tensor(data["input_id"], dtype=torch.long),
                'input_mask': torch.tensor(data["input_mask"], dtype=torch.long),
                'segment_id': torch.tensor(data["segment_id"], dtype=torch.long)
            }

        return output
