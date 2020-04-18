
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import src.config as config
import src.model_utils as mutils
from src.dataset import CustomDataset


def predict(df, model, device, label_list, description_col=config.TEXT_COLUMN):

    test_dataset = CustomDataset(
            description=df[description_col].values
    )
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            sampler=test_sampler,
            num_workers=2
    )

    all_logits = None
    model.eval()

    tk0 = tqdm(test_data_loader, total=len(test_data_loader))
    for step, batch in enumerate(tk0):
        input_ids = batch['input_id']
        input_mask= batch['input_mask']
        segment_ids = batch['segment_id']

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits, _ = model(input_ids, segment_ids, input_mask)
            logits = logits.sigmoid()

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    return pd.merge(df, pd.DataFrame(all_logits, columns=label_list), left_index=True, right_index=True)


def serve_inference(data, label_list: list, model: str = None, description_col=config.TEXT_COLUMN, thresh=None):

    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Data provided needs to be either a file path 'str', or pd.DataFrame "
                         f"object. Found {type(data)}")

    if not model:
        model = mutils.generate_model_filename()
        if not model.is_file():
            raise ValueError(f"Model filename generated does not exist: {model}")

    if (not thresh) or (not isinstance(thresh, float)) or (not isinstance(thresh, dict)):
        print(f"Loading label logit thresholds from {config.LOGIT_THR_FILE}")
        with open(config.LOGIT_THR_FILE) as f:
            thresh = json.load(f)
    else:
        raise ValueError("Threshold object passed needs to be either 'float', 'dict', or None.")

    model = mutils.get_model(which_model=config.CLASSIFIER,
                             num_labels=len(label_list),
                             use_checkpoint=True,
                             model_ckpt=model)

    logits = predict(df, model, config.DEVICE, label_list, description_col)
    preds = mutils.logits_to_discrete(logits, label_list, thresh)

    return logits, preds
