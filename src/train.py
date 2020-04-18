
import json
import torch
import pandas as pd

from sklearn import metrics
from scipy.sparse import csr_matrix

import src.config as config
import src.dataset_utils as dutils
import src.model_utils as mutils
import src.engine as engine
import src.serve as serve
from src.dataset import CustomDataset

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():

    mutils.set_seeds(config.SEED)

    df_train = pd.read_csv(config.TRAIN_SET_FILE)
    df_valid = pd.read_csv(config.VALID_SET_FILE)

    # < if any extra train data processing needed (e.g. data augmentation), insert here >
    
    # retrieve columns of interest - description, and target column
    df_train = df_train[[config.TEXT_COLUMN, config.TARGET]]
    df_valid = df_valid[[config.TEXT_COLUMN, config.TARGET]]

    df_train.columns = ['description', 'target']
    df_valid.columns = ['description', 'target']

    # pivot target column into N one-hot binary columns
    df_train = dutils.multi_label_binarize(df_train)
    df_valid = dutils.multi_label_binarize(df_valid)

    LABEL_LIST = list(filter(lambda x: x != 'description', df_train.columns))
    NUM_LABELS = len(LABEL_LIST)

    # print some sense checks
    print(df_train.head())
    print(df_valid.head())
    
    # train batch size
    train_batch_size = int(config.TRAIN_BATCH_SIZE / config.GRADIENT_ACCUMULATION_STEPS)
    
    # DATA LOADERS
    train_dataset = CustomDataset(
            description=df_train['description'].values,
            target=df_train[LABEL_LIST].values
    )
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            num_workers=4
    )

    valid_dataset = CustomDataset(
            description=df_valid['description'].values,
            target=df_valid[LABEL_LIST].values
    )
    valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            num_workers=2
    )

    num_train_steps = int(len(df_train) / train_batch_size * config.NUM_EPOCHS)

    # Setup GPU parameters
    device = config.DEVICE
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.SEED)

    print("device: {} n_gpu: {}, 16-bits training: {}".format(device, n_gpu, config.FP16))

    # Get Model
    model = mutils.get_model(config.CLASSIFIER, NUM_LABELS, config.USE_CHECKPOINT, config.MODEL_CKPT)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare optimizer
    optimizer = AdamW(mutils.optimizer_params(model),
                      lr=config.LEARNING_RATE,
                      correct_bias=False
                      )
    num_warmup_steps = num_train_steps * config.WARMUP_PROPORTION
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps)

    print("Start training")
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        extra_train_args = {'epoch': epoch, 'num_train_steps': num_train_steps, 'n_gpu': n_gpu}
        engine.train_epoch(train_data_loader, model, optimizer, device, scheduler, **extra_train_args)
        eval_loss, eval_score, eval_auc_score = engine.evaluate(valid_data_loader, model, device, LABEL_LIST)

        print(f"Validation Results after epoch {epoch+1}/{config.NUM_EPOCHS}: ")
        print(f"loss = {eval_loss}")
        print(f"score = {eval_score}")

        for label, auc_score in eval_auc_score.items():
            if label == 'micro':
                print(f"auc_score total micro average = {auc_score}")
            else:
                print(f"auc_score for {label} = {auc_score}")

    print("Training finished")

    print("Overall Validation results using non-optimal global thr of 0.5: ")
    valid_logits = serve.predict(df_valid, model, device, LABEL_LIST)
    valid_preds = mutils.logits_to_discrete(valid_logits, LABEL_LIST, 0.5)
    print(metrics.classification_report(
            csr_matrix(df_valid[LABEL_LIST].values.tolist()),
            csr_matrix(valid_preds[LABEL_LIST].values.tolist()),
            target_names=LABEL_LIST))

    print("Optimizing logit thresholds against training data")
    thr_score_fn, lower_better, extra_args = mutils.get_thr_optimization_params(config.OPTIMIZE_LOGIT_THR_FOR,
                                                                                config.OPTIMIZE_LOGIT_THR_INDEP)
    train_logits = serve.predict(df_train, model, device, LABEL_LIST)
    best_threshold = mutils.get_optimal_threshold(df_train,
                                                  train_logits,
                                                  LABEL_LIST,
                                                  thr_score_fn,
                                                  config.OPTIMIZE_LOGIT_THR_INDEP,
                                                  lower_better,
                                                  **extra_args if extra_args else None)
    print(f"Optimized thresholds: {best_threshold}")
    valid_preds = mutils.logits_to_discrete(valid_logits, LABEL_LIST, best_threshold)

    print("Overall results after thr optimization: ")
    print(metrics.classification_report(
            csr_matrix(df_valid[LABEL_LIST].values.tolist()),
            csr_matrix(valid_preds[LABEL_LIST].values.tolist()),
            target_names=LABEL_LIST))

    print("Saving model, optimized thresholds, and validation results.")
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save model itself
    model_filename = mutils.generate_model_filename()
    torch.save(model_to_save.state_dict(), model_filename)
    print(f"Model checkpoint saved to -> {model_filename}")

    best_thr_filename = config.OUTPUT_MODEL_PATH / config.LOGIT_THR_FILE
    print(f"Saving thresholds to -> {best_thr_filename}")
    with open(best_thr_filename, 'w') as f:
        json.dump(best_threshold, f)

    print("Done. Exiting Application.")


if __name__ == "__main__":
    run()
