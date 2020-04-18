
import torch
import numpy as np
from tqdm import tqdm

import src.config as config
from src.model_utils import AverageMeter, fbeta, roc_auc_score, warmup_linear


def train_epoch(data_loader, model, optimizer, device, scheduler, **kwargs):
    model.train()  # sets model in training mode
    model.zero_grad()  # clears old gradients from the last step

    losses = AverageMeter()
    scores = AverageMeter()

    total_iterations = len(data_loader)
    tk0 = tqdm(data_loader, total=total_iterations)

    for step, batch in enumerate(tk0):

        input_ids = batch["input_id"]
        input_mask = batch["input_mask"]
        segment_ids = batch["segment_id"]
        label_ids = batch["label_ids"]

        # send tensors to device
        input_ids = input_ids.to(device, dtype=torch.long)
        input_mask = input_mask.to(device, dtype=torch.long)
        segment_ids = segment_ids.to(device, dtype=torch.long)
        label_ids = label_ids.to(device, dtype=torch.long)

        # forward
        outputs, loss = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=label_ids)

        if kwargs['n_gpu'] > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        loss.backward()

        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # modify learning rate with special warm up BERT uses
            global_step = (total_iterations * kwargs['epoch']) + (step + 1)
            lr_this_step = config.LEARNING_RATE * warmup_linear(global_step / kwargs['num_train_steps'],
                                                                config.WARMUP_PROPORTION)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # compute fbeta score for training samples
        fbeta_score = fbeta(outputs, label_ids, thresh=0.3, beta=2)

        # keep track of losses and scores
        losses.update(loss.item(), input_ids.size(0))
        scores.update(fbeta_score, input_ids.size(0))
        tk0.set_postfix(loss=losses.avg, score=scores.avg)


def evaluate(data_loader, model, device, label_list):

    all_outputs = None
    all_labels = None

    model.eval()
    losses = AverageMeter()
    scores = AverageMeter()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, batch in enumerate(tk0):
            input_ids = batch["input_id"]
            input_mask = batch["input_mask"]
            segment_ids = batch["segment_id"]
            label_ids = batch["label_ids"]

            # send tensors to device
            input_ids = input_ids.to(device, dtype=torch.long)
            input_mask = input_mask.to(device, dtype=torch.long)
            segment_ids = segment_ids.to(device, dtype=torch.long)
            label_ids = label_ids.to(device, dtype=torch.long)

            # forward
            outputs, loss = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=label_ids)

            eval_score = fbeta(outputs, label_ids, thresh=0.3, beta=2.0)

            if all_outputs is None:
                all_outputs = outputs.detach().cpu().numpy()
            else:
                all_outputs = np.concatenate((all_outputs, outputs.detach().cpu().numpy()), axis=0)

            if all_labels is None:
                all_labels = label_ids.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)

            # keep track of losses and scores
            losses.update(loss.item(), input_ids.size(0))
            scores.update(eval_score, input_ids.size(0))
            tk0.set_postfix(loss=losses.avg, score=scores.avg)

    eval_loss = losses.avg
    eval_score = scores.avg
    eval_auc_score = roc_auc_score(all_outputs, all_labels, label_list)

    return eval_loss, eval_score, eval_auc_score
