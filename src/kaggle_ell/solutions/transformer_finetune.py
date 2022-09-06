import gc
import math
import os
import time
from typing import Mapping
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import mean_squared_error
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from kaggle_ell.solution import Solution
from kaggle_ell.solution_factory import SolutionFactory

logger = logging.getLogger(__name__)

class TrainDataset(Dataset):
    def __init__(self, cfg, df, tokenizer, target_cols):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[target_cols].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = self.tokenizer(self.texts[item], padding=True, truncation=True, max_length=self.cfg.max_length)
        #label = torch.tensor(self.labels[item], dtype=torch.float)
        inputs['labels'] = self.labels[item]
        return inputs



class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.backbone, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            logger.info(self.config)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.backbone, config=self.config)
        else:
            self.model = AutoModel(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 6)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output

class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores

def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(CFG, fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, inputs in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = inputs['labels']
        del inputs['labels']
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader),
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))

        wandb.log({f"[fold{fold}] loss": losses.val,
                   f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(CFG, valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, inputs in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = inputs['labels']
        del inputs['labels']
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def train_loop(model, folds, fold, train_cfg, data_cfg, artifacts_path, device, tokenizer, target_cols):
    logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[target_cols].values

    train_dataset = TrainDataset(data_cfg, train_folds, tokenizer, target_cols)
    valid_dataset = TrainDataset(data_cfg, valid_folds, tokenizer, target_cols)

    train_loader = DataLoader(train_dataset,
                              batch_size=train_cfg.batch_size,
                              shuffle=True,
                              num_workers=train_cfg.num_workers, pin_memory=True, drop_last=True,
                              collate_fn=DataCollatorWithPadding(tokenizer))
    valid_loader = DataLoader(valid_dataset,
                              batch_size=train_cfg.batch_size * 2,
                              shuffle=False,
                              num_workers=train_cfg.num_workers, pin_memory=True, drop_last=False,
                              collate_fn=DataCollatorWithPadding(tokenizer))

    # ====================================================
    # model & optimizer
    # ====================================================
    model.to(device)
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=train_cfg.encoder_lr,
                                                decoder_lr=train_cfg.decoder_lr,
                                                weight_decay=train_cfg.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=train_cfg.encoder_lr, eps=train_cfg.eps, betas=train_cfg.betas)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles
            )
        return scheduler

    num_train_steps = int(len(train_folds) / train_cfg.batch_size * train_cfg.epochs)
    scheduler = get_scheduler(train_cfg, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.SmoothL1Loss(reduction='mean')  # RMSELoss(reduction="mean")

    best_score = np.inf

    checkpoint_path = os.path.join(artifacts_path, f"fold{fold}_best.pth")

    for epoch in range(train_cfg.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_cfg, fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(train_cfg, valid_loader, model, criterion, device)

        # scoring
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        logger.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        logger.info(f'Epoch {epoch + 1} - Score: {score:.4f}  Scores: {scores}')

        wandb.log({f"[fold{fold}] epoch": epoch + 1,
                   f"[fold{fold}] avg_train_loss": avg_loss,
                   f"[fold{fold}] avg_val_loss": avg_val_loss,
                   f"[fold{fold}] score": score})

        if best_score > score:
            best_score = score
            logger.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                       checkpoint_path)

    predictions = torch.load(checkpoint_path,
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[[f"pred_{c}" for c in train_cfg.target_cols]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds

def get_result(oof_df, target_cols):
    labels = oof_df[target_cols].values
    preds = oof_df[[f"pred_{c}" for c in target_cols]].values
    score, scores = get_score(labels, preds)
    logger.info(f'Score: {score:<.4f}  Scores: {scores}')

# This is still work in progress
@SolutionFactory.register('transformer_finetune')
class TransformerFinetune(Solution):
    def do_train(self, train_data: pd.DataFrame, data_cfg: Mapping, train_cfg: Mapping, model_cfg: Mapping, env_cfg: Mapping):

        train = self.competition_data_manager.load_train_data()
        
        Fold = MultilabelStratifiedKFold(n_splits=train_cfg.n_fold, shuffle=True, random_state=train_cfg.seed)
        target_cols = self.competition_data_manager.LABEL_COLUMNS
        for n, (train_index, val_index) in enumerate(Fold.split(train, train[target_cols])):
            train.loc[val_index, 'fold'] = int(n)
        train['fold'] = train['fold'].astype(int)

        tokenizer = AutoTokenizer.from_pretrained(model_cfg.backbone)
        tokenizer.save_pretrained(os.path.join(env_cfg.artifacts_path, 'tokenizer'))

        oof_df = pd.DataFrame()
        for fold in range(train_cfg.n_fold):
            if fold in train_cfg.trn_fold:
                model = CustomModel(model_cfg, config_path=None, pretrained=True)
                torch.save(model.config, os.path.join(env_cfg.artifacts_path, 'config.pth'))

                _oof_df = train_loop(model, train, fold, train_cfg, data_cfg, env_cfg.artifacts_path, env_cfg.device,
                                     tokenizer, target_cols)
                oof_df = pd.concat([oof_df, _oof_df])
                logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df, target_cols)
        oof_df = oof_df.reset_index(drop=True)
        logger.info(f"========== CV ==========")
        get_result(oof_df, target_cols)
        oof_df.to_pickle(os.path.join(env_cfg.artifacts_path, 'oof_df.pkl'))

    def do_predict(self, input_data: pd.DataFrame, inference_cfg: Mapping, artifacts_path: str):
        pass