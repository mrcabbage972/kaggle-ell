import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, List, Dict, Any
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import mean_squared_error
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, EarlyStoppingCallback, \
    DataCollator, Seq2SeqTrainer, Seq2SeqTrainingArguments
import datasets
import numpy as np

from kaggle_ell.solution import Solution
from kaggle_ell.solution_factory import SolutionFactory
from kaggle_ell.solutions.transformer_finetune import MCRMSE

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# prepares labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
@dataclass
class T2TDataCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([(example['input_ids']) for example in batch])
        labels = torch.stack([(example['labels']) for example in batch])
        labels[labels[:, :] == 0] = -100
        attention_mask = torch.stack([(example['attention_mask']) for example in batch])
        decoder_attention_mask = torch.stack([(example['decoder_attention_mask']) for example in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'decoder_attention_mask': decoder_attention_mask
        }


# process the examples in input and target text format and the eos token at the end
def add_input_texts(row):
    row['source_text'] = 'regression: {} source: {}'.format(row['task'], row['full_text'])
    row['target_text'] = str(row['score'])
    return row

# tokenize the examples
def convert_to_features(example_batch, tokenizer, data_cfg):
    input_encodings = tokenizer.batch_encode_plus(example_batch['source_text'],
                                                  padding=True,
                                                  max_length=data_cfg.source_max_length,
                                                  truncation=True)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'],
                                                   padding=True,
                                                   max_length=data_cfg.target_max_length,
                                                   truncation=True)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings


def get_train_test_split(train_all_ds, val_fold_idx, data_cfg):
    split_ds = {'train': train_all_ds.filter(lambda row: row['fold'] != val_fold_idx),
                'test': train_all_ds.filter(lambda row: row['fold'] == val_fold_idx)}

    if data_cfg.max_train_samples > 0:
        split_ds['train'] = split_ds['train'].select(range(data_cfg.max_train_samples))

    if data_cfg.max_val_samples > 0:
        split_ds['test'] = split_ds['test'].select(range(data_cfg.max_val_samples))

    return split_ds


def transform_datasets_for_train(split_ds, tokenizer, data_cfg):
    columns = ['input_ids', 'labels', 'attention_mask', 'decoder_attention_mask']
    encoded_ds = {}
    for key, ds in split_ds.items():
        #remove_cols = set(ds.features.keys()) - set(['input_ids', 'attention_mask', 'label', 'token_type_ids'])
        #remove_columns = remove_cols
        encoded_ds[key] = ds.map(lambda x: convert_to_features(x, tokenizer, data_cfg), batched=True)
        encoded_ds[key].set_format(type='torch', columns=columns)

    return encoded_ds

def is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def decoded2numeric(decoded_labels):
    return [float(x) if is_float(x) else 2.5 for x in decoded_labels]


def get_score_mse(y_trues, y_preds):
    score = mean_squared_error(y_trues, y_preds)
    return score

def get_score_mcrmse(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores

def get_result(oof_df, target_cols):
    labels = oof_df[target_cols].values
    preds = oof_df[[f"pred_{c}" for c in target_cols]].values
    score, scores = get_score_mcrmse(labels, preds)
    logger.info(f'Score: {score:<.4f}  Scores: {scores}')

def compute_metrics(tokenizer, eval_pred):
    predictions, labels = eval_pred

    unpadded_labels = len(labels) * [None]
    for idx, label in enumerate(labels):
        pad_idx = np.where(label <= 1)
        if len(pad_idx[0]) == 0:
            unpadded = label
        else:
            unpadded = label[:pad_idx[0].min()]
        unpadded_labels[idx] = unpadded

    decoded_labels = tokenizer.batch_decode(unpadded_labels)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    numeric_labels = decoded2numeric(decoded_labels)
    numeric_preds = decoded2numeric(decoded_preds)
    
    return {
        'mse': get_score_mse(numeric_labels, numeric_preds)
    }


def del_old_checkpoints(best_model_checkpoint_path):
    parent_dir = Path(best_model_checkpoint_path).parent.absolute()
    basename = os.path.basename(best_model_checkpoint_path)
    for subdir in os.listdir(parent_dir):
        if subdir.startswith('checkpoint') and not subdir == basename:
            logger.info(f'deleting dir {subdir} in {parent_dir}')
            shutil.rmtree(os.path.join(parent_dir, subdir))

def gather_results(group):
    pass

def train_loop(model, train_ds, fold_idx, train_cfg, data_cfg, artifacts_path, device, tokenizer, target_cols):
    raw_ds_dict = get_train_test_split(train_ds, fold_idx, data_cfg)
    ds_dict = transform_datasets_for_train(raw_ds_dict, tokenizer, data_cfg)

    train_args = Seq2SeqTrainingArguments(
        f"model{fold_idx}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=train_cfg.learning_rate,
        per_device_train_batch_size=train_cfg.train_batch_size,
        per_device_eval_batch_size=train_cfg.eval_batch_size,
        num_train_epochs=train_cfg.epochs,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        save_total_limit=1,
        log_level='error',
        optim=train_cfg.optim,
        fp16=train_cfg.fp16,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        generation_max_length=data_cfg.target_max_length,
        predict_with_generate=True
    )

    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=ds_dict["train"].shuffle(train_cfg.seed),
        eval_dataset=ds_dict["test"],
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(tokenizer, x),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_cfg.early_stopping_patience)],
        data_collator=T2TDataCollator()
    )

    trainer.evaluate()
    trainer.train()

    os.remove(os.path.join(trainer.state.best_model_checkpoint, 'optimizer.pt'))
    del_old_checkpoints(trainer.state.best_model_checkpoint)

    raw_preds = trainer.predict(ds_dict["test"]).predictions
    oof_decoded = tokenizer.batch_decode(raw_preds, skip_special_tokens=True)
    oof_preds = decoded2numeric(oof_decoded)
    preds_df = raw_ds_dict["test"].to_pandas()
    preds_df['pred'] = oof_preds

    gathered_preds_df = preds_df.pivot(index='text_id', columns='task', values='pred')
    gathered_preds_df.columns = [f"pred_{c}" for c in gathered_preds_df.columns]

    gathered_labels_df = preds_df.pivot(index='text_id', columns='task', values='score')

    return gathered_preds_df.merge(preds_df[['text_id', 'full_text']].drop_duplicates(), on='text_id').merge(gathered_labels_df, on='text_id')


@SolutionFactory.register('text2text')
class Text2Text(Solution):
    def do_train(self, train_data: pd.DataFrame, data_cfg: Mapping, train_cfg: Mapping, model_cfg: Mapping, env_cfg: Mapping):
        tokenizer = T5Tokenizer.from_pretrained(model_cfg.backbone)

        raw_train_df = self.competition_data_manager.load_train_data()

        target_cols = self.competition_data_manager.LABEL_COLUMNS
        train_ds = self.preprocess_train(train_cfg, raw_train_df, target_cols)

        tokenizer.save_pretrained(os.path.join(env_cfg.artifacts_path, 'tokenizer'))

        #possible_scores = [x / 2 for x in range(2, 11)]
        #score_token_lookup = [x.encode(x) for x in possible_scores]

        oof_df = pd.DataFrame()
        for fold in range(train_cfg.n_fold):
            if fold in train_cfg.trn_fold:
                model = T5ForConditionalGeneration.from_pretrained(model_cfg.backbone)
                torch.save(model.config, os.path.join(env_cfg.artifacts_path, 'config.pth'))

                _oof_df = train_loop(model, train_ds, fold, train_cfg, data_cfg, env_cfg.artifacts_path, env_cfg.device,
                                     tokenizer, target_cols)
                oof_df = pd.concat([oof_df, _oof_df])
                logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df, target_cols)
        oof_df = oof_df.reset_index(drop=True)
        logger.info(f"========== CV ==========")
        get_result(oof_df, target_cols)
        oof_df.to_pickle(os.path.join(env_cfg.artifacts_path, 'oof_df.pkl'))


    def preprocess_train(self, train_cfg, train_df, target_cols):
        Fold = MultilabelStratifiedKFold(n_splits=train_cfg.n_fold, shuffle=True, random_state=train_cfg.seed)
        for n, (train_index, val_index) in enumerate(Fold.split(train_df, train_df[target_cols])):
            train_df.loc[val_index, 'fold'] = int(n)
        train_df['fold'] = train_df['fold'].astype(int)
        single_score_df = pd.concat([pd.concat([train_df[['text_id', 'full_text', x, 'fold']].rename(columns={x: 'score'}),
                                                pd.Series(len(train_df) * [x], name='task')], axis=1)
                                     for x in self.competition_data_manager.LABEL_COLUMNS], axis=0)
        single_score_df = single_score_df.sort_values(['text_id', 'task'])
        return datasets.Dataset.from_pandas(single_score_df.reset_index(drop=True)).map(add_input_texts)

    def do_predict(self, input_data: pd.DataFrame, data_cfg: Mapping, inference_cfg: Mapping, model_cfg: Mapping,
                   env_cfg: Mapping):
        pass
