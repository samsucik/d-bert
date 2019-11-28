import os
import sys
import argparse
from types import SimpleNamespace

import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adadelta
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import read_args
from dbert.distill.data import find_dataset, set_seed, replace_embeds, list_field_mappings
import dbert.distill.model as mod

# pytorch-transformers imports
from examples.utils_glue import compute_metrics
from examples.distillation.distiller_from_finetuned import Distiller

def evaluate(params: SimpleNamespace):
    """
    params contains:
    - dataset
    - model
    - task_name
    """
    params.dataset.init_epoch()
    params.model.eval()
    preds, targets = None, None
    for batch in params.dataset:
        with torch.no_grad():
            logits = params.model(batch.sentence)
        labels_pred = logits.max(1)[1]
        if preds is None:
            preds = labels_pred.detach().cpu().numpy()
            targets = batch.label.detach().cpu().numpy()
        else:
            preds = np.append(preds, labels_pred.detach().cpu().numpy(), axis=0)
            targets = np.append(targets, batch.label.detach().cpu().numpy(), axis=0)
    result = compute_metrics(params.task_name, preds, targets)
    return result

def extend_args(args):
    pairing = {
        "output_dir": "workspace",
        "temperature": "distill_temperature",
        "alpha_ce": "ce_lambda",
        "alpha_mse": "distill_lambda",
        "n_epochs": "epochs",
        "learning_rate": "lr",
        "max_grad_norm": "clip_grad",
        "task_name": "dataset_name"
    }
    defaults = {
        "gradient_accumulation_steps": 1,
        "use_hard_labels": False,
        "max_steps": -1,
        "log_interval": 100,
        "evaluate_during_training": True,
        "checkpoint_interval": -1,
        "n_gpu": 0 if args.device == "cpu" else 1
    }
    for k, v in pairing.items():
        args.__dict__[k] = args.__dict__[v]
    for k, v in defaults.items():
        args.__dict__[k] = v
    return SimpleNamespace(**args.__dict__)

def main():
    args = read_args(default_config="confs/kim_cnn_sst2.json")
    set_seed(args.seed)
    args = extend_args(args)
    print(args)
    torch.cuda.deterministic = True
    dataset_cls = find_dataset(args.dataset_name)
    training_iter, dev_iter, test_iter = dataset_cls.iters(args.dataset_path, args.vectors_file, args.vectors_dir,
        batch_size=args.batch_size, device=args.device, train=args.train_file, dev=args.dev_file, test=args.test_file)

    args.dataset = training_iter.dataset
    model = mod.BiRNNModel(args).to(args.device)
    args.dataset = None
    distiller = Distiller(params=args,
                          dataset_train=training_iter,
                          dataset_eval=dev_iter,
                          student=model,
                          evaluate_fn=evaluate,
                          student_type="LSTM")
    distiller.train()

if __name__ == "__main__":
    main()
