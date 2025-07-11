import sys
import os

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
import random
import datetime
import numpy as np

sys.path.append('.')
from core.utils.train_utils import Metric_Recoder
from model.cnn.textcnn import TextCNN
from utils import init_tensorboard, get_dataset_dataloader, save_model
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--tmp_dir',default="tmp_dir",type = str,help="the temp dir for holding data")
parser.add_argument('--tmp_model',default="tmp_model",type = str,help="the temp dir for holding model")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training")
parser.add_argument("--num_epochs", default=10, type=int, help="Number of epochs for training")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for the optimizer")
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--max_len", default=700, type=int, help="Maximum sequence length for padding. For character-level tokenization, 2100 is recommended for the PKDD dataset because it includes headers, while 700 is recommended for other datasets.")
parser.add_argument("--emb_dim", default=512, type=int, help="Dimensionality of the embeddings")
parser.add_argument("--dropout", default=None, type=float, help="Dropout rate for the model")
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--dataset", default="fpad", type=str, help="Dataset to use, one of [csic, pkdd, fpad, cve]")
parser.add_argument("--token", default="char", type=str, help="one of [char, word]")
parser.add_argument("--optimizer", default='adam', type=str, help="Optimizer to use, one of [adam, adamw]")
parser.add_argument("--decay_ratio", default=1.0, type=float, help="Decay ratio for the learning rate scheduler, 1.0 means no decay")
parser.add_argument("--decay_step", default=1, type=int, help="Step size for the learning rate scheduler")
parser.add_argument("--model", default='textcnn', type=str, help="Model to use, we use textcnn")
parser.add_argument("--use_tb", action="store_true", help="Flag to use tensorboard for logging") 
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--explain_flag", default=True, help="Flag to use explanation-based tokenization (alignment with HTTP structure)")
parser.add_argument("--poison_ratio", default=0.0, type=float, help="Proportion of the data to be label-poisoned")
parser.add_argument("--train_path", default="train.jsonl", type=str, help="Path to the training dataset")
parser.add_argument("--test_path", default="test.jsonl", type=str, help="Path to the testing dataset")

# Parse arguments
args = parser.parse_args()

if args.use_tb:
    writer, hyperpara_dict = init_tensorboard(args)
else:
    writer = None

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(args.seed)

tmp_dir = args.tmp_dir
# dataset_path = args.tmp_dir
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
print(device)

# %%
train_path = os.path.join(tmp_dir, args.train_path)
test_path = os.path.join(tmp_dir, args.test_path)

trainset, trainloader, testset, testloader, word2id, id2word = get_dataset_dataloader(train_path, test_path, args.token, args.dataset, args.explain_flag, args)
vocab_size = len(word2id)

print("vocab_size:", vocab_size)
NUM_CLS = trainset.get_number_of_classes()
print("NUM_CLS:", NUM_CLS)
print("Training set stats report:")
trainset.report_class_stats()
print("Testing set stats report:")
testset.report_class_stats()

_dropout = args.dropout if args.dropout is not None else 0.0
model_args = {
    "vocab_size": vocab_size,
    "emb_dim": args.emb_dim,
    "dropout": _dropout,
    "n_class": NUM_CLS
}
net = TextCNN(**model_args).to(device)

print(f"{args.model} args:")
for k, v in model_args.items():
    print(f"\t{k}: {v}")
criterion = nn.CrossEntropyLoss()
if args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.optimizer == 'adamw':
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_ratio)
print('num parameters:', sum(param.numel() for param in net.parameters()))
rec = Metric_Recoder(kpi='f1') #  macro_f1 in multi-class setting, positive_f1 in binary setting


train(net, args.num_epochs, trainloader, testloader, optimizer, criterion, scheduler, device, NUM_CLS, rec, writer=writer)
print('Finished Training')


print(f"{args.model} args:")
for k, v in model_args.items():
    print(f"\t{k}: {v}")
print("best result: ", rec.get_best())
if writer:
    writer.add_hparams(hyperpara_dict, rec.get_best())

model_save_name = f"textcnn-{args.max_len}-{args.dataset.upper()}-{args.emb_dim}-{args.dropout}-{args.seed}"

model_save_path = os.path.join(args.tmp_model, f"{model_save_name}.pth")

os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
save_model(rec.get_best_model_state_dict(), word2id, id2word, args, model_args, model_save_path, NUM_CLS)
print("Model saved to: ", model_save_path)