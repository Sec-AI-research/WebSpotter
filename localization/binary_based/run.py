# %%
import torch
import os
import sys
import random
import numpy as np
sys.path.append('.')
sys.path.append('./localization/post_explain')
from localization.post_explain.explain import load_datasets, initialize_tokenizer
from binary_feature import get_binary_dataset,get_binary_dataset_plus
from binary_model import B_Model, evaluate_model_withlog_waf, evaluate_model_withlog
from local_embedding import Sentence_embedding
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--feature_method',type=str, required=True, help="one of ['text', 'textemb', 'score', 'score_sort', 'score_sort_with_textemb']")
parser.add_argument('--dataset', type=str, required=True, help="one of ['fpad', 'csic', 'pkdd', 'cve']")
parser.add_argument('--train_path',type=str, required=True)
parser.add_argument('--test_path', type=str, required=True)
parser.add_argument('--k', type=int, default=10, help="number of importance scores to use")
parser.add_argument('--sample_rate', type=float, default=0.01, help="rate of location-labeled training data to use")
parser.add_argument('--emb_model', type=str, default='nomic-v1.5')
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--emb_prefix', type=str, default='classification: ')
parser.add_argument('--cls_model', type=str, default='rf')
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument('--output_path', type=str, default=None, help="Optional path to save special case study output files")
parser.add_argument('--poison_ratio', type=float, default=0.0, help="Poison ratio to be applied to the dataset")
args = parser.parse_args()
print(args)
dataset_name = args.dataset
train_data_path = args.train_path
test_data_path = args.test_path
feature_method = args.feature_method
K = args.k
sample_rate = args.sample_rate
cls_model = args.cls_model
device = f'cuda:{args.gpu}'
poison_ratio = args.poison_ratio  # Capture poison ratio from command line

if feature_method in ['textemb', 'score_sort_with_textemb']:
    emb_model = Sentence_embedding(model_name=args.emb_model, dim=args.emb_dim, device=device, prefix=args.emb_prefix)
else:
    emb_model = None
if feature_method == 'text':
    assert cls_model == 'textcnn'

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(args.seed)

http_tokenizer_with_httpinfo = initialize_tokenizer(dataset_name, "char") 
train_dataset, train_data_json, train_labels, train_Y = load_datasets(train_data_path, "cpu")
test_dataset, test_data_json, test_labels, test_Y = load_datasets(test_data_path, "cpu")

for req in train_dataset:
    if hasattr(req, 'importance_scores'):
        req.importance_scores = [score for _, score in req.importance_scores]
for req in test_dataset:
    if hasattr(req, 'importance_scores'):
        req.importance_scores = [score for _, score in req.importance_scores]

if sample_rate == 1.0:
    train_binary_dataset = get_binary_dataset(train_dataset, train_data_json, dataset_name, http_tokenizer_with_httpinfo, feature_method=feature_method, k=K, balance=1.0, emb_model=emb_model, poison_ratio=poison_ratio)
else:
    train_binary_dataset = get_binary_dataset_plus(train_dataset, train_data_json, dataset_name, http_tokenizer_with_httpinfo, feature_method=feature_method, k=K, balance=1.0, sample_rate=sample_rate, emb_model=emb_model, poison_ratio=poison_ratio)

model = B_Model(train_binary_dataset, model_name=cls_model, device=device)
train_start_time = time.time()
model.train()

train_end_time = time.time()
print(f"Training time: {train_end_time - train_start_time} seconds")

if feature_method == 'text':
    test_binary_dataset = get_binary_dataset(test_dataset, test_data_json, dataset_name, http_tokenizer_with_httpinfo, feature_method=feature_method)
    test_string_list = [data[0] for data in test_binary_dataset]
    model.predict_batch(test_string_list)  # for cache


start_time = time.time()

evaluate_model_withlog(model, dataset_name, test_dataset, test_data_json, http_tokenizer_with_httpinfo, args.output_path,feature_method=feature_method, k=K, emb_model=emb_model)

end_time = time.time()
print(f"Testing time: {end_time - start_time} seconds")
