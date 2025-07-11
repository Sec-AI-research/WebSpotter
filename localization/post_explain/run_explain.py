import sys
import os
import time
import argparse
from tqdm import tqdm
import torch
import random
import subprocess
import numpy as np
import warnings
warnings.filterwarnings("ignore")
sys.path.append('.')
from explain import load_datasets, initialize_explanation_method, initialize_tokenizer, perform_explanation, tokens_to_padded_indices,write_data,analyze_attacks_accuracy,identify_attack_payload,evaluate_dataset_performance, get_location_ground_truth
from captum.attr import remove_interpretable_embedding_layer
from classification.utils import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',type=str, required=True)
parser.add_argument('--dataset', type=str, required=True, choices=['csic','pkdd','fpad','cve'])
parser.add_argument('--test_path', type=str, required=True)
parser.add_argument('--outputdir', type=str, required=True, help="Directory to store output files")
parser.add_argument("--token", default="char", type=str, help="one of [char, word]")
parser.add_argument('--explain_method', type=str, default='ig', choices=['feature','lime','lemna','ig', 'kernelshap', 'ng'])
parser.add_argument('--mean_weight', type=float, default=1.0)
parser.add_argument('--stv_weight', type=float, default=1.0)
parser.add_argument('--token_aggretion_method', type=str, default='abs_dot', choices=['abs_dot','sum','abs_sum'])
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--seed", default=0, type=int)

args = parser.parse_args()
dataset_type = args.dataset
test_path = args.test_path
outputdir = args.outputdir
explain_method = args.explain_method
mean_weight = args.mean_weight
stv_weight = args.stv_weight
token_aggretion_method = args.token_aggretion_method
token = args.token

device = torch.device("cuda:{}".format(args.gpu))

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(args.seed)

import json
def main():
    net, word2id, _id2word, _args, _n_class = load_model(args.model_path)
    max_len = _args['max_len']
    net.to(device)
    
    # Initialize counters
    total_count = 0
    precision_not_one_count = 0
    recall_not_one_count = 0
    misclassified_count = 0
    # Initialize lists to store evaluation scores for each sample
    all_precision = []
    all_recall = []
    all_f1_scores = []
    all_jaccard_index = []
    all_accurary = []

    # Ensure the output directory exists
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # Create output filenames
    targetfile = os.path.join(args.outputdir, "results.txt")  # Assuming 'results.txt' as a default output filename

    # Call the function to load datasets
    test_dataset, test_data_json, test_labels, test_Y = load_datasets(test_path, device)
    # Initialize explanation method
    lig, interpretable_emb = initialize_explanation_method(net, explain_method)
    # Initialize tokenizer
    http_tokenizer_with_httpinfo = initialize_tokenizer(dataset_type, token)
    total_time_start = time.time()

    with open(targetfile, "w") as file:

        for i in tqdm(range(len(test_dataset))):
            # Evaluate malicious samples
            label = test_labels[i].to(device)
            if label.item() != 0:
                # Load and process data
                token_list, alignment = http_tokenizer_with_httpinfo(test_dataset[i])
                token_lengths = [len(chars) for _, chars in alignment]  # Length of each token

                tokens_original = [orig for orig, _ in alignment]  # Original text of each token
                token_ids = tokens_to_padded_indices(token_list, word2id, max_len)  # Convert tokens to indices
                input_data = torch.tensor(token_ids, device=device).unsqueeze(0)
                
                # Perform explanation
                final_importance_scores, pred_label, pred_prob = perform_explanation(
                    net, input_data, explain_method, lig, interpretable_emb, device, 
                    max_len, word2id, token_lengths, alignment, tokens_original, token_aggretion_method
                )

                # Identify potential attacks
                suspected_attacks, mean_score, std_dev, threshold = identify_attack_payload(final_importance_scores, mean_weight, stv_weight)
                
                # Compute evaluation metrics
                precision, recall, f1_score, accurary, jaccard_index, ulocation, attacks = analyze_attacks_accuracy(
                            dataset_type, test_data_json[i], suspected_attacks, final_importance_scores
                        )

                # Store final_importance_scores
                test_dataset[i].importance_scores = final_importance_scores
                test_dataset[i].precision = precision
                test_dataset[i].recall = recall
                test_dataset[i].f1_score = f1_score
                
                # Save the precision, recall, and F1 score of the current sample
                all_precision.append(precision)
                all_recall.append(recall)
                all_f1_scores.append(f1_score)
                all_accurary.append(accurary)
                all_jaccard_index.append(jaccard_index)

                request = test_dataset[i]
                original_text = f"Method:{request.method} URL:{request.url} Body:{request.body}".strip()
                write_data(file, original_text, pred_label, pred_prob, test_Y[i], final_importance_scores, suspected_attacks, mean_score, std_dev, threshold, attacks, ulocation, precision, recall, f1_score, accurary, jaccard_index)
                
                # Update counters and handle special cases for output
                total_count += 1

        
    # Record total execution time and statistics
    total_time_end = time.time()
    print("Method:", explain_method)
    print("Dataset:", dataset_type)
    print("Total Execution Time:", total_time_end - total_time_start)
    print("Total Count:", total_count)
    
    # Evaluate performance metrics for the entire dataset
    avg_precision, avg_recall, avg_f1_score, avg_accurary, avg_jaccard_index = evaluate_dataset_performance(
        all_precision, all_recall, all_f1_scores, all_accurary, all_jaccard_index
    )
    
    # # Print evaluation results for the entire dataset
    # print(f"Average Precision: {avg_precision:.4f}")
    # print(f"Average Recall: {avg_recall:.4f}")
    # print(f"Average F1 Score: {avg_f1_score:.4f}")
    # print(f"Average Accuracy: {avg_accurary:.4f}")
    # print(f"Average Jaccard Index: {avg_jaccard_index:.4f}")

    # Export the test dataset with scores included
    test_dataset_withscore_path = os.path.join(outputdir, f"{test_path.split('/')[-1]}_withscore")
    test_dataset.dump_datset(test_dataset_withscore_path)


if __name__ == "__main__":
    main()
