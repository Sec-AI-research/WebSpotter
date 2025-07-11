import json
from core.inputter import HTTPDataset
import torch
import torch.nn.functional as F
from metrics import get_ground_truth_poc,check_payload_accuracy_poc, check_payload_accuracy_fpad, check_payload_accuracy_pkdd ,check_payload_accuracy_csic , identify_attack_payload, evaluate_dataset_performance, get_ground_truth_fpad,get_ground_truth_pkdd,get_ground_truth_csic
from lemna import LemnaModel
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr import (
    LayerIntegratedGradients, LayerDeepLift, GradientShap, KernelShap, Lime, Saliency, 
    ShapleyValueSampling, LRP, configure_interpretable_embedding_layer, LimeBase ,FeatureAblation
)
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule, IdentityRule

def initialize_explanation_method(net, explain_method):
    """
    Initializes the explanation method for a neural network model.

    Args:
        net: The neural network model to be explained.
        explain_method (str): The explanation method to use. Supported methods include:
            - "integratedgradients(ig)": Uses integrated gradients to compute feature importance by accumulating gradients along a path.
            - "kernelshap": Uses a kernel-based approximation of Shapley values to compute feature importance.
            - "lime": Fits a local interpretable model (e.g., linear) around the prediction to approximate feature importance.
            - "naivegradients(ng)": Computes feature importance directly using input gradients.
            - "lemna": Builds a local linear surrogate model tailored for security-related datasets.
            - "feature": Uses feature ablation by removing individual features and observing the impact on predictions.

    Returns:
        - lig: The initialized explanation method object.
        - interpretable_emb: The interpretable embedding layer (if required by the method), 
            or None if not needed.
    """
    interpretable_emb = None
    lig = None
    if explain_method == "ig":
        # Setting multiply_by_inputs=True returns the product of the embedding features and their attribution scores
        lig = LayerIntegratedGradients(net, net.embedding,multiply_by_inputs=True)
    elif explain_method == "kernelshap":
        interpretable_emb = configure_interpretable_embedding_layer(net, 'embedding')
        lig = KernelShap(net)
    elif explain_method == "lime":
        interpretable_emb = configure_interpretable_embedding_layer(net, 'embedding')
        lig = Lime(net)
    elif explain_method == "ng":
        interpretable_emb = configure_interpretable_embedding_layer(net, 'embedding')
        lig = Saliency(net)
    elif explain_method == "lemna":
        interpretable_emb = configure_interpretable_embedding_layer(net, 'embedding')
        lig = Lime(net, interpretable_model = LemnaModel())
    elif explain_method == "feature": 
        interpretable_emb = configure_interpretable_embedding_layer(net, 'embedding')
        lig = FeatureAblation(net)

    return lig, interpretable_emb


def initialize_tokenizer(dataset, token_type,deep_char=False):
    if token_type == "word":
        if dataset == "pkdd":
            from core.preprocess import word_tokenizer_with_http_level_alignment_furl_header
            return word_tokenizer_with_http_level_alignment_furl_header
        else:
            from core.preprocess import word_tokenizer_with_http_level_alignment
            return word_tokenizer_with_http_level_alignment
    elif token_type == "char":
        if dataset == "pkdd":
            from core.preprocess import char_tokenizer_with_http_level_alignment_furl_header
            return char_tokenizer_with_http_level_alignment_furl_header
        else:
            from core.preprocess import char_tokenizer_with_http_level_alignment
            return char_tokenizer_with_http_level_alignment
    else:
        raise ValueError("Invalid token_type")
    

def load_datasets(test_path, device):
    with open(test_path, 'r') as file:
        test_data_json = [json.loads(line) for line in file]

    # Load HTTP dataset
    test_dataset = HTTPDataset.load_from(test_path)
    
    # Extract labels and convert to tensor
    test_Y = [req.label for req in test_dataset]
    test_labels = torch.tensor(test_Y, device=device)

    return test_dataset, test_data_json, test_labels, test_Y

def write_data(file, original_text, pred_label, pred_prob, true_label, final_importance_scores, suspected_attacks, mean_score, std_dev, threshold, location, ulocation, precision, recall, f1_score, accurary,jaccard_index):
    file.write("Original text: {}\n".format(original_text))
    file.write("Predicted label: {}, Prob: {:.4f}\n".format(pred_label, pred_prob))
    file.write("True label: {}\n".format(true_label))
    file.write("http Contributions:\n")
    for word, importance in final_importance_scores:
        file.write("part: {}, score : {:.4f}\n".format(word, importance))
    file.write("Identified Attack Payloads: (mean_score:{:.4f},std_dev:{:.4f},threshold:{:.4f})\n".format(mean_score, std_dev, threshold))
    for part, score in suspected_attacks:
        file.write("Part: {}, Score: {:.4f}\n".format(part, score))
    file.write("Locations tag: {}\n".format(location))
    file.write("used Locations: {}\n".format(ulocation))
    file.write("Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}, Accurary: {:.4f}, Jaccard Index: {:.4f}\n".format(precision, recall, f1_score, accurary,jaccard_index))
    file.write("\n")
    
def tokens_to_padded_indices(chars, word2id, max_len, pad_token="<PAD>", unk_token="<UNK>"):
    indices = [word2id.get(char, word2id.get(unk_token, 0)) for char in chars]
    pad_length = max_len - len(indices)
    if pad_length > 0:
        indices += [word2id.get(pad_token, 0)] * pad_length
    else:
        indices = indices[:max_len]
    
    return indices

def create_feature_mask_token(input_emb, token_lengths, device):
    batch_size, max_length, embedding_size = input_emb.shape

    feature_mask = torch.arange(max_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    total_token_length = sum(token_lengths)

    if total_token_length > max_length:
        total_token_length = max_length

    feature_mask[:, total_token_length:] = total_token_length

    feature_mask = feature_mask.unsqueeze(2).repeat(1, 1, embedding_size)

    return feature_mask

def aggregate_token_importance(scores_list, tokens_list, token_lengths_list):

    aggregated_scores = []
    score_index = 0  

    for token, length in zip(tokens_list, token_lengths_list):
        token_score = sum(scores_list[score_index:score_index+length])
        aggregated_scores.append((token, token_score))
        score_index += length 

    return aggregated_scores

def aggregate_token_importance_waf(scores_list, tokens_list, token_lengths_list):
    aggregated_scores = []
    score_index = 0  

    for token, length in zip(tokens_list, token_lengths_list):
        character_scores = scores_list[score_index:score_index+length]
        token_score = sum(character_scores)
        aggregated_scores.append((token, token_score, character_scores))
        score_index += length 

    return aggregated_scores


def calculate_token_importances(embedded_input, attributions, max_len, token_aggretion_method):
    tokens_importances = []
    sequence_length = embedded_input.size(1) 

    for idx in range(sequence_length):
        if idx < max_len:
            char_embedding = embedded_input[0, idx]
            char_attributions = attributions[idx]
            if token_aggretion_method == "abs_dot":
                token_importance = char_attributions.sum().item()
            elif token_aggretion_method == "sum":
                safe_embedding = char_embedding.clone()
                safe_embedding[safe_embedding == 0] = 1  
                token_importance = (char_attributions / safe_embedding).sum().item()
            elif token_aggretion_method == "abs_sum":
                safe_embedding = char_embedding.clone()
                safe_embedding[safe_embedding == 0] = 1 
                token_importance = (abs(char_attributions / safe_embedding)).sum().item()
        else:
            token_importance = 0
            print("Embedding length short for index:", idx)

        tokens_importances.append(token_importance)

    return tokens_importances

def perform_explanation(net, input_data, explain_method, lig, interpretable_emb, device, max_len, word2id, token_lengths, alignment, tokens_original, token_aggretion_method):
    final_importance_scores = []
    if explain_method == "ig":
        outputs = net(input_data)
        pred_label = outputs.argmax(dim=1).item()
        pred_prob = F.softmax(outputs, dim=1)[0, pred_label].item()
        attributions = lig.attribute(input_data, target=pred_label,n_steps=50)
        attributions = attributions.squeeze(0)
        embedded_input = net.embedding(input_data)
    elif explain_method == "kernelshap" or explain_method == "lime" or explain_method == "lemna":
        embedded_input = interpretable_emb.indices_to_embeddings(input_data)
        reference_emb = torch.zeros_like(embedded_input)
        outputs = net(embedded_input)
        pred_label = outputs.argmax(dim=1).item()
        pred_prob = F.softmax(outputs, dim=1)[0, pred_label].item()
        feature_mask = create_feature_mask_token(embedded_input,token_lengths,device)
        attributions = lig.attribute(embedded_input, baselines=reference_emb, n_samples=50,target=pred_label,feature_mask=feature_mask,return_input_shape=False)
        attributions = attributions.squeeze(0)
    elif explain_method == "ng":
        embedded_input = interpretable_emb.indices_to_embeddings(input_data)
        outputs = net(embedded_input)
        pred_label = outputs.argmax(dim=1).item()
        pred_prob = F.softmax(outputs, dim=1)[0, pred_label].item()
        attributions = lig.attribute(embedded_input, target=pred_label)
        attributions = attributions.squeeze(0)
    elif explain_method == "feature":
        embedded_input = interpretable_emb.indices_to_embeddings(input_data)
        reference_emb = torch.zeros_like(embedded_input)
        outputs = net(embedded_input)
        pred_label = outputs.argmax(dim=1).item()
        pred_prob = F.softmax(outputs, dim=1)[0, pred_label].item()
        feature_mask = create_feature_mask_token(embedded_input,token_lengths,device)
        attributions = lig.attribute(embedded_input, baselines=reference_emb, target=pred_label,feature_mask=feature_mask)
        attributions = attributions.squeeze(0)
        attributions = attributions[:, 0]


    if explain_method == "kernelshap" or explain_method == "lime" or explain_method == "lemna" or explain_method == "feature":
        attributions_list = attributions.tolist()
        final_importance_scores = aggregate_token_importance(attributions_list, tokens_original, token_lengths)  
    else:
        tokens_importances = calculate_token_importances(embedded_input, attributions, max_len, token_aggretion_method)
        final_importance_scores = aggregate_token_importance(tokens_importances, tokens_original, token_lengths)       

    return final_importance_scores, pred_label, pred_prob


def analyze_attacks_accuracy(dataset_type, test_data_json_i, suspected_attacks, final_importance_scores):
    if dataset_type == "fpad" or dataset_type == "cve":
        if 'attacks' in test_data_json_i:
            attacks = test_data_json_i['attacks']
            metrics, ulocation = check_payload_accuracy_poc(attacks, suspected_attacks, len(final_importance_scores))
        else:
            attacks = test_data_json_i.get('location', [])
            metrics, ulocation = check_payload_accuracy_fpad(attacks, suspected_attacks, len(final_importance_scores))
    elif dataset_type == "pkdd":
        attacks = test_data_json_i.get('attacks', [])
        metrics, ulocation = check_payload_accuracy_pkdd(attacks, suspected_attacks, len(final_importance_scores))
    elif dataset_type == "csic":
        attacks = test_data_json_i.get('attacks', [])
        metrics, ulocation = check_payload_accuracy_csic(attacks, suspected_attacks, len(final_importance_scores))
    else:
        raise ValueError("Unsupported dataset type")

    precision, recall, f1_score, accurary, jaccard_index = metrics
    
    return precision, recall, f1_score, accurary, jaccard_index, ulocation, attacks

def get_location_ground_truth(dataset_type, test_data_json_i, http_split_part):
    if dataset_type == "fpad" or dataset_type == "cve":
        if 'attacks' in test_data_json_i :
            attacks = test_data_json_i.get('attacks', [])
            ground_truth = get_ground_truth_poc(attacks, http_split_part)
        else:
            attacks = test_data_json_i.get('location', [])
            ground_truth = get_ground_truth_fpad(attacks, http_split_part)
    elif dataset_type == "pkdd" :
        attacks = test_data_json_i.get('attacks', [])
        ground_truth = get_ground_truth_pkdd(attacks, http_split_part)
    elif dataset_type == "csic":
        attacks = test_data_json_i.get('attacks', [])
        ground_truth = get_ground_truth_csic(attacks, http_split_part)
    else:
        raise ValueError("Unsupported dataset type")
    
    return ground_truth
