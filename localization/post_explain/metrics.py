import re
from urllib.parse import unquote_plus, unquote

def identify_attack_payload(final_importance_scores, mean_weight=1, stv_weight=1):
    # Calculate mean and standard deviation of scores
    scores = [score for _, score in final_importance_scores]
    mean_score = sum(scores) / len(scores)
    std_dev = (sum([(x - mean_score) ** 2 for x in scores]) / len(scores)) ** 0.5
    threshold = mean_score*mean_weight + std_dev*stv_weight  # Setting the threshold as mean + standard deviation

    attack_payloads = [(part, score) for part, score in final_importance_scores if score >= threshold]

    return attack_payloads, mean_score, std_dev, threshold

def extract_params(part):
    if ": " in part:
        tokens = part.split(': ', 1) 
    else:
        tokens = part.split('=', 1) 
    
    return set(token.rstrip('?').rstrip().rstrip('/') for token in tokens)


from urllib.parse import unquote_plus

def check_payload_accuracy_fpad(location_tags, suspected_attacks, total_num):
    """
    Evaluate the accuracy of model-identified attack payloads compared to ground-truth attack locations.
    Designed specifically for the FPAD dataset.
    """
    relevant_tags = ['<AddedURLkey>', '<ModifiedURLkey>', '<AddedBodykey>', '<ModifiedBodykey>', '<Path>']
    location_params = []
    for tag in location_tags:
        if any(tag.startswith(prefix) for prefix in relevant_tags):
            if tag.startswith('<Path>'):
                path_content = tag[len('<Path>'):]
                if path_content == '/':
                    location_params.append('/')
                else:
                    path_parts = tag[len('<Path>'):].split('/')
                    for part in path_parts:
                        if part:
                            decoded_part = unquote_plus(part).rstrip()
                            location_params.append('/' + decoded_part)
            else:
                prefix = next(prefix for prefix in relevant_tags if tag.startswith(prefix))
                param_name = tag[len(prefix):]
                location_params.append(param_name)

    identified_params_list = []
    for part, _ in suspected_attacks:
        part_identified_set = set()
        part_identified_set.add(part.rstrip('?').rstrip().rstrip('/'))
        tokens = extract_params(part)
        part_identified_set.update(tokens)
        identified_params_list.append(part_identified_set)

    return calculate_metrics(location_params, identified_params_list, total_num), location_params

def check_payload_accuracy_pkdd(attacks, suspected_attacks, total_num):
    """
    Evaluate the accuracy of model-identified attack payloads compared to ground-truth attack locations.
    Designed specifically for the PKDD dataset.
    """
    location_params = []
    for param in attacks:
        location_params.append(unquote_plus(param).rstrip('?').rstrip())

    identified_params_list = []
    for part, _ in suspected_attacks:
        part_identified_set = set()
        part_identified_set.add(part.rstrip('?').rstrip())
        tokens = extract_params(part)
        part_identified_set.update(tokens)
        identified_params_list.append(part_identified_set)

    return calculate_metrics(location_params, identified_params_list, total_num),location_params

def check_payload_accuracy_poc(attacks, suspected_attacks, total_num):
    """
    Evaluate the accuracy of model-identified attack payloads compared to ground-truth attack locations.
    Designed specifically for the CVE dataset.
    """
    location_params = []
    for param in attacks:
        if param.startswith('<Path>'):
            path_content = param[len('<Path>'):]
            if path_content == '/':  
                location_params.append('/')
            else:
                path_parts = path_content.split('/')
                for part in path_parts:
                    if part:  
                        decoded_part = unquote_plus(part).rstrip()
                        location_params.append('/' + decoded_part)
        else:
            location_params.append(param.rstrip('?').rstrip().rstrip('/'))

    identified_params_list = []
    for part, _ in suspected_attacks:
        part_identified_set = set()
        part_identified_set.add(part.rstrip('?').rstrip().rstrip('/'))
        tokens = extract_params(part)
        part_identified_set.update(tokens)
        identified_params_list.append(part_identified_set)

    return calculate_metrics(location_params, identified_params_list, total_num),location_params

def check_payload_accuracy_csic(attacks, suspected_attacks, total_num):
    """
    Evaluate the accuracy of model-identified attack payloads compared to ground-truth attack locations.
    Designed specifically for the CSIC dataset.
    """
    location_params = []
    for param in attacks:
        location_params.append(param.rstrip('?').rstrip().rstrip('/'))

    identified_params_list = []
    for part, _ in suspected_attacks:
        part_identified_set = set()
        part_identified_set.add(part.rstrip('?').rstrip().rstrip('/'))
        tokens = extract_params(part)
        part_identified_set.update(tokens)
        identified_params_list.append(part_identified_set)

    return calculate_metrics(location_params, identified_params_list, total_num),location_params


def calculate_metrics(location_params, identified_params_list, total_num):
    """
    Calculate precision, recall, F1 score, accuracy, and Jaccard index 
    based on ground-truth attack locations and identified attack payloads.
    """
    if not location_params:
        precision = 0
        recall = 0
        f1_score = 0
        jaccard_index = 0
        accuracy = 0
        return precision, recall, f1_score, accuracy, jaccard_index

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_true_negatives = 0

    for identified_params in identified_params_list:
        match_found = False
        for param in location_params:
            if any(param in sub_param for sub_param in identified_params):
                match_found = True
                break
        if match_found:
            total_true_positives += 1
        else:
            total_false_positives += 1

    total_false_negatives = len(location_params) - total_true_positives
    total_true_negatives = total_num - (total_true_positives + total_false_positives + total_false_negatives)

    precision = 0
    recall = 0
    f1_score = 0
    jaccard_index = 0

    # Calculate precision and recall
    if total_true_positives + total_false_positives > 0:
        precision = total_true_positives / (total_true_positives + total_false_positives)
    if total_true_positives + total_false_negatives > 0:
        recall = total_true_positives / (total_true_positives + total_false_negatives)

    # Calculate F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    # Calculate Jaccard index
    union_size = total_true_positives + total_false_negatives + total_false_positives
    jaccard_index = total_true_positives / union_size

    # Calculate accuracy
    if total_num > 0:
        accuracy = (total_true_positives + total_true_negatives) / total_num

    return precision, recall, f1_score, accuracy, jaccard_index



def evaluate_dataset_performance(all_precision, all_recall, all_f1_scores, all_accuracy, all_jaccard_index):
    num = len(all_precision)
    avg_precision = sum(all_precision) / num
    avg_recall = sum(all_recall) / num
    avg_f1_score = sum(all_f1_scores) / num
    avg_accurary = sum(all_accuracy) / num
    avg_jaccard_index = sum(all_jaccard_index) / num
    
    return avg_precision, avg_recall, avg_f1_score, avg_accurary, avg_jaccard_index

def get_ground_truth_fpad(location_tags, http_msu):
    """
    Extract ground-truth attack labels for the FPAD dataset by identifying whether each token matches the 
    known attack locations (location_tags).
    """
    ground_truth = []
    relevant_tags = ['<AddedURLkey>', '<ModifiedURLkey>', '<AddedBodykey>', '<ModifiedBodykey>', '<Path>']
    location_params = []

    # Parse location tags to extract relevant parameters
    for tag in location_tags:
        if any(tag.startswith(prefix) for prefix in relevant_tags):
            if tag.startswith('<Path>'):
                path_content = tag[len('<Path>'):]
                if path_content == '/':  # Handle special case for root path
                    location_params.append('/')
                else:
                    path_parts = tag[len('<Path>'):].split('/')
                    for part in path_parts:
                        if part:
                            decoded_part = unquote_plus(part).rstrip()  # Decode and clean up
                            location_params.append('/' + decoded_part)
            else:
                prefix = next(prefix for prefix in relevant_tags if tag.startswith(prefix))
                param_name = tag[len(prefix):]
                location_params.append(param_name)

    params_list = []
    for part in http_msu:
        part_identified_set = set()
        # Add the raw, unprocessed part (trim trailing spaces, '?' and '/')
        part_identified_set.add(part.rstrip('?').rstrip().rstrip('/'))
        # Extract and add segmented tokens from the part
        tokens = extract_params(part)
        part_identified_set.update(tokens)
        params_list.append(part_identified_set)

    # Match tokens with ground-truth locations to generate labels
    for identified_params in params_list:
        if any(param in identified_params for param in location_params):
            ground_truth.append(1)  # Match found: mark as attack
        else:
            ground_truth.append(0)  # No match: mark as non-attack

    return ground_truth

def get_ground_truth_pkdd(attacks, http_msu):
    """
    Extract ground-truth attack labels for the PKDD dataset by matching identified tokens against known attacks.
    """
    ground_truth = []
    location_params = []

    # Process attack parameters to remove special characters and extra spaces
    for param in attacks:
        location_params.append(unquote_plus(param).rstrip('?').rstrip())  # Clean and decode attack parameters

    params_list = []
    for part in http_msu:
        part_identified_set = set()
        # Add the raw token after removing trailing spaces, '?' and '/'
        part_identified_set.add(part.rstrip('?').rstrip())
        # Extract and add segmented tokens from the part
        tokens = extract_params(part)
        part_identified_set.update(tokens)
        params_list.append(part_identified_set)

    # Match identified tokens with ground-truth attack locations
    for identified_params in params_list:
        match_found = False
        for param in location_params:
            if any(param in sub_param for sub_param in identified_params):
                match_found = True
                break
        if match_found:
            ground_truth.append(1)  # Match found: mark as attack
        else:
            ground_truth.append(0)  # No match: mark as non-attack

    return ground_truth

def get_ground_truth_csic(attacks, http_token):
    """
    Extract ground-truth attack labels for the CSIC dataset by matching identified tokens against known attacks.
    """
    ground_truth = []
    location_params = []

    # Process attack parameters by cleaning trailing characters for consistency
    for param in attacks:
        location_params.append(param.rstrip('?').rstrip().rstrip('/'))

    params_list = []
    for part in http_token:
        part_identified_set = set()
        # Add the raw token after removing trailing characters
        part_identified_set.add(part.rstrip('?').rstrip().rstrip('/'))
        # Extract and add segmented tokens
        tokens = extract_params(part)
        part_identified_set.update(tokens)
        params_list.append(part_identified_set)

    # Compare identified tokens with attack locations
    for identified_params in params_list:
        if any(param in identified_params for param in location_params):
            ground_truth.append(1)  # Match found: mark as attack
        else:
            ground_truth.append(0)  # No match: mark as non-attack

    return ground_truth


def get_ground_truth_poc(attacks, http_token):
    """
    Extract ground-truth attack labels for the PoC dataset by comparing identified tokens with attack locations, 
    including processing for path-based attacks.
    """
    ground_truth = []
    location_params = []

    # Process attacks to extract parameters or paths
    for param in attacks:
        if param.startswith('<Path>'):
            # Handle <Path> tags by removing the tag and splitting the path
            path_content = param[len('<Path>'):]
            if path_content == '/':  # Handle the case where the path is just a single slash
                location_params.append('/')
            else:
                path_parts = path_content.split('/')
                for part in path_parts:
                    if part:  # Add only non-empty parts
                        decoded_part = unquote_plus(part).rstrip()  # Decode and clean each part
                        location_params.append('/' + decoded_part)
        else:
            # Handle non-path parameters by cleaning trailing characters
            location_params.append(param.rstrip('?').rstrip().rstrip('/'))

    params_list = []
    for part in http_token:
        part_identified_set = set()
        # Add the raw token after cleaning trailing characters
        part_identified_set.add(part.rstrip('?').rstrip().rstrip('/'))
        # Extract and add segmented tokens
        tokens = extract_params(part)
        part_identified_set.update(tokens)
        params_list.append(part_identified_set)

    # Compare identified tokens with attack locations
    for identified_params in params_list:
        if any(param in identified_params for param in location_params):
            ground_truth.append(1)  # Match found: mark as attack
        else:
            ground_truth.append(0)  # No match: mark as non-attack

    return ground_truth


