import copy
import random
from tqdm import tqdm
from localization.post_explain.explain import get_location_ground_truth, load_datasets, initialize_tokenizer

def get_binary_sample(req, location_ground_truth, index, feature_method='text', k=10, emb_model=None):
    http_token = [part for part, _ in req.alignment]
    assert len(http_token) == len(location_ground_truth)
    label = location_ground_truth[index]
    if feature_method == 'text':
        target_text = http_token[index]
        return target_text, label  # target_text: str, label: int
    elif feature_method == 'textemb':
        assert emb_model is not None
        target_text = http_token[index]
        emb = emb_model(target_text)
        return emb, label  # emb: list of float, label: int
    elif feature_method == 'score':
        assert len(http_token) == len(req.importance_scores)
        scores = copy.deepcopy(req.importance_scores)
        feature = [0 for _ in range(k)]
        # move importance_scores[index] to the first, the others not change
        feature[0] = scores[index]
        scores.pop(index)
        for i in range(1, k):
            if i-1 >= len(scores):
                break
            feature[i] = scores[i-1]
        return feature, label  # feature: list of float, label: int
    elif feature_method == 'score_sort':
        assert len(http_token) == len(req.importance_scores)
        scores = copy.deepcopy(req.importance_scores)
        feature = [0 for _ in range(k)]
        feature[0] = scores[index]
        scores.pop(index)
        scores.sort(reverse=True)
        for i in range(1, k):
            if i-1 >= len(scores):
                break
            feature[i] = scores[i-1]
        return feature, label  # feature: list of float, label: int
    elif feature_method == 'score_sort_with_textemb':
        feature_score, _ = get_binary_sample(req, location_ground_truth, index, feature_method='score_sort', k=k)
        feature_textemb, _ = get_binary_sample(req, location_ground_truth, index, feature_method='textemb', emb_model=emb_model)
        feature = feature_score + feature_textemb
        return feature, label  # feature: list of float (dim: k + emb_dim), label: int
    else:
        raise ValueError(f"feature_method: {feature_method} is not supported")

def get_binary_dataset(dataset, data_json, dataset_name, tokenizer, feature_method='text', k=10, balance=None, sample_rate=1.0, emb_model=None, poison_ratio=0):
    """
    Generates a binary dataset for model training or evaluation.

    Returns:
        list: A list of tuples, where each tuple contains a feature and a label.
    """
    binary_dataset = []

    for index in tqdm(range(len(dataset))):
        # Skip samples without importance scores
        if not hasattr(dataset[index], 'importance_scores'):
            continue
        
        # Apply sampling logic based on sample_rate
        if random.random() > sample_rate:
            continue
        
        _, alignment = tokenizer(dataset[index])
        http_token = [part for part, _ in alignment]

        # Prepare embeddings if an embedding model is provided
        if emb_model is not None:
            prepare_embedding(http_token, emb_model)

        # Get ground truth for the token locations
        location_ground_truth = get_location_ground_truth(dataset_name, data_json[index], http_token)

        # Apply poisoning logic to the ground truth
        location_ground_truth = poison_ground_truth(location_ground_truth, poison_ratio)

        # Add alignment information to the current request
        req = dataset[index]
        req.alignment = alignment

        # Generate binary samples from tokens
        for i in range(1, len(http_token)):  # Start from 1 to ignore method part
            try:
                feature, label = get_binary_sample(req, location_ground_truth, i, feature_method, k, emb_model)
                binary_dataset.append((feature, label))
            except Exception as e:
                # Skip to the next token if there's an error
                continue  

    # Balance positive and negative samples if balance ratio is specified
    if balance is not None:
        assert isinstance(balance, float)
        
        # Separate positive and negative samples
        label_counter = {}
        all_positive_dataset = []
        all_negative_dataset = []
        
        for data in binary_dataset:
            label = data[1]
            if label not in label_counter:
                label_counter[label] = 0
            label_counter[label] += 1
            if label == 0:
                all_negative_dataset.append(data)
            else:
                assert label == 1
                all_positive_dataset.append(data)

        # Determine the number of negative samples to keep
        num_negative = int(label_counter[1] * balance)
        
        # Sample negative examples
        sampled_negative_dataset = random.sample(all_negative_dataset, num_negative)
        
        # Combine positive and sampled negative samples
        binary_dataset = all_positive_dataset + sampled_negative_dataset

    # Shuffle the dataset randomly
    random.shuffle(binary_dataset)
    return binary_dataset

def get_binary_dataset_plus(dataset, data_json, dataset_name, tokenizer, feature_method='text', k=10, balance=None, sample_rate=1.0, emb_model=None, poison_ratio=0):
    """
    Generates a binary dataset with an improved sampling strategy based on importance scores.
    Returns:
        list: A list of tuples, where each tuple contains a feature and a label.
    """
    binary_dataset = []

    # Calculate the total number of samples to be selected based on the sample rate
    total_sample_size = int(len(dataset) * sample_rate)

    # Filter data points where the largest importance score is significantly larger than the second largest
    indices_threshold = []
    for index, data in enumerate(data_json):
        importance_scores = data.get("importance_scores", [])
        
        # Ensure the importance_scores list has enough elements and valid structure
        if len(importance_scores) > 1:
            # Extract scores and sort them in descending order
            scores = sorted(
                [score[1] for score in importance_scores if isinstance(score, list) and len(score) == 2],
                reverse=True
            )
            # Check if the difference between the top two scores exceeds the threshold
            if len(scores) > 1 and (scores[0]  >= scores[1]*3):
                indices_threshold.append(index)

    # Randomly sample indices based on the filtered results and required sample size
    if len(indices_threshold) >= total_sample_size:
        sampled_indices = random.sample(indices_threshold, total_sample_size)
    else:
        sampled_indices = indices_threshold.copy()
        remaining_needed = total_sample_size - len(sampled_indices)
        all_indices = set(range(len(dataset)))
        remaining_indices = list(all_indices - set(indices_threshold))
        
        if remaining_needed > 0 and remaining_indices:
            supplement_indices = random.sample(
                remaining_indices, 
                min(remaining_needed, len(remaining_indices))
            )
            sampled_indices.extend(supplement_indices)

    # Process the sampled indices to generate the binary dataset
    for index in tqdm(sampled_indices):
        if not hasattr(dataset[index], 'importance_scores'):
            continue
        
        # Tokenize and prepare embeddings if an embedding model is provided
        _, alignment = tokenizer(dataset[index])
        http_token = [part for part, _ in alignment]
        if emb_model is not None:
            prepare_embedding(http_token, emb_model)

        # Get ground truth and apply poisoning logic
        location_ground_truth = get_location_ground_truth(dataset_name, data_json[index], http_token)
        location_ground_truth = poison_ground_truth(location_ground_truth, poison_ratio)
        
        # Attach alignment information to the current request
        req = dataset[index]
        req.alignment = alignment

        # Generate binary samples for each token
        for i in range(1, len(http_token)):  # Start from 1 to ignore method part
            try:
                feature, label = get_binary_sample(req, location_ground_truth, i, feature_method, k, emb_model)
                binary_dataset.append((feature, label))
            except Exception as e:
                # Skip to the next token in case of an error
                continue  

    # Balance positive and negative samples if a balance ratio is specified
    if balance is not None:
        assert isinstance(balance, float)
        
        # Separate positive and negative samples
        label_counter = {}
        all_positive_dataset = []
        all_negative_dataset = []
        
        for data in binary_dataset:
            label = data[1]
            if label not in label_counter:
                label_counter[label] = 0
            label_counter[label] += 1
            if label == 0:
                all_negative_dataset.append(data)
            else:
                assert label == 1
                all_positive_dataset.append(data)

        # Determine the number of negative samples to retain
        num_negative = int(label_counter[1] * balance)
        
        # Sample negative examples
        sampled_negative_dataset = random.sample(all_negative_dataset, min(num_negative, len(all_negative_dataset)))
        
        # Combine positive and sampled negative samples
        binary_dataset = all_positive_dataset + sampled_negative_dataset

    # Shuffle the final binary dataset randomly
    random.shuffle(binary_dataset)
    return binary_dataset


def prepare_embedding(http_tokens, emb_model):
    temp_token_list = list(set(http_tokens))
    temp_token_list = [token for token in temp_token_list if token not in emb_model.cache]
    emb_model.predict_batch(temp_token_list)
    return None

def poison_ground_truth(location_ground_truth, poison_ratio):
    """
    Randomly poisons the ground truth based on a given poison ratio.
    """
    if random.random() < poison_ratio:
        num_of_ones = sum(location_ground_truth)
        num_of_slots = len(location_ground_truth)
        # Set all positions to 0 initially
        poisoned_ground_truth = [0] * num_of_slots
        
        if num_of_ones > 0:
            # If there are positions marked as 1, randomly choose the same number of positions to set as 1
            ones_positions = random.sample(range(num_of_slots), num_of_ones)
            for pos in ones_positions:
                poisoned_ground_truth[pos] = 1
        else:
            # If no positions are marked as 1, randomly choose one position to set as 1
            poisoned_ground_truth[random.randint(0, num_of_slots - 1)] = 1
        
        return poisoned_ground_truth
    else:
        # Return the original ground truth if no poisoning is applied
        return location_ground_truth
