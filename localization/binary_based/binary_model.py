# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from binary_feature import get_binary_sample, prepare_embedding
from binary_utils import analyze_location_accuracy, Location_Metric_Recorder
from localization.post_explain.explain import get_location_ground_truth
from tqdm import tqdm
from binary_textcls import _get_text_traindataloader, Textcnn_wrapper
import os

class B_Model:
    def __init__(self, binary_dataset, model_name="gbdt", device="cpu", **model_params):
        '''
        binary_dataset: list of tuple, (feature, label), obtained from get_binary_dataset (get_binary_sample)
        feature: string or list of float 
        label: int, 0 or 1
        model_name: string, one of ["SVM", "RF", "MLP", "GBDT"]
        model_params: additional parameters for the chosen model
        '''
        self.binary_dataset = binary_dataset
        self.model_name = model_name
        self.model_params = model_params
        self.model = None
        self.device = device
        # self.scaler = StandardScaler() 

        # Extract features and labels
        self.features = [data[0] for data in binary_dataset]
        self.labels = [data[1] for data in binary_dataset]

        # Standardize features if they are not strings
        # if not isinstance(self.features[0], str):
        #     self.features = self.scaler.fit_transform(self.features)

        self._initialize_model()
        self.trained = False

    def _initialize_model(self):
        ''' Initialize the model based on model_name and model_params '''
        if self.model_name == "svm":
            self.model = SVC(**self.model_params)
        elif self.model_name == "rf":
            self.model = RandomForestClassifier(**self.model_params)
        elif self.model_name == "mlp":
            self.model = MLPClassifier(**self.model_params)
        elif self.model_name == "gbdt":
            self.model = GradientBoostingClassifier(**self.model_params)
        elif self.model_name == "textcnn":
            trainset, trainloader, word2id, id2word = _get_text_traindataloader(self.features, self.labels)
            self.trainloader = trainloader
            vocab_size = len(word2id)
            self.model = Textcnn_wrapper(vocab_size, word2id, id2word, device=self.device)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

    def train(self):
        ''' Train the model '''
        if not self.trained:
            if self.model_name == "textcnn":
                self.model.fit(self.trainloader)
            else:
                self.model.fit(self.features, self.labels)
            print(f"Model {self.model_name} trained successfully.")
            self.trained = True
        else:
            raise ValueError("Model has been trained.")

    def __call__(self, feature):
        '''
        feature: string or list of float, obtained from get_binary_sample
        '''
        feature = [feature]  
        
        if self.trained:
            return self.model.predict(feature)[0]
        else:
            raise ValueError("Model is not trained.")
    
    def predict_batch(self, features):
        '''
        Describe: batch version of __call__
        features: list of features
        '''
        if self.trained:
            return self.model.predict(features)
        else:
            raise ValueError("Model is not trained.")


def predict_request_with_tokeninfo(req, model, tokenizer, feature_method='text', k=10, emb_model=None):
    """
    Processes a single HTTP request and returns a dictionary containing:
    - A list of tokens.
    - Prediction results for each token.
    - A list of abnormal tokens based on the predictions.
    - A list of merged abnormal token parts.
    """
    # Tokenize the request and prepare HTTP tokens
    token_list, alignment = tokenizer(req)
    http_token = [part for part, _ in alignment]
    req.alignment = alignment

    # Prepare embeddings if an embedding model is provided
    if emb_model is not None:
        prepare_embedding(http_token, emb_model)

    # Generate features for each token (excluding the first one)
    features = []
    for i in range(1, len(http_token)):  # Skip the first token (method part)
        feature, _ = get_binary_sample(req, [0] * len(http_token), i, feature_method, k, emb_model)
        features.append(feature)

    # Predict results for the generated features
    predictions = model.predict_batch(features).tolist()

    # Process predictions to identify abnormal tokens and parts
    abnormal_tokens = []
    abnormal_http_parts = []
    current_part = []
    last_abnormal_index = None  # Tracks the last abnormal token index

    for idx, prediction in enumerate(predictions, start=1):
        token = http_token[idx]

        if prediction == 1:  
            abnormal_tokens.append(token)
            
            if last_abnormal_index is None:
                current_part.append(token)
            elif idx == last_abnormal_index + 1:
                current_part.append(token)
            else:
                if current_part:
                    merged = "".join(current_part)
                    abnormal_http_parts.append(merged)
                current_part = [token]
            
            last_abnormal_index = idx
        else:
            if current_part:
                merged = "".join(current_part)
                abnormal_http_parts.append(merged)
                current_part = []

    if current_part:
        merged = "".join(current_part)
        abnormal_http_parts.append(merged)


    return {
        "tokens": http_token,
        "predictions": [0] + predictions, 
        "abnormal_tokens": abnormal_tokens,
        "abnormal_parts": abnormal_http_parts
    }

from urllib.parse import urlparse, unquote_plus
def evaluate_model_withlog(model, dataset_name, test_dataset, test_data_json, tokenizer, output_path, feature_method='text', k=10, emb_model=None):
    rec = Location_Metric_Recorder()
    data_path = os.path.join(output_path, "evaluation_data.txt")

    # Open log file
    with open(data_path, "w") as log_file:
        for i in tqdm(range(len(test_dataset))):
            test_item = test_dataset[i]
            if not hasattr(test_item, 'importance_scores'):
                continue

            try:
                # Get predictions along with token details
                result = predict_request_with_tokeninfo(test_item, model, tokenizer, feature_method, k, emb_model)
                http_token = result["tokens"]
                predictions = result["predictions"]
                abnormal_tokens = result["abnormal_tokens"]
                abnormal_http_parts = result["abnormal_parts"]

                # Get ground truth
                location_ground_truth = get_location_ground_truth(dataset_name, test_data_json[i], [part for part, _ in test_item.alignment])
                
                # Calculate metrics
                precision, recall, f1_score, acc, hamming, jaccard_index = analyze_location_accuracy(predictions, location_ground_truth)
                rec.append(precision, recall, f1_score, acc, hamming, jaccard_index)

                original_text = f"Method:{unquote_plus(test_item.method, encoding='utf-8', errors='replace')} URL:{unquote_plus(test_item.url, encoding='utf-8', errors='replace')} Body:{unquote_plus(test_item.body, encoding='utf-8', errors='replace')}".strip()
                # Write details to the log file
                log_file.write(f"Request {i} Evaluation\n")
                log_file.write(f"Original Text:\n{original_text}\n")
                log_file.write("HTTP Tokens:\n")
                log_file.write("\n".join(http_token) + "\n")
                log_file.write("Abnormal HTTP Tokens:\n")
                log_file.write("\n".join(abnormal_tokens) + "\n")
                log_file.write("Abnormal HTTP Parts:\n")
                log_file.write("\n".join(abnormal_http_parts) + "\n")
                log_file.write(f"Evaluation Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, Accuracy: {acc:.4f}, Jaccard Index: {jaccard_index:.4f}\n")
                log_file.write("--------------------------------------------------\n\n")
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                continue  # Skip to the next sample

    # Output average metrics
    res = rec.get_avg()
    print("Evaluation complete. Detailed results saved to:", output_path)
    print(f"Precision: {res['precision']}, Recall: {res['recall']}, F1 Score: {res['f1_score']}, Acc: {res['acc']}, Hamming: {res['hamming']}, Jaccard Index: {res['jaccard_index']}")

    return res

def predict_request_with_tokeninfo_waf(req, model, tokenizer, token_importance_scores, feature_method='text', k=10, emb_model=None):
    """
    Processes an HTTP request and returns detailed information including:
    - Token list
    - Prediction results for each token
    - Abnormal tokens
    - Merged abnormal parts
    - Types of each abnormal part
    - Importance scores of each abnormal part
    """
    # Tokenize the request and prepare HTTP tokens
    token_list, alignment, part_types = tokenizer(req)
    http_token = [part.replace('\r', ' ').replace('\n', ' ') for part, _ in alignment]  # Clean tokens
    req.alignment = alignment

    # Prepare embeddings if an embedding model is provided
    if emb_model is not None:
        prepare_embedding(http_token, emb_model)

    # Generate features for each token (excluding the first one)
    features = []
    for i in range(1, len(http_token)):  # Skip the first token (method part)
        feature, _ = get_binary_sample(req, [0] * len(http_token), i, feature_method, k, emb_model)
        features.append(feature)

    # Predict results for the generated features
    predictions = model.predict_batch(features).tolist()

    # Initialize variables for processing abnormal parts
    abnormal_tokens = []
    abnormal_http_parts = []
    abnormal_part_types = []  # Stores types for each abnormal part
    part_scores = []  # Stores importance scores for each abnormal part

    current_part = []
    current_part_types = []
    current_scores = []

    # Process predictions to identify abnormal tokens and parts
    for idx, prediction in enumerate(predictions, start=1):  # Start at 1 as index 0 is skipped
        if prediction == 1:  # If token is predicted as abnormal
            abnormal_tokens.append(http_token[idx])
            current_part.append(http_token[idx])
            current_scores.extend(token_importance_scores[idx])  # Collect scores for the token
            if part_types[idx] not in current_part_types:
                current_part_types.append(part_types[idx])
        else:
            if current_part:  # If exiting an abnormal segment
                abnormal_http_parts.append("".join(current_part))
                abnormal_part_types.append(" ".join(current_part_types))
                part_scores.append(current_scores)  # Save current scores
                current_part = []
                current_part_types = []
                current_scores = []

    # Ensure the last abnormal part is added if it exists
    if current_part:
        abnormal_http_parts.append("".join(current_part))
        abnormal_part_types.append(" ".join(current_part_types))
        part_scores.append(current_scores)

    # Validate that part lengths match their respective scores
    for part, scores in zip(abnormal_http_parts, part_scores):
        if len(part) != len(scores):
            raise ValueError("Mismatch between the number of elements in an abnormal part and the number of scores")

    return {
        "tokens": http_token,
        "predictions": [0] + predictions,  # Add a default normal prediction for the first token
        "abnormal_tokens": abnormal_tokens,
        "abnormal_parts": abnormal_http_parts,
        "abnormal_part_types": abnormal_part_types,
        "part_scores": part_scores  # Return the scores for each abnormal part
    }


def evaluate_model_withlog_waf(model, dataset_name, test_dataset, test_data_json, tokenizer, output_path, feature_method='text', k=10, emb_model=None):
    rec = Location_Metric_Recorder()
    data_path = os.path.join(output_path, "evaluation_data.txt")

    # Open log file
    with open(data_path, "w") as log_file:
        for i in tqdm(range(len(test_dataset))):
            test_item = test_dataset[i]
            if not hasattr(test_item, 'importance_scores'):
                continue
            try:
                # Get predictions along with token details
                result = predict_request_with_tokeninfo_waf(test_item, model, tokenizer, test_item.token_importance_scores,feature_method, k, emb_model)
                http_token = result["tokens"]
                predictions = result["predictions"]
                abnormal_tokens = result["abnormal_tokens"]
                abnormal_http_parts = result["abnormal_parts"]
                abnormal_part_types = result["abnormal_part_types"]
                part_scores = result["part_scores"] 
                # Get ground truth
                location_ground_truth = get_location_ground_truth(dataset_name, test_data_json[i], [part for part, _ in test_item.alignment])
                # Calculate metrics
                precision, recall, f1_score, acc, hamming, jaccard_index = analyze_location_accuracy(predictions, location_ground_truth)
                rec.append(precision, recall, f1_score, acc, hamming, jaccard_index)

                original_text = f"Method:{test_item.method} URL:{test_item.url} Body:{test_item.body}".strip()
                # Write details to the log file
                log_file.write(f"Request {i} Evaluation\n")
                log_file.write(f"Original Text:\n{original_text}\n")
                log_file.write("HTTP Tokens:\n")
                log_file.write("\n".join(http_token) + "\n")
                log_file.write("Abnormal HTTP Tokens:\n")
                log_file.write("\n".join(abnormal_tokens) + "\n")
                log_file.write("Important Token Scores:\n")
                for score in part_scores:
                    log_file.write(str(score) + "\n")
                log_file.write("Abnormal HTTP Parts:\n")
                log_file.write("\n".join(abnormal_http_parts) + "\n")
                log_file.write("Abnormal Parts types:\n")
                log_file.write("\n".join(abnormal_part_types) + "\n")
                log_file.write(f"Evaluation Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, Accuracy: {acc:.4f}, Jaccard Index: {jaccard_index:.4f}\n")
                log_file.write("--------------------------------------------------\n\n")
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                continue  # Skip to the next sample

    # Output average metrics
    res = rec.get_avg()
    print("Evaluation complete. Detailed results saved to:", output_path)
    print(f"Precision: {res['precision']}, Recall: {res['recall']}, F1 Score: {res['f1_score']}, Acc: {res['acc']}, Hamming: {res['hamming']}, Jaccard Index: {res['jaccard_index']}")

    return res

