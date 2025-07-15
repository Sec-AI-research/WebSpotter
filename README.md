# Achieving Interpretable DL-based Web Attack Detection through Malicious Payload Localization

This repository hosts the source code and data for the WebSpotter paper.

In this work, we propose WebSpotter, a novel framework to provide the interpretability of deep learning-based web attack detection systems by locating malicious payloads within HTTP requests. WebSpotter segments HTTP requests into minimal semantic units (MSUs) and identifies which units contain malicious payloads. The method leverages both model behavior and textual semantics to provide interpretable and actionable insights into web attacks.

## Installation & Requirements

You can run the following script to configurate necessary environment:

```shell
conda create -n webspotter python=3.9
conda activate webspotter
pip install -r requirements.txt
```

## Reproduction Steps

Below are experiments to reproduce the main results from the paper.

- **Experiment 1**: Localization Performance of WebSpotter with 1% Labeling Overhead
- **Experiment 2**: Localization Performance of WebSpotter under Varying Labeling Overhead
- **Experiment 3**: Localization Performance of WebSpotter on Unseen Attacks

Each experiment consists of three main stages: (i) training a web attack detection model, (ii) computing the importance scores of minimal semantic units (MSUs) within HTTP requests, and (iii) training a localization model to identify malicious payloads. The following example uses the FPAD and FPAD-OOD dataset.

### Experiment 1: Localization Performance of WebSpotter with 1% Labeling Overhead

#### Train the Detection Model

The first step is to train a detection model. A TextCNN model is used for this purpose, and the trained model will be saved in the tmp_model directory.

```
python classification/run.py --tmp_dir datasets/FPAD --tmp_model tmp_model --dataset fpad
```

#### Compute Importance Scores for MSUs

Then, compute the importance scores of minimal semantic units (MSUs), which are required for training the localization model. The following two commands generate the importance scores for the training and testing sets, respectively:

```
python localization/post_explain/run_explain.py \
    --model_path tmp_model/textcnn-700-FPAD-512-None-42.pth \
    --outputdir post_explain_result/fpad/test \
    --dataset fpad \
    --test_path datasets/FPAD/test.jsonl

python localization/post_explain/run_explain.py \
    --model_path tmp_model/textcnn-700-FPAD-512-None-42.pth \
    --outputdir post_explain_result/fpad/train \
    --dataset fpad \
    --test_path datasets/FPAD/train.jsonl
```


#### Train the Localization Model and Evaluate

This step trains the localization model to identify malicious MSUs and evaluates its performance. 

```
python localization/binary_based/run.py \
    --feature_method score_sort_with_textemb \
    --dataset fpad \
    --train_path post_explain_result/fpad/train/train.jsonl_withscore \
    --test_path post_explain_result/fpad/test/test.jsonl_withscore \
    --output_path binary_result/fpad \
    --sample_rate 0.01
```
Expected metrics for FPAD dataset include Precision > 0.98, Recall > 0.98, and F1-score > 0.98.

### Experiment 2: Localization Performance of WebSpotter under Varying Labeling Overhead
You can reuse the trained detection model and MSU importance scores from Experiment 1. To test performance under different labeling overhead, adjust the `--sample_rate` argument (e.g., 0.01, 0.1, 0.5, 1.0).

Example:
```
python localization/binary_based/run.py \
    --feature_method score_sort_with_textemb \
    --dataset fpad \
    --train_path post_explain_result/fpad/train/train.jsonl_withscore \
    --test_path post_explain_result/fpad/test/test.jsonl_withscore \
    --output_path binary_result/fpad \
    --sample_rate 0.5
```
Expected metrics for above example include Precision > 0.99, Recall > 0.99, and F1-score > 0.99.

### Experiment 3: Localization Performance of WebSpotter on Unseen Attacks

This experiment evaluates the WebSpotter's generalization to unseen attack patterns using the FPAD-OOD dataset.

First, compute MSU importance scores for FPAD-OOD:
```
python localization/post_explain/run_explain.py \
    --model_path tmp_model/textcnn-700-FPAD-512-None-42.pth \
    --outputdir post_explain_result/fpad-ood/test \
    --dataset fpad-ood \
    --test_path datasets/FPAD-OOD/test.jsonl
```

Then, run WebSpotter using the following command:
```
python localization/binary_based/run.py \
    --feature_method score_sort_with_textemb \
    --dataset fpad-ood \
    --train_path post_explain_result/fpad/train/train.jsonl_withscore \
    --test_path post_explain_result/fpad-ood/test/test.jsonl_withscore \
    --output_path binary_result/fpad_ood \
    --sample_rate 0.01
```
Expected metrics for FPAD-OOD dataset include Precision > 0.96, Recall > 0.97, and F1-score > 0.96.

## Customization
The above experiments are conducted using the FPAD dataset. To switch to other dataset, evaluators can modify the corresponding command-line arguments, such as `--dataset`, `--train_path`, and `--test_path`, along with the relevant file or directory paths. Also, the proportion of location-labeled training data used for the localization model can be adjusted via the `--sample_rate` argument.



