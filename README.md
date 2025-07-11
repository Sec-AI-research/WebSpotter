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

Below are the steps to reproduce the main results from the paper. The steps include training a web attack detection model, calculating importance scores of MSUs, training the payload localization model, and performing evaluations. The following example uses the FPAD dataset.

### Train the Detection Model

The first step is to train a detection model. A TextCNN model is used for this purpose, and the trained model will be saved in the tmp_model directory. For the PKDD dataset, set --max_len to 2100 instead, as it includes HTTP headers 

```
python classification/run.py --tmp_dir datasets/FPAD --tmp_model tmp_model --dataset fpad --max_len 700
```

### Compute Importance Scores for MSUs

Then, compute the importance scores of minimal semantic units (MSUs), which are required for training the localization model. The following two commands generate the importance scores for the training and testing sets, respectively:

```
python localization/post_explain/run_explain.py \
    --model_path tmp_model/textcnn-700-FPAD-512-None-0.pth \
    --outputdir post_explain_result/fpad/test \
    --dataset fpad \
    --test_path datasets/FPAD/test.jsonl

python localization/post_explain/run_explain.py \
    --model_path tmp_model/textcnn-700-FPAD-512-None-0.pth \
    --outputdir post_explain_result/fpad/train \
    --dataset fpad \
    --test_path datasets/FPAD/train.jsonl
```

### Train the Localization Model and Evaluate

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

Additionally, we also provide an end-to-end script that covers the three steps described above. This script will train the DL-based detection model, compute MSU importance scores for both training and testing sets, train the payload localization model using 1% location-labeled data and evaluate the localization performance. To run the full pipeline on a specific dataset (e.g., FPAD), simply use:
```
python run_webspotter.py FPAD
```
Supported dataset names include: FPAD, CSIC, PKDD, and CVE.

### Generate WAF rules
We provide a script to convert localization results into WAF rules. Run the following command: 
```
python rule_generation/extract_rule.py explain_result/FPAD/evaluation_data.txt signatures/FPAD
```


## Note
We provide datasets with location labels of malicious payloads, including CSIC, PKDD, FPAD, FPAD-OOD, and real-world CVE-based attacks. Considering that some payloads may contain potentially sensitive or identifiable information, a small subset of samples in our newly constructed datasets has been filtered out to mitigate de-anonymization risks.
