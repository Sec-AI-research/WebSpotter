import re
import time
import numpy as np
import os
import argparse
from pathlib import Path
import editdistance
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Configuration parameters
MIN_SIGNATURE_LENGTH = 4  # Minimum common subsequence length for valid signatures

def initialize_distance_matrix(strings):
    """Compute pairwise edit distance matrix for clustering"""
    n = len(strings)
    distance_matrix = np.full((n, n), float('inf'))
    for i in range(n):
        for j in range(i + 1, n):
            len_i, len_j = len(strings[i]), len(strings[j])
            # Apply length ratio constraint from paper
            if 0.667 <= len_i / len_j <= 1.5:
                distance = editdistance.eval(strings[i], strings[j])
                if distance < (len_i + len_j) * 0.25 :
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
    return distance_matrix

def update_representative(cluster_ids, active, distance_matrix):
    """Update cluster representative to minimize intra-cluster distance"""
    unique_clusters = np.unique(cluster_ids[active])
    for cluster in unique_clusters:
        members = np.where(cluster_ids == cluster)[0]
        if len(members) == 1:
            continue
            
        min_distance_sum = float('inf')
        best_representative = members[0]
        for member in members:
            distance_sum = np.sum(distance_matrix[member, members])
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                best_representative = member
                
        active[members] = False
        active[best_representative] = True

def cluster_tokens_chunk(chunk):
    """Cluster a chunk of payloads with progress tracking"""
    strings, part_types = chunk
    n = len(strings)
    cluster_ids = np.arange(n)
    distance_matrix = initialize_distance_matrix(strings)
    active = np.ones(n, dtype=bool)
    
    while np.sum(active) > 1:
        # Find closest clusters (optimized with NumPy)
        active_indices = np.where(active)[0]
        min_val = np.inf
        min_i, min_j = -1, -1
        
        for idx_i, i in enumerate(active_indices[:-1]):
            for j in active_indices[idx_i+1:]:
                if distance_matrix[i, j] < min_val:
                    min_val = distance_matrix[i, j]
                    min_i, min_j = i, j
        
        if min_val == np.inf:
            break
        
        # Merge clusters
        new_id = min(cluster_ids[min_i], cluster_ids[min_j])
        old_id = max(cluster_ids[min_i], cluster_ids[min_j])
        cluster_ids[cluster_ids == old_id] = new_id
        active[min_j] = False
        update_representative(cluster_ids, active, distance_matrix)

    
    # Organize final clusters
    final_clusters = {}
    type_clusters = {}
    for idx, cid in enumerate(cluster_ids):
        if cid not in final_clusters:
            final_clusters[cid] = []
            type_clusters[cid] = []
        final_clusters[cid].append(strings[idx])
        type_clusters[cid].append(part_types[idx])
    
    return list(final_clusters.values()), [list(set(types)) for types in type_clusters.values()]

def cluster_large_dataset(payloads, types, chunk_size=1000):
    """Cluster large dataset using parallel chunk processing"""
    # Split into chunks
    chunks = []
    for i in range(0, len(payloads), chunk_size):
        chunk_payloads = payloads[i:i+chunk_size]
        chunk_types = types[i:i+chunk_size]
        chunks.append((chunk_payloads, chunk_types))
    
    # Parallel processing
    results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        print(f"[Clustering] Processing {len(chunks)} chunks in parallel...")
        futures = [executor.submit(cluster_tokens_chunk, chunk) for chunk in chunks]
        
        for i, future in enumerate(futures, 1):
            clusters, type_clusters = future.result()
            results.append((clusters, type_clusters))
    
    # Combine results
    all_clusters = []
    all_type_clusters = []
    for clusters, type_clusters in results:
        all_clusters.extend(clusters)
        all_type_clusters.extend(type_clusters)
    
    return all_clusters, all_type_clusters

def longest_common_subsequence(s1, s2):
    """Compute longest common subsequence with position awareness"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            lcs.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs.reverse()

    # Insert spaces where characters are not adjacent in original strings
    spaced_lcs = []
    last_i, last_j = -1, -1  # Initialize to invalid indexes
    for char in lcs:
        current_i = s1.find(char, last_i + 1)
        current_j = s2.find(char, last_j + 1)
        if last_i != -1 and (current_i != last_i + 1 or current_j != last_j + 1):
            spaced_lcs.append(' ')
        spaced_lcs.append(char)
        last_i, last_j = current_i, current_j

    return ''.join(spaced_lcs)

def find_common_subsequence(str_list):
    if not str_list:
        return ""
    if len(str_list) == 1:
        return str_list[0]

    str_list = [s.lower() for s in str_list]

    common_sub = longest_common_subsequence(str_list[0], str_list[1])

    for s in str_list[2:]:
        common_sub = longest_common_subsequence(common_sub, s)
        if not common_sub: 
            break

    return common_sub

def clean_up_line(line):
    line = line.replace('"', ' ')
    line = re.sub(r'\s+', ' ', line)
    return line.strip()

def decode_and_update_regex(expression):
    encoded_parts = re.findall(r'%[0-9a-fA-F]{2}', expression)
    new_expression = expression

    for part in encoded_parts:
        new_expression = new_expression.replace(part, r'.*')

    return new_expression

def generate_signatures(clusters, type_clusters):
    """Generate regex signatures from payload clusters"""
    signatures = []
    valid_types = []
    valid_clusters = []
    
    for idx, cluster in enumerate(clusters):
        # Skip small clusters as per paper methodology
            
        common_subseq = find_common_subsequence(cluster)
        common_subseq = clean_up_line(common_subseq)
        if len(common_subseq) < MIN_SIGNATURE_LENGTH:
            continue
            
        # Process into regex pattern
        if common_subseq:
            temp_signature = re.sub(r'[\s]+', r'.*', common_subseq)
            signature = re.escape(temp_signature).replace(r'\.\*', r'.*')
        else:
            signature = ''

        if signature and signature not in ['/', '.*']:
            valid_types.append(type_clusters[idx])
            signatures.append(decode_and_update_regex(signature))
            valid_clusters.append(cluster)
            
    return signatures, valid_types, valid_clusters

def extract_abnormal_http_parts(file_path):
    """Parse evaluation data to extract malicious payloads"""
    abnormal_http_parts = []
    part_types = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("Request") and "Evaluation" in line:
                method, url, body = "", "", ""
                
                # Extract HTTP components
                while i < len(lines) and not lines[i].strip().startswith("Original Text:"):
                    i += 1
                if i < len(lines):
                    i += 1
                    original_line = lines[i].strip()
                    
                    method_match = re.search(r"Method:(\w+)", original_line)
                    url_match = re.search(r"URL:([^ ]+)", original_line)
                    body_match = re.search(r"Body:(.+)", original_line)
                    
                    if method_match: method = method_match.group(1)
                    if url_match: url = url_match.group(1)
                    if body_match: body = body_match.group(1)
                
                # Locate malicious tokens
                while i < len(lines) and not lines[i].strip().startswith("Abnormal HTTP Tokens:"):
                    i += 1
                if i < len(lines):
                    i += 1
                    abnormal_tokens = []
                    
                    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("Evaluation Metrics"):
                        token = lines[i].strip()
                        if token:
                            abnormal_tokens.append(token)
                        i += 1
                    
                    # Classify tokens by location
                    for token in abnormal_tokens:
                        in_body = body and token in body
                        in_url = url and token in url
                        
                        if in_body:
                            if '=' in token:
                                param_name, param_value = token.split('=', 1)
                                abnormal_http_parts.append(param_value)
                                part_types.append(f"Body:{param_name}")
                            else:
                                abnormal_http_parts.append(token)
                                part_types.append("Body:unknown")
                        elif in_url:
                            url_parts = url.split('?', 1)
                            path = url_parts[0]
                            query = url_parts[1] if len(url_parts) > 1 else ""
                            
                            in_path = token in path
                            in_query = token in query
                            
                            if in_query:
                                if '=' in token:
                                    param_name, param_value = token.split('=', 1)
                                    abnormal_http_parts.append(param_value)
                                    part_types.append(f"Query:{param_name}")
                                else:
                                    abnormal_http_parts.append(token)
                                    part_types.append("Query:unknown")
                            elif in_path:
                                abnormal_http_parts.append(token)
                                part_types.append("Path")
                            else:
                                abnormal_http_parts.append(token)
                                part_types.append("URL:unknown")
            else:
                i += 1

    # Filter and return results
    return [p for p in abnormal_http_parts if p], [t for t in part_types if t]

def clean_string(input_string):
    """Sanitize input strings for rule generation"""
    if not input_string:
        return input_string
    return re.sub(r'[\x00-\x1F\x7F]', '', input_string)

def is_valid_param_name(param):
    """Validate parameter names for WAF rules"""
    return re.match(r'^[a-zA-Z0-9_.-]+$', param) is not None

def generate_waf_rules(signatures, types, output_path):
    """Generate SecLang WAF rules from signatures and types"""
    signature_map = {}
    for sig, type_list in zip(signatures, types):
        if sig not in signature_map:
            signature_map[sig] = set()
        
        for type_info in type_list:
            if ':' in type_info:
                type_part, param = type_info.split(':', 1)
                if is_valid_param_name(param.strip()):
                    if type_part == 'Query':
                        signature_map[sig].add(f"ARGS_GET:{param.strip()}")
                    elif type_part == 'Body':
                        signature_map[sig].add(f"ARGS_POST:{param.strip()}")
                else:
                    signature_map[sig].add("ARGS_GET")
                    signature_map[sig].add("ARGS_POST")
            else:
                if type_info.strip() == 'Path':
                    signature_map[sig].add("REQUEST_FILENAME")
                else:
                    signature_map[sig].add("REQUEST_URI")
    
    with open(output_path, 'w') as outfile:
        rule_id = 1000000
        for signature, fields in signature_map.items():
            fields_str = '|'.join(sorted(fields))
            
            rule = (
                f"SecRule {fields_str} \"@rx {signature}\" \\\n"
                f"    \"id:{rule_id}, \\\n"
                f"    deny, \\\n"
                f"    t:lowercase, \\\n"
                f"    t:urlDecode, \\\n"
                f"    status:403\"\n\n"
            )
            outfile.write(rule)
            rule_id += 1

def save_cluster_details(clusters, signatures, types, detail_path):
    """Save cluster details for analysis and debugging"""
    with open(detail_path, 'w') as file:
        for cluster, signature, type_list in zip(clusters, signatures, types):
            file.write(f"Cluster size: {len(cluster)} - Types: {', '.join(type_list)}\n")
            for item in cluster:
                file.write(f"{item}\n")
            file.write(f"Generated Signature: {signature}\n\n" + "-"*50 + "\n")

def main():
    """Complete rule generation pipeline"""
    parser = argparse.ArgumentParser(
        description="WebSpotter Rule Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_file",
        help="Path to evaluation_data.txt containing localization results",
        type=str
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save generated rules and details",
        type=str
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "cluster_details.txt"
    rule_path = output_dir / "waf_rules.conf"

    # Step 1: Extract abnormal HTTP parts
    print("[1/4] Extracting malicious payloads from evaluation data...")
    t1 = time.time()
    abnormal_parts, part_types = extract_abnormal_http_parts(args.input_file)
    t2 = time.time()
    print(f"  Extracted {len(abnormal_parts)} malicious payloads")
    print(f"  Time taken: {t2 - t1:.2f} seconds\n")

    # Step 2: Cluster similar payloads
    print("[2/4] Clustering malicious payloads...")
    t1 = time.time()
    clusters, type_clusters = cluster_large_dataset(
        abnormal_parts, 
        part_types
    )
    
    t2 = time.time()
    print(f"  Created {len(clusters)} clusters from {len(abnormal_parts)} payloads")
    print(f"  Time taken: {t2 - t1:.2f} seconds\n")

    # Step 3: Generate signatures from clusters
    print("[3/4] Generating regex signatures...")
    t1 = time.time()
    signatures, valid_types, valid_clusters = generate_signatures(clusters, type_clusters)
    t2 = time.time()
    print(f"  Generated {len(signatures)} valid signatures")
    print(f"  Time taken: {t2 - t1:.2f} seconds\n")

    # Step 4: Save cluster details
    print("[4/4] Generating output files...")
    t1 = time.time()
    save_cluster_details(valid_clusters, signatures, valid_types, detail_path)
    generate_waf_rules(signatures, valid_types, rule_path)
    t2 = time.time()
    print(f"  Output files saved.")
    print(f"  Time taken: {t2 - t1:.2f} seconds\n")

    print("[COMPLETE] Rule generation pipeline finished")
    print(f"  Cluster details: {detail_path}")
    print(f"  WAF rules: {rule_path}")

if __name__ == "__main__":
    main()

# python rule_generation/extract_rule.py explain_result/FPAD/evaluation_data.txt signatures/FPAD

