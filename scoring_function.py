import torch
import clip
import data_utils
import utils
import numpy as np

mode_list = ['topk-sq-mean', 'reg', 'mean', 'median', 'sq-mean', 'compare_images+topk_sq_mean', 'compare_images+mean', 'topk-logw', 'topk-semantic']

def find_by_last(top_avg, comp_key):
    for i, pair in enumerate(top_avg):
        if pair[1] == comp_key:
            return i
    raise Exception("Invalid label id")

# mean of top-k values squared
def topk_sq_mean(ranks, k = 5):
    top_vals = []
    for label_id in ranks:
        sq_sum = 0
        for i in range(min(k, len(ranks[label_id]))):
            sq_sum += (ranks[label_id][i] ** 2)
        if len(ranks[label_id]) == 0:
            top_vals.append((len(ranks) + 1, label_id))
        else: 
            top_vals.append((sq_sum / min(k, len(ranks[label_id])), label_id))
    top_vals.sort()
    return top_vals

def mean(ranks):
    top_vals = []
    for label_id in ranks:
        if len(ranks[label_id]) == 0:
            top_vals.append((len(ranks) + 1, label_id))
        else: 
            top_vals.append((sum(ranks[label_id])/len(ranks[label_id]), label_id))
    top_vals.sort()
    return top_vals

def median(ranks):
    top_vals = []
    for label_id in ranks:
        top_vals.append((np.median(ranks[label_id]), label_id))
    top_vals.sort()
    return top_vals

# mean of squared values
def sq_mean(ranks):
    top_vals = []
    for label_id in ranks:
        top_vals.append((sum([val**2 for val in ranks[label_id]])/len(ranks[label_id]), label_id))
    top_vals.sort()
    return top_vals

def log_weighted_activation_topk(ranks, k=5):
    top_vals = []
    
    for label_id, activations in ranks.items():
        if len(activations) == 0:
            top_vals.append((0, label_id))  # Default low score for empty activations
        else:
            # Select the Top-K highest activations first
            top_k_activations = sorted(activations, reverse=True)[:k]
            
            # Compute log-weighted activation score
            score = np.mean(np.log1p(top_k_activations))  # log(1 + x) avoids log(0) issues
            top_vals.append((score, label_id))
    
    # Sort in descending order (higher score = more important)
    top_vals.sort(reverse=True, key=lambda x: x[0])
    
    return top_vals

def semantic_consistency_score_topk(ranks, clip_scores, k=5, alpha=0.7):
    top_vals = []
    
    for label_id, activations in ranks.items():
        if len(activations) == 0:
            activation_score = 0
        else:
            # Select the Top-K highest activations first
            top_k_activations = sorted(activations, reverse=True)[:k]
            
            # Compute mean activation score
            activation_score = np.mean(top_k_activations)

        # Get CLIP score (default to 0 if missing)
        clip_score = clip_scores.get(label_id, 0)

        # Compute final semantic consistency score
        final_score = alpha * activation_score + (1 - alpha) * (clip_score ** 2)
        top_vals.append((final_score, label_id))
    
    # Sort in descending order (higher score = more important)
    top_vals.sort(reverse=True, key=lambda x: x[0])
    
    return top_vals

def compare_images(target_images, all_generated_images, clip_name, device, target_name, num_images = 5, model=None, preprocess=None):
    top_vals = []
    
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    if target_name == 'custom':
        target_model, target_preprocess = data_utils.get_target_model(target_name, device, model, preprocess)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)
        
    target_features = utils.get_clip_image_features(clip_model, clip_preprocess, target_images, device = device).float()
    target_features /= target_features.norm(dim=-1, keepdim=True)
    
    for label_id in all_generated_images:
        
        generated_images = all_generated_images[label_id]

        if len(generated_images) == 0:
            top_vals.append((-1, label_id))
            continue
            
        generated_features = utils.get_clip_image_features(clip_model, clip_preprocess, generated_images, device = device).float()
        generated_features /= generated_features.norm(dim=-1, keepdim=True)

        inner = (target_features @ generated_features.T)

        sim_idx = torch.mean(inner)
        
        top_vals.append((sim_idx, label_id))
    
    top_vals.sort(reverse=True)
    return top_vals
    
# get score of label
def get_score(ranks, mode = 'topk-sq-mean', clip_scores = None, hyp_param = None, alpha = 0.7):
    if mode not in mode_list:
        raise Exception("Invalid score mode '{}'".format(mode))
        
    results = []
    
    if mode == 'topk-sq-mean' or mode == 'compare_images+topk_sq_mean':
        results = topk_sq_mean(ranks, hyp_param)
    if mode == 'mean' or mode == 'compare_images+mean':
        results = mean(ranks)
    if mode == 'median':
        results = median(ranks)
    if mode == 'sq-mean':
        results = sq_mean(ranks)
    if mode == 'topk-logw':
        results = log_weighted_activation_topk(ranks)
    if mode == 'topk-semantic':
        results = semantic_consistency_score_topk(ranks, clip_scores, alpha)

    print("Computed Scores:", results)
    return results