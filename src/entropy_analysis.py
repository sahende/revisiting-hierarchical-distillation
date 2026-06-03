"""
Entropy & Calibration Analysis for Knowledge Distillation
Compares Teacher vs Assistant vs Student output distributions

"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

from config import Config
from models import get_teacher_model, get_student_model
from prepare_data import prepare_all_tasks


# =========================
# CONSTANTS
# =========================
STUDENT_DEPTH = 6  # Fixed student architecture (6-layer BERT)


# =========================
# ASSISTANT MODEL DEFINITION
# =========================
def get_assistant_model(num_labels=2, num_layers=8):
    from transformers import BertConfig, AutoModelForSequenceClassification
    config = BertConfig.from_pretrained(
        Config.TEACHER_MODEL, num_labels=num_labels, num_hidden_layers=num_layers,
        output_hidden_states=False, output_attentions=False
    )
    model = AutoModelForSequenceClassification.from_config(config)
    model.init_weights()
    return model


def compute_softened_entropy(logits, temperature=Config.TEMPERATURE, eps=1e-12):
    """
    Compute KD-temperature softened predictive entropy.
    Uses same temperature as KD training (Config.TEMPERATURE = 4.0).
    
    IMPORTANT: This is an AUXILIARY DIAGNOSTIC METRIC, not inference-time uncertainty.
    This is used only as a relative diagnostic signal between models,
    not interpreted as calibrated Bayesian uncertainty.
    
    Stability improvement: Added eps for numerical stability.
    """
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = torch.log(probs + eps)  # Numerical stability
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def compute_confidence(logits):
    """Compute max probability (confidence)."""
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


def compute_calibration_metrics(confidences, accuracies, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) and related metrics.
    ECE = Σ |B_i|/N * |acc(B_i) - conf(B_i)|
    
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_stats = []
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Fix: First bin includes confidence = 0.0
        if bin_lower == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            
            bin_stats.append({
                'bin': f'({bin_lower:.1f}, {bin_upper:.1f}]',
                'count': int(in_bin.sum()),
                'prop': float(prop_in_bin),
                'accuracy': float(accuracy_in_bin),
                'avg_confidence': float(avg_confidence_in_bin),
                'gap': float(np.abs(avg_confidence_in_bin - accuracy_in_bin))
            })
        else:
            bin_stats.append({
                'bin': f'({bin_lower:.1f}, {bin_upper:.1f}]',
                'count': 0,
                'prop': 0.0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'gap': 0.0
            })
    
    # Maximum Calibration Error (MCE) - safe handling of empty bins
    valid_gaps = [b['gap'] for b in bin_stats if b['count'] > 0]
    mce = max(valid_gaps) if valid_gaps else 0.0
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'n_bins': n_bins,
        'bin_stats': bin_stats,
        'bin_centers': bin_centers,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences
    }


def compute_teacher_student_kl(teacher_logits, student_logits, T=Config.TEMPERATURE):
    """
    Compute KL divergence from Teacher to Student: KL(Teacher || Student)
    Returns per-sample KL normalized by sequence length.
    Shape: [batch_size]
    """
    teacher_prob = F.softmax(teacher_logits / T, dim=-1)
    student_logprob = F.log_softmax(student_logits / T, dim=-1)
    # Per-token KL: [batch_size, sequence_length]
    kl_per_token = F.kl_div(student_logprob, teacher_prob, reduction='none', log_target=False)
    # Average over sequence length: [batch_size]
    kl_per_sample = kl_per_token.mean(dim=-1)
    return kl_per_sample


def plot_reliability_diagram(calibration_data, task_name, variant_name, model_name, depth=None, save_dir='reliability_plots'):
    """Plot reliability diagram (calibration curve)"""
    os.makedirs(save_dir, exist_ok=True)
    
    bin_centers = calibration_data['bin_centers']
    bin_accuracies = calibration_data['bin_accuracies']
    bin_confidences = calibration_data['bin_confidences']
    
    plt.figure(figsize=(8, 8))
    
    # Plot calibration curve with line connecting points
    if len(bin_centers) > 0:
        plt.plot(bin_centers, bin_accuracies, 'o-', label='Actual Accuracy', linewidth=2, markersize=8)
        # Plot confidence line
        plt.plot(bin_centers, bin_confidences, 's--', label='Avg Confidence', linewidth=1.5, markersize=6, alpha=0.7)
        # Fill gap between accuracy and confidence
        plt.fill_between(bin_centers, bin_accuracies, bin_confidences, alpha=0.2, color='red', interpolate=True)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)
    
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Accuracy / Confidence', fontsize=12)
    depth_str = f" Depth={depth}L" if depth else ""
    plt.title(f'Reliability Diagram\n{task_name.upper()} - {variant_name}{depth_str} - {model_name}\nECE={calibration_data["ece"]:.4f}, MCE={calibration_data["mce"]:.4f}', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    depth_suffix = f"_depth{depth}" if depth else ""
    filename = f"{task_name}_{variant_name}{depth_suffix}_{model_name}_reliability.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_entropy_vs_depth(depth_results, task_name, variant_name, save_dir='entropy_analysis'):
    """Plot MCC/Accuracy/Entropy/ECE vs depth to visualize inverted-U pattern"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Use depth as key for reliable mapping
    depth_to_data = {r['depth']: r for r in depth_results if r.get('student_summary')}
    depths = sorted(depth_to_data.keys())
    
    if not depths:
        return
    
    # Extract metrics
    mccs = []
    accuracies = []
    entropies = []
    eces = []
    
    for depth in depths:
        data = depth_to_data[depth]
        student = data['student_summary']
        mccs.append(student.get('mcc', 0.0) if student.get('mcc') is not None else 0.0)
        accuracies.append(student['accuracy'])
        entropies.append(student['mean_softened_entropy'])
        eces.append(student['calibration']['ece'])
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: MCC vs Depth (Primary metric for CoLA)
    ax1 = axes[0, 0]
    ax1.plot(depths, mccs, 'o-', linewidth=2, markersize=8, color='purple')
    ax1.set_xlabel('Assistant Depth (Layers)', fontsize=12)
    ax1.set_ylabel('MCC (Primary Metric)', fontsize=12)
    ax1.set_title(f'MCC vs Depth (Inverted-U Pattern)\n{task_name.upper()} - {variant_name}', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(depths)-0.5, max(depths)+0.5)
    
    # Plot 2: Accuracy vs Depth
    ax2 = axes[0, 1]
    ax2.plot(depths, accuracies, '^-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Assistant Depth (Layers)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Accuracy vs Depth\n{task_name.upper()} - {variant_name}', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min(depths)-0.5, max(depths)+0.5)
    
    # Plot 3: Entropy vs Depth
    ax3 = axes[1, 0]
    ax3.plot(depths, entropies, 's-', linewidth=2, markersize=8, color='blue')
    ax3.set_xlabel('Assistant Depth (Layers)', fontsize=12)
    ax3.set_ylabel('Softened Entropy (T=4.0)', fontsize=12)
    ax3.set_title(f'Entropy vs Depth\n{task_name.upper()} - {variant_name}', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(min(depths)-0.5, max(depths)+0.5)
    
    # Plot 4: ECE vs Depth
    ax4 = axes[1, 1]
    ax4.plot(depths, eces, 'd-', linewidth=2, markersize=8, color='red')
    ax4.set_xlabel('Assistant Depth (Layers)', fontsize=12)
    ax4.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
    ax4.set_title(f'Calibration Error vs Depth\n{task_name.upper()} - {variant_name}', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(min(depths)-0.5, max(depths)+0.5)
    
    plt.tight_layout()
    filename = f"{task_name}_{variant_name}_depth_analysis.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_subsampling_comparison(subsampling_results, task_name, save_dir='entropy_analysis'):
    """Compare original vs 3668 vs 2490 at fixed depth (3L)"""
    os.makedirs(save_dir, exist_ok=True)
    
    variants = ['original', 'subsampled_3668', 'subsampled_2490']
    variant_labels = ['Original', '3668 Samples', '2490 Samples']
    colors = ['blue', 'orange', 'red']
    
    # Extract metrics
    mccs = []
    accuracies = []
    entropies = []
    eces = []
    
    for variant in variants:
        if variant in subsampling_results and subsampling_results[variant]:
            student = subsampling_results[variant].get('student_summary')
            if student:
                mccs.append(student.get('mcc', 0.0) if student.get('mcc') is not None else 0.0)
                accuracies.append(student['accuracy'])
                entropies.append(student['mean_softened_entropy'])
                eces.append(student['calibration']['ece'])
            else:
                mccs.append(0.0)
                accuracies.append(0.0)
                entropies.append(0.0)
                eces.append(0.0)
        else:
            mccs.append(0.0)
            accuracies.append(0.0)
            entropies.append(0.0)
            eces.append(0.0)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x_pos = np.arange(len(variants))
    
    # Plot 1: MCC Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos, mccs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(variant_labels, rotation=15, ha='right')
    ax1.set_ylabel('MCC (Primary Metric)', fontsize=12)
    ax1.set_title(f'MCC: Subsampling Effect (Depth=3L)\n{task_name.upper()}', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars1, mccs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center', fontsize=9)
    
    # Plot 2: Accuracy Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(variant_labels, rotation=15, ha='right')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Accuracy: Subsampling Effect (Depth=3L)\n{task_name.upper()}', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center', fontsize=9)
    
    # Plot 3: Entropy Comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, entropies, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(variant_labels, rotation=15, ha='right')
    ax3.set_ylabel('Softened Entropy (T=4.0)', fontsize=12)
    ax3.set_title(f'Entropy: Subsampling Effect (Depth=3L)\n{task_name.upper()}', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, entropies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center', fontsize=9)
    
    # Plot 4: ECE Comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, eces, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(variant_labels, rotation=15, ha='right')
    ax4.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
    ax4.set_title(f'Calibration Error: Subsampling Effect (Depth=3L)\n{task_name.upper()}', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars4, eces):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    filename = f"{task_name}_subsampling_comparison.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_entropy_histogram(entropies_dict, task_name, variant_name, depth=None, save_dir='entropy_histograms'):
    """Plot entropy distribution histograms for comparison"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    colors = {'teacher': 'black', 'assistant': 'blue', 'no_distill': 'orange', 
              'direct_kd': 'green', 'hkd_student': 'red'}
    
    for model_name, entropies in entropies_dict.items():
        if entropies is not None and len(entropies) > 0:
            plt.hist(entropies, bins=50, alpha=0.5, label=model_name,
                    color=colors.get(model_name, 'gray'))
    
    depth_str = f" Depth={depth}L" if depth else ""
    plt.xlabel('Softened Predictive Entropy (T=4.0) - Auxiliary Diagnostic Metric', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Entropy Distribution (KD Temperature)\n{task_name.upper()} - {variant_name}{depth_str}\nNOTE: Not inference-time uncertainty - relative diagnostic only', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    depth_suffix = f"_depth{depth}" if depth else ""
    filename = f"{task_name}_{variant_name}{depth_suffix}_entropy_histogram.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def summarize_entropy_distribution(entropy_list, percentiles=[0, 25, 50, 75, 100]):
    """Summarize entropy distribution with percentiles instead of full list"""
    if not entropy_list:
        return None
    entropy_array = np.array(entropy_list)
    return {
        'min': float(np.min(entropy_array)),
        'max': float(np.max(entropy_array)),
        'mean': float(np.mean(entropy_array)),
        'std': float(np.std(entropy_array)),
        'percentiles': {f'p{p}': float(np.percentile(entropy_array, p)) for p in percentiles}
    }


def analyze_models(teacher, models_dict, dataloader, device, task_name, depth=3, suffix="", assistant_depth=None):
    """
    Comprehensive entropy and calibration analysis for multiple models.
    Includes teacher entropy, confidence, ECE.
    
    This is an AUXILIARY analysis supporting the main claim.
    """
    teacher.eval()
    for model in models_dict.values():
        if model is not None:
            model.eval()
    
    results = {
        'task': task_name,
        'depth': depth,
        'assistant_depth': assistant_depth,
        'student_depth': STUDENT_DEPTH,
        'suffix': suffix,
        'models': {name: defaultdict(list) for name in ['teacher'] + list(models_dict.keys())},
        'cross_model': defaultdict(list),
    }
    
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Analyzing {task_name}{suffix} depth={assistant_depth}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]
            
            labels = batch["labels"]
            all_labels.extend(labels.cpu().numpy())
            
            # Teacher outputs
            t_out = teacher(**inputs)
            t_logits = t_out.logits
            
            # Teacher metrics
            results['models']['teacher']['entropy'].append(compute_softened_entropy(t_logits))
            results['models']['teacher']['confidence'].append(compute_confidence(t_logits))
            results['models']['teacher']['predictions'].append(torch.argmax(t_logits, dim=-1))
            
            # Student outputs
            student_outputs = {}
            for model_name, model in models_dict.items():
                if model is None:
                    continue
                s_out = model(**inputs)
                s_logits = s_out.logits
                student_outputs[model_name] = s_logits
                
                results['models'][model_name]['entropy'].append(compute_softened_entropy(s_logits))
                results['models'][model_name]['confidence'].append(compute_confidence(s_logits))
                results['models'][model_name]['predictions'].append(torch.argmax(s_logits, dim=-1))
            
            # Cross-model KL divergences
            for model_name, s_logits in student_outputs.items():
                kl_key = f'kl_teacher_to_{model_name}'
                kl_per_sample = compute_teacher_student_kl(t_logits, s_logits)
                results['cross_model'][kl_key].append(kl_per_sample)
            
            # Assistant → HKD Student
            if 'assistant' in student_outputs and 'hkd_student' in student_outputs:
                kl_per_sample = compute_teacher_student_kl(student_outputs['assistant'], student_outputs['hkd_student'])
                results['cross_model']['kl_assistant_to_hkd'].append(kl_per_sample)
    
    all_labels_tensor = torch.tensor(all_labels, device=device)
    
    summaries = {}
    
    for model_name, model_data in results['models'].items():
        if not model_data['entropy']:
            summaries[model_name] = None
            continue
            
        for metric in ['entropy', 'confidence', 'predictions']:
            model_data[metric] = torch.cat(model_data[metric])
        
        predictions = model_data['predictions']
        accuracy = (predictions == all_labels_tensor).float()
        
        
        preds_cpu = predictions.detach().cpu().numpy()
        labels_cpu = all_labels_tensor.cpu().numpy()
        mcc = matthews_corrcoef(labels_cpu, preds_cpu)
        entropy_list = model_data['entropy'].cpu().numpy().tolist()
        
        summaries[model_name] = {
            'mean_softened_entropy': float(model_data['entropy'].mean()),
            'std_softened_entropy': float(model_data['entropy'].std()),
            'entropy_summary': summarize_entropy_distribution(entropy_list),
            'mean_confidence': float(model_data['confidence'].mean()),
            'std_confidence': float(model_data['confidence'].std()),
            'accuracy': float(accuracy.mean()),
            'mcc': float(mcc) if mcc is not None else None,
            'calibration': compute_calibration_metrics(
                model_data['confidence'].cpu().numpy(),
                accuracy.cpu().numpy()
            )
        }
    
    cross_summaries = {}
    for key, values_list in results['cross_model'].items():
        if values_list:
            all_kl_values = torch.cat(values_list, dim=0)
            cross_summaries[key] = {
                'mean': float(all_kl_values.mean()),
                'std': float(all_kl_values.std()),
                'distribution_summary': summarize_entropy_distribution(all_kl_values.cpu().numpy().tolist())
            }
    
    return summaries, cross_summaries


def load_teacher_model(task, device, suffix=""):
    """Load teacher model with variant suffix."""
    teacher = get_teacher_model(task, num_labels=2)
    teacher_path = os.path.join(Config.MODEL_SAVE_PATH, f"teacher_{task}{suffix}.pt")
    
    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  ✓ Teacher loaded from {teacher_path}")
        return teacher
    else:
        print(f"  ✗ Teacher not found: {teacher_path}")
        return None


def load_model_with_variants(task, device, depth=3, suffix=""):
    """Load all model variants for a given task and suffix."""
    models = {}
    trained_models_dir = os.path.join(Config.MODEL_SAVE_PATH, "trained_models")
    base_model_dir = Config.MODEL_SAVE_PATH
    
    # No Distillation Student
    no_distill_path = os.path.join(base_model_dir, f"student_no_distill_{task}{suffix}.pt")
    if os.path.exists(no_distill_path):
        student = get_student_model(num_labels=2)
        student.load_state_dict(torch.load(no_distill_path, map_location=device))
        student.to(device).eval()
        models['no_distill'] = student
        print(f"  ✓ No Distill loaded from {no_distill_path}")
    else:
        models['no_distill'] = None
    
    # Direct KD Student
    direct_kd_path = os.path.join(base_model_dir, f"student_kd_{task}{suffix}.pt")
    if os.path.exists(direct_kd_path):
        student = get_student_model(num_labels=2)
        student.load_state_dict(torch.load(direct_kd_path, map_location=device))
        student.to(device).eval()
        models['direct_kd'] = student
        print(f"  ✓ Direct KD loaded from {direct_kd_path}")
    else:
        models['direct_kd'] = None
    
    # Assistant Model
    assistant_path = os.path.join(trained_models_dir, f"{task}_assistant_{depth}L_seed42{suffix}.pt")
    if not os.path.exists(assistant_path):
        assistant_path = os.path.join(base_model_dir, f"m2_assistant_{task}_{depth}L{suffix}.pt")
    
    if os.path.exists(assistant_path):
        assistant = get_assistant_model(num_labels=2, num_layers=depth)
        assistant.load_state_dict(torch.load(assistant_path, map_location=device))
        assistant.to(device).eval()
        models['assistant'] = assistant
        print(f"  ✓ Assistant ({depth}L) loaded from {assistant_path}")
    else:
        models['assistant'] = None
    
    # HKD Student
    hkd_path = os.path.join(trained_models_dir, f"{task}_student_{depth}Lassistant_seed42{suffix}.pt")
    if not os.path.exists(hkd_path):
        hkd_path = os.path.join(base_model_dir, f"m2_hkd_{task}_{depth}L{suffix}.pt")
    if not os.path.exists(hkd_path):
        hkd_path = os.path.join(base_model_dir, f"student_distilled_{task}{suffix}.pt")
    
    if os.path.exists(hkd_path):
        student = get_student_model(num_labels=2)
        student.load_state_dict(torch.load(hkd_path, map_location=device))
        student.to(device).eval()
        models['hkd_student'] = student
        print(f"  ✓ HKD Student (from {depth}L assistant → {STUDENT_DEPTH}L student) loaded from {hkd_path}")
    else:
        models['hkd_student'] = None
    
    return models


def detect_inverted_u_peak(depth_to_metric, metric_name='mcc'):
    """Detect peak in inverted-U pattern using depth-based dictionary."""
    depths = sorted(depth_to_metric.keys())
    if len(depths) < 3:
        return None, False, 0.0
    
    metrics = [depth_to_metric[d] for d in depths]
    peak_idx = np.argmax(metrics)
    peak_depth = depths[peak_idx]
    is_inverted_u = (peak_idx > 0 and peak_idx < len(depths) - 1)
    
    left_edge = metrics[0]
    right_edge = metrics[-1]
    peak_value = metrics[peak_idx]
    avg_edge = (left_edge + right_edge) / 2
    improvement = (peak_value - avg_edge) / (avg_edge + 1e-8)
    
    return peak_depth, is_inverted_u, improvement


def main():
    device = Config.DEVICE
    
    print("\n" + "=" * 80)
    print("  AUXILIARY ANALYSIS: SOFTENED ENTROPY & CALIBRATION")
    print("  =================================================")
    print("=" * 80)
    
    
    # CoLA depths for inverted-U analysis 
    COLA_DEPTHS = [1, 3, 6, 10]
    FIXED_DEPTH = 3  # For subsampling comparison
    
    # Subsampling variants for CoLA comparison (suffixes for model loading and data preparation)
    COLA_SUBSAMPLING_VARIANTS = {
        '': 'original',
        '_3668': 'subsampled_3668',
        '_2490': 'subsampled_2490'
    }
    
    all_results = {}
    
    # For depth analysis storage 
    cola_depth_results = []
    
    # For subsampling comparison storage 
    cola_subsampling_results = {}
    
    # Process CoLA - Depth Analysis 
    print(f"\n{'#'*60}")
    print(f"  COLA DEPTH ANALYSIS (ORIGINAL DATASET ONLY)")
    print(f"  Analyzing depths: {COLA_DEPTHS} for inverted-U pattern")
    print(f"{'#'*60}")
    
    task = 'cola'
    variant_suffix = ''  # ORIGINAL dataset only
    variant_name = 'original'
    
    # Prepare data with full dataset
    target_data, _ = prepare_all_tasks([task], {})
    test_loader = target_data[task]['test']
    
    # Load Teacher
    teacher = load_teacher_model(task, device, variant_suffix)
    if teacher is not None:
        depth_results_list = []
        
        for depth in COLA_DEPTHS:
            print(f"\n  --- Analyzing Assistant Depth: {depth}L (Student is fixed {STUDENT_DEPTH}L) ---")
            
            models = load_model_with_variants(task, device, depth, variant_suffix)
            valid_models = {name: model for name, model in models.items() if model is not None}
            
            if not valid_models:
                print(f"  ✗ No valid models found for depth {depth}")
                continue
            
            summaries, cross_summaries = analyze_models(
                teacher, valid_models, test_loader, device, task, depth, variant_suffix, 
                assistant_depth=depth
            )
            
            task_key = f"{task}{variant_suffix}_depth{depth}"
            all_results[task_key] = {
                'task': task,
                'variant': variant_name,
                'analysis_type': 'depth_analysis',
                'assistant_depth': depth,
                'student_depth': STUDENT_DEPTH,
                'temperature': Config.TEMPERATURE,
                'teacher_summary': summaries.get('teacher'),
                'student_summaries': {k: v for k, v in summaries.items() if k != 'teacher'},
                'cross_model_kl': cross_summaries
            }
            
            depth_results_list.append({
                'depth': depth,
                'student_summary': summaries.get('hkd_student'),
                'teacher_summary': summaries.get('teacher')
            })
            
            student_summary = summaries.get('hkd_student')
            if student_summary:
                mcc_str = f"{student_summary['mcc']:.4f}" if student_summary.get('mcc') is not None else "N/A"
                print(f"    HKD Student ({depth}L assistant → {STUDENT_DEPTH}L student):")
                print(f"      MCC: {mcc_str}")
                print(f"      Accuracy: {student_summary['accuracy']:.4f}")
                print(f"      Softened Entropy: {student_summary['mean_softened_entropy']:.4f}")
                print(f"      ECE: {student_summary['calibration']['ece']:.4f}")
                
                if summaries.get('teacher'):
                    plot_reliability_diagram(
                        summaries['teacher']['calibration'],
                        task, variant_name, 'teacher', depth=depth
                    )
                plot_reliability_diagram(
                    student_summary['calibration'],
                    task, variant_name, 'hkd_student', depth=depth
                )
        
        cola_depth_results = depth_results_list
        plot_entropy_vs_depth(depth_results_list, task, variant_name)
        
        del teacher
        torch.cuda.empty_cache()
    
    # Process CoLA - Subsampling Comparison 
    print(f"\n{'#'*60}")
    print(f"  COLA SUBSAMPLING COMPARISON (Depth={FIXED_DEPTH}L ONLY)")
    print(f"  Comparing: ORIGINAL vs 3668 vs 2490 samples")
    print(f"{'#'*60}")
    
    task = 'cola'
    depth = FIXED_DEPTH
    
    for variant_suffix, variant_name in COLA_SUBSAMPLING_VARIANTS.items():
        print(f"\n  --- Analyzing: {variant_name} (depth={depth}L) ---")
        
        # Prepare data with appropriate subsampling
        if variant_suffix:
            subsample_size = int(variant_suffix.strip('_'))
            task_subsample_sizes = {'cola': subsample_size}
        else:
            task_subsample_sizes = {}
        
        target_data, _ = prepare_all_tasks([task], task_subsample_sizes)
        test_loader = target_data[task]['test']
        
        # Load Teacher
        teacher = load_teacher_model(task, device, variant_suffix)
        if teacher is None:
            print(f"  ✗ Teacher not found for {variant_name}")
            continue
        
        models = load_model_with_variants(task, device, depth, variant_suffix)
        valid_models = {name: model for name, model in models.items() if model is not None}
        
        if not valid_models:
            print(f"  ✗ No valid models found for {variant_name}")
            continue
        
        summaries, cross_summaries = analyze_models(
            teacher, valid_models, test_loader, device, task, depth, variant_suffix, 
            assistant_depth=depth
        )
        
        task_key = f"{task}{variant_suffix}_depth{depth}"
        all_results[task_key] = {
            'task': task,
            'variant': variant_name,
            'analysis_type': 'subsampling_comparison',
            'assistant_depth': depth,
            'student_depth': STUDENT_DEPTH,
            'temperature': Config.TEMPERATURE,
            'teacher_summary': summaries.get('teacher'),
            'student_summaries': {k: v for k, v in summaries.items() if k != 'teacher'},
            'cross_model_kl': cross_summaries
        }
        
        cola_subsampling_results[variant_name] = {
            'student_summary': summaries.get('hkd_student'),
            'teacher_summary': summaries.get('teacher')
        }
        
        student_summary = summaries.get('hkd_student')
        if student_summary:
            mcc_str = f"{student_summary['mcc']:.4f}" if student_summary.get('mcc') is not None else "N/A"
            print(f"    HKD Student ({depth}L assistant → {STUDENT_DEPTH}L student):")
            print(f"      MCC: {mcc_str}")
            print(f"      Accuracy: {student_summary['accuracy']:.4f}")
            print(f"      Softened Entropy: {student_summary['mean_softened_entropy']:.4f}")
            print(f"      ECE: {student_summary['calibration']['ece']:.4f}")
        
        del teacher
        torch.cuda.empty_cache()
    
    # Plot subsampling comparison
    if cola_subsampling_results:
        plot_subsampling_comparison(cola_subsampling_results, 'cola')
    
    # Process other tasks 
    OTHER_TASKS = ['mrpc', 'rte']
    FIXED_DEPTH = 3
    
    for task in OTHER_TASKS:
        print(f"\n{'#'*60}")
        print(f"  TASK: {task.upper()} - Depth={FIXED_DEPTH}L")
        print(f"{'#'*60}")
        
        target_data, _ = prepare_all_tasks([task], {})
        test_loader = target_data[task]['test']
        
        teacher = load_teacher_model(task, device, '')
        if teacher is None:
            print(f"  ✗ Teacher not found for {task}")
            continue
        
        models = load_model_with_variants(task, device, FIXED_DEPTH, '')
        valid_models = {name: model for name, model in models.items() if model is not None}
        
        if not valid_models:
            print(f"  ✗ No valid models found for {task}")
            continue
        
        summaries, cross_summaries = analyze_models(
            teacher, valid_models, test_loader, device, task, FIXED_DEPTH, '', 
            assistant_depth=FIXED_DEPTH
        )
        
        task_key = f"{task}_depth{FIXED_DEPTH}"
        all_results[task_key] = {
            'task': task,
            'variant': 'original',
            'analysis_type': 'standard',
            'assistant_depth': FIXED_DEPTH,
            'student_depth': STUDENT_DEPTH,
            'temperature': Config.TEMPERATURE,
            'teacher_summary': summaries.get('teacher'),
            'student_summaries': {k: v for k, v in summaries.items() if k != 'teacher'},
            'cross_model_kl': cross_summaries
        }
        
        student_summary = summaries.get('hkd_student')
        if student_summary:
            print(f"    HKD Student ({FIXED_DEPTH}L assistant → {STUDENT_DEPTH}L student):")
            print(f"      Accuracy: {student_summary['accuracy']:.4f}")
            print(f"      Softened Entropy: {student_summary['mean_softened_entropy']:.4f}")
            print(f"      ECE: {student_summary['calibration']['ece']:.4f}")
        
        del teacher
        torch.cuda.empty_cache()
    
    # Save results
    output_path = os.path.join(Config.RESULTS_PATH, "entropy_analysis_complete.json")
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    all_results_serializable = convert_to_serializable(all_results)
    with open(output_path, 'w') as f:
        json.dump(all_results_serializable, f, indent=2)
    
    # ================================================================
    # SUMMARY TABLES
    # ================================================================
    print(f"\n{'='*100}")
    print("  AUXILIARY ANALYSIS COMPLETED")
    print("  ============================")
    print("  This analysis provides supporting observations for M2.")
    print(f"  Main claim remains: Non-monotonic depth-performance relationship (inverted-U).")
    print(f"{'='*100}")
    
    # Table 1: CoLA Depth Analysis 
    if cola_depth_results:
        print(f"\n  {'='*60}")
        print("  TABLE 1: COLA INVERTED-U PATTERN (ORIGINAL DATASET)")
        print("  HKD Student Performance vs Assistant Depth")
        print("  Primary Metric: MCC")
        print(f"  {'='*60}")
        print(f"  {'Depth':<10} {'MCC':<12} {'Accuracy':<12} {'Soft Entropy':<16} {'ECE':<12}")
        print(f"  {'─'*65}")
        
        depth_to_mcc = {}
        for result in sorted(cola_depth_results, key=lambda x: x['depth']):
            depth = result['depth']
            student = result.get('student_summary')
            if student:
                mcc = student.get('mcc', 0.0) if student.get('mcc') is not None else 0.0
                depth_to_mcc[depth] = mcc
                print(f"  {depth}L{'':<7} {mcc:<12.4f} {student['accuracy']:<12.4f} "
                      f"{student['mean_softened_entropy']:<16.4f} {student['calibration']['ece']:<12.4f}")
        
        peak_depth, is_inverted_u, improvement = detect_inverted_u_peak(depth_to_mcc, 'mcc')
        print(f"\n  → Inverted-U pattern detection:")
        if is_inverted_u:
            print(f"    ✓ Peak detected at depth {peak_depth}L")
            print(f"    ✓ Improvement over edges: {improvement:.2%}")
        else:
            print(f"    ⚠ Clear inverted-U pattern not detected")
    
    # Table 2: CoLA Subsampling Comparison 
    if cola_subsampling_results:
        print(f"\n  {'='*60}")
        print(f"  TABLE 2: COLA SUBSAMPLING EFFECT (DEPTH={FIXED_DEPTH}L)")
        print("  Comparing ORIGINAL vs 3668 vs 2490 samples")
        print(f"  {'='*60}")
        print(f"  {'Variant':<20} {'MCC':<12} {'Accuracy':<12} {'Soft Entropy':<16} {'ECE':<12}")
        print(f"  {'─'*70}")
        
        for variant_name in ['original', 'subsampled_3668', 'subsampled_2490']:
            if variant_name in cola_subsampling_results:
                student = cola_subsampling_results[variant_name].get('student_summary')
                if student:
                    mcc = student.get('mcc', 0.0) if student.get('mcc') is not None else 0.0
                    print(f"  {variant_name:<20} {mcc:<12.4f} {student['accuracy']:<12.4f} "
                          f"{student['mean_softened_entropy']:<16.4f} {student['calibration']['ece']:<12.4f}")
    
    # Table 3: Other Tasks Summary
    print(f"\n  {'='*60}")
    print(f"  TABLE 3: OTHER TASKS (Depth={FIXED_DEPTH}L)")
    print(f"  {'='*60}")

    
    print(f"  {'Task':<10} {'MCC':<10} {'Accuracy':<12} {'Soft Entropy':<16} {'ECE':<12}")
    print(f"  {'─'*65}")

    for task in OTHER_TASKS:
        task_key = f"{task}_depth{FIXED_DEPTH}"
        if task_key in all_results:
            student_summaries = all_results[task_key].get('student_summaries', {})
            hkd_student = student_summaries.get('hkd_student')
            if hkd_student:
                mcc_val = hkd_student.get('mcc', 0.0) if hkd_student.get('mcc') is not None else 0.0
                print(f"  {task.upper():<10} {mcc_val:<10.4f} {hkd_student['accuracy']:<12.4f} "
                    f"{hkd_student['mean_softened_entropy']:<16.4f} {hkd_student['calibration']['ece']:<12.4f}")
    # Discussion/Limitation note
    print(f"\n{'='*100}")
    print("  DISCUSSION / LIMITATION")
    print("  =======================")
    print(f"\n{'='*100}")
    print(f"✓ Results saved to: {output_path}")
    print(f"✓ Reliability diagrams saved to: reliability_plots/")
    print(f"✓ Depth analysis plots saved to: entropy_analysis/")
    print(f"✓ Subsampling comparison plot saved to: entropy_analysis/cola_subsampling_comparison.png")


if __name__ == "__main__":
    main()