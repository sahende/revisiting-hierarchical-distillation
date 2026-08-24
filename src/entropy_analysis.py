"""
Entropy & Calibration Analysis for Knowledge Distillation
Compares Teacher vs No Distill vs Direct KD vs HKD Students across ALL depths

"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

from config import Config
from models import get_teacher_model, get_student_model
from prepare_data import prepare_all_tasks


# =========================
# CONSTANTS
# =========================
ALL_DEPTHS = [1, 2, 4, 6, 8, 10]

MODELS_DIR = Config.MODEL_SAVE_PATH
ALL_DATASET_DIR = os.path.join(MODELS_DIR, "all_dataset")
TEACHERS_DIR = os.path.join(MODELS_DIR, "teachers")
M2_MODELS_DIR = os.path.join(MODELS_DIR, "m2_models")


# =========================
# ASSISTANT/STUDENT MODEL LOADER
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


# =========================
# METRICS
# =========================
def compute_softened_entropy(logits, temperature=Config.TEMPERATURE, eps=1e-12):
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = torch.log(probs + eps)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def compute_confidence(logits):
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


def compute_calibration_metrics(confidences, accuracies, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if bin_lower == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


# =========================
# ANALYZE ALL MODELS
# =========================
def analyze_models(teacher, models_dict, dataloader, device, task_name):
    """Analyze teacher + all student models."""
    teacher.eval()
    for model in models_dict.values():
        if model is not None:
            model.eval()
    
    all_teacher_entropy = []
    all_teacher_conf = []
    all_teacher_preds = []
    all_labels = []
    
    model_data = {name: {'entropy': [], 'conf': [], 'preds': []} 
                  for name in models_dict.keys()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  {task_name}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]
            
            labels = batch["labels"]
            all_labels.extend(labels.cpu().numpy())
            
            # Teacher
            t_out = teacher(**inputs)
            all_teacher_entropy.append(compute_softened_entropy(t_out.logits))
            all_teacher_conf.append(compute_confidence(t_out.logits))
            all_teacher_preds.append(torch.argmax(t_out.logits, dim=-1))
            
            # Other models
            for name, model in models_dict.items():
                if model is None:
                    continue
                s_out = model(**inputs)
                model_data[name]['entropy'].append(compute_softened_entropy(s_out.logits))
                model_data[name]['conf'].append(compute_confidence(s_out.logits))
                model_data[name]['preds'].append(torch.argmax(s_out.logits, dim=-1))
    
    all_labels_tensor = torch.tensor(all_labels, device=device)
    
    # Teacher summary
    t_entropy = torch.cat(all_teacher_entropy)
    t_conf = torch.cat(all_teacher_conf)
    t_preds = torch.cat(all_teacher_preds)
    t_acc = (t_preds == all_labels_tensor).float()
    t_mcc = matthews_corrcoef(all_labels_tensor.cpu().numpy(), t_preds.cpu().numpy())
    
    teacher_summary = {
        'mean_softened_entropy': float(t_entropy.mean()),
        'accuracy': float(t_acc.mean()),
        'mcc': float(t_mcc),
        'calibration': {'ece': compute_calibration_metrics(t_conf.cpu().numpy(), t_acc.cpu().numpy())}
    }
    
    # Student summaries
    student_summaries = {}
    for name, data in model_data.items():
        if not data['entropy']:
            continue
        
        entropy = torch.cat(data['entropy'])
        conf = torch.cat(data['conf'])
        preds = torch.cat(data['preds'])
        acc = (preds == all_labels_tensor).float()
        mcc = matthews_corrcoef(all_labels_tensor.cpu().numpy(), preds.cpu().numpy())
        
        student_summaries[name] = {
            'mean_softened_entropy': float(entropy.mean()),
            'accuracy': float(acc.mean()),
            'mcc': float(mcc),
            'calibration': {'ece': compute_calibration_metrics(conf.cpu().numpy(), acc.cpu().numpy())}
        }
    
    return {'teacher': teacher_summary, 'students': student_summaries}


# =========================
# MODEL LOADING HELPERS
# =========================
def load_teacher(task_name, dataset_size_str, device):
    """Load teacher model."""
    teacher = get_teacher_model(task_name, num_labels=2)
    path = Config.get_model_path(task_name, "baselines", "teacher.pt", dataset_size=dataset_size_str)
    
    if os.path.exists(path):
        teacher.load_state_dict(torch.load(path, map_location=device))
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"    ✓ Teacher: {os.path.basename(path)}")
        return teacher
    print(f"    ✗ Teacher not found (size={dataset_size_str})")
    return None


def load_all_models(task_name, dataset_size_str, device):
    """
    Load all student models for ALL depths.
    """
    models = {}
    
    # HKD Students for all depths
    for depth in ALL_DEPTHS:
        hkd_path = Config.get_model_path(task_name, "hierarchical_kd", f"student_hkd_depth{depth}L.pt", dataset_size=dataset_size_str)
        if os.path.exists(hkd_path):
            student = get_student_model(num_labels=2)
            student.load_state_dict(torch.load(hkd_path, map_location=device))
            student.to(device).eval()
            models[f'HKD {depth}L ($\\rightarrow$6L)'] = student
            print(f"    ✓ HKD {depth}L Student: {os.path.basename(hkd_path)}")
        else:
            print(f"    ✗ HKD {depth}L Student NOT FOUND (task={task_name}, size={dataset_size_str})")
    
    # No Distill (6L, no KD)
    no_distill_path = Config.get_model_path(task_name, "baselines", "student_no_distill.pt", dataset_size=dataset_size_str)
    if os.path.exists(no_distill_path):
        student = get_student_model(num_labels=2)
        student.load_state_dict(torch.load(no_distill_path, map_location=device))
        student.to(device).eval()
        models['No Distill (6L)'] = student
        print(f"    ✓ No Distill: {os.path.basename(no_distill_path)}")
    else:
        print(f"    ✗ No Distill NOT FOUND (task={task_name}, size={dataset_size_str})")
    
    # Direct KD (12L→6L)
    direct_kd_path = Config.get_model_path(task_name, "hierarchical_kd", "direct_kd.pt", dataset_size=dataset_size_str)
    if not os.path.exists(direct_kd_path):
        direct_kd_path = Config.get_model_path(task_name, "standard_kd", "student_distilled.pt", dataset_size=dataset_size_str)
        
    if os.path.exists(direct_kd_path):
        student = get_student_model(num_labels=2)
        student.load_state_dict(torch.load(direct_kd_path, map_location=device))
        student.to(device).eval()
        models['Direct KD (6L)'] = student
        print(f"    ✓ Direct KD: {os.path.basename(direct_kd_path)}")
    else:
        print(f"    ✗ Direct KD NOT FOUND (task={task_name}, size={dataset_size_str})")
    
    return models


def run_config(task_name, variant_label, dataset_size_str, task_subsample_sizes, device, all_results):
    """Run analysis for one configuration."""
    print(f"\n{'─'*60}")
    print(f"  {task_name.upper()} - {variant_label}")
    print(f"{'─'*60}")
    
    target_data, _ = prepare_all_tasks([task_name], task_subsample_sizes)
    test_loader = target_data[task_name]['test']
    
    teacher = load_teacher(task_name, dataset_size_str, device)
    if teacher is None:
        return
    
    models = load_all_models(task_name, dataset_size_str, device)
    if not models:
        print(f"    ✗ No models found")
        return
    
    summaries = analyze_models(teacher, models, test_loader, device, task_name)
    
    key = f"{task_name}_{variant_label}"
    all_results[key] = summaries
    
    # Print summary
    t = summaries['teacher']
    print(f"    {'Teacher':<25} Acc={t['accuracy']:.4f} MCC={t['mcc']:.4f} "
          f"Entropy={t['mean_softened_entropy']:.4f} ECE={t['calibration']['ece']:.4f}")
    
    for name, s in summaries['students'].items():
        mcc_str = f"MCC={s['mcc']:.4f}" if s.get('mcc') is not None else ""
        print(f"    {name:<25} Acc={s['accuracy']:.4f} {mcc_str} "
              f"Entropy={s['mean_softened_entropy']:.4f} ECE={s['calibration']['ece']:.4f}")
    
    del teacher
    torch.cuda.empty_cache()


def main():
    device = Config.DEVICE
    
    print("\n" + "=" * 70)
    print("  ENTROPY & CALIBRATION ANALYSIS (All Depths)")
    print("=" * 70)
    
    all_results = {}
    
    # ================================================================
    # ALL CONFIGURATIONS
    # ================================================================
    configs = []
    for task in Config.TASKS:
        for size in Config.get_task_sizes(task):
            subsample = Config.size_to_subsample(task, size)
            label = str(size)
            configs.append((task, label, str(size), subsample))
    
    for task, variant, size_str, subsample in configs:
        run_config(task, variant, size_str, subsample, device, all_results)
    
    # ================================================================
    # SAVE JSON
    # ================================================================
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert_to_serializable(v) for v in obj]
        return obj
    
    output_path = os.path.join(Config.RESULTS_PATH, "entropy_analysis_all_depths.json")
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    print(f"\n✓ JSON saved to: {output_path}")


if __name__ == "__main__":
    main()
