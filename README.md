# Revisiting Hierarchical Knowledge Distillation: Depth Dynamics Across Data Regimes
  
*Izmir Institute of Technology, Spring 2026*

---

## Project Overview

This project investigates **hierarchical knowledge distillation (HKD)** for compressing BERT-base (109M parameters) into a compact 6-layer student model (67M parameters) across four GLUE benchmark tasks: **RTE**, **MRPC**, **CoLA**, and **SST-2**.

The core research question: *When do intermediate "assistant" models between teacher and student actually help?*

---

## Key Contributions

- Systematic depth ablation: 6 assistant depths (1L–10L) × 5 seeds × 4 tasks  
- Controlled data-scale experiments: CoLA (8.5K, 3.7K, 2.5K), MRPC (3.7K, 2.5K)  
- Representation analysis: CKA and SVCCA for understanding learned structure  
- Calibration analysis: Soft entropy, ECE, MCC across all configurations  
- Practical guidance: HKD helps when data > 3K and teacher-student gap > 0.20  

---

## Key Findings

| Finding | Evidence |
|--------|----------|
| HKD benefits are data-gated | U-curve collapses below ~3K samples |
| HKD benefits are gap-gated | Gains only when gap > 0.20 |
| Optimal assistant depth: 1–4L | Deeper assistants overfit or collapse |
| SST-2: HKD fails at all scales | Task too easy (gap < 0.13) |
| CKA is uniformly high | Representation similarity ≠ performance |

---

## Repository Structure

```text
.
├── src/
│   ├── config.py
│   ├── prepare_data.py
│   ├── train_baseline.py
│   ├── train_distill.py
│   ├── hierarchical_knowledge_distillation_all.py
│   ├── cka_svcca.py
│   ├── entropy_analysis.py
│   ├── evaluate.py
│   └── plot.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (6GB VRAM recommended)
- Git

---

## Installation

git clone https://github.com/sahende/revisiting-hierarchical-distillation.git
cd revisiting-hierarchical-distillation

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

---

## Running Experiments


Step 1:
python src/train_baseline.py

Step 2:
python src/train_distill.py

Step 3:
python src/hierarchical_knowledge_distillation_all.py

Step 4:
python src/cka_svcca.py

Step 5:
python src/entropy_analysis.py

Step 6:
python src/plot.py

---

## Experimental Design

Task: RTE, MRPC, CoLA, SST-2  
Depth: 1, 2, 4, 6, 8, 10  
Data: CoLA (8.5K / 3.7K / 2.5K), MRPC (3.7K / 2.5K)  
Seeds: 42–82  

---

## Hyperparameters

Teacher: BERT-base (12L)  
Student: 6-layer BERT  
Temperature: 4.0  
Alpha: 0.5  
LR: 5e-5 / 2.5e-5  
Epochs: 10  
Batch size: 16  

---

## Citation


```bibtex
@inproceedings{simsek2026hkdDepth,
  author    = {Şimşek, Şahende and Kaçıkan, Egemen and Onan, Aytuğ},
  title     = {Hierarchical Knowledge Distillation Under Data Scarcity: An Empirical Study of Depth Dynamics},
  booktitle = {2026 International Artificial Intelligence and Data Processing Symposium (IDAP)},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/IDAP71869.2026.11728118},
  url       = {https://doi.org/10.1109/IDAP71869.2026.11728118}
}
```

---

## Author

Şahende Şimşek  
Izmir Institute of Technology  
Spring 2026
