# The Game of Keys and Queries: Parallelism and Cognitive Geometry in Chinese Regulated Verse

This repository accompanies the article "The Game of Keys and Queries: Parallelism and Cognitive Geometry in Chinese Regulated Verse" by Maciej Kurzynski, Xiaotong Xu, and Yu Feng.

## Abstract

Language models represent word meanings as vectors in a multidimensional space. Building on this property, this study offers a geometric perspective on parallelism in classical Chinese poetry, complementing traditional symbolic interpretations. To automatically detect parallelism in poetic verse, the authors trained a BERT-based classifier on a dataset of over 140,000 regulated poems (lüshi 律詩), achieving performance on par with state-of-the-art generative models such as GPT-4.1 and DeepSeek R1. Unlike general purpose models, the custom classifier offers unique insights into how poetic meaning is encoded geometrically. The analysis shows that parallel lines exhibit alignment in the model's attention patterns: the 'key' vectors of corresponding characters point in the same direction, while this alignment disappears in non-parallel lines. This finding is interpreted through Peter Gärdenfors's theory of cognitive semantics, which posits that humans make sense of the world by organizing experience into distinct conceptual regions. The authors argue that parallelism serves as a bridging mechanism that temporarily unites these disparate domains of meaning, suggesting a deeper, geometric order that underlies language itself.

## Quick Start

1. **Setup environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Prepare training data:**
```bash
# Standard dataset (no community filtering)
python prepare_data.py --input-dir data/poems --test-data data/test.json --output train.json

# Community-filtered dataset
python make_communities.py --input train.json --output char_communities.json
python prepare_data.py --input-dir data/poems --communities char_communities.json --test-data data/test.json --output train_filtered.json
```

3. **Train model:**
```bash
# Standard training
python finetune.py --input train.json --model SIKU-BERT/sikubert --output models/standard

# with community filtering
python finetune.py --input train_filtered.json --model SIKU-BERT/sikubert --output models/filtered
```

4. **Evaluate model:**
```bash
python evaluate.py --model models/standard/sikubert-parallelism-best --input test.json
```

5. **Classify couplets using external APIs:**
```bash
# Batch processing from JSON file
python classify_api.py --input data/test.json --endpoint https://api.deepseek.com --model deepseek-reasoner --api-key YOUR_API_KEY

# Single couplet classification
python classify_api.py --input "中岁历三台，旬月典邦政" --endpoint https://api.poe.com/v1 --model Claude-Opus-4.1 --api-key YOUR_API_KEY
```

## Repository Contents

- `prepare_data.py` - Extract couplets from poetry CSV files with optional test data filtering
- `make_communities.py` - Character community detection using network analysis
- `finetune.py` - Fine-tune BERT models for parallelism detection
- `evaluate.py` - Evaluate trained models
- `classify_api.py` - Classify couplets using external APIs (supports both batch JSON files and single couplet strings)
- `figures/` - Jupyter notebooks for generating paper figures

## Data Format

Training and test data use JSON format with parallel/non-parallel couplet pairs:
```json
[
  {
    "line1": "调笑辄酬答",
    "line2": "嘲谑无惭沮", 
    "label": 1
  }
]
```

Where `label` is 1 for parallel couplets, 0 for non-parallel.

## Citation

Kurzynski, M., Xu, X., & Feng, Y. (2025). The Game of Keys and Queries: Parallelism and Cognitive Geometry in Chinese Regulated Verse. *IJHAC: A Journal of Digital Humanities*, 19(2), 143-157. doi:10.3366/ijhac.2025.0355