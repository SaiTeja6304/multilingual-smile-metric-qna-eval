# Multilingual SMILE: A Composite Lexical-Semantic Metric for Question-Answering Evaluation


This repository provides a **Multilingual** implementation of **SMILE: Semantic Metric Integrating Lexical Exactness**, a novel metric for evaluating natural language generation.

## What is SMILE?
SMILE is a lightweight and reliable evaluation metric for textual and visual question answering tasks. Unlike traditional metrics like ROUGE, METEOR, and Exact Match that focus purely on lexical overlap, or embedding-based metrics like BERTScore that overlook lexical precision, SMILE strikes a balance by combining sentence-level semantics, keyword-level understanding, and exact lexical matching. This hybrid approach offers a more comprehensive and interpretable evaluation, aligning closely with human judgment while avoiding the cost, bias, and inconsistency often associated with LLM-based metrics.

### Multilingual Support
This fork extends SMILE to support native, robust evaluation across the following 8 languages:
- **`ar`** (Arabic)
- **`bn`** (Bengali)
- **`en`** (English)
- **`fi`** (Finnish)
- **`ja`** (Japanese)
- **`ko`** (Korean)
- **`ru`** (Russian)
- **`te`** (Telugu)

The codebase automatically selects the appropriate models (e.g. `paraphrase-multilingual-mpnet-base-v2`, `bert-base-multilingual-cased`, `BLEURT-20`), applies appropriate language-specific stemming and NLTK stopwords, and uses character-level tokenization for languages without space-delimited words (e.g. Japanese, Korean).

## Directory Structure

```
multilingual-smile-metric-qna-eval/
├── smile/
│   ├── __init__.py
│   └── smile.py                 # Core SMILE implementation
├── pyscripts/
│   ├── multilingual_utils.py    # Language configuration and utilities
│   ├── generate_scores.py       # Main scoring script for all metrics
│   ├── generate_syn_ans.py      # Synthetic answer generation
│   ├── eval_perf.py             # Correlation analysis and evaluation
│   ├── eval_gpt_perf.py         # GPT-based evaluation script
│   ├── utils.py                 # Utility functions
│   ├── conversations.py         # LLM conversation templates
│   └── view_results.py          # Results visualization
├── scripts/
│   ├── example_generate_scores.sh   # Example: Generate SMILE scores
│   ├── example_syn_ans.sh           # Example: Generate synthetic answers
│   └── example_gpt_eval.sh          # Example: Run GPT-based evaluation
├── datasets/
│   ├── full_set/                # Full evaluation datasets
│   │   └── syn_ans/
│   │       └── syn_model-llama-3.2-3b-instruct/
│   │           └── *.jsonl
│   ├── subset_200/              # 200-sample subsets for quick experiments
│   │   └── syn_ans/
│   │       └── syn_model-llama-3.2-3b-instruct/
│   │           └── *.jsonl
│   └── human_eval/              # Human evaluation annotations
│       └── reviewer_*.csv
├── sample_data/
│   ├── test_input.jsonl         # Sample multilingual LLM predictions
│   ├── test_ground_truth.jsonl  # Sample multilingual ground truth references
│   └── sample_input.json        # Sample input for quick testing
├── main.py                      # Main multilingual evaluation entry point
├── smile_sample_usage.py        # Quick-start sample script
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

Clone this repository and install the dependencies:

```bash
git clone git@github.com:SaiTeja6304/multilingual-smile-metric-qna-eval.git
cd multilingual-smile-metric-qna-eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, you can also install using pip:
```bash
pip install git+https://github.com/SaiTeja6304/multilingual-smile-metric-qna-eval.git
```

## Quick Run

Run the main evaluation script on the included sample data to quickly verify your setup:

```bash
python3 main.py --input sample_data/test_input.jsonl --ground-truth sample_data/test_ground_truth.jsonl --output sample_data/test_results.csv --metrics exact_match rouge --verbose
```

What this does:
- Loads the sample predictions and ground-truth answers spanning 8 languages.
- Groups the processing by language for efficient model loading.
- Computes `exact_match` and `rouge_l` scores.
- Saves the per-question results, per-language summaries, and overall summary to a CSV file.

## Usage

You can use SMILE natively from the command line.

### Input Data Format
The input data for the evaluation script should in in JSON or JSONL format. Each entry in the file should be a dictionary containing the following keys:
- **id** or **question_id**: A unique identifier for the question.
- **question**: The question text.
- **answer** or **answers**: The ground-truth answer(s) for the question. This can be a string or list of strings (for multiple references).
- **syn_ans**: Synthetic answers generated for the question against each answer(s). Not required in case `use_ans` flag is set.
- **pred** or **answer**: The predicted answer(s) for the question.
- **lang**: (Optional in `input`, required in `ground-truth`) The 2-letter ISO code for the language (e.g. `ar`, `ja`, `en`).

Example Ground-Truth JSONL line:
```json
{"question_id": "1", "question": "What is the capital of France?", "answers": ["Paris"], "lang": "en"}
```

Example Input Prediction JSONL line:
```json
{"question_id": "1", "question": "What is the capital of France?", "answer": "Paris is the capital of France."}
```



### Using `main.py`
The `main.py` script is the primary entry point for large-scale multilingual evaluation. It accepts both input predictions and ground-truth JSONL files, aligns them by `question_id`, groups by language for efficiency, computes all specified metrics using a max-over-references policy, and outputs a results file containing per-question metrics and an aggregate summary.

To compute SMILE and all other metrics and save the output as a CSV (or JSONL):
```bash
python3 main.py \
      --input path/to/input_answers.jsonl \
      --ground-truth path/to/ground_truth.jsonl \
      --output path/to/output_results.csv \
      --verbose
```

If you only want to compute specific metrics:
```bash
python3 main.py --input inputs.jsonl --ground-truth gt.jsonl --metrics smile rouge exact_match
```

## Supported Metrics

| Metric | Description |
|--------|-------------|
| **SMILE** | Our proposed composite lexical-semantic metric |
| **ROUGE-L** | Longest common subsequence F1 |
| **BERTScore** | Contextual embedding similarity |
| **METEOR** | Token-level matching with synonyms |
| **Exact Match** | Exact string matching |
| **sBERT** | Sentence-BERT cosine similarity |
| **BLEURT** | Learned evaluation metric |
| **MoverScore** | Earth mover's distance with BERT embeddings |


## Configuration Options

### Embedding Models
- **`ember-v1`**: Default embedding model for SMILE when `lang="en"`
- **`paraphrase-multilingual-mpnet-base-v2`**: Default embedding model for SMILE & sBERT when using non-English languages.
- **`bert-base-multilingual-cased`**: Default model for BERTScore & MoverScore when using non-English languages.


## Datasets

The sample datasets include synthetic answers generated using Llama-3.2-3B-Instruct for:

| Category | Datasets |
|----------|----------|
| **Language QA** | HotpotQA, MRQA, MuSiQue, NaturalQuestions, TriviaQA |
| **Image QA** | DocVQA, TextVQA, POPE |
| **Video QA** | TGIF, MSVD, MSRVTT |

## Notes

1. All paths in the scripts are relative and should work from the package root directory.
2. GPU is recommended for faster embedding generation.
3. For GPT-based evaluation, you need to provide your own OpenAI API key.
4. The `subset_200` contains 200 samples per dataset for faster experimentation.

## Troubleshooting

### MoverScore Compatibility Issues

If you're using MoverScore (`--eval_mode moverscore`) and encounter errors, you may need to patch the installed `moverscore_v2.py` file:

#### Issue 1: CUDA Device Error
```
AssertionError: Torch not compiled with CUDA enabled
```
**Cause:** The library hardcodes `device = 'cuda'`, failing on machines without CUDA (e.g., macOS).

#### Issue 2: NumPy `np.float` Deprecation
```
AttributeError: module 'numpy' has no attribute 'float'
```
**Cause:** `np.float` was deprecated in NumPy 1.20 and removed in NumPy 2.0.

#### Fix (one-liner)
Run this command to patch both issues:
```bash
sed -i '' -e "s/^device = 'cuda'$/device = 'cuda' if torch.cuda.is_available() else 'cpu'/" \
          -e 's/np\.float)/float)/g' \
    .venv/lib/python3.11/site-packages/moverscore_v2.py
```
> **Note:** Adjust the path based on your Python version and virtual environment location.

## Citation For Original SMILE Paper

If you use this code or the SMILE metric in your research, please cite:

```
@inproceedings{smile2025,
  title={SMILE: A Composite Lexical-Semantic Metric for Question-Answering Evaluation},
  author={...},
  booktitle={Proceedings of ARR 2025},
  year={2025},
  url={https://arxiv.org/abs/2406.XXXX}
}
```
## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. 

You are free to:
- **Share**: Copy and redistribute the material in any medium or format.
- **Adapt**: Remix, transform, and build upon the material.

Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial**: You may not use the material for commercial purposes.

For more details, see the [full license text](https://creativecommons.org/licenses/by-nc/4.0/).

> **Note**: This release is for research purposes only. This release should not be used to develop models that compete with OpenAI. This release should not be used to improve any other large language model (excluding Llama 2 or derivative works thereof).

## Contributors

- [Sai Teja Sunku]()
- [Ronak Wani]()


We welcome contributions! Please open an issue or pull request.

**For more details of the original SMILE paper, see the [paper on arXiv](https://arxiv.org/abs/2406.XXXX).**
