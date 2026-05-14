import time
import json
import bert_score
import numpy as np
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
import os
import unicodedata
import importlib
import sys as _sys

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# specific to METEOR Implementation
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

from multilingual_utils import (
    normalize_lang, rouge_preprocess, multilingual_tokenize,
    get_bertscore_lang, get_bertscore_model, get_sbert_model,
    get_moverscore_model, unicode_normalize, is_char_tokenize_lang,
)


class _MultilingualRougeTokenizer:
    """Custom tokenizer for RougeScorer supporting non-space-delimited languages.
    RougeScorer expects an object with a .tokenize(text) method, not a bare function."""
    def __init__(self, lang):
        self.lang = lang
    def tokenize(self, text):
        return multilingual_tokenize(text, self.lang)

def compute_rouge_score(metrics:list=['rougeL'], pred_col='pred', sub_metrics=['fmeasure'], ref_data=None, ans_idx:int=1, lang:str='en'):
    """
    Computes ROUGE scores between reference and candidate sentences.
    Supports multilingual evaluation with language based tokenization.

    Parameters:
        metrics (list): List of ROUGE metrics to compute (e.g., ['rouge1', 'rouge2','rougeL']).
        pred_col (str): Name of the prediction column (default: 'pred').
        sub_metrics (list): List of sub-metrics to extract (e.g., ['fmeasure']).
        ref_data (list): List of data samples, each containing answer and prediction.
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).
        lang (str): Language for evaluation (default: 'en').

    Returns:
        dict: Dictionary of ROUGE scores for each metric and sub-metric.
    """
    lang = normalize_lang(lang)
    ans, preds = [], []
    for data in ref_data:
        # index - ans_idx is the answer, last index is the prediction
        ans.append(str(data[ans_idx]))
        preds.append(data[-1])

    # Initialize ROUGE scorer
    # Disable stemmer for non-English
    tokenizer = _MultilingualRougeTokenizer(lang) if lang != 'en' else None
    use_stemmer = (lang == 'en')
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=use_stemmer, tokenizer=tokenizer)

    rouge_rslts = {metric: {sub_metric:[] for sub_metric in sub_metrics} for metric in metrics}

    for ref, cand in tqdm(zip(ans, preds), total=len(ans)):
        scores = scorer.score(ref, cand)
        for key, data in rouge_rslts.items():
            for metric in sub_metrics:
                if metric == 'fmeasure':
                    data[metric].append(scores[key].fmeasure)

    
    return rouge_rslts

def compute_bert_score(inp_data, pred_col='pred', ans_idx:int=1, lang:str='en'):
    """
    Computes BERTScore precision, recall, and F1 between reference and prediction strings.
    Uses language based BERT model.

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        pred_col (str): Name of the prediction column (default: 'pred').
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).
        lang (str): Language for evaluation.

    Returns:
        dict: Dictionary with BERTScore precision ('P'), recall ('R'), and F1 ('F1') lists.
    """
    lang = normalize_lang(lang)
    # Extract ans & pred
    # index-1 is the 'answer', last index is the prediction
    ans = [str(data[ans_idx]) for data in inp_data]
    pred = [str(data[-1]) for data in inp_data]
    
    # Language based model
    bertscore_lang = get_bertscore_lang(lang)
    bertscore_model = get_bertscore_model(lang)
    
    bert_p, bert_r, bert_f1 = bert_score.score(
        pred, ans, 
        lang=bertscore_lang,
        model_type=bertscore_model,
        verbose=True
    )
    
    bert_result = {'P':[], 'R':[], 'F1': []}
    for p,r,f1 in zip(bert_p, bert_r, bert_f1):
        bert_result['P'].append(p.item())
        bert_result['R'].append(r.item())
        bert_result['F1'].append(f1.item())

    return bert_result


class _IdentityStemmer:
    """Not performing any stemmer for non-English METEOR, returns tokens unchanged."""
    def stem(self, token: str) -> str:
        return token

def compute_meteor_score(inp_data, pred_col='pred', ans_idx:int=1, lang:str='en'):
    """
    Calculates the METEOR score between a reference and prediction text.
    For non-English, disables stemmer and WordNet.

    Args:
        inp_data (list): List of data samples, each containing answer and prediction.
        pred_col (str): Name of the prediction column (default: 'pred').
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).
        lang (str): Language for evaluation (default: 'en').

    Returns:
        dict: Dictionary with METEOR scores ('meteor').
    """
    lang = normalize_lang(lang)
    ans = [str(data[ans_idx]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    result = {'meteor':[]}
    
    if lang == 'en':
        # English - full METEOR with WordNet and stemmer
        for ref, cand in tqdm(zip(ans, preds), total=len(ans)):
            tokenized_reference = word_tokenize(ref)
            tokenized_hypothesis = word_tokenize(cand)
            result['meteor'].append(meteor_score([tokenized_reference], tokenized_hypothesis))
    else:
        # Non-English - character-level or space tokenization, no stemmer/WordNet
        for ref, cand in tqdm(zip(ans, preds), total=len(ans)):
            tokenized_reference = multilingual_tokenize(ref, lang)
            tokenized_hypothesis = multilingual_tokenize(cand, lang)
            try:
                score = meteor_score(
                    [tokenized_reference], tokenized_hypothesis,
                    stemmer=_IdentityStemmer(),
                    wordnet=None,
                    alpha=0.9, beta=3.0, gamma=0.5
                )
            except Exception:
                ref_set = set(tokenized_reference)
                hyp_set = set(tokenized_hypothesis)
                if len(ref_set) == 0:
                    score = 0.0
                else:
                    precision = len(ref_set & hyp_set) / len(hyp_set) if len(hyp_set) > 0 else 0.0
                    recall = len(ref_set & hyp_set) / len(ref_set)
                    if precision + recall == 0:
                        score = 0.0
                    else:
                        score = (10 * precision * recall) / (recall + 9 * precision)
                    
            result['meteor'].append(score)
    
    return result

def compute_exact_match(inp_data, pred_col='pred', ans_idx:int=1, lang:str='en'):
    """
    Computes the exact match(after lowercasing and unicode normalization) between reference and prediction strings.

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        pred_col (str): Name of the prediction column (default: 'pred').
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).
        lang (str): Language for evaluation.

    Returns:
        dict: Dictionary with exact match results ('exact_match'), 1 if exact match, else 0.
    """
    ans = [str(data[ans_idx]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    result = {'exact_match':[]}
    for ref, cand in tqdm(zip(ans, preds), total=len(ans)):
        # Unicode normalize for consistent comparison
        tokenized_reference = unicode_normalize(ref.lower() if not ref.isdigit() else ref)
        tokenized_hypothesis = unicode_normalize(cand.lower() if not cand.isdigit() else cand)
        result['exact_match'].append(int(tokenized_reference == tokenized_hypothesis))
    
    return result

def compute_sbert_score(inp_data, ans_idx:int=1, lang:str='en'):
    """
    Computes cosine similarity between sentence embeddings of reference and prediction strings using SBERT.
    Uses language based model.

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).
        lang (str): Language for evaluation.

    Returns:
        np.ndarray: Array of cosine similarity scores for each sample.
    """
    lang = normalize_lang(lang)
    ans = [str(data[ans_idx]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    # Language based SBERT model
    model_name = get_sbert_model(lang)
    model = SentenceTransformer(model_name)
    ans_embs = model.encode(ans)
    pred_embs = model.encode(preds)

    # Generate cosine-similarities
    sims = np.diagonal(cosine_similarity(ans_embs, pred_embs))

    return sims

def compute_bleurt_score(inp_data, ans_idx=1, checkpoint='BLEURT-20', lang:str='en'):
    """
    Computes BLEURT scores between reference and prediction strings using Google's learned metric.
    
    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).
        checkpoint (str): BLEURT checkpoint to use (default: 'BLEURT-20'). 
                         Options: 'BLEURT-20', 'BLEURT-20-D12', 'BLEURT-20-D6' (smaller, faster).
        lang (str): Language for evaluation.

    Returns:
        dict: Dictionary with 'scores' key containing list of BLEURT scores for each sample.
    """
    from evaluate import load
    
    ans = [str(data[ans_idx]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    # Initialize BLEURT scorer
    # BLEURT-20 is multilingual
    bleurt = load('bleurt', checkpoint)
    
    # Compute scores in batches for efficiency
    scores = []
    batch_size = 32  # Process in batches to avoid memory issues
    
    for i in tqdm(range(0, len(ans), batch_size), desc="Computing BLEURT"):
        batch_refs = ans[i:i+batch_size]
        batch_preds = preds[i:i+batch_size]
        batch_scores = bleurt.compute(predictions=batch_preds, references=batch_refs)
        scores.extend(batch_scores['scores'])
    
    return {'scores': scores}

def compute_moverscore(inp_data, ans_idx=1, model='bert-base-uncased', n_gram=2, device='cpu', lang:str='en'):
    """
    Computes MoverScore between reference and prediction strings using contextualized embeddings and Word Mover's Distance.
    Uses language based BERT model.
    
    Based on Zhao et al. (EMNLP 2019), recommended configuration for QA tasks:
    - model: 'bert-base-uncased' or BERT fine-tuned on MNLI for best correlation with human judgments
    - n_gram: 2 (bigrams) - captures phrase-level context and word order, shown to outperform unigrams
    - For QA evaluation, bigrams are particularly effective as they capture multi-word answer phrases

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).
        model (str): Model to use for embeddings (default: 'distilbert-base-uncased').
                    Recommended for QA: 'bert-base-uncased' (best quality, slower) or 
                    'distilbert-base-uncased' (faster, slightly lower quality).
                    Paper's best: BERT fine-tuned on MNLI dataset.
                    Other options: 'roberta-base', 'roberta-large', 'albert-base-v2'.
                    For multilingual: 'bert-base-multilingual-cased'.
        n_gram (int): N-gram level for matching (default: 2).
                     1 = unigrams (individual words)
                     2 = bigrams (word pairs, recommended for QA - captures phrases like "New York")
                     3 = trigrams (better for longer contexts)
        device (str): Device to use for computation (default: 'cuda').
        lang (str): Language for evaluation.

    Returns:
        dict: Dictionary with 'scores' key containing list of MoverScore values for each sample.
        
    References:
        Zhao et al. (2019). MoverScore: Text Generation Evaluating with Contextualized 
        Embeddings and Earth Mover Distance. EMNLP 2019.
        Paper findings: n_gram=2 with BERT-MNLI achieved highest correlation with human judgments.
    """
    lang = normalize_lang(lang)

    # Auto-select model for non-English
    if lang != 'en':
        model = get_moverscore_model(lang)

    current_model = os.environ.get('MOVERSCORE_MODEL', '')
    os.environ['MOVERSCORE_MODEL'] = model

    if 'moverscore_v2' in _sys.modules and current_model != model:
        # Model changed since last import, reload so the new env var is set
        importlib.reload(_sys.modules['moverscore_v2'])
    
    try:
        from moverscore_v2 import get_idf_dict, word_mover_score
    except ImportError:
        raise ImportError(
            "MoverScore not found. Please install it using:\n"
            "pip install -U git+https://github.com/AIPHES/emnlp19-moverscore.git"
        )

    # Apply CPU and numpy compatibility patches programmatically,
    # so users don't need to manually patch the installed package.
    import moverscore_v2 as _msv2
    import numpy as _np
    if not hasattr(_np, 'float'):
        _np.float = float  # fix np.float removed in NumPy 2.0
    # Ensure device is set correctly regardless of what the module loaded with
    _msv2.device = device
    
    ans = [str(data[ans_idx]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    # Compute IDF dictionary for references (used for weighting)
    idf_dict_ref = get_idf_dict(ans)
    idf_dict_hyp = get_idf_dict(preds)
    
    # Compute MoverScore
    # The function returns scores for each reference-hypothesis pair
    # n_gram=2 recommended by paper for better correlation with human judgments in QA tasks
    scores = word_mover_score(
        ans, 
        preds, 
        idf_dict_ref, 
        idf_dict_hyp,
        stop_words=[], 
        n_gram=n_gram, 
        remove_subwords=True,
        batch_size=32,
        device=device
    )
    
    return {'scores': scores}

def time_exec(start_time, end_time, title):
    """
    Prints the elapsed time between start_time and end_time with a custom title.

    Parameters:
        start_time (float): Start time in seconds (as returned by time.time()).
        end_time (float): End time in seconds (as returned by time.time()).
        title (str): Description of the timed operation.
    """
    elapsed_time = end_time - start_time
    print(f' > {title}: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

def read_json(inp_file):
    """
    Reads a JSON or JSONL file and returns its contents.

    Parameters:
        inp_file (str): Path to the input JSON or JSONL file.

    Returns:
        object: Parsed data from the file.
    """
    if inp_file[-1] == 'l':
        # if it a .jsonl file
        with open(inp_file,'r') as f:
            inp_data = [json.loads(line) for line in f]
    else:
        inp_data = json.load(open(inp_file))

    return inp_data