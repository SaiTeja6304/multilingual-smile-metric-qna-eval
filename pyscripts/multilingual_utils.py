import re
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer


SUPPORTED_LANGUAGES = [
    "ar", "bn", "en", "fi",
    "ja", "ko", "ru", "te",
]

# Languages where words are NOT separated by spaces.
_CHAR_TOKENIZE_LANGS = {"ja", "ko"}

# NLTK supported stopwords corpus names
_NLTK_STOPWORD_LANGS = {
    "ar": "arabic",
    "en": "english",
    "fi": "finnish",
    "ru": "russian",
}

# NLTK supported Snowball stemmer language names
_SNOWBALL_LANGS = {
    "ar": "arabic",
    "fi": "finnish",
    "ru": "russian",
}

# Languages not supported by paraphrase-multilingual-mpnet-base-v2
_LABSE_LANGS = {"bn", "te"}

# Stopword lists for languages without NLTK support
_BENGALI_STOPWORDS = {
    "এবং", "এই", "একটি", "একজন", "এক", "কিন্তু", "কি", "কে", "করে",
    "করা", "করেন", "তা", "তার", "তিনি", "তো", "দিয়ে", "না", "নিয়ে",
    "পরে", "বলে", "বা", "হয়", "হয়ে", "হলে", "হলো", "আর", "আমি",
    "আপনি", "ও", "সে", "যে", "থেকে", "মধ্যে", "প্রতি", "জন্য",
}

_JAPANESE_STOPWORDS = {
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ",
    "さ", "ある", "いる", "も", "する", "から", "な", "こと", "として",
    "い", "や", "れる", "など", "なっ", "ない", "この", "ため", "その",
    "あっ", "よう", "また", "もの", "という", "あり", "まで", "られ",
    "なる", "へ", "か", "だ", "これ", "によって", "により", "おり",
    "より", "による", "ず", "なり", "られる", "において", "について",
}

_KOREAN_STOPWORDS = {
    "이", "그", "저", "것", "수", "등", "들", "및", "에", "의", "를",
    "으로", "에서", "와", "과", "는", "은", "가", "도", "로", "만",
    "하다", "있다", "되다", "이다", "않다", "없다", "같다", "때문",
    "그리고", "하지만", "그러나", "또는", "즉", "또한",
}

_TELUGU_STOPWORDS = {
    "మరియు", "ఈ", "ఒక", "కానీ", "ఏమి", "ఎవరు", "చేసి", "చేయు",
    "అది", "అతని", "ఆమె", "అయితే", "తో", "లేదు", "కోసం", "నుండి",
    "లో", "పై", "కు", "యొక్క", "ఆ", "ఇది", "వారు", "మీరు", "నేను",
    "అన్ని", "ఉంది", "ఉన్న", "అయిన", "గా", "వల్ల", "ద్వారా",
}


def normalize_lang(lang: str) -> str:
    """Normalize language name to lowercase form."""
    lang = lang.strip().lower()
    aliases = {
        "finnish": "fi", "finish": "fi", "fin": "fi", 
        "english": "en", "eng": "en",
        "arabic": "ar", "bengali": "bn", "japanese": "ja",
        "korean": "ko", "russian": "ru", "telugu": "te"
    }
    lang = aliases.get(lang, lang)
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: '{lang}'. Supported: {SUPPORTED_LANGUAGES}")
    return lang


def get_bertscore_lang(lang: str) -> str:
    """Return language code used by the bert-score library."""
    return normalize_lang(lang)


def get_bertscore_model(lang: str) -> str:
    """Return the appropriate model for BERTScore based on language."""
    lang = normalize_lang(lang)
    if lang == "en":
        return "roberta-large"
    return "bert-base-multilingual-cased"


def get_stopwords(lang: str) -> set:
    """Return a set of stopwords for the given language."""
    lang = normalize_lang(lang)
    if lang in _NLTK_STOPWORD_LANGS:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words(_NLTK_STOPWORD_LANGS[lang]))
    
    _custom = {
        "bn": _BENGALI_STOPWORDS,
        "ja": _JAPANESE_STOPWORDS,
        "ko": _KOREAN_STOPWORDS,
        "te": _TELUGU_STOPWORDS,
    }
    return _custom.get(lang, set())


def get_stemmer(lang: str):
    """
    Return a stemmer/lemmatizer instance for the given language.
    
    Returns:
        - WordNetLemmatizer for English
        - SnowballStemmer for Arabic, Finnish, Russian
        - None for Bengali, Japanese, Korean, Telugu
    """
    lang = normalize_lang(lang)
    if lang == "en":
        return WordNetLemmatizer()
    if lang in _SNOWBALL_LANGS:
        return SnowballStemmer(_SNOWBALL_LANGS[lang])
    return None


def is_char_tokenize_lang(lang: str) -> bool:
    """Whether this language needs character-level tokenization."""
    return normalize_lang(lang) in _CHAR_TOKENIZE_LANGS


def multilingual_tokenize(text: str, lang: str) -> list:
    """
    Tokenize text appropriately for the given language.
    - For space-delimited languages (English, Finnish, Russian), split on whitespace.
    - For remaining languages, character-level tokenization.

    Returns a list of tokens.
    """
    lang = normalize_lang(lang)
    text = unicode_normalize(text)
    if lang in _CHAR_TOKENIZE_LANGS:
        # Character-level - each character is a token
        return [ch for ch in text if not ch.isspace()]
    else:
        return text.split()


def rouge_preprocess(text: str, lang: str) -> str:
    """
    Pre-process text for ROUGE scoring.
    For character-level languages, insert spaces between every character.
    For space-delimited languages, return it in same way as given.
    """
    lang = normalize_lang(lang)
    text = unicode_normalize(text)
    if lang in _CHAR_TOKENIZE_LANGS:
        return " ".join(ch for ch in text if not ch.isspace())
    return text


def unicode_normalize(text: str) -> str:
    """Apply NFC unicode normalization."""
    return unicodedata.normalize("NFC", text)


def get_smile_emb_model(lang: str) -> str:
    """
    Return the embedding model name for SMILE.
    - English: ember-v1
    - Bengali, Telugu: LaBSE
    - Others: paraphrase-multilingual-mpnet-base-v2
    """
    lang = normalize_lang(lang)
    if lang == "en":
        return "ember-v1"
    if lang in _LABSE_LANGS:
        return "LaBSE"
    return "paraphrase-multilingual-mpnet-base-v2"


def get_sbert_model(lang: str) -> str:
    """
    Return the sBERT model name for cosine similarity.
    - English: all-roberta-large-v1
    - Bengali, Telugu: LaBSE
    - Others: paraphrase-multilingual-mpnet-base-v2
    """
    lang = normalize_lang(lang)
    if lang == "en":
        return "sentence-transformers/all-roberta-large-v1"
    if lang in _LABSE_LANGS:
        return "sentence-transformers/LaBSE"
    return "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def get_moverscore_model(lang: str) -> str:
    """
    Return the model name for MoverScore.
    - English: bert-base-uncased
    - Others: bert-base-multilingual-cased
    """
    lang = normalize_lang(lang)
    if lang == "en":
        return "bert-base-uncased"
    return "bert-base-multilingual-cased"
