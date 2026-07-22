"""Stylometric feature extraction.

Two complementary feature families are combined (see ``build_vectorizer``):

1. Character n-gram TF-IDF (char_wb, 2-4) -- captures spelling habits,
   morphology, favoured letter combinations. This is the workhorse of modern
   authorship attribution and is robust even with a modest corpus.

2. Hand-crafted style scalars (``StyleScalars``) -- sentence/word length,
   punctuation rhythm, vocabulary richness and function-word rates. These are
   topic-independent, so they describe *how* someone writes rather than *what*
   about, which is exactly what we want when the impersonator changes subject.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

# A compact list of common English function words. Their relative rates are a
# classic, topic-independent authorship signal (Mosteller & Wallace, 1964).
FUNCTION_WORDS = [
    "the", "a", "an", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "to", "from",
    "in", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "should", "now", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "would", "could",
]

_PUNCT = [".", ",", "!", "?", ";", ":", "'", '"', "-", "(", ")"]
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+")


class StyleScalars(BaseEstimator, TransformerMixin):
    """Transform raw texts into a dense matrix of topic-independent style stats."""

    feature_names_ = (
        ["avg_word_len", "avg_sent_len", "type_token_ratio", "hapax_ratio",
         "uppercase_ratio", "digit_ratio", "whitespace_ratio", "chars_per_word"]
        + [f"punct_{p}" for p in _PUNCT]
        + [f"fw_{w}" for w in FUNCTION_WORDS]
    )

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X):
        rows = [self._vectorize(doc) for doc in X]
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)

    @staticmethod
    def _vectorize(text: str) -> list[float]:
        text = text or ""
        n_chars = max(len(text), 1)
        words = _WORD_RE.findall(text.lower())
        n_words = max(len(words), 1)
        sentences = [s for s in _SENT_RE.split(text) if s.strip()]
        n_sents = max(len(sentences), 1)

        word_lengths = [len(w) for w in words]
        unique = set(words)
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        hapax = sum(1 for c in counts.values() if c == 1)

        feats = [
            float(np.mean(word_lengths)) if word_lengths else 0.0,   # avg_word_len
            n_words / n_sents,                                       # avg_sent_len
            len(unique) / n_words,                                   # type_token_ratio
            hapax / n_words,                                         # hapax_ratio
            sum(1 for c in text if c.isupper()) / n_chars,           # uppercase_ratio
            sum(1 for c in text if c.isdigit()) / n_chars,           # digit_ratio
            sum(1 for c in text if c.isspace()) / n_chars,           # whitespace_ratio
            n_chars / n_words,                                       # chars_per_word
        ]
        # Punctuation rates per character.
        for p in _PUNCT:
            feats.append(text.count(p) / n_chars)
        # Function-word rates per word.
        for w in FUNCTION_WORDS:
            feats.append(counts.get(w, 0) / n_words)
        return feats


def build_vectorizer() -> Pipeline:
    """Return the full feature pipeline used to embed a document.

    Output rows are L2-normalizable sparse vectors combining char n-grams and
    scaled style scalars. The pipeline is fit on the target corpus during
    training and then reused unchanged at scoring time.
    """
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True,
        lowercase=True,
    )
    style = Pipeline([
        ("scalars", StyleScalars()),
        ("scale", MaxAbsScaler()),  # keep sparse-friendly, no mean-centering
    ])
    union = FeatureUnion([
        ("char", char_tfidf),
        ("style", style),
    ])
    return Pipeline([("features", union)])


def to_csr(matrix) -> csr_matrix:
    """Ensure a matrix is CSR for consistent cosine math downstream."""
    return csr_matrix(matrix)
