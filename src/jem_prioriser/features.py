"""Bug representation: from text and categories to numeric vectors.

Two outputs:
- `transform_sparse`: TF-IDF (summary + description) + one-hot, a sparse matrix of ~43,000
  columns, used by the logistic regression baseline.
- `transform`: a DENSE, CONTINUOUS, standardised version of ~400 dimensions (SVD of the TF-IDF,
  i.e. LSA, + one-hot). This is the space the JEM works in: Langevin sampling differentiates
  the energy with respect to the input, which requires a continuous space of reasonable size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["product", "component", "type", "op_sys"]


class BugFeaturizer:
    def __init__(self, svd_dim: int = 256, min_category_count: int = 20, seed: int = 0):
        # Separate vocabularies: a term in the summary does not weigh the same as in the description.
        self.tfidf_summary = TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True, strip_accents="unicode"
        )
        self.tfidf_description = TfidfVectorizer(
            ngram_range=(1, 2), min_df=5, max_features=30000, sublinear_tf=True, strip_accents="unicode"
        )
        # Rare components (< 20 bugs) are grouped into a single "infrequent" category.
        self.onehot = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=min_category_count, sparse_output=True
        )
        self.svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
        self.scaler = StandardScaler()

    def _text_matrix(self, df: pd.DataFrame, fit: bool) -> sp.csr_matrix:
        s = self.tfidf_summary.fit_transform(df["summary"]) if fit else self.tfidf_summary.transform(df["summary"])
        d = (
            self.tfidf_description.fit_transform(df["description"])
            if fit else self.tfidf_description.transform(df["description"])
        )
        return sp.hstack([s, d]).tocsr()

    def fit(self, df: pd.DataFrame) -> "BugFeaturizer":
        text = self._text_matrix(df, fit=True)
        # `stop_words_` keeps every discarded term (hundreds of thousands): useless for transforming
        # new bugs, and it would add tens of MB to the serialised model.
        for vectorizer in (self.tfidf_summary, self.tfidf_description):
            vectorizer.stop_words_ = None
        cat = self.onehot.fit_transform(df[CATEGORICAL])
        lsa = self.svd.fit_transform(text)
        # The projection matrix (svd_dim x ~43,000 terms) takes 88 MB in float64; float32 is enough.
        self.svd.components_ = self.svd.components_.astype(np.float32)
        self.scaler.fit(np.hstack([lsa, cat.toarray()]))
        return self

    def transform_sparse(self, df: pd.DataFrame) -> sp.csr_matrix:
        return sp.hstack([self._text_matrix(df, fit=False), self.onehot.transform(df[CATEGORICAL])]).tocsr()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        lsa = self.svd.transform(self._text_matrix(df, fit=False))
        cat = self.onehot.transform(df[CATEGORICAL]).toarray()
        return self.scaler.transform(np.hstack([lsa, cat])).astype(np.float32)

    @property
    def dim(self) -> int:
        return int(self.scaler.n_features_in_)

    def explained_variance(self) -> float:
        """Share of the TF-IDF variance retained by the SVD."""
        return float(self.svd.explained_variance_ratio_.sum())
