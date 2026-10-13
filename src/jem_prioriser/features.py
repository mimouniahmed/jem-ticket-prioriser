"""Représentation des bugs : du texte et des catégories vers des vecteurs de nombres.

Deux sorties :
- `transform_sparse` : TF-IDF (titre + description) + one-hot, matrice creuse de ~50 000 colonnes.
  Utilisée par la régression logistique de référence.
- `transform`        : version DENSE et CONTINUE de ~300 dimensions (SVD du TF-IDF = LSA,
  + one-hot), standardisée. C'est l'espace dans lequel travaille le JEM : l'échantillonnage
  par dynamique de Langevin dérive l'énergie par rapport à l'entrée, ce qui exige un espace
  continu et de dimension raisonnable.
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
        # Deux vocabulaires séparés : un mot dans le titre n'a pas le même poids que dans la description.
        self.tfidf_summary = TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True, strip_accents="unicode"
        )
        self.tfidf_description = TfidfVectorizer(
            ngram_range=(1, 2), min_df=5, max_features=30000, sublinear_tf=True, strip_accents="unicode"
        )
        # Les composants rares (< 20 bugs) sont regroupés en une catégorie « autre ».
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
        # `stop_words_` garde tous les termes écartés (des centaines de milliers) : inutile pour
        # transformer de nouveaux bugs, et ça alourdirait le modèle sauvegardé de plusieurs dizaines de Mo.
        for vectorizer in (self.tfidf_summary, self.tfidf_description):
            vectorizer.stop_words_ = None
        cat = self.onehot.fit_transform(df[CATEGORICAL])
        lsa = self.svd.fit_transform(text)
        # La matrice de projection (svd_dim x ~43 000 mots) fait 88 Mo en float64 : float32 suffit.
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
        """Part de la variance du TF-IDF conservée par la SVD."""
        return float(self.svd.explained_variance_ratio_.sum())
