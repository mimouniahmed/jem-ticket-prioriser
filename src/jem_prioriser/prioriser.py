"""L'outil de triage : featurizer + réseau entraîné -> backlog ordonné et bugs atypiques signalés.

Il est enregistré dans MLflow comme un modèle « pyfunc » : n'importe quel code peut le
recharger avec `mlflow.pyfunc.load_model("models:/jem-prioriser@champion")` et appeler
`.predict(dataframe_de_bugs)`, sans connaître PyTorch ni scikit-learn.
"""

from __future__ import annotations

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from .cleaning import CLASSES
from .features import BugFeaturizer
from .metrics import urgency
from .model import MLP, energy_from_logits


class Prioriser:
    def __init__(self, featurizer: BugFeaturizer, model: MLP, train_energies: np.ndarray, review_quantile: float = 0.95):
        self.featurizer = featurizer
        self.model = model.cpu().eval()
        self.train_energies = np.sort(train_energies)
        # Au-delà de cette énergie (95e percentile des bugs d'entraînement) : bug atypique -> revue humaine
        self.review_threshold = float(np.quantile(train_energies, review_quantile))

    @torch.no_grad()
    def predict_raw(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        logits = self.model(torch.from_numpy(self.featurizer.transform(df)))
        return torch.softmax(logits, 1).numpy(), energy_from_logits(logits).numpy()

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retourne les bugs triés du plus urgent au moins urgent, avec de quoi justifier le tri."""
        probs, energy = self.predict_raw(df)
        out = pd.DataFrame(index=df.index)
        if "id" in df:
            out["id"] = df["id"]
        out["summary"] = df["summary"]
        out["predicted"] = [CLASSES[i] for i in probs.argmax(1)]
        out["confidence"] = probs.max(1).round(3)
        out["urgency"] = urgency(probs).round(4)
        for i, c in enumerate(CLASSES):
            out[f"p_{c}"] = probs[:, i].round(3)
        out["energy"] = energy.round(3)
        out["energy_percentile"] = (np.searchsorted(self.train_energies, energy) / len(self.train_energies) * 100).round(1)
        out["needs_review"] = energy > self.review_threshold
        out = out.sort_values("urgency", ascending=False).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Prioriser":
        return joblib.load(path)


class PrioriserPyfunc(mlflow.pyfunc.PythonModel):
    """Adaptateur MLflow : charge le Prioriser sauvegardé et expose .predict(df)."""

    def load_context(self, context) -> None:
        self.prioriser = Prioriser.load(context.artifacts["prioriser"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return self.prioriser.score(model_input)
