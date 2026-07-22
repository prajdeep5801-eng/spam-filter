"""Authorship-verification model.

Given a corpus written by ONE target person, decide how likely a new email was
written by that same person -- even from an unfamiliar address.

Two modes, chosen automatically:

* one-class (default): only the target's emails are available. We build a style
  centroid and measure cosine similarity of a new email to it. A leave-one-out
  pass over the corpus estimates the genuine similarity distribution, which
  calibrates a 0-1 confidence via a logistic squashing function.

* supervised: if you also supply impostor emails (other people's writing), we
  train a logistic-regression classifier on target-vs-impostor. This yields a
  genuine probability and a reportable cross-validated AUC.

The output ``likelihood`` in one-class mode is a calibrated confidence, not a
frequentist probability -- with no negatives there is no ground truth for
"not the author". Supply impostors for a true probability. Both modes expose
the same ``score()`` API so the Gmail runner does not care which is active.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize

from .features import build_vectorizer


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


class StylometryModel:
    def __init__(self):
        self.pipeline = None          # fitted feature pipeline
        self.mode = None              # "one_class" | "supervised"
        # one-class state
        self.centroid_ = None         # np.ndarray, L2-normalized
        self.calib_s0_ = None         # similarity mapped to likelihood 0.5
        self.calib_scale_ = None      # logistic steepness
        self.genuine_mu_ = None
        self.genuine_sigma_ = None
        # supervised state
        self.clf_ = None
        self.cv_auc_ = None
        # metadata
        self.n_target_ = 0
        self.n_impostor_ = 0

    # ------------------------------------------------------------------ fit
    def fit(self, target_texts: list[str], impostor_texts: list[str] | None = None,
            *, calib_z: float = 1.0) -> "StylometryModel":
        if len(target_texts) < 2:
            raise ValueError("Need at least 2 target documents to build a profile.")

        impostor_texts = impostor_texts or []
        self.n_target_ = len(target_texts)
        self.n_impostor_ = len(impostor_texts)

        self.pipeline = build_vectorizer()
        all_texts = target_texts + impostor_texts
        # Fit feature space on everything we have so impostor vocabulary is seen.
        self.pipeline.fit(all_texts)

        Xt = normalize(csr_matrix(self.pipeline.transform(target_texts)))

        if impostor_texts:
            self.mode = "supervised"
            Xi = normalize(csr_matrix(self.pipeline.transform(impostor_texts)))
            X = vstack([Xt, Xi])
            y = np.array([1] * len(target_texts) + [0] * len(impostor_texts))
            self.clf_ = LogisticRegression(
                max_iter=2000, class_weight="balanced", C=1.0)
            self.clf_.fit(X, y)
            # Cross-validated AUC as an honest quality read-out (if feasible).
            try:
                folds = min(5, len(target_texts), len(impostor_texts))
                if folds >= 2:
                    self.cv_auc_ = float(np.mean(cross_val_score(
                        LogisticRegression(max_iter=2000, class_weight="balanced"),
                        X, y, cv=folds, scoring="roc_auc")))
            except Exception:
                self.cv_auc_ = None
            # Keep a centroid too, for an interpretable similarity read-out.
            self._fit_centroid(Xt, calib_z)
        else:
            self.mode = "one_class"
            self._fit_centroid(Xt, calib_z)

        return self

    def _fit_centroid(self, Xt: csr_matrix, calib_z: float) -> None:
        centroid = np.asarray(Xt.mean(axis=0)).ravel()
        self.centroid_ = normalize(centroid.reshape(1, -1))[0]

        # Leave-one-out genuine similarity distribution.
        sims = []
        n = Xt.shape[0]
        total = np.asarray(Xt.sum(axis=0)).ravel()
        for i in range(n):
            row = np.asarray(Xt.getrow(i).todense()).ravel()
            loo = (total - row) / (n - 1)
            loo_n = normalize(loo.reshape(1, -1))[0]
            sims.append(float(np.dot(row, loo_n)))
        sims = np.array(sims)

        self.genuine_mu_ = float(sims.mean())
        self.genuine_sigma_ = float(max(sims.std(), 1e-3))
        # Map similarity -> confidence. s0 (=> 0.5) sits calib_z sigmas below the
        # genuine mean, so typical genuine mail scores comfortably above 0.5 and
        # stylistically distant mail falls below it.
        self.calib_s0_ = self.genuine_mu_ - calib_z * self.genuine_sigma_
        self.calib_scale_ = self.genuine_sigma_

    # ---------------------------------------------------------------- score
    def _similarity(self, text: str) -> float:
        vec = normalize(csr_matrix(self.pipeline.transform([text])))
        row = np.asarray(vec.todense()).ravel()
        return float(np.dot(row, self.centroid_))

    def score(self, text: str) -> dict:
        """Return a dict with ``likelihood`` (0-1) and supporting diagnostics."""
        sim = self._similarity(text)
        z = (sim - self.genuine_mu_) / self.genuine_sigma_

        if self.mode == "supervised":
            vec = normalize(csr_matrix(self.pipeline.transform([text])))
            proba = float(self.clf_.predict_proba(vec)[0, 1])
            likelihood = proba
        else:
            likelihood = float(_sigmoid((sim - self.calib_s0_) / self.calib_scale_))

        return {
            "likelihood": round(likelihood, 4),
            "similarity": round(sim, 4),
            "z_score": round(z, 3),
            "mode": self.mode,
        }

    # ------------------------------------------------------------- reporting
    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "n_target": self.n_target_,
            "n_impostor": self.n_impostor_,
            "genuine_similarity_mean": round(self.genuine_mu_, 4),
            "genuine_similarity_std": round(self.genuine_sigma_, 4),
            "calibration_s0_(likelihood=0.5)": round(self.calib_s0_, 4),
            "cv_auc": round(self.cv_auc_, 4) if self.cv_auc_ is not None else None,
        }
