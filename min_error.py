"""
min_error.py — mini bibliothèque "Minimerror" pour perceptron

Objectif
--------
Fournir une implémentation simple de l'algorithme Minimerror (apprentissage
de perceptron) utilisable sur différents formats :
- CSV (via pandas si disponible)
- pandas.DataFrame
- numpy.ndarray
- listes/tuples (dataset "à la main")

Exemples rapides
---------------
1) Données numpy (X: (n, d), y: (n,))

    import numpy as np
    from min_error import MinimErrorPerceptron

    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
    y = np.array([1, 1, 1, -1], dtype=float)
    clf = MinimErrorPerceptron(max_iter=200, eta=0.1, temp=1.0).fit(X, y)
    print(clf.coef_, clf.intercept_)
    print(clf.predict(X))

2) CSV avec colonne cible

    from min_error import fit_from_csv
    model = fit_from_csv("data.csv", label_col="label", max_iter=200, eta=0.1, temp=1.0)

Notes
-----
- Les labels attendus sont idéalement dans {-1, +1}. Si vos labels sont {0,1}
  ou strings, utilisez `label_map` dans `fit_from_csv` ou `MinimErrorPerceptron.fit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def stabilite(w: np.ndarray, x: np.ndarray, y: float) -> float:
    """
    Stabilité (marge signée) : gamma = y * <w, x>

    Paramètres
    ---------
    w : (d,) ndarray
    x : (d,) ndarray
    y : float (souvent -1/+1)
    """
    return float(y) * float(np.dot(w, x))


def minim_error(
    L_ens: Sequence[Tuple[ArrayLike, float]],
    w: ArrayLike,
    maxIter: int,
    eta: float,
    temp: float,
) -> np.ndarray:
    """
    Algorithme Minimerror pour l'apprentissage des perceptrons.

    Paramètres
    ---------
    L_ens : séquence de (x, y)
        x : vecteur de caractéristiques (d,)
        y : label (idéalement -1/+1)
    w : vecteur initial (d,)
    maxIter : nombre d'itérations (epochs)
    eta : taux d'apprentissage
    temp : température (beta = 1/temp)

    Retour
    ------
    w : ndarray (d,)
        poids appris
    """
    if temp == 0:
        raise ValueError("temp ne doit pas être 0 (beta = 1/temp).")

    X = [np.asarray(ex[0], dtype=float) for ex in L_ens]
    y = [float(ex[1]) for ex in L_ens]
    w = np.asarray(w, dtype=float).copy()
    beta = 1.0 / float(temp)

    for _ in range(int(maxIter)):
        for mu in range(len(X)):
            gamma = stabilite(w, X[mu], y[mu])
            arg = (beta * gamma) / 2.0
            grad_scalar = (beta / 4.0) * (1.0 - np.tanh(arg) ** 2)
            w = w + float(eta) * float(grad_scalar) * y[mu] * X[mu]

    return w


def _maybe_import_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None
    return pd


def _to_numpy_xy(
    data: Any,
    y: Optional[Any] = None,
    *,
    label_col: Optional[Union[int, str]] = None,
    feature_cols: Optional[Sequence[Union[int, str]]] = None,
    dropna: bool = True,
    label_map: Optional[Dict[Any, float]] = None,
    add_bias: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convertit plusieurs formats en (X, y) numpy.

    Formats supportés :
    - (X, y) numpy/array-likes
    - pandas.DataFrame (+ label_col)
    - chemin CSV (str/Path) (+ label_col)
    - liste de tuples (x, y)
    """
    pd = _maybe_import_pandas()

    # Cas 1: dataset en liste de (x, y)
    if y is None and isinstance(data, (list, tuple)) and len(data) > 0:
        first = data[0]
        if isinstance(first, (list, tuple)) and len(first) == 2 and not isinstance(first[0], (int, float)):
            X_list = [np.asarray(xi, dtype=float) for xi, _ in data]  # type: ignore[misc]
            y_list = [yi for _, yi in data]  # type: ignore[misc]
            X_arr = np.asarray(X_list, dtype=float)
            y_arr = np.asarray(y_list, dtype=object)
            return _finalize_xy(X_arr, y_arr, label_map=label_map, add_bias=add_bias)

    # Cas 2: fichier CSV
    if y is None and isinstance(data, (str, bytes)):
        if pd is None:
            raise ImportError("pandas est requis pour charger un CSV (pip install pandas).")
        df = pd.read_csv(data)  # type: ignore[arg-type]
        return _to_numpy_xy(
            df,
            y=None,
            label_col=label_col,
            feature_cols=feature_cols,
            dropna=dropna,
            label_map=label_map,
            add_bias=add_bias,
        )

    # Cas 3: pandas DataFrame
    if y is None and pd is not None and hasattr(data, "__class__") and data.__class__.__name__ == "DataFrame":
        if label_col is None:
            raise ValueError("Pour un DataFrame, `label_col` est obligatoire.")
        df = data
        if dropna:
            df = df.dropna()

        if feature_cols is None:
            if isinstance(label_col, int):
                cols = list(df.columns)
                label_name = cols[label_col]
            else:
                label_name = label_col
            feature_cols = [c for c in df.columns if c != label_name]
        X_arr = df.loc[:, list(feature_cols)].to_numpy(dtype=float)
        y_arr = df.loc[:, label_col].to_numpy(dtype=object)
        return _finalize_xy(X_arr, y_arr, label_map=label_map, add_bias=add_bias)

    # Cas 4: (X, y) séparés
    if y is not None:
        X_arr = np.asarray(data, dtype=float)
        y_arr = np.asarray(y, dtype=object)
        return _finalize_xy(X_arr, y_arr, label_map=label_map, add_bias=add_bias)

    # Cas 5: numpy avec dernière colonne = y (optionnel)
    if isinstance(data, np.ndarray):
        if label_col is None:
            raise ValueError("Si `data` est un ndarray unique, fournissez `label_col` ou passez (X, y).")
        X_arr = np.delete(data, label_col, axis=1).astype(float, copy=False)
        y_arr = data[:, label_col]
        return _finalize_xy(X_arr, y_arr, label_map=label_map, add_bias=add_bias)

    raise TypeError(
        "Format de données non supporté. Utilisez: (X,y), DataFrame+label_col, chemin CSV+label_col, "
        "ou liste de (x,y)."
    )


def _finalize_xy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    label_map: Optional[Dict[Any, float]],
    add_bias: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2:
        raise ValueError(f"X doit être 2D (n,d). Reçu shape={X.shape}.")
    if y.ndim != 1:
        y = y.reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Incohérence tailles: X a {X.shape[0]} lignes, y a {y.shape[0]} éléments.")

    y_num: np.ndarray
    if label_map is not None:
        try:
            y_num = np.asarray([label_map[yi] for yi in y], dtype=float)
        except KeyError as e:
            raise ValueError(f"label_map ne contient pas la clé {e}.") from e
    else:
        # essai conversion directe
        try:
            y_num = y.astype(float)
        except Exception:
            raise ValueError("Labels non numériques: fournissez `label_map` pour convertir vers -1/+1.")

    # Normalisation simple de labels fréquents {0,1} -> {-1,+1} si possible
    uniq = set(np.unique(y_num))
    if uniq == {0.0, 1.0}:
        y_num = (2.0 * y_num) - 1.0

    if add_bias:
        bias = np.ones((X.shape[0], 1), dtype=float)
        X = np.concatenate([bias, X], axis=1)

    return X.astype(float, copy=False), y_num.astype(float, copy=False)


@dataclass
class MinimErrorPerceptron:
    """
    Modèle perceptron entraîné par Minimerror.

    Paramètres
    ---------
    max_iter : int
        nombre d'epochs
    eta : float
        taux d'apprentissage
    temp : float
        température (beta = 1/temp)
    """

    max_iter: int = 200
    eta: float = 0.1
    temp: float = 1.0

    w_: Optional[np.ndarray] = None  # inclut le biais si add_bias=True

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        *,
        label_col: Optional[Union[int, str]] = None,
        feature_cols: Optional[Sequence[Union[int, str]]] = None,
        label_map: Optional[Dict[Any, float]] = None,
        add_bias: bool = True,
        w0: Optional[ArrayLike] = None,
        dropna: bool = True,
    ) -> "MinimErrorPerceptron":
        Xn, yn = _to_numpy_xy(
            X,
            y,
            label_col=label_col,
            feature_cols=feature_cols,
            dropna=dropna,
            label_map=label_map,
            add_bias=add_bias,
        )

        if w0 is None:
            w0_arr = np.zeros(Xn.shape[1], dtype=float)
        else:
            w0_arr = np.asarray(w0, dtype=float)
            if w0_arr.shape != (Xn.shape[1],):
                raise ValueError(f"w0 doit être de shape {(Xn.shape[1],)}. Reçu {w0_arr.shape}.")

        L_ens = [(Xn[i], float(yn[i])) for i in range(Xn.shape[0])]
        self.w_ = minim_error(L_ens, w0_arr, self.max_iter, self.eta, self.temp)
        return self

    def decision_function(self, X: Any, *, add_bias: bool = True) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Modèle non entraîné: appelez .fit() avant.")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if add_bias:
            bias = np.ones((X_arr.shape[0], 1), dtype=float)
            X_arr = np.concatenate([bias, X_arr], axis=1)
        if X_arr.shape[1] != self.w_.shape[0]:
            raise ValueError(f"Dimension invalide: X a {X_arr.shape[1]} colonnes, w a {self.w_.shape[0]}.")
        return X_arr @ self.w_

    def predict(self, X: Any, *, add_bias: bool = True) -> np.ndarray:
        scores = self.decision_function(X, add_bias=add_bias)
        pred = np.where(scores >= 0.0, 1.0, -1.0)
        return pred

    def score(self, X: Any, y: Any, *, add_bias: bool = True, label_map: Optional[Dict[Any, float]] = None) -> float:
        Xn, yn = _to_numpy_xy(X, y, label_map=label_map, add_bias=add_bias)
        # Xn inclut déjà le biais; decision_function ajouterait un biais une 2e fois, donc add_bias=False
        scores = (Xn @ self.w_) if self.w_ is not None else None
        if scores is None:
            raise RuntimeError("Modèle non entraîné: appelez .fit() avant.")
        pred = np.where(scores >= 0.0, 1.0, -1.0)
        return float(np.mean(pred == yn))

    @property
    def intercept_(self) -> float:
        if self.w_ is None:
            raise RuntimeError("Modèle non entraîné: appelez .fit() avant.")
        return float(self.w_[0])

    @property
    def coef_(self) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Modèle non entraîné: appelez .fit() avant.")
        return self.w_[1:].copy()


def fit_from_csv(
    csv_path: str,
    *,
    label_col: Union[int, str],
    feature_cols: Optional[Sequence[Union[int, str]]] = None,
    max_iter: int = 200,
    eta: float = 0.1,
    temp: float = 1.0,
    label_map: Optional[Dict[Any, float]] = None,
    add_bias: bool = True,
    dropna: bool = True,
) -> MinimErrorPerceptron:
    """
    Convenience: entraîne un modèle à partir d'un CSV.
    """
    model = MinimErrorPerceptron(max_iter=max_iter, eta=eta, temp=temp)
    return model.fit(
        csv_path,
        y=None,
        label_col=label_col,
        feature_cols=feature_cols,
        label_map=label_map,
        add_bias=add_bias,
        dropna=dropna,
    )


