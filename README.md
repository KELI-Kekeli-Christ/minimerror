# Minimerror (perceptron) — mini bibliothèque Python

Ce dépôt contient une implémentation simple de **Minimerror** pour apprendre un perceptron, utilisable sur plusieurs formats de données (CSV / DataFrame / numpy / listes).

## Installation

Dépendances minimales :
- `numpy`

Optionnel (pour CSV / DataFrame) :
- `pandas`

## Usage

### 1) Entraîner sur numpy

```python
import numpy as np
from min_error import MinimErrorPerceptron

X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
y = np.array([1, 1, 1, -1], dtype=float)

clf = MinimErrorPerceptron(max_iter=200, eta=0.1, temp=1.0).fit(X, y)
print("w =", clf.w_)
print("pred =", clf.predict(X))
```

### 2) Entraîner depuis un CSV

```python
from min_error import fit_from_csv

# Exemple: colonne "label" contient la classe
model = fit_from_csv(
    "data.csv",
    label_col="label",
    max_iter=200,
    eta=0.1,
    temp=1.0,
)
```

### 3) Labels non numériques (mapping)

```python
from min_error import MinimErrorPerceptron

label_map = {"spam": 1.0, "ham": -1.0}
clf = MinimErrorPerceptron().fit("data.csv", label_col="target", label_map=label_map)
```

## Notes

- Labels recommandés : **-1 / +1**. Si vous avez **0/1**, la lib convertit automatiquement en **-1/+1**.
- Un biais est ajouté par défaut (`add_bias=True`) : le modèle apprend `intercept_` + `coef_`.

