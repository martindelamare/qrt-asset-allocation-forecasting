# QRT Asset Allocation Performance Forecasting

Prévoir si le rendement futur de chacune des 278 allocations systématiques sera
positif ([QRT × ENS Challenge Data](https://challengedata.ens.fr)). Métrique :
accuracy. Tout le travail tient dans un notebook :
[`qrt_equal3.ipynb`](qrt_equal3.ipynb).

## Résultats

Hors échantillon, sur un holdout de 504 dates mises de côté avant toute modélisation :

| Modèle | Accuracy |
| --- | ---: |
| Toujours « hausse » (baseline) | 50.6 % |
| Régression logistique | 52.3 % |
| **Ensemble equal3** | **52.4 %** |

- Contre la baseline : **+1.8 pt**, IC 95 % [+0.9 ; +2.8] (bootstrap par dates).
- Contre la logistique : +0.12 pt, IC [−0.10 ; +0.34]. Ce gain n'est **pas**
  significatif.

Le notebook recalcule aussi une validation croisée en 5 folds groupés par date
(527 073 lignes) :

| Modèle | Accuracy | Delta vs logistique | IC 95 % |
| --- | ---: | ---: | ---: |
| Logistique | 52.49 % | – | – |
| Ridge `asinh` | 52.49 % | 0.00 pt | [−0.10 ; +0.10] |
| LightGBM | 52.53 % | +0.04 pt | [−0.15 ; +0.23] |
| **equal3** | **52.65 %** | **+0.16 pt** | [+0.06 ; +0.26] |

## Méthode

- **Validation groupée par date (`TS`)**. Les allocations d'une même date partagent
  un facteur commun. Couper une date entre train et validation ferait donc fuiter
  ce facteur.
- **Design appris dans chaque fold**, avec deux blocs :
  - variables brutes imputées et standardisées, indicateurs de manquants,
    one-hot `GROUP`/`ALLOCATION` ;
  - 18 variables relatives à la date courante : rang, écart à la médiane et
    z-score robuste dans le batch.
- **Trois modèles sur ce même design** :
  - une régression logistique ridge ;
  - une ridge sur `asinh(target)`, qui exploite l'amplitude des rendements sans
    déplacer la frontière de signe ;
  - un LightGBM à hyperparamètres figés, peu corrélé aux deux autres.
- **Ensemble equal3** : moyenne des trois probabilités, seuil 0.5, aucun poids
  optimisé. Chaque modèle est entraîné sur trois sous-échantillons de 80 % des dates.

## Fuites évitées

- `ROW_ID` n'est jamais une variable. Un 1-NN sur `ROW_ID` atteint 100 %
  in-sample, par pure mémorisation.
- `TS` sert uniquement à grouper. Les dates anonymisées pourraient être
  partiellement remises dans l'ordre : la cible d'une date réapparaît dans
  `RET_1` de la date suivante, avec 88 % de concordance de signe. Ce lien est
  exclu, car l'exploiter serait une fuite et non une prévision.

## Reproduire

1. Télécharger les données du challenge. Elles ne sont pas incluses dans ce dépôt.
2. Placer les quatre CSV dans `data/raw/`, ou définir `QRT_DATA_DIR`.
3. Installer les dépendances : `pip install -r requirements.txt`.
4. Exécuter le notebook (environ 6 minutes). Il écrit `submission_equal3.csv`.

## Limites

- Les écarts entre modèles (0.1 à 0.2 pt) sont bien plus petits que le bruit du
  leaderboard (environ ±1.4 pt).
- Le gain de l'ensemble vient surtout des dates à peu d'allocations.
- Les variables relatives à la date supposent que toutes ses allocations sont
  connues au moment de prédire.
