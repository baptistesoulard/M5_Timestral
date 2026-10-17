# M5 × Timestral

Banc d'essai de **Timestral** (modèles de fondation de séries temporelles de Mistral, oct. 2026) sur le concours **M5 Walmart**, dans sa configuration officielle :

- 30 490 séries quotidiennes article × magasin (10 magasins, 3 États), agrégées en **42 840 séries sur 12 niveaux** ;
- historique d_1 → d_1941, **horizon 28 jours** (d_1942 → d_1969, période d'évaluation finale du concours) ;
- métriques officielles : **WRMSSE** (piste précision) et **WSPL** (piste incertitude, 9 quantiles).

Timestral est utilisé en **zero-shot** (aucun entraînement), sur CPU.

## Ce que le banc mesure

| Question | Comment la lire |
|---|---|
| Timestral fait-il mieux qu'un modèle statistique classique ? | WRMSSE face au naïf saisonnier, aux méthodes intermittentes (Croston, TSB…) et à AutoETS saisonnier, proche d'un Holt-Winters type SAP IBP |
| À quelle distance du top 10 du concours ? | Les prévisions des 10 premières équipes (LightGBM très travaillés) sont rescorées avec le même code |
| Les covariables aident-elles ? | `--covariates none` → `calendar` (SNAP, événements, jour de semaine) → `full` (+ prix) |
| Médiane ou espérance ? | Sur les articles intermittents la médiane vaut souvent 0, ce que la RMSSE pénalise : les deux variantes sont enregistrées |
| Où le modèle est-il bon ? | `results/by_level.csv` : agrégats lisses (niveaux 1-9) contre articles intermittents (niveaux 10-12) |
| Les quantiles sont-ils exploitables (stock de sécurité) ? | WSPL face à des quantiles empiriques par jour de semaine |

## Référence déjà calculée

Scores obtenus avec ce code sur la période d'évaluation finale (plus bas = meilleur). L'implémentation retrouve à l'identique le score officiel du vainqueur (0,520), du naïf saisonnier (0,847) et du naïf (1,752), ainsi que les poids officiels (`weights_evaluation.csv`).

| Méthode | WRMSSE |
|---|---|
| M5 #1 YJ_STU (vainqueur) | 0,5204 |
| M5 #2 Matthias | 0,5282 |
| M5 #10 DenisKokosinskiy_STU | 0,5475 |
| AutoETS saisonnier (7 jours) | 0,6726 |
| Naïf saisonnier 7 jours | 0,8470 |
| Moyenne mobile 28 jours | 0,9462 |
| TSB | 0,9536 |
| ADIDA | 0,9546 |
| Croston SBA | 0,9572 |
| iMAPA | 0,9595 |
| SES | 0,9615 |
| Naïf | 1,7520 |

Piste incertitude : quantiles empiriques par jour de semaine (26 semaines) → **WSPL 0,2428**.

AutoETS saisonnier, proche d'un Holt-Winters type SAP IBP, est la meilleure méthode statistique. Les méthodes « plates » (SES, Croston, moyenne mobile) font moins bien que le naïf saisonnier : en bottom-up, elles perdent la saisonnalité hebdomadaire, qui pèse lourd dans les niveaux agrégés.

## Site public (GitHub Pages, gratuit)

Le site public est statique : Timestral est exécuté une fois, gratuitement, sur les serveurs GitHub Actions
(`.github/workflows/site.yml`), puis le site affiche les prévisions précalculées. Il est publié sur la branche
`gh-pages` (adresse : `https://baptistesoulard.github.io/M5_Timestral/`).

- `bench/configs.json` : configurations calculées (taille du modèle, covariables, nombre de jobs parallèles).
- `bench/run.json` : `"mode": "probe"` pour un test rapide sur quelques centaines de séries, `"full"` pour le run complet et la publication ; `"only"` limite à certaines configurations, `"reuse_run_id"` réutilise les prévisions d'un run précédent.
- Chaque push modifiant ces fichiers relance le calcul. Le rapport (scores, durées, erreurs) est écrit sur la branche `bench-results`.
- Covariables testées : `none` (ventes seules), `calendar` (SNAP, événements, jour de semaine), `full` (+ prix relatif, niveaux article), `rich` (+ indice de prix, profondeur de remise, part d'articles en vente, jour du mois, jours avant / depuis un événement, fermeture de Noël, type d'événement, pour tous les niveaux).

Le même calcul peut tourner sur un PC : `python export_static.py forecast …` puis `python export_static.py build`, le site est alors dans `public/`.

## Appli de démo (locale)

`app.py` est une appli web (Gradio) bilingue, anglais et français (langue du navigateur par défaut, bascule en haut de page), qui fait tourner Timestral en direct. Les textes sont dans `app_texts.py`.

- **Explorer une série** : n'importe quelle série des 12 niveaux, prévue à la demande, avec bandes de quantiles, réel, naïf saisonnier, TSB, AutoETS et vainqueur du concours ; KPI et lecture « niveau de service » (stock dimensionné sur un quantile).
- **Mini-benchmark** : WRMSSE et WSPL niveau par niveau sur les 154 séries agrégées et un échantillon d'articles, en une à deux minutes ; classement complet en dessous.
- **Comprendre** : le concours M5, les modèles de fondation pour séries temporelles et Timestral.
- **Vos données** : un CSV (une colonne date + une colonne par série, ou `serie, date, valeur`), en prévision ou en backtest, avec export des quantiles.

```bash
pip install gradio plotly
python app.py                 # http://localhost:7860 (télécharge les données si besoin)
```

### Mise en ligne de l'appli interactive sur Google Cloud Run (gratuit pour une démo, payé à l'usage au-delà)

Le `Dockerfile` construit l'appli avec les données M5 et les poids Timestral 5M et 28M intégrés.
Dans la console Google Cloud : *Cloud Run* → *Déployer un conteneur* → *Déployer en continu depuis un dépôt*
(ce repo GitHub, type de build *Dockerfile*), région `europe-west1`, accès public, 2 vCPU, 4 Gio de mémoire,
0 instance minimum, 1 instance maximum, délai de requête 3 600 s, affinité de session activée.

Ou en ligne de commande, depuis le dossier du repo :

```bash
gcloud run deploy timestral-m5 --source . --region europe-west1 --allow-unauthenticated \
  --cpu 2 --memory 4Gi --min-instances 0 --max-instances 1 --timeout 3600 --session-affinity
```

### Mise en ligne sur Hugging Face Spaces (abonnement PRO requis)

```bash
python build_space.py         # assemble dist/space/
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli upload <compte>/timestral-m5 dist/space . --repo-type space
```

Ou depuis le site : *New Space* → SDK **Gradio** → matériel **CPU basic** → onglet *Files* → glisser le contenu de `dist/space/`. Le premier démarrage dure quelques minutes (installation, téléchargement des données M5 et des poids). Choisir un Space **privé** si la démo ne doit pas être publique.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Étapes

```bash
# 1. Données M5 complètes (copie publique Nixtla, test inclus) + soumissions du top 10
python download_data.py

# 2. Benchmarks statistiques (quelques secondes ; --ets ajoute AutoETS, ~1 h sur 2 cœurs)
python run_baselines.py --stats --ets

# 3. Timestral : estimer d'abord la durée sur votre machine
python run_timestral.py --model mistralai/Timestral-28M --dry-run 500

# 4. Runs Timestral (les poids se téléchargent une fois depuis Hugging Face)
python run_timestral.py --model mistralai/Timestral-28M --covariates none
python run_timestral.py --model mistralai/Timestral-28M --covariates calendar
python run_timestral.py --model mistralai/Timestral-28M --covariates full

# 5. Scores
python evaluate.py          # → results/scores.md, scores.csv, by_level.csv

# 6. Mettre à jour le classement affiché par l'appli
python build_app_assets.py
```

### Plan d'expériences conseillé

| # | Commande | Ce qu'on apprend |
|---|---|---|
| E1 | `--model mistralai/Timestral-28M --covariates none` | Niveau brut du modèle, ventes seules |
| E2 | `… --covariates calendar` | Apport des événements, SNAP et jour de semaine |
| E3 | `… --covariates full` | Apport du prix (niveaux article) |
| E4 | `--model mistralai/Timestral-209M` + meilleur réglage | Gain du grand modèle contre coût de calcul |
| E5 | `… --max-context 365` puis `112` | Sensibilité à un historique court (lancements, ruptures de série) |

Options utiles : `--levels 12` (article × magasin seulement, piste précision uniquement, plus rapide), `--no-symmetric` (désactive le double passage, deux fois plus rapide), `--threads N`, `--name` pour nommer le run.

## Choix de méthode

- **Prévision directe de chaque niveau** : Timestral prévoit les 42 840 séries. Pour le WRMSSE officiel, seule la prévision article × magasin est agrégée (bottom-up, règle du concours) ; le score « direct » est donné à titre d'information. Le WSPL utilise les quantiles directs, car des quantiles ne s'additionnent pas.
- **Zéros de lancement masqués** : avant la première vente, l'article n'est pas référencé ; ces jours sont passés en valeurs manquantes.
- **Covariables** connues sur l'horizon : SNAP (indicateur de l'État, ou part des États pour les séries multi-États), événement du jour et jour de semaine (catégoriels, encodés par Timestral), prix relatif au prix médian de la série (niveaux 10 à 12).
- **Quantiles M5** : Timestral produit 21 quantiles (0,01 → 0,99). Les niveaux M5 sont interpolés ; 0,005 et 0,995 sont extrapolés linéairement, ce qui pénalise légèrement les queues.
- **Espérance** : intégrale de la grille de quantiles.

## Limites à garder en tête

- **Contamination possible** : M5 figure dans des benchmarks publics (fev-bench notamment) sur lesquels Timestral a été évalué. Le pré-entraînement est majoritairement synthétique, mais un bon score n'est pas une preuve de généralisation.
- **Sell-out retail quotidien** : la demande M5 est mesurée en caisse et censurée par les ruptures en rayon. Elle diffère d'une demande B2B mensuelle en commandes.
- **Durée** : sur CPU, compter de quelques minutes (5M, 28M) à plus d'une heure (209M, toutes séries, covariables). L'encodage des covariables catégorielles ajoute environ 10 ms par série.

## Structure

```
app.py               appli de démo (Gradio)
build_app_assets.py  prévisions de référence et classement embarqués dans l'appli
build_space.py       dossier prêt à déposer sur Hugging Face Spaces
download_data.py     données M5 + soumissions de référence
run_baselines.py     benchmarks statistiques
run_timestral.py     prévisions Timestral
evaluate.py          scores WRMSSE / WSPL par niveau
m5bench/data.py      chargement, hiérarchie 12 niveaux, covariables
m5bench/metrics.py   WRMSSE, WSPL, interpolation des quantiles
m5bench/store.py     format des fichiers de prévision (forecasts/*.npz)
m5bench/timestral_runner.py  chargement de Timestral et prévision par lots
```

Sources : [Timestral (Mistral)](https://mistral.ai/news/timestral/) · [code d'inférence](https://github.com/mistralai/timestral) · [données et soumissions M5 (Nixtla)](https://github.com/Nixtla/m5-forecasts) · [concours M5 Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
