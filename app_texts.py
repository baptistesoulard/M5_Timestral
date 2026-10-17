"""Interface texts of the demo app, in English and French."""

LANGS = [("English", "en"), ("Français", "fr")]

LEVEL_LABELS = {
    "en": {"Level1": "Total", "Level2": "State", "Level3": "Store", "Level4": "Category", "Level5": "Department",
           "Level6": "State × category", "Level7": "State × department", "Level8": "Store × category",
           "Level9": "Store × department", "Level10": "Item", "Level11": "Item × state", "Level12": "Item × store"},
    "fr": {"Level1": "Total", "Level2": "État", "Level3": "Magasin", "Level4": "Catégorie", "Level5": "Rayon",
           "Level6": "État × catégorie", "Level7": "État × rayon", "Level8": "Magasin × catégorie",
           "Level9": "Magasin × rayon", "Level10": "Article", "Level11": "Article × État",
           "Level12": "Article × magasin"},
}

METHOD_LABELS = {
    "en": {"snaive": "Seasonal naive", "tsb": "TSB (intermittent)", "ets": "Seasonal AutoETS",
           "winner": "M5 winner (YJ_STU)"},
    "fr": {"snaive": "Naïf saisonnier", "tsb": "TSB (intermittent)", "ets": "AutoETS saisonnier",
           "winner": "Vainqueur M5 (YJ_STU)"},
}

LEADERBOARD_NAMES = {
    "en": {"snaive": "Seasonal naive (7 days)", "naive": "Naive (last value)", "ma28": "28-day moving average",
           "ses": "Simple exponential smoothing", "sba": "Croston SBA", "tsb": "TSB", "adida": "ADIDA",
           "imapa": "iMAPA", "ets": "Seasonal AutoETS", "median": "median", "mean": "mean",
           "benchmark": "Statistical", "concours M5": "Competition (LightGBM…)", "timestral": "Timestral zero-shot"},
    "fr": {"snaive": "Naïf saisonnier 7 jours", "naive": "Naïf (veille)", "ma28": "Moyenne mobile 28 jours",
           "ses": "Lissage exponentiel simple", "sba": "Croston SBA", "tsb": "TSB", "adida": "ADIDA",
           "imapa": "iMAPA", "ets": "AutoETS saisonnier", "median": "médiane", "mean": "espérance",
           "benchmark": "Statistique", "concours M5": "Concours (LightGBM…)", "timestral": "Timestral zero-shot"},
}

FREQ_NAMES = {"en": {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly"},
              "fr": {"D": "quotidienne", "W": "hebdomadaire", "M": "mensuelle", "Q": "trimestrielle"}}

T = {
    "en": {
        # header
        "intro": """<div class="hero"><h1>Timestral × M5 Walmart</h1>
<p>Mistral's time-series foundation model, used without any training (zero-shot), against statistical methods
and the winner of the M5 forecasting competition: 42,840 daily Walmart sales series, 28-day horizon, official
test period May 23 – June 19, 2016.</p></div>""",
        "test_banner": 'Test mode: "{model}" model with random weights. Forecasts are meaningless; only the '
                       "interface is being tested.",
        "lang_label": "Language",
        # tabs
        "tab_explore": "Explore", "tab_bench": "Benchmark", "tab_data": "Your data", "tab_learn": "Learn",
        # explore
        "how_to_read": "How to read this screen",
        "how_to_read_body": """Pick a series on the left, or tap an example. Timestral sees the sales history up to
May 22, 2016 and forecasts the next 28 days; the black line is what Walmart actually sold.

- **Orange line and bands**: Timestral's median forecast, with the ranges where demand should fall 80% and 98% of the time.
- **Dotted lines**: other methods, from simple rules to the competition winner.
- **RMSSE**: forecast error relative to a naive forecast, the competition's metric. Lower is better; below 1 beats naive.
- **Supply chain view**: what happens if each day's stock is set at a quantile of the forecast.

The **Learn** tab explains the competition and the model in more detail.""",
        "examples": "**Examples**",
        "ex_total": "Walmart total", "ex_store": "Store CA_3 · Foods", "ex_fast": "Fast-moving item",
        "ex_inter": "Intermittent item",
        "level": "Aggregation level", "store": "Store", "state": "State", "category": "Category",
        "department": "Department", "item": "Item",
        "settings": "Model settings", "model_size": "Model size",
        "model_info": "5M and 28M: fast. 209M: most accurate, a few seconds.",
        "covariates": "Covariates known in advance",
        "cov_none": "Sales only", "cov_calendar": "+ calendar (events, SNAP, day of week)",
        "cov_full": "+ calendar and price",
        "cov_rich": "+ rich covariates (price index, discounts, assortment, event timing)",
        "ctx": "History given to the model (days)", "ctx_info": "Reduce to simulate a short history",
        "compare": "Compare with", "show_days": "History shown (days)",
        "go": "Forecast with Timestral", "plot_label": "28-day forecast",
        "table_label": "Errors over the test period",
        "svc_title": "Supply chain view: what service level do the quantiles deliver?",
        "svc_body": "Each day's stock is set at the chosen quantile of the Timestral forecast. Red bars are days "
                    "when demand exceeded that stock.",
        "svc_target": "Target service level (%)",
        # chart
        "history": "History", "band98": "Timestral · 98% interval", "band80": "Timestral · 80% interval",
        "median": "Timestral · median", "mean": "Timestral · mean", "actual_test": "Actual (test)",
        "forecast_start": "forecast start", "units_day": "Units sold / day", "total_series": "Walmart total (10 stores)",
        "actual_demand": "Actual demand", "stock_at": "Stock set at the {target}% quantile", "median_short": "Median",
        "units": "Units / day",
        # table
        "col_method": "Method", "col_bias": "Bias", "col_total": "28-day total", "actual": "Actual",
        # kpis
        "k_rmsse": "Timestral RMSSE", "k_lower": "lower is better", "k_best_stat": "best statistical method (RMSSE {v})",
        "k_vs_winner": "vs M5 winner", "k_band": "Actual inside 80% band", "k_band_v": "{n}/28 days",
        "k_band_sub": "≈ 22 days expected", "k_bias": "Bias over 28 days", "k_bias_sub": "median vs actual",
        "k_time": "Compute time",
        "reading": "**Reading.** On this series ({zs} zero-sales days over the last year), Timestral **{verdict}** "
                   "the best statistical method. Actual demand stayed inside the 80% band on **{n} of 28 days**. "
                   "On RMSSE, Timestral's {which} does better{extra}",
        "beats": "beats", "trails": "trails", "w_mean": "mean", "w_median": "median",
        "reading_extra": "; on an intermittent series the median is often 0 and underestimates demand.",
        "s_covered": "Days covered", "s_covered_sub": "target {t}% · {n}/28 days", "s_fill": "Fill rate (volume)",
        "s_fill_sub": "{v} units not covered", "s_surplus": "Cumulative surplus", "s_surplus_sub": "units above demand",
        "s_median": "With the median only", "s_median_sub": "days covered, for comparison",
        # benchmark
        "bench_intro": "Timestral forecasts live **all 154 aggregate series** (levels 1 to 9) and a **sample** of "
                       "the item levels. Scores are computed as in the competition, on the same series for every method.",
        "b_cov": "Covariates", "b_n": "Series drawn per item level (10, 11, 12)", "b_seed": "Random seed",
        "b_run": "Run the mini-benchmark", "b_loading": "Loading the model", "b_progress": "Timestral: {d}/{n} series",
        "b_level": "Level", "b_series": "Series", "b_wspl_t": "WSPL Timestral", "b_wspl_e": "WSPL empirical quantiles",
        "b_overall": "Overall score (mean of 12)", "b_yaxis": "Weighted RMSSE",
        "b_summary": "**{n} series forecast in {s} s.** Timestral overall score: **{t}**",
        "b_best_stat": "best statistical method ({name}): **{v}**", "b_winner": "competition winner: **{v}**",
        "b_quant": "Quantiles: Timestral WSPL **{t}** vs **{e}** for empirical day-of-week quantiles.",
        "b_note": "*Levels 1 to 9 are complete; levels 10 to 12 are estimated on the drawn sample. Timestral "
                  "forecasts each series directly; the other methods are aggregated from the item × store level "
                  "(competition rule).*",
        "lb_title": "### Full leaderboard (42,840 series, official WRMSSE)",
        "lb_note": "*Full Timestral runs appear here after `run_timestral.py`, `evaluate.py` and `build_app_assets.py`.*",
        "lb_type": "Type",
        # your data
        "data_intro": "Forecast your own series with Timestral. **Format**: a `date` column then one column per "
                      "series, or three columns `series, date, value`. Frequency is detected automatically (day, week, "
                      "month, quarter).\n\n⚠️ This is a public app: do not upload confidential data. Run the app "
                      "locally instead (`python app.py`) so your data stays on your machine.",
        "u_file": "CSV file", "u_example": "Load the example: monthly sales of the 7 M5 departments",
        "u_h": "Horizon (0 = automatic)", "u_h_info": "Automatic: 28 days, 13 weeks, 12 months or 4 quarters",
        "u_bt": "Backtest: hide the last periods and measure the error", "u_go": "Forecast",
        "u_series": "Series shown", "u_metrics": "Backtest error (WAPE = absolute error / actual volume)",
        "u_dl": "Forecasts (CSV: p10, p50, p90, mean)",
        "u_info": "**{n} series** · **{f}** frequency · {p} points · horizon **{h}** · {mode}",
        "u_bt_mode": "backtest on the last periods", "u_fc_mode": "forecast of the future",
        "u_hist": "History seen by the model", "u_hidden": "Actual (hidden from the model)", "u_band": "80% interval",
        "u_avg": "Average", "u_snaive": "Seasonal naive", "u_naive": "Naive", "col_series": "Series",
        # errors
        "e_select": "Incomplete selection: pick every element of the series.",
        "e_load": "Could not load {name}: {err}",
        "e_nodate": "No date column found. Expected format: a date column, then one column per series (or the "
                    "columns series, date, value).",
        "e_nofile": "Upload a CSV file first (or load the example).",
        "e_short": "History too short for this horizon.",
    },
    "fr": {
        "intro": """<div class="hero"><h1>Timestral × M5 Walmart</h1>
<p>Le modèle de fondation de séries temporelles de Mistral, utilisé sans entraînement (zero-shot), face aux
méthodes statistiques et au vainqueur du concours de prévision M5 : 42 840 séries de ventes quotidiennes Walmart,
horizon 28 jours, période de test officielle du 23 mai au 19 juin 2016.</p></div>""",
        "test_banner": "Mode test : modèle « {model} » à poids aléatoires. Les prévisions n'ont aucune valeur, "
                       "l'interface seule est testée.",
        "lang_label": "Langue",
        "tab_explore": "Explorer", "tab_bench": "Benchmark", "tab_data": "Vos données", "tab_learn": "Comprendre",
        "how_to_read": "Comment lire cet écran",
        "how_to_read_body": """Choisissez une série à gauche, ou un exemple. Timestral voit l'historique des ventes
jusqu'au 22 mai 2016 et prévoit les 28 jours suivants ; la courbe noire montre ce que Walmart a réellement vendu.

- **Courbe et bandes orange** : la prévision médiane de Timestral, et les plages où la demande devrait tomber 80 % et 98 % du temps.
- **Pointillés** : les autres méthodes, de la règle simple au vainqueur du concours.
- **RMSSE** : l'erreur de prévision rapportée à celle d'une prévision naïve, la métrique du concours. Plus bas = meilleur ; sous 1, on bat le naïf.
- **Lecture supply chain** : ce qui se passe si le stock de chaque jour est fixé sur un quantile de la prévision.

L'onglet **Comprendre** explique le concours et le modèle plus en détail.""",
        "examples": "**Exemples**",
        "ex_total": "Total Walmart", "ex_store": "Magasin CA_3 · Alimentaire", "ex_fast": "Article à forte rotation",
        "ex_inter": "Article intermittent",
        "level": "Niveau d'agrégation", "store": "Magasin", "state": "État", "category": "Catégorie",
        "department": "Rayon", "item": "Article",
        "settings": "Réglages du modèle", "model_size": "Taille du modèle",
        "model_info": "5M et 28M : rapides. 209M : le plus précis, quelques secondes.",
        "covariates": "Covariables connues à l'avance",
        "cov_none": "Ventes seules", "cov_calendar": "+ calendrier (événements, SNAP, jour de semaine)",
        "cov_full": "+ calendrier et prix",
        "cov_rich": "+ covariables enrichies (indice de prix, remises, assortiment, proximité des événements)",
        "ctx": "Historique fourni au modèle (jours)", "ctx_info": "Réduire pour simuler un historique court",
        "compare": "Comparer avec", "show_days": "Historique affiché (jours)",
        "go": "Prévoir avec Timestral", "plot_label": "Prévision à 28 jours",
        "table_label": "Erreurs sur la période de test",
        "svc_title": "Lecture supply chain : quel niveau de service avec les quantiles ?",
        "svc_body": "Le stock de chaque jour est fixé au quantile choisi de la prévision Timestral. Les barres rouges "
                    "sont les jours où la demande a dépassé ce stock.",
        "svc_target": "Niveau de service visé (%)",
        "history": "Historique", "band98": "Timestral · intervalle 98 %", "band80": "Timestral · intervalle 80 %",
        "median": "Timestral · médiane", "mean": "Timestral · espérance", "actual_test": "Réel (test)",
        "forecast_start": "début de la prévision", "units_day": "Unités vendues / jour",
        "total_series": "Total Walmart (10 magasins)",
        "actual_demand": "Demande réelle", "stock_at": "Stock dimensionné au quantile {target} %",
        "median_short": "Médiane", "units": "Unités / jour",
        "col_method": "Méthode", "col_bias": "Biais", "col_total": "Total 28 jours", "actual": "Réel",
        "k_rmsse": "RMSSE Timestral", "k_lower": "plus bas = meilleur",
        "k_best_stat": "meilleure méthode statistique (RMSSE {v})", "k_vs_winner": "vs vainqueur M5",
        "k_band": "Réel dans la bande 80 %", "k_band_v": "{n}/28 j", "k_band_sub": "attendu ≈ 22 jours sur 28",
        "k_bias": "Biais sur 28 jours", "k_bias_sub": "médiane vs réel", "k_time": "Temps de calcul",
        "reading": "**Lecture.** Sur cette série ({zs} de jours sans vente sur la dernière année), Timestral "
                   "**{verdict}** que la meilleure méthode statistique. Le réel est resté dans la bande 80 % "
                   "**{n} jours sur 28**. Pour la RMSSE, {which} de Timestral fait mieux{extra}",
        "beats": "fait mieux", "trails": "fait moins bien", "w_mean": "l'espérance", "w_median": "la médiane",
        "reading_extra": " : sur une série intermittente, la médiane vaut souvent 0 et sous-estime la demande.",
        "s_covered": "Jours couverts", "s_covered_sub": "objectif {t} % · {n}/28 jours",
        "s_fill": "Taux de service en volume", "s_fill_sub": "{v} unités non couvertes",
        "s_surplus": "Surplus cumulé", "s_surplus_sub": "unités au-delà de la demande",
        "s_median": "Avec la médiane seule", "s_median_sub": "jours couverts, pour comparaison",
        "bench_intro": "Timestral prévoit en direct **les 154 séries agrégées** (niveaux 1 à 9) et un **échantillon** "
                       "des niveaux article. Les scores sont calculés comme au concours, sur les mêmes séries pour "
                       "toutes les méthodes.",
        "b_cov": "Covariables", "b_n": "Séries tirées par niveau article (10, 11, 12)", "b_seed": "Graine du tirage",
        "b_run": "Lancer le mini-benchmark", "b_loading": "Chargement du modèle", "b_progress": "Timestral : {d}/{n} séries",
        "b_level": "Niveau", "b_series": "Séries", "b_wspl_t": "WSPL Timestral",
        "b_wspl_e": "WSPL quantiles empiriques", "b_overall": "Score global (moyenne des 12)",
        "b_yaxis": "RMSSE pondérée",
        "b_summary": "**{n} séries prévues en {s} s.** Score global Timestral : **{t}**",
        "b_best_stat": "meilleure méthode statistique ({name}) : **{v}**", "b_winner": "vainqueur du concours : **{v}**",
        "b_quant": "Quantiles : WSPL Timestral **{t}** contre **{e}** pour des quantiles empiriques par jour de semaine.",
        "b_note": "*Les niveaux 1 à 9 sont complets ; les niveaux 10 à 12 sont estimés sur l'échantillon tiré. "
                  "Timestral prévoit chaque série directement, les autres méthodes sont agrégées depuis le niveau "
                  "article × magasin (règle du concours).*",
        "lb_title": "### Classement complet (42 840 séries, WRMSSE officiel)",
        "lb_note": "*Les runs Timestral complets apparaissent ici après `run_timestral.py`, `evaluate.py` et "
                   "`build_app_assets.py`.*",
        "lb_type": "Type",
        "data_intro": "Prévoyez vos propres séries avec Timestral. **Format** : une colonne `date` puis une colonne "
                      "par série, ou bien trois colonnes `serie, date, valeur`. Fréquence détectée automatiquement "
                      "(jour, semaine, mois, trimestre).\n\n⚠️ Sur une appli publique, n'envoyez pas de données "
                      "confidentielles : lancez plutôt l'appli en local (`python app.py`), les données restent alors "
                      "sur votre poste.",
        "u_file": "Fichier CSV", "u_example": "Charger l'exemple : ventes mensuelles des 7 rayons M5",
        "u_h": "Horizon (0 = automatique)", "u_h_info": "Automatique : 28 jours, 13 semaines, 12 mois ou 4 trimestres",
        "u_bt": "Backtest : masquer les dernières périodes et mesurer l'erreur", "u_go": "Prévoir",
        "u_series": "Série affichée", "u_metrics": "Erreur en backtest (WAPE = erreur absolue / volume réel)",
        "u_dl": "Prévisions (CSV : p10, p50, p90, espérance)",
        "u_info": "**{n} série(s)** · fréquence **{f}** · {p} points · horizon **{h}** · {mode}",
        "u_bt_mode": "backtest sur les dernières périodes", "u_fc_mode": "prévision du futur",
        "u_hist": "Historique vu par le modèle", "u_hidden": "Réel (masqué au modèle)", "u_band": "Intervalle 80 %",
        "u_avg": "Moyenne", "u_snaive": "Naïf saisonnier", "u_naive": "Naïf", "col_series": "Série",
        "e_select": "Sélection incomplète : choisissez chaque élément de la série.",
        "e_load": "Impossible de charger {name} : {err}",
        "e_nodate": "Aucune colonne de dates trouvée. Format attendu : une colonne date, puis une colonne par série "
                    "(ou bien les colonnes serie, date, valeur).",
        "e_nofile": "Chargez d'abord un fichier CSV (ou l'exemple).",
        "e_short": "Historique trop court pour cet horizon.",
    },
}

PIPELINE = {
    "en": [("1", "History", "The series' past sales, up to 32,768 steps."),
           ("2", "Patches", "Values are cut into blocks of 32 time steps, the model's “words”."),
           ("3", "Transformer", "Attention across time, and across related series and covariates."),
           ("4", "21 quantiles", "A full range per day, from the 1% to the 99% quantile.")],
    "fr": [("1", "Historique", "Les ventes passées de la série, jusqu'à 32 768 pas de temps."),
           ("2", "Patchs", "Les valeurs sont découpées en blocs de 32 pas, les « mots » du modèle."),
           ("3", "Transformer", "Attention dans le temps, et entre séries liées et covariables."),
           ("4", "21 quantiles", "Une plage complète par jour, du quantile 1 % au quantile 99 %.")],
}

LEARN = {
    "en": """
## The M5 competition

The M competitions, run since 1982 by Spyros Makridakis, are the reference benchmark of the forecasting
community. The fifth edition, **M5**, ran on Kaggle in 2020 with real data from **Walmart**, the world's largest
retailer, and drew more than 5,000 teams.

**The task.** Forecast the daily unit sales of **3,049 products** in **10 stores** across California, Texas and
Wisconsin, for the next **28 days**. The history covers January 2011 to May 2016 (1,941 days), with selling
prices, calendar events (holidays, sports finals, religious festivals) and **SNAP** days, when US food-stamp
benefits are paid out and food sales rise.

**A real product hierarchy.** Items roll up into 7 departments and 3 categories (Foods, Hobbies, Household),
stores roll up into states. Each item × store series is also part of **11 aggregate views**, from the item's
total to the whole of Walmart: 42,840 series in all.
""",
    "en_levels_title": "The 12 levels",
    "en_after_levels": """
**Why it is hard.** At the bottom of the hierarchy most series are **intermittent**: many days with zero sales,
then a few units. Promotions, price changes and stock-outs blur the signal, and the history mixes products
that launch and disappear. This is very close to the daily reality of demand planning.

**How it is scored.** Two tracks ran in parallel:
- **Accuracy (WRMSSE)**: the error of each series is divided by the error of a naive “same as yesterday”
  forecast, then weighted by the series' dollar sales over the last 28 days. The 12 levels count equally.
- **Uncertainty (WSPL)**: the same idea for 9 quantiles, from 0.5% to 99.5%, which is what a service-level or
  safety-stock decision needs.

**What it taught.** For the first time in the M series, machine learning beat classical statistics clearly.
Almost every top team used **LightGBM** gradient-boosted trees trained on all series at once, with prices and
calendar features. The winner (WRMSSE 0.520) was about 22% more accurate than the best statistical benchmark,
exponential smoothing (0.671). These solutions took weeks of feature engineering and tuning on this one dataset.

## Foundation models for time series

A foundation model is a large neural network **pre-trained once** on a huge and varied collection of series,
then used as is on data it has never seen. This is the same idea as language models: instead of learning words,
the model learns what trends, seasonality, noise and breaks look like, across thousands of domains.

**What changes for a planner.**
- **Zero-shot**: no training per product or per dataset. Give it a history, get a forecast in milliseconds.
- **Probabilistic by design**: the output is a range of quantiles, not a single number, which maps directly to
  service levels and safety stock.
- **Cross-learning built in**: the model brings patterns learned elsewhere, which helps short histories and
  new products.

**Known players.** Amazon's Chronos, Google's TimesFM, Salesforce's Moirai, Nixtla's TimeGPT, and since
October 2026 Mistral's Timestral.

**Limits to keep in mind.** They do not know your business rules (allocations, constraints, promotions planned
by marketing) unless you pass them as covariates. Very sparse, lumpy demand remains hard. Public benchmark scores
can be flattered when test data overlaps with what the model saw during pre-training.

## Timestral

Released by **Mistral AI** on October 15, 2026, Timestral is a family of four models: **5M, 28M, 123M and 209M**
parameters. Most of its pre-training data is **synthetic**: series built from trends, seasonal cycles and noise,
combined into related groups.
""",
    "en_after_pipeline": """
**Key features.**
- **Long context**: up to 32,768 time steps of history, and up to 1,024 steps forecast in one pass.
- **Multivariate and covariates**: related series are forecast jointly; covariates known only in the past or
  also in the future (prices, events, calendar), including categorical ones, can be passed in.
- **CPU-friendly**: Mistral reports about 10 ms per forecast for the 5M model and 140 ms for the 209M model on a
  standard 8-thread server, with no GPU.
- **Open**: inference code under Apache 2.0, weights on Hugging Face under a modified MIT licence. Fine-tuning on
  your own data is offered through Mistral Forge.

**Reported results.** Mistral states that each size ranks first for its size on the three public forecasting
benchmarks (GIFT-Eval, TIME and fev-bench), and that the 28M model beats Amazon's Chronos-2, about four times
larger. These are Mistral's own measurements.

## What this demo tests

This app puts Timestral, **unchanged and untrained on M5**, in the exact setting of the competition: same history,
same 28-day test period, same metric. It answers three practical questions:
1. How does a zero-shot model compare with a seasonal exponential smoothing, close to what planning tools such as
   SAP IBP run by default?
2. How far is it from solutions hand-tuned for weeks on this dataset?
3. Are its quantiles usable to set stock for a target service level?

**Caveats.** M5 appears in public benchmarks on which Timestral was evaluated, so a good score does not prove it
generalises. M5 measures point-of-sale sales, censored by stock-outs, not B2B orders. The 0.5% and 99.5%
quantiles are extrapolated from Timestral's 1%–99% grid.

**Sources.** [Timestral announcement (Mistral)](https://mistral.ai/news/timestral/) ·
[Timestral code](https://github.com/mistralai/timestral) ·
[M5 competition (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy) ·
[M5 data and top submissions (Nixtla)](https://github.com/Nixtla/m5-forecasts) ·
[Source code of this demo](https://github.com/baptistesoulard/M5_Timestral)
""",
    "fr": """
## Le concours M5

Les concours M, organisés depuis 1982 par Spyros Makridakis, sont la référence de la communauté de la
prévision. La cinquième édition, **M5**, s'est tenue sur Kaggle en 2020 avec des données réelles de **Walmart**,
premier distributeur mondial, et a réuni plus de 5 000 équipes.

**L'exercice.** Prévoir les ventes quotidiennes en unités de **3 049 produits** dans **10 magasins** de
Californie, du Texas et du Wisconsin, sur les **28 jours** suivants. L'historique couvre janvier 2011 à mai 2016
(1 941 jours), avec les prix de vente, les événements du calendrier (fêtes, finales sportives, fêtes religieuses)
et les jours **SNAP**, où les aides alimentaires américaines sont versées et les ventes alimentaires montent.

**Une vraie hiérarchie produit.** Les articles se regroupent en 7 rayons et 3 catégories (Alimentaire, Loisirs,
Maison), les magasins en États. Chaque série article × magasin entre aussi dans **11 vues agrégées**, du total de
l'article à l'ensemble de Walmart : 42 840 séries au total.
""",
    "fr_levels_title": "Les 12 niveaux",
    "fr_after_levels": """
**Pourquoi c'est difficile.** En bas de la hiérarchie, la plupart des séries sont **intermittentes** : beaucoup
de jours sans vente, puis quelques unités. Promotions, changements de prix et ruptures brouillent le signal, et
l'historique mêle des produits qui arrivent et disparaissent. C'est très proche du quotidien d'un demand planner.

**Comment c'est noté.** Deux pistes en parallèle :
- **Précision (WRMSSE)** : l'erreur de chaque série est divisée par celle d'une prévision naïve « comme hier »,
  puis pondérée par le chiffre d'affaires de la série sur les 28 derniers jours. Les 12 niveaux comptent autant.
- **Incertitude (WSPL)** : le même principe pour 9 quantiles, de 0,5 % à 99,5 %, ce dont a besoin une décision de
  niveau de service ou de stock de sécurité.

**Ce qu'on en a appris.** Pour la première fois dans la série M, le machine learning a nettement battu les
statistiques classiques. Presque toutes les équipes de tête ont utilisé des arbres de gradient boosting
**LightGBM** entraînés sur toutes les séries à la fois, avec prix et variables calendaires. Le vainqueur
(WRMSSE 0,520) était environ 22 % plus précis que le meilleur benchmark statistique, le lissage exponentiel (0,671).
Ces solutions ont demandé des semaines de construction de variables et de réglages sur ce seul jeu de données.

## Les modèles de fondation pour séries temporelles

Un modèle de fondation est un grand réseau de neurones **pré-entraîné une fois** sur une collection immense et
variée de séries, puis utilisé tel quel sur des données qu'il n'a jamais vues. C'est l'idée des modèles de
langage : au lieu d'apprendre des mots, le modèle apprend à quoi ressemblent tendances, saisonnalités, bruit et
ruptures, dans des milliers de domaines.

**Ce que ça change pour un planificateur.**
- **Zero-shot** : aucun entraînement par produit ou par jeu de données. On donne un historique, on obtient une
  prévision en quelques millisecondes.
- **Probabiliste par construction** : la sortie est une plage de quantiles, pas un chiffre unique, ce qui se
  traduit directement en niveau de service et stock de sécurité.
- **Apprentissage croisé intégré** : le modèle apporte des motifs appris ailleurs, utile pour les historiques
  courts et les nouveaux produits.

**Les acteurs connus.** Chronos d'Amazon, TimesFM de Google, Moirai de Salesforce, TimeGPT de Nixtla, et depuis
octobre 2026 Timestral de Mistral.

**Les limites à garder en tête.** Ils ignorent vos règles métier (allocations, contraintes, promotions prévues
par le marketing) si on ne les leur passe pas en covariables. Une demande très clairsemée et irrégulière reste
difficile. Les scores sur benchmarks publics peuvent être flattés quand les données de test recoupent ce que le
modèle a vu au pré-entraînement.

## Timestral

Publié par **Mistral AI** le 15 octobre 2026, Timestral est une famille de quatre modèles : **5M, 28M, 123M et
209M** paramètres. L'essentiel de ses données de pré-entraînement est **synthétique** : des séries construites à
partir de tendances, de cycles saisonniers et de bruit, combinées en groupes liés.
""",
    "fr_after_pipeline": """
**Caractéristiques clés.**
- **Long contexte** : jusqu'à 32 768 pas d'historique, et jusqu'à 1 024 pas prévus en un seul passage.
- **Multivarié et covariables** : les séries liées sont prévues ensemble ; on peut fournir des covariables
  connues seulement dans le passé ou aussi dans le futur (prix, événements, calendrier), y compris catégorielles.
- **Sur CPU** : Mistral annonce environ 10 ms par prévision pour le modèle 5M et 140 ms pour le 209M sur un
  serveur standard à 8 threads, sans GPU.
- **Ouvert** : code d'inférence sous Apache 2.0, poids sur Hugging Face sous licence MIT modifiée. L'ajustement
  sur vos propres données est proposé via Mistral Forge.

**Résultats annoncés.** Mistral indique que chaque taille est première à taille égale sur les trois benchmarks
publics de prévision (GIFT-Eval, TIME et fev-bench), et que le modèle 28M bat Chronos-2 d'Amazon, environ quatre
fois plus gros. Ce sont les mesures de Mistral.

## Ce que teste cette démo

Cette appli place Timestral, **tel quel et sans entraînement sur M5**, exactement dans les conditions du
concours : même historique, même période de test de 28 jours, même métrique. Elle répond à trois questions
pratiques :
1. Comment un modèle zero-shot se compare-t-il à un lissage exponentiel saisonnier, proche de ce que les outils
   de planification comme SAP IBP utilisent par défaut ?
2. À quelle distance est-il de solutions réglées pendant des semaines sur ce jeu de données ?
3. Ses quantiles sont-ils exploitables pour fixer un stock visant un niveau de service ?

**Précautions.** M5 figure dans des benchmarks publics sur lesquels Timestral a été évalué : un bon score ne
prouve pas qu'il généralise. M5 mesure des ventes en caisse, censurées par les ruptures, pas des commandes B2B.
Les quantiles 0,5 % et 99,5 % sont extrapolés depuis la grille 1 %–99 % de Timestral.

**Sources.** [Annonce de Timestral (Mistral)](https://mistral.ai/news/timestral/) ·
[Code de Timestral](https://github.com/mistralai/timestral) ·
[Concours M5 (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy) ·
[Données et meilleures soumissions M5 (Nixtla)](https://github.com/Nixtla/m5-forecasts) ·
[Code source de cette démo](https://github.com/baptistesoulard/M5_Timestral)
""",
}
