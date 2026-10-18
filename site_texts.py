"""Website texts that are not in app_texts.py: the VN1 "Learn" page and the glossary (English and French).

Rendered to HTML by export_static.py (markdown) and shipped in public/data/learn.json and glossary.json.
"""

from __future__ import annotations

VN1_LEARN = {
    "en": """
## The VN1 competition

**VN1** is a forecasting competition organised in 2024 by **Nicolas Vandeput** (SupChains), with Flieber and
Syrup Tech. Unlike M5, it is a **supply chain planning** case: weekly sales of e-commerce brands, by product and
**warehouse**, the level at which replenishment is decided.

**The task.** Forecast **13 weeks** of sales for **15,053 product × warehouse series** (several client brands,
each with its own warehouses). The history runs from July 2020 to early January 2024 (183 weeks, phases 0 and 1);
the final test period (phase 2) runs from **8 January to 1 April 2024**. Prices are given for the past only.

**How it is scored.** One number, close to what a demand planner tracks every month:

**score = MAE% + |Bias%|**, with MAE% = Σ|forecast − sales| / Σ sales and Bias% = Σ(forecast − sales) / Σ sales,
summed over all series and weeks. Lower is better. Bias counts twice: a forecast that is systematically high
(excess stock) or low (lost sales) is penalised even if it looks accurate series by series.

**What it is like.** Many series are short, lumpy or stop selling (products phased out), and volumes differ by
several orders of magnitude between products. The top 5 forecasts and two official benchmarks (naive and 12-week
moving average) are published by the organisers, so every score on this site is computed with the same rule on the
same weeks.

**Why it is a useful second test.** VN1 is not one of the benchmarks Mistral reports results on, its data is more
recent than M5's and the scoring rule is the one used in practice. The data has been public since 2024, so an
overlap with Timestral's pre-training data cannot be excluded either.
""",
    "en_tests": """
## What this demo tests on VN1

Timestral, **unchanged and untrained on VN1**, forecasts the 13 test weeks of the 15,053 series from their history.
Two inputs are compared: sales only, and sales plus the **relative price** (price divided by the series' usual price)
as a covariate known only in the past, since VN1 gives no future prices. The point forecast is the median or
the mean of Timestral's distribution, or the quantile that gave the best average score on four backtests
covering the quarters of 2023 (the last one is phase 1), applied once to the test weeks: this is the forecast shown in the leaderboard.

**Sources.** [VN1 data, top 5 forecasts and scoring code (SupChains)](https://github.com/SupChains/VN1) ·
[Timestral announcement (Mistral)](https://mistral.ai/news/timestral/) ·
[Source code of this demo](https://github.com/baptistesoulard/M5_Timestral)
""",
    "fr": """
## Le concours VN1

**VN1** est un concours de prévision organisé en 2024 par **Nicolas Vandeput** (SupChains), avec Flieber et
Syrup Tech. Contrairement au M5, c'est un cas de **planification supply chain** : des ventes hebdomadaires de marques
e-commerce, par produit et par **entrepôt**, la maille à laquelle se décide le réapprovisionnement.

**L'exercice.** Prévoir **13 semaines** de ventes pour **15 053 séries produit × entrepôt** (plusieurs marques
clientes, chacune avec ses entrepôts). L'historique va de juillet 2020 à début janvier 2024 (183 semaines, phases 0
et 1) ; la période de test finale (phase 2) va du **8 janvier au 1er avril 2024**. Les prix ne sont fournis que pour
le passé.

**La notation.** Un seul chiffre, proche de ce qu'un demand planner suit chaque mois :

**score = MAE % + |Biais %|**, avec MAE % = Σ|prévision − ventes| / Σ ventes et Biais % = Σ(prévision − ventes) /
Σ ventes, cumulés sur toutes les séries et toutes les semaines. Plus bas = meilleur. Le biais compte deux fois : une
prévision systématiquement haute (surstock) ou basse (ventes perdues) est pénalisée même si elle paraît précise
série par série.

**À quoi ressemblent les données.** Beaucoup de séries sont courtes, erratiques ou s'arrêtent (produits en fin de
vie), et les volumes varient de plusieurs ordres de grandeur d'un produit à l'autre. Les 5 meilleures prévisions et
deux benchmarks officiels (naïf et moyenne mobile 12 semaines) sont publiés par les organisateurs : tous les scores
de ce site sont calculés avec la même règle, sur les mêmes semaines.

**Pourquoi c'est un second test utile.** VN1 ne fait pas partie des benchmarks sur lesquels Mistral publie des
résultats, ses données sont plus récentes que celles du M5 et la règle de notation est celle utilisée en pratique.
Les données sont publiques depuis 2024 : un recouvrement avec les données de pré-entraînement de Timestral ne peut
donc pas non plus être exclu.
""",
    "fr_tests": """
## Ce que teste cette démo sur VN1

Timestral, **tel quel et sans entraînement sur VN1**, prévoit les 13 semaines de test des 15 053 séries à partir de
leur historique. Deux entrées sont comparées : les ventes seules, et les ventes plus le **prix relatif** (prix divisé
par le prix habituel de la série) comme covariable connue seulement dans le passé, VN1 ne donnant pas les prix
futurs. La prévision ponctuelle est la médiane ou l'espérance de la distribution de Timestral, ou le quantile qui a donné le meilleur score moyen sur quatre backtests
couvrant les trimestres de 2023 (le dernier est la phase 1), appliqué une seule fois aux semaines de test : c'est la prévision retenue dans le classement.

**Sources.** [Données VN1, top 5 et code de notation (SupChains)](https://github.com/SupChains/VN1) ·
[Annonce de Timestral (Mistral)](https://mistral.ai/news/timestral/) ·
[Code source de cette démo](https://github.com/baptistesoulard/M5_Timestral)
""",
}

# (section, [(anchor, term EN, term FR, definition EN, definition FR)])
GLOSSARY = [
    (("Reading a forecast", "Lire une prévision"), [
        ("point", "Point forecast", "Prévision ponctuelle",
         "A single number per period, e.g. 120 units next week. Simple to use, but silent about risk.",
         "Un seul chiffre par période, par exemple 120 unités la semaine prochaine. Simple à utiliser, mais muet sur le risque."),
        ("quantile", "Quantile · P10, P50, P90", "Quantile · P10, P50, P90",
         "The P90 is the quantity that demand should stay below 90% of the time. A probabilistic forecast gives several "
         "quantiles; Timestral gives 21, from P1 to P99. Setting stock at the P90 targets a 90% service level.",
         "Le P90 est la quantité sous laquelle la demande devrait rester 90 % du temps. Une prévision probabiliste donne "
         "plusieurs quantiles ; Timestral en donne 21, du P1 au P99. Dimensionner le stock au P90 vise 90 % de niveau de service."),
        ("median-mean", "Median vs mean", "Médiane vs espérance",
         "The median (P50) is the middle scenario; the mean (expected value) is the average over all scenarios. For a "
         "slow mover that sells 0 most days, the median is 0 while the mean is, say, 0.3 units a day: only the mean adds "
         "up correctly to store or company totals.",
         "La médiane (P50) est le scénario central ; l'espérance est la moyenne de tous les scénarios. Pour un article lent "
         "qui vend 0 la plupart des jours, la médiane vaut 0 alors que l'espérance vaut par exemple 0,3 unité par jour : "
         "seule l'espérance s'additionne correctement en totaux magasin ou entreprise."),
        ("band", "80% band (prediction interval)", "Bande 80 % (intervalle de prévision)",
         "The range between P10 and P90. If the forecast is well calibrated, actual demand falls inside about 80% of the "
         "time; much less means the model is overconfident.",
         "La plage entre P10 et P90. Si la prévision est bien calibrée, la demande réelle y tombe environ 80 % du temps ; "
         "beaucoup moins signifie que le modèle est trop sûr de lui."),
        ("calibration", "Calibrated quantile (point forecast)", "Quantile calibré (prévision ponctuelle)",
         "Instead of the median, the point forecast is taken at the quantile that gave the best score on an earlier "
         "period (here four backtests covering 2023), then applied unchanged to the period evaluated. A P55 corrects a median that "
         "systematically under-forecasts, without looking at the weeks being scored.",
         "Au lieu de la médiane, la prévision ponctuelle est prise au quantile qui a donné le meilleur score sur une "
         "période antérieure (ici quatre backtests couvrant 2023), puis appliqué tel quel à la période évaluée. Un P55 corrige une "
         "médiane qui sous-prévoit systématiquement, sans regarder les semaines notées."),
        ("horizon", "Horizon", "Horizon",
         "How far ahead the forecast goes: 28 days for M5, 13 weeks for VN1.",
         "La distance de prévision : 28 jours pour le M5, 13 semaines pour VN1."),
        ("intermittent", "Intermittent (lumpy) demand", "Demande intermittente (erratique)",
         "Many periods with zero sales, then a few units. Typical of slow movers and spare parts; averages and "
         "percentage errors behave badly on it.",
         "Beaucoup de périodes sans vente, puis quelques unités. Typique des articles lents et des pièces de rechange ; "
         "moyennes et erreurs en pourcentage s'y comportent mal."),
        ("hierarchy", "Hierarchy · bottom-up vs direct", "Hiérarchie · bottom-up vs direct",
         "M5 series are nested (item × store → store → state → total). Bottom-up: forecast the finest level and add up "
         "(the official M5 rule). Direct: forecast every level on its own.",
         "Les séries M5 sont emboîtées (article × magasin → magasin → État → total). Bottom-up : prévoir la maille la plus "
         "fine puis additionner (règle officielle du M5). Direct : prévoir chaque niveau séparément."),
        ("covariates", "Covariates", "Covariables",
         "Extra inputs that help explain sales: price, promotions, holidays, day of week. Known in the future (calendar, "
         "planned prices) or only in the past (realised prices in VN1). Categorical covariates are labels, such as the "
         "event name.",
         "Variables explicatives ajoutées aux ventes : prix, promotions, jours fériés, jour de la semaine. Connues dans le "
         "futur (calendrier, prix planifiés) ou seulement dans le passé (prix réalisés dans VN1). Les covariables "
         "catégorielles sont des libellés, comme le nom d'un événement."),
        ("snap", "SNAP", "SNAP",
         "US food-stamp programme. Benefits are paid on fixed days per state, and food sales rise on those days (M5).",
         "Programme américain d'aide alimentaire. Les aides sont versées à jours fixes selon l'État, et les ventes "
         "alimentaires montent ces jours-là (M5)."),
        ("zero-shot", "Foundation model · zero-shot", "Modèle de fondation · zero-shot",
         "A model pre-trained once on a very large set of series and used as is on new data, with no training on it "
         "(zero-shot). Fine-tuning would retrain it on your data; it is not done here.",
         "Un modèle pré-entraîné une fois sur un très grand ensemble de séries et utilisé tel quel sur de nouvelles données, "
         "sans entraînement dessus (zero-shot). Le fine-tuning le réentraînerait sur vos données ; il n'est pas fait ici."),
    ]),
    (("Accuracy indicators", "Indicateurs de performance"), [
        ("mae", "MAE · MAE%", "MAE · MAE %",
         "Mean absolute error: average gap between forecast and actual, in units. MAE% divides the total absolute error "
         "by total sales, so it can be compared across products (VN1).",
         "Erreur absolue moyenne : écart moyen entre prévision et réel, en unités. Le MAE % divise l'erreur absolue totale "
         "par les ventes totales, pour comparer des produits entre eux (VN1)."),
        ("wape", "WAPE", "WAPE",
         "Weighted absolute percentage error = Σ|forecast − actual| / Σ actual: the same formula as MAE%. Unlike MAPE, it "
         "does not explode on days with zero sales.",
         "Erreur absolue en pourcentage pondérée = Σ|prévision − réel| / Σ réel : même formule que le MAE %. Contrairement "
         "au MAPE, elle n'explose pas les jours sans vente."),
        ("bias", "Bias · Bias%", "Biais · Biais %",
         "Σ(forecast − actual) / Σ actual. Positive: the forecast is too high (excess stock); negative: too low (stock-outs, "
         "lost sales). Errors in both directions cancel out, so a good bias does not mean a good forecast.",
         "Σ(prévision − réel) / Σ réel. Positif : prévision trop haute (surstock) ; négatif : trop basse (ruptures, ventes "
         "perdues). Les erreurs dans les deux sens se compensent : un bon biais ne garantit pas une bonne prévision."),
        ("vn1-score", "VN1 score", "Score VN1",
         "MAE% + |Bias%|, over all series and weeks. Rewards forecasts that are both accurate and unbiased. Lower is better.",
         "MAE % + |Biais %|, sur toutes les séries et semaines. Récompense les prévisions à la fois précises et sans biais. "
         "Plus bas = meilleur."),
        ("rmsse", "RMSSE", "RMSSE",
         "Root mean squared scaled error: the forecast's root mean squared error divided by that of a naive “same as "
         "yesterday” forecast over the history. Scale-free, so a 10-unit and a 10,000-unit series can be compared; "
         "below 1 is better than that naive yardstick.",
         "Erreur quadratique moyenne mise à l'échelle : la racine de l'erreur quadratique moyenne de la prévision, divisée "
         "par celle d'une prévision naïve « comme la veille » sur l'historique. Sans unité : une série de 10 unités et une "
         "de 10 000 se comparent ; sous 1, on fait mieux que ce naïf."),
        ("wrmsse", "WRMSSE", "WRMSSE",
         "The official M5 accuracy score: RMSSE of the 42,840 series, weighted by their dollar sales over the last 28 days, "
         "the 12 levels counting equally. Lower is better; the winner scored 0.520.",
         "Le score officiel de précision du M5 : RMSSE des 42 840 séries, pondérées par leur chiffre d'affaires des 28 "
         "derniers jours, les 12 niveaux comptant à parts égales. Plus bas = meilleur ; le vainqueur a obtenu 0,520."),
        ("spl", "Pinball loss · SPL", "Perte pinball · SPL",
         "Error of a quantile. For the P90, being short costs 0.9 per unit and being over costs 0.1 per unit, which is how a "
         "90% service level trades lost sales against stock. SPL is this loss scaled like the RMSSE.",
         "Erreur d'un quantile. Pour le P90, manquer coûte 0,9 par unité et dépasser 0,1 par unité : c'est l'arbitrage "
         "ventes perdues contre stock d'un niveau de service de 90 %. La SPL est cette perte mise à l'échelle comme la RMSSE."),
        ("wspl", "WSPL", "WSPL",
         "The official score of the M5 Uncertainty track: SPL averaged over 9 quantiles (P0.5 to P99.5) and weighted like "
         "the WRMSSE. Lower is better; the winner scored 0.154.",
         "Le score officiel de la piste Incertitude du M5 : SPL moyennée sur 9 quantiles (P0,5 à P99,5) et pondérée comme "
         "la WRMSSE. Plus bas = meilleur ; le vainqueur a obtenu 0,154."),
        ("coverage", "Coverage", "Couverture",
         "Share of periods where actual demand fell inside the 80% band. Close to 80% means well calibrated quantiles.",
         "Part des périodes où la demande réelle est tombée dans la bande 80 %. Proche de 80 % = quantiles bien calibrés."),
        ("service", "Service level · fill rate", "Niveau de service · taux de service",
         "Days (or weeks) covered: share of periods where stock covered all demand. Fill rate: share of the demand volume "
         "served from stock. A 90% cycle service level usually gives a fill rate well above 90%.",
         "Jours (ou semaines) couverts : part des périodes où le stock a couvert toute la demande. Taux de service en "
         "volume : part des quantités demandées servies depuis le stock. 90 % de jours couverts donnent en général un taux "
         "de service en volume nettement supérieur à 90 %."),
    ]),
    (("Forecasting methods", "Méthodes de prévision"), [
        ("naive", "Naive", "Naïf",
         "Next periods = last observed value. The yardstick every method should beat.",
         "Les prochaines périodes = dernière valeur observée. L'étalon que toute méthode doit battre."),
        ("snaive", "Seasonal naive", "Naïf saisonnier",
         "Repeats the last season: same day last week (M5), same week last year (VN1).",
         "Répète la dernière saison : même jour la semaine précédente (M5), même semaine l'an dernier (VN1)."),
        ("ma", "Moving average (MA12, MA28)", "Moyenne mobile (MA12, MA28)",
         "Flat forecast equal to the average of the last 12 weeks or 28 days.",
         "Prévision plate égale à la moyenne des 12 dernières semaines ou des 28 derniers jours."),
        ("ses", "Simple exponential smoothing (SES)", "Lissage exponentiel simple (SES)",
         "A weighted average where recent periods weigh more, the weight decaying exponentially. Flat forecast that "
         "adapts to level changes.",
         "Une moyenne pondérée où les périodes récentes pèsent plus, le poids décroissant de façon exponentielle. "
         "Prévision plate qui s'adapte aux changements de niveau."),
        ("ets", "AutoETS (Holt-Winters)", "AutoETS (Holt-Winters)",
         "Exponential smoothing of level, trend and seasonality (Error-Trend-Season), the variant being chosen "
         "automatically per series. The method behind many planning tools; here with a weekly season on M5 and a yearly "
         "season on VN1.",
         "Lissage exponentiel du niveau, de la tendance et de la saisonnalité (Erreur-Tendance-Saison), la variante étant "
         "choisie automatiquement pour chaque série. La méthode de nombreux outils de planification ; ici avec une saison "
         "hebdomadaire sur M5 et annuelle sur VN1."),
        ("arima", "AutoARIMA (SARIMA)", "AutoARIMA (SARIMA)",
         "Explains each value by the previous values and the previous errors of the series, after removing trend by "
         "differencing; the seasonal version (SARIMA) adds the same terms one season back. Orders are chosen automatically "
         "per series. Season: 7 days on M5, 52 weeks on VN1.",
         "Explique chaque valeur par les valeurs et les erreurs précédentes de la série, après avoir retiré la tendance par "
         "différenciation ; la version saisonnière (SARIMA) ajoute les mêmes termes une saison en arrière. Les ordres sont "
         "choisis automatiquement pour chaque série. Saison : 7 jours sur M5, 52 semaines sur VN1."),
        ("croston", "Croston · SBA", "Croston · SBA",
         "For intermittent demand: forecasts separately the size of a sale when there is one and the interval between "
         "sales; demand per period = size ÷ interval. SBA (Syntetos-Boylan) corrects Croston's tendency to over-forecast.",
         "Pour la demande intermittente : prévoit séparément la taille d'une vente quand il y en a une et l'intervalle "
         "entre deux ventes ; demande par période = taille ÷ intervalle. SBA (Syntetos-Boylan) corrige la tendance de "
         "Croston à surestimer."),
        ("tsb", "TSB", "TSB",
         "Teunter-Syntetos-Babai, a Croston variant: it updates the probability of selling every period, including periods "
         "without sales, so the forecast fades out when an item stops selling. Suited to products at end of life.",
         "Teunter-Syntetos-Babai, une variante de Croston : elle met à jour la probabilité de vendre à chaque période, y "
         "compris sans vente, si bien que la prévision s'éteint quand un article ne se vend plus. Adaptée aux produits en "
         "fin de vie."),
        ("adida", "ADIDA · iMAPA", "ADIDA · iMAPA",
         "Group periods into larger buckets (fewer zeros), forecast the buckets with exponential smoothing, then spread the "
         "result back. iMAPA does it at several bucket sizes and averages them.",
         "Regroupe les périodes en paquets plus larges (moins de zéros), prévoit ces paquets par lissage exponentiel, puis "
         "redistribue le résultat. iMAPA le fait avec plusieurs tailles de paquets et en fait la moyenne."),
        ("lightgbm", "LightGBM (competition winners)", "LightGBM (vainqueurs des concours)",
         "Gradient-boosted decision trees trained on all series at once with engineered features (recent sales, prices, "
         "calendar). Used by most top M5 teams, after weeks of tuning on that dataset.",
         "Arbres de décision boostés entraînés sur toutes les séries à la fois avec des variables construites (ventes "
         "récentes, prix, calendrier). Utilisés par la plupart des meilleures équipes du M5, après des semaines de réglage "
         "sur ce jeu de données."),
        ("empirical", "Empirical quantiles", "Quantiles empiriques",
         "Quantiles read directly from recent history, per day of week (last 26 weeks): a simple probabilistic yardstick.",
         "Quantiles lus directement dans l'historique récent, par jour de semaine (26 dernières semaines) : un étalon "
         "probabiliste simple."),
        ("timestral", "Timestral", "Timestral",
         "Mistral AI's time-series foundation model (5M to 209M parameters). Reads the history, and covariates if given, "
         "and returns 21 quantiles per period in one pass, without training on the data. The mean is computed from those "
         "quantiles.",
         "Le modèle de fondation de séries temporelles de Mistral AI (5 à 209 millions de paramètres). Lit l'historique, et "
         "les covariables si on les fournit, et renvoie 21 quantiles par période en une passe, sans entraînement sur les "
         "données. L'espérance est calculée à partir de ces quantiles."),
    ]),
]

GLOSSARY_INTRO = {
    "en": "The indicators and methods used on this site, in plain words. Lower is better for every error indicator.",
    "fr": "Les indicateurs et méthodes utilisés sur ce site, en termes simples. Pour tous les indicateurs d'erreur, plus bas = meilleur.",
}
