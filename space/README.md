---
title: Timestral × M5
emoji: 📈
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Timestral de Mistral face au benchmark M5 Walmart
---

# Timestral × M5 Walmart

Démo interactive du modèle de fondation de séries temporelles **Timestral** (Mistral) sur le benchmark
**M5 Walmart** : 42 840 séries de ventes quotidiennes, horizon 28 jours, période de test officielle.

- **Explorer une série** : prévision en direct d'une série (total, magasin, rayon, article…), comparée au réel,
  au naïf saisonnier, à AutoETS, à TSB et au vainqueur du concours ; lecture du niveau de service obtenu
  en dimensionnant le stock sur un quantile.
- **Mini-benchmark** : score WRMSSE / WSPL niveau par niveau sur un échantillon, en une à deux minutes.
- **Vos données** : prévision ou backtest de vos propres séries (CSV).

Code et benchmark complet : [github.com/baptistesoulard/M5_Timestral](https://github.com/baptistesoulard/M5_Timestral)
