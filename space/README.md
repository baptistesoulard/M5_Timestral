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
short_description: Mistral's Timestral on the M5 Walmart forecasting benchmark
---

# Timestral × M5 Walmart

Interactive demo of **Timestral**, Mistral's time-series foundation model, on the **M5 Walmart** benchmark:
42,840 daily sales series, 28-day horizon, official test period. The model is used zero-shot, on CPU.
The app is available in **English and French** (follows the browser language, switch at the top).

- **Explore**: live forecast of any series (total, store, department, item…) against actuals,
  seasonal naive, AutoETS, TSB and the competition winner, plus the service level reached when stock is set
  at a forecast quantile.
- **Benchmark**: WRMSSE / WSPL level by level on a sample, in one to two minutes.
- **Your data**: forecast or backtest your own series (CSV).
- **Learn**: the M5 competition, time-series foundation models and Timestral, explained.

Code and full benchmark: [github.com/baptistesoulard/M5_Timestral](https://github.com/baptistesoulard/M5_Timestral)
