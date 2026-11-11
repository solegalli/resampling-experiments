[![License https://github.com/solegalli/imbalanced-data-myths-mistakes-solutions/blob/main/LICENSE](https://img.shields.io/badge/license-BSD-success.svg)](https://github.com/solegalli/imbalanced-data-myths-mistakes-solutions/blob/main/LICENSE)
[![Sponsorship https://www.trainindata.com/](https://img.shields.io/badge/Powered%20By-TrainInData-orange.svg)](https://www.trainindata.com/)

# Undersampling Research

This repository contains research and analysis on undersampling methods for imbalanced classification datasets.

## Overview

We test undersampling methods across various datasets to determine if and when they add value over commonly used ensemble models (i.e., random forests, XGBoost, CatBoost, etc). This research provides empirical evidence to help practitioners decide whether undersampling is beneficial for their specific use cases.

## This branch: early stopping vs successive halving

The experiments in this branch test whether limiting the number of boosting rounds improves CatBoost, LightGBM and XGBoost. Previously, these models were tuned with successive halving (SH) over the number of trees. Here, they are tuned with Optuna and early stopping (patience of 50 rounds), with and without undersampling (RUS and CNN).

**Conclusion**:

- Without undersampling, early stopping gives the same performance as SH, with fewer trees.
- After undersampling, early stopping improves LightGBM (RUS and CNN) and CatBoost (RUS). XGBoost does not change.
- Even so, the undersampled models are still not better than the same models trained on the imbalanced data.

See `notebooks/01-analysis-of-ensembles-expanded.ipynb`, `notebooks/05-analysis-of-rus-expanded.ipynb` and `notebooks/05-analysis-of-cnn-expanded.ipynb`.

## Results and Discussion

The findings from this research are discussed in detail in our book:

[**Imbalanced Data: Myth, Mistakes and Modern Solutions**](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

[![Book Cover](https://github.com/solegalli/imbalanced-data-myths-mistakes-solutions/blob/main/MOCKUP_BOOK.jpg?raw=true)](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

This comprehensive guide explores common misconceptions about handling imbalanced datasets and provides practical, modern solutions based on empirical research.

## Contents

- **Notebooks**: Analysis and evaluation of resampling methods across different datasets and models
- **Functions**: Reusable evaluation and utility functions
- **Models**: Trained models and results from experiments
