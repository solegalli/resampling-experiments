"""
This module contains functions to load the datasets for the analysis.
The datasets to be analysed were decided in the notebooks located in
notebooks/exploratory-data-analysis.

Preprocessing steps: categorical variables are arbitrary
ordinal-encoded, constant features are dropped, and the data is split 70/30 with
``random_state=0``. Missing values, when present, are imputed as 3 times the maximum
value of the variable for numerical variables or with the string missing for 
categorical variables.
Feature engineering steps are applied before splitting; the resulting leakage 
is negligible and applied identically to every model, keeping the comparison fair.
"""
import warnings
import zipfile
from io import BytesIO
from urllib.request import urlopen

import numpy as np
import pandas as pd
from feature_engine.encoding import OrdinalEncoder
from feature_engine.imputation import CategoricalImputer, EndTailImputer
from feature_engine.selection import DropConstantFeatures
from imblearn.datasets import fetch_datasets
from keel_ds import load_data
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

DATASETS_IMBLEARN = [
    "abalone_19",
    "arrhythmia",
    "car_eval_4",
    "coil_2000",
    "ecoli",
    "isolet",
    "letter_img",
    "libras_move",
    "mammography",
    "oil",
    "optical_digits",
    "ozone_level",
    "pen_digits",
    "protein_homo",
    "satimage",
    "scene",
    "sick_euthyroid",
    "solar_flare_m0",
    "spectrometer",
    "thyroid_sick",
    "us_crime",
    "webpage",
    "wine_quality",
    "yeast_me2",
]

DATASETS_KEEL = [
    "cleveland-0_vs_4",
    "dermatology-6",
    "glass-0-1-4-6_vs_2",
    "kddcup-buffer_overflow_vs_back",
    "kr-vs-k-one_vs_fifteen",
    "led7digit-0-2-4-5-6-7-8-9_vs_1",
    "page-blocks-1-3_vs_4",
    "pima",
    "poker-8-9_vs_5",
    "shuttle-2_vs_5",
]

DATASETS_SEPARABLE = [
    "dermatology-6",
    "arrhythmia",
    "kddcup-buffer_overflow_vs_back",
    "kr-vs-k-one_vs_fifteen",
    "shuttle-2_vs_5",
]

DATASETS_UCI_OPENML = [
    "diabetes130",
    "default_credit",
    "htru2",
    "credit_fraud",
    "secom",
    "bank-marketing",
    "telco",
    "adult",
]

DATASETS_LS = [
    dataset
    for dataset in dict.fromkeys(DATASETS_IMBLEARN + DATASETS_KEEL + DATASETS_UCI_OPENML)
    if dataset not in DATASETS_SEPARABLE
]


def load_dataset(dataset):
    if dataset in DATASETS_KEEL:
        # load dataset from keel
        data = load_data(dataset, type_data="imbalanced", raw=True)

        target = data.iloc[:, -1:]
        target = np.where(target == "negative", 0, 1)
        target = target.ravel()

        data = data.iloc[:, :-1]

        if dataset == "kddcup-buffer_overflow_vs_back":
            data = DropConstantFeatures().fit_transform(data)

        # some datasets contain categorical variables
        if dataset in [
            "kddcup-buffer_overflow_vs_back",
            "cleveland-0_vs_4",
            "kr-vs-k-one_vs_fifteen",
        ]:
            data = OrdinalEncoder(encoding_method="arbitrary").fit_transform(data)

        # separate dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            target,
            test_size=0.3,
            random_state=0,
        )

    elif dataset in DATASETS_UCI_OPENML:
        X, y = load_uci_openml(dataset)
        # separate dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=0,
        )

    else:
        # load dataset from imbalanced learn
        data = fetch_datasets()[dataset]
        data.target = np.where(data.target < 0, 0, 1)

        # remove constant features
        if dataset in ["arrhythmia", "oil", "optical_digits", "thyroid_sick"]:
            data.data = DropConstantFeatures().fit_transform(data.data)

        # separate dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.3,
            random_state=0,
        )

    return X_train, X_test, y_train, y_test


def load_uci_openml(dataset):
    datasets = {
        "diabetes130": 296,
        "default_credit": 350,
        "htru2": 372,
    }

    if dataset == "credit_fraud":
        data = fetch_openml(data_id=1597, as_frame=True)
        X = data.data.copy()
        y = data.target.copy().astype(int)

    elif dataset == "secom":
        # SECOM is not available through the ucimlrepo API, so we download it
        #  directly from the UCI static file server.
        SECOM_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
        with urlopen(SECOM_URL) as resp:
            archive = zipfile.ZipFile(BytesIO(resp.read()))
        with archive.open("secom.data") as f:
            X = pd.read_csv(f, sep=r"\s+", header=None, na_values="NaN")
        with archive.open("secom_labels.data") as f:
            labels = pd.read_csv(f, sep=r"\s+", header=None)
        X.columns = [f"feature_{i}" for i in range(X.shape[1])]
        y = labels.iloc[:, 0].replace(
            -1, 0
        )

        X = EndTailImputer(imputation_method="max", fold=3).fit_transform(X)
        X = DropConstantFeatures().fit_transform(X)

    elif dataset == "bank-marketing":
        data = fetch_openml(
            name="bank-marketing", version=1, as_frame=True, parser="auto"
        )
        X = data.data
        y = (data.target == "2").astype(int)
        X = OrdinalEncoder(encoding_method="arbitrary").fit_transform(X)

    elif dataset == "adult":
        adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        X = adult.data
        y = adult.target
        y = y.map({"<=50K": 0, ">50K": 1}).astype(int)

        X = CategoricalImputer().fit_transform(X)
        X = OrdinalEncoder(encoding_method="arbitrary").fit_transform(X)

    elif dataset == "telco":
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
        y = (df["Churn"] == "Yes").astype(int)
        X = df.drop(columns=["customerID", "Churn"])

        # Replace empty string with nan
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["TotalCharges"] = X["TotalCharges"].replace(" ", pd.NA)
        X = EndTailImputer(imputation_method="max", fold=3).fit_transform(X)
        X = OrdinalEncoder(encoding_method="arbitrary").fit_transform(X)

    else:
        # ucimlrepo reads the source CSV without specifying dtypes, which
        # raises a DtypeWarning for diabetes130's mixed-type diagnosis
        # columns (diag_1/2/3). We coerce those columns ourselves below,
        # so the warning is safe to suppress here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
            data = fetch_ucirepo(id=datasets[dataset])
        X = data.data.features.copy()
        y = data.data.targets.squeeze().copy()

    if dataset == "diabetes130":
        diag_cols = ["diag_1", "diag_2", "diag_3"]

        # Create new columns to capture non-numeric values
        for col in diag_cols:
            numeric_converted = pd.to_numeric(X[col], errors="coerce")
            non_numeric_mask = numeric_converted.isnull() & X[col].notna()

            # New variable: non-numeric values only (numerical become NaN)
            X[f"{col}_categorical"] = X[col].where(non_numeric_mask, other=None)

            # Original variable: replace non-numeric with NaN
            X[col] = numeric_converted

        X = EndTailImputer(imputation_method="max", fold=3).fit_transform(X)
        X = CategoricalImputer().fit_transform(X)
        X = OrdinalEncoder(encoding_method="arbitrary").fit_transform(X)
        X = DropConstantFeatures().fit_transform(X)
        y = (y == "<30").astype(int).to_numpy()

    return X, y
