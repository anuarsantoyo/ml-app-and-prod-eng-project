"""Model serving: load the trained artifact and predict a price from the
~12 essential features a buyer provides.

Every other model input is left as NaN; the pipeline's own imputers fill
them with the training median (numeric) / most-frequent (categorical), so
the model always sees its full set of features.
"""
import os

import joblib
import numpy as np
import pandas as pd

_ART = None
_NEIGHBORHOODS = []


def _load():
    global _ART, _NEIGHBORHOODS
    if _ART is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "house_price_model.joblib")
        _ART = joblib.load(path)
        _NEIGHBORHOODS = _extract_neighborhoods(_ART)
    return _ART


def _extract_neighborhoods(art):
    pre = art["model"].regressor_.named_steps["pre"]
    onehot = pre.named_transformers_["cat"].named_steps["onehot"]
    for col, cats in zip(art["categorical_features"], onehot.categories_):
        if col == "Neighborhood":
            return sorted(cats)
    return []


def valid_neighborhoods():
    _load()
    return list(_NEIGHBORHOODS)


_STYLE_ALIASES = {
    "1story": "1Story", "1 story": "1Story", "1-story": "1Story", "one story": "1Story",
    "2story": "2Story", "2 story": "2Story", "2-story": "2Story", "two story": "2Story",
    "1.5story": "1.5Fin", "1.5 story": "1.5Fin", "1.5": "1.5Fin",
    "2.5story": "2.5Fin", "2.5 story": "2.5Fin",
    "split": "SFoyer", "split foyer": "SFoyer", "split level": "SLvl",
}


def _norm_style(s):
    s = str(s).strip().lower()
    for k, v in _STYLE_ALIASES.items():
        if s == k or s in k:
            return v
    return None  # unknown -> leave NaN (imputer fills the mode)


def _to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("yes", "y", "true", "1", "on")


def predict_price(
    *,
    bedrooms,
    bathrooms,
    sqft,
    year_built,
    lot_size,
    neighborhood,
    condition,
    garage,
    basement_sqft,
    central_air,
    house_style,
    fireplaces,
):
    """Predict fair market price from a buyer's essential features.

    Returns dict with predicted_price_usd, rmse_usd, model — or an error
    dict if the neighborhood is not a known Ames neighborhood.
    """
    art = _load()

    hood = str(neighborhood).strip()
    if hood not in _NEIGHBORHOODS:
        return {
            "error": f"Unknown neighborhood '{hood}'. "
            f"Valid Ames neighborhoods: {', '.join(_NEIGHBORHOODS)}"
        }

    cols = art["numeric_features"] + art["categorical_features"]
    row = {c: np.nan for c in cols}

    row["BedroomAbvGr"] = int(bedrooms)
    full = int(float(bathrooms))
    half = 1 if (float(bathrooms) - full) >= 0.5 else 0
    row["FullBath"] = full
    row["HalfBath"] = half
    row["GrLivArea"] = float(sqft)
    row["YearBuilt"] = int(year_built)
    row["LotArea"] = float(lot_size)
    row["Neighborhood"] = hood
    row["OverallQual"] = int(condition)
    row["GarageCars"] = int(garage)
    row["TotalBsmtSF"] = float(basement_sqft)
    row["CentralAir"] = "Y" if _to_bool(central_air) else "N"
    style = _norm_style(house_style)
    if style is not None:
        row["HouseStyle"] = style
    row["Fireplaces"] = int(fireplaces)

    df = pd.DataFrame([row], columns=cols)
    price = float(art["model"].predict(df)[0])
    return {
        "predicted_price_usd": round(price, 2),
        "rmse_usd": round(art["metrics"]["rmse"], 2),
        "model": art["model_name"],
    }
