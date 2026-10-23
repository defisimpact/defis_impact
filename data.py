#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data.py — Chargement et constantes
TPI · Analyse Financière (Dash)
"""

"""
data.py — Chargement et constantes
TPI · Analyse Financière (Dash)

⚠️ Version adaptée au schéma UNIFIÉ produit par build_all_parquets.py :
   - colonnes génériques communes à toutes les sources :
       company_name, isin, sector, macro_sector,
       score, score_percentile, quintile, ticker, source
   - un seul fichier de prix partagé : prix_journaliers.parquet
     (au lieu d'un fichier de prix par source comme avant)
"""

import os
import requests
import pandas as pd
from functools import lru_cache


# ── CHEMINS ───────────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR          = '/Users/hamidi/Desktop/tpi_dash'
GITHUB_RELEASE_URL = "https://github.com/Ihssane-Hamidi/DEFIS_Analyse/releases/download/v1.0/"

# Un seul fichier de prix, partagé par toutes les sources (tous les tickers
# résolus, toutes sources confondues, y sont réunis).
FILES = {
    'mq_metriques':    'mq_metriques.parquet',
    'act_metriques':   'act_metriques.parquet',
    'ca_metriques':    'ca_metriques.parquet',
    'cp_metriques':    'cp_metriques.parquet',
    'defis_metriques': 'defis_metriques.parquet',
    'prix':            'prix_journaliers.parquet',
}

# ── CONSTANTES ────────────────────────────────────────────────────────────────
PERIODS_LABELS = {
    '2023':      '2023',
    '2024':      '2024',
    '2025':      '2025',
    '2026':      '2026',
    '2023_2026': '2023–2026',
}

QUINTILE_COLORS = {
    'Q1': '#ef4444',
    'Q2': '#f97316',
    'Q3': '#a3a3a3',
    'Q4': '#86efac',
    'Q5': '#16a34a',
}

PLOTLY_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=30, b=0),
)


# ── TÉLÉCHARGEMENT ────────────────────────────────────────────────────────────
def get_parquet(key):
    filename = FILES[key]

    local = os.path.join(LOCAL_DIR, filename)
    if os.path.exists(local):
        return local

    project_data = os.path.join(BASE_DIR, 'data', filename)
    if os.path.exists(project_data):
        return project_data

    tmp_path = os.path.join('/tmp', filename)
    if os.path.exists(tmp_path):
        return tmp_path

    url = GITHUB_RELEASE_URL + filename
    print(f"Téléchargement {filename}...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(tmp_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    return tmp_path


# ── CHARGEMENT (mis en cache) ─────────────────────────────────────────────────
@lru_cache(maxsize=None)
def load_mq():
    return pd.read_parquet(get_parquet('mq_metriques'))

@lru_cache(maxsize=None)
def load_act():
    return pd.read_parquet(get_parquet('act_metriques'))

@lru_cache(maxsize=None)
def load_ca():
    return pd.read_parquet(get_parquet('ca_metriques'))

@lru_cache(maxsize=None)
def load_defis():
    return pd.read_parquet(get_parquet('defis_metriques'))

@lru_cache(maxsize=None)
def load_cp():
    return pd.read_parquet(get_parquet('cp_metriques'))


@lru_cache(maxsize=None)
def load_prix():
    """Fichier de prix journaliers partagé par toutes les sources."""
    df = pd.read_parquet(get_parquet('prix'))
    df.index = pd.to_datetime(df.index)
    return df

# Alias rétro-compatibles : anciennement un fichier de prix par source,
# désormais tous pointent vers le même fichier partagé (les colonnes
# non pertinentes pour une source donnée sont simplement ignorées par les
# lookups faits via 'ticker' dans le reste de l'app).
load_mq_prix    = load_prix
load_act_prix   = load_prix
load_ca_prix    = load_prix
load_defis_prix = load_prix
load_cp_prix    = load_prix


@lru_cache(maxsize=None)
def load_brent():
    filename = 'brent.parquet'

    for path in [
        os.path.join(LOCAL_DIR,   filename),
        os.path.join(BASE_DIR, 'data', filename),
        os.path.join(BASE_DIR,    filename),
    ]:
        if os.path.exists(path):
            return pd.read_parquet(path)['Close']

    try:
        url = GITHUB_RELEASE_URL + filename
        print(f"Téléchargement {filename} depuis GitHub...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        tmp_path = os.path.join('/tmp', filename)
        with open(tmp_path, 'wb') as f:
            f.write(r.content)
        return pd.read_parquet(tmp_path)['Close']
    except Exception as e:
        print(f"ERREUR brent : {e}")
        return pd.Series(dtype=float)


# ── HELPERS QUINTILES ─────────────────────────────────────────────────────────
def _add_quintiles(valid, score_col, quintile_col, pct_col):
    """
    Ajoute quintile_col et pct_col si absents.
    Utilise rank(pct=True) + cut (pas qcut) pour éviter les erreurs
    'Bin edges not unique' quand le score a beaucoup de valeurs identiques
    (c'est la méthode utilisée dans build_all_parquets.py).
    """
    if pct_col not in valid.columns:
        valid[pct_col] = valid[score_col].rank(pct=True)

    if quintile_col not in valid.columns:
        m = valid[score_col].notna()
        if m.sum() >= 5:
            ranks = valid.loc[m, score_col].rank(method='first', pct=True)
            valid.loc[m, quintile_col] = pd.cut(
                ranks,
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                include_lowest=True,
            ).astype(str)
        else:
            valid[quintile_col] = pd.NA

    return valid


def _normalize_quintile_col(valid, col):
    """
    Convertit une colonne de quintiles entiers (1–5) en labels 'Q1'–'Q5'.
    Ne touche à rien si la colonne contient déjà des strings 'Q1'–'Q5'.
    """
    if col not in valid.columns:
        return valid
    sample = valid[col].dropna()
    if sample.empty:
        return valid
    # Déjà au bon format
    if sample.astype(str).str.match(r'^Q[1-5]$').all():
        return valid
    # Mapper int → 'Q{n}'
    mapping = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5',
               '1': 'Q1', '2': 'Q2', '3': 'Q3', '4': 'Q4', '5': 'Q5'}
    valid[col] = valid[col].map(mapping)
    return valid


# ── PRÉPARATION DES DATAFRAMES ────────────────────────────────────────────────
# NOTE : pas de @lru_cache ici — lru_cache ne supporte pas les DataFrames
#        (non hashables). Le cache est inutile car ces fonctions ne sont
#        appelées qu'une seule fois au démarrage dans app.py.
#
# Toutes les fonctions ci-dessous partagent désormais le même schéma unifié :
#   'company_name', 'score', 'quintile', 'score_percentile', 'ticker', ...
# Elles exposent en plus 'Company Name' (alias capitalisé, pour compat avec
# le code d'affichage existant qui référence encore ce nom de colonne).

def _prepare_valid_generic(df):
    """
    Filtre commun à toutes les sources sur le schéma unifié :
      - ticker résolu
      - Rendement_2023_2026 disponible
      - quintile/score_percentile complétés si jamais absents
      - alias 'Company Name' ajouté pour compat affichage
    """
    df = df.loc[:, ~df.columns.duplicated()]

    valid = df[
        df['ticker'].notna() &
        (df['ticker'].astype(str) != 'None') &
        df['Rendement_2023_2026'].notna()
    ].copy()

    # 'quintile' peut contenir la string 'nan' selon la source d'origine → NaN réel
    if 'quintile' in valid.columns:
        valid['quintile'] = valid['quintile'].replace('nan', pd.NA)

    valid = _normalize_quintile_col(valid, 'quintile')
    valid = _add_quintiles(valid, 'score', 'quintile', 'score_percentile')

    valid['Company Name'] = valid['company_name']

    return valid


def prepare_valid_mq(df_mq):
    """Filtre MQ (schéma unifié)."""
    return _prepare_valid_generic(df_mq)


def prepare_valid_act(df_act):
    """Filtre ACT (schéma unifié)."""
    return _prepare_valid_generic(df_act)


def prepare_valid_ca(df_ca):
    """Filtre CA (schéma unifié)."""
    return _prepare_valid_generic(df_ca)


def prepare_valid_defis(df_defis):
    """Filtre DEFIS (schéma unifié). Rappel : pas de score exploitable pour
    cette source, donc 'score'/'score_percentile'/'quintile' resteront NaN."""
    return _prepare_valid_generic(df_defis)


def prepare_valid_cp(df_cp):
    """Filtre CP (schéma unifié)."""
    return _prepare_valid_generic(df_cp)
