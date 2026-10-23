"""
pages/ols.py — Page Régression OLS
TPI · Analyse Financière (Dash)
"""

import pandas as pd
from dash import html, dcc, dash_table, Input, Output
from charts import plot_scatter_ols, plot_coefficients_secteurs
from utils import prepare_ols_data, winsorize, run_ols, sig_stars, calc_momentum
from data import PERIODS_LABELS
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
def layout(ctx: dict):
    periods_options = [
        {'label': v, 'value': k}
        for k, v in PERIODS_LABELS.items()
    ]
    dep_options = [
        {'label': 'Rendement',    'value': 'Rendement'},
        {'label': 'Volatilité',   'value': 'Volatilite'},
        {'label': 'Sharpe',       'value': 'Sharpe'},
        {'label': 'Max Drawdown', 'value': 'MaxDrawdown'},
    ]
    model_options = [
        {'label': 'Modèle 1 · Simple (Score + Secteur)',          'value': 'simple'},
        {'label': 'Modèle 2 · Interaction (Score × Secteur)',     'value': 'interaction'},
        {'label': 'Modèle 3 · Fama-French (+ Taille + B/M)',      'value': 'fama_french'},
        {'label': 'Modèle 4 · Momentum (+ 12-1 mois)',            'value': 'momentum'},
    ]

    return html.Div([
        dcc.Store(id='ols-dataset-local', data=ctx.get('dataset_key', 'mq')),
        html.H2(
            'Régression OLS · Score → Performance financière',
            style={'fontSize': '16px', 'fontWeight': '500',
                   'color': '#e6edf3', 'marginBottom': '16px'},
        ),

        # Note temporelle
        html.Div(className='note-box', style={'marginBottom': '16px'}, children=[
            "⚠ Note méthodologique : les scores TPI sont issus de l'année fiscale 2023–2024, "
            "publiés en 2025. Les régressions sur Rendement 2023 et 2024 ont une valeur "
            "descriptive uniquement (look-ahead bias). ",
            html.Strong("Le Rendement 2025 constitue le test prédictif de référence."),
        ]),

        # ── Sélecteurs ───────────────────────────────────────────────────────
        html.Div(className='row-3', style={'marginBottom': '16px'}, children=[
            html.Div([
                html.Div('Période', className='kpi-label'),
                dcc.Dropdown(
                    id='ols-period',
                    options=periods_options,
                    value='2025',        # ← 2025 par défaut (test prédictif)
                    clearable=False,
                    style=_dd_style(),
                ),
            ]),
            html.Div([
                html.Div('Variable expliquée', className='kpi-label'),
                dcc.Dropdown(
                    id='ols-dep',
                    options=dep_options,
                    value='Rendement',
                    clearable=False,
                    style=_dd_style(),
                ),
            ]),
            html.Div([
                html.Div('Modèle', className='kpi-label'),
                dcc.Dropdown(
                    id='ols-model',
                    options=model_options,
                    value='simple',
                    clearable=False,
                    style=_dd_style(),
                ),
            ]),
        ]),

        # ── Résultats dynamiques ──────────────────────────────────────────────
        dcc.Loading(
            type='circle', color='#1f6feb',
            children=html.Div(id='ols-results'),
        ),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════
def register_callbacks(app, data: dict):

    @app.callback(
        Output('ols-results',  'children'),
        Input('ols-period',    'value'),
        Input('ols-dep',       'value'),
        Input('ols-model',     'value'),
        Input('ols-dataset-local', 'data'),
    )
    def update_ols(period, dep_choice, model_type, dataset):
        dataset = dataset or 'mq'
    
        dataset_config = {
            'mq': dict(
                valid=data['valid_mq'], prices=data['prices_mq'],
                score_col='score', secteur_col='macro_sector',
                quintile_col='quintile', pct_col='score_percentile',
                total_panel=len(data['df_mq']),
            ),
            'act': dict(
                valid=data['valid_act'], prices=data['prices_act'],
                score_col=data['col_score_act'], secteur_col=data['col_secteur_act'],
                quintile_col='quintile', pct_col='score_percentile',
                total_panel=len(data['df_act']),
            ),
            'ca': dict(
                valid=data['valid_ca'], prices=data['prices_ca'],
                score_col=data['col_score_ca'], secteur_col=data['col_secteur_ca'],
                quintile_col=data['col_quintile_ca'], pct_col=data['col_pct_ca'],
                total_panel=len(data['df_ca']),
            ),
            'cp': dict(
                valid=data['valid_cp'], prices=data['prices_cp'],
                score_col=data['col_score_cp'], secteur_col=data['col_secteur_cp'],
                quintile_col=data['col_quintile_cp'], pct_col=data['col_pct_cp'],
                total_panel=len(data['df_cp']),
            ),
            'defis': dict(
                valid=data['valid_defis'], prices=data['prices_defis'],
                score_col=data['col_score_defis'], secteur_col=data['col_secteur_defis'],
                quintile_col=data['col_quintile_defis'], pct_col=data['col_pct_defis'],
                total_panel=len(data['df_defis']),
            ),
        }
    
        cfg = dataset_config.get(dataset, dataset_config['mq'])
        valid        = cfg['valid']
        prices       = cfg['prices']
        score_col    = cfg['score_col']
        secteur_col  = cfg['secteur_col']
        quintile_col = cfg['quintile_col']
        pct_col      = cfg['pct_col']
        total_panel  = cfg['total_panel']
    
        dep_var = f'{dep_choice}_{period}'

        # ── NOUVEAU : Momentum calculé à la volée sur les prix déjà en mémoire ──
        # Mapping approximatif période → date de référence pour le momentum
        # (à ajuster selon vos conventions de PERIODS_LABELS)
        PERIOD_TO_DATE_REF = {
            '2023': pd.Timestamp('2023-01-01'),
            '2024': pd.Timestamp('2024-01-01'),
            '2025': pd.Timestamp('2025-01-01'),
            '2026': pd.Timestamp('2026-01-01'),
            '2023-2026': pd.Timestamp('2023-01-01'),
        }
        
        if model_type == 'momentum' and 'ticker' in valid.columns:
            date_ref = PERIOD_TO_DATE_REF.get(period)
            mom_dict = calc_momentum(prices, valid['ticker'].unique(), date_ref=date_ref)
            valid['Momentum'] = valid['ticker'].map(mom_dict)

        # ── Préparation données ───────────────────────────────────────────────
        cols_needed = [score_col, secteur_col, dep_var, 'LogMarketCap', 'BookToMarket', quintile_col]
        if model_type == 'momentum':
            cols_needed.append('Momentum')
    
        df_ols = valid[[c for c in cols_needed if c in valid.columns]].dropna().copy()
        df_ols = prepare_ols_data(df_ols, score_col, secteur_col, momentum_col='Momentum' if model_type == 'momentum' else None)
        df_ols[dep_var] = winsorize(df_ols[dep_var])

        n_ols      = len(df_ols)
        n_secteurs = df_ols[secteur_col].nunique()
    


        # ── Sécurité échantillon ──────────────────────────────────────────────
        if n_ols < (n_secteurs + 15):
            return html.Div(
                f"Données insuffisantes pour {PERIODS_LABELS.get(period, period)} "
                f"({n_ols} observations, {n_secteurs} secteurs). "
                "Choisissez une période plus longue.",
                className='warn-box',
            )

        # ── Régression ────────────────────────────────────────────────────────
        model = run_ols(df_ols, dep_var, secteur_col, model_type)
        if model is None:
            return html.Div(
                "Le calcul a échoué (instabilité numérique). "
                "Essayez le modèle Simple.",
                className='warn-box',
            )
        
        coef = model.params.get('Score_std', 0)
        pval = model.pvalues.get('Score_std', 1)
        
        # ── NOUVEAU : coefficient momentum si le modèle l'inclut ──────────────────
        coef_mom = model.params.get('Momentum_std')
        pval_mom = model.pvalues.get('Momentum_std')

        metric_cards_children = [
            _metric_card('Coef. Score (std)', f'{coef:+.3f}', pval < 0.05),
            _metric_card('p-value',           f'{pval:.3f} {sig_stars(pval)}', pval < 0.05),
            _metric_card('R² ajusté',         f'{model.rsquared_adj:.3f}', True),
            _metric_card('Observations',      f'{int(model.nobs)} / {total_panel}', True),
        ]
        if coef_mom is not None:
            metric_cards_children.append(
                _metric_card('Coef. Momentum (std)', f'{coef_mom:+.3f}', pval_mom < 0.05)
            )
        
        metric_cards = html.Div(
            className='row-4' if coef_mom is None else 'row-5',
            children=metric_cards_children,
        )

        # Note contexte
        context_note = html.Div(
            className='note-box', style={'marginTop': '12px'},
            children=[
                f"Régression sur {n_ols} entreprises "
                f"({n_ols/total_panel:.0%} du panel initial) · "
                "OLS avec erreurs robustes HC3 · Score standardisé (Z-score) · "
                f"Secteurs regroupés (N < 8).",
            ],
        )

        # Badge période prédictive
        period_badge = html.Div()
        if period == '2025':
            period_badge = html.Div(
                "✓ Test prédictif valide — le score était publié avant cette période.",
                className='note-box',
                style={'marginTop': '8px', 'borderColor': '#3fb950',
                       'background': '#052e16', 'color': '#3fb950'},
            )
        elif period in ('2023', '2024'):
            period_badge = html.Div(
                "⚠ Valeur descriptive uniquement — look-ahead bias : "
                "le score était publié après cette période.",
                className='warn-box',
                style={'marginTop': '8px'},
            )

        # ── Graphique interaction secteurs ─────────────────────────────────
        coef_secteurs_block = html.Div()
        if model_type == 'interaction':
            data_sect = []
            for s in sorted(df_ols[secteur_col].unique()):
                p_g = model.params.get('Score_std', 0)
                key = f'Score_std:C({secteur_col})[T.{s}]'
                if key in model.params:
                    p_g += model.params[key]
                data_sect.append({'Secteur': s, 'Modèle': 'Coefficient', 'Coefficient': p_g})

            coef_secteurs_block = html.Div([
                html.Div('Effet du score par secteur (pentes)', className='section-title'),
                html.Div(className='card', children=[
                    html.Div(className='card-body', children=[
                        dcc.Graph(
                            figure=plot_coefficients_secteurs(data_sect),
                            config={'displayModeBar': False},
                        ),
                    ]),
                ]),
            ])

        # ── Scatter Score vs variable ─────────────────────────────────────────
        dep_labels = {
            'Rendement': 'Rendement', 'Volatilite': 'Volatilité',
            'Sharpe': 'Sharpe', 'MaxDrawdown': 'Max Drawdown',
        }
        scatter = html.Div([
            html.Div(
                f'Score standardisé vs {dep_labels.get(dep_choice, dep_choice)} '
                f'· {PERIODS_LABELS.get(period, period)}',
                className='section-title',
            ),
            html.Div(className='card', children=[
                html.Div(className='card-body', children=[
                    dcc.Graph(
                        figure=plot_scatter_ols(
                            df_ols, dep_var, quintile_col,
                            y_label=dep_labels.get(dep_choice, dep_choice),
                        ),
                        config={'displayModeBar': False},
                    ),
                ]),
            ]),
        ])

        # ── Tableau coefficients secteur ──────────────────────────────────────
        coef_table = _build_coef_table(model, secteur_col)

        # ── Summary OLS ───────────────────────────────────────────────────────
        summary_block = html.Details([
            html.Summary(
                'Consulter le rapport statistique complet (OLS Summary)',
                style={'cursor': 'pointer', 'color': '#58a6ff',
                       'fontSize': '12px', 'padding': '10px 0'},
            ),
            html.Pre(
                model.summary().as_text(),
                style={'fontSize': '10px', 'color': '#8b949e',
                       'overflowX': 'auto', 'whiteSpace': 'pre',
                       'marginTop': '10px'},
            ),
        ])

        return html.Div([
            metric_cards,
            context_note,
            period_badge,
            html.Div(style={'marginTop': '16px'}, children=[coef_secteurs_block]),
            html.Div(style={'marginTop': '14px'}, children=[scatter]),
            html.Div(style={'marginTop': '14px'}, children=[coef_table]),
            html.Div(style={'marginTop': '14px'}, children=[summary_block]),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _build_coef_table(model, secteur_col):
    """Tableau des coefficients du modèle avec p-values et étoiles."""
    rows = []
    for param, coef in model.params.items():
        pval  = model.pvalues.get(param, 1)
        stars = sig_stars(pval)
        # Nettoyage du nom de paramètre
        label = (param
                 .replace(f'C({secteur_col})[T.', '')
                 .replace(']', '')
                 .replace('Score_std', 'Score (std)')
                 .replace('Momentum_std', 'Momentum (std)') 
                 .replace('Intercept', 'Constante')
                 .replace('LogMarketCap', 'Log(MarketCap)')
                 .replace('BookToMarket', 'Book/Market'))
        rows.append({
            'Paramètre':  label,
            'Coef.':      f'{coef:+.4f}',
            'p-value':    f'{pval:.3f}',
            'Sig.':       stars,
        })

    return html.Div([
        html.Div('Coefficients du modèle', className='section-title'),
        html.Div(className='card', children=[
            dash_table.DataTable(
                data=rows,
                columns=[{'name': c, 'id': c} for c in rows[0].keys()] if rows else [],
                style_table={'overflowX': 'auto', 'maxHeight': '320px',
                             'overflowY': 'auto'},
                style_cell={
                    'backgroundColor': '#0d1117',
                    'color':           '#c9d1d9',
                    'fontSize':        '12px',
                    'fontFamily':      'IBM Plex Mono, monospace',
                    'border':          '0.5px solid #21262d',
                    'padding':         '6px 14px',
                    'textAlign':       'right',
                },
                style_header={
                    'backgroundColor': '#161b22',
                    'color':           '#6e7681',
                    'fontSize':        '10px',
                    'fontWeight':      '400',
                    'textTransform':   'uppercase',
                    'letterSpacing':   '0.07em',
                    'border':          '0.5px solid #21262d',
                    'textAlign':       'left',
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Paramètre'},
                        'textAlign':  'left',
                        'fontFamily': 'IBM Plex Sans, sans-serif',
                        'color':      '#8b949e',
                    },
                    {
                        'if': {'filter_query': '{Sig.} = "★★★"'},
                        'color': '#3fb950',
                    },
                    {
                        'if': {'filter_query': '{Sig.} = "★★"'},
                        'color': '#56d364',
                    },
                    {
                        'if': {'filter_query': '{Sig.} = "★"'},
                        'color': '#d29922',
                    },
                ],
            ),
        ]),
    ])


def _metric_card(label, value, ok=True):
    color = '#3fb950' if ok else '#f85149'
    return html.Div(className='metric-card', children=[
        html.Div(label, className='metric-label'),
        html.Div(value, className='metric-value', style={'color': color}),
    ])


def _dd_style():
    return {
        'backgroundColor': '#161b22',
        'color':           '#e6edf3',
        'border':          '0.5px solid #30363d',
        'borderRadius':    '6px',
        'fontSize':        '12px',
    }

