"""Phenological-window analysis for the Field Crops Research version.

Uses the existing state panel (painel_uf_ano.csv: IBGE yields, production-weighted NASA POWER monthly
rainfall p0..p5, root-zone soil wetness g0..g5 and Tmax t0..t5) and the CONAB series that separate first-
and second-crop maize (dados_conab/conab_*_uf_ano.csv).

Monthly columns: soybean and maize in RS/SC cover Oct(t-1)..Mar(t); maize elsewhere covers Jan..Jun.

Outputs (resultados/):
  F1_resposta_por_fase.csv      slopes (log-points per SD) of the yield anomaly on rainfall, soil wetness and Tmax
                                of each phenological window, pooled by crop group, block-bootstrap 95% CI
  F2_enso_para_clima_fase.csv   slope of each standardized phase variable on the ONI (where ENSO enters the season)
  F3_skill_por_conjunto.csv     LOYO skill (ridge, RF) with season means / phase windows / monthly predictors
  F4_conab_enso.csv             ONI slopes for CONAB first- and second-crop maize vs IBGE total maize, by state
  F5_conab_fase.csv             phase slopes for CONAB second-crop maize (MT, PR, MS, GO, SP, MG) and
                                first-crop maize (RS, SC)
  fase_fenologica.log
"""
import os, sys, numpy as np, pandas as pd, warnings
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.makedirs('resultados', exist_ok=True)
LOG = open('resultados/fase_fenologica.log', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
rng = np.random.default_rng(0)
PARTES = os.environ.get('PARTES', 'F1,F2,F3,F4,F5').split(',')

# ---------------------------------------------------------------- data
p = pd.read_csv('dados_derivados/painel_uf_ano.csv')
p = p[(p.ano >= 1990) & (p.ano <= 2025)].copy()
p['ly'] = np.log(p.rend_kgha)
SUL = {42, 43}                     # SC, RS: first-crop maize, Oct–Mar window
SOJA_B = {41, 42, 43}              # PR, SC, RS: later soybean calendar
UF_SIGLA = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB',
            26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC', 43: 'RS',
            50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'}
p['sigla'] = p.cod_uf.map(UF_SIGLA)

def grupo(r):
    if r.cultura == 'soja':
        return 'soybean South' if r.cod_uf in SOJA_B else 'soybean Centre-North'
    return 'maize first crop (South)' if r.cod_uf in SUL else 'maize second crop'
p['grupo'] = p.apply(grupo, axis=1)

# phase windows as month-index pairs into p0..p5 (inclusive)
# soybean Centre-North (Oct..Mar): establishment Oct–Nov, flowering Dec, grain filling Jan–Feb   (Mar = harvest)
# soybean South        (Oct..Mar): establishment Oct–Nov, flowering Dec–Jan, grain filling Feb–Mar
# maize second crop    (Jan..Jun): establishment Jan–Feb, flowering Mar–Apr, grain filling May–Jun
# maize first crop S   (Oct..Mar): establishment Oct–Nov, flowering Dec–Jan, grain filling Feb–Mar
JANELAS = {
    'soybean Centre-North':     {'est': (0, 1), 'flo': (2, 2), 'fill': (3, 4)},
    'soybean South':            {'est': (0, 1), 'flo': (2, 3), 'fill': (4, 5)},
    'maize second crop':        {'est': (0, 1), 'flo': (2, 3), 'fill': (4, 5)},
    'maize first crop (South)': {'est': (0, 1), 'flo': (2, 3), 'fill': (4, 5)},
}
MESES = {'soybean Centre-North': ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'], 'soybean South': ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'],
         'maize second crop': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 'maize first crop (South)': ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']}
FASES = ['est', 'flo', 'fill']

def fase_vars(d):
    d = d.copy()
    for g, jan in JANELAS.items():
        m = d.grupo == g
        for f, (a, b) in jan.items():
            cols = list(range(a, b + 1))
            d.loc[m, f'P_{f}'] = d.loc[m, [f'p{i}' for i in cols]].sum(1) * 30.4 * len(cols) / len(cols)  # mm/day * ~days -> mm (sum of monthly means*30.4)
            d.loc[m, f'G_{f}'] = d.loc[m, [f'g{i}' for i in cols]].mean(1)
            d.loc[m, f'T_{f}'] = d.loc[m, [f't{i}' for i in cols]].mean(1)
    return d
p = fase_vars(p)
PH = [f'{v}_{f}' for f in FASES for v in 'PGT']
SEAS = ['prec', 'gwet', 'tmax']
SEAS_ALL = ['prec', 'tmax', 'tmed', 'rh', 'gwet', 'rad']
MON = [f'{v}{i}' for v in 'pgt' for i in range(6)]
for v in PH + SEAS_ALL + MON:
    g = p.groupby(['cod_uf', 'cultura'])[v]
    p[v + '_z'] = (p[v] - g.transform('mean')) / (g.transform('std') + 1e-9)
p['oni'] = np.where(p.cultura == 'soja', p.oni_ndj_prev, np.where(p.cod_uf.isin(SUL), p.oni_ndj_prev, p.oni_mam))
p['oni_ndj'] = p.oni_ndj_prev

# ---------------------------------------------------------------- helpers
def anomalia(d, anos_fit=None):
    """log-yield anomaly from a quadratic trend per state×crop, fitted on anos_fit (default: all rows)."""
    out = pd.Series(index=d.index, dtype=float)
    for k, g in d.groupby(['cod_uf', 'cultura']):
        gf = g if anos_fit is None else g[g.ano.isin(anos_fit)]
        if len(gf) < 8: continue
        z = np.polyfit(gf.ano - 2000, gf.ly, 2); out[g.index] = g.ly - np.polyval(z, g.ano - 2000)
    return out

def ols(Xm, y):
    Xm = np.column_stack([np.ones(len(y)), Xm]); b, *_ = np.linalg.lstsq(Xm, y, rcond=None); return b[1:]

def block_boot(d, cols, y_fn, B=2000, bloco=3):
    """Pooled OLS of the anomaly on cols with a moving-block bootstrap of years (trend refitted per replicate)."""
    anos = np.array(sorted(d.ano.unique())); n = len(anos)
    d = d.dropna(subset=cols).copy()
    y0 = y_fn(d); m0 = y0.notna(); b0 = ols(d.loc[m0, cols].values, y0[m0].values)
    bs = []
    for _ in range(B):
        starts = rng.integers(0, n - bloco + 1, size=int(np.ceil(n / bloco)))
        sel = np.concatenate([anos[s:s + bloco] for s in starts])[:n]
        # resample years with replacement: stack the rows of the chosen years
        dd = pd.concat([d[d.ano == a] for a in sel], ignore_index=True)
        yy = y_fn(dd); mm = yy.notna()
        if mm.sum() < len(cols) + 5: continue
        bs.append(ols(dd.loc[mm, cols].values, yy[mm].values))
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    pv = 2 * np.minimum((bs <= 0).mean(0), (bs >= 0).mean(0))
    return b0, lo, hi, pv, int(m0.sum())

def tendencia_anom(dd):
    return anomalia(dd)

# ---------------------------------------------------------------- F1: response by phase
grandes = p[p.ano == 2025].query('prod_t > 2e5')[['cod_uf', 'cultura']]
pg = p.merge(grandes, on=['cod_uf', 'cultura'])
if 'F1' in PARTES:
    log('== F1: yield anomaly on phase variables (pooled by group, states with 2025 production > 0.2 Mt) ==')
    grandes = p[p.ano == 2025].query('prod_t > 2e5')[['cod_uf', 'cultura']]
    pg = p.merge(grandes, on=['cod_uf', 'cultura'])
    rows = []
    for g, d in pg.groupby('grupo'):
        cols = [c + '_z' for c in PH]
        b, lo, hi, pv, n = block_boot(d, cols, tendencia_anom)
        for c, bb, l, h, q in zip(cols, b, lo, hi, pv):
            v, f = c.replace('_z', '').split('_')
            rows.append(dict(group=g, phase=f, variable={'P': 'rainfall', 'G': 'soil wetness', 'T': 'Tmax'}[v], slope_lp=100 * bb, lo=100 * l, hi=100 * h, p=q, n=n,
                             states=d.sigla.nunique()))
        log(g, 'n =', n, 'states =', d.sigla.nunique())
    F1 = pd.DataFrame(rows); F1.to_csv('resultados/F1_resposta_por_fase.csv', index=False)
    log(F1.round(2).to_string(index=False))

# ---------------------------------------------------------------- F2: where ENSO enters the season
if 'F2' in PARTES:
    log('\n== F2: standardized phase variable on ONI (NDJ) by group ==')
    rows = []
    for g, d in pg.groupby('grupo'):
        for c in PH:
            dd = d.dropna(subset=[c + '_z', 'oni_ndj'])
            b, lo, hi, pv, n = block_boot(dd, ['oni_ndj'], lambda x: x[c + '_z'], B=1000)
            v, f = c.split('_')
            rows.append(dict(group=g, phase=f, variable={'P': 'rainfall', 'G': 'soil wetness', 'T': 'Tmax'}[v], sd_per_degC=b[0], lo=lo[0], hi=hi[0], p=pv[0], n=n))
    F2 = pd.DataFrame(rows); F2.to_csv('resultados/F2_enso_para_clima_fase.csv', index=False)
    log(F2.round(2).to_string(index=False))

# ---------------------------------------------------------------- F3: LOYO skill by predictor set
if 'F3' in PARTES:
    log('\n== F3: leave-one-year-out skill by predictor set ==')
    w = pd.read_csv('dados_derivados/pesos_municipais.csv', dtype={'uf': str})
    w['g'] = w[['Soja (em grão)', 'Milho (em grão)']].sum(1)
    cen = w.groupby('uf').apply(lambda g: pd.Series({'lat': np.average(g.lat.fillna(g.lat.mean()), weights=g.g + 1e-9),
                                                     'lon': np.average(g.lon.fillna(g.lon.mean()), weights=g.g + 1e-9)})).reset_index()
    cen['cod_uf'] = cen.uf.astype(int)
    pm = p.merge(cen[['cod_uf', 'lat', 'lon']], on='cod_uf')
    pm = pm.merge(grandes, on=['cod_uf', 'cultura']).dropna(subset=[c + '_z' for c in MON]).copy()
    SETS = {'season means (3)': [c + '_z' for c in SEAS], 'phase windows (9)': [c + '_z' for c in PH], 'monthly values (18)': [c + '_z' for c in MON],
            'phase windows + ONI': [c + '_z' for c in PH] + ['oni'],
        'full set as submitted (6 season means + 18 monthly + ONI)': [c + '_z' for c in SEAS_ALL] + [c + '_z' for c in MON] + ['oni']}
    def Xmat(d, cols):
        Z = d[cols].copy(); Z['soja'] = (d.cultura == 'soja').astype(int); Z['lat'] = d.lat; Z['lon'] = d.lon
        for c in cols[:3]: Z[c + '_x_lat'] = d[c] * d.lat
        return Z
    def tendencia(tr, te):
        fit = pd.Series(index=tr.index, dtype=float); out = pd.Series(index=te.index, dtype=float)
        for k, g in tr.groupby(['cod_uf', 'cultura']):
            z = np.polyfit(g.ano - 2000, g.ly, 2); fit[g.index] = np.polyval(z, g.ano - 2000)
            m = (te.cod_uf == k[0]) & (te.cultura == k[1]); out[te.index[m]] = np.polyval(z, te.ano[m] - 2000)
        return fit, out
    preds = {(s, m): pd.Series(index=pm.index, dtype=float) for s in SETS for m in ['ridge', 'rf']}
    ptrend = pd.Series(index=pm.index, dtype=float)
    for ano in sorted(pm.ano.unique()):
        tr, te = pm[pm.ano != ano], pm[pm.ano == ano]
        ftr, fte = tendencia(tr, te); ytr = tr.ly - ftr; ptrend[te.index] = fte
        for s, cols in SETS.items():
            Xtr, Xte = Xmat(tr, cols), Xmat(te, cols); mu, sd = Xtr.mean(), Xtr.std() + 1e-9
            r = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((Xtr - mu) / sd, ytr); preds[(s, 'ridge')][te.index] = fte + r.predict((Xte - mu) / sd)
            rf = RandomForestRegressor(500, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=2).fit(Xtr, ytr)
            preds[(s, 'rf')][te.index] = fte + rf.predict(Xte)
        print(ano, end=' ', flush=True)
    print()
    rows = []
    e0 = pm.ly - ptrend
    anos_u = np.array(sorted(pm.ano.unique()))
    for (s, m), pr in preds.items():
        e = pm.ly - pr
        for c in ['soja', 'milho']:
            k = pm.cultura == c
            sk = 1 - (e[k] ** 2).sum() / (e0[k] ** 2).sum()
            # bootstrap of years for CI
            bs = []
            for _ in range(1000):
                sel = rng.choice(anos_u, size=len(anos_u), replace=True)
                kk = pd.concat([pm.index[k & (pm.ano == a)].to_series() for a in sel])
                bs.append(1 - (e.loc[kk] ** 2).sum() / (e0.loc[kk] ** 2).sum())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(predictor_set=s, model=m, crop={'soja': 'soybean', 'milho': 'maize'}[c], skill=sk, lo=lo, hi=hi, rmse=np.sqrt((e[k] ** 2).mean())))
    F3 = pd.DataFrame(rows); F3.to_csv('resultados/F3_skill_por_conjunto.csv', index=False)
    log(F3.round(3).to_string(index=False))

# ---------------------------------------------------------------- F4/F5: CONAB first- and second-crop maize
if 'F4' in PARTES:
    log('\n== F4: ONI slope (% per °C) for CONAB second-crop / first-crop maize vs IBGE total maize ==')
    SIG2COD = {v: k for k, v in UF_SIGLA.items()}
    def carrega(k):
        c = pd.read_csv(f'dados_conab/conab_{k}_uf_ano.csv'); c = c[(c.ano >= 1990) & (c.ano <= 2025) & c.rend.notna() & (c.rend > 0)].copy()
        c['cod_uf'] = c.uf.map(SIG2COD); c['ly'] = np.log(c.rend); c['sigla'] = c.uf; return c
    c2, c1 = carrega('milho2'), carrega('milho1')
    clima_m = p[p.cultura == 'milho'][['cod_uf', 'ano', 'oni_ndj_prev', 'oni_mam'] + [c + '_z' for c in PH] + [c for c in PH]]
    def oni_slopes(c, nome, oni_col, min_n=25):
        rows = []
        for uf, g in c.groupby('sigla'):
            if len(g) < min_n: continue
            oni_ano = p[['ano', 'oni_ndj_prev', 'oni_mam']].drop_duplicates('ano')
            g = g.drop(columns=[c for c in ['oni_ndj_prev', 'oni_mam'] if c in g.columns]).merge(oni_ano, on='ano'); g['cultura'] = nome
            if len(g) < min_n: continue
            b, lo, hi, pv, n = block_boot(g, [oni_col], tendencia_anom, B=1000)
            rows.append(dict(series=nome, state=uf, slope_pct_per_degC=100 * (np.exp(b[0]) - 1), lo=100 * (np.exp(lo[0]) - 1), hi=100 * (np.exp(hi[0]) - 1), p=pv[0], n=n))
        return rows
    rows = oni_slopes(c2, 'CONAB second crop', 'oni_ndj_prev') + oni_slopes(c1, 'CONAB first crop', 'oni_ndj_prev')
    ib = p[(p.cultura == 'milho')].copy(); ib['sigla'] = ib.cod_uf.map(UF_SIGLA)
    rows += oni_slopes(ib[ib.sigla.isin(c2.sigla.unique()) | ib.sigla.isin(c1.sigla.unique())].assign(rend=lambda d: d.rend_kgha), 'IBGE total maize', 'oni_ndj_prev')
    F4 = pd.DataFrame(rows); F4.to_csv('resultados/F4_conab_enso.csv', index=False)
    log(F4.round(2).to_string(index=False))

if 'F5' in PARTES:
    log('\n== F5: phase slopes for CONAB second-crop maize (MT, PR, MS, GO, SP, MG) and first-crop maize (RS, SC) ==')
    rows = []
    d2 = c2[c2.sigla.isin(['MT', 'PR', 'MS', 'GO', 'SP', 'MG'])].merge(clima_m, on=['cod_uf', 'ano']); d2['cultura'] = 'milho2'
    d1 = c1[c1.sigla.isin(['RS', 'SC'])].merge(clima_m, on=['cod_uf', 'ano']); d1['cultura'] = 'milho1'
    for nome, d in [('CONAB second crop (MT, PR, MS, GO, SP, MG)', d2), ('CONAB first crop (RS, SC)', d1)]:
        for fam, v in {'rainfall': 'P', 'soil wetness': 'G', 'Tmax': 'T'}.items():
            cols = [f'{v}_{f}_z' for f in FASES]
            b, lo, hi, pv, n = block_boot(d, cols, tendencia_anom)
            for f, bb, l, h, q in zip(FASES, b, lo, hi, pv):
                rows.append(dict(series=nome, family=fam, phase=f, slope_lp=100 * bb, lo=100 * l, hi=100 * h, p=q, n=n))
        log(nome, 'n =', n)
    F5 = pd.DataFrame(rows); F5.to_csv('resultados/F5_conab_fase.csv', index=False)
    log(F5.round(2).to_string(index=False))

    # share of second crop in the six states and nationally (CONAB), 2025
    tot = c1.groupby('ano')['prod'].sum() + c2.groupby('ano')['prod'].sum()
    log('\nShare of second crop in national maize (CONAB):', {a: round(c2[c2.ano == a]['prod'].sum() / tot[a], 2) for a in [1995, 2005, 2015, 2025]})
    log('Share of the six states in second-crop production 2025:', round(d2[d2.ano == 2025]['prod'].sum() / c2[c2.ano == 2025]['prod'].sum(), 3))
LOG.close()
