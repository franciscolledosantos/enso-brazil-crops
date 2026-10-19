"""Produtividade (anomalia log em relação à tendência da UF) ~ clima da safra + ENSO.
Validação: deixar um ano-safra inteiro de fora (todas as UF). A tendência quadrática por UF é ajustada só nos anos de treino.
Modelos: tendência pura (referência), ridge, efeitos mistos (inclinação do ONI por UF), LightGBM, floresta aleatória, GP."""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run relative to the repository root
_os.makedirs('resultados', exist_ok=True); _os.makedirs('dados/power', exist_ok=True)
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
warnings.filterwarnings('ignore')
p = pd.read_csv('resultados/painel_uf_ano.csv'); p = p[(p.ano >= 1990) & (p.prod_t > 2e5)].copy()
p['ly'] = np.log(p.rend_kgha)
CL = ['prec', 'tmax', 'tmed', 'rh', 'gwet', 'rad'] + [f'{v}{i}' for v in 'pgt' for i in range(6)]
for v in CL:   # anomalias padronizadas por UF e cultura (climatologia de todo o período: o clima é exógeno)
    g = p.groupby(['uf', 'cultura'])[v]; p[v + '_a'] = (p[v] - g.transform('mean')) / g.transform('std')
p['oni'] = np.where(p.cultura == 'soja', p.oni_ndj_prev, np.where(p.cod_uf.astype(str).isin(['43', '42']), p.oni_ndj_prev, p.oni_mam))
F = [v + '_a' for v in CL] + ['oni']
def tendencia(tr, te):
    out = pd.Series(index=te.index, dtype=float); fit = pd.Series(index=tr.index, dtype=float)
    for k, g in tr.groupby(['uf', 'cultura']):
        z = np.polyfit(g.ano - 2000, g.ly, 2); fit[g.index] = np.polyval(z, g.ano - 2000)
        m = (te.uf == k[0]) & (te.cultura == k[1]); out[te.index[m]] = np.polyval(z, te.ano[m] - 2000)
    return fit, out
def X(d):
    Z = d[F].copy(); Z['soja'] = (d.cultura == 'soja').astype(int); Z['lat'] = d.lat; Z['lon'] = d.lon
    for v in ['prec_a', 'tmax_a', 'gwet_a', 'oni']:
        Z[v + '_x_lat'] = d[v] * d.lat
    return Z
# coordenadas da UF (centro da produção) para interações espaciais
w = pd.read_csv('dados/pesos_municipais.csv', dtype={'uf': str})
w['g'] = w[['Soja (em grão)', 'Milho (em grão)']].sum(1)
cen = w.groupby('uf').apply(lambda g: pd.Series({'lat': np.average(g.lat.fillna(g.lat.mean()), weights=g.g + 1e-9), 'lon': np.average(g.lon.fillna(g.lon.mean()), weights=g.g + 1e-9)})).reset_index()
cen['uf'] = cen.uf.astype(int); p = p.merge(cen, left_on='cod_uf', right_on='uf', suffixes=('', '_c'))
pred = {m: pd.Series(index=p.index, dtype=float) for m in ['trend', 'ridge', 'mixed', 'lgbm', 'rf', 'gp']}
for ano in sorted(p.ano.unique()):
    tr = p[p.ano != ano]; te = p[p.ano == ano]
    ftr, fte = tendencia(tr, te); ytr = tr.ly - ftr
    Xtr, Xte = X(tr), X(te); mu, sd = Xtr.mean(), Xtr.std() + 1e-9
    pred['trend'][te.index] = fte
    r = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((Xtr - mu) / sd, ytr); pred['ridge'][te.index] = fte + r.predict((Xte - mu) / sd)
    # efeitos mistos aproximados: ridge com interações UF×(ONI, chuva, Tmax) (encolhimento para o efeito comum)
    def Xm(d):
        Z = X(d)
        for u in p.uf.unique():
            for v in ['oni', 'prec_a', 'tmax_a', 'gwet_a']:
                Z[f'{v}_{u}_{d.cultura.iloc[0] if False else ""}'] = 0
        for c in ['soja', 'milho']:
            for u in p.uf.unique():
                m = ((d.uf == u) & (d.cultura == c)).astype(float)
                for v in ['oni', 'prec_a', 'tmax_a', 'gwet_a']:
                    Z[f'{v}_{u}_{c}'] = m * d[v]
        return Z
    Mtr, Mte = Xm(tr), Xm(te); mu2, sd2 = Mtr.mean(), Mtr.std() + 1e-9
    r2 = RidgeCV(alphas=np.logspace(-1, 3, 20)).fit((Mtr - mu2) / sd2, ytr); pred['mixed'][te.index] = fte + r2.predict((Mte - mu2) / sd2)
    g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=1.0, verbose=-1).fit(Xtr, ytr)
    pred['lgbm'][te.index] = fte + g.predict(Xte)
    rf = RandomForestRegressor(500, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1).fit(Xtr, ytr)
    pred['rf'][te.index] = fte + rf.predict(Xte)
    k = ConstantKernel() * RBF(length_scale=3.0) + WhiteKernel()
    gp = GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit((Xtr - mu) / sd, ytr)
    pred['gp'][te.index] = fte + gp.predict((Xte - mu) / sd)
    print(ano, flush=True)
P = p[['uf', 'cultura', 'ano', 'ly', 'prod_t', 'area_ha', 'oni', 'rend_kgha']].copy()
for m, s in pred.items(): P['pred_' + m] = s
P.to_csv('resultados/predicoes_LOYO_v2.csv', index=False)
fortes = [1983, 1998, 2016, 2024]
rows = []
for m in pred:
    e = P.ly - P['pred_' + m]; e0 = P.ly - P.pred_trend
    for nome, msk in [('all', P.ano > 0), ('strong El Niño', P.ano.isin(fortes))]:
        for c in ['soja', 'milho']:
            k = msk & (P.cultura == c)
            rows.append(dict(model=m, subset=nome, crop=c, rmse=np.sqrt((e[k] ** 2).mean()), skill_vs_trend=1 - (e[k] ** 2).sum() / (e0[k] ** 2).sum()))
R = pd.DataFrame(rows); R.to_csv('resultados/metricas_LOYO_v2.csv', index=False); print(R.round(3).to_string())
