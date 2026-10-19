"""v2 (após revisão): bootstrap por blocos móveis de 3 anos com reajuste da tendência em cada réplica; p bilateral e
correção de Benjamini–Hochberg; cenário de análogos por ano (1998, 2016, 2024) além da média; habilidade de um
modelo só com o ONI pré-safra (NDJ) e inclinação por UF, deixando um ano de fora."""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run relative to the repository root
_os.makedirs('resultados', exist_ok=True); _os.makedirs('dados/power', exist_ok=True)
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
p = pd.read_csv('resultados/painel_uf_ano.csv'); p = p[p.ano >= 1990].copy()
p['oni'] = p.oni_ndj_prev; p['ly'] = np.log(p.rend_kgha)
ult = p[p.ano == 2025].set_index(['uf', 'cultura']); keep = ult[ult.prod_t > 5e5].index
rng = np.random.default_rng(0); FORTES = [1998, 2016, 2024]
rows = []
def anomalia(g):
    z = np.polyfit(g.ano - 2000, g.ly, 2); return g.ly - np.polyval(z, g.ano - 2000), z
for (uf, c), g in p.groupby(['uf', 'cultura']):
    if (uf, c) not in keep: continue
    g = g.sort_values('ano').reset_index(drop=True); a, z = anomalia(g); b = np.polyfit(g.oni, a, 1)[0]
    n = len(g); L = 3; bs = []
    for k in range(2000):
        ini = rng.integers(0, n - L + 1, int(np.ceil(n / L))); idx = np.concatenate([np.arange(i, i + L) for i in ini])[:n]
        gg = g.iloc[idx]; aa, _ = anomalia(gg); bs.append(np.polyfit(gg.oni, aa, 1)[0])
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5]); pv = min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    y25 = np.exp(np.polyval(z, 25)); area = ult.loc[(uf, c), 'area_ha']; base = area * y25 / 1e9
    r = dict(uf=uf, cultura=c, n=n, prod_2025_Mt=ult.loc[(uf, c), 'prod_t'] / 1e6, trend_prod_2025_Mt=base, beta_oni=b, lo=lo, hi=hi, p=pv,
             dprod_oni25_Mt=base * (np.exp(b * 2.5) - 1))
    for y in FORTES:
        v = a[g.ano == y]; r[f'anom_{y}'] = float(v.iloc[0]) if len(v) else np.nan
        r[f'dprod_{y}_Mt'] = base * (np.exp(r[f'anom_{y}']) - 1) if len(v) else 0.0
    r['anom_fortes'] = np.nanmean([r[f'anom_{y}'] for y in FORTES]); r['dprod_analogo_Mt'] = base * (np.exp(r['anom_fortes']) - 1)
    rows.append(r)
R = pd.DataFrame(rows)
# Benjamini–Hochberg
o = np.argsort(R.p.values); m = len(R); q = np.empty(m); prev = 1
for rank, i in enumerate(o[::-1]):
    k = m - rank; prev = min(prev, R.p.values[i] * m / k); q[i] = prev
R['q_BH'] = q
R = R.sort_values(['cultura', 'prod_2025_Mt'], ascending=[False, False]); R.to_csv('resultados/enso_por_uf_v2.csv', index=False)
print(R[['uf', 'cultura', 'n', 'beta_oni', 'lo', 'hi', 'p', 'q_BH']].round(3).to_string())
for c in ['soja', 'milho']:
    s = R[R.cultura == c]
    print(c, 'n UF', len(s), 'base Mt', round(s.trend_prod_2025_Mt.sum(), 1), 'linear', round(s.dprod_oni25_Mt.sum(), 2),
          'perdas', round(s[s.dprod_oni25_Mt < 0].dprod_oni25_Mt.sum(), 2), 'ganhos', round(s[s.dprod_oni25_Mt > 0].dprod_oni25_Mt.sum(), 2),
          '| analogo medio', round(s.dprod_analogo_Mt.sum(), 2), '| por ano', {y: round(s[f'dprod_{y}_Mt'].sum(), 2) for y in FORTES})
# habilidade pré-safra: só ONI NDJ com inclinação por UF (ridge), deixando um ano de fora
q2 = p[p.prod_t > 2e5].copy(); out = []
for c in ['soja', 'milho']:
    d = q2[q2.cultura == c].copy(); e = []; e0 = []; fe = []; fe0 = []
    for ano in sorted(d.ano.unique()):
        tr = d[d.ano != ano]; te = d[d.ano == ano]
        if te.empty: continue
        ftr = pd.Series(index=tr.index, dtype=float); fte = pd.Series(index=te.index, dtype=float)
        for u, g in tr.groupby('uf'):
            z = np.polyfit(g.ano - 2000, g.ly, 2); ftr[g.index] = np.polyval(z, g.ano - 2000)
            mm = te.uf == u; fte[te.index[mm]] = np.polyval(z, te.ano[mm] - 2000)
        te = te[fte.notna()]; fte = fte.dropna()
        def X(dd):
            Z = pd.DataFrame({'oni': dd.oni.values}, index=dd.index)
            for u in d.uf.unique(): Z['oni_' + u] = (dd.uf == u).astype(float) * dd.oni
            return Z
        rg = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X(tr), tr.ly - ftr)
        pr = fte + rg.predict(X(te)); e += list(te.ly - pr); e0 += list(te.ly - fte)
        if ano in FORTES: fe += list(te.ly - pr); fe0 += list(te.ly - fte)
    e, e0, fe, fe0 = map(np.array, (e, e0, fe, fe0))
    out.append(dict(crop=c, n=len(e), skill_all=1 - (e ** 2).sum() / (e0 ** 2).sum(), skill_strong=1 - (fe ** 2).sum() / (fe0 ** 2).sum()))
print(pd.DataFrame(out).round(3)); pd.DataFrame(out).to_csv('resultados/skill_preseason_oni.csv', index=False)
