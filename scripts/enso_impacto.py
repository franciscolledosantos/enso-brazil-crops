"""Efeito do ENSO na produtividade por UF e cenário 2026/27.
(1) anomalia log da produtividade (tendência quadrática por UF, 1990–2025); (2) inclinação no ONI por UF com
bootstrap por blocos de anos; (3) análogos: média das anomalias nos El Niño fortes; (4) conversão em toneladas com a
área e a tendência de 2025; (5) peso no mercado mundial (USDA PSD 2026/27)."""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run relative to the repository root
_os.makedirs('resultados', exist_ok=True); _os.makedirs('dados/power', exist_ok=True)
import numpy as np, pandas as pd
p = pd.read_csv('resultados/painel_uf_ano.csv'); p = p[(p.ano >= 1990)].copy()
p['oni'] = p.oni_ndj_prev   # pico do evento (NDJ antes da colheita), para soja e milho
p['ly'] = np.log(p.rend_kgha)
ult = p[p.ano == 2025].set_index(['uf', 'cultura'])
keep = ult[ult.prod_t > 5e5].index
rows, an = [], []
rng = np.random.default_rng(0)
FORTES = [1983, 1998, 2016, 2024]
for (uf, c), g in p.groupby(['uf', 'cultura']):
    if (uf, c) not in keep: continue
    g = g.sort_values('ano'); z = np.polyfit(g.ano - 2000, g.ly, 2); g = g.assign(a=g.ly - np.polyval(z, g.ano - 2000))
    b = np.polyfit(g.oni, g.a, 1)[0]
    bs = []
    anos = g.ano.values
    for k in range(2000):
        s = rng.choice(len(g), len(g)); bs.append(np.polyfit(g.oni.values[s], g.a.values[s], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    fort = g[g.ano.isin(FORTES)].a.mean()
    y25 = np.exp(np.polyval(z, 25)); area = ult.loc[(uf, c), 'area_ha']; prod = ult.loc[(uf, c), 'prod_t']
    rows.append(dict(uf=uf, cultura=c, prod_2025_Mt=prod / 1e6, beta_oni=b, lo=lo, hi=hi, anom_fortes=fort, n=len(g),
                     dprod_oni25_Mt=area * y25 * (np.exp(b * 2.5) - 1) / 1e9,
                     dprod_analogo_Mt=area * y25 * (np.exp(fort) - 1) / 1e9))
    an.append(g[['uf', 'cultura', 'ano', 'a', 'oni']])
R = pd.DataFrame(rows).sort_values(['cultura', 'prod_2025_Mt'], ascending=[False, False])
R.to_csv('resultados/enso_por_uf.csv', index=False); pd.concat(an).to_csv('resultados/anomalias_uf.csv', index=False)
print(R.round(3).to_string())
for c in ['soja', 'milho']:
    s = R[R.cultura == c]; print(c, 'Δ ONI=+2.5 (Mt):', s.dprod_oni25_Mt.sum().round(2), ' análogos (Mt):', s.dprod_analogo_Mt.sum().round(2),
          ' perdas:', s[s.dprod_oni25_Mt < 0].dprod_oni25_Mt.sum().round(2), ' ganhos:', s[s.dprod_oni25_Mt > 0].dprod_oni25_Mt.sum().round(2))
