"""Acrescenta ao painel variáveis mensais da safra (chuva, umidade do solo, Tmax) ponderadas pela produção."""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run relative to the repository root
_os.makedirs('resultados', exist_ok=True); _os.makedirs('dados/power', exist_ok=True)
import numpy as np, pandas as pd
C = pd.read_csv('dados/clima_celulas_mensal.csv'); cel = pd.read_csv('dados/celulas_clima.csv')
w = pd.read_csv('dados/pesos_municipais.csv', dtype={'cod': str, 'uf': str}); w['clat'] = np.floor(w.lat) + 0.5; w['clon'] = np.floor(w.lon) + 0.5
p = pd.read_csv('resultados/painel_uf_ano.csv')
out = []
for cul, nome in [('Soja (em grão)', 'soja'), ('Milho (em grão)', 'milho')]:
    q = w.groupby(['uf', 'clat', 'clon'])[cul].sum().reset_index().merge(cel[['clat', 'clon']], on=['clat', 'clon'])
    q['w'] = q[cul] / q.groupby('uf')[cul].transform('sum'); q = q[q.w > 0]
    m = C.merge(q[['uf', 'clat', 'clon', 'w']], on=['clat', 'clon'])
    for v in ['PRECTOTCORR', 'GWETROOT', 'T2M_MAX']: m[v] = m[v] * m.w
    a = m.groupby(['uf', 'ano', 'mes'])[['PRECTOTCORR', 'GWETROOT', 'T2M_MAX']].sum().reset_index()
    a['uf'] = a.uf.astype(int)
    for (uf, ano), _ in p[p.cultura == nome].groupby(['cod_uf', 'ano']):
        sul = uf in (42, 43)
        meses = ([(ano - 1, k) for k in (10, 11, 12)] + [(ano, k) for k in (1, 2, 3)]) if (nome == 'soja' or sul) else [(ano, k) for k in (1, 2, 3, 4, 5, 6)]
        row = dict(cod_uf=uf, ano=ano, cultura=nome)
        sub = a[a.uf == uf].set_index(['ano', 'mes'])
        for i, (y, mm) in enumerate(meses):
            if (y, mm) in sub.index:
                r = sub.loc[(y, mm)]; row[f'p{i}'] = r.PRECTOTCORR; row[f'g{i}'] = r.GWETROOT; row[f't{i}'] = r.T2M_MAX
        out.append(row)
M = pd.DataFrame(out); p = p.merge(M, on=['cod_uf', 'ano', 'cultura'], how='left'); p.to_csv('resultados/painel_uf_ano.csv', index=False); print(p.shape)
