"""Painel UF × ano: produtividade (IBGE/PAM), clima da safra ponderado pela produção (NASA POWER) e ENSO (ONI)."""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run relative to the repository root
_os.makedirs('resultados', exist_ok=True); _os.makedirs('dados/power', exist_ok=True)
import json, glob, numpy as np, pandas as pd
cel = pd.read_csv('dados/celulas_clima.csv')
w = pd.read_csv('dados/pesos_municipais.csv', dtype={'cod': str, 'uf': str})
gr = ['Soja (em grão)', 'Milho (em grão)', 'Arroz (em casca)', 'Feijão (em grão)', 'Trigo (em grão)']
w['clat'] = np.floor(w.lat) + 0.5; w['clon'] = np.floor(w.lon) + 0.5
# clima mensal por célula
reg = []
for f in glob.glob('dados/power/c_*.json'):
    d = json.load(open(f)); la, lo = d['geometry']['coordinates'][1], d['geometry']['coordinates'][0]
    P = d['properties']['parameter']
    df = pd.DataFrame(P); df = df[~df.index.str.endswith('13')]
    df['ano'] = df.index.str[:4].astype(int); df['mes'] = df.index.str[4:].astype(int)
    df['clat'] = round(la, 1); df['clon'] = round(lo, 1); reg.append(df)
C = pd.concat(reg).replace(-999, np.nan)
C.to_csv('dados/clima_celulas_mensal.csv', index=False)
pes = {}
for cul in ['Soja (em grão)', 'Milho (em grão)']:
    p = w.groupby(['uf', 'clat', 'clon'])[cul].sum().reset_index()
    p = p.merge(cel[['clat', 'clon']], on=['clat', 'clon'])
    p['w'] = p[cul] / p.groupby('uf')[cul].transform('sum'); pes[cul] = p[p.w > 0]
# estação de crescimento: soja out(t-1)–mar(t); milho (total, dominado pela 2ª safra) fev–jun(t)
def sazonal(cul, meses_prev, meses_ano):
    p = pes[cul]
    out = []
    for ano in range(1982, 2026):
        sel = C[((C.ano == ano - 1) & C.mes.isin(meses_prev)) | ((C.ano == ano) & C.mes.isin(meses_ano))]
        m = sel.groupby(['clat', 'clon']).agg(prec=('PRECTOTCORR', 'mean'), tmax=('T2M_MAX', 'mean'), tmed=('T2M', 'mean'),
                                              rh=('RH2M', 'mean'), gwet=('GWETROOT', 'mean'), rad=('ALLSKY_SFC_SW_DWN', 'mean')).reset_index()
        q = p.merge(m, on=['clat', 'clon'])
        for v in ['prec', 'tmax', 'tmed', 'rh', 'gwet', 'rad']:
            q[v] = q[v] * q.w
        a = q.groupby('uf')[['prec', 'tmax', 'tmed', 'rh', 'gwet', 'rad']].sum().reset_index(); a['ano'] = ano; out.append(a)
    return pd.concat(out)
cs = sazonal('Soja (em grão)', [10, 11, 12], [1, 2, 3]); cs['cultura'] = 'soja'
cm2 = sazonal('Milho (em grão)', [], [2, 3, 4, 5, 6]); cm1 = sazonal('Milho (em grão)', [10, 11, 12], [1, 2])
sul = ['43', '42']          # RS e SC: milho de 1ª safra (out–fev); demais UF: 2ª safra dominante (fev–jun)
cm = pd.concat([cm1[cm1.uf.astype(str).isin(sul)], cm2[~cm2.uf.astype(str).isin(sul)]]); cm['cultura'] = 'milho'
clima = pd.concat([cs, cm])
pam = pd.read_csv('dados/pam_uf_1974_2025.csv', dtype={'D1C': str})
pam = pam[pam.cultura.isin(['Soja (em grão)', 'Milho (em grão)'])]
pv = pam.pivot_table(index=['D1C', 'uf', 'ano', 'cultura'], columns='variavel', values='valor').reset_index()
pv['cultura'] = pv.cultura.map({'Soja (em grão)': 'soja', 'Milho (em grão)': 'milho'})
pv = pv.rename(columns={'D1C': 'cod_uf', 'Quantidade produzida': 'prod_t', 'Área colhida': 'area_ha', 'Rendimento médio da produção': 'rend_kgha', 'Área plantada': 'area_plant_ha'})
oni = pd.read_csv('dados/oni_1980_2026.csv')
o = oni.pivot(index='ano', columns='estacao', values='oni')
enso = pd.DataFrame({'ano': o.index, 'oni_son_prev': o['SON'].shift(1), 'oni_ndj_prev': o['NDJ'].shift(1), 'oni_djf': o['DJF'], 'oni_mam': o['MAM']}).reset_index(drop=True)
painel = pv.merge(clima, left_on=['cod_uf', 'ano', 'cultura'], right_on=['uf', 'ano', 'cultura'], suffixes=('', '_c')).drop(columns='uf_c').merge(enso, on='ano')
painel.to_csv('resultados/painel_uf_ano.csv', index=False)
print(painel.shape, painel.groupby('cultura').cod_uf.nunique().to_dict(), painel.ano.min(), painel.ano.max())
