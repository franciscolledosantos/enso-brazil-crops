"""Análises pedidas pelo revisor (17/10/2026). Saídas em resultados/.
A. Todos os anos de El Niño (categoria pelo pico do ONI do evento) com anomalia nacional de soja e milho a áreas de 2025.
B. Regressão anomalia ~ chuva MAM (milho 2ª safra) / DJF (soja) por UF, com bootstrap de blocos; e anomalia de milho ~ ONI MAM.
C. Ablação: ridge e floresta aleatória com e sem ONI, deixando um ano de fora.
D. Importância por permutação (floresta aleatória, por grupo de variáveis), deixando um ano de fora.
E. IC por bootstrap por ano para a habilidade (Table 2 / Fig. 5), inclusive o modelo "só ONI" (SON para soja, NDJ para milho) e os anos fortes separados.
F. Regressão agrupada anomalia ~ ONI × região (teste conjunto do padrão espacial).
G. Table 1 em % (exp(beta)-1)."""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run relative to the repository root
_os.makedirs('resultados', exist_ok=True); _os.makedirs('dados/power', exist_ok=True)
import os, numpy as np, pandas as pd, warnings, os
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore'); os.makedirs('resultados', exist_ok=True)
OUT = 'resultados/'
rng = np.random.default_rng(0)
p0 = pd.read_csv('resultados/painel_uf_ano.csv'); p0 = p0[p0.ano >= 1990].copy(); p0['ly'] = np.log(p0.rend_kgha)
ult = p0[p0.ano == 2025].set_index(['uf', 'cultura']); keep = ult[ult.prod_t > 5e5].index
def anomalia(g):
    z = np.polyfit(g.ano - 2000, g.ly, 2); return g.ly - np.polyval(z, g.ano - 2000), z
# ---------- A. todos os anos de El Niño
oni = pd.read_csv('dados/oni_1980_2026.csv'); o = oni.pivot(index='ano', columns='estacao', values='oni')
seq = ['JAS', 'ASO', 'SON', 'OND', 'NDJ']; seq2 = ['DJF', 'JFM', 'FMA', 'MAM']
pico = {}
for y in range(1990, 2026):
    vals = list(o.loc[y - 1, seq].values) + list(o.loc[y, seq2].values); pico[y] = np.nanmax(vals)
def cat(v):
    return 'very strong' if v >= 2.0 else 'strong' if v >= 1.5 else 'moderate' if v >= 1.0 else 'weak' if v >= 0.5 else '-'
# anomalias por UF (quadrática, 1990-2025) e produção-tendência de 2025
A = []; base = {}
for (uf, c), g in p0.groupby(['uf', 'cultura']):
    if (uf, c) not in keep: continue
    g = g.sort_values('ano').reset_index(drop=True); a, z = anomalia(g)
    y25 = np.exp(np.polyval(z, 25)); base[(uf, c)] = ult.loc[(uf, c), 'area_ha'] * y25 / 1e9
    for yy, aa in zip(g.ano, a): A.append(dict(uf=uf, cultura=c, ano=yy, a=aa, base=base[(uf, c)]))
A = pd.DataFrame(A)
rows = []
for y in range(1990, 2026):
    r = dict(harvest_year=y, oni_ndj=o.loc[y - 1, 'NDJ'], oni_peak=pico[y], category=cat(pico[y]))
    for c in ['soja', 'milho']:
        s = A[(A.cultura == c) & (A.ano == y)]
        tot = s.base.sum(); d = (s.base * (np.exp(s.a) - 1)).sum()
        r[f'{c}_dMt_2025areas'] = d; r[f'{c}_pct_2025areas'] = 100 * d / tot
        r[f'{c}_mean_state_anom_pct'] = 100 * (np.exp(s.a.mean()) - 1)
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(OUT + 'A_todos_anos_anomalia_nacional.csv', index=False)
EN = T[T.oni_ndj >= 0.5]; print('--- A. anos El Niño (NDJ >= 0.5)'); print(EN.round(2).to_string())
EN.to_csv(OUT + 'A_anos_ElNino_tabela.csv', index=False)
for th in [1.5, 2.0]:
    s = T[T.oni_peak >= th]; print(f'peak>={th}: anos', list(s.harvest_year), 'milho<0:', (s.milho_pct_2025areas < 0).sum(), '/', len(s), 'soja<0:', (s.soja_pct_2025areas < 0).sum())
# ---------- B. chuva em estação por UF
def reg_boot(x, y, ano, B=2000, L=3):
    b = np.polyfit(x, y, 1)[0]; n = len(x); bs = []
    for k in range(B):
        ini = rng.integers(0, n - L + 1, int(np.ceil(n / L))); idx = np.concatenate([np.arange(i, i + L) for i in ini])[:n]
        bs.append(np.polyfit(x[idx], y[idx], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5]); r = np.corrcoef(x, y)[0, 1]
    return b, lo, hi, r
rows = []
for (uf, c), g in p0.groupby(['uf', 'cultura']):
    if (uf, c) not in keep: continue
    g = g.sort_values('ano').reset_index(drop=True); a, z = anomalia(g); a = a.values
    sul = uf in ('Rio Grande do Sul', 'Santa Catarina')
    # chuva: milho 2ª safra -> p2..p4 = mar-mai; soja e milho do Sul -> p2..p4 = dez-fev (meses out..mar)
    chuva = g[['p2', 'p3', 'p4']].mean(1).values; janela = 'Dec–Feb' if (c == 'soja' or sul) else 'Mar–May'
    zc = (chuva - chuva.mean()) / chuva.std()
    ok = np.isfinite(zc) & np.isfinite(a)
    b, lo, hi, r = reg_boot(zc[ok], a[ok], g.ano.values[ok])
    row = dict(uf=uf, cultura=c, n=ok.sum(), rain_window=janela, beta_rain_logpts_per_sd=100 * b, lo=100 * lo, hi=100 * hi, r_rain=r)
    if c == 'milho':
        x = g.oni_mam.values; b2, lo2, hi2, r2 = reg_boot(x, a, g.ano.values)
        row.update(beta_oniMAM_logpts_per_C=100 * b2, lo_oniMAM=100 * lo2, hi_oniMAM=100 * hi2, r_oniMAM=r2)
        x = g.oni_ndj_prev.values; b3, lo3, hi3, r3 = reg_boot(x, a, g.ano.values); row.update(r_oniNDJ=r3)
        # umidade do solo abr-mai (g3,g4)
        gw = g[['g3', 'g4']].mean(1).values; zg = (gw - gw.mean()) / gw.std(); b4, lo4, hi4, r4 = reg_boot(zg, a, g.ano.values)
        row.update(beta_gwet_AprMay=100 * b4, lo_gwet=100 * lo4, hi_gwet=100 * hi4, r_gwet=r4)
    rows.append(row)
B = pd.DataFrame(rows); B.to_csv(OUT + 'B_chuva_estacao_por_UF.csv', index=False)
print('--- B. chuva em estação'); print(B.round(2).to_string())
# correlação média ponderada pela produção, por cultura e grupo
B2 = B.merge(ult.reset_index()[['uf', 'cultura', 'prod_t']], on=['uf', 'cultura'])
for c in ['soja', 'milho']:
    s = B2[B2.cultura == c]; print(c, 'r chuva ponderado pela produção 2025:', round(np.average(s.r_rain, weights=s.prod_t), 2), '| n UF com IC>0:', (s.lo > 0).sum(), '/', len(s))
# ---------- C/D/E. modelos deixando um ano de fora
p = pd.read_csv('resultados/painel_uf_ano.csv'); p = p[(p.ano >= 1990) & (p.prod_t > 2e5)].copy(); p['ly'] = np.log(p.rend_kgha)
CL = ['prec', 'tmax', 'tmed', 'rh', 'gwet', 'rad'] + [f'{v}{i}' for v in 'pgt' for i in range(6)]
for v in CL:
    g = p.groupby(['uf', 'cultura'])[v]; p[v + '_a'] = (p[v] - g.transform('mean')) / g.transform('std')
p['oni'] = np.where(p.cultura == 'soja', p.oni_ndj_prev, np.where(p.cod_uf.astype(str).isin(['43', '42']), p.oni_ndj_prev, p.oni_mam))
F = [v + '_a' for v in CL] + ['oni']
w = pd.read_csv('dados/pesos_municipais.csv', dtype={'uf': str}); w['g'] = w[['Soja (em grão)', 'Milho (em grão)']].sum(1)
cen = w.groupby('uf').apply(lambda g: pd.Series({'lat': np.average(g.lat.fillna(g.lat.mean()), weights=g.g + 1e-9), 'lon': np.average(g.lon.fillna(g.lon.mean()), weights=g.g + 1e-9)})).reset_index()
cen['uf'] = cen.uf.astype(int); p = p.merge(cen, left_on='cod_uf', right_on='uf', suffixes=('', '_c')).reset_index(drop=True)
def tendencia(tr, te):
    out = pd.Series(index=te.index, dtype=float); fit = pd.Series(index=tr.index, dtype=float)
    for k, g in tr.groupby(['uf', 'cultura']):
        z = np.polyfit(g.ano - 2000, g.ly, 2); fit[g.index] = np.polyval(z, g.ano - 2000)
        m = (te.uf == k[0]) & (te.cultura == k[1]); out[te.index[m]] = np.polyval(z, te.ano[m] - 2000)
    return fit, out
def X(d, com_oni=True):
    Z = d[F].copy(); Z['soja'] = (d.cultura == 'soja').astype(int); Z['lat'] = d.lat; Z['lon'] = d.lon
    for v in ['prec_a', 'tmax_a', 'gwet_a', 'oni']: Z[v + '_x_lat'] = d[v] * d.lat
    if not com_oni: Z = Z.drop(columns=['oni', 'oni_x_lat'])
    return Z
grupos = {'ONI': ['oni', 'oni_x_lat'], 'Rain, season mean': ['prec_a', 'prec_a_x_lat'], 'Tmax, season mean': ['tmax_a', 'tmax_a_x_lat'],
          'Soil wetness, season mean': ['gwet_a', 'gwet_a_x_lat'], 'Tmean/RH/radiation, season': ['tmed_a', 'rh_a', 'rad_a'],
          'Rain, months 1–2': ['p0_a', 'p1_a'], 'Rain, months 3–4': ['p2_a', 'p3_a'], 'Rain, months 5–6': ['p4_a', 'p5_a'],
          'Soil wetness, months 1–2': ['g0_a', 'g1_a'], 'Soil wetness, months 3–4': ['g2_a', 'g3_a'], 'Soil wetness, months 5–6': ['g4_a', 'g5_a'],
          'Tmax, months 1–2': ['t0_a', 't1_a'], 'Tmax, months 3–4': ['t2_a', 't3_a'], 'Tmax, months 5–6': ['t4_a', 't5_a'],
          'Location (lat, lon, crop)': ['lat', 'lon', 'soja']}
pred = {m: pd.Series(index=p.index, dtype=float) for m in ['trend', 'ridge', 'ridge_noONI', 'rf', 'rf_noONI', 'oni_only', 'oni_only_ndj', 'oni_only_son']}
perm = {c: {g: 0.0 for g in grupos} for c in ['soja', 'milho']}; perm_base = {c: 0.0 for c in ['soja', 'milho']}
# ONI pré-safra por cultura: SON para soja (antes da semeadura), NDJ para milho (conhecido na semeadura da 2ª safra)
p['oni_pre'] = np.where(p.cultura == 'soja', p.oni_son_prev, p.oni_ndj_prev)
def Xo(d, col, ufs):
    Z = pd.DataFrame({'oni': d[col].values}, index=d.index)
    for u in ufs: Z['oni_' + u] = (d.uf == u).astype(float) * d[col]
    return Z
for ano in sorted(p.ano.unique()):
    tr = p[p.ano != ano]; te = p[p.ano == ano]
    ftr, fte = tendencia(tr, te); ytr = tr.ly - ftr; pred['trend'][te.index] = fte
    for nome, com in [('ridge', True), ('ridge_noONI', False)]:
        Xtr, Xte = X(tr, com), X(te, com); mu, sd = Xtr.mean(), Xtr.std() + 1e-9
        r = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((Xtr - mu) / sd, ytr); pred[nome][te.index] = fte + r.predict((Xte - mu) / sd)
    for nome, com in [('rf', True), ('rf_noONI', False)]:
        Xtr, Xte = X(tr, com), X(te, com)
        rf = RandomForestRegressor(500, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=-1).fit(Xtr, ytr)
        pr = rf.predict(Xte); pred[nome][te.index] = fte + pr
        if com:   # importância por permutação (no ano deixado de fora)
            e_base = (te.ly - fte - pr) ** 2
            for c in ['soja', 'milho']:
                k = (te.cultura == c).values; perm_base[c] += e_base[k].sum()
            for gname, cols in grupos.items():
                Xp = Xte.copy()
                for col in cols: Xp[col] = rng.permutation(Xp[col].values)
                e_p = (te.ly - fte - rf.predict(Xp)) ** 2
                for c in ['soja', 'milho']:
                    k = (te.cultura == c).values; perm[c][gname] += e_p[k].sum()
    # só ONI, inclinação por UF e cultura
    for nome, col in [('oni_only', 'oni_pre'), ('oni_only_ndj', 'oni_ndj_prev'), ('oni_only_son', 'oni_son_prev')]:
        for c in ['soja', 'milho']:
            trc = tr[tr.cultura == c]; tec = te[te.cultura == c]; ufs = p[p.cultura == c].uf.unique()
            if tec.empty: continue
            rg = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Xo(trc, col, ufs), ytr[trc.index])
            pred[nome][tec.index] = fte[tec.index] + rg.predict(Xo(tec, col, ufs))
    print(ano, flush=True)
P = p[['uf', 'cultura', 'ano', 'ly', 'prod_t', 'oni', 'oni_pre']].copy()
for m, s in pred.items(): P['pred_' + m] = s
# juntar as predições dos modelos originais (mixed, lgbm, gp) do arquivo v2
P2 = pd.read_csv('resultados/predicoes_LOYO_v2.csv')[['uf', 'cultura', 'ano', 'pred_ridge', 'pred_mixed', 'pred_lgbm', 'pred_rf', 'pred_gp', 'pred_trend']]
P = P.merge(P2, on=['uf', 'cultura', 'ano'], suffixes=('', '_v2'))
assert np.allclose(P.pred_trend, P.pred_trend_v2); P.to_csv(OUT + 'E_predicoes_LOYO_revisao.csv', index=False)
print('coerência ridge novo vs v2 (corr):', np.corrcoef(P.pred_ridge, P.pred_ridge_v2)[0, 1].round(4))
# D. importância por permutação
D = pd.DataFrame({c: {g: 100 * (perm[c][g] / perm_base[c] - 1) for g in grupos} for c in ['soja', 'milho']})
D.index.name = 'predictor group'; D.columns = ['soybean_MSE_increase_pct', 'maize_MSE_increase_pct']
D = D.sort_values('maize_MSE_increase_pct', ascending=False); D.to_csv(OUT + 'D_importancia_permutacao_RF.csv'); print('--- D. importância por permutação (% de aumento do EQM fora da amostra)'); print(D.round(1).to_string())
# C/E. habilidade com IC por bootstrap por ano
FORTES = [1998, 2016, 2024]
MOD = {'trend': 'Trend only', 'ridge_v2': 'Ridge', 'mixed': 'Ridge + state slopes', 'rf_v2': 'Random forest', 'lgbm': 'LightGBM', 'gp': 'Gaussian process',
       'ridge_noONI': 'Ridge without ONI', 'rf_noONI': 'Random forest without ONI', 'oni_only': 'Pre-season ONI only', 'oni_only_ndj': 'ONI only (NDJ both crops)', 'oni_only_son': 'ONI only (SON both crops)'}
def skill(P, m, c, anos):
    k = (P.cultura == c) & P.ano.isin(anos); e = P.ly[k] - P['pred_' + m][k]; e0 = P.ly[k] - P.pred_trend[k]
    return 1 - (e ** 2).sum() / (e0 ** 2).sum()
anos = sorted(P.ano.unique()); rows = []
for m, nome in MOD.items():
    if m == 'trend': continue
    for c in ['soja', 'milho']:
        n_all = ((P.cultura == c)).sum(); n_str = ((P.cultura == c) & P.ano.isin(FORTES)).sum()
        s_all = skill(P, m, c, anos); s_str = skill(P, m, c, FORTES)
        # bootstrap por ano (1000 réplicas): reamostrar anos com reposição e recalcular a habilidade
        ba = []; bsx = []
        for k in range(1000):
            ya = rng.choice(anos, len(anos), replace=True)
            # habilidade com anos repetidos: soma dos erros por ano ponderada pela multiplicidade
            kk = (P.cultura == c); e = (P.ly - P['pred_' + m])[kk] ** 2; e0 = (P.ly - P.pred_trend)[kk] ** 2
            se = e.groupby(P.ano[kk]).sum(); se0 = e0.groupby(P.ano[kk]).sum()
            ba.append(1 - se.reindex(ya).sum() / se0.reindex(ya).sum())
            ys = rng.choice(FORTES, 3, replace=True); bsx.append(1 - se.reindex(ys).sum() / se0.reindex(ys).sum())
        r = dict(model=nome, crop=c, n_all=n_all, skill_all=s_all, lo_all=np.percentile(ba, 2.5), hi_all=np.percentile(ba, 97.5),
                 n_strong=n_str, skill_strong=s_str, lo_strong=np.percentile(bsx, 2.5), hi_strong=np.percentile(bsx, 97.5))
        for y in FORTES: r[f'skill_{y}'] = skill(P, m, c, [y])
        rows.append(r)
E = pd.DataFrame(rows); E.to_csv(OUT + 'E_habilidade_IC_bootstrap.csv', index=False)
print('--- E. habilidade com IC (bootstrap por ano)'); print(E.round(3).to_string())
# ---------- F. regressão agrupada anomalia ~ ONI × região
REG = {'South': ['Paraná', 'Santa Catarina', 'Rio Grande do Sul'], 'Centre-West and Southeast': ['Mato Grosso', 'Mato Grosso do Sul', 'Goiás', 'Minas Gerais', 'São Paulo', 'Rondônia'],
       'North and Northeast (incl. Matopiba)': ['Maranhão', 'Tocantins', 'Piauí', 'Bahia', 'Pará', 'Sergipe']}
A2 = A.merge(p0[['uf', 'cultura', 'ano', 'oni_ndj_prev']], on=['uf', 'cultura', 'ano']); A2['reg'] = A2.uf.map({u: r for r, us in REG.items() for u in us})
rows = []
for c in ['soja', 'milho']:
    d = A2[A2.cultura == c].copy(); anos_c = sorted(d.ano.unique())
    def fit(dd):
        out = {}
        for r in REG:
            s = dd[dd.reg == r]
            if len(s) > 5: out[r] = np.polyfit(s.oni_ndj_prev, s.a, 1)[0]
        # diferença Sul - (centro-oeste+norte)
        s1 = dd[dd.reg == 'South']; s2 = dd[dd.reg != 'South']
        out['South minus rest'] = np.polyfit(s1.oni_ndj_prev, s1.a, 1)[0] - np.polyfit(s2.oni_ndj_prev, s2.a, 1)[0]
        return out
    est = fit(d); bs = {k: [] for k in est}; n = len(anos_c); L = 3
    for k in range(2000):
        ini = rng.integers(0, n - L + 1, int(np.ceil(n / L))); idx = np.concatenate([np.arange(i, i + L) for i in ini])[:n]
        ya = [anos_c[i] for i in idx]; dd = pd.concat([d[d.ano == y] for y in ya])
        for kk, v in fit(dd).items(): bs[kk].append(v)
    for kk, v in est.items():
        b = np.array(bs[kk]); pv = min(1, 2 * min((b <= 0).mean(), (b >= 0).mean()))
        rows.append(dict(crop=c, term=kk, n_state_years=int((d.reg == kk).sum()) if kk in REG else len(d), beta_logpts_per_C=100 * v, lo=100 * np.percentile(b, 2.5), hi=100 * np.percentile(b, 97.5), p_boot=pv))
Fr = pd.DataFrame(rows); Fr.to_csv(OUT + 'F_regressao_agrupada_regiao.csv', index=False); print('--- F. ONI x região'); print(Fr.round(3).to_string())
# ---------- G. Table 1 em percentagem
R = pd.read_csv('resultados/enso_por_uf_v2.csv')
for k in ['beta_oni', 'lo', 'hi']: R[k + '_pct'] = 100 * (np.exp(R[k]) - 1)
R['dprod_analogo_Mt'] = R['dprod_analogo_Mt']
R.to_csv(OUT + 'G_table1_percent.csv', index=False)
print('--- G. Table 1 em %'); print(R[['uf', 'cultura', 'beta_oni', 'beta_oni_pct', 'lo_pct', 'hi_pct', 'q_BH', 'dprod_analogo_Mt']].round(2).to_string())
for c in ['soja', 'milho']:
    s = R[R.cultura == c]; print(c, 'analogo medio total', round(s.dprod_analogo_Mt.sum(), 2), '| +2.0:', round((s.trend_prod_2025_Mt * (np.exp(s.beta_oni * 2.0) - 1)).sum(), 2), '| +3.0:', round((s.trend_prod_2025_Mt * (np.exp(s.beta_oni * 3.0) - 1)).sum(), 2),
          '| perdas (outras UF)', round(s[s.dprod_oni25_Mt < 0].dprod_oni25_Mt.sum(), 2), 'ganhos RS+SC', round(s[s.dprod_oni25_Mt > 0].dprod_oni25_Mt.sum(), 2))
print('fim')
