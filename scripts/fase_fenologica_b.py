"""Companion to fase_fenologica.py: phase responses estimated one variable family at a time
(rainfall of the three phases; soil wetness of the three phases; Tmax of the three phases), to avoid the
within-phase collinearity of rainfall and soil wetness, pooled by group and for the main states.
Also partial R² of each family. Outputs resultados/F1b_resposta_por_fase_familia.csv and F1c_por_estado.csv."""
import os, numpy as np, pandas as pd, warnings, importlib.util, sys
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# reuse data preparation from the main script without running its analyses
src = open('codigo/fase_fenologica.py').read().split('# ---------------------------------------------------------------- F1')[0]
exec(compile(src, 'prep', 'exec'))
LOG = open('resultados/fase_fenologica_b.log', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
grandes = p[p.ano == 2025].query('prod_t > 2e5')[['cod_uf', 'cultura']]
pg = p.merge(grandes, on=['cod_uf', 'cultura'])
FAM = {'rainfall': 'P', 'soil wetness': 'G', 'Tmax': 'T'}

def r2_parcial(d, cols):
    y = anomalia(d); m = y.notna() & d[cols].notna().all(1)
    X = np.column_stack([np.ones(m.sum()), d.loc[m, cols].values]); b, *_ = np.linalg.lstsq(X, y[m].values, rcond=None)
    res = y[m].values - X @ b; return 1 - res.var() / y[m].var()

rows = []
for g, d in pg.groupby('grupo'):
    for fam, v in FAM.items():
        cols = [f'{v}_{f}_z' for f in FASES]
        b, lo, hi, pv, n = block_boot(d, cols, anomalia, B=2000)
        r2 = r2_parcial(d, cols)
        for f, bb, l, h, q in zip(FASES, b, lo, hi, pv):
            rows.append(dict(group=g, family=fam, phase=f, slope_lp=100 * bb, lo=100 * l, hi=100 * h, p=q, n=n, states=d.sigla.nunique(), R2_family=r2))
    # all nine together, for reference
    rows.append(dict(group=g, family='all nine', phase='-', slope_lp=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(d), states=d.sigla.nunique(),
                     R2_family=r2_parcial(d, [c + '_z' for c in PH])))
    log(g)
F1b = pd.DataFrame(rows); F1b.to_csv('resultados/F1b_resposta_por_fase_familia.csv', index=False)
log(F1b.round(2).to_string(index=False))

# main states, rainfall and soil wetness families
log('\n== main states ==')
rows = []
ESTADOS = {'soja': ['MT', 'PR', 'RS', 'GO', 'MS', 'MG', 'BA', 'MA', 'TO', 'PI', 'SP'], 'milho': ['MT', 'PR', 'GO', 'MS', 'MG', 'SP', 'RS', 'SC', 'BA']}
for cul, ufs in ESTADOS.items():
    for uf in ufs:
        d = pg[(pg.cultura == cul) & (pg.sigla == uf)]
        if len(d) < 25: continue
        for fam in ['rainfall', 'soil wetness', 'Tmax']:
            v = FAM[fam]; cols = [f'{v}_{f}_z' for f in FASES]
            b, lo, hi, pv, n = block_boot(d, cols, anomalia, B=1000)
            for f, bb, l, h, q in zip(FASES, b, lo, hi, pv):
                rows.append(dict(crop=cul, state=uf, group=d.grupo.iloc[0], family=fam, phase=f, slope_lp=100 * bb, lo=100 * l, hi=100 * h, p=q, n=n))
F1c = pd.DataFrame(rows); F1c.to_csv('resultados/F1c_por_estado.csv', index=False)
sig = F1c[(F1c.p < 0.05)]
log(sig.round(1).to_string(index=False))
LOG.close()
