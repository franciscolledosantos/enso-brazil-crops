"""Fig. 5 (FCR): interannual series of the phase variables that matter, by crop group, 1990–2025, with El Niño harvests marked.
Also prints the late-season rainfall anomaly of the strong El Niño harvests (1992, 1998, 2010, 2016, 2024)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
src=open('codigo/fase_fenologica.py').read().split('# ---------------------------------------------------------------- F1')[0]
exec(compile(src,'prep','exec'))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4df"; C_RAIN, C_SOIL, C_TMAX = "#256abf", "#2a9d7c", "#b73737"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK, "axes.edgecolor": INK2, "axes.spines.top": False, "axes.spines.right": False})
grandes = p[p.ano == 2025].query('prod_t > 2e5')[['cod_uf', 'cultura']]; pg = p.merge(grandes, on=['cod_uf', 'cultura'])
oni = p[['ano','oni_ndj_prev']].drop_duplicates('ano').set_index('ano').oni_ndj_prev
elnino = oni[oni >= 0.5].index.tolist(); forte = oni[oni >= 1.5].index.tolist()
# production-weighted group means of the raw phase variables (weights: 2025 production)
w25 = p[p.ano == 2025].set_index(['cod_uf','cultura']).prod_t
def serie(d, var):
    d = d.copy(); d['w'] = [w25.get((a, b), np.nan) for a, b in zip(d.cod_uf, d.cultura)]
    return d.groupby('ano').apply(lambda g: np.average(g[var], weights=g.w))
seis = pg[(pg.cultura == 'milho') & (pg.sigla.isin(['MT','PR','MS','GO','SP','MG']))]
sul = pg[pg.grupo == 'soybean South']; cn = pg[pg.grupo == 'soybean Centre-North']
panels = [("(a) Second-crop maize, six main states: flowering rainfall (Mar–Apr)", serie(seis, 'P_flo'), C_RAIN, 'mm'),
          ("(b) Second-crop maize, six main states: grain-filling Tmax (May–Jun)", serie(seis, 'T_fill'), C_TMAX, '°C'),
          ("(c) Soybean, South: flowering rainfall (Dec–Jan)", serie(sul, 'P_flo'), C_RAIN, 'mm'),
          ("(d) Soybean, Centre-North: grain-filling Tmax (Jan–Feb)", serie(cn, 'T_fill'), C_TMAX, '°C')]
fig, axs = plt.subplots(4, 1, figsize=(7.4, 7.6), sharex=True)
for ax, (tit, s, col, unit) in zip(axs, panels):
    s = s.loc[1990:2025]; m = s.mean()
    for a in elnino: ax.axvspan(a - 0.5, a + 0.5, color="#f3e6c8" if a not in forte else "#e8c98a", lw=0, zorder=0)
    ax.axhline(m, color=INK2, lw=0.7, ls=':'); ax.plot(s.index, s.values, color=col, lw=1.6); ax.plot(s.index, s.values, 'o', color=col, ms=3, mec='white', mew=0.5)
    ax.set_title(tit, loc='left', fontsize=8.5); ax.set_ylabel(unit); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for a in forte: ax.text(a, ax.get_ylim()[1], str(a), ha='center', va='top', fontsize=6.5, color=INK2)
axs[-1].set_xlabel("Harvest year"); axs[-1].set_xlim(1989.3, 2025.7)
fig.text(0.01, 0.005, "Shaded: El Niño harvests (NDJ ONI ≥ 0.5 °C); darker: strong and very strong events (≥ 1.5 °C), labelled. Dotted line: 1990–2025 mean. Series are production-weighted means of the states in each group.", fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("fcr/figuras/Fig5_climate_windows.png", dpi=300, bbox_inches="tight", facecolor="white")
# anomalies of the strong El Niño harvests, six-state second crop
for var, lab in [('P_flo', 'Mar–Apr rain'), ('G_fill', 'May–Jun soil wetness'), ('T_fill', 'May–Jun Tmax')]:
    s = serie(seis, var).loc[1990:2025]; z = (s - s.mean()) / s.std()
    print(lab, {a: round(z[a], 2) for a in forte})
