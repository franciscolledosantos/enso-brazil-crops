import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run relative to the repository root
_os.makedirs('resultados', exist_ok=True); _os.makedirs('dados/power', exist_ok=True)
import os, numpy as np, pandas as pd, matplotlib, json
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from shapely.geometry import shape
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
F = 'figuras/'; os.makedirs(F, exist_ok=True)
# Fig 1: participação do Brasil no mercado mundial (USDA PSD)
s = pd.read_csv('dados/psd_subset.csv')
w = s[s.Attribute_Description.isin(['Production', 'Exports'])].groupby(['Commodity_Description', 'Attribute_Description', 'Market_Year']).Value.sum()
b = s[(s.Country_Name == 'Brazil') & s.Attribute_Description.isin(['Production', 'Exports'])].groupby(['Commodity_Description', 'Attribute_Description', 'Market_Year']).Value.sum()
sh = (b / w).unstack(level=[0, 1]).loc[1980:2026]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for (c, a), cor, ls in [(('Oilseed, Soybean', 'Production'), '#2e7d32', '-'), (('Oilseed, Soybean', 'Exports'), '#2e7d32', '--'),
                        (('Corn', 'Production'), '#f9a825', '-'), (('Corn', 'Exports'), '#f9a825', '--')]:
    ax.plot(sh.index, sh[(c, a)] * 100, color=cor, ls=ls, lw=1.6, label=f"{'Soybean' if 'Soy' in c else 'Maize'} – {a.lower()}")
ax.axvspan(2025.5, 2026.5, color='#ddd'); ax.text(2026, 3, 'USDA\n2026/27', ha='center', fontsize=6)
ax.set_ylabel("Brazil's share of world total (%)"); ax.set_xlabel('Marketing year'); ax.legend(frameon=False, fontsize=7); ax.set_ylim(0, 70)
fig.tight_layout(); fig.savefig(F + 'Fig1_brazil_share.png', dpi=300); plt.close()
# Fig 2: mapa do efeito do ONI por UF (soja e milho)
R = pd.read_csv('resultados/enso_por_uf_v2.csv')
g = json.load(open('dados/raw/municipios.geojson'))
import pandas as pd
# contornos por UF: dissolver municípios (agregação simples pelas geometrias)
from shapely.ops import unary_union
geoms = {}
for ft in g['features']:
    uf = str(ft['properties']['codarea'])[:2]; geoms.setdefault(uf, []).append(shape(ft['geometry']).buffer(0))
ufs = {k: unary_union(v) for k, v in geoms.items()}
nome2cod = pd.read_csv('dados/pam_uf_1974_2025.csv', dtype={'D1C': str}).drop_duplicates('uf').set_index('uf').D1C.to_dict()
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
cm = plt.get_cmap('RdBu'); norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-32, vmax=32)
for ax, c, tit in [(axs[0], 'soja', '(a) Soybean'), (axs[1], 'milho', '(b) Maize')]:
    r = R[R.cultura == c].set_index('uf')
    for uf, geo in ufs.items():
        nome = [k for k, v in nome2cod.items() if v == uf]; nome = nome[0] if nome else None
        polys = geo.geoms if geo.geom_type == 'MultiPolygon' else [geo]
        if nome in r.index:
            v = 100 * (np.exp(r.loc[nome, 'beta_oni'] * 2.5) - 1); sig = (r.loc[nome, 'hi'] < 0) or (r.loc[nome, 'lo'] > 0)
            for pg in polys:
                x, y = pg.exterior.xy; ax.fill(x, y, color=cm(norm(v)), ec='k' if sig else '#888', lw=1.0 if sig else 0.3)
        else:
            for pg in polys:
                x, y = pg.exterior.xy; ax.fill(x, y, color='#f2f2f2', ec='#bbb', lw=0.3)
    ax.set_xlim(-74.5, -34.0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(tit, loc='left', fontsize=9)
    for sp in ax.spines.values(): sp.set_visible(False)
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
fig.colorbar(sm, ax=axs, shrink=0.7, label='Yield change for ONI = +2.5 °C (%)')
fig.savefig(F + 'Fig2_enso_effect_map.png', dpi=300, bbox_inches='tight'); plt.close()
# Fig 3: anomalias nos El Niño fortes x ONI (MT, Matopiba, RS, PR)
A = pd.read_csv('resultados/anomalias_uf.csv')
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
grp = {'Mato Grosso': '#c62828', 'Matopiba (MA, TO, PI, BA)': '#ef6c00', 'Paraná': '#1565c0', 'Rio Grande do Sul': '#2e7d32'}
mtp = ['Maranhão', 'Tocantins', 'Piauí', 'Bahia']
for ax, c, tit in [(axs[0], 'soja', '(a) Soybean'), (axs[1], 'milho', '(b) Maize')]:
    for k, cor in grp.items():
        sub = A[(A.cultura == c) & (A.uf.isin(mtp) if k.startswith('Matopiba') else A.uf == k)].groupby('ano').agg(a=('a', 'mean'), oni=('oni', 'first'))
        sub['pct'] = 100 * (np.exp(sub.a) - 1); ax.scatter(sub.oni, sub.pct, s=10, color=cor, alpha=0.7, label=k)
        z = np.polyfit(sub.oni, sub.pct, 1); xx = np.linspace(-2, 2.7, 10); ax.plot(xx, np.polyval(z, xx), color=cor, lw=1.2)
    ax.axhline(0, color='#999', lw=0.6); ax.axvline(0, color='#999', lw=0.6)
    ax.set_xlabel('ONI (NDJ before harvest, °C)'); ax.set_title(tit, loc='left', fontsize=9)
axs[0].set_ylabel('Detrended yield anomaly (%)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7, ncol=4, loc='lower center')
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(F + 'Fig3_anomaly_vs_oni.png', dpi=300); plt.close()
# Fig 4: cenário 2026/27 por UF (Mt), ONI linear e análogos
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=False)
for ax, c, tit in [(axs[0], 'soja', '(a) Soybean'), (axs[1], 'milho', '(b) Maize')]:
    r = R[R.cultura == c].sort_values('dprod_analogo_Mt')
    y = np.arange(len(r))
    ax.barh(y - 0.2, r.dprod_oni25_Mt, 0.4, color='#5c6bc0', label='Linear ONI effect (ONI = +2.5)')
    ax.barh(y + 0.2, r.dprod_analogo_Mt, 0.4, color='#ef6c00', label='Strong El Niño analogues')
    ax.set_yticks(y); ax.set_yticklabels(r.uf, fontsize=7); ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('Change in production (Mt)'); ax.set_title(tit, loc='left', fontsize=9)
axs[1].legend(frameon=False, fontsize=6.5, loc='upper left')
fig.tight_layout(); fig.savefig(F + 'Fig4_scenario_2026_27.png', dpi=300); plt.close()
# Fig 5: habilidade preditiva (LOYO) com IC por bootstrap por ano; anos fortes separados (revisão 17/10)
E = pd.read_csv('resultados/E_habilidade_IC_bootstrap.csv')
ordem = ['Ridge', 'Ridge + state slopes', 'Random forest', 'LightGBM', 'Gaussian process', 'Ridge without ONI', 'Random forest without ONI', 'Pre-season ONI only']
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
x = np.arange(len(ordem))
for k, (c, cor, lab) in enumerate([('soja', '#2e7d32', 'Soybean'), ('milho', '#f9a825', 'Maize')]):
    m = E[E.crop == c].set_index('model').reindex(ordem)
    axs[0].bar(x + (k - 0.5) * 0.38, m.skill_all, 0.38, color=cor, label=lab)
    axs[0].errorbar(x + (k - 0.5) * 0.38, m.skill_all, yerr=[m.skill_all - m.lo_all, m.hi_all - m.skill_all], fmt='none', ecolor='k', lw=0.8, capsize=2)
    for j, (yy, mk) in enumerate([(1998, 'o'), (2016, 's'), (2024, '^')]):
        v = m[f'skill_{yy}'].clip(lower=-0.6)
        axs[1].scatter(x + (k - 0.5) * 0.38, v, marker=mk, s=16, color=cor, edgecolor='k', lw=0.4, label=f'{yy}' if k == 0 else None)
        for xi, (vv, raw) in enumerate(zip(v, m[f'skill_{yy}'])):
            if raw < -0.6: axs[1].annotate(f'{raw:.1f}', (xi + (k - 0.5) * 0.38, -0.6), fontsize=5.5, ha='right' if k == 0 else 'left', va='top', xytext=(-2 if k == 0 else 2, -4), textcoords='offset points')
for ax, tit in [(axs[0], '(a) All years, 1990–2025 (95% CI)'), (axs[1], '(b) Strong El Niño harvests, by year')]:
    ax.set_xticks(x); ax.set_xticklabels(ordem, rotation=40, ha='right', fontsize=6.5); ax.axhline(0, color='k', lw=0.6)
    ax.axvline(4.5, color='#aaa', lw=0.6, ls=':'); ax.set_title(tit, loc='left', fontsize=9)
axs[0].set_ylabel('MSE skill vs. trend only'); axs[0].legend(frameon=False, fontsize=7, loc='upper right'); axs[1].legend(frameon=False, fontsize=6.5, title='Harvest', title_fontsize=6.5, loc='upper right', ncol=3)
axs[0].set_ylim(-0.7, 0.95)
fig.tight_layout(); fig.savefig(F + 'Fig5_skill_LOYO.png', dpi=300); plt.close()
print('ok')
