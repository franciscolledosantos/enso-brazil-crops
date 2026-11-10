"""Fig. do mapa (v4, 10/11/2026: sem contorno preto; significância só pelo rótulo em negrito com valor). Base: v3, 07/11/2026: mapa classificado do efeito do ONI por UF.
Método da skill dataviz: escala divergente com 4 degraus por braço (azul 250–650 / vermelho de mesma
luminosidade OKLCH) e cinza neutro no centro, ambos os braços validados com validate_palette.js --ordinal;
rótulos de valor só onde o IC exclui zero; texto em tinta neutra; legenda discreta.
Lê dados_derivados/enso_por_uf_v2.csv e dados_derivados/brazil-states.geojson.
Saída: figuras/Fig2_enso_effect_map.png. Rodar a partir da pasta do artigo."""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely.geometry import shape

INK, INK2, INK_LIGHT = "#0b0b0b", "#52514e", "#f0efec"
BLUE = ["#86b6ef", "#5598e7", "#256abf", "#104281"]      # +1–5, +5–10, +10–20, >+20
RED = ["#ef9790", "#e36b65", "#b73737", "#7a1b1d"]       # −1–5, −5–10, −10–20, <−20
NEUTRAL = "#f0efec"
EDGES = [-20, -10, -5, -1, 1, 5, 10, 20]
COLORS = RED[::-1] + [NEUTRAL] + BLUE                      # 9 classes, de <−20 a >+20
LABELS = ["< −20", "−20 to −10", "−10 to −5", "−5 to −1", "−1 to +1", "+1 to +5", "+5 to +10", "+10 to +20", "> +20"]
DARK = {RED[2], RED[3], BLUE[2], BLUE[3]}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK})
R = pd.read_csv("dados_derivados/enso_por_uf_v2.csv")
g = json.load(open("dados_derivados/brazil-states.geojson", encoding="utf-8"))
ufs = {f["properties"]["name"]: (f["properties"]["sigla"], shape(f["geometry"]).buffer(0)) for f in g["features"]}

pos = {"Mato Grosso": (-55.8, -13.0), "Pará": (-52.5, -4.5), "Bahia": (-41.8, -12.3), "Minas Gerais": (-44.8, -18.5),
       "Goiás": (-49.6, -15.9), "Mato Grosso do Sul": (-54.8, -20.6), "São Paulo": (-49.0, -22.2), "Paraná": (-51.6, -24.6),
       "Rio Grande do Sul": (-53.6, -29.8), "Maranhão": (-45.3, -5.3), "Tocantins": (-48.3, -10.2), "Piauí": (-42.9, -7.6),
       "Rondônia": (-62.9, -10.9)}
fora = {"Santa Catarina": (-44.0, -27.6), "Sergipe": (-33.6, -10.8)}

def classe(v):
    return COLORS[int(np.digitize(v, EDGES))]

fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.4))
for ax, c, tit in [(axs[0], "soja", "(a) Soybean"), (axs[1], "milho", "(b) Maize")]:
    r = R[R.cultura == c].set_index("uf")
    for nome, (sig, geo) in ufs.items():
        polys = geo.geoms if geo.geom_type == "MultiPolygon" else [geo]
        if nome in r.index:
            v = 100 * (np.exp(r.loc[nome, "beta_oni"] * 2.5) - 1)
            s = (r.loc[nome, "hi"] < 0) or (r.loc[nome, "lo"] > 0)
            fill = classe(v)
            for pg in polys:
                x, y = pg.exterior.xy
                ax.fill(x, y, color=fill, ec="white", lw=0.6, zorder=1)      # separador claro entre preenchimentos
            vr = int(round(v))
            txt = f"{sig}\n{vr:+d}" if s else sig                              # valor só onde o IC exclui zero
            cor = INK_LIGHT if fill in DARK else (INK if s else INK2)
            kw = dict(fontsize=6.4, ha="center", va="center", fontweight="bold" if s else "normal", linespacing=0.95, zorder=5)
            if nome in fora:
                rp = geo.representative_point()
                ax.annotate(txt, xy=(rp.x, rp.y), xytext=fora[nome], color=INK if s else INK2,
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=0, shrinkB=0), **kw)
            else:
                lon, lat = pos.get(nome, (geo.representative_point().x, geo.representative_point().y))
                ax.text(lon, lat, txt, color=cor, **kw)
        else:
            for pg in polys:
                x, y = pg.exterior.xy
                ax.fill(x, y, facecolor="white", ec="#c8c7c2", lw=0.4, hatch="////", zorder=0)
    ax.set_xlim(-74.5, -31.0); ax.set_ylim(-34.5, 5.8); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(tit, loc="left", fontsize=9, fontweight="bold", color=INK)
    for sp in ax.spines.values(): sp.set_visible(False)
plt.rcParams["hatch.color"] = "#c8c7c2"; plt.rcParams["hatch.linewidth"] = 0.4

# legenda discreta: classes em uma linha, depois os dois símbolos auxiliares
h1 = [Patch(facecolor=col, edgecolor="white", lw=0.6, label=lab) for col, lab in zip(COLORS, LABELS)]
leg1 = fig.legend(handles=h1, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=9, frameon=False, fontsize=6.6,
                  handlelength=1.6, handleheight=1.1, columnspacing=0.9, handletextpad=0.4,
                  title="Yield change for ONI = +2.5 °C (%, exp(2.5β) − 1)", title_fontsize=7.5)
leg1.get_title().set_color(INK)
h2 = [Line2D([0], [0], color="none", label="Bold label with value: 95% bootstrap CI excludes zero"),
      Patch(facecolor="white", edgecolor="#c8c7c2", hatch="////", label="Not analysed (2025 production < 0.5 Mt)")]
fig.legend(handles=h2, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=6.8,
           handlelength=1.2, columnspacing=2.5)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.2, wspace=0.03)
fig.savefig("fcr/figuras/Fig2_enso_effect_map.png", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
print("ok")
