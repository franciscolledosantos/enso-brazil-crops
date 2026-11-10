"""Figures for the Field Crops Research version (09/11/2026).
Fig. 1  phenological windows and phase responses (CONAB second-crop maize, IBGE total maize, soybean South, soybean Centre-North)
Fig. 3  where ENSO enters the season: slope of each standardized phase variable on the NDJ ONI, by group
Fig. 4  leave-one-year-out skill by predictor set
Style: dataviz skill — thin marks, direct labels, neutral ink, categorical hues in fixed order, no dual axes.
Run from the article folder: python3 codigo/figuras_fcr.py"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4df"
C_RAIN, C_SOIL, C_TMAX = "#256abf", "#2a9d7c", "#b73737"      # rainfall, soil wetness, Tmax (fixed order)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
FASES = ["est", "flo", "fill"]; FASE_LAB = {"est": "Establishment", "flo": "Flowering", "fill": "Grain filling"}
FAM = [("rainfall", C_RAIN, "Rainfall"), ("soil wetness", C_SOIL, "Soil wetness"), ("Tmax", C_TMAX, "Tmax")]

F1b = pd.read_csv("resultados/F1b_resposta_por_fase_familia.csv")
F5 = pd.read_csv("resultados/F5_conab_fase.csv")
F2 = pd.concat([pd.read_csv("resultados/F2_enso_para_clima_fase.csv"), pd.read_csv("resultados/F2b_enso_fase_seis_estados.csv")])
F3 = pd.read_csv("resultados/F3_skill_por_conjunto.csv")

def painel_resposta(ax, df, titulo, xlim=(-25, 25)):
    """dot + 95% CI per phase (rows) and family (colour offset)."""
    ax.axvline(0, color=INK2, lw=0.8)
    for i, f in enumerate(FASES):
        for j, (fam, col, lab) in enumerate(FAM):
            r = df[(df.phase == f) & (df.family == fam)]
            if r.empty: continue
            r = r.iloc[0]; y = 2 - i + (1 - j) * 0.22
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.6, solid_capstyle="round", alpha=0.9)
            ax.plot(r.slope_lp, y, "o", color=col, ms=4.5, mec="white", mew=0.6)
            if r.p < 0.05:
                ax.text(r.hi + 0.6 if r.slope_lp >= 0 else r.lo - 0.6, y, f"{r.slope_lp:+.0f}", va="center", ha="left" if r.slope_lp >= 0 else "right", fontsize=7, color=INK)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([FASE_LAB[f] for f in FASES]); ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(*xlim); ax.set_xlabel("Yield response (log-points per SD)"); ax.set_title(titulo, loc="left", fontsize=8.5, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)

# ---------------------------------------------------------------- Fig 1
fig = plt.figure(figsize=(7.4, 6.6))
gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1, 1], hspace=0.85, wspace=0.35)
ax0 = fig.add_subplot(gs[0, :])
# calendar: months Sep..Jul; bars per group and phase
MESES = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
mi = {m: i for i, m in enumerate(MESES)}
cal = [("Soybean, Centre-North", [("est", "Oct", "Nov"), ("flo", "Dec", "Dec"), ("fill", "Jan", "Feb")]),
       ("Soybean, South", [("est", "Oct", "Nov"), ("flo", "Dec", "Jan"), ("fill", "Feb", "Mar")]),
       ("Second-crop maize", [("est", "Jan", "Feb"), ("flo", "Mar", "Apr"), ("fill", "May", "Jun")]),
       ("First-crop maize, South", [("est", "Oct", "Nov"), ("flo", "Dec", "Jan"), ("fill", "Feb", "Mar")])]
PH_COL = {"est": "#cfd8e3", "flo": "#8fb3d9", "fill": "#3f6fa8"}
for k, (nome, fases) in enumerate(cal):
    y = len(cal) - 1 - k
    for f, a, b in fases:
        ax0.add_patch(Rectangle((mi[a] - 0.45, y - 0.32), mi[b] - mi[a] + 0.9, 0.64, color=PH_COL[f], lw=0))
        ax0.text((mi[a] + mi[b]) / 2, y, FASE_LAB[f].split()[0], ha="center", va="center", fontsize=6.5, color=INK if f != "fill" else "white")
    ax0.text(-0.9, y, nome, ha="right", va="center", fontsize=7.5, color=INK)
ax0.set_xlim(-0.6, len(MESES) - 0.4); ax0.set_ylim(-0.6, len(cal) - 0.4)
ax0.set_xticks(range(len(MESES))); ax0.set_xticklabels(MESES); ax0.set_yticks([])
for s in ["left"]: ax0.spines[s].set_visible(False)
ax0.set_title("(a) Phenological windows used for each crop group (soybean and first-crop windows start in the calendar year before harvest)", loc="left", fontsize=8.5)
# response panels
d_c2 = F5[F5.series.str.startswith("CONAB second")]
painel_resposta(fig.add_subplot(gs[1, 0]), d_c2, "(b) Second-crop maize, CONAB series\n(MT, PR, MS, GO, SP, MG; n = 205)")
painel_resposta(fig.add_subplot(gs[1, 1]), F1b[F1b.group == "maize second crop"], "(c) Maize, IBGE total (first + second crop)\n(14 states; n = 504)")
painel_resposta(fig.add_subplot(gs[2, 0]), F1b[F1b.group == "soybean South"], "(d) Soybean, South\n(PR, SC, RS; n = 108)", xlim=(-32, 32))
painel_resposta(fig.add_subplot(gs[2, 1]), F1b[F1b.group == "soybean Centre-North"], "(e) Soybean, Centre-North\n(13 states; n = 447)", xlim=(-12, 12))
# legend
h = [plt.Line2D([], [], color=c, marker="o", lw=1.6, ms=4.5, mec="white", label=l) for _, c, l in FAM]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.01))
fig.savefig("figuras/Fig1_phase_response.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3
ORD = ["soybean South", "maize first crop (South)", "soybean Centre-North", "maize second crop", "maize second crop, six states"]
TIT = {"soybean South": "(a) Soybean, South", "maize first crop (South)": "(b) First-crop maize, South (RS, SC)",
       "soybean Centre-North": "(c) Soybean, Centre-North (13 states)", "maize second crop": "(d) Maize, 14 states outside the South", "maize second crop, six states": "(e) Second-crop maize, six main states"}
fig, axs = plt.subplots(3, 2, figsize=(7.4, 7.2), sharex=True); axs = axs.ravel(); axs[5].axis("off")
for ax, g in zip(axs, ORD):
    d = F2[F2.group == g]; ax.axvline(0, color=INK2, lw=0.8)
    for i, f in enumerate(FASES):
        for j, (fam, col, lab) in enumerate(FAM):
            r = d[(d.phase == f) & (d.variable == fam)].iloc[0]; y = 2 - i + (1 - j) * 0.22
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.6, solid_capstyle="round"); ax.plot(r.sd_per_degC, y, "o", color=col, ms=4.5, mec="white", mew=0.6)
            if r.p < 0.05: ax.text(r.hi + 0.02 if r.sd_per_degC >= 0 else r.lo - 0.02, y, f"{r.sd_per_degC:+.2f}", va="center", ha="left" if r.sd_per_degC >= 0 else "right", fontsize=7)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([FASE_LAB[f] for f in FASES]); ax.set_ylim(-0.6, 2.6); ax.set_xlim(-0.55, 1.0)
    ax.set_title(TIT[g], loc="left", fontsize=8.5); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for ax in [axs[3], axs[4]]: ax.set_xlabel("Change in phase variable per °C of NDJ ONI (SD)")
axs[2].tick_params(labelbottom=False)
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig("figuras/Fig3_enso_pathway.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4
SETS = ["season means (3)", "phase windows (9)", "phase windows + ONI", "monthly values (18)", "full set as submitted (6 season means + 18 monthly + ONI)"]
LAB = ["Season\nmeans (3)", "Phase\nwindows (9)", "Phases\n+ ONI (10)", "Monthly\nvalues (18)", "Extended\nset (25)"]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
for ax, crop, tit in [(axs[0], "soybean", "(a) Soybean"), (axs[1], "maize", "(b) Maize (IBGE total)")]:
    for j, (m, col, lab) in enumerate([("ridge", "#52514e", "Ridge regression"), ("rf", "#256abf", "Random forest")]):
        d = F3[(F3.crop == crop) & (F3.model == m)].set_index("predictor_set").loc[SETS]
        x = np.arange(len(SETS)) + (j - 0.5) * 0.3
        ax.bar(x, d.skill, width=0.26, color=col, lw=0, label=lab)
        ax.errorbar(x, d.skill, yerr=[d.skill - d.lo, d.hi - d.skill], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0)
        for xi, v in zip(x, d.skill): ax.text(xi, max(v, 0) + 0.012, f"{v:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(range(len(SETS))); ax.set_xticklabels(LAB, fontsize=6.5)
    ax.set_title(tit, loc="left", fontsize=8.5); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(-0.15, 0.3)
axs[0].set_ylabel("Skill vs trend-only forecast (1 − MSE ratio)"); axs[0].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig("figuras/Fig4_skill_predictor_sets.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("ok")
