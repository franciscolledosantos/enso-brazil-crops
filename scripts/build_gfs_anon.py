"""Builds the Ecological Informatics manuscript: title page, abstract, keywords, main text double
spaced with continuous line numbers and page numbers, declarations, references, tables and figures at the end.
Source: manuscript_ijc.md. Figures: ../figuras/en/. Output: Manuscript_IJC_rainfall_RJ.docx"""
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

A = Path(__file__).resolve().parent.parent
FIG = A / "figuras"
FIGS = {1: "Fig1_brazil_share.png", 2: "Fig2_enso_effect_map.png", 3: "Fig3_anomaly_vs_oni.png", 4: "Fig4_scenario_2026_27.png",
        5: "Fig5_skill_LOYO.png"}
md = (A / "manuscrito_GFS_anon.md").read_text(encoding="utf-8")

doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
    setattr(sec, m, Cm(2.5))
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 2.0
st.paragraph_format.space_after = Pt(0)

# continuous line numbers
ln = OxmlElement("w:lnNumType")
ln.set(qn("w:countBy"), "1")
ln.set(qn("w:restart"), "continuous")
ln.set(qn("w:distance"), "283")
sec._sectPr.append(ln)
# page number in footer
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    r = fp.add_run()
    if tag:
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag); r._r.append(e)
    else:
        e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; r._r.append(e)


def runs(p, text, size=None, bold=False):
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^)", text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("^"):
            r = p.add_run(part[1:-1]); r.font.superscript = True
        elif part.startswith("*"):
            r = p.add_run(part[1:-1]); r.italic = True
        else:
            r = p.add_run(part)
        if bold:
            r.bold = True
        if size:
            r.font.size = Pt(size)
    return p


def par(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, bold=False, spacing=None, indent=False, before=0, after=0):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    if spacing:
        pf.line_spacing = spacing
    if indent:
        pf.first_line_indent = Cm(1.0)
    return runs(p, text, size, bold)


def sem_numero(p):
    ppr = p._p.get_or_add_pPr()
    e = OxmlElement("w:suppressLineNumbers"); ppr.append(e)


def tabela(linhas):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in linhas if not re.match(r"^\|\s*-", l)]
    nc = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=nc)
    t.style = "Table Grid"
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            cell = t.cell(i, j); cell.text = ""
            p = cell.paragraphs[0]; p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            sem_numero(p)
            runs(p, c, 8 if nc > 8 else 9 if nc > 6 else 10, bold=(i == 0))
            p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(2), Pt(2)
    # larguras: primeira coluna larga (nomes), demais iguais
    tot = 16.0
    # larguras proporcionais ao conteúdo (maior palavra e comprimento médio por coluna)
    import math
    need = []
    for j in range(nc):
        cells = [r[j] for r in rows[1:]] or [rows[0][j]]
        longest_word = max((len(w) for c in cells + [rows[0][j]] for w in c.split()), default=4)
        avg = sum(len(c) for c in cells) / max(1, len(cells))
        need.append(max(1.4 * longest_word, avg, 4) + 1)
    larg = [tot * n / sum(need) for n in need]
    t.autofit = False
    tblPr = t._tbl.tblPr
    tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(int(tot * 567))); tw.set(qn("w:type"), "dxa"); tblPr.append(tw)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    grid = t._tbl.find(qn("w:tblGrid"))
    for g in list(grid):
        grid.remove(g)
    for w in larg:
        gc = OxmlElement("w:gridCol"); gc.set(qn("w:w"), str(int(w * 567))); grid.append(gc)
    for row in t.rows:
        for j, cell in enumerate(row.cells):
            tcPr = cell._tc.get_or_add_tcPr()
            for e in tcPr.findall(qn("w:tcW")):
                tcPr.remove(e)
            cw = OxmlElement("w:tcW"); cw.set(qn("w:w"), str(int(larg[j] * 567))); cw.set(qn("w:type"), "dxa"); tcPr.insert(0, cw)
    cm_ = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), "40"); e.set(qn("w:type"), "dxa"); cm_.append(e)
    t._tbl.tblPr.append(cm_)
    b = OxmlElement("w:tblBorders")
    for nome, val in [("top", "single"), ("bottom", "single"), ("insideH", "single"), ("left", "nil"), ("right", "nil"),
                      ("insideV", "nil")]:
        e = OxmlElement(f"w:{nome}"); e.set(qn("w:val"), val)
        if val == "single":
            e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000")
        b.append(e)
    t._tbl.tblPr.append(b)
    doc.add_paragraph()


# ---------- title page
meta = dict(re.findall(r"^% (\w+): (.*)$", md, flags=re.M))
par(meta["TITLE"], WD_ALIGN_PARAGRAPH.CENTER, 14, True, after=12)
par("Running title: " + meta["RUNNING"], WD_ALIGN_PARAGRAPH.LEFT, after=12)
(meta["AUTHORS"] != "-") and par(meta["AUTHORS"].replace(";", ","), WD_ALIGN_PARAGRAPH.LEFT, after=6)
for k in ("AFF1", "AFF2", "AFF3"):
    if k in meta:
        tx = meta[k]
        par("^" + tx[0] + "^" + tx[2:], WD_ALIGN_PARAGRAPH.LEFT, 11, spacing=1.15)
(meta["CORR"] != "-") and par(meta["CORR"], WD_ALIGN_PARAGRAPH.LEFT, after=6, before=12)
(meta["ORCID"] != "-") and par("ORCID: " + meta["ORCID"], WD_ALIGN_PARAGRAPH.LEFT)
par("**Keywords:** " + meta["KEYWORDS"], WD_ALIGN_PARAGRAPH.LEFT, before=12)

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ---------- body
corpo = md.split("# Abstract", 1)[1]
linhas = ("# Abstract" + corpo).split("\n")
i, em_refs, primeira_legenda = 0, False, True
while i < len(linhas):
    l = linhas[i]
    if not l.strip() or l.startswith("% "):
        i += 1; continue
    if l.startswith("|"):
        bloco = []
        while i < len(linhas) and linhas[i].startswith("|"):
            bloco.append(linhas[i]); i += 1
        tabela(bloco); continue
    m = re.match(r"^\[FIGURE (\d)\]", l)
    if m:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(FIG / FIGS[int(m.group(1))]), width=Cm(13 if m.group(1) == "1" else 16))
        sem_numero(p)
        i += 1; continue
    if l.startswith("# "):
        txt = l[2:].strip()
        if txt in ("References", "Tables", "Figures", "Supplementary material"):
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            primeira_legenda = True
        em_refs = txt == "References"
        par(txt, WD_ALIGN_PARAGRAPH.LEFT, 12, True, before=12, after=6)
        i += 1; continue
    if l.startswith("## "):
        par(l[3:].strip(), WD_ALIGN_PARAGRAPH.LEFT, 12, False, before=6, after=3).runs[0].italic = True
        i += 1; continue
    if l.startswith("**Table") or l.startswith("**Figure"):
        if not primeira_legenda:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        primeira_legenda = False
        p = par(l, WD_ALIGN_PARAGRAPH.LEFT, 11, spacing=1.15, before=6, after=6)
        p.paragraph_format.keep_with_next = True
        i += 1; continue
    if l.startswith("- "):
        p = par(l[2:], WD_ALIGN_PARAGRAPH.JUSTIFY)
        p.paragraph_format.left_indent = Cm(0.8); p.paragraph_format.first_line_indent = Cm(-0.4)
        p.runs[0].text = "• " + p.runs[0].text
        i += 1; continue
    if em_refs:
        p = par(l, WD_ALIGN_PARAGRAPH.LEFT)
        p.paragraph_format.left_indent = Cm(1.0); p.paragraph_format.first_line_indent = Cm(-1.0)
        i += 1; continue
    par(l, indent=True)
    i += 1

saida = A / "Manuscript_GFS_anonymized.docx"
doc.save(saida)
print("ok", saida.name)
