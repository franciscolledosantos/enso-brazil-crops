"""Gera Cover_letter_GlobalFoodSecurity.docx a partir de cover_letter_GFS.md (criado na revisão de 17/10/2026)."""
from docx import Document
from docx.shared import Pt, Cm
from pathlib import Path
A = Path(__file__).resolve().parent.parent
d = Document(); st = d.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(12)
for m in ('top_margin', 'bottom_margin', 'left_margin', 'right_margin'): setattr(d.sections[0], m, Cm(2.5))
for para in (A / 'cover_letter_GFS.md').read_text(encoding='utf-8').strip().split('\n\n'):
    p = d.add_paragraph(para.replace('\n', ' ')); p.paragraph_format.space_after = Pt(10)
d.save(A / 'Cover_letter_GlobalFoodSecurity.docx'); print('ok')
