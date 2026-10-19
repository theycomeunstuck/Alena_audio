"""Build the research report from the reviewed Markdown using bundled Python.

No generation calls, application mutations, or changes to frozen research inputs.
Visual QA remains a separate required step; see the report build instructions.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "research-report-2026-10-20"
SOURCE = REPORT / "Отчет_память_TeachCopilot.md"
OUTPUT = REPORT / "Отчет_память_TeachCopilot.docx"


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    el = OxmlElement("w:shd")
    el.set(qn("w:fill"), fill)
    tc_pr.append(el)


def set_font(style, size, bold=False):
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Times New Roman")


def table(doc, rows):
    cols = len(rows[0])
    tab = doc.add_table(rows=1, cols=cols)
    tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    tab.autofit = False
    widths = {
        (3, "Метод"): [1.0, 4.2, 1.7],
        (5, "Метод"): [1.2, 1.0, 0.8, 1.95, 1.95],
        (4, "Метод и стадия"): [2.6, 1.2, 1.2, 1.9],
        (5, "Сессии"): [0.7, 1.0, 0.8, 2.2, 2.2],
        (2, "Пункт задания"): [2.5, 4.4],
    }.get((cols, rows[0][0]), [6.9 / cols] * cols)
    for col, width in zip(tab.columns, widths):
        col.width = Inches(width)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement(f"w:{side}")
        for name, val in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            edge.set(qn(f"w:{name}"), val)
        borders.append(edge)
    tab._tbl.tblPr.append(borders)
    for i, row_data in enumerate(rows):
        row = tab.rows[0] if i == 0 else tab.add_row()
        no_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(no_split)
        if i == 0:
            repeat = OxmlElement("w:tblHeader")
            row._tr.get_or_add_trPr().append(repeat)
        for j, value in enumerate(row_data):
            cell = row.cells[j]
            cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tc_pr = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for side in ("top", "left", "bottom", "right"):
                edge = OxmlElement(f"w:{side}")
                edge.set(qn("w:w"), "85")
                edge.set(qn("w:type"), "dxa")
                margins.append(edge)
            tc_pr.append(margins)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.keep_with_next = False
            numeric = bool(re.fullmatch(r"[\d ,]+", value))
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if numeric else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(value)
            run.font.size = Pt(10.5)
            run.bold = i == 0
            if i == 0:
                shade(cell, "E7EDF2")
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def build(source=SOURCE, output=OUTPUT):
    text = source.read_text(encoding="utf-8")
    if "\u2014" in text:
        raise ValueError("Use en dash in report prose")
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(0.7)
    section.left_margin = section.right_margin = Inches(0.8)
    normal = doc.styles["Normal"]
    set_font(normal, 11.5)
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.widow_control = True
    for name, size in (("Title", 19), ("Subtitle", 12), ("Heading 1", 14), ("Heading 2", 12)):
        style = doc.styles[name]
        set_font(style, size, name != "Subtitle")
        style.paragraph_format.space_before = Pt(12 if name.startswith("Heading") else 0)
        style.paragraph_format.space_after = Pt(7)
        style.paragraph_format.keep_with_next = True
        ppr = style.element.get_or_add_pPr()
        for border in list(ppr.findall(qn("w:pBdr"))):
            ppr.remove(border)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith("# "):
            doc.add_paragraph(line[2:], "Title")
        elif line.startswith("## "):
            heading = line[3:]
            if heading.startswith(("1 ", "12 ")):
                doc.add_page_break()
            doc.add_paragraph(heading, "Heading 1")
        elif line.startswith("### "):
            doc.add_paragraph(line[4:], "Heading 2")
        elif line.startswith("|"):
            rows = [[x.strip() for x in line.strip("|").split("|")]]
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                i += 1
                if all(re.fullmatch(r"[:\- ]+", x) for x in row):
                    continue
                rows.append(row)
            table(doc, rows)
        elif line.startswith("- "):
            p = doc.add_paragraph(line[2:], "List Bullet")
            p.paragraph_format.space_after = Pt(4)
        else:
            p = doc.add_paragraph(line)
            if line.startswith("["):
                for run in p.runs:
                    run.font.size = Pt(10.5)
            if line in ("Отчёт по учебному проекту", "20 октября 2026 года"):
                p.paragraph_format.space_after = Pt(10)
    doc.core_properties.title = lines[0][2:]
    doc.core_properties.subject = "Сравнение четырёх методов памяти и прототип TeachCopilot"
    doc.core_properties.author = "Климов М.Ю."
    doc.core_properties.keywords = "TeachCopilot, долговременная память, образовательный ИИ"
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.source, args.output)
