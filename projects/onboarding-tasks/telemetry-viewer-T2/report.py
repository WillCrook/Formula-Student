"""Render editable output/report.md using a deliberately small Markdown subset."""
from pathlib import Path
import re
import io
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from PIL import Image as PILImage
ROOT=Path(__file__).resolve().parent

def render_report():
    source=ROOT/"output/report.md"
    target=ROOT/"output/pdf/telemetry_report_draft.pdf"
    target.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Body",fontName="Helvetica",fontSize=9,leading=12,spaceAfter=6))
    styles["Heading1"].fontSize=17
    styles["Heading1"].leading=20
    styles["Heading1"].spaceAfter=9
    styles["Heading2"].fontSize=11
    styles["Heading2"].leading=14
    styles["Heading2"].spaceBefore=5
    styles["Heading2"].spaceAfter=5
    styles["Heading1"].keepWithNext=1
    styles["Heading2"].keepWithNext=1
    styles["Body"].allowOrphans=0
    styles["Body"].allowWidows=0
    story=[]
    text=source.read_text()
    appendix=source.parent/"workbench_assumptions.md"
    if appendix.exists():
        text+="\n<!-- pagebreak -->\n"+appendix.read_text()
    lines=text.splitlines()
    width=A4[0]-72
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1; continue
        if line=="<!-- pagebreak -->":
            story.append(PageBreak())
        elif line.startswith("# "):
            story.append(Paragraph(escape(line[2:]),styles["Heading1"]))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]),styles["Heading2"]))
        elif line.startswith("!["):
            match=re.fullmatch(r"!\[(.*?)\]\((.*?)\)",line)
            path=source.parent/match[2]
            with PILImage.open(path) as im:
                w,h=im.size
            story.append(Image(str(path),width=width,height=width*h/w))
            story.append(Spacer(1,5))
        elif line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].startswith("|"):
                cells=[c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+",c) for c in cells):
                    rows.append(cells)
                i+=1
            t=Table(rows,colWidths=[width/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign="LEFT")
            t.setStyle(TableStyle([
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
                ("FONTSIZE",(0,0),(-1,-1),8),
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dcece9")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f7f7")]),
                ("BOTTOMPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),3),
                ("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#577c77"))]))
            story.extend([t,Spacer(1,6)])
            continue
        else:
            story.append(Paragraph(escape(line),styles["Body"]))
        i+=1
    def footer(canvas,doc):
        canvas.setFont("Helvetica",8)
        canvas.setFillColor(colors.HexColor("#526266"))
        canvas.drawString(36,22,"TBRe VDG | Task 2 | Review draft")
        canvas.drawRightString(A4[0]-36,22,str(doc.page))
    stream=io.BytesIO()
    doc=SimpleDocTemplate(stream,pagesize=A4,rightMargin=36,leftMargin=36,topMargin=30,bottomMargin=34,
                         title="TBRe VDG Task 2 - Telemetry analysis draft",author="Will Crook")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    temporary=target.with_suffix(".tmp.pdf")
    temporary.write_bytes(stream.getvalue())
    temporary.replace(target)
    print(target)
    return target

if __name__=="__main__":
    render_report()
