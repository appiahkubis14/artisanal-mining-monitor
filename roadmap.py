from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

pdf_path = "Planetary_PhD_Roadmap_2027.pdf"

doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleStyle",
    parent=styles["Heading1"],
    alignment=TA_CENTER,
    fontSize=24,
    leading=30,
    textColor=colors.HexColor("#123B5D"),
    spaceAfter=20,
)

heading_style = ParagraphStyle(
    "HeadingStyle",
    parent=styles["Heading2"],
    fontSize=16,
    leading=22,
    textColor=colors.HexColor("#123B5D"),
    spaceBefore=16,
    spaceAfter=10,
)

body_style = ParagraphStyle(
    "BodyStyle",
    parent=styles["BodyText"],
    fontSize=11,
    leading=18,
    spaceAfter=10,
)

quote_style = ParagraphStyle(
    "QuoteStyle",
    parent=styles["BodyText"],
    fontSize=11,
    leading=18,
    leftIndent=18,
    rightIndent=18,
    backColor=colors.HexColor("#F4F4F4"),
    borderPadding=12,
    textColor=colors.HexColor("#333333"),
    spaceAfter=16,
)

elements = []

title = Paragraph(
    "Planetary Science PhD Roadmap for Samuel",
    title_style
)
elements.append(title)

intro = """
This roadmap outlines a realistic pathway from Earth Observation and multi-sensor
fusion engineering into planetary science research in North America. The roadmap
assumes completion of your master’s degree in 2027 and focuses on transitioning
into a fully funded PhD programme in the United States or Canada.
"""
elements.append(Paragraph(intro, body_style))

elements.append(Paragraph("1. Statement of Purpose Draft", heading_style))

sop = """
“I grew up in Ghana, where smallholder farmers feed millions of people but often
have no early warning when drought will strike. I decided to build it myself.

Using Sentinel-2 data and AI-driven Earth Observation workflows, I built five
operational systems: crop type mapping at 10 m resolution, NDVI forecasting with
LSTM attention, UAV-based plant counting using YOLOv8, illegal mining detection
using U-Net and SAR change tracking, and road condition assessment with monocular
depth estimation.

These projects taught me how to fuse optical imagery, SAR, depth maps, and IMU data
into operational digital twin architectures capable of functioning in difficult environments.

I am not simply an agriculture student. I am a multi-sensor fusion engineer.

Now I want to apply these same skills to planetary surfaces. A PhD at [University X]
working with [Professor Y] would allow me to adapt my Earth digital twin systems to Mars,
using HiRISE, CRISM, and MRO datasets for terrain hazard detection, rover navigation,
surface change analysis, and orbiter–rover data fusion.

My long-term goal is to become a planetary research engineer at NASA JPL or ESA,
building digital twins that support future Mars and lunar missions.

My open-source systems demonstrate that I can build.
My presentations demonstrate that I can communicate.
My publications demonstrate that I can publish.
A PhD at [University X] is the next step.”
"""
elements.append(Paragraph(sop, quote_style))

elements.append(Paragraph("2. Updated Timeline (2026–2035)", heading_style))

timeline_data = [
    ["Year", "Milestone", "Goal"],
    ["2026", "Begin Toulouse research and thesis", "Develop planetary-transferable research"],
    ["2027", "Complete master’s degree", "Graduate with publications and strong thesis"],
    ["Late 2026 – Early 2027", "Submit PhD applications", "Apply to 10–12 planetary programmes"],
    ["Mid 2027", "Receive admissions decisions", "Accept funded PhD offer"],
    ["Fall 2027", "Begin PhD programme", "Start planetary science research"],
    ["2028–2029", "Qualifying exams and research", "Publish planetary-focused papers"],
    ["2030–2032", "Core PhD thesis research", "Develop Mars digital twin architectures"],
    ["2032–2033", "Graduate and pursue postdoc", "Apply for NASA/ESA fellowships"],
    ["2034–2035", "Transition into research role", "NASA JPL, APL, ESA, CSA, or academia"]
]

timeline_table = Table(
    timeline_data,
    colWidths=[1.4*inch, 2.7*inch, 2.7*inch],
    repeatRows=1
)

timeline_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B5D")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
elements.append(timeline_table)
elements.append(Spacer(1, 16))

elements.append(Paragraph("3. Target University Strategy", heading_style))

uni_text = """
<b>Reach Schools</b><br/>
Caltech, MIT, Stanford, Johns Hopkins APL

<br/><br/>

<b>Target Schools</b><br/>
University of Colorado Boulder, Arizona State University,
University of Arizona, Brown University, UT Austin

<br/><br/>

<b>Safety Schools</b><br/>
Purdue University, Georgia Tech,
Western University (Canada), York University (Canada)

<br/><br/>

Recommended strategy: Apply to 10–12 programmes total.
"""
elements.append(Paragraph(uni_text, body_style))

elements.append(Paragraph("4. Recommendation Letter Strategy", heading_style))

letters_data = [
    ["Letter Source", "Purpose"],
    ["Bachelor’s Supervisor (KNUST)", "Confirms independent engineering and research ability"],
    ["Master’s Supervisor (Europe)", "Validates advanced research maturity and technical depth"],
    ["External Researcher (ESA/CNES/Conference Contact)", "Provides international academic credibility"]
]

letters_table = Table(
    letters_data,
    colWidths=[3.2*inch, 3.0*inch],
    repeatRows=1
)

letters_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5D3A")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]))
elements.append(letters_table)

elements.append(Paragraph("5. Funding Strategy", heading_style))

funding = """
Most top US and Canadian PhD programmes in planetary science fully fund admitted students.

Funding generally includes:
• Full tuition waiver<br/>
• Annual stipend (US: $30k–$50k, Canada: $25k–$35k)<br/>
• Research or teaching assistantship<br/>
• Health insurance support

External scholarships such as Fulbright, Vanier Canada, Erasmus+, or DAAD can help,
but the main strategy should be admission into fully funded programmes.
"""
elements.append(Paragraph(funding, body_style))

elements.append(Paragraph("6. Immigration and Career Positioning", heading_style))

immigration = """
The United States provides the strongest direct access to NASA-connected planetary research,
especially through JPL, APL, and major planetary science centres.

Canada offers easier permanent residence pathways while maintaining strong access
to North American research networks and NASA collaborations.

Recommended strategy:
• Primary target: US planetary science PhD programmes<br/>
• Secondary backup: Canadian planetary science programmes<br/>
• Long-term goal: NASA JPL, NASA postdoctoral fellowships, ESA, or CSA
"""
elements.append(Paragraph(immigration, body_style))

elements.append(Paragraph("7. Final Strategic Positioning", heading_style))

final_text = """
Your strongest competitive advantage is your demonstrated ability to independently
build operational multi-sensor Earth Observation systems.

Your projects already demonstrate:
• Surface change detection<br/>
• Digital twin engineering<br/>
• Autonomous anomaly detection<br/>
• AI-based remote sensing workflows<br/>
• Multi-modal sensor fusion<br/>
• Operational deployment experience

The next step is translating these Earth-based systems into planetary science language
through publications, conference presentations, and a strong master’s thesis.

Your Ghanaian engineering background, Copernicus training,
AI engineering experience, and operational EO systems create a rare and highly
compelling profile for planetary science PhD admissions.
"""
elements.append(Paragraph(final_text, body_style))

doc.build(elements)

print(f"PDF successfully created at: {pdf_path}")
