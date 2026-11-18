"""Build the PathSnap technical report PDF from the finalized aggregate results."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "PathSnap_Hackathon_Report.pdf"

NAVY = colors.HexColor("#10243A")
TEAL = colors.HexColor("#087E8B")
PALE = colors.HexColor("#EAF3F5")
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#596875")
LINE = colors.HexColor("#D8E0E5")
WHITE = colors.white

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=26, leading=30, textColor=NAVY, alignment=TA_LEFT, spaceAfter=5))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, textColor=MUTED, spaceAfter=12))
styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceBefore=2, spaceAfter=8))
styles.add(ParagraphStyle(name="Subsection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=TEAL, spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.7, leading=10, textColor=MUTED))
styles.add(ParagraphStyle(name="Metric", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=TEAL, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="MetricLabel", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.6, leading=9, textColor=MUTED, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="Cell", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10, textColor=INK))
styles.add(ParagraphStyle(name="CellHead", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.7, leading=10, textColor=WHITE))


def P(text, style="Body2"):
    return Paragraph(text, styles[style])


def make_table(rows, widths, header=True, font=7.8):
    converted = []
    for ri, row in enumerate(rows):
        converted.append([P(str(value), "CellHead" if header and ri == 0 else "Cell") for value in row])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for ri in range(1 if header else 0, len(rows)):
        if ri % 2 == (1 if header else 0):
            commands.append(("BACKGROUND", (0, ri), (-1, ri), PALE))
    table.setStyle(TableStyle(commands))
    return table


def page_chrome(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 11 * mm, w, 11 * mm, stroke=0, fill=1)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(18 * mm, h - 7.3 * mm, "PATHSNAP  /  TECHNICAL REPORT")
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 14 * mm, w - 18 * mm, 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18 * mm, 9 * mm, "Development results use heuristic route labels; evaluator data stayed outside model development.")
    canvas.drawRightString(w - 18 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=19 * mm, bottomMargin=19 * mm, title="PathSnap Hackathon Technical Report", author="PathSnap Team")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=page_chrome)])
    story = []

    # Page 1: problem, system design, data, features.
    story.extend([
        Spacer(1, 6 * mm),
        P("PathSnap", "CoverTitle"),
        P("GPS trajectory map matching with candidate scoring and HMM/Viterbi route reconstruction", "Subtitle"),
        P("Executive summary", "Section"),
        P("PathSnap converts noisy GPS observations into a likely directed-road route. A Bayesian-tuned Extra Trees model scores nearby road candidates; a Viterbi decoder uses those probabilities and road connectivity to produce a coherent sequence. The prediction service is exposed through FastAPI and a browser map interface.", "Body2"),
    ])
    metrics = [
        [P("76.03%", "Metric"), P("82.20%", "Metric"), P("6 variants", "Metric")],
        [P("Tuned Extra Trees classifier Top-1", "MetricLabel"), P("After HMM / Viterbi", "MetricLabel"), P("5 families + tuned Extra Trees", "MetricLabel")],
    ]
    mt = Table(metrics, colWidths=[doc.width / 3] * 3, rowHeights=[12 * mm, 12 * mm])
    mt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([mt, Spacer(1, 4 * mm), P("System flow", "Subsection")])
    flow_rows = [[P("GPS trip", "Cell"), P("Nearby directed road arcs", "Cell"), P("Seven-feature classifier", "Cell"), P("HMM / Viterbi", "Cell"), P("FastAPI + map UI", "Cell")]]
    flow = Table(flow_rows, colWidths=[doc.width / 5] * 5, rowHeights=[15 * mm])
    flow.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.extend([flow, Spacer(1, 4 * mm), P("Dataset and partitions", "Subsection")])
    data_rows = [
        ["Partition", "Trips", "GPS observations", "Use"],
        ["train", "80", "208,417", "Feature building, training, grouped CV, tuning"],
        ["student_test", "10", "19,160", "Development evaluation after model choice"],
        ["evaluator_test", "10", "Not inspected", "Reserved for final evaluator run; folder/file counts only"],
    ]
    story.extend([make_table(data_rows, [doc.width * .17, doc.width * .10, doc.width * .19, doc.width * .54]), Spacer(1, 3 * mm), P("Preprocessing and features", "Subsection"), P("Each trip supplies GPS points, road nodes, directed arcs, and a reference route. Coordinates are projected to a local metre-based plane. A spatial grid finds up to eight road-segment candidates within 80 m. A monotone route alignment labels candidate rows when the aligned route arc is within 50 m; observations outside this threshold remain unlabeled. The seven model inputs are time delta, GPS speed, candidate distance, road length, GPS movement bearing, road bearing, and bearing difference.", "Body2"), P("The training partition contains 159,244 route-labeled observations out of 208,417 GPS observations (76.40%). The student-test partition contains 18,846 route-labeled observations out of 19,160 (98.36%). Correct route candidates are absent from the candidate set for 2,026 training labels and 190 student-test labels; these are counted as prediction misses.", "Small")])

    story.extend([PageBreak(), P("Model selection", "Section"), P("Five model families were compared with five-fold GroupKFold so that entire trips stayed together. Observation-level Top-1 measures candidate ranking for each GPS point; route-level Top-1 after HMM is closer to the final map-matching objective. The permitted student-test development comparison was also used to choose the final configuration, so its results are not an untouched benchmark.", "Body2")])
    model_rows = [
        ["Model", "Top-1 mean +/- std", "Candidate macro F1", "Candidate log loss"],
        ["Logistic Regression", "75.63% +/- 2.87 pp", "0.7883", "0.2628"],
        ["Extra Trees", "75.61% +/- 2.61 pp", "0.7555", "0.3627"],
        ["Hist Gradient Boosting", "73.94% +/- 2.68 pp", "0.8113", "0.2274"],
        ["Random Forest", "73.89% +/- 2.85 pp", "0.7877", "0.2944"],
        ["Decision Tree", "70.42% +/- 3.40 pp", "0.7518", "0.6472"],
    ]
    story.extend([make_table(model_rows, [doc.width * .31, doc.width * .25, doc.width * .22, doc.width * .22]), Spacer(1, 4 * mm), P("Selected model", "Subsection"), P("The final artifact is Extra Trees with 160 trees, maximum depth 16, minimum leaf size 10, square-root feature sampling, and balanced class weights, fitted on all 80 training trips. On the five-fold baseline comparison, Logistic Regression and untuned Extra Trees were effectively tied on Top-1. A 10-trial Gaussian-process search over Extra Trees settings used three grouped folds on training trips only. Its best configuration scored 75.89% +/- 2.76 percentage points, a modest improvement over untuned Extra Trees (about 75.76%) and Logistic Regression (about 75.62%) on those same screening folds. The 10-trip student-test comparison then favored tuned Extra Trees after HMM route decoding.", "Body2"), P("Metric tradeoff", "Subsection"), P("Hist Gradient Boosting remains best on candidate macro F1 and log loss (0.8136 and 0.2340 on student-test). Tuned Extra Trees is selected for route reconstruction because it had the highest student-test weighted and per-trip HMM Top-1. These are different measures, and the route results are development-stage results that informed model choice.", "Body2"), P("The public repository contains the final model artifact and aggregate summaries. Raw data and large candidate tables remain local.", "Small")])

    story.extend([PageBreak(), P("Route decoding and evaluation", "Section"), P("The HMM uses the classifier probabilities as emission scores. Transitions receive the best score for staying on the same directed arc, a smaller score for moving to a connected directed arc, and a penalty for a disconnected jump. The sequence is restarted after a missing candidate or a GPS gap longer than 60 seconds.", "Body2")])
    eval_rows = [
        ["Student-test model", "Cand. F1", "Cand. log loss", "HMM weighted Top-1", "HMM macro by trip"],
        ["Logistic Regression", "0.7955", "0.2706", "81.55%", "84.45%"],
        ["Decision Tree", "0.7474", "0.6371", "76.33%", "78.33%"],
        ["Random Forest", "0.7816", "0.3088", "80.02%", "82.53%"],
        ["Extra Trees, untuned", "0.7517", "0.3747", "81.68%", "84.80%"],
        ["Hist Gradient Boosting", "0.8136", "0.2340", "79.62%", "82.21%"],
        ["Extra Trees, tuned", "0.7495", "0.3832", "82.20%", "85.19%"],
    ]
    story.extend([make_table(eval_rows, [doc.width * .29, doc.width * .12, doc.width * .15, doc.width * .22, doc.width * .22]), Spacer(1, 4 * mm), P("Development-test details", "Subsection"), P("Every comparison row used the same 10 student-test trips; fitting used the 80 training trips. There were 18,846 heuristic route-labeled observations and 190 observations without a positive candidate (1.01%); these count as misses. For tuned Extra Trees, classifier-only weighted Top-1 was 76.03%, rising to 82.20% after HMM. Mean continuity was 99.50%. Hist Gradient Boosting had better candidate F1 and log loss, while tuned Extra Trees produced the highest route scores in this development comparison. Local Folium maps include toggleable GPS, nearby-network, reference-route, and HMM-prediction layers; coordinate-bearing HTML stays local.", "Body2"), P("Deployment", "Subsection"), P("Laptop 1 runs `uvicorn api.app:app --host 0.0.0.0 --port 8000` with PATHSNAP_PARTITION set to evaluator_test and the evaluator data kept on that machine. Laptop 2 opens the host laptop's IP address in a browser. The API performs inference first and reads reference routes server-side only for evaluation; the UI receives the predicted route and aggregate score, never the reference labels.", "Body2"), P("Limitations and integrity", "Subsection"), P("All current labels are generated by a spatial alignment heuristic, not independently verified map-matching ground truth. Long gaps, repeated edges, and intersections can lower alignment coverage or assign an imperfect label. Results are provisional and measure agreement with this heuristic. The evaluator partition was not used for fitting or tuning; only the student-test development comparison informed model selection.", "Body2"), P("Team split: Person 1 - data and features; Person 2 - model comparison and tuning; Person 3 - HMM/Viterbi decoding; Person 4 - FastAPI, browser UI, and two-laptop demonstration.", "Small")])

    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
