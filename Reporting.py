# reporting_generate_report_staticplot_header_improved.py
from pathlib import Path
import pandas as pd
import re
import sys
import plotly.express as px
from datetime import datetime
import weather_data

# Pillow for accurate logo sizing
from PIL import Image as PILImage

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle

# ---------- USER CONFIG ----------
INPUT_DIR = Path(r"C:\Users\pauls\OneDrive\Desktop\USC\Fall '25\QT Insite Automation\10_14_Test_750V\10_14_Test_750V\10_14_Test-2000")
OUTPUT_DIR = INPUT_DIR / "Summary"
OUTPUT_CSV = OUTPUT_DIR / "measurements_summary.csv"
OUTPUT_PLOT_PNG = OUTPUT_DIR / "resistance_plot.png"
OUTPUT_PDF = OUTPUT_DIR / "Circuit_Integrity_Report.pdf"
LOGO_FILENAME = "AtlanticElectricLogo.png"  # expected in cwd
COMPANY_NAME = "Atlantic Electric"
FNAME_TS_RE = re.compile(r'(\d{8})_(\d{6})')
# PNG size (pixels) for a high-quality static plot
PNG_WIDTH = 1400
PNG_HEIGHT = 700
# Max logo size in inches (keeps it small and proportional)
MAX_LOGO_WIDTH_IN = 1.6
MAX_LOGO_HEIGHT_IN = 1.0
# ----------------------------------

def extract_datetime_from_filename(fname: str):
    m = FNAME_TS_RE.search(fname)
    if not m:
        return None
    date_part, time_part = m.group(1), m.group(2)
    try:
        return pd.to_datetime(f"{date_part}_{time_part}", format="%Y%m%d_%H%M%S")
    except Exception:
        return None

# --- replace collect_measurements() in your script with this ---
def collect_measurements():
    """Read CSVs and return DataFrame with datetime, resistance, plus day/time columns."""
    if not INPUT_DIR.exists():
        print("INPUT_DIR does not exist:", INPUT_DIR)
        sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(INPUT_DIR.glob("*.csv")):
        dt = extract_datetime_from_filename(f.name)
        try:
            df = pd.read_csv(f, skiprows=1)
            raw_val = df["Final Measurement (A/Ω)"].iloc[0]
            resistance = pd.to_numeric(raw_val, errors="coerce")
        except Exception as e:
            print(f"{f.name}: ERROR -> {e}")
            resistance = pd.NA

        rows.append({"datetime": dt, "resistance": resistance})
        print(f"{f.name} | datetime: {dt} | resistance: {resistance}")

    result_df = pd.DataFrame(rows)
    # ensure proper datetimes
    result_df["datetime"] = pd.to_datetime(result_df["datetime"], errors="coerce")
    result_df = result_df.sort_values("datetime").reset_index(drop=True)

    # NEW: add day and time_str columns
    # day: YYYY-MM-DD (string) — useful for grouping or coloring by day
    # time_str: HH:MM (string) — text label for ticks / hover
    result_df["day"] = result_df["datetime"].dt.strftime("%Y-%m-%d")
    result_df["time_str"] = result_df["datetime"].dt.strftime("%H:%M")

    # save CSV (includes new columns)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved summary to: {OUTPUT_CSV}\n")
    return result_df


def create_static_plot_png(df: pd.DataFrame):
    """
    Static Plotly PNG:
    - HH:MM tick labels (tickvals/ticktext)
    - dashed vertical line at each midnight
    - small MM/DD label under the FIRST measurement of each day
    - Atlantic Electric color theme
    """
    import plotly.graph_objects as go
    import pandas as pd
    from datetime import timedelta

    if df.empty:
        print("No data to plot.")
        return False

    # brand colors
    red = "#A6192E"
    dark_gray = "#231F20"
    light_gray = "#E6E6E6"

    # ensure proper datetimes and sorting
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    x = df["datetime"].tolist()
    y = df["resistance"].tolist()

    start = df["datetime"].min()
    end = df["datetime"].max()

    # choose tick frequency based on span
    span_seconds = (end - start).total_seconds()
    if span_seconds <= 2 * 3600:
        freq = "10min"
    elif span_seconds <= 6 * 3600:
        freq = "15min"
    elif span_seconds <= 24 * 3600:
        freq = "30min"
    else:
        freq = "1H"

    # align tick_start to freq boundary
    tick_start = pd.to_datetime(start)
    if freq.endswith("H"):
        tick_start = tick_start.floor("H")
    else:
        minutes = int(freq.replace("min", ""))
        tick_start = tick_start - pd.Timedelta(minutes=(tick_start.minute % minutes),
                                               seconds=tick_start.second,
                                               microseconds=tick_start.microsecond)

    tick_vals = pd.date_range(start=tick_start, end=end + pd.Timedelta(minutes=1), freq=freq).to_pydatetime().tolist()
    tick_text = [t.strftime("%H:%M") for t in tick_vals]

    # day boundaries (midnight) shapes
    days = pd.to_datetime(df["datetime"]).dt.floor("D").unique()
    shapes = []
    for d in days:
        d = pd.to_datetime(d)
        if d >= start and d <= end:
            shapes.append(dict(
                type="line", x0=d, x1=d, xref="x",
                y0=0, y1=1, yref="paper",
                line=dict(color=light_gray, width=1, dash="dash"),
                opacity=0.6
            ))

    # Determine the first measurement datetime for each day (to place MM/DD labels)
    first_per_day = (
        df.groupby(df["datetime"].dt.floor("D"), sort=True)["datetime"]
          .min()
          .dt.to_pydatetime()
          .tolist()
    )

    # Build annotations: small MM/DD under first occurrence of each day
    annotations = []
    for dt in first_per_day:
        if dt >= start and dt <= end:
            annotations.append(dict(
                x=dt,
                y=-0.12,
                xref="x",
                yref="paper",
                text=dt.strftime("%m/%d"),
                showarrow=False,
                font=dict(size=12, color="black")
            ))

    # Main figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        line=dict(width=2, color=red),
        marker=dict(size=6, color=red),
        hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Resistance: %{y:,.2f} Ω<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_white",
        title="Resistance Measurements Over Time",
        title_font=dict(size=18, family="Arial", color=dark_gray),
        font=dict(family="Arial", size=12, color=dark_gray),
        margin=dict(l=60, r=30, t=80, b=110),
        width=PNG_WIDTH,
        height=PNG_HEIGHT,
        shapes=shapes,
        annotations=annotations
    )

    # Force HH:MM labels via explicit tickvals/ticktext
    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=0,
        showgrid=True,
        gridcolor=light_gray,
        title="Datetime"
    )

    fig.update_yaxes(
        title="Resistance (Ohms)",
        showgrid=True,
        gridcolor=light_gray,
        tickformat="~s"
    )

    # small left/right padding
    fig.update_xaxes(range=[start - timedelta(minutes=1), end + timedelta(minutes=1)])

    # Export static PNG (requires kaleido)
    try:
        fig.write_image(str(OUTPUT_PLOT_PNG), width=PNG_WIDTH, height=PNG_HEIGHT, scale=1)
        print(f"Wrote static PNG plot to: {OUTPUT_PLOT_PNG}")
        return True
    except Exception as e:
        print("ERROR: Plotly static PNG export failed. Install kaleido: pip install kaleido")
        print("Plot export error:", e)
        return False

def integrate_weather(df: pd.DataFrame, output_dir: Path, icao: str = "KCAE") -> Path | None:
    """
    Fetch hourly weather for the date range in df, match to resistance datetimes
    (nearest hour), create a weather PNG at output_dir/"weather_plot.png", and
    return the Path on success or None on failure.
    Requires weather_data.py with:
      get_hourly_weather(start_date, end_date, icao_code)
      match_weather_to_resistance(res_df, hourly_df)
      create_weather_plot_png(matched_weather_df, daily_df, out_png)
    """
    from pathlib import Path
    import pandas as pd
    import weather_data

    try:
        if df is None or df.empty:
            print("No resistance measurements; skipping weather integration.")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weather_png = output_dir / "weather_plot.png"

        # derive date range from df datetimes (use local dates)
        start_dt = pd.to_datetime(df["datetime"].min())
        end_dt = pd.to_datetime(df["datetime"].max())
        if pd.isna(start_dt) or pd.isna(end_dt):
            print("Could not determine start/end dates from measurements; skipping weather.")
            return None

        start = start_dt.date().strftime("%Y-%m-%d")
        end = end_dt.date().strftime("%Y-%m-%d")

        print(f"Fetching weather for {start} -> {end} at ICAO {icao} ...")
        try:
            # Some versions expect named param icao_code, some expect positional
            hourly_df, daily_df, airport_name = weather_data.get_hourly_weather(start, end, icao_code=icao)
        except TypeError:
            hourly_df, daily_df, airport_name = weather_data.get_hourly_weather(start, end, icao)

        if hourly_df is None or hourly_df.empty:
            print("Hourly weather fetch returned no data; skipping weather plot.")
            return None

        matched = weather_data.match_weather_to_resistance(df, hourly_df)
        if matched is None or matched.empty:
            print("No matched weather rows for resistance datetimes; skipping weather plot.")
            return None

        created = weather_data.create_weather_plot_png(matched, daily_df, str(weather_png))
        if created:
            print(f"Weather plot created at: {weather_png}")
            return weather_png
        else:
            print("Weather plot creation failed.")
            return None

    except Exception as exc:
        print("Weather integration failed:", exc)
        return None


def _build_logo_rlimage(logo_path: Path):
    """
    Load logo with Pillow, compute scaled drawWidth/drawHeight in points (ReportLab),
    and return an RL Image object sized to fit within MAX_LOGO_*_IN while preserving aspect.
    """
    if not logo_path.exists():
        return None

    try:
        pil = PILImage.open(str(logo_path))
    except Exception as e:
        print("Warning: Pillow cannot open logo:", e)
        return None

    # pixel dimensions
    w_px, h_px = pil.size
    # attempt to get DPI; default to 72 if not present
    dpi_x, dpi_y = pil.info.get("dpi", (72, 72))
    if dpi_x == 0:
        dpi_x = 72
    if dpi_y == 0:
        dpi_y = 72

    # convert to inches
    w_in = w_px / dpi_x
    h_in = h_px / dpi_y

    # scale to fit in max footprint while preserving aspect ratio
    scale = min(MAX_LOGO_WIDTH_IN / w_in, MAX_LOGO_HEIGHT_IN / h_in, 1.0)  # don't upscale
    draw_w_pts = (w_in * scale) * inch
    draw_h_pts = (h_in * scale) * inch

    try:
        rl_img = RLImage(str(logo_path))
        rl_img.drawWidth = draw_w_pts
        rl_img.drawHeight = draw_h_pts
        return rl_img
    except Exception as e:
        print("Warning: ReportLab failed to create RL Image:", e)
        return None

def build_pdf_report(df: pd.DataFrame, png_available: bool):
    """
    Build PDF report using Atlantic Electric red theme.
    - Red header for table (#A6192E)
    - 'N/A' for missing values
    - Formats resistance into kΩ, MΩ, or GΩ
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle

    doc = SimpleDocTemplate(str(OUTPUT_PDF), pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # Header: logo left, title text right (aligned top-right)
    logo_path = Path.cwd() / LOGO_FILENAME
    logo_rl = _build_logo_rlimage(logo_path)
    left_cell = logo_rl if logo_rl else Paragraph("", styles["Normal"])

    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"],
                                 fontName="Helvetica-Bold", fontSize=18,
                                 leading=20, alignment=2)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
                                    fontSize=11, leading=14, alignment=2)
    small_grey = ParagraphStyle("SmallGrey", parent=styles["Normal"],
                                fontSize=9, textColor=colors.grey, alignment=2)

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    right_cell = [
        Paragraph(f"{COMPANY_NAME}", title_style),
        Paragraph("Circuit Integrity Report", subtitle_style),
        Paragraph(f"Report generated: {gen_time}", small_grey)
    ]

    header_table = Table([[left_cell, right_cell]],
                         colWidths=[1.6 * inch, 5.9 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 12))

        # Plot image (resistance)
    if png_available and OUTPUT_PLOT_PNG.exists():
        try:
            img = RLImage(str(OUTPUT_PLOT_PNG),
                          width=6.9 * inch, height=3.6 * inch)
            story.append(img)
            story.append(Spacer(1, 12))
        except Exception as e:
            print("Warning: failed to embed PNG plot:", e)
            story.append(Paragraph("Plot preview not available.",
                                   styles["Normal"]))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph(
            "Plot preview not available. Install kaleido and re-run to embed it.",
            styles["Normal"]))
        story.append(Spacer(1, 8))

    # --- Weather plot (embedded under resistance plot, if it exists) ---
    weather_png_path = OUTPUT_DIR / "weather_plot.png"
    if weather_png_path.exists():
        try:
            weather_img = RLImage(str(weather_png_path), width=6.9 * inch, height=2.6 * inch)
            story.append(weather_img)
            story.append(Spacer(1, 12))
        except Exception as e:
            print("Warning: failed to embed weather PNG:", e)
            # continue on to table


    # ---- Table ----
    table_data = [["Date", "Resistance"]]
    for _, row in df.iterrows():
        dt = row["datetime"]
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt) else ""
        res = row["resistance"]

        if pd.isna(res):
            res_str = "N/A"
        else:
            try:
                ohms = float(res)
                if ohms >= 1e9:
                    res_str = f"{ohms/1e9:,.2f} GΩ"
                elif ohms >= 1e6:
                    res_str = f"{ohms/1e6:,.2f} MΩ"
                elif ohms >= 1e3:
                    res_str = f"{ohms/1e3:,.2f} kΩ"
                else:
                    res_str = f"{ohms:,.0f} Ω"
            except Exception:
                res_str = "N/A"

        table_data.append([dt_str, res_str])

    # Atlantic Electric colors
    ATLANTIC_RED = colors.HexColor("#A6192E")
    DARK_GRAY = colors.HexColor("#231F20")

    table = Table(table_data, colWidths=[3.5 * inch, 3.5 * inch], repeatRows=1)
    table_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ATLANTIC_RED),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GRAY),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ])
    table.setStyle(table_style)
    story.append(table)
    story.append(Spacer(1, 12))

    note_style = ParagraphStyle("note", parent=styles["Normal"],
                                fontSize=9, textColor=colors.grey)
    story.append(Paragraph(
        "Notes: 'Final Measurement' values converted to appropriate scale (Ω, kΩ, MΩ, or GΩ). "
        "'N/A' indicates missing or invalid data.", note_style))

    try:
        doc.build(story)
        print(f"✅ PDF report created: {OUTPUT_PDF}")
    except Exception as e:
        print("ERROR creating PDF:", e)




def main():
    df = collect_measurements()
    png_ok = create_static_plot_png(df)

    # integrate weather (creates OUTPUT_DIR/weather_plot.png if successful)
    try:
        weather_png = integrate_weather(df, OUTPUT_DIR, icao="KCAE")
    except Exception as e:
        print("Weather integration threw an exception:", e)
        weather_png = None

    build_pdf_report(df, png_ok)


if __name__ == "__main__":
    main()
