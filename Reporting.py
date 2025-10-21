"""
Circuit Integrity Report Generator

This script generates a comprehensive PDF report for circuit integrity testing,
including resistance measurements over time and correlated weather data.

Features:
- Reads IR test CSV files and extracts resistance measurements
- Creates high-quality resistance vs time plots
- Integrates weather data from nearby airport stations
- Generates a professional PDF report with plots and tabular data

Author: Joseph Pauls
"""

from pathlib import Path
import pandas as pd
import re
import sys
from datetime import datetime, timedelta
import plotly.graph_objects as go
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
import weather_data

# ========== CONFIGURATION ==========
INPUT_DIR = Path(r"C:\Users\pauls\OneDrive\Desktop\USC\Fall '25\QT Insite Automation\10_14_Test_750V\10_14_Test_750V\10_14_Test-2000")
OUTPUT_DIR = INPUT_DIR / "Summary"
OUTPUT_CSV = OUTPUT_DIR / "measurements_summary.csv"
OUTPUT_PLOT_PNG = OUTPUT_DIR / "resistance_plot.png"
OUTPUT_PDF = OUTPUT_DIR / "Circuit_Integrity_Report.pdf"
LOGO_FILENAME = "AtlanticElectricLogo.png"
COMPANY_NAME = "Atlantic Electric"

# Plot dimensions (pixels)
PNG_WIDTH = 1400
PNG_HEIGHT = 700
WEATHER_HEIGHT = 500

# Logo constraints (inches)
MAX_LOGO_WIDTH_IN = 1.6
MAX_LOGO_HEIGHT_IN = 1.0

# Brand colors
ATLANTIC_RED = "#A6192E"
DARK_GRAY = "#231F20"
LIGHT_GRAY = "#E6E6E6"

# Filename pattern for extracting timestamps
FNAME_TS_RE = re.compile(r'(\d{8})_(\d{6})')
# ====================================

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


def _calculate_tick_frequency(start: datetime, end: datetime) -> str:
    """
    Determine appropriate tick frequency based on time span.
    
    Args:
        start: Start datetime
        end: End datetime
        
    Returns:
        Frequency string ('10min', '15min', '30min', or '1H')
    """
    span_hours = (end - start).total_seconds() / 3600
    
    if span_hours <= 2:
        return "10min"
    elif span_hours <= 6:
        return "15min"
    elif span_hours <= 24:
        return "30min"
    else:
        return "1H"


def _align_tick_start(start: datetime, freq: str) -> datetime:
    """
    Align tick start time to frequency boundary.
    
    Args:
        start: Start datetime
        freq: Frequency string
        
    Returns:
        Aligned datetime
    """
    tick_start = pd.to_datetime(start)
    
    if freq.endswith("H"):
        return tick_start.floor("H")
    else:
        minutes = int(freq.replace("min", ""))
        return tick_start - pd.Timedelta(
            minutes=(tick_start.minute % minutes),
            seconds=tick_start.second,
            microseconds=tick_start.microsecond
        )


def _create_day_separators(days: pd.DatetimeIndex, start: datetime, end: datetime) -> list:
    """
    Create vertical line shapes for day boundaries.
    
    Args:
        days: Unique day timestamps
        start: Plot start time
        end: Plot end time
        
    Returns:
        List of plotly shape dictionaries
    """
    shapes = []
    for day in days:
        day = pd.to_datetime(day)
        if start <= day <= end:
            shapes.append(dict(
                type="line", x0=day, x1=day, xref="x", y0=0, y1=1, yref="paper",
                line=dict(color=LIGHT_GRAY, width=1, dash="dash"), opacity=0.6
            ))
    return shapes


def _create_date_annotations(datetimes: pd.Series, start: datetime, end: datetime, 
                            y_position: float = -0.1) -> list:
    """
    Create MM/DD date label annotations for first measurement of each day.
    
    Args:
        datetimes: Series of datetime values
        start: Plot start time
        end: Plot end time
        y_position: Vertical position relative to plot (paper coordinates)
        
    Returns:
        List of plotly annotation dictionaries
    """
    first_per_day = (
        datetimes.dt.floor("D")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    
    # Map each day to its first occurrence in the original series
    first_times = datetimes.groupby(datetimes.dt.floor("D")).min()
    
    annotations = []
    for day in first_per_day:
        first_time = first_times.loc[day]
        if start <= first_time <= end:
            annotations.append(dict(
                x=first_time, y=y_position, xref="x", yref="paper",
                text=first_time.strftime("%m/%d"), showarrow=False,
                font=dict(size=16, family="Arial", color=DARK_GRAY)
            ))
    
    return annotations


def create_static_plot_png(df: pd.DataFrame) -> bool:
    """
    Generate resistance vs time plot as static PNG with Atlantic Electric branding.
    
    Creates a high-quality plotly chart showing resistance measurements over time,
    with time-of-day labels on the x-axis and MM/DD date markers below. Includes
    day separator lines and automatic tick frequency adjustment based on time span.
    
    Args:
        df: DataFrame with 'datetime' and 'resistance' columns
        
    Returns:
        True if plot created successfully, False otherwise
    """
    if df.empty:
        print("⚠️  No data to plot.")
        return False
    
    # Prepare data
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    x = df["datetime"].tolist()
    y = df["resistance"].tolist()
    start, end = df["datetime"].min(), df["datetime"].max()
    
    # Calculate time axis parameters
    freq = _calculate_tick_frequency(start, end)
    tick_start = _align_tick_start(start, freq)
    tick_vals = pd.date_range(
        start=tick_start, 
        end=end + pd.Timedelta(minutes=1), 
        freq=freq
    ).to_pydatetime().tolist()
    tick_text = [t.strftime("%H:%M") for t in tick_vals]
    
    # Create visual elements
    days = pd.to_datetime(df["datetime"]).dt.floor("D").unique()
    shapes = _create_day_separators(days, start, end)
    annotations = _create_date_annotations(df["datetime"], start, end, y_position=-0.1)
    
    # Build plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, 
        mode="lines+markers", 
        name="Resistance",
        line=dict(width=2.5, color=ATLANTIC_RED),
        marker=dict(size=7, color=ATLANTIC_RED),
        hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Resistance: %{y:,.2f} Ω<extra></extra>"
    ))
    
    # Configure layout
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Resistance Measurements Over Time",
            x=0.01, xanchor="left",
            font=dict(size=24, family="Arial", color=DARK_GRAY, weight="bold")
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, 
            xanchor="center", x=0.5,
            font=dict(size=16, family="Arial", color=DARK_GRAY)
        ),
        margin=dict(l=80, r=50, t=100, b=130),
        width=PNG_WIDTH,
        height=PNG_HEIGHT,
        shapes=shapes,
        annotations=annotations,
        font=dict(family="Arial", size=16, color=DARK_GRAY)
    )
    
    # Configure axes
    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=0,
        showgrid=True,
        gridcolor=LIGHT_GRAY,
        gridwidth=1,
        zeroline=False,
        title=dict(text="Datetime", font=dict(size=18, family="Arial", color=DARK_GRAY)),
        tickfont=dict(size=16, family="Arial", color=DARK_GRAY),
        range=[start - timedelta(minutes=1), end + timedelta(minutes=1)]
    )
    
    fig.update_yaxes(
        title=dict(text="Resistance (Ohms)", font=dict(size=18, family="Arial", color=DARK_GRAY)),
        tickfont=dict(size=16, family="Arial", color=DARK_GRAY),
        showgrid=True,
        gridcolor=LIGHT_GRAY,
        gridwidth=1,
        zeroline=False,
        tickformat="~s"
    )
    
    # Export to PNG
    try:
        fig.write_image(str(OUTPUT_PLOT_PNG), width=PNG_WIDTH, height=PNG_HEIGHT, scale=1)
        print(f"✅ Resistance plot saved: {OUTPUT_PLOT_PNG}")
        return True
    except Exception as e:
        print(f"❌ Plot export failed: {e}")
        print("   Install kaleido: pip install kaleido")
        return False

def integrate_weather(df: pd.DataFrame, output_dir: Path, icao: str = "KCAE") -> Path | None:
    """
    Fetch weather data and create weather plot for the measurement time range.
    
    Retrieves hourly weather data from OpenMeteo API for the nearest airport station,
    matches it to resistance measurement timestamps, and generates a weather plot
    PNG showing temperature and precipitation over time.
    
    Args:
        df: DataFrame with 'datetime' column defining the time range
        output_dir: Directory where weather_plot.png will be saved
        icao: ICAO airport code for weather station (default: "KCAE" for Columbia, SC)
        
    Returns:
        Path to the created weather plot PNG, or None if generation failed
        
    Dependencies:
        Requires weather_data.py module with:
        - get_hourly_weather(start_date, end_date, icao_code)
        - match_weather_to_resistance(res_df, hourly_df)
        - create_weather_plot_png(matched_df, daily_df, out_png, png_width, png_height)
    """
    try:
        # Validate input data
        if df is None or df.empty:
            print("No resistance measurements; skipping weather integration.")
            return None

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weather_png = output_dir / "weather_plot.png"

        # Extract date range from measurement datetimes
        start_dt = pd.to_datetime(df["datetime"].min())
        end_dt = pd.to_datetime(df["datetime"].max())
        
        if pd.isna(start_dt) or pd.isna(end_dt):
            print("Could not determine start/end dates from measurements; skipping weather.")
            return None

        start_date = start_dt.date().strftime("%Y-%m-%d")
        end_date = end_dt.date().strftime("%Y-%m-%d")

        # Fetch hourly weather data from API
        print(f"Fetching weather for {start_date} → {end_date} at ICAO {icao}...")
        try:
            # Try named parameter first (newer API)
            hourly_df, daily_df, airport_name = weather_data.get_hourly_weather(
                start_date, end_date, icao_code=icao
            )
        except TypeError:
            # Fall back to positional argument (older API)
            hourly_df, daily_df, airport_name = weather_data.get_hourly_weather(
                start_date, end_date, icao
            )

        # Validate weather data retrieved
        if hourly_df is None or hourly_df.empty:
            print("Hourly weather fetch returned no data; skipping weather plot.")
            return None

        # Match weather timestamps to resistance measurement timestamps
        matched = weather_data.match_weather_to_resistance(df, hourly_df)
        if matched is None or matched.empty:
            print("No matched weather rows for resistance datetimes; skipping weather plot.")
            return None

        # Generate weather plot PNG
        created = weather_data.create_weather_plot_png(
            matched, daily_df, str(weather_png), 
            png_width=PNG_WIDTH, png_height=WEATHER_HEIGHT
        )
        
        if created:
            print(f"✅ Weather plot created: {weather_png}")
            return weather_png
        else:
            print("❌ Weather plot creation failed.")
            return None

    except Exception as exc:
        print(f"❌ Weather integration failed: {exc}")
        return None


def _build_logo_rlimage(logo_path: Path):
    """
    Load and size logo image for PDF embedding.
    
    Loads logo using Pillow, calculates scaled dimensions to fit within maximum
    logo size constraints while preserving aspect ratio, and returns a ReportLab
    Image object ready for PDF insertion.
    
    Args:
        logo_path: Path to logo image file
        
    Returns:
        ReportLab Image object with proper sizing, or None if loading fails
        
    Notes:
        - Respects MAX_LOGO_WIDTH_IN and MAX_LOGO_HEIGHT_IN constants
        - Preserves aspect ratio
        - Will not upscale images smaller than max constraints
        - Defaults to 72 DPI if image has no DPI metadata
    """
    if not logo_path.exists():
        return None

    try:
        pil_img = PILImage.open(str(logo_path))
    except Exception as e:
        print(f"⚠️  Pillow cannot open logo: {e}")
        return None

    # Get pixel dimensions
    width_px, height_px = pil_img.size
    
    # Extract DPI from image metadata, default to 72 if not present
    dpi_x, dpi_y = pil_img.info.get("dpi", (72, 72))
    if dpi_x == 0:
        dpi_x = 72
    if dpi_y == 0:
        dpi_y = 72

    # Convert pixels to inches
    width_in = width_px / dpi_x
    height_in = height_px / dpi_y

    # Scale to fit within max footprint while preserving aspect ratio
    # (Don't upscale if image is already smaller than constraints)
    scale = min(MAX_LOGO_WIDTH_IN / width_in, MAX_LOGO_HEIGHT_IN / height_in, 1.0)
    draw_width_pts = (width_in * scale) * inch
    draw_height_pts = (height_in * scale) * inch

    # Create ReportLab Image object with calculated dimensions
    try:
        rl_img = RLImage(str(logo_path))
        rl_img.drawWidth = draw_width_pts
        rl_img.drawHeight = draw_height_pts
        return rl_img
    except Exception as e:
        print(f"⚠️  ReportLab failed to create Image: {e}")
        return None

def build_pdf_report(df: pd.DataFrame, png_available: bool):
    """
    Generate professional PDF report with Atlantic Electric branding.
    
    Creates a comprehensive PDF report including:
    - Company logo and report header
    - Resistance vs time plot (if available)
    - Weather plot (if available)
    - Tabular data with formatted resistance values
    
    Args:
        df: DataFrame with 'datetime' and 'resistance' columns
        png_available: Whether resistance plot PNG was successfully created
        
    Notes:
        - Uses Atlantic Electric brand colors (red #A6192E, dark gray #231F20)
        - Formats resistance values as kΩ, MΩ, or GΩ for readability
        - Displays 'N/A' for missing values
        - Embeds both resistance and weather plots if available
        - Saves PDF to OUTPUT_PDF path
    """
    # Initialize PDF document
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF), 
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()
    story = []

    # ========== HEADER SECTION ==========
    # Create logo element (left side)
    logo_path = Path.cwd() / LOGO_FILENAME
    logo_rl = _build_logo_rlimage(logo_path)
    left_cell = logo_rl if logo_rl else Paragraph("", styles["Normal"])

    # Create title text elements (right side, top-aligned)
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=18,
        leading=20, alignment=2  # alignment=2 is right-aligned
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=11, leading=14, alignment=2
    )
    small_grey = ParagraphStyle(
        "SmallGrey", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey, alignment=2
    )

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    right_cell = [
        Paragraph(f"{COMPANY_NAME}", title_style),
        Paragraph("Circuit Integrity Report", subtitle_style),
        Paragraph(f"Report generated: {gen_time}", small_grey)
    ]

    # Assemble header table
    header_table = Table(
        [[left_cell, right_cell]],
        colWidths=[1.6 * inch, 5.9 * inch]
    )
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 12))

    # ========== RESISTANCE PLOT ==========
    if png_available and OUTPUT_PLOT_PNG.exists():
        try:
            resistance_img = RLImage(
                str(OUTPUT_PLOT_PNG),
                width=6.9 * inch, 
                height=3.6 * inch
            )
            story.append(resistance_img)
            story.append(Spacer(1, 12))
        except Exception as e:
            print(f"⚠️  Failed to embed resistance plot: {e}")
            story.append(Paragraph(
                "Resistance plot preview not available.",
                styles["Normal"]
            ))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph(
            "Resistance plot preview not available. Install kaleido and re-run to embed it.",
            styles["Normal"]
        ))
        story.append(Spacer(1, 8))

    # ========== WEATHER PLOT ==========
    weather_png_path = OUTPUT_DIR / "weather_plot.png"
    if weather_png_path.exists():
        try:
            # Weather plot: 1400x500px embedded at 6.9"x2.5" to match aspect ratio
            weather_img = RLImage(
                str(weather_png_path), 
                width=6.9 * inch, 
                height=2.5 * inch
            )
            story.append(weather_img)
            story.append(Spacer(1, 12))
        except Exception as e:
            print(f"⚠️  Failed to embed weather plot: {e}")

    # ========== DATA TABLE ==========
    # Build table data with formatted resistance values
    table_data = [["Date", "Resistance"]]  # Header row
    
    for _, row in df.iterrows():
        # Format datetime
        dt = row["datetime"]
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt) else ""
        
        # Format resistance with appropriate units
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

    # Create and style table with Atlantic Electric branding
    table = Table(
        table_data, 
        colWidths=[3.5 * inch, 3.5 * inch], 
        repeatRows=1  # Repeat header on page breaks
    )
    
    atlantic_red = colors.HexColor("#A6192E")
    dark_gray = colors.HexColor("#231F20")
    
    table_style = TableStyle([
        # Header row styling
        ("BACKGROUND", (0, 0), (-1, 0), atlantic_red),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        
        # Data rows styling
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), dark_gray),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        
        # Grid
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ])
    table.setStyle(table_style)
    story.append(table)
    story.append(Spacer(1, 12))

    # ========== BUILD PDF ==========
    try:
        doc.build(story)
        print(f"✅ PDF report created: {OUTPUT_PDF}")
    except Exception as e:
        print(f"❌ PDF creation failed: {e}")




def main():
    """
    Main execution function for Circuit Integrity Report Generator.
    
    Orchestrates the complete report generation workflow:
    1. Collect resistance measurements from CSV files
    2. Generate resistance vs time plot
    3. Fetch and integrate weather data
    4. Build comprehensive PDF report
    
    Output files created in OUTPUT_DIR:
    - measurements_summary.csv: Raw measurement data
    - resistance_plot.png: Resistance over time visualization
    - weather_plot.png: Weather conditions over time (if successful)
    - Circuit_Integrity_Report.pdf: Complete professional report
    """
    print("=" * 60)
    print("Circuit Integrity Report Generator")
    print("=" * 60)
    
    # Step 1: Collect measurements from CSV files
    print("\n[1/4] Collecting measurements from CSV files...")
    df = collect_measurements()
    
    # Step 2: Generate resistance plot
    print("\n[2/4] Generating resistance plot...")
    png_ok = create_static_plot_png(df)
    
    # Step 3: Integrate weather data
    print("\n[3/4] Integrating weather data...")
    try:
        weather_png = integrate_weather(df, OUTPUT_DIR, icao="KCAE")
        if weather_png:
            print(f"     Weather data successfully integrated")
        else:
            print(f"     Weather integration skipped (no data available)")
    except Exception as e:
        print(f"⚠️   Weather integration error: {e}")
        weather_png = None
    
    # Step 4: Build PDF report
    print("\n[4/4] Building PDF report...")
    build_pdf_report(df, png_ok)
    
    print("\n" + "=" * 60)
    print("Report generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
