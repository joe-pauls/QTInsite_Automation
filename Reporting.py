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
import math
from datetime import datetime, timedelta
import plotly.graph_objects as go
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
import weather_data

# ========== CONFIGURATION ==========
DEFAULT_INPUT_DIR = Path(r"C:\Users\user\Desktop\QTInsite_Automation\QTInsite_Automation\1700VDwell60")
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_DIR / "Summary"

INPUT_DIR = DEFAULT_INPUT_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_ROOT
OUTPUT_CSV = OUTPUT_DIR / "measurements_summary.csv"
OUTPUT_PLOT_PNG = OUTPUT_DIR / "resistance_plot.png"
OUTPUT_NORMALIZED_PLOT_PNG = OUTPUT_DIR / "resistance_normalized_plot.png"
OUTPUT_WEATHER_PLOT_PNG = OUTPUT_DIR / "weather_plot.png"
OUTPUT_PDF = OUTPUT_DIR / "Circuit_Integrity_Report.pdf"
LOGO_FILENAME = "AtlanticElectricLogo.png"
COMPANY_NAME = "Atlantic Electric"

# Plot dimensions (pixels)
PNG_WIDTH = 1100
PNG_HEIGHT = 580
WEATHER_HEIGHT = 420

PLOT_IMAGE_WIDTH_IN = 6.85
RESISTANCE_HEIGHT_SCALE = 0.9
RESISTANCE_PLOT_HEIGHT_IN = PLOT_IMAGE_WIDTH_IN * (PNG_HEIGHT / PNG_WIDTH) * RESISTANCE_HEIGHT_SCALE
WEATHER_PLOT_HEIGHT_IN = PLOT_IMAGE_WIDTH_IN * (WEATHER_HEIGHT / PNG_WIDTH)

DATE_LABEL_Y_POSITION = -0.18
STANDARD_PLOT_MARGIN = {
    "l": 80,
    "r": 110,
    "t": 80,
    "b": 100,
}

# Logo constraints (inches)
MAX_LOGO_WIDTH_IN = 1.6
MAX_LOGO_HEIGHT_IN = 1.0

# Brand colors
ATLANTIC_RED = "#A6192E"
DARK_GRAY = "#231F20"
LIGHT_GRAY = "#7C7C7E"
DEFAULT_RAIN_AXIS_MAX = 1.0
MINOR_SEPARATOR_HOURS = 6

# Filename pattern for extracting timestamps
FNAME_TS_RE = re.compile(r'(\d{8})_(\d{6})')
# ====================================

def configure_paths(*, input_dir: Path | str | None = None, output_dir: Path | str | None = None) -> Path:
    """Configure global input/output paths for the current report run."""
    global INPUT_DIR, OUTPUT_DIR, OUTPUT_CSV, OUTPUT_PLOT_PNG, OUTPUT_NORMALIZED_PLOT_PNG, OUTPUT_WEATHER_PLOT_PNG, OUTPUT_PDF

    if input_dir is not None:
        INPUT_DIR = Path(input_dir)
    else:
        INPUT_DIR = Path(INPUT_DIR)

    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir)
    else:
        OUTPUT_DIR = Path(DEFAULT_OUTPUT_ROOT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV = OUTPUT_DIR / "measurements_summary.csv"
    OUTPUT_PLOT_PNG = OUTPUT_DIR / "resistance_plot.png"
    OUTPUT_NORMALIZED_PLOT_PNG = OUTPUT_DIR / "resistance_normalized_plot.png"
    OUTPUT_WEATHER_PLOT_PNG = OUTPUT_DIR / "weather_plot.png"
    OUTPUT_PDF = OUTPUT_DIR / "Circuit_Integrity_Report.pdf"
    return OUTPUT_DIR


def get_default_output_root() -> Path:
    """Return the default directory where report artifacts are stored."""
    return Path(DEFAULT_OUTPUT_ROOT)


def generate_recent_weather_preview(
    *,
    days: int = 7,
    icao_code: str = "KCUB",
    output_dir: Path | str | None = None,
    rain_axis_max: float | None = None,
) -> Path | None:
    """Fetch recent weather from the API and create a standalone preview plot.

    Args:
        days: Number of days of history to include (default 7)
        icao_code: Weather station ICAO identifier (default KCUB)
        output_dir: Optional target directory for preview artifacts
        rain_axis_max: Optional rainfall axis ceiling in inches for scaling bars
    """
    if days <= 0:
        raise ValueError("days must be positive")

    # Determine date range
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    try:
        hourly_df, daily_df, airport_name = weather_data.get_hourly_weather(
            start_str, end_str, icao_code=icao_code
        )
    except RuntimeError as exc:  # Missing dependencies or API issues
        print(f"[ERROR] Weather preview failed: {exc}")
        return None
    except Exception as exc:
        print(f"[ERROR] Weather preview failed with unexpected error: {exc}")
        return None

    # Reuse weather plotting with minimal transformation
    preview_df = hourly_df.rename(columns={"date": "matched_hour"}).copy()
    preview_df["matched_hour"] = pd.to_datetime(preview_df["matched_hour"])

    if output_dir is None:
        target_dir = get_default_output_root() / "weather_previews"
    else:
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    out_path = target_dir / f"weather_preview_last_{days}_days.png"
    csv_path = target_dir / f"weather_preview_last_{days}_days.csv"

    # Export raw hourly weather data for inspection
    try:
        hourly_df.to_csv(csv_path, index=False)
    except Exception as exc:
        print(f"[WARN] Failed to save weather preview CSV: {exc}")
        csv_path = None

    ok = weather_data.create_weather_plot_png(
        preview_df,
        daily_df,
        str(out_path),
        title=f"{airport_name} Weather — Last {days} Days",
        rain_axis_max=rain_axis_max,
        station_label=airport_name,
    )

    if ok:
        print(f"[OK] Weather preview plot saved to: {out_path}")
        if csv_path:
            print(f"[OK] Weather preview CSV saved to: {csv_path}")
        return out_path

    return None


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

    return result_df


def _calculate_tick_frequency(start: datetime, end: datetime) -> str:
    """
    Determine appropriate tick frequency based on time span.
    
    Args:
        start: Start datetime
        end: End datetime
        
    Returns:
        Frequency string ('10min', '15min', '30min', or '1d')
    """
    span_hours = (end - start).total_seconds() / 3600
    
    if span_hours <= 2:
        return "10min"
    elif span_hours <= 6:
        return "15min"
    elif span_hours <= 24:
        return "30min"
    else:
        return "1d"


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
    freq_lower = freq.lower()

    if freq_lower.endswith("h"):
        return tick_start.floor("h")
    if freq_lower.endswith("d"):
        return tick_start.floor("d")
    if freq_lower.endswith("min"):
        minutes = int(freq_lower.replace("min", ""))
        return tick_start - pd.Timedelta(
            minutes=(tick_start.minute % minutes),
            seconds=tick_start.second,
            microseconds=tick_start.microsecond
        )
    return tick_start


def _create_day_separators(days: pd.DatetimeIndex, start: datetime, end: datetime) -> list:
    """Create vertical line shapes for day boundaries plus intermediate 6-hour guides."""
    shapes: list[dict] = []
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # Ensure we know the midnight boundaries that should get the stronger styling
    major_lines: set[pd.Timestamp] = set()
    for day in pd.to_datetime(days):
        boundary = pd.to_datetime(day).floor("D")
        if start_dt <= boundary <= end_dt:
            shapes.append(dict(
                type="line",
                x0=boundary,
                x1=boundary,
                xref="x",
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color=LIGHT_GRAY, width=1.2),  # Solid line
                opacity=0.45,
            ))
            major_lines.add(boundary)

    window_start = start_dt.floor("D")
    window_end = end_dt.ceil("D")
    minor_times = pd.date_range(start=window_start, end=window_end, freq=f"{MINOR_SEPARATOR_HOURS}H")

    for ts in minor_times:
        if ts in major_lines:
            continue
        if start_dt <= ts <= end_dt:
            shapes.append(dict(
                type="line",
                x0=ts,
                x1=ts,
                xref="x",
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color=LIGHT_GRAY, width=1.0, dash="dash"),  # Dashed line
                opacity=0.45,
            ))

    return shapes


def _create_date_annotations(datetimes: pd.Series, start: datetime, end: datetime,
                            y_position: float = DATE_LABEL_Y_POSITION) -> list:
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


def _infer_circuit_configuration() -> str:
    """Infer circuit grounding from file naming conventions."""
    try:
        first_file = next(INPUT_DIR.glob("*.csv"))
    except StopIteration:
        return "Unknown"

    name = first_file.stem.lower()
    if "ungrounded" in name or "un_grounded" in name:
        return "Ungrounded"
    if "grounded" in name:
        return "Grounded"
    return "Unknown"


def _format_duration(delta: timedelta | pd.Timedelta | None) -> str:
    if delta is None:
        return "N/A"

    td = pd.to_timedelta(delta, errors="coerce")
    if pd.isna(td):
        return "N/A"

    seconds = int(td.total_seconds())
    if seconds <= 0:
        return "N/A"

    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")

    if not parts:
        return "Less than 1 minute"
    return ", ".join(parts)


def _format_interval(delta: timedelta | pd.Timedelta | None) -> str:
    if delta is None:
        return "N/A"

    td = pd.to_timedelta(delta, errors="coerce")
    if pd.isna(td):
        return "N/A"

    total_seconds = td.total_seconds()
    if total_seconds <= 0:
        return "N/A"

    # Round to the nearest minute
    total_minutes = round(total_seconds / 60)

    if total_minutes == 0:
        return "Less than 1 minute"
    
    days, remainder_minutes = divmod(total_minutes, 1440)  # 1440 minutes in a day
    hours, minutes = divmod(remainder_minutes, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")

    return ", ".join(parts) if parts else "Less than 1 minute"


def _describe_sampling_frequency(datetimes: pd.Series | None) -> str:
    if datetimes is None:
        return "N/A"

    if isinstance(datetimes, pd.Series):
        series = datetimes
    else:
        series = pd.Series(datetimes)

    times = pd.to_datetime(series, errors="coerce").dropna().sort_values().drop_duplicates()
    if times.shape[0] < 2:
        return "N/A"

    intervals = times.diff().dropna()
    if intervals.empty:
        return "N/A"

    median_delta = intervals.median()
    return f"Every {_format_interval(median_delta)}"


def _format_resistance_value(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        ohms = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if ohms >= 1e9:
        return f"{ohms / 1e9:,.2f} GΩ"
    if ohms >= 1e6:
        return f"{ohms / 1e6:,.2f} MΩ"
    if ohms >= 1e3:
        return f"{ohms / 1e3:,.2f} kΩ"
    return f"{ohms:,.0f} Ω"


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
        print("[WARN] No data to plot.")
        return False
    
    # Prepare data
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    x = df["datetime"].tolist()
    y = df["resistance"].tolist()
    start, end = df["datetime"].min(), df["datetime"].max()
    
    # Calculate time axis parameters
    measurement_count = len(df)
    use_day_only = measurement_count > 10

    tick_vals: list[datetime] = []
    tick_text: list[str] = []
    annotations: list[dict] = []

    if use_day_only:
        # > 10 measurements: Ticks at midnight are labeled "MM/DD",
        # and ticks at 6-hour intervals are labeled "HH:MM".
        major_ticks = df["datetime"].dt.floor("D").unique().to_pydatetime().tolist()

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        window_start = start_dt.floor("D")
        window_end = end_dt.ceil("D")
        all_hourly_ticks = pd.date_range(start=window_start, end=window_end, freq=f"{MINOR_SEPARATOR_HOURS}H")
        
        minor_ticks = []
        for ts in all_hourly_ticks:
            if ts.hour != 0 and start_dt <= ts <= end_dt:
                minor_ticks.append(ts.to_pydatetime())

        tick_vals = sorted(major_ticks + minor_ticks)
        tick_text = []
        for t in tick_vals:
            if t.hour == 0:
                tick_text.append(t.strftime("%m/%d"))
            else:
                tick_text.append(t.strftime("%H:%M"))
    else:
        # <= 10 measurements: Ticks are at measurement times, labeled "HH:MM".
        # MM/DD labels are added as annotations below the axis.
        tick_vals = df["datetime"].tolist()
        tick_text = [d.strftime("%H:%M") for d in tick_vals]
        
        if not df.empty:
            first_times_per_day = df.groupby(df["datetime"].dt.floor("D"))["datetime"].min()
            for dt in first_times_per_day:
                annotations.append(dict(
                    x=dt, y=DATE_LABEL_Y_POSITION, xref="x", yref="paper",
                    text=dt.strftime("%m/%d"), showarrow=False,
                    font=dict(size=16, family="Arial", color=DARK_GRAY)
                ))
    
    # Create visual elements
    days = pd.to_datetime(df["datetime"]).dt.floor("D").unique()
    shapes = _create_day_separators(days, start, end)
    
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
            text="Insulation Resistance Measurements Over Time",
            x=0.01, xanchor="left",
            font=dict(size=24, family="Arial", color=DARK_GRAY, weight="bold")
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, 
            xanchor="center", x=0.5,
            font=dict(size=16, family="Arial", color=DARK_GRAY)
        ),
    margin=dict(STANDARD_PLOT_MARGIN),
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
        print(f"[OK] Resistance plot saved: {OUTPUT_PLOT_PNG}")
        return True
    except Exception as e:
        print(f"[ERROR] Plot export failed: {e}")
        print("   Install kaleido: pip install kaleido")
        return False


def create_normalized_plot_png(df: pd.DataFrame) -> bool:
    """Create normalized resistance plot if normalized data is available."""
    if df.empty or "resistance_normalized_40c" not in df.columns:
        return False

    series = pd.to_numeric(df["resistance_normalized_40c"], errors="coerce")
    if series.dropna().empty:
        return False

    df_plot = df.copy()
    df_plot["datetime"] = pd.to_datetime(df_plot["datetime"])
    df_plot = df_plot.sort_values("datetime").reset_index(drop=True)

    x = df_plot["datetime"].tolist()
    y = series.tolist()
    start, end = df_plot["datetime"].min(), df_plot["datetime"].max()

    measurement_count = len(df_plot)
    use_day_only = measurement_count > 10

    tick_vals: list[datetime] = []
    tick_text: list[str] = []
    annotations: list[dict] = []

    if use_day_only:
        # > 10 measurements: Ticks at midnight are labeled "MM/DD",
        # and ticks at 6-hour intervals are labeled "HH:MM".
        major_ticks = df_plot["datetime"].dt.floor("D").unique().to_pydatetime().tolist()

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        window_start = start_dt.floor("D")
        window_end = end_dt.ceil("D")
        all_hourly_ticks = pd.date_range(start=window_start, end=window_end, freq=f"{MINOR_SEPARATOR_HOURS}H")
        
        minor_ticks = []
        for ts in all_hourly_ticks:
            if ts.hour != 0 and start_dt <= ts <= end_dt:
                minor_ticks.append(ts.to_pydatetime())

        tick_vals = sorted(major_ticks + minor_ticks)
        tick_text = []
        for t in tick_vals:
            if t.hour == 0:
                tick_text.append(t.strftime("%m/%d"))
            else:
                tick_text.append(t.strftime("%H:%M"))
    else:
        # <= 10 measurements: Ticks are at measurement times, labeled "HH:MM".
        # MM/DD labels are added as annotations below the axis.
        tick_vals = df_plot["datetime"].tolist()
        tick_text = [d.strftime("%H:%M") for d in tick_vals]
        
        if not df_plot.empty:
            first_times_per_day = df_plot.groupby(df_plot["datetime"].dt.floor("D"))["datetime"].min()
            for dt in first_times_per_day:
                annotations.append(dict(
                    x=dt, y=DATE_LABEL_Y_POSITION, xref="x", yref="paper",
                    text=dt.strftime("%m/%d"), showarrow=False,
                    font=dict(size=16, family="Arial", color=DARK_GRAY)
                ))

    days = pd.to_datetime(df_plot["datetime"]).dt.floor("D").unique()
    shapes = _create_day_separators(days, start, end)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name="Normalized Resistance",
        line=dict(width=2.5, color=ATLANTIC_RED),
        marker=dict(size=7, color=ATLANTIC_RED),
        hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Normalized: %{y:,.2f} Ω<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Insulation Resistance (Normalized to 40°C)",
            x=0.01,
            xanchor="left",
            font=dict(size=24, family="Arial", color=DARK_GRAY, weight="bold")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=16, family="Arial", color=DARK_GRAY)
        ),
    margin=dict(STANDARD_PLOT_MARGIN),
        width=PNG_WIDTH,
        height=PNG_HEIGHT,
        shapes=shapes,
        annotations=annotations,
        font=dict(family="Arial", size=16, color=DARK_GRAY)
    )

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
        tickfont=dict(size=16, family="Arial", color=DARK_GRAY),
        range=[start - pd.Timedelta(minutes=1), end + pd.Timedelta(minutes=1)]
    )

    fig.update_yaxes(
        title=dict(text="Normalized Resistance (Ohms)", font=dict(size=18, family="Arial", color=DARK_GRAY)),
        tickfont=dict(size=16, family="Arial", color=DARK_GRAY),
        showgrid=True,
        gridcolor=LIGHT_GRAY,
        gridwidth=1,
        zeroline=False,
        tickformat="~s"
    )

    try:
        fig.write_image(str(OUTPUT_NORMALIZED_PLOT_PNG), width=PNG_WIDTH, height=PNG_HEIGHT, scale=1)
        print(f"[OK] Normalized resistance plot saved: {OUTPUT_NORMALIZED_PLOT_PNG}")
        return True
    except Exception as exc:
        print(f"[ERROR] Normalized plot export failed: {exc}")
        return False

def integrate_weather(df: pd.DataFrame, output_dir: Path, icao: str = "KCUB") -> tuple[Path | None, pd.DataFrame | None, str | None]:
    """
    Fetch weather data and create weather plot for the measurement time range.
    
    Retrieves hourly weather data from OpenMeteo API for the nearest airport station,
    matches it to resistance measurement timestamps, and generates a weather plot
    PNG showing temperature and precipitation over time.
    
    Args:
        df: DataFrame with 'datetime' column defining the time range
        output_dir: Directory where weather_plot.png will be saved
        icao: ICAO airport code for weather station (default: "KCUB" for Columbia, SC)
        
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
            return None, None, None

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weather_png = OUTPUT_WEATHER_PLOT_PNG

        # Extract date range from measurement datetimes
        start_dt = pd.to_datetime(df["datetime"].min())
        end_dt = pd.to_datetime(df["datetime"].max())

        if pd.isna(start_dt) or pd.isna(end_dt):
            print("Could not determine start/end dates from measurements; skipping weather.")
            return None, None, None

        start_date = start_dt.date().strftime("%Y-%m-%d")
        end_date = end_dt.date().strftime("%Y-%m-%d")

        # Fetch hourly weather data from API
        print(f"Fetching weather for {start_date} -> {end_date} at ICAO {icao}...")
        try:
            hourly_df, daily_df, airport_name = weather_data.get_hourly_weather(
                start_date, end_date, icao_code=icao
            )
        except TypeError:
            hourly_df, daily_df, airport_name = weather_data.get_hourly_weather(
                start_date, end_date, icao
            )

        if hourly_df is None or hourly_df.empty:
            print("Hourly weather fetch returned no data; skipping weather plot.")
            return None, None, airport_name

        matched = weather_data.match_weather_to_resistance(df, hourly_df)
        if matched is None or matched.empty:
            print("No matched weather rows for resistance datetimes; skipping weather plot.")
            return None, None, airport_name

        # Build plotting frame using the full hourly series so rainfall matches preview styling
        tz_name = weather_data.TIMEZONE
        start_local = pd.to_datetime(start_dt)
        end_local = pd.to_datetime(end_dt)
        if start_local.tzinfo is None or start_local.tzinfo.utcoffset(start_local) is None:
            start_local = start_local.tz_localize(tz_name)
        else:
            start_local = start_local.tz_convert(tz_name)
        if end_local.tzinfo is None or end_local.tzinfo.utcoffset(end_local) is None:
            end_local = end_local.tz_localize(tz_name)
        else:
            end_local = end_local.tz_convert(tz_name)

        hourly_window = hourly_df[(hourly_df["date"] >= start_local.floor("h")) & (hourly_df["date"] <= end_local.ceil("h"))].copy()
        if hourly_window.empty:
            hourly_window = hourly_df.copy()

        plot_df = hourly_window.rename(columns={"date": "matched_hour"})
        plot_df["res_datetime"] = plot_df["matched_hour"]

        created = weather_data.create_weather_plot_png(
            plot_df,
            daily_df,
            str(weather_png),
            png_width=PNG_WIDTH,
            png_height=WEATHER_HEIGHT,
            rain_axis_max=DEFAULT_RAIN_AXIS_MAX,
            station_label=airport_name,
        )

        if created:
            print(f"[OK] Weather plot created: {weather_png}")
            return weather_png, matched, airport_name

        print("[ERROR] Weather plot creation failed.")
        return None, matched, airport_name

    except Exception as exc:
        print(f"[ERROR] Weather integration failed: {exc}")
        return None, None, None


def _enrich_with_weather(df: pd.DataFrame, matched: pd.DataFrame | None) -> pd.DataFrame:
    """Merge matched weather metrics with resistance data frame."""
    if matched is None or matched.empty:
        return df

    enriched = df.copy()
    enriched["datetime"] = pd.to_datetime(enriched["datetime"], errors="coerce")
    if enriched["datetime"].dt.tz is None:
        enriched["_datetime_local"] = enriched["datetime"].dt.tz_localize(
            weather_data.TIMEZONE,
            nonexistent="shift_forward",
            ambiguous="NaT",
        )
    else:
        enriched["_datetime_local"] = enriched["datetime"].dt.tz_convert(weather_data.TIMEZONE)

    weather = matched.copy()
    weather["res_datetime"] = pd.to_datetime(weather["res_datetime"], errors="coerce")

    enriched = enriched.merge(
        weather,
        left_on="_datetime_local",
        right_on="res_datetime",
        how="left"
    )

    enriched.drop(columns=["_datetime_local", "res_datetime"], inplace=True, errors="ignore")

    enriched["temperature_f"] = pd.to_numeric(enriched.get("temperature_2m"), errors="coerce")
    enriched["relative_humidity_percent"] = pd.to_numeric(
        enriched.get("relative_humidity_2m"), errors="coerce"
    )
    enriched["precipitation_in"] = pd.to_numeric(enriched.get("precipitation"), errors="coerce")

    enriched.drop(columns=["temperature_2m", "relative_humidity_2m", "precipitation", "matched_hour"], inplace=True, errors="ignore")
    return enriched


def _compute_normalized_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute resistance normalized to 40°C using provided temperature column."""
    if "temperature_f" not in df.columns:
        df["resistance_normalized_40c"] = pd.NA
        return df

    def _norm(row):
        temp_f = row.get("temperature_f")
        resistance = row.get("resistance")
        if pd.isna(temp_f) or pd.isna(resistance):
            return pd.NA
        temp_c = (temp_f - 32) * (5.0 / 9.0)
        kt = 0.5 * math.exp((40 - temp_c) / 10)
        try:
            return float(resistance) * kt
        except Exception:
            return pd.NA

    df["resistance_normalized_40c"] = df.apply(_norm, axis=1)
    return df


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
        print(f"[WARN] Pillow cannot open logo: {e}")
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
        print(f"[WARN] ReportLab failed to create Image: {e}")
        return None

def build_pdf_report(
    df: pd.DataFrame,
    *,
    resistance_plot_ok: bool,
    normalized_plot_ok: bool,
    weather_plot_ok: bool,
    airport_name: str | None = None,
):
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

    datetimes = pd.to_datetime(df.get("datetime"), errors="coerce").dropna().sort_values()
    test_start = datetimes.iloc[0] if not datetimes.empty else None
    test_end = datetimes.iloc[-1] if not datetimes.empty else None
    duration_str = _format_duration(test_end - test_start if test_start is not None and test_end is not None else None)
    window_str = (
        f"{test_start.strftime('%Y-%m-%d %H:%M')} to {test_end.strftime('%Y-%m-%d %H:%M')}"
        if test_start is not None and test_end is not None else "N/A"
    )
    sampling_freq = _describe_sampling_frequency(df.get("datetime"))

    raw_series = pd.to_numeric(df.get("resistance"), errors="coerce")
    raw_min = _format_resistance_value(raw_series.min())
    raw_avg = _format_resistance_value(raw_series.dropna().mean())
    raw_max = _format_resistance_value(raw_series.max())

    if "resistance_normalized_40c" in df.columns:
        norm_series = pd.to_numeric(df.get("resistance_normalized_40c"), errors="coerce")
        if norm_series.dropna().empty:
            norm_summary = "N/A"
        else:
            norm_min = _format_resistance_value(norm_series.min())
            norm_avg = _format_resistance_value(norm_series.dropna().mean())
            norm_max = _format_resistance_value(norm_series.max())
            norm_summary = f"Min {norm_min} | Avg {norm_avg} | Max {norm_max}"
    else:
        norm_summary = "N/A"

    temp_series = pd.to_numeric(df.get("temperature_f"), errors="coerce")
    if temp_series.dropna().empty:
        temp_summary = "N/A"
    else:
        temp_min = temp_series.min()
        temp_max = temp_series.max()
        temp_summary = f"{temp_min:.1f} °F to {temp_max:.1f} °F"

    rain_series = pd.to_numeric(df.get("precipitation_in"), errors="coerce")
    if rain_series.dropna().empty:
        rain_total = "N/A"
    else:
        rain_total = f"{rain_series.sum():.2f} in"

    circuit_config = _infer_circuit_configuration()

    summary_rows = [
        ["Metric", "Value"],
        ["Test Voltage", "2500 V"],
        ["Circuit Configuration", circuit_config],
        ["Measurement Window", window_str],
        ["Test Duration", duration_str],
        ["Sampling Frequency", sampling_freq],
        ["Resistance (raw)", f"Min {raw_min} | Avg {raw_avg} | Max {raw_max}"],
        ["Resistance (normalized)", norm_summary],
        ["Temperature Range", temp_summary],
        ["Total Rainfall", rain_total],
    ]

    story.append(Paragraph("Test Summary", ParagraphStyle(
        "SummaryHeading",
        parent=styles["Heading2"],
        spaceAfter=12,
    )))

    summary_table = Table(summary_rows, colWidths=[2.8 * inch, 4.7 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ATLANTIC_RED)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F3F3F3")]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 24))
    story.append(PageBreak())

    # ========== RESISTANCE PLOT ==========
    if resistance_plot_ok and OUTPUT_PLOT_PNG.exists():
        try:
            resistance_img = RLImage(
                str(OUTPUT_PLOT_PNG),
                width=PLOT_IMAGE_WIDTH_IN * inch,
                height=RESISTANCE_PLOT_HEIGHT_IN * inch
            )
            story.append(resistance_img)
        except Exception as e:
            print(f"[WARN] Failed to embed resistance plot: {e}")
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

    # ========== NORMALIZED RESISTANCE PLOT ==========
    if normalized_plot_ok and OUTPUT_NORMALIZED_PLOT_PNG.exists():
        try:
            normalized_img = RLImage(
                str(OUTPUT_NORMALIZED_PLOT_PNG),
                width=PLOT_IMAGE_WIDTH_IN * inch,
                height=RESISTANCE_PLOT_HEIGHT_IN * inch
            )
            story.append(normalized_img)
        except Exception as exc:
            print(f"[WARN] Failed to embed normalized plot: {exc}")

    

    # ========== WEATHER PLOT ==========
    weather_png_path = OUTPUT_WEATHER_PLOT_PNG
    if weather_plot_ok and weather_png_path.exists():
        try:
            # Weather plot embedded with shared dimensions for consistent alignment
            weather_img = RLImage(
                str(weather_png_path), 
                width=PLOT_IMAGE_WIDTH_IN * inch,
                height=WEATHER_PLOT_HEIGHT_IN * inch
            )
            story.append(weather_img)
        except Exception as e:
            print(f"[WARN] Failed to embed weather plot: {e}")
    elif airport_name:
        story.append(Paragraph(
            f"Weather station: {airport_name}",
            styles["Normal"]
        ))

    # ========== DATA TABLE ==========
    # Build table data with formatted resistance values
    story.append(PageBreak())
    story.append(Paragraph("Measurement Data", ParagraphStyle(
        "SummaryHeading",
        parent=styles["Heading2"],
        spaceAfter=12
    )))
    table_data = [[
        "Date",
        "Resistance",
        "Normalized (40°C)",
        "Temp (°F)",
        "Humidity (%)",
        "Rainfall (in)"
    ]]
    
    for _, row in df.iterrows():
        # Format datetime
        dt = row["datetime"]
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt) else ""
        
        # Format resistance with appropriate units
        res_str = _format_resistance_value(row.get("resistance"))
        norm_str = _format_resistance_value(row.get("resistance_normalized_40c"))

        temp_val = row.get("temperature_f")
        hum_val = row.get("relative_humidity_percent")
        rain_val = row.get("precipitation_in")

        temp_str = f"{float(temp_val):.1f}" if pd.notna(temp_val) else "N/A"
        hum_str = f"{float(hum_val):.0f}" if pd.notna(hum_val) else "N/A"
        rain_str = f"{float(rain_val):.2f}" if pd.notna(rain_val) else "N/A"

        table_data.append([dt_str, res_str, norm_str, temp_str, hum_str, rain_str])

    # Create and style table with Atlantic Electric branding
    table = Table(
        table_data, 
        colWidths=[2.0 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch], 
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
        print(f"[OK] PDF report created: {OUTPUT_PDF}")
    except Exception as e:
        print(f"[ERROR] PDF creation failed: {e}")




def _run_report_pipeline() -> dict[str, object]:
    """Execute complete report workflow and return artifact summary."""
    print("=" * 60)
    print("Circuit Integrity Report Generator")
    print("=" * 60)

    print("\n[1/5] Collecting measurements from CSV files...")
    df = collect_measurements()

    print("\n[2/5] Generating resistance plot...")
    resistance_plot_ok = create_static_plot_png(df)

    print("\n[3/5] Integrating weather data...")
    weather_png, matched_weather, airport_name = integrate_weather(df, OUTPUT_DIR, icao="KCUB")
    weather_plot_ok = weather_png is not None and Path(weather_png).exists()

    enriched_df = _enrich_with_weather(df, matched_weather)
    enriched_df = _compute_normalized_resistance(enriched_df)

    print("\n[4/5] Generating normalized resistance plot...")
    normalized_plot_ok = create_normalized_plot_png(enriched_df)

    enriched_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved enriched summary to: {OUTPUT_CSV}")

    print("\n[5/5] Building PDF report...")
    build_pdf_report(
        enriched_df,
        resistance_plot_ok=resistance_plot_ok,
        normalized_plot_ok=normalized_plot_ok,
        weather_plot_ok=weather_plot_ok,
        airport_name=airport_name,
    )

    print("\n" + "=" * 60)
    print("Report generation complete!")
    print("=" * 60)

    return {
        "data": enriched_df,
        "resistance_plot": OUTPUT_PLOT_PNG if resistance_plot_ok else None,
        "normalized_plot": OUTPUT_NORMALIZED_PLOT_PNG if normalized_plot_ok else None,
        "weather_plot": OUTPUT_WEATHER_PLOT_PNG if weather_plot_ok else None,
        "pdf": OUTPUT_PDF,
        "summary_csv": OUTPUT_CSV,
        "airport_name": airport_name,
    }


def generate_report(output_dir: Path | str | None = None) -> dict[str, object]:
    """Entry point for other scripts to run the report pipeline with a target directory."""
    configure_paths(output_dir=output_dir)
    return _run_report_pipeline()


def main():
    """CLI entry point using configured defaults."""
    generate_recent_weather_preview()
    configure_paths()
    _run_report_pipeline()


if __name__ == "__main__":
    main()