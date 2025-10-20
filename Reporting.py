# reporting_with_plotly.py
from pathlib import Path
import pandas as pd
import re
import sys
import plotly.express as px

INPUT_DIR = r"C:\Users\user\Downloads\TestCSV\10_14_Test_750V\10_14_Test-2000"
OUTPUT_DIR = INPUT_DIR + r"\Summary"
OUTPUT_FILE = OUTPUT_DIR + r"\measurements_summary.csv"
OUTPUT_PLOT_HTML = OUTPUT_DIR + r"\resistance_plot.html"
OUTPUT_PLOT_PNG = OUTPUT_DIR + r"\resistance_plot.png"

FNAME_TS_RE = re.compile(r'(\d{8})_(\d{6})')

def extract_datetime_from_filename(fname: str):
    m = FNAME_TS_RE.search(fname)
    if not m:
        return None
    date_part, time_part = m.group(1), m.group(2)
    try:
        return pd.to_datetime(f"{date_part}_{time_part}", format="%Y%m%d_%H%M%S")
    except Exception:
        return None

def collect_measurements():
    """Read CSVs and return a DataFrame with datetime and resistance."""
    p = Path(INPUT_DIR)
    if not p.exists():
        print("INPUT_DIR does not exist:", INPUT_DIR)
        sys.exit(1)

    rows = []
    for f in sorted(p.glob("*.csv")):
        dt = extract_datetime_from_filename(f.name)
        try:
            df = pd.read_csv(f, skiprows=1)
            raw_val = df["Final Measurement (A/Ω)"].iloc[0]
            resistance = pd.to_numeric(raw_val, errors="coerce")
        except Exception as e:
            print(f"{f.name}: ERROR -> {e}")
            resistance = pd.NA

        rows.append({"datetime": dt, "resistance": resistance, "source_file": f.name})
        print(f"{f.name} | datetime: {dt} | resistance: {resistance}")

    result_df = pd.DataFrame(rows)
    result_df["datetime"] = pd.to_datetime(result_df["datetime"], errors="coerce")
    result_df = result_df.sort_values("datetime").reset_index(drop=True)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved summary to: {OUTPUT_FILE}\n")
    return result_df

def plot_results(df: pd.DataFrame):
    """Create a polished resistance-over-time plot using Plotly."""
    fig = px.line(
        df,
        x="datetime",
        y="resistance",
        title="Resistance Measurements Over Time",
        markers=True,
        hover_data={"datetime": "|%Y-%m-%d %H:%M:%S", "resistance": ":.2f"},
    )
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=22, family="Arial, bold"),
        xaxis_title="Datetime",
        yaxis_title="Resistance (A/Ω)",
        font=dict(size=14),
        hoverlabel=dict(bgcolor="white", font_size=13),
        margin=dict(l=50, r=30, t=80, b=60),
    )
    # Save as interactive HTML and high-res PNG (good for reports)
    fig.write_html(OUTPUT_PLOT_HTML)
    try:
        fig.write_image(OUTPUT_PLOT_PNG, scale=3)
    except Exception:
        print("⚠️  Plotly image export requires 'kaleido' package. Install it via: pip install kaleido")
    fig.show()

def main():
    df = collect_measurements()
    print(df[["datetime", "resistance", "source_file"]].to_string(index=False))
    plot_results(df)

if __name__ == "__main__":
    main()
