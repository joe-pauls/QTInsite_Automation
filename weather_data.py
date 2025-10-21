# weather_data.py
"""
Weather helper module for Reporting.py

Provides:
- get_hourly_weather(start_date, end_date, icao_code)
- match_weather_to_resistance(res_df, hourly_df)
- create_weather_plot_png(matched_weather_df, daily_df, out_png)

Depends on the same packages used previously in your uploaded script:
  - pandas, numpy, plotly, openmeteo_requests, requests_cache, airportsdata, retry_requests
(If you already have these installed for Reporting.py, no extra installs needed.)
"""
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go

# --- If you have the openmeteo / airport utilities in your uploaded script, import them ---
# The uploaded weather script used: openmeteo_requests, requests_cache, airportsdata, retry_requests
# If those modules are available in your environment, uncomment imports below and use the fetch function.
try:
    import openmeteo_requests
    import requests_cache
    import airportsdata
    from retry_requests import retry
    _HAS_OPENMETEO = True
except Exception:
    _HAS_OPENMETEO = False
    # If openmeteo modules aren't available, user must provide hourly_df/daily_df manually or install libs.

# --- Public functions ---

def get_hourly_weather(start_date, end_date, icao_code="KCAE"):
    """
    Fetch hourly weather (temperature_2m, relative_humidity_2m, precipitation) and daily precipitation sums.
    Returns (hourly_df, daily_df, airport_name).
    hourly_df: column 'date' is timezone-aware in America/New_York
    daily_df: column 'date' is timezone-aware in America/New_York (midnight)
    """
    if not _HAS_OPENMETEO:
        raise RuntimeError("openmeteo_requests and supporting packages are not available in the environment.")

    # find airport coords
    def icao_to_coordinates(icao_code):
        airports = airportsdata.load()
        if icao_code in airports:
            return airports[icao_code]['lat'], airports[icao_code]['lon']
        return None, None

    latitude, longitude = icao_to_coordinates(icao_code)
    if latitude is None:
        raise ValueError(f"Invalid ICAO code: {icao_code}")

    airport_name = airportsdata.load()[icao_code]["name"] if icao_code in airportsdata.load() else icao_code

    # Use cached session + retries like your uploaded script
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": (pd.to_datetime(start_date) - pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
        "end_date": pd.to_datetime(end_date).strftime('%Y-%m-%d'),
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation"],
        "daily": ["precipitation_sum"],
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "UTC"   # fetch in UTC then convert to America/New_York below
    }

    # Request (matches previous file's approach)
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()
    daily = response.Daily()

    # Build hourly dataframe (the uploaded script used Time(), TimeEnd(), Interval())
    hourly_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    hourly_df = pd.DataFrame({
        "date": hourly_index,
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "precipitation": hourly.Variables(2).ValuesAsNumpy()
    })

    daily_index = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )
    daily_df = pd.DataFrame({
        "date": daily_index,
        "precipitation_sum": daily.Variables(0).ValuesAsNumpy()
    })

    # rolling precipitation sum (example: 3-day rolling like in uploaded script)
    daily_df["rolling_precipitation_sum"] = daily_df["precipitation_sum"].rolling(window=3, min_periods=1).sum()

    # Convert to America/New_York for display/merging
    hourly_df["date"] = hourly_df["date"].dt.tz_convert("America/New_York")
    daily_df["date"] = daily_df["date"].dt.tz_convert("America/New_York")

    return hourly_df, daily_df, airport_name


def match_weather_to_resistance(res_df, hourly_df):
    """
    Given res_df with a 'datetime' column and hourly_df with 'date' (tz-aware),
    round each resistance datetime to nearest hour and return a DataFrame with:
      - original datetime (res_datetime)
      - matched_hour (rounded)
      - temperature_2m, relative_humidity_2m, precipitation (for that hour)
    Only hours that exist in hourly_df are returned; missing hours show NaN.
    """
    if "datetime" not in res_df.columns:
        raise ValueError("res_df must contain a 'datetime' column")

    # Ensure timezone alignment: convert res datetimes to America/New_York (same as hourly_df)
    res = res_df.copy()
    res["datetime"] = pd.to_datetime(res["datetime"])
    # if naive, treat as local America/New_York
    if res["datetime"].dt.tz is None:
        res["datetime"] = res["datetime"].dt.tz_localize("America/New_York")
    else:
        res["datetime"] = res["datetime"].dt.tz_convert("America/New_York")

    # Round to nearest hour
    res["rounded_hour"] = res["datetime"].dt.round("H")

    # Prepare hourly_df index for fast lookup
    hourly = hourly_df.copy()
    # ensure hourly 'date' tz-aware America/New_York
    hourly["date"] = pd.to_datetime(hourly["date"])
    if hourly["date"].dt.tz is None:
        hourly["date"] = hourly["date"].dt.tz_localize("America/New_York")
    else:
        hourly["date"] = hourly["date"].dt.tz_convert("America/New_York")
    hourly = hourly.set_index("date")

    # For each rounded_hour, lookup in hourly
    def lookup_hour(dt):
        try:
            row = hourly.loc[dt]
            # if multiple rows (unlikely), take first
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return pd.Series({
                "temperature_2m": row["temperature_2m"],
                "relative_humidity_2m": row["relative_humidity_2m"],
                "precipitation": row["precipitation"]
            })
        except KeyError:
            return pd.Series({"temperature_2m": np.nan, "relative_humidity_2m": np.nan, "precipitation": np.nan})

    matched = res[["datetime", "rounded_hour"]].copy()
    matched = pd.concat([matched, matched["rounded_hour"].apply(lookup_hour)], axis=1)
    # Keep rounded_hour for plotting/labeling
    matched = matched.rename(columns={"datetime": "res_datetime", "rounded_hour": "matched_hour"})

    return matched


def create_weather_plot_png(matched_weather_df, daily_df, out_png, title="Weather (temp/humidity/rainfall)"):
    """
    Improved weather plot that aligns x-axis ticks with the resistance plot and uses
    matched_weather_df rows (which should include 'matched_hour' for each resistance sample).

    - Temperature: red line, left axis
    - Humidity: blue line, right axis
    - Rainfall: thin bars, right axis
    - X-axis ticks forced to HH:MM (same logic as resistance plot)
    - Vertical day separators and MM/DD labels under first point per day
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from datetime import timedelta
    from pathlib import Path

    # defensive checks
    if matched_weather_df is None or matched_weather_df.empty:
        print("No matched weather data to plot.")
        return False

    # copy and ensure datetimes
    df = matched_weather_df.copy()
    df["matched_hour"] = pd.to_datetime(df["matched_hour"])
    df = df.sort_values("matched_hour").reset_index(drop=True)

    # Extract series (coerce to numeric)
    x = df["matched_hour"].tolist()
    temp = pd.to_numeric(df.get("temperature_2m", pd.Series([np.nan]*len(df))), errors="coerce").tolist()
    hum = pd.to_numeric(df.get("relative_humidity_2m", pd.Series([np.nan]*len(df))), errors="coerce").tolist()
    # For rainfall at the matched times we prefer hourly precipitation if present in df,
    # otherwise use daily_df (display as daily bars) — try hourly first:
    if "precipitation" in df.columns:
        rain_at_hours = pd.to_numeric(df["precipitation"], errors="coerce")
        use_hourly_rain = True
    else:
        # fall back to daily totals (bars at midnight)
        use_hourly_rain = False
        if daily_df is None or daily_df.empty:
            daily = pd.DataFrame(columns=["date", "rain"])
        else:
            daily = daily_df.copy()
            daily["date"] = pd.to_datetime(daily["date"])
            if "rolling_precipitation_sum" in daily.columns:
                daily["rain"] = pd.to_numeric(daily["rolling_precipitation_sum"], errors="coerce")
            elif "precipitation_sum" in daily.columns:
                daily["rain"] = pd.to_numeric(daily["precipitation_sum"], errors="coerce")
            else:
                daily["rain"] = pd.to_numeric(daily.get("precipitation", pd.Series([np.nan]*len(daily))), errors="coerce")

    # Colors & style
    temp_color = "#A6192E"    # Atlantic red
    hum_color = "#2A66D9"     # blue
    rain_color = "#4C4EA3"    # darker blue/purple
    light_gray = "#E6E6E6"
    dark_gray = "#231F20"

    # Determine tick frequency using same rules as resistance plot
    start = pd.to_datetime(df["matched_hour"].min())
    end = pd.to_datetime(df["matched_hour"].max())
    span_seconds = (end - start).total_seconds()
    if span_seconds <= 2 * 3600:
        freq = "10min"
    elif span_seconds <= 6 * 3600:
        freq = "15min"
    elif span_seconds <= 24 * 3600:
        freq = "30min"
    else:
        freq = "1H"

    tick_start = start
    if freq.endswith("H"):
        tick_start = tick_start.floor("H")
    else:
        minutes = int(freq.replace("min", ""))
        tick_start = tick_start - pd.Timedelta(minutes=(tick_start.minute % minutes),
                                               seconds=tick_start.second,
                                               microseconds=tick_start.microsecond)

    tick_vals = pd.date_range(start=tick_start, end=end + pd.Timedelta(minutes=1), freq=freq).to_pydatetime().tolist()
    tick_text = [t.strftime("%H:%M") for t in tick_vals]

    # Day separators and first-per-day MM/DD labels
    days = pd.to_datetime(df["matched_hour"]).dt.floor("D").unique()
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

    # first measurement per day for label placement
    first_per_day = df.groupby(df["matched_hour"].dt.floor("D"))["matched_hour"].min().tolist()
    annotations = []
    for f in first_per_day:
        annotations.append(dict(
            x=f, y=-0.15, xref="x", yref="paper",
            text=pd.to_datetime(f).strftime("%m/%d"),
            showarrow=False, font=dict(size=10, color=dark_gray)
        ))

    # Build figure
    fig = go.Figure()

    # Temperature (left axis)
    fig.add_trace(go.Scatter(
        x=x, y=temp,
        name="Temperature (°F)",
        mode="lines+markers",
        line=dict(color=temp_color, width=2),
        marker=dict(size=6),
        yaxis="y1",
        hovertemplate="%{x|%b %d %H:%M}<br>Temp: %{y:.1f}°F<extra></extra>"
    ))

    # Humidity (right axis)
    fig.add_trace(go.Scatter(
        x=x, y=hum,
        name="Humidity (%)",
        mode="lines+markers",
        line=dict(color=hum_color, width=2),
        marker=dict(size=6),
        yaxis="y2",
        hovertemplate="%{x|%b %d %H:%M}<br>Humidity: %{y:.0f}%<extra></extra>"
    ))

    # Rainfall: prefer hourly precipitation (thin bars), else daily bars
    if use_hourly_rain:
        # narrow bar width: use a fraction of tick spacing (in ms)
        # compute approximate bar width as 60% of interval between ticks (or 10 minutes default)
        if len(tick_vals) >= 2:
            interval_ms = (pd.to_datetime(tick_vals[1]) - pd.to_datetime(tick_vals[0])).total_seconds() * 1000
            bar_width = max(1, interval_ms * 0.25)  # 25% of tick spacing
        else:
            bar_width = 15 * 60 * 1000  # 15 minutes in ms
        fig.add_trace(go.Bar(
            x=x,
            y=rain_at_hours,
            name="Rainfall (in)",
            marker=dict(color=rain_color, opacity=0.6),
            yaxis="y2",
            width=bar_width,
            hovertemplate="%{x|%b %d %H:%M}<br>Rain: %{y:.2f} in<extra></extra>"
        ))
    else:
        if not daily.empty:
            # thin daily bars
            day_width = 24 * 3600 * 1000 * 0.6
            fig.add_trace(go.Bar(
                x=daily["date"],
                y=daily["rain"],
                name="Rainfall (in)",
                marker=dict(color=rain_color, opacity=0.6),
                yaxis="y2",
                width=day_width,
                hovertemplate="%{x|%b %d}<br>Rain: %{y:.2f} in<extra></extra>"
            ))

        # --- Layout (modern syntax, matches resistance plot) ---
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.01, xanchor="left",
                   font=dict(size=16, color=dark_gray)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        margin=dict(l=60, r=80, t=70, b=100),
        width=PNG_WIDTH if "PNG_WIDTH" in globals() else 1200,
        height=360,
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            title="Datetime",
            type="date",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=0,
            showgrid=True,
            gridcolor=light_gray,
            zeroline=False,
            tickfont=dict(family="Arial", size=10, color=dark_gray),
            title_font=dict(family="Arial", size=12, color=dark_gray)
        ),
        yaxis=dict(
            title=dict(text="Temperature (°F)",
                       font=dict(family="Arial", size=12, color=temp_color)),
            tickfont=dict(family="Arial", size=10, color=temp_color),
            showgrid=True,
            gridcolor=light_gray,
            zeroline=False,
        ),
        yaxis2=dict(
            title=dict(text="Humidity (%) / Rain (in)",
                       font=dict(family="Arial", size=12, color=hum_color)),
            tickfont=dict(family="Arial", size=10, color=hum_color),
            overlaying="y",
            side="right",
            rangemode="tozero"
        )
    )


    # pad x range slightly
    if len(x) >= 1:
        pad = timedelta(minutes=1)
        fig.update_xaxes(range=[pd.to_datetime(x[0]) - pad, pd.to_datetime(x[-1]) + pad])

    # write image
    try:
        out_path = Path(out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(out_path), width=PNG_WIDTH if "PNG_WIDTH" in globals() else 1200,
                        height=360, scale=1)
        print(f"Wrote weather PNG to: {out_path}")
        return True
    except Exception as e:
        print("Failed to write weather PNG:", e)
        return False

