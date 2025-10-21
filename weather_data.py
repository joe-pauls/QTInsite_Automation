"""
Weather Data Integration Module

Provides weather data fetching, matching, and visualization for Circuit Integrity Reports.

This module integrates with Open-Meteo Archive API to fetch historical weather data
from nearby airport stations and correlates it with resistance measurements.

Features:
- Fetch hourly weather data (temperature, humidity, precipitation)
- Match weather data to resistance measurement timestamps
- Generate weather visualization plots matching resistance plot styling

Dependencies:
- openmeteo_requests: Open-Meteo API client
- requests_cache: HTTP request caching
- airportsdata: Airport location database
- retry_requests: Request retry logic
- pandas, numpy, plotly: Data processing and visualization

Author: Joseph Pauls
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go

# ========== DEPENDENCY CHECKS ==========
try:
    import openmeteo_requests
    import requests_cache
    import airportsdata
    from retry_requests import retry
    _HAS_OPENMETEO = True
except ImportError:
    _HAS_OPENMETEO = False
    # If openmeteo modules aren't available, user must install:
    # pip install openmeteo-requests requests-cache airportsdata retry-requests

# ========== CONFIGURATION ==========
# Atlantic Electric brand colors (matching Reporting.py)
ATLANTIC_RED = "#A6192E"
DARK_GRAY = "#231F20"
LIGHT_GRAY = "#E6E6E6"

# Weather plot colors
TEMP_COLOR = ATLANTIC_RED  # Temperature uses brand red
HUM_COLOR = "#2A66D9"      # Humidity uses blue
RAIN_COLOR = "#4C4EA3"     # Rainfall uses darker blue/purple

# Default timezone for weather data
TIMEZONE = "America/New_York"

# Weather API configuration
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE_DIR = ".cache"
API_RETRY_COUNT = 5
API_RETRY_BACKOFF = 0.2

# Font sizes (matching resistance plot)
TITLE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 16
DATE_LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16
# ====================================
# ========== HELPER FUNCTIONS ==========

def _icao_to_coordinates(icao_code: str) -> tuple[float | None, float | None]:
    """
    Convert ICAO airport code to latitude/longitude coordinates.
    
    Args:
        icao_code: Four-letter ICAO airport code (e.g., "KCAE")
        
    Returns:
        Tuple of (latitude, longitude) or (None, None) if code invalid
    """
    airports = airportsdata.load()
    if icao_code in airports:
        return airports[icao_code]['lat'], airports[icao_code]['lon']
    return None, None


def _get_airport_name(icao_code: str) -> str:
    """
    Get airport name from ICAO code.
    
    Args:
        icao_code: Four-letter ICAO airport code
        
    Returns:
        Airport name, or the ICAO code if not found
    """
    airports = airportsdata.load()
    if icao_code in airports:
        return airports[icao_code]["name"]
    return icao_code


def _calculate_tick_frequency(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """
    Determine appropriate tick frequency based on time span.
    
    Args:
        start: Start timestamp
        end: End timestamp
        
    Returns:
        Frequency string ('10min', '15min', '30min', or '1H')
    """
    span_seconds = (end - start).total_seconds()
    
    if span_seconds <= 2 * 3600:  # <= 2 hours
        return "10min"
    elif span_seconds <= 6 * 3600:  # <= 6 hours
        return "15min"
    elif span_seconds <= 24 * 3600:  # <= 24 hours
        return "30min"
    else:
        return "1H"


def _align_tick_start(start: pd.Timestamp, freq: str) -> pd.Timestamp:
    """
    Align tick start time to frequency boundary.
    
    Args:
        start: Start timestamp
        freq: Frequency string from _calculate_tick_frequency
        
    Returns:
        Aligned timestamp
    """
    if freq.endswith("H"):
        return start.floor("H")
    else:
        minutes = int(freq.replace("min", ""))
        return start - pd.Timedelta(
            minutes=(start.minute % minutes),
            seconds=start.second,
            microseconds=start.microsecond
        )


def _create_day_separators(days: pd.DatetimeIndex, start: pd.Timestamp, 
                          end: pd.Timestamp) -> list:
    """
    Create vertical line shapes for day boundaries.
    
    Args:
        days: Unique day timestamps (floored to midnight)
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


def _create_date_annotations(datetimes: pd.Series, y_position: float = -0.16) -> list:
    """
    Create MM/DD date label annotations for first measurement of each day.
    
    Args:
        datetimes: Series of datetime values
        y_position: Vertical position relative to plot (paper coordinates)
        
    Returns:
        List of plotly annotation dictionaries
    """
    first_per_day = datetimes.groupby(datetimes.dt.floor("D")).min().tolist()
    
    annotations = []
    for first_time in first_per_day:
        annotations.append(dict(
            x=first_time, y=y_position, xref="x", yref="paper",
            text=pd.to_datetime(first_time).strftime("%m/%d"), showarrow=False,
            font=dict(size=DATE_LABEL_FONT_SIZE, family="Arial", color=DARK_GRAY)
        ))
    
    return annotations


# ========== PUBLIC API FUNCTIONS ==========

def get_hourly_weather(start_date: str, end_date: str, icao_code: str = "KCAE"):
    """
    Fetch hourly and daily weather data from Open-Meteo Archive API.
    
    Retrieves temperature, humidity, and precipitation data for the specified date range
    from the nearest airport weather station. Data is returned in America/New_York timezone.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        icao_code: Four-letter ICAO airport code (default: "KCAE" for Columbia, SC)
        
    Returns:
        Tuple of (hourly_df, daily_df, airport_name) where:
        - hourly_df: DataFrame with columns ['date', 'temperature_2m', 'relative_humidity_2m', 'precipitation']
        - daily_df: DataFrame with columns ['date', 'precipitation_sum', 'rolling_precipitation_sum']
        - airport_name: Name of the weather station
        
    Raises:
        RuntimeError: If required dependencies (openmeteo_requests, etc.) are not installed
        ValueError: If ICAO code is invalid
        
    Notes:
        - Fetches 7 days before start_date for rolling precipitation calculations
        - Uses cached requests to minimize API calls
        - Includes automatic retry logic for failed requests
        - All timestamps are timezone-aware (America/New_York)
    """
    # Validate dependencies
    if not _HAS_OPENMETEO:
        raise RuntimeError(
            "Required weather API dependencies not installed. "
            "Install with: pip install openmeteo-requests requests-cache airportsdata retry-requests"
        )

    # Get airport coordinates
    latitude, longitude = _icao_to_coordinates(icao_code)
    if latitude is None:
        raise ValueError(f"Invalid ICAO code: {icao_code}")

    airport_name = _get_airport_name(icao_code)

    # Set up cached HTTP session with retry logic
    cache_session = requests_cache.CachedSession(CACHE_DIR, expire_after=-1)
    retry_session = retry(cache_session, retries=API_RETRY_COUNT, backoff_factor=API_RETRY_BACKOFF)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Configure API request parameters
    # Fetch 7 days before start for rolling precipitation calculations
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": (pd.to_datetime(start_date) - pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
        "end_date": pd.to_datetime(end_date).strftime('%Y-%m-%d'),
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation"],
        "daily": ["precipitation_sum"],
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "UTC"  # Fetch in UTC, convert to local timezone below
    }

    # Make API request
    responses = openmeteo.weather_api(WEATHER_API_URL, params=params)
    response = responses[0]
    hourly = response.Hourly()
    daily = response.Daily()

    # Build hourly DataFrame
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

    # Build daily DataFrame
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

    # Calculate 3-day rolling precipitation sum
    daily_df["rolling_precipitation_sum"] = (
        daily_df["precipitation_sum"]
        .rolling(window=3, min_periods=1)
        .sum()
    )

    # Convert timestamps to local timezone
    hourly_df["date"] = hourly_df["date"].dt.tz_convert(TIMEZONE)
    daily_df["date"] = daily_df["date"].dt.tz_convert(TIMEZONE)

    return hourly_df, daily_df, airport_name


def match_weather_to_resistance(res_df: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match weather data to resistance measurement timestamps.
    
    For each resistance measurement datetime, finds the nearest hour in the weather data
    and returns the temperature, humidity, and precipitation for that hour.
    
    Args:
        res_df: DataFrame with 'datetime' column (resistance measurements)
        hourly_df: DataFrame with 'date' column (hourly weather data)
        
    Returns:
        DataFrame with columns:
        - res_datetime: Original resistance measurement datetime
        - matched_hour: Rounded hour matched to weather data
        - temperature_2m: Temperature in °F
        - relative_humidity_2m: Relative humidity percentage
        - precipitation: Precipitation in inches
        
    Raises:
        ValueError: If res_df doesn't contain 'datetime' column
        
    Notes:
        - Rounds each resistance datetime to nearest hour for matching
        - Handles timezone conversion automatically (assumes America/New_York)
        - Returns NaN for weather values if no matching hour found
        - Missing hours in weather data will show as NaN in output
    """
    # Validate input
    if "datetime" not in res_df.columns:
        raise ValueError("res_df must contain a 'datetime' column")

    # Prepare resistance data
    res = res_df.copy()
    res["datetime"] = pd.to_datetime(res["datetime"])
    
    # Ensure timezone alignment to America/New_York
    if res["datetime"].dt.tz is None:
        res["datetime"] = res["datetime"].dt.tz_localize(TIMEZONE)
    else:
        res["datetime"] = res["datetime"].dt.tz_convert(TIMEZONE)

    # Round each datetime to nearest hour for matching
    res["rounded_hour"] = res["datetime"].dt.round("H")

    # Prepare hourly weather data for lookup
    hourly = hourly_df.copy()
    hourly["date"] = pd.to_datetime(hourly["date"])
    
    # Ensure timezone alignment
    if hourly["date"].dt.tz is None:
        hourly["date"] = hourly["date"].dt.tz_localize(TIMEZONE)
    else:
        hourly["date"] = hourly["date"].dt.tz_convert(TIMEZONE)
    
    hourly = hourly.set_index("date")

    # Lookup function to find weather for each hour
    def lookup_hour(dt):
        """Look up weather data for a specific datetime."""
        try:
            row = hourly.loc[dt]
            # Handle multiple rows (unlikely but possible)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return pd.Series({
                "temperature_2m": row["temperature_2m"],
                "relative_humidity_2m": row["relative_humidity_2m"],
                "precipitation": row["precipitation"]
            })
        except KeyError:
            # Hour not found in weather data
            return pd.Series({
                "temperature_2m": np.nan,
                "relative_humidity_2m": np.nan,
                "precipitation": np.nan
            })

    # Match weather to each resistance measurement
    matched = res[["datetime", "rounded_hour"]].copy()
    weather_cols = matched["rounded_hour"].apply(lookup_hour)
    matched = pd.concat([matched, weather_cols], axis=1)
    
    # Rename columns for clarity
    matched = matched.rename(columns={
        "datetime": "res_datetime",
        "rounded_hour": "matched_hour"
    })

    return matched


def create_weather_plot_png(matched_weather_df: pd.DataFrame, daily_df: pd.DataFrame, 
                           out_png: str, title: str = "Weather (temp/humidity/rainfall)", 
                           png_width: int = 1400, png_height: int = 500) -> bool:
    """
    Generate weather plot PNG with Atlantic Electric branding.
    
    Creates a dual-axis plot showing temperature (°F) on left axis and humidity (%)/
    precipitation (in) on right axis. Includes day separators and MM/DD date labels,
    with styling matching the resistance plot.
    
    Args:
        matched_weather_df: DataFrame from match_weather_to_resistance() with columns:
                           ['res_datetime', 'matched_hour', 'temperature_2m', 
                            'relative_humidity_2m', 'precipitation']
        daily_df: DataFrame from get_hourly_weather() with daily precipitation data
        out_png: Output file path for PNG image
        title: Plot title (default: "Weather (temp/humidity/rainfall)")
        png_width: Image width in pixels (default: 1400)
        png_height: Image height in pixels (default: 500)
        
    Returns:
        True if plot created successfully, False otherwise
        
    Notes:
        - Uses Atlantic Electric brand colors
        - Font sizes match resistance plot (24pt title, 18pt axes, 16pt ticks)
        - Precipitation displayed as bars (hourly if available, otherwise daily)
        - Automatic tick frequency based on time span
        - MM/DD date labels positioned at y=-0.16
    """
    # Validate input
    if matched_weather_df is None or matched_weather_df.empty:
        print("⚠️  No matched weather data to plot.")
        return False

    # Prepare data
    df = matched_weather_df.copy()
    df["matched_hour"] = pd.to_datetime(df["matched_hour"])
    df = df.sort_values("matched_hour").reset_index(drop=True)
    
    x = df["matched_hour"].tolist()
    temp = pd.to_numeric(
        df.get("temperature_2m", pd.Series([np.nan] * len(df))), 
        errors="coerce"
    ).tolist()
    hum = pd.to_numeric(
        df.get("relative_humidity_2m", pd.Series([np.nan] * len(df))), 
        errors="coerce"
    ).tolist()
    
    # Determine rainfall data source (hourly vs daily)
    if "precipitation" in df.columns:
        rain_at_hours = pd.to_numeric(df["precipitation"], errors="coerce")
        use_hourly_rain = True
    else:
        use_hourly_rain = False
        # Prepare daily data as fallback
        if daily_df is None or daily_df.empty:
            daily = pd.DataFrame(columns=["date", "rain"])
        else:
            daily = daily_df.copy()
            daily["date"] = pd.to_datetime(daily["date"])
            # Use rolling precipitation if available, otherwise regular sum
            if "rolling_precipitation_sum" in daily.columns:
                daily["rain"] = pd.to_numeric(daily["rolling_precipitation_sum"], errors="coerce")
            elif "precipitation_sum" in daily.columns:
                daily["rain"] = pd.to_numeric(daily["precipitation_sum"], errors="coerce")
            else:
                daily["rain"] = pd.to_numeric(
                    daily.get("precipitation", pd.Series([np.nan] * len(daily))), 
                    errors="coerce"
                )

    # Calculate time range and tick parameters
    start = pd.to_datetime(df["matched_hour"].min())
    end = pd.to_datetime(df["matched_hour"].max())
    
    freq = _calculate_tick_frequency(start, end)
    tick_start = _align_tick_start(start, freq)
    
    tick_vals = pd.date_range(
        start=tick_start, 
        end=end + pd.Timedelta(minutes=1), 
        freq=freq
    ).to_pydatetime().tolist()
    tick_text = [t.strftime("%H:%M") for t in tick_vals]

    # Create visual elements
    days = pd.to_datetime(df["matched_hour"]).dt.floor("D").unique()
    shapes = _create_day_separators(days, start, end)
    annotations = _create_date_annotations(df["matched_hour"], y_position=-0.16)

    # ========== BUILD PLOT ==========
    fig = go.Figure()

    # Temperature trace (left y-axis)
    fig.add_trace(go.Scatter(
        x=x, y=temp, 
        name="Temperature (°F)", 
        mode="lines+markers",
        line=dict(color=TEMP_COLOR, width=2.5),
        marker=dict(size=7, color=TEMP_COLOR),
        yaxis="y1",
        hovertemplate="%{x|%b %d %H:%M}<br>Temp: %{y:.1f}°F<extra></extra>"
    ))

    # Humidity trace (right y-axis)
    fig.add_trace(go.Scatter(
        x=x, y=hum, 
        name="Humidity (%)", 
        mode="lines+markers",
        line=dict(color=HUM_COLOR, width=2.5),
        marker=dict(size=7, color=HUM_COLOR),
        yaxis="y2",
        hovertemplate="%{x|%b %d %H:%M}<br>Humidity: %{y:.0f}%<extra></extra>"
    ))

    # Rainfall trace (right y-axis, bars)
    if use_hourly_rain:
        # Calculate bar width based on tick frequency
        if len(tick_vals) >= 2:
            interval_ms = (
                pd.to_datetime(tick_vals[1]) - pd.to_datetime(tick_vals[0])
            ).total_seconds() * 1000
            bar_width = max(1, interval_ms * 0.25)
        else:
            bar_width = 15 * 60 * 1000  # 15 minutes default
            
        fig.add_trace(go.Bar(
            x=x, y=rain_at_hours, 
            name="Rainfall (in)",
            marker=dict(color=RAIN_COLOR, opacity=0.6),
            yaxis="y2", 
            width=bar_width,
            hovertemplate="%{x|%b %d %H:%M}<br>Rain: %{y:.2f} in<extra></extra>"
        ))
    else:
        # Use daily precipitation data
        if not daily.empty:
            day_width = 24 * 3600 * 1000 * 0.6  # 60% of a day width
            fig.add_trace(go.Bar(
                x=daily["date"], y=daily["rain"], 
                name="Rainfall (in)",
                marker=dict(color=RAIN_COLOR, opacity=0.6),
                yaxis="y2", 
                width=day_width,
                hovertemplate="%{x|%b %d}<br>Rain: %{y:.2f} in<extra></extra>"
            ))

    # ========== CONFIGURE LAYOUT ==========
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=title,
            x=0.01, xanchor="left",
            font=dict(size=TITLE_FONT_SIZE, family="Arial", color=DARK_GRAY, weight="bold")
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            font=dict(size=LEGEND_FONT_SIZE, family="Arial", color=DARK_GRAY)
        ),
        margin=dict(l=80, r=100, t=100, b=130),
        width=png_width,
        height=png_height,
        shapes=shapes,
        annotations=annotations,
        font=dict(family="Arial", size=TICK_FONT_SIZE, color=DARK_GRAY)
    )

    # Configure X-axis
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
        title=dict(
            text="Datetime", 
            font=dict(size=AXIS_LABEL_FONT_SIZE, family="Arial", color=DARK_GRAY)
        ),
        tickfont=dict(size=TICK_FONT_SIZE, family="Arial", color=DARK_GRAY),
        range=[start - timedelta(minutes=1), end + timedelta(minutes=1)]
    )

    # Configure Y-axes
    # Left axis: Temperature
    fig.update_yaxes(
        title=dict(
            text="Temperature (°F)", 
            font=dict(size=AXIS_LABEL_FONT_SIZE, family="Arial", color=TEMP_COLOR)
        ),
        tickfont=dict(size=TICK_FONT_SIZE, family="Arial", color=TEMP_COLOR),
        showgrid=True,
        gridcolor=LIGHT_GRAY,
        gridwidth=1,
        zeroline=False
    )

    # Right axis: Humidity / Precipitation
    fig.update_layout(
        yaxis2=dict(
            title=dict(
                text="Humidity (%) / Rain (in)", 
                font=dict(size=AXIS_LABEL_FONT_SIZE, family="Arial", color=HUM_COLOR)
            ),
            tickfont=dict(size=TICK_FONT_SIZE, family="Arial", color=HUM_COLOR),
            overlaying="y",
            side="right",
            rangemode="tozero"
        )
    )

    # ========== EXPORT PNG ==========
    try:
        out_path = Path(out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(out_path), width=png_width, height=png_height, scale=1)
        print(f"✅ Weather plot saved: {out_path}")
        return True
    except Exception as e:
        print(f"❌ Weather plot export failed: {e}")
        print("   Install kaleido: pip install kaleido")
        return False

