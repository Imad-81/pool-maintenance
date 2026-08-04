#!/usr/bin/env python3
"""
Open-Meteo Daily Weather Data Fetcher & CSV Exporter

Features:
- Downloads maximum available daily weather variables from Open-Meteo API.
- Accepts location by City Name (via Open-Meteo Geocoding API) or Latitude/Longitude.
- Handles custom date ranges (start_date to end_date in YYYY-MM-DD format).
- Seamlessly routes between Historical Archive API and Forecast API based on dates.
- Translates WMO weather codes into human-readable descriptions.
- Zero external dependencies required (uses Python standard library).
- Exposes both a reusable Python function and a Command-Line Interface (CLI).
"""

import argparse
import csv
from datetime import datetime, date, timedelta
import json
import os
import sys
import urllib.parse
import urllib.request

# Open-Meteo API Endpoints
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# Full list of daily variables supported by Open-Meteo
ALL_DAILY_VARIABLES = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "sunrise",
    "sunset",
    "daylight_duration",
    "sunshine_duration",
    "uv_index_max",
    "uv_index_clear_sky_max",
    "precipitation_sum",
    "rain_sum",
    "showers_sum",
    "snowfall_sum",
    "precipitation_hours",
    "precipitation_probability_max",
    "precipitation_probability_min",
    "precipitation_probability_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

# Map WMO weather codes to human-readable descriptions
WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def geocode_location(location_name: str) -> dict:
    """
    Search for a location by name using Open-Meteo Geocoding API.
    Returns a dictionary with name, country, latitude, and longitude.
    """
    params = {
        "name": location_name,
        "count": 1,
        "language": "en",
        "format": "json"
    }
    url = f"{GEOCODING_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "OpenMeteoFetcher/1.0"})

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        results = data.get("results", [])
        if not results:
            raise ValueError(f"No coordinates found for location: '{location_name}'")
        
        best = results[0]
        return {
            "name": best.get("name"),
            "country": best.get("country", ""),
            "admin1": best.get("admin1", ""),
            "latitude": float(best["latitude"]),
            "longitude": float(best["longitude"]),
            "timezone": best.get("timezone", "auto")
        }
    except Exception as err:
        print(f"[Error] Geocoding failed: {err}", file=sys.stderr)
        raise


def _fetch_single_range(api_url: str, params: dict) -> dict:
    """Helper function to perform an HTTP request to Open-Meteo API."""
    query = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{api_url}?{query}"
    req = urllib.request.Request(full_url, headers={"User-Agent": "OpenMeteoFetcher/1.0"})

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP Error {http_err.code} from {api_url}: {err_body}") from http_err
    except Exception as err:
        raise RuntimeError(f"Request failed: {err}") from err


def fetch_daily_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "auto",
    temperature_unit: str = "celsius",
    wind_speed_unit: str = "kmh",
    precipitation_unit: str = "mm",
) -> tuple[list[dict], dict]:
    """
    Fetch daily weather data for given coordinates and date range (YYYY-MM-DD).
    Returns (rows, units_dict).
    """
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    if start_dt > end_dt:
        raise ValueError(f"start_date ({start_date}) cannot be after end_date ({end_date}).")

    # Open-Meteo Archive API supports historical up to ~5 days ago
    cutoff_date = date.today() - timedelta(days=5)

    date_ranges = []
    if start_dt <= cutoff_date:
        arch_end = min(end_dt, cutoff_date)
        date_ranges.append((ARCHIVE_API_URL, start_dt.strftime("%Y-%m-%d"), arch_end.strftime("%Y-%m-%d")))
    if end_dt > cutoff_date:
        fore_start = max(start_dt, cutoff_date + timedelta(days=1))
        date_ranges.append((FORECAST_API_URL, fore_start.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")))

    combined_daily = {}
    units = {}
    elevation = None

    for api_endpoint, s_str, e_str in date_ranges:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": s_str,
            "end_date": e_str,
            "daily": ALL_DAILY_VARIABLES,
            "timezone": timezone,
            "temperature_unit": temperature_unit,
            "wind_speed_unit": wind_speed_unit,
            "precipitation_unit": precipitation_unit,
        }

        res = _fetch_single_range(api_endpoint, params)
        if elevation is None:
            elevation = res.get("elevation", 0.0)

        raw_daily = res.get("daily", {})
        raw_units = res.get("daily_units", {})
        units.update(raw_units)

        for key, val_list in raw_daily.items():
            if key not in combined_daily:
                combined_daily[key] = []
            combined_daily[key].extend(val_list)

    if not combined_daily or "time" not in combined_daily:
        raise RuntimeError("No daily data received from Open-Meteo API.")

    # Convert dictionary of lists into a list of row dicts
    num_records = len(combined_daily["time"])
    rows = []

    for i in range(num_records):
        row = {
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": elevation,
            "date": combined_daily["time"][i],
        }

        # Weather condition description
        code = combined_daily.get("weather_code", [None] * num_records)[i]
        row["weather_code"] = code
        row["weather_description"] = WMO_WEATHER_CODES.get(code, "Unknown") if code is not None else ""

        # Map remaining variables
        for var in ALL_DAILY_VARIABLES:
            if var == "weather_code":
                continue
            val = combined_daily.get(var, [None] * num_records)[i]
            row[var] = val

        rows.append(row)

    return rows, units


def save_to_csv(rows: list[dict], output_file: str, units: dict = None) -> None:
    """Save rows of weather data to a CSV file."""
    if not rows:
        print("[Warning] No rows to write to CSV.", file=sys.stderr)
        return

    fieldnames = list(rows[0].keys())

    # Build directory if needed
    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully saved {len(rows)} weather records to '{output_file}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Download comprehensive daily weather data from Open-Meteo API and save to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    loc_group = parser.add_mutually_exclusive_group(required=True)
    loc_group.add_argument("--location", type=str, help="City name or address (e.g. 'London', 'New York, US', 'Tokyo')")
    loc_group.add_argument("--lat", type=float, help="Latitude of target location")

    parser.add_argument("--lon", type=float, help="Longitude of target location (required if --lat is used)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="weather_data.csv", help="Output CSV file path")
    parser.add_argument("--timezone", type=str, default="auto", help="Timezone (e.g. 'auto', 'UTC', 'America/New_York')")
    parser.add_argument("--temp-unit", choices=["celsius", "fahrenheit"], default="celsius", help="Temperature unit")
    parser.add_argument("--wind-unit", choices=["kmh", "ms", "mph", "kn"], default="kmh", help="Wind speed unit")
    parser.add_argument("--precip-unit", choices=["mm", "inch"], default="mm", help="Precipitation unit")

    args = parser.parse_args()

    # Determine coordinates
    if args.location:
        print(f"Geocoding location: '{args.location}'...")
        loc_info = geocode_location(args.location)
        lat = loc_info["latitude"]
        lon = loc_info["longitude"]
        tz = loc_info["timezone"] if args.timezone == "auto" else args.timezone
        loc_label = f"{loc_info['name']}, {loc_info['country']}"
        print(f"Resolved to: {loc_label} (Lat: {lat}, Lon: {lon})")
    else:
        if args.lon is None:
            parser.error("--lon is required when --lat is specified.")
        lat = args.lat
        lon = args.lon
        tz = args.timezone
        print(f"Using coordinates: Lat {lat}, Lon {lon}")

    print(f"Fetching daily weather data from {args.start} to {args.end}...")
    rows, units = fetch_daily_weather(
        latitude=lat,
        longitude=lon,
        start_date=args.start,
        end_date=args.end,
        timezone=tz,
        temperature_unit=args.temp_unit,
        wind_speed_unit=args.wind_unit,
        precipitation_unit=args.precip_unit,
    )

    save_to_csv(rows, args.output, units=units)


if __name__ == "__main__":
    main()
