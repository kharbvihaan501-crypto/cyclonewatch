"""Illustrative coastal exposure for a few Odisha towns (used by app/app.py).

ILLUSTRATIVE ONLY. Distances, not a forecast. Does not model storm surge, wind fields or flooding.

- Town coordinates: approximate values supplied by the team.
- Coastline: a coarse, hand-entered polyline of the Odisha coast (approximate points along the
  shore from the Andhra Pradesh border to the West Bengal border). It is good to a few km, enough
  to split "within 20 km of the coast" from "inland" for these towns, and nothing finer.
"""
import numpy as np

EARTH_RADIUS_KM = 6371.0
COASTAL_KM = 20.0  # within this distance of the coast = "coastal"

TOWNS = {  # name: (lat N, lon E), approximate
    "Puri": (19.81, 85.83),
    "Paradip": (20.32, 86.61),
    "Gopalpur": (19.26, 84.91),
    "Balasore": (21.49, 86.93),
    "Bhubaneswar": (20.30, 85.82),
    "Cuttack": (20.46, 85.88),
    "Angul": (20.84, 85.10),
    "Keonjhar": (21.63, 85.58),
}

# Approximate Odisha shoreline, south-west to north-east (lat N, lon E).
COAST = [
    (19.07, 84.77),  # near the Andhra Pradesh border
    (19.26, 84.93),  # Gopalpur
    (19.45, 85.10),
    (19.62, 85.42),  # Chilika outer shore
    (19.79, 85.82),  # Puri
    (19.88, 86.10),  # Konark
    (20.08, 86.43),
    (20.26, 86.68),  # Paradip
    (20.58, 86.85),
    (20.80, 86.98),  # Dhamra
    (21.10, 86.90),
    (21.45, 87.03),  # Chandipur
    (21.60, 87.48),  # near the West Bengal border
]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _dense_coast(step_deg: float = 0.01) -> np.ndarray:
    pts = []
    for (a_lat, a_lon), (b_lat, b_lon) in zip(COAST[:-1], COAST[1:]):
        n = max(2, int(np.hypot(b_lat - a_lat, b_lon - a_lon) / step_deg))
        pts.append(np.column_stack([np.linspace(a_lat, b_lat, n), np.linspace(a_lon, b_lon, n)]))
    return np.vstack(pts)


_DENSE = _dense_coast()


def coast_km(lat: float, lon: float) -> float:
    """Distance (km) from a point to the approximate coastline polyline."""
    return float(haversine_km(lat, lon, _DENSE[:, 0], _DENSE[:, 1]).min())


TOWN_COAST_KM = {name: coast_km(*ll) for name, ll in TOWNS.items()}


def exposure_rows(storm_lat: float, storm_lon: float) -> list[dict]:
    """One row per town: distance from coast, distance from the storm centre, exposure note."""
    rows = []
    for name, (lat, lon) in TOWNS.items():
        c = TOWN_COAST_KM[name]
        coastal = c <= COASTAL_KM
        rows.append({
            "town": name, "lat": lat, "lon": lon, "coastal": coastal,
            "coast_km": c, "storm_km": float(haversine_km(storm_lat, storm_lon, lat, lon)),
            "note": ("higher exposure to wind and surge" if coastal else
                     "lower wind exposure, but heavy rain and river flooding can still affect them"),
        })
    return sorted(rows, key=lambda r: r["storm_km"])
