# app.py
import streamlit as st
from geopy.geocoders import Nominatim
from math import radians, sin, cos, atan2, sqrt
import time
import csv
import io
from typing import List, Tuple

# ----------------- KONFIGURATION -----------------
# HOME-Adresse fest im Code (Standard). Du kannst sie hier editieren.
HOME_ADDRESS_DEFAULT = "Vohwinkeler Str. 107, 42329 Wuppertal, Germany"

# user agent für Nominatim (bitte ggf. deine E-Mail angeben)
NOMINATIM_USER_AGENT = "nn_streamlit_app_your_email@example.com"

# Pause zwischen Geocoding-Anfragen (in Sekunden)
GEOCODE_PAUSE_S = 1.0
# -------------------------------------------------

st.set_page_config(page_title="NN-Sortierer (Streamlit)", layout="centered")

st.title("Nearest-Neighbor Sortierer — Adressen (Text-only)")

st.markdown(
    """
    Gib eine Liste von Adressen ein (eine pro Zeile) oder lade eine .txt/.csv Datei hoch.
    Die **Home-Adresse** wird zum Sortieren verwendet, erscheint aber **nicht** in der Ausgabe.
    """
)

# Sidebar: Optionen
st.sidebar.header("Einstellungen")
use_fixed_home = st.sidebar.checkbox("Home aus Code verwenden (fest)", value=True)
home_from_code = HOME_ADDRESS_DEFAULT
home_input = st.sidebar.text_input("Home-Adresse (falls nicht fest)", value=HOME_ADDRESS_DEFAULT)
pause_s = st.sidebar.number_input("Pause zwischen Geocoding-Requests (s)", value=GEOCODE_PAUSE_S, min_value=0.0, step=0.1)
user_agent = st.sidebar.text_input("Nominatim user-agent (empfohlen: Email)", value=NOMINATIM_USER_AGENT)

# Input: Datei oder Textfeld
uploaded_file = st.file_uploader("adressen.txt / adressen.csv hochladen (optional)", type=["txt", "csv"])
st.markdown("oder")
text_input = st.text_area("Adressen (eine pro Zeile)", height=200, placeholder="Vohwinkeler Str. 107, 42329 Wuppertal, Germany\nBundesallee 76, 42103 Wuppertal, Germany\n...")

# Option: Koordinaten-Format unterstützen (lat,lon,label)
st.info("Tipp: Wenn du Offline-Nutzung willst, kannst du auch Zeilen im Format `lat,lon,Label` eingeben (z.B. `51.255,7.133,Mein Ziel`).")

# Hilfsfunktionen
def parse_addresses_from_file(f) -> List[str]:
    content = f.read().decode("utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return lines

def parse_addresses_from_text(t: str) -> List[str]:
    return [line.strip() for line in t.splitlines() if line.strip()]

def parse_latlon_line(line: str):
    # prüfe Format lat,lon,label
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2:
        try:
            lat = float(parts[0])
            lon = float(parts[1])
            label = ", ".join(parts[2:]) if len(parts) > 2 else f"{lat},{lon}"
            return (lat, lon, label)
        except ValueError:
            return None
    return None

@st.cache_data(show_spinner=False)
def geocode_address_cached(address: str, user_agent_local: str) -> Tuple[float, float]:
    geolocator = Nominatim(user_agent=user_agent_local)
    loc = geolocator.geocode(address, timeout=10)
    if loc is None:
        raise ValueError(f"Geokodierung fehlgeschlagen für: {address}")
    return (loc.latitude, loc.longitude)

def haversine_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    R = 6371.0
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(hav), sqrt(1-hav))
    return R * c

def nearest_neighbor_sort_by_coords(coords_map: dict, start_coord: Tuple[float,float]) -> List[str]:
    unvisited = set(coords_map.keys())
    route = []
    current_coord = start_coord
    while unvisited:
        nearest = min(unvisited, key=lambda a: haversine_km(current_coord, coords_map[a]))
        route.append(nearest)
        unvisited.remove(nearest)
        current_coord = coords_map[nearest]
    return route

# === Input sammeln ===
addresses = []
if uploaded_file is not None:
    try:
        addresses = parse_addresses_from_file(uploaded_file)
    except Exception as e:
        st.error(f"Fehler beim Lesen der Datei: {e}")
else:
    addresses = parse_addresses_from_text(text_input)

if not addresses:
    st.warning("Keine Adressen eingegeben. Bitte Adressen eingeben oder Datei hochladen.")
    st.stop()

# Home bestimmen
if use_fixed_home:
    home_addr = home_from_code
else:
    home_addr = home_input.strip() or home_from_code

st.write("Start (Home):", home_addr)
st.write(f"Anzahl Ziele: {len(addresses)} (Home wird nicht in der Ausgabedatei erscheinen)")

# Geokodierung / Parsing
coords_map = {}  # address -> (lat,lon)
geocode_errors = []
with st.spinner("Geokodieren und sortieren..."):
    # 1) falls Zeilen als lat,lon,label gegeben sind, nutze direkt
    for line in addresses:
        parsed = parse_latlon_line(line)
        if parsed:
            lat, lon, label = parsed
            coords_map[label] = (lat, lon)
        else:
            # normale Adresse: geokodieren (cached)
            try:
                latlon = geocode_address_cached(line, user_agent)
                coords_map[line] = latlon
                # pause, um höflich zu Nominatim zu sein (nur wenn nicht im cache)
                time.sleep(pause_s)
            except Exception as e:
                geocode_errors.append((line, str(e)))

    # Geokodiere Home (falls Home auch lat,lon format)
    home_parsed = parse_latlon_line(home_addr)
    if home_parsed:
        home_coord = (home_parsed[0], home_parsed[1])
    else:
        try:
            home_coord = geocode_address_cached(home_addr, user_agent)
            time.sleep(pause_s)
        except Exception as e:
            st.error(f"Home-Geokodierung fehlgeschlagen: {e}")
            st.stop()

    # Wenn Geocoding-Fehler, anzeigen (aber wir versuchen weiter)
    if geocode_errors:
        st.warning("Einige Adressen konnten nicht geokodiert werden. Sie werden ignoriert:")
        for ln, err in geocode_errors:
            st.write(f"- {ln}  →  {err}")

    # Entferne fehlgeschlagene Adressen aus der Map
    valid_coords_map = {a: coords_map[a] for a in coords_map.keys() if a not in [e[0] for e in geocode_errors]}

    if not valid_coords_map:
        st.error("Keine gültigen geokodierten Adressen vorhanden.")
        st.stop()

    # NN sort
    sorted_list = nearest_neighbor_sort_by_coords(valid_coords_map, home_coord)

# Ergebnis anzeigen + Download
st.subheader("Sortierte Adressen (Nearest Neighbor)")
st.text_area("Ergebnis (eine Adresse pro Zeile)", value="\n".join(sorted_list), height=250)

# Download als TXT
def generate_txt(data_list: List[str]) -> bytes:
    txt = "\n".join(data_list)
    return txt.encode("utf-8")

def generate_csv_bytes(data_list: List[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    for r in data_list:
        writer.writerow([r])
    return buf.getvalue().encode("utf-8")

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download .txt", data=generate_txt(sorted_list), file_name="sortierte_adressen.txt", mime="text/plain")
with col2:
    st.download_button("Download .csv", data=generate_csv_bytes(sorted_list), file_name="sortierte_adressen.csv", mime="text/csv")

st.success("Fertig — die Liste wurde sortiert. Home wird nur zum Sortieren verwendet.")
st.caption("Hinweis: Die Sortierung benutzt Luftlinien (Haversine). Für straßenspezifische Reihenfolge ist ein Routing-Distance-API nötig (Google/OSRM).")
