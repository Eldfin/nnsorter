# app.py
import streamlit as st
from geopy.geocoders import GoogleV3
from math import radians, sin, cos, atan2, sqrt
import csv
import io
from typing import List, Tuple
import os

# ----------------- KONFIGURATION -----------------
HOME_ADDRESS_DEFAULT = "Rudolf-Diesel-Straße 2, 35463 Fernwald, Germany"
# -------------------------------------------------

st.set_page_config(page_title="NN-Sortierer (Streamlit)", layout="centered")
st.title("Nearest-Neighbor Sortierer — Adressen (Text-only)")

# ----------------- Hilfsfunktionen -----------------
def parse_addresses_from_file(f) -> List[str]:
    content = f.read().decode("utf-8")
    return [line.strip() for line in content.splitlines() if line.strip()]

def parse_addresses_from_text(t: str) -> List[str]:
    return [line.strip() for line in t.splitlines() if line.strip()]

def parse_latlon_line(line: str):
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

def generate_txt(data_list: List[str]) -> bytes:
    return "\n".join(data_list).encode("utf-8")

def generate_csv_bytes(data_list: List[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    for r in data_list:
        writer.writerow([r])
    return buf.getvalue().encode("utf-8")

# ----------------- Google API Key aus Secrets -----------------
def get_google_api_key():
    if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.environ.get("GOOGLE_API_KEY")

GOOGLE_API_KEY = get_google_api_key()
if not GOOGLE_API_KEY:
    st.error("Kein Google API Key gefunden. Bitte als Streamlit Secret oder Environment Variable setzen.")
    st.stop()

# ----------------- Geocoding-Funktion (Google API) -----------------
def geocode_address_google(address: str):
    g = GoogleV3(api_key=GOOGLE_API_KEY, timeout=10)
    loc = g.geocode(address)
    if loc is None:
        raise ValueError(f"Adresse nicht gefunden: {address}")
    return (loc.latitude, loc.longitude)

# ----------------- UI: Home-Adresse + Formular -----------------
home_addr = st.text_input("Home-Adresse (Start-Adresse)", value=HOME_ADDRESS_DEFAULT)

with st.form(key="input_form"):
    uploaded_file = st.file_uploader("adressen.txt / adressen.csv hochladen", type=["txt","csv"])
    
    # Vorbefüllen des Textfeldes, wenn Datei hochgeladen
    prefill_text = ""
    if uploaded_file is not None:
        try:
            file_addresses = parse_addresses_from_file(uploaded_file)
            prefill_text = "\n".join(file_addresses)
        except Exception as e:
            st.error(f"Fehler beim Lesen der Datei: {e}")
    
    st.markdown("**Adressen (eine pro Zeile)**")
    text_input = st.text_area("Adressen", height=200, value=prefill_text, placeholder="Vohwinkeler Str. 107, 42329 Wuppertal, Germany\nBundesallee 76, 42103 Wuppertal, Germany\n...")
    
    st.markdown("**Priorisierte Adressen (optional, diese werden zuerst besucht, eine pro Zeile):**")
    priority_input = st.text_area("Priorisierte Adressen", height=100, placeholder="Adresse 1\nAdresse 2\n...")

    submit = st.form_submit_button("Sortieren")

if not submit:
    st.info("Gib Adressen ein und klicke auf 'Sortieren', um die Reihenfolge zu berechnen.")
    st.stop()

# ----------------- Adressen sammeln -----------------
addresses = parse_addresses_from_text(text_input)
if not addresses:
    st.warning("Keine Adressen eingegeben.")
    st.stop()

st.write("Start (Home):", home_addr)
st.write(f"Anzahl Ziele: {len(addresses)}")

# ----------------- Geokodierung -----------------
coords_map = {}
geocode_errors = []

with st.spinner("Geokodieren..."):
    for line in addresses:
        parsed = parse_latlon_line(line)
        if parsed:
            lat, lon, label = parsed
            coords_map[label] = (lat, lon)
        else:
            try:
                coords_map[line] = geocode_address_google(line)
            except Exception as e:
                geocode_errors.append((line, str(e)))

    # Home-Adresse geokodieren
    home_parsed = parse_latlon_line(home_addr)
    if home_parsed:
        home_coord = (home_parsed[0], home_parsed[1])
    else:
        try:
            home_coord = geocode_address_google(home_addr)
        except Exception as e:
            st.error(f"Home-Geokodierung fehlgeschlagen: {e}")
            st.stop()

    # Priorisierte Adressen
    priority_list = [line.strip() for line in priority_input.splitlines() if line.strip()]
    priority_coords = []
    for addr in priority_list:
        try:
            priority_coords.append((addr, geocode_address_google(addr)))
        except Exception as e:
            st.warning(f"Priorisierte Adresse konnte nicht geokodiert werden und wird ignoriert: {addr} → {e}")

    # Entferne priorisierte Adressen aus normalen Zielen (falls doppelt vorhanden)
    remaining_coords_map = {a: coords_map[a] for a in coords_map if a not in [p[0] for p in priority_coords]}

    # NN-Sortierung ab der letzten priorisierten Adresse
    if priority_coords:
        start_coord_nn = priority_coords[-1][1]
    else:
        start_coord_nn = home_coord

    sorted_remaining = nearest_neighbor_sort_by_coords(remaining_coords_map, start_coord_nn)

    # Endgültige Route: Home + priorisierte Adressen + restliche Adressen
    final_route = [home_addr] + [p[0] for p in priority_coords] + sorted_remaining

# ----------------- Ergebnisanzeige + Download -----------------
st.subheader("Sortierte Adressen (Nearest Neighbor)")
st.text_area("Ergebnis (eine Adresse pro Zeile)", value="\n".join(final_route), height=250)

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download .txt", data=generate_txt(final_route), file_name="sortierte_adressen.txt", mime="text/plain")
with col2:
    st.download_button("Download .csv", data=generate_csv_bytes(final_route), file_name="sortierte_adressen.csv", mime="text/csv")

st.success("Fertig — die Liste wurde sortiert.")
