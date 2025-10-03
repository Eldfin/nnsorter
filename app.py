# app.py
import streamlit as st
from geopy.geocoders import GoogleV3
from math import radians, sin, cos, atan2, sqrt
import csv
import io
from typing import List, Tuple
import os

# ----------------- KONFIGURATION -----------------
HOME_ADDRESS_DEFAULT = "Vohwinkeler Str. 107, 42329 Wuppertal, Germany"
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

# ----------------- Geocoding-Funktion (Google API, schnell) -----------------
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
    st.markdown("oder")
    text_input = st.text_area("Adressen (eine pro Zeile)", height=200, placeholder="Vohwinkeler Str. 107, 42329 Wuppertal, Germany\nBundesallee 76, 42103 Wuppertal, Germany\n...")
    submit = st.form_submit_button("Sortieren")

if not submit:
    st.info("Gib Adressen ein und klicke auf 'Sortieren', um die Reihenfolge zu berechnen.")
    st.stop()

# ----------------- Adressen sammeln -----------------
addresses = []
if uploaded_file is not None:
    try:
        addresses = parse_addresses_from_file(uploaded_file)
    except Exception as e:
        st.error(f"Fehler beim Lesen der Datei: {e}")
        st.stop()
else:
    addresses = parse_addresses_from_text(text_input)

if not addresses:
    st.warning("Keine Adressen eingegeben. Bitte Adressen eingeben oder Datei hochladen.")
    st.stop()

st.write("Start (Home):", home_addr)
st.write(f"Anzahl Ziele: {len(addresses)}")

# ----------------- Geokodierung -----------------
coords_map = {}
geocode_errors = []

with st.spinner("Geokodieren und sortieren..."):
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

    if geocode_errors:
        st.warning("Einige Adressen konnten nicht geokodiert werden und werden ignoriert:")
        for ln, err in geocode_errors:
            st.write(f"- {ln} → {err}")

    valid_coords_map = {a: coords_map[a] for a in coords_map.keys() if a not in [e[0] for e in geocode_errors]}

    if not valid_coords_map:
        st.error("Keine gültigen geokodierten Adressen vorhanden.")
        st.stop()

    sorted_list = nearest_neighbor_sort_by_coords(valid_coords_map, home_coord)

# ----------------- Ergebnisanzeige + Download -----------------
st.subheader("Sortierte Adressen (Nearest Neighbor)")
st.text_area("Ergebnis (eine Adresse pro Zeile)", value="\n".join(sorted_list), height=250)

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download .txt", data=generate_txt(sorted_list), file_name="sortierte_adressen.txt", mime="text/plain")
with col2:
    st.download_button("Download .csv", data=generate_csv_bytes(sorted_list), file_name="sortierte_adressen.csv", mime="text/csv")

st.success("Fertig — die Liste wurde sortiert.")
