# app.py
# Streamlit NN-Sortierer mit Kamera-OCR (Google Vision) unter Verwendung eines einzigen GOOGLE_API_KEY
import streamlit as st
import os
import io
import csv
import base64
import requests
import hashlib
import re
from typing import List, Tuple
from math import radians, sin, cos, atan2, sqrt

# Optional: lokale OCR (falls du Tesseract auf dem Host installierst)
try:
    from PIL import Image, ImageOps
    import pytesseract
    TESSERACT_AVAILABLE = True
    try:
        _ = pytesseract.get_tesseract_version()
        TESSERACT_BINARY_OK = True
    except Exception:
        TESSERACT_BINARY_OK = False
except Exception:
    TESSERACT_AVAILABLE = False
    TESSERACT_BINARY_OK = False

# Geopy (Google geocoding)
try:
    from geopy.geocoders import GoogleV3
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

# ----------------- KONFIGURATION -----------------
HOME_ADDRESS_DEFAULT = "Rudolf-Diesel-Straße 2, 35463 Fernwald, Germany"
st.set_page_config(page_title="NN-Sortierer (Streamlit) mit Kamera-OCR", layout="centered")
st.title("Nearest-Neighbor Sortierer — Adressen (mit Kamera-OCR)")

# ----------------- Hilfsfunktionen -----------------
def get_google_api_key():
    if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.environ.get("GOOGLE_API_KEY")

GOOGLE_API_KEY = get_google_api_key()

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

# ----------------- Google Vision OCR (server-side) -----------------
def ocr_image_with_google_vision(image_bytes: bytes, api_key: str) -> str:
    """Sendet Bildbytes an Google Vision TEXT_DETECTION und gibt erkannten Text zurück."""
    if not api_key:
        raise RuntimeError("Kein GOOGLE_API_KEY gesetzt für Vision OCR.")
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [
            {
                "image": {"content": b64},
                "features": [{"type": "TEXT_DETECTION", "maxResults": 1}],
                "imageContext": {"languageHints": ["de"]}
            }
        ]
    }
    resp = requests.post(url, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    try:
        annotations = data["responses"][0].get("textAnnotations", [])
        if annotations:
            return annotations[0].get("description", "")  # gesamter erkannter Text
        return ""
    except Exception:
        return ""

def ocr_image_to_text(image_bytes_io: io.BytesIO) -> str:
    """Versucht lokal pytesseract, sonst Google Vision (benötigt GOOGLE_API_KEY)."""
    # lokales tesseract (wenn vorhanden)
    if TESSERACT_AVAILABLE and TESSERACT_BINARY_OK:
        img = Image.open(image_bytes_io)
        img = ImageOps.exif_transpose(img).convert("L")
        if max(img.size) < 1200:
            img = img.resize((int(img.size[0]*2), int(img.size[1]*2)), Image.LANCZOS)
        return pytesseract.image_to_string(img, lang='deu+eng')

    # fallback: Google Vision
    if GOOGLE_API_KEY:
        return ocr_image_with_google_vision(image_bytes_io.getvalue(), GOOGLE_API_KEY)

    raise RuntimeError("Weder lokales Tesseract noch GOOGLE_API_KEY für Vision verfügbar.")

# ----------------- Heuristische Extraktion einer Adresse aus OCR-Text -----------------
def extract_address_from_text(text: str) -> str:
    if not text:
        return None
    # Normalize newlines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = "\n".join(lines)
    # 1) Suche PLZ + Ort (z.B. 12345 Musterstadt)
    m = re.search(r"(\d{5}\s+[A-Za-zÄÖÜäöüß\\-\\. ]{2,})", joined)
    if m:
        # finde Zeile mit PLZ, nehme vorherige Zeile als Straße (falls vorhanden)
        for i, ln in enumerate(lines):
            if m.group(1) in ln:
                street = lines[i-1] if i-1 >= 0 else None
                if street and re.search(r"\d", street):  # enthält Hausnummer
                    return f"{street}, {lines[i]}"
                return lines[i]
    # 2) Suche Zeile mit Straße + Hausnummer Muster
    for ln in lines:
        if re.search(r"\b\S+\s+\d+[a-zA-Z]?\b", ln):  # einfache Straße + Nr Erkennung
            return ln
    # 3) fallback: erste Zeile
    return lines[0] if lines else None

# ----------------- Google Geocoding (geopy) -----------------
def geocode_address_google(address: str):
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY nicht gesetzt für Geocoding.")
    if not GEOPY_AVAILABLE:
        raise RuntimeError("geopy nicht installiert. pip install geopy")
    g = GoogleV3(api_key=GOOGLE_API_KEY, timeout=10)
    loc = g.geocode(address)
    if loc is None:
        raise ValueError(f"Adresse nicht gefunden: {address}")
    return (loc.latitude, loc.longitude)

# ----------------- SESSION STATE Initialisierung -----------------
if 'scanning' not in st.session_state:
    st.session_state['scanning'] = False
if 'scan_idx' not in st.session_state:
    st.session_state['scan_idx'] = 0
if 'camera_last_hash' not in st.session_state:
    st.session_state['camera_last_hash'] = None
if 'scanned_text' not in st.session_state:
    st.session_state['scanned_text'] = ""

# ----------------- UI: Home-Adresse / Datei / Kamera-Steuerung -----------------
home_addr = st.text_input("Home-Adresse (Startpunkt)", value=HOME_ADDRESS_DEFAULT)

uploaded_file = st.file_uploader("adressen.txt / adressen.csv hochladen (optional)", type=["txt","csv"])
if uploaded_file is not None:
    try:
        content = uploaded_file.read().decode("utf-8")
        st.session_state['scanned_text'] = "\n".join([line.strip() for line in content.splitlines() if line.strip()])
    except Exception as e:
        st.error(f"Fehler beim Lesen der Datei: {e}")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button("Start Kamera-Scan"):
        st.session_state['scanning'] = True
with col2:
    if st.button("Stop Kamera-Scan"):
        st.session_state['scanning'] = False
with col3:
    auto_add = st.checkbox("Automatisch erkannte Adresse sofort einfügen", value=True)

# Prüfe ob OCR möglich ist (lokal oder via Vision key)
ocr_possible = (TESSERACT_AVAILABLE and TESSERACT_BINARY_OK) or bool(GOOGLE_API_KEY)
if not ocr_possible:
    st.warning(
        "Weder lokales Tesseract noch GOOGLE_API_KEY verfügbar. Kamera-Scan deaktiviert.\n"
        "Setze GOOGLE_API_KEY als Streamlit Secret oder Environment-Variable oder installiere Tesseract auf dem Host."
    )

# Kamera-Input (wird neu gerendert, wenn scan_idx erhöht wird)
if st.session_state['scanning']:
    if not ocr_possible:
        st.info("Kamera-Scan ist deaktiviert, weil keine OCR-Möglichkeit gefunden wurde.")
    else:
        st.markdown("**Kamera aktiv — mache ein Foto einer Adresse. Nach der Erkennung bleibt die Kamera zur weiteren Nutzung bereit.**")
        cam_key = f"cam_{st.session_state['scan_idx']}"
        img_file = st.camera_input("Foto aufnehmen", key=cam_key)
        if img_file is not None:
            img_bytes = img_file.getvalue()
            # Hash prüfen um Doppelverarbeitung zu vermeiden
            h = hashlib.sha256(img_bytes).hexdigest()
            if h == st.session_state.get('camera_last_hash'):
                st.info("Dieses Foto wurde bereits verarbeitet. Bitte neues Foto aufnehmen.")
            else:
                st.session_state['camera_last_hash'] = h
                with st.spinner("OCR wird ausgeführt..."):
                    try:
                        text = ocr_image_to_text(io.BytesIO(img_bytes))
                        addr = extract_address_from_text(text)
                        if addr:
                            if auto_add:
                                existing = st.session_state.get('scanned_text', '').strip()
                                if existing:
                                    st.session_state['scanned_text'] = existing + "\n" + addr
                                else:
                                    st.session_state['scanned_text'] = addr
                                st.success(f"Adresse erkannt und hinzugefügt: {addr}")
                            else:
                                st.info(f"Erkannte Adresse: {addr}")
                                if st.button("Erkannte Adresse einfügen"):
                                    existing = st.session_state.get('scanned_text', '').strip()
                                    if existing:
                                        st.session_state['scanned_text'] = existing + "\n" + addr
                                    else:
                                        st.session_state['scanned_text'] = addr
                                    st.success("Adresse eingefügt.")
                        else:
                            st.warning("Keine Adresse im Bild erkannt. Rohtext (zur Prüfung):")
                            st.text_area("OCR Rohtext", value=text, height=150)
                    except Exception as e:
                        st.error(f"OCR fehlgeschlagen: {e}")
                # erhöht scan_idx damit ein neues camera_input-Widget gerendert wird und Nutzer sofort weiterscannen kann
                st.session_state['scan_idx'] += 1

# Haupt Adress-Textfeld (vorbefüllt durch Scans oder Datei)
initial_text = st.session_state.get('scanned_text', '')
text_input = st.text_area("Adressen (eine pro Zeile)", height=200, value=initial_text, placeholder="Adresse 1\nAdresse 2\n...")
priority_input = st.text_area("Priorisierte Adressen (optional, werden zuerst besucht)", height=100, placeholder="Adresse A\nAdresse B\n...")

submit = st.button("Sortieren")
if not submit:
    st.info("Gib Adressen ein (oder scanne sie per Kamera) und klicke auf 'Sortieren', um die Reihenfolge zu berechnen.")
    st.stop()

# ----------------- Adressen sammeln und Geokodieren -----------------
addresses = parse_addresses_from_text(text_input)
if not addresses:
    st.warning("Keine Adressen eingegeben.")
    st.stop()

coords_map = {}
geocode_errors = []
if not GEOPY_AVAILABLE:
    st.error("geopy nicht installiert. Installiere mit: pip install geopy")
    st.stop()

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

    remaining_coords_map = {a: coords_map[a] for a in coords_map if a not in [p[0] for p in priority_coords]}

    if priority_coords:
        start_coord_nn = priority_coords[-1][1]
    else:
        start_coord_nn = home_coord

    sorted_remaining = nearest_neighbor_sort_by_coords(remaining_coords_map, start_coord_nn)
    final_route = [p[0] for p in priority_coords] + sorted_remaining

# ----------------- Ergebnisse -----------------
st.subheader("Sortierte Adressen")
st.text_area("Ergebnis (eine Adresse pro Zeile)", value="\n".join(final_route), height=250)

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download .txt", data=generate_txt(final_route), file_name="sortierte_adressen.txt", mime="text/plain")
with col2:
    st.download_button("Download .csv", data=generate_csv_bytes(final_route), file_name="sortierte_adressen.csv", mime="text/csv")

if geocode_errors:
    st.warning(f"Es gab {len(geocode_errors)} Geokodier-Fehler. Siehe Details unten.")
    for a, e in geocode_errors:
        st.write(f"- {a} → {e}")

st.success("Fertig — die Liste wurde sortiert.")
