# app.py — erweitert um Kamera-OCR-Scanning
# Deutschsprachige UI

import streamlit as st
from geopy.geocoders import GoogleV3
from math import radians, sin, cos, atan2, sqrt
import csv
import io
from typing import List, Tuple
import os
import re
import hashlib

# --- optionale OCR-Bibliotheken ---
try:
    from PIL import Image, ImageOps
    import pytesseract
    TESSERACT_AVAILABLE = True
    try:
        # prüfe, ob tesseract-binary vorhanden ist
        _ = pytesseract.get_tesseract_version()
        TESSERACT_BINARY_OK = True
    except Exception:
        TESSERACT_BINARY_OK = False
except Exception:
    TESSERACT_AVAILABLE = False
    TESSERACT_BINARY_OK = False

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

# ----------------- Geocoding-Funktion -----------------

def geocode_address_google(address: str):
    g = GoogleV3(api_key=GOOGLE_API_KEY, timeout=10)
    loc = g.geocode(address)
    if loc is None:
        raise ValueError(f"Adresse nicht gefunden: {address}")
    return (loc.latitude, loc.longitude)

# ----------------- OCR Hilfsfunktionen -----------------

def ocr_image_with_google_vision(image_bytes: bytes, api_key: str) -> str:
    """Sendet ein Bild an Google Vision API (TEXT_DETECTION) und gibt den erkannten Text zurück.
    Erwartet die Rohbytes des Bildes sowie einen gültigen API-Key (Serverseitig, z.B. st.secrets oder ENV).
    """
    import base64
    import requests

    if not api_key:
        raise RuntimeError("Kein GOOGLE_VISION_API_KEY gesetzt.")
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
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
            return annotations[0].get("description", "")
        return ""
    except Exception:
        return ""


def ocr_image_to_text(image_bytes_io: io.BytesIO) -> str:
    """Versucht zunächst, pytesseract zu verwenden (falls lokal installiert).
    Falls pytesseract nicht verfügbar ist, fällt diese Funktion auf
    Google Vision API zurück (benötigt st.secrets['GOOGLE_VISION_API_KEY'] oder ENV).
    """
    # 1) lokales Tesseract wenn möglich
    if TESSERACT_AVAILABLE and TESSERACT_BINARY_OK:
        img = Image.open(image_bytes_io)
        img = ImageOps.exif_transpose(img).convert("L")
        if max(img.size) < 1200:
            img = img.resize((int(img.size[0]*2), int(img.size[1]*2)), Image.LANCZOS)
        return pytesseract.image_to_string(img, lang='deu+eng')

    # 2) fallback auf Google Vision API
    api_key = None
    if hasattr(st, "secrets"):
        api_key = st.secrets.get("GOOGLE_VISION_API_KEY")
    if not api_key:
        api_key = os.environ.get("GOOGLE_VISION_API_KEY")

    if api_key:
        return ocr_image_with_google_vision(image_bytes_io.getvalue(), api_key)

    raise RuntimeError("Weder pytesseract noch GOOGLE_VISION_API_KEY verfügbar. Bitte Tesseract installieren oder GOOGLE_VISION_API_KEY setzen.")

def extract_address_from_text(text: str):(text: str) -> str:
    """Heuristische Extraktion einer Adresse aus OCR-Text.
    Gibt die gefundene Adresse als einzelne Zeile zurück oder None.
    Diese Funktion ist intentionally konservativ; du kannst sie an dein Datenformat anpassen.
    """
    if not text:
        return None
    # 1) Suche nach PLZ + Ortsname (z.B. "35463 Fernwald")
    m = re.search(r"(\d{5}\s+[A-Za-zÄÖÜäöüß\-\. ]{2,})", text)
    if m:
        # versuche vorhergehende Zeile als Straße+Hausnummer hinzuzufügen
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # finde Index der Zeile die die PLZ enthält
        for i, ln in enumerate(lines):
            if m.group(1) in ln:
                street = lines[i-1] if i-1 >= 0 else None
                if street and re.search(r"\d", street):
                    return f"{street}, {lines[i]}"
                return lines[i]
    # 2) Falls keine PLZ gefunden: suche nach Muster Straße + Hausnummer
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if re.search(r"\d", ln) and len(ln) > 6:
            # einfache Annahme: enthält Ziffer => evtl. Straße+Nr
            return ln
    # 3) fallback: erste nicht-leere Zeile (sehr konservativ)
    return lines[0] if lines else None

# ----------------- UI: Home-Adresse + Adresseneingabe + Kamera-Scanner -----------------

home_addr = st.text_input("Home-Adresse (Startpunkt)", value=HOME_ADDRESS_DEFAULT)

# Datei hochladen und Textfeld automatisch befüllen
uploaded_file = st.file_uploader("adressen.txt / adressen.csv hochladen (optional)", type=["txt","csv"])
prefill_text = ""
if uploaded_file is not None:
    try:
        file_addresses = parse_addresses_from_file(uploaded_file)
        prefill_text = "\n".join(file_addresses)
    except Exception as e:
        st.error(f"Fehler beim Lesen der Datei: {e}")

# Session-state initialisieren für Scanner
if 'scanning' not in st.session_state:
    st.session_state['scanning'] = False
if 'camera_last_hash' not in st.session_state:
    st.session_state['camera_last_hash'] = None
if 'scanned_text' not in st.session_state:
    st.session_state['scanned_text'] = prefill_text

# Scanner-Steuerung
col_scan1, col_scan2, col_scan3 = st.columns([1,1,2])
with col_scan1:
    if st.button("Start Kamera-Scan"):
        st.session_state['scanning'] = True
with col_scan2:
    if st.button("Stop Kamera-Scan"):
        st.session_state['scanning'] = False
with col_scan3:
    st.checkbox("Automatisch erkannte Adresse sofort einfügen", value=True, key='auto_add_addr')

# Falls pytesseract/Tesseract nicht verfügbar: Hinweis
if not (TESSERACT_AVAILABLE and TESSERACT_BINARY_OK):
    st.warning("OCR (pytesseract/Tesseract) ist nicht vollständig verfügbar. Kamera-Scan ist deaktiviert.\n" 
               "Installationshinweise: pip install pillow pytesseract und auf Debian/Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-deu")

# Kamera-Scanning UI (bleibt sichtbar, solange 'scanning' True ist)
if st.session_state['scanning']:
    st.markdown("**Kamera aktiv — nimm ein Foto einer Adresse auf. Nach der Erkennung bleibt die Kamera aktiv, du kannst mehrere Adressen nacheinander scannen.**")
    if TESSERACT_AVAILABLE and TESSERACT_BINARY_OK:
        img_file = st.camera_input("Kamera: Adresse aufnehmen")
        if img_file is not None:
            img_bytes = img_file.getvalue()
            h = hashlib.sha256(img_bytes).hexdigest()
            if h != st.session_state['camera_last_hash']:
                st.session_state['camera_last_hash'] = h
                try:
                    text = ocr_image_to_text(io.BytesIO(img_bytes))
                    addr = extract_address_from_text(text)
                    if addr:
                        if st.session_state.get('auto_add_addr', True):
                            # füge zur internen Scanned-Text-Liste hinzu
                            existing = st.session_state.get('scanned_text', '').strip()
                            if existing:
                                st.session_state['scanned_text'] = existing + '\n' + addr
                            else:
                                st.session_state['scanned_text'] = addr
                            st.success(f"Adresse erkannt und hinzugefügt: {addr}")
                        else:
                            st.info(f"Adresse erkannt: {addr}")
                            if st.button("Erkannte Adresse einfügen"):
                                existing = st.session_state.get('scanned_text', '').strip()
                                if existing:
                                    st.session_state['scanned_text'] = existing + '\n' + addr
                                else:
                                    st.session_state['scanned_text'] = addr
                                st.success("Adresse eingefügt.")
                    else:
                        st.warning("Keine Adresse im Bild erkannt. Rohtext (zur Prüfung):")
                        st.text_area("OCR Rohtext", value=text, height=150)
                except Exception as e:
                    st.error(f"OCR fehlgeschlagen: {e}")
            else:
                st.info("Dasselbe Bild wurde bereits verarbeitet — bitte ein neues Foto aufnehmen.")
    else:
        st.info("OCR-Bibliothek nicht verfügbar — siehe Installationshinweis oben.")

# Das Haupt-Textfeld für Adressen: wird mit gescannten Adressen vorbefüllt (falls vorhanden) oder mit Datei-Inhalt
initial_text = st.session_state.get('scanned_text', '') if st.session_state.get('scanned_text') else prefill_text
text_input = st.text_area("Adressen (eine pro Zeile)", height=200, value=initial_text, placeholder="Adresse 1\nAdresse 2\n...")
priority_input = st.text_area("Priorisierte Adressen (optional, werden zuerst besucht)", height=100, placeholder="Adresse A\nAdresse B\n...")

submit = st.button("Sortieren")

if not submit:
    st.info("Gib Adressen ein (oder scanne sie per Kamera) und klicke auf 'Sortieren', um die Reihenfolge zu berechnen.")
    st.stop()

# ----------------- Adressen sammeln -----------------
addresses = parse_addresses_from_text(text_input)
if not addresses:
    st.warning("Keine Adressen eingegeben.")
    st.stop()

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

    # Restliche Adressen ohne Duplikate der Prioritäten
    remaining_coords_map = {a: coords_map[a] for a in coords_map if a not in [p[0] for p in priority_coords]}

    # NN-Sortierung startet vom letzten priorisierten Punkt oder Home, falls keine Priorität
    if priority_coords:
        start_coord_nn = priority_coords[-1][1]
    else:
        start_coord_nn = home_coord

    sorted_remaining = nearest_neighbor_sort_by_coords(remaining_coords_map, start_coord_nn)

    # Endgültige Route = Priorisierte Adressen + restliche Adressen (Home wird nicht in der Liste gezeigt)
    final_route = [p[0] for p in priority_coords] + sorted_remaining

# ----------------- Ergebnisanzeige + Download -----------------
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
