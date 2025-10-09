# app.py
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
import streamlit.components.v1 as components

# ----------------- Konfiguration -----------------
HOME_ADDRESS_DEFAULT = "Rudolf-Diesel-Straße 2, 35463 Fernwald, Germany"
st.set_page_config(page_title="NN-Sortierer mit Kamera-OCR", layout="centered")
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

# ----------------- Google Vision OCR -----------------
def ocr_image_with_google_vision(image_bytes: bytes, api_key: str) -> str:
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
            return annotations[0].get("description", "")
        return ""
    except Exception:
        return ""

def ocr_image_to_text(image_bytes_io: io.BytesIO) -> str:
    if GOOGLE_API_KEY:
        return ocr_image_with_google_vision(image_bytes_io.getvalue(), GOOGLE_API_KEY)
    raise RuntimeError("GOOGLE_API_KEY für Vision OCR nicht gesetzt.")

# ----------------- Extraktion Adresse -----------------
def extract_address_from_text(text: str) -> str:
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = "\n".join(lines)
    m = re.search(r"(\d{5}\s+[A-Za-zÄÖÜäöüß\\-\\. ]{2,})", joined)
    if m:
        for i, ln in enumerate(lines):
            if m.group(1) in ln:
                street = lines[i-1] if i-1 >= 0 else None
                if street and re.search(r"\d", street):
                    return f"{street}, {lines[i]}"
                return lines[i]
    for ln in lines:
        if re.search(r"\b\S+\s+\d+[a-zA-Z]?\b", ln):
            return ln
    return lines[0] if lines else None

# ----------------- Google Geocoding -----------------
try:
    from geopy.geocoders import GoogleV3
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

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

# ----------------- Session State -----------------
if 'scanning' not in st.session_state:
    st.session_state['scanning'] = False
if 'camera_image' not in st.session_state:
    st.session_state['camera_image'] = None
if 'scanned_text' not in st.session_state:
    st.session_state['scanned_text'] = ""

# ----------------- UI -----------------
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

ocr_possible = bool(GOOGLE_API_KEY)
if not ocr_possible:
    st.warning("GOOGLE_API_KEY nicht gesetzt. Kamera-Scan deaktiviert.")

# ----------------- Kamera Input -----------------
if st.session_state['scanning'] and ocr_possible:
    st.markdown("**Kamera aktiv — Foto aufnehmen. Nach der Erkennung bleibt die Kamera bereit.**")

    camera_html = '''
    <div style="max-width:480px; margin:auto; text-align:center;">
        <div style="overflow:hidden; max-height:320px; border:1px solid #ddd;">
            <video id="video" autoplay playsinline style="width:100%;"></video>
        </div>
        <div style="margin-top:8px;">
            <button id="capture" style="width:150px; height:40px; font-size:16px;">Foto aufnehmen</button>
            <span id="status" style="margin-left:8px;"></span>
        </div>
        <canvas id="canvas" style="display:none;"></canvas>
    </div>
    <script>
    (async function(){
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const status = document.getElementById('status');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video:{facingMode:{ideal:'environment'}, width:{ideal:1280}, height:{ideal:720}},
                audio:false
            });
            video.srcObject = stream;
            await video.play();
            const [track] = stream.getVideoTracks();
            const capabilities = track.getCapabilities();
            if(capabilities.zoom){
                track.applyConstraints({ advanced: [{ zoom: capabilities.zoom.max }] });
            }
            status.textContent='Kamera bereit';
        } catch(e){status.textContent='Kamera nicht verfügbar: '+e; return;}

        document.getElementById('capture').addEventListener('click', ()=>{
            const w = video.videoWidth, h = video.videoHeight;
            const cropW = Math.floor(w*0.8), cropH = Math.floor(h*0.5);
            const sx = Math.floor((w-cropW)/2), sy = Math.floor((h-cropH)/2);
            canvas.width = cropW; canvas.height = cropH;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video,sx,sy,cropW,cropH,0,0,cropW,cropH);
            const dataUrl = canvas.toDataURL('image/jpeg',0.9);
            window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setComponentValue',value:dataUrl}, '*');
            status.textContent='Foto gesendet';
        });
    })();
    </script>
    '''
    components.html(camera_html, height=400, key="camera_block")

# ----------------- OCR Verarbeitung -----------------
if st.session_state['camera_image']:
    img_bytes = base64.b64decode(st.session_state['camera_image'].split(",")[1])
    text = ocr_image_to_text(io.BytesIO(img_bytes))
    addr = extract_address_from_text(text)
    if addr:
        existing = st.session_state.get('scanned_text', '').strip()
        if existing:
            st.session_state['scanned_text'] = existing + "\n" + addr
        else:
            st.session_state['scanned_text'] = addr
        st.success(f"Adresse erkannt: {addr}")

# ----------------- Adress-Textfeld -----------------
text_input = st.text_area("Adressen (eine pro Zeile)", height=200, value=st.session_state.get('scanned_text', ''), placeholder="Adresse 1\nAdresse 2\n...")
priority_input = st.text_area("Priorisierte Adressen (optional)", height=100, placeholder="Adresse A\nAdresse B\n...")

submit = st.button("Sortieren")
if not submit:
    st.info("Adressen eingeben oder scannen und dann 'Sortieren' klicken.")
    st.stop()

# ----------------- Geokodierung und NN-Sortierung -----------------
addresses = parse_addresses_from_text(text_input)
if not addresses:
    st.warning("Keine Adressen eingegeben.")
    st.stop()

coords_map = {}
geocode_errors = []
if not GEOPY_AVAILABLE:
    st.error("geopy nicht installiert. pip install geopy")
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

    home_parsed = parse_latlon_line(home_addr)
    if home_parsed:
        home_coord = (home_parsed[0], home_parsed[1])
    else:
        try:
            home_coord = geocode_address_google(home_addr)
        except Exception as e:
            st.error(f"Home-Geokodierung fehlgeschlagen: {e}")
            st.stop()

    priority_list = [line.strip() for line in priority_input.splitlines() if line.strip()]
    priority_coords = []
    for addr in priority_list:
        try:
            priority_coords.append((addr, geocode_address_google(addr)))
        except Exception as e:
            st.warning(f"Priorisierte Adresse konnte nicht geokodiert werden: {addr} → {e}")

    remaining_coords_map = {a: coords_map[a] for a in coords_map if a not in [p[0] for p in priority_coords]}
    start_coord_nn = priority_coords[-1][1] if priority_coords else home_coord
    sorted_remaining = nearest_neighbor_sort_by_coords(remaining_coords_map, start_coord_nn)
    final_route = [p[0] for p in priority_coords] + sorted_remaining

# ----------------- Ergebnis -----------------
st.subheader("Sortierte Adressen")
st.text_area("Ergebnis", value="\n".join(final_route), height=250)
col1, col2 = st.columns(2)
with col1:
    st.download_button("Download .txt", data=generate_txt(final_route), file_name="sortierte_adressen.txt", mime="text/plain")
with col2:
    st.download_button("Download .csv", data=generate_csv_bytes(final_route), file_name="sortierte_adressen.csv", mime="text/csv")

if geocode_errors:
    st.warning(f"Es gab {len(geocode_errors)} Geokodier-Fehler:")
    for a,e in geocode_errors:
        st.write(f"- {a} → {e}")

st.success("Fertig — die Liste wurde sortiert.")
