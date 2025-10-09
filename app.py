import streamlit as st
from geopy.geocoders import GoogleV3
from math import radians, sin, cos, atan2, sqrt
import csv
import io
from typing import List, Tuple
import os
import base64
import hashlib
import requests

# ----------------- KONFIGURATION -----------------
HOME_ADDRESS_DEFAULT = "Rudolf-Diesel-Straße 2, 35463 Fernwald, Germany"
# -------------------------------------------------

st.set_page_config(page_title="NN-Sortierer (Streamlit)", layout="centered")
st.title("Nearest-Neighbor Sortierer — Adressen (Text-only + Kamera OCR)")

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
GOOGLE_API_KEY = None
if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Kein Google API Key gefunden. Bitte als Streamlit Secret oder Environment Variable setzen.")
    st.stop()

# ----------------- Google-Geocoding -----------------
def geocode_address_google(address: str):
    g = GoogleV3(api_key=GOOGLE_API_KEY, timeout=10)
    loc = g.geocode(address)
    if loc is None:
        raise ValueError(f"Adresse nicht gefunden: {address}")
    return (loc.latitude, loc.longitude)

# ----------------- OCR via Google Vision -----------------
def ocr_image_with_google_vision(img_bytes: bytes, api_key: str) -> str:
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    payload = {
        "requests": [{
            "image": {"content": base64.b64encode(img_bytes).decode('utf-8')},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data['responses'][0].get('fullTextAnnotation', {}).get('text', '')

# ----------------- Text zu Adresse extrahieren -----------------
def extract_address_from_text(text: str) -> str:
    # einfache Heuristik: nehme die Zeilen, die mindestens 3 Wörter haben
    for line in text.splitlines():
        if len(line.strip().split()) >= 3:
            return line.strip()
    return text.strip() if text else ''

# ----------------- UI -----------------
home_addr = st.text_input("Home-Adresse (Startpunkt)", value=HOME_ADDRESS_DEFAULT)

uploaded_file = st.file_uploader("adressen.txt / adressen.csv hochladen (optional)", type=["txt","csv"])
prefill_text = ""
if uploaded_file is not None:
    try:
        file_addresses = parse_addresses_from_file(uploaded_file)
        prefill_text = "\n".join(file_addresses)
    except Exception as e:
        st.error(f"Fehler beim Lesen der Datei: {e}")

# Kamera/OCR Session State
if 'scanning' not in st.session_state:
    st.session_state['scanning'] = True
if 'scanned_text' not in st.session_state:
    st.session_state['scanned_text'] = ''

# Kamera HTML (Rückkamera, mittlerer Bereich crop)
import streamlit.components.v1 as components
camera_html = '''
<div>
<video id="video" autoplay playsinline style="width:100%; max-width:560px; border:1px solid #ddd"></video>
<div style="margin-top:8px;">
<button id="capture">Foto aufnehmen</button>
<span id="status" style="margin-left:8px"></span>
</div>
<canvas id="canvas" style="display:none"></canvas>
</div>
<script>
(async function(){
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
let stream = null;
try { stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}, width:{ideal:1280}, height:{ideal:720}}, audio:false}); video.srcObject = stream; await video.play(); status.textContent='Kamera bereit'; }
catch(e){status.textContent='Kamera nicht verfügbar: '+e; return;}
document.getElementById('capture').addEventListener('click', ()=>{
const w=video.videoWidth, h=video.videoHeight;
const cropW=Math.floor(w*0.8), cropH=Math.floor(h*0.5), sx=Math.floor((w-cropW)/2), sy=Math.floor((h-cropH)/2);
canvas.width=cropW; canvas.height=cropH;
const ctx=canvas.getContext('2d'); ctx.drawImage(video,sx,sy,cropW,cropH,0,0,cropW,cropH);
const dataUrl=canvas.toDataURL('image/jpeg',0.9);
window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setComponentValue',value:dataUrl},'*');
status.textContent='Foto gesendet';
});
})();
</script>
'''
component_value = components.html(camera_html, height=480)

# Foto verarbeiten und OCR
if component_value:
    try:
        if isinstance(component_value, dict): s = component_value.get('value') or str(component_value)
        else: s = str(component_value)
        if s.startswith('data:'):
            header,b64=s.split(',',1)
            img_bytes = base64.b64decode(b64)
            h = hashlib.sha256(img_bytes).hexdigest()
            if h != st.session_state.get('camera_last_hash'):
                st.session_state['camera_last_hash'] = h
                with st.spinner('OCR via Google Vision...'):
                    text = ocr_image_with_google_vision(img_bytes, GOOGLE_API_KEY)
                    addr = extract_address_from_text(text)
                    if addr:
                        existing = st.session_state.get('scanned_text','').strip()
                        if existing: st.session_state['scanned_text'] = existing+'\n'+addr
                        else: st.session_state['scanned_text'] = addr
                        st.success(f"Adresse erkannt und hinzugefügt: {addr}")
    except Exception as e:
        st.error(f'Fehler bei der Verarbeitung des Kamerabildes: {e}')

# Textfeld für alle Adressen
text_input = st.text_area("Adressen (eine pro Zeile)", height=200, value=st.session_state.get('scanned_text','') or prefill_text, placeholder="Adresse 1\nAdresse 2\n...")
priority_input = st.text_area("Priorisierte Adressen (optional, werden zuerst besucht)", height=100, placeholder="Adresse A\nAdresse B\n...")
submit = st.button("Sortieren")

if submit:
    addresses = parse_addresses_from_text(text_input)
    if not addresses:
        st.warning("Keine Adressen eingegeben.")
        st.stop()

    coords_map = {}
    geocode_errors = []
    for line in addresses:
        parsed = parse_latlon_line(line)
        if parsed: coords_map[parsed[2]]=(parsed[0],parsed[1])
        else:
            try: coords_map[line]=geocode_address_google(line)
            except Exception as e: geocode_errors.append((line,str(e)))

    home_parsed = parse_latlon_line(home_addr)
    if home_parsed: home_coord=(home_parsed[0],home_parsed[1])
    else:
        try: home_coord=geocode_address_google(home_addr)
        except Exception as e: st.error(f"Home-Geokodierung fehlgeschlagen: {e}"); st.stop()

    priority_list=[line.strip() for line in priority_input.splitlines() if line.strip()]
    priority_coords=[]
    for addr in priority_list:
        try: priority_coords.append((addr,geocode_address_google(addr)))
        except Exception as e: st.warning(f"Priorisierte Adresse konnte nicht geokodiert werden: {addr} -> {e}")

    remaining_coords_map={a:coords_map[a] for a in coords_map if a not in [p[0] for p in priority_coords]}
    start_coord_nn = priority_coords[-1][1] if priority_coords else home_coord
    sorted_remaining = nearest_neighbor_sort_by_coords(remaining_coords_map, start_coord_nn)
    final_route=[p[0] for p in priority_coords]+sorted_remaining

    st.subheader("Sortierte Adressen")
    st.text_area("Ergebnis (eine Adresse pro Zeile)", value="\n".join(final_route), height=250)
    col1,col2=st.columns(2)
    with col1: st.download_button("Download .txt", data=generate_txt(final_route), file_name="sortierte_adressen.txt", mime="text/plain")
    with col2: st.download_button("Download .csv", data=generate_csv_bytes(final_route), file_name="sortierte_adressen.csv", mime="text/csv")
    st.success("Fertig — die Liste wurde sortiert.")
