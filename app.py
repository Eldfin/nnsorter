# app.py
# Streamlit NN-Sortierer mit client-seitigem Kamera-Scanner (Tesseract.js)
# - bevorzugt Rückkamera
# - kontinuierliches Scannen
# - automatische Einfügung, wenn eine akzeptable Adresse erkannt wurde (auch mehrzeilig)

import streamlit as st
import os
import io
import csv
import re
from typing import List, Tuple
from math import radians, sin, cos, atan2, sqrt
import time

# Geopy für Geocoding (verwendet deinen GOOGLE_API_KEY falls gesetzt)
try:
    from geopy.geocoders import GoogleV3
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

st.set_page_config(page_title="NN-Sortierer mit Live-Kamera-OCR", layout="centered")
st.title("Nearest-Neighbor Sortierer — Live Kamera-OCR (Rückkamera bevorzugt)")

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

# ----------------- Verbesserte Heuristik für mehrzeilige Adressen -----------------
def extract_address_from_text_block(text: str) -> str:
    """
    Versucht aus einem (mehrzeiligen) OCR-Textblock eine plausible Adresse zu extrahieren.
    - sucht nach PLZ+Ort (5-stellige PLZ)
    - wenn gefunden: kombiniert die vorhergehende(n) Zeile(n) als Straße, wenn sinnvoll
    - fallback: sucht nach Straße + Hausnummer in beliebiger Zeile
    - sonst: gibt die ersten 3 Linien zusammen als Kandidat zurück
    """
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    joined = "\n".join(lines)

    # Suche PLZ (DE) + Ort
    m = re.search(r"\b(\d{5})\s+([A-Za-zÄÖÜäöüß\-\. ]{2,})\b", joined)
    if m:
        # finde Index der Zeile die die PLZ enthält
        for i, ln in enumerate(lines):
            if m.group(1) in ln:
                # nimm vorherige Zeile(n) für Straße, falls diese Hausnummer enthält
                street = lines[i-1] if i-1 >= 0 else None
                if street and re.search(r"\d", street):
                    return f"{street}, {lines[i]}"
                # evtl. zwei vorherige Zeilen (Firma + Straße)
                if i-2 >= 0 and re.search(r"\d", lines[i-2]):
                    return f"{lines[i-2]}, {lines[i-1]}, {lines[i]}"
                return lines[i]  # nur PLZ/Ort-Zeile
    # Suche nach Straße + Hausnummer in einer Zeile
    for ln in lines:
        if re.search(r"\b\S+\s+\d+[a-zA-Z]?(?:\s*[-/]\s*\d+)?\b", ln):
            return ln
    # fallback: kombiniere die ersten bis zu 3 Zeilen
    return ", ".join(lines[:3])

# ----------------- Session state -----------------
if 'scanned_text' not in st.session_state:
    st.session_state['scanned_text'] = ""
if 'last_component_value' not in st.session_state:
    st.session_state['last_component_value'] = None

# ----------------- UI: Kamera-Component (HTML+Tesseract.js) -----------------
st.markdown("### Live-Kamera OCR (Rückkamera bevorzugt)")
st.markdown(
    "Diese Komponente verwendet Tesseract.js im Browser. "
    "Erkanntes Ergebnis wird automatisch an die App gesendet, sobald eine plausible Adresse gefunden wurde."
)
st.markdown("**Hinweis:** Browser muss HTTPS unterstützen für Kamerazugriff (oder `localhost`).")

# Hier bauen wir ein HTML-Widget, das die Kamera anspricht, Tesseract.js lädt,
# kontinuierlich Frames verarbeitet und bei erkannter Adresse das Ergebnis an Streamlit zurückmeldet.
# Wir benutzen window.parent.postMessage mit streamlit:setComponentValue, damit Streamlit das Ergebnis als Rückgabewert erhält.
html_code = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Live OCR</title>
  </head>
  <body>
    <div>
      <video id="video" autoplay playsinline style="width:100%;max-width:560px;border:1px solid #ddd"></video>
      <div style="margin-top:8px;">
        <button id="startBtn">Start</button>
        <button id="stopBtn">Stop</button>
        <span id="status"></span>
      </div>
      <div id="last" style="margin-top:8px;color:green;font-weight:bold;"></div>
    </div>
    <canvas id="canvas" style="display:none;"></canvas>

    <!-- Tesseract.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/tesseract.js@4.1.1/dist/tesseract.min.js"></script>

    <script>
      const statusEl = document.getElementById('status');
      const lastEl = document.getElementById('last');
      let stream = null;
      let video = document.getElementById('video');
      let canvas = document.getElementById('canvas');
      let scanning = false;
      let worker = null;
      let minConfidence = 0.6; // optionaler Schwellenwert (nicht strikt)
      let lastSent = null;

      async function startWorker() {{
        statusEl.textContent = 'Lade OCR-Worker...';
        worker = Tesseract.createWorker({{
          logger: m => {{
            // du kannst m.status/m.progress zur Anzeige nutzen
            // console.log(m);
          }}
        }});
        await worker.load();
        try {{
          await worker.loadLanguage('deu');
          await worker.initialize('deu');
        }} catch (e) {{
          // fallback Englisch falls kein 'deu' verfügbar
          await worker.loadLanguage('eng');
          await worker.initialize('eng');
        }}
        statusEl.textContent = 'OCR bereit';
      }}

      async function startCamera() {{
        statusEl.textContent = 'Kamera startet...';
        // versuche explizit Rückkamera
        const constraints = {{ video: {{ facingMode: {{ ideal: "environment" }} }}, audio: false }};
        try {{
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        }} catch (e) {{
          // fallback: any camera
          try {{
            stream = await navigator.mediaDevices.getUserMedia({{ video: true, audio: false }});
          }} catch (err) {{
            statusEl.textContent = 'Kamera nicht verfügbar: ' + err;
            return;
          }}
        }}
        video.srcObject = stream;
        await video.play();
        statusEl.textContent = 'Kamera läuft';
      }}

      function stopCamera() {{
        scanning = false;
        if (stream) {{
          stream.getTracks().forEach(t => t.stop());
          stream = null;
        }}
        if (video) {{
          video.pause();
          video.srcObject = null;
        }}
        statusEl.textContent = 'Gestoppt';
      }}

      async function scanLoop() {{
        if (!worker) {{
          await startWorker();
        }}
        scanning = true;
        statusEl.textContent = 'Scannen...';
        while (scanning) {{
          // capture frame
          const w = video.videoWidth;
          const h = video.videoHeight;
          if (!w || !h) {{
            await new Promise(r => setTimeout(r, 300));
            continue;
          }}
          // Zeichne auf Canvas (ggf. nur mittleren Bereich croppen um Performance zu erhöhen)
          const ctx = canvas.getContext('2d');
          // wir nehmen mittleren Bereich mit 0.9*Breite und 0.5*Höhe (adressen sind oft in der Mitte)
          const cropW = Math.floor(w * 0.9);
          const cropH = Math.floor(h * 0.5);
          const sx = Math.floor((w - cropW) / 2);
          const sy = Math.floor((h - cropH) / 2);
          canvas.width = cropW;
          canvas.height = cropH;
          ctx.drawImage(video, sx, sy, cropW, cropH, 0, 0, cropW, cropH);

          // verkleinere für Performance falls sehr groß
          const maxDim = 1000;
          let targetW = canvas.width;
          let targetH = canvas.height;
          if (Math.max(targetW, targetH) > maxDim) {{
            const scale = maxDim / Math.max(targetW, targetH);
            targetW = Math.round(targetW * scale);
            targetH = Math.round(targetH * scale);
          }}
          // toDataURL
          const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
          const blob = await (await fetch(dataUrl)).blob();

          // OCR (async)
          try {{
            const {{" data ": result }} = await worker.recognize(blob);
            const text = result ? result.text : '';
            if (text && text.trim().length>5) {{
              // heuristik: suche PLZ oder Straße+Nr
              const txt = text.replace(/\\r/g,'\\n');
              const plzRe = /\\b\\d{{5}}\\b/;
              const streetRe = /\\b\\S+\\s+\\d+[a-zA-Z]?\\b/;
              if (plzRe.test(txt) || streetRe.test(txt)) {{
                // optional: kombiniere mehrere Zeilen, trimme
                let lines = txt.split(/\\n+/).map(s=>s.trim()).filter(Boolean);
                let candidate = '';
                // priorisiere PLZ+Ort Zeile plus vorherige Zeile (Straße)
                for (let i=0;i<lines.length;i++) {{
                  if (plzRe.test(lines[i])) {{
                    let street = i>0 ? lines[i-1] : '';
                    if (street && /\\d/.test(street)) candidate = street + ', ' + lines[i];
                    else candidate = lines[i];
                    break;
                  }}
                }}
                if (!candidate) {{
                  // fallback: erste Zeile mit Straße+Nr
                  for (let ln of lines) {{
                    if (streetRe.test(ln)) {{ candidate = ln; break; }}
                  }}
                }}
                if (!candidate) candidate = lines.slice(0,3).join(', ');

                // vermeide mehrfaches Senden desselben Strings
                const now = Date.now();
                if (candidate && candidate !== lastSent) {{
                  lastSent = candidate;
                  lastEl.textContent = 'Erkannt: ' + candidate;
                  statusEl.textContent = 'Adresse gefunden — sende an App';
                  // send value back to Streamlit
                  const message = {{
                    isStreamlitMessage: true,
                    type: 'streamlit:setComponentValue',
                    value: candidate
                  }};
                  window.parent.postMessage(message, '*');
                  // pause kurz, damit Nutzer Zeit hat zu sehen was erkannt wurde
                  await new Promise(r => setTimeout(r, 1500));
                }}
              }}
            }}
          }} catch (ocrErr) {{
            console.error('OCR Error', ocrErr);
          }}
          // Warte zwischen Scans (anpassen für Geschwindigkeit/Performance)
          await new Promise(r => setTimeout(r, 1200));
        }}
      }}

      document.getElementById('startBtn').addEventListener('click', async () => {{
        await startCamera();
        scanLoop();
      }});
      document.getElementById('stopBtn').addEventListener('click', () => {{
        stopCamera();
      }});

      // Automatisch starten wenn erlaubt:
      (async () => {{
        try {{
          // versuche automatisch zu starten (wird evtl. Browser-Popup auslösen)
          await startCamera();
          // nicht sofort OCR starten, warte auf Benutzer-Klick für UX (optional)
        }} catch(e) {{
          // ignore
        }}
      }})();
    </script>
  </body>
</html>
"""

# Rendere das HTML-Widget und empfange Rückgabewert (die erkannte Adresse -> component_value)
import streamlit.components.v1 as components
component_value = components.html(html_code, height=600, scrolling=True)

# components.html gibt den letzten streamlit:setComponentValue zurück — falls eine Adresse geschickt wurde,
# wird sie in component_value geliefert. Wir speichern in session_state und fügen ins Textfeld ein.
if component_value:
    # merge in scanned_text falls noch nicht enthalten
    prev = st.session_state.get('scanned_text', '').strip()
    cand = component_value.strip()
    if cand and cand not in prev.splitlines():
        if prev:
            st.session_state['scanned_text'] = prev + "\n" + cand
        else:
            st.session_state['scanned_text'] = cand
        st.success(f"Adresse übernommen: {cand}")
    st.session_state['last_component_value'] = component_value

# ----------------- Restliche UI (Adressen, Sortierung) -----------------
home_addr = st.text_input("Home-Adresse (Startpunkt)", value="Rudolf-Diesel-Straße 2, 35463 Fernwald, Germany")

# Haupt-Textfeld vorbefüllen mit gescannten Adressen
text_input = st.text_area("Adressen (eine pro Zeile)", height=200, value=st.session_state.get('scanned_text', ''), placeholder="Adresse 1\nAdresse 2\n...")
priority_input = st.text_area("Priorisierte Adressen (optional, werden zuerst besucht)", height=100, placeholder="Adresse A\nAdresse B\n...")

if st.button("Sortieren"):
    addresses = parse_addresses_from_text(text_input)
    if not addresses:
        st.warning("Keine Adressen eingegeben.")
        st.stop()

    if not GEOPY_AVAILABLE:
        st.error("geopy nicht installiert. pip install geopy")
        st.stop()

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

        remaining_coords_map = {a: coords_map[a] for a in coords_map if a not in [p[0] for p in priority_coords]}

        if priority_coords:
            start_coord_nn = priority_coords[-1][1]
        else:
            start_coord_nn = home_coord

        sorted_remaining = nearest_neighbor_sort_by_coords(remaining_coords_map, start_coord_nn)
        final_route = [p[0] for p in priority_coords] + sorted_remaining

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
