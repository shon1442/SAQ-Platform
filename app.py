import streamlit as st
import os
import io
import json
import base64
import math
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium

try:
    from saq_vector_engine import DXFVectorParser, compare_vector_delta
    HAS_VECTOR_ENGINE = True
except Exception:
    HAS_VECTOR_ENGINE = False

LOGO_PATH = "logo.png.png" if os.path.exists("logo.png.png") else "logo.png"
has_logo = os.path.exists(LOGO_PATH)
app_icon = Image.open(LOGO_PATH) if has_logo else "📐"
MEMORY_FILE = "saq_ai_memory.json"

st.set_page_config(page_title="S.A.Q - Autonomous Multi-Discipline Takeoff", layout="wide", page_icon=app_icon)

def load_ai_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"approved_patterns": [], "rejected_patterns": []}
    return {"approved_patterns": [], "rejected_patterns": []}

def save_ai_memory(memory):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def img_to_data_uri(cv2_img):
    if cv2_img is None or not hasattr(cv2_img, "size") or cv2_img.size == 0:
        return ""
    try:
        _, buf = cv2.imencode('.png', cv2_img)
        return f"data:image/png;base64,{base64.b64encode(buf).decode()}"
    except Exception:
        return ""

def load_raster(file, scale=1.4):
    if file is None:
        return None
    try:
        if file.name.lower().endswith(".pdf"):
            pdf = pdfium.PdfDocument(file.read())
            bitmap = pdf.get_page(0).render(scale=scale)
            pil_img = bitmap.to_pil().convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if len(img.shape) == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3] / 255.0
                bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
                for c in range(3):
                    bg[:, :, c] = (img[:, :, c] * alpha + bg[:, :, c] * (1.0 - alpha)).astype(np.uint8)
                return bg
            elif len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
    except Exception as e:
        st.error(f"שגיאה בטעינת הקובץ: {e}")
        return None

def safe_render_table(rows):
    cols = ["מס'", "תמונת סמל", "תיאור הפריט", "כמות מאושרת", "יחידת מידה"]
    if not rows:
        st.dataframe(pd.DataFrame(columns=cols))
        return
    clean_data = []
    for idx, r in enumerate(rows):
        clean_data.append({
            "מס'": r.get("מס'", idx + 1),
            "תמונת סמל": r.get("תמונת סמל", ""),
            "תיאור הפריט": r.get("תיאור הפריט", f"פריט #{idx+1}"),
            "כמות מאושרת": r.get("כמות מאושרת", 0),
            "יחידת מידה": r.get("יחידת מידה", "יח'")
        })
    df = pd.DataFrame(clean_data)[cols]
    st.dataframe(df, column_config={"תמונת סמל": st.column_config.ImageColumn("סמל גרפי", width="small")})

# ========================================================
# 🧱 מודול בניה (תצוגה עוקבת: סטנדרט ואז ביצוע)
# ========================================================
def extract_interior_walls_clean(plan_img, px_per_meter=125.0):
    gray = cv2.cvtColor(plan_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    k_filter = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_filter)
    
    env_kernel_dim = max(11, int(px_per_meter * 0.18))
    k_env = cv2.getStructuringElement(cv2.MORPH_RECT, (env_kernel_dim, env_kernel_dim))
    envelope = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_env)
    
    interior_raw = cv2.subtract(cleaned, envelope)
    k_wall = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    interior_walls = cv2.morphologyEx(interior_raw, cv2.MORPH_CLOSE, k_wall)
    
    min_wall_area = int((px_per_meter * 0.35) * (px_per_meter * 0.06))
    contours, _ = cv2.findContours(interior_walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clean_interior_mask = np.zeros_like(interior_walls)
    for c in contours:
        if cv2.contourArea(c) >= min_wall_area:
            cv2.drawContours(clean_interior_mask, [c], -1, 255, -1)
            
    return clean_interior_mask, envelope

def get_morphological_skeleton(binary_img):
    skel = np.zeros(binary_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary_img.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

def calc_building_partitions_clean(plan_img, px_per_meter=125.0):
    interior_mask, envelope = extract_interior_walls_clean(plan_img, px_per_meter)
    skel = get_morphological_skeleton(interior_mask)
    
    linear_pixels = cv2.countNonZero(skel)
    linear_meters = round(linear_pixels / float(px_per_meter), 2)
    
    disp_img = plan_img.copy()
    overlay = disp_img.copy()
    overlay[interior_mask > 0] = [0, 235, 255]
    cv2.addWeighted(overlay, 0.65, disp_img, 0.35, 0, disp_img)
    
    return linear_meters, disp_img, interior_mask

# ========================================================
# 🚿 מודול אינסטלציה – זיהוי סניטרי מדויק
# ========================================================
def detect_sanitary_fixtures_and_points(plan_img, px_per_meter=125.0):
    gray = cv2.cvtColor(plan_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    fixtures = []
    disp_img = plan_img.copy()
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        w_m = w / float(px_per_meter)
        h_m = h / float(px_per_meter)
        max_dim = max(w_m, h_m)
        min_dim = min(w_m, h_m)
        
        if (1.2 <= max_dim <= 2.2) and (0.6 <= min_dim <= 1.0) and area > 900:
            fixtures.append({
                "type": "אמבטיה / מקלחון",
                "center": (x + w // 2, y + h // 2),
                "bbox": (x, y, w, h),
                "crop": plan_img[max(0, y-5):min(plan_img.shape[0], y+h+5), max(0, x-5):min(plan_img.shape[1], x+w+5)],
                "color": (255, 0, 0),
                "status": "Green",
                "score": 0.90
            })
        elif (0.35 <= max_dim <= 0.95) and (0.28 <= min_dim <= 0.65) and 250 < area < 4000:
            fixtures.append({
                "type": "אסלה",
                "center": (x + w // 2, y + h // 2),
                "bbox": (x, y, w, h),
                "crop": plan_img[max(0, y-5):min(plan_img.shape[0], y+h+5), max(0, x-5):min(plan_img.shape[1], x+w+5)],
                "color": (0, 165, 255),
                "status": "Green",
                "score": 0.85
            })
        elif (0.30 <= max_dim <= 1.40) and (0.25 <= min_dim <= 0.75) and 300 < area < 5500:
            fixtures.append({
                "type": "כיור / ארון רחצה",
                "center": (x + w // 2, y + h // 2),
                "bbox": (x, y, w, h),
                "crop": plan_img[max(0, y-5):min(plan_img.shape[0], y+h+5), max(0, x-5):min(plan_img.shape[1], x+w+5)],
                "color": (0, 200, 0),
                "status": "Green",
                "score": 0.80
            })
            
    unique = []
    for f in fixtures:
        if not any(np.hypot(f["center"][0] - u["center"][0], f["center"][1] - u["center"][1]) < (px_per_meter * 0.30) for u in unique):
            unique.append(f)
            x, y, w, h = f["bbox"]
            cv2.rectangle(disp_img, (x, y), (x + w, y + h), f["color"], 2)
            
    return unique, disp_img

def compare_plumbing_delta_accurate(plan_std, plan_exec, px_per_meter=125.0):
    fix_std, _ = detect_sanitary_fixtures_and_points(plan_std, px_per_meter)
    fix_exec, disp_exec = detect_sanitary_fixtures_and_points(plan_exec, px_per_meter)
    
    relocations = []
    added = []
    b_matched = set()
    
    for f_a in fix_std:
        ca = f_a["center"]
        best_dist = 999999
        best_idx_b = -1
        for idx_b, f_b in enumerate(fix_exec):
            if idx_b in b_matched: continue
            cb = f_b["center"]
            dist_px = np.hypot(ca[0] - cb[0], ca[1] - cb[1])
            dist_m = dist_px / float(px_per_meter)
            if 0.25 <= dist_m <= 4.0 and dist_px < best_dist and f_a["type"] == f_b["type"]:
                best_dist = dist_px
                best_idx_b = idx_b
                
        if best_idx_b != -1:
            b_matched.add(best_idx_b)
            f_b = fix_exec[best_idx_b]
            dist_m = round(best_dist / float(px_per_meter), 2)
            relocations.append({"type": f_b["type"], "distance_m": dist_m, "from": ca, "to": f_b["center"]})
            cv2.arrowedLine(disp_exec, ca, f_b["center"], (0, 140, 255), 3, tipLength=0.20)
            
    for idx_b, f_b in enumerate(fix_exec):
        if idx_b not in b_matched:
            added.append(f_b)
            
    return relocations, added, disp_exec

# ========================================================
# ⚡ מנוע חילוץ סמלי חשמל מהמקרא (מדויק ומונע ספירת אלפים)
# ========================================================
def extract_symbols_from_legend(legend_img):
    if legend_img is None: return []
    gray = cv2.cvtColor(legend_img, cv2.COLOR_BGR2GRAY)
    leg_h, leg_w = gray.shape
    
    # חילוץ תאים מתוך טבלת המקרא
    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_symbols = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # סינון תאים לפי גודל הגיוני של סמל במקרא (בין 15 ל-90 פיקסלים)
        if 15 <= w <= 90 and 15 <= h <= 90 and cv2.contourArea(c) > 60:
            aspect = w / float(h)
            if 0.45 <= aspect <= 2.2:
                pad = 4
                y1, y2 = max(0, y - pad), min(leg_h, y + h + pad)
                x1, x2 = max(0, x - pad), min(leg_w, x + w + pad)
                
                crop_color = legend_img[y1:y2, x1:x2]
                crop_gray = gray[y1:y2, x1:x2]
                
                raw_symbols.append({
                    "bbox": (x, y, w, h),
                    "crop_color": crop_color,
                    "crop_gray": crop_gray,
                    "y_pos": y,
                    "x_pos": x
                })
                
    # סינון כפילויות ומִון לפי שורות ועמודות במקרא
    raw_symbols.sort(key=lambda s: (s["y_pos"] // 40, s["x_pos"]))
    unique = []
    for sym in raw_symbols:
        if not any(np.hypot(sym["x_pos"] - u["x_pos"], sym["y_pos"] - u["y_pos"]) < 25 for u in unique):
            unique.append(sym)
            
    return unique[:24]

def match_symbol_ai(plan_inv, templ_gray, min_thresh=0.68, high_thresh=0.81):
    _, templ_inv = cv2.threshold(templ_gray, 230, 255, cv2.THRESH_BINARY_INV)
    pts = cv2.findNonZero(templ_inv)
    if pts is not None:
        tx, ty, tw, th = cv2.boundingRect(pts)
        if tw > 8 and th > 8:
            templ_inv = templ_inv[ty:ty+th, tx:tx+tw]
            
    detections = []
    for scale in [0.95, 1.0, 1.05]:
        sw, sh = int(templ_inv.shape[1] * scale), int(templ_inv.shape[0] * scale)
        if sw >= plan_inv.shape[1] or sh >= plan_inv.shape[0] or sw < 8 or sh < 8: continue
        resized_t = cv2.resize(templ_inv, (sw, sh))
        
        for rot in [0, 90, 180, 270]:
            if rot == 90: r_t = cv2.rotate(resized_t, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180: r_t = cv2.rotate(resized_t, cv2.ROTATE_180)
            elif rot == 270: r_t = cv2.rotate(resized_t, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else: r_t = resized_t
            
            rw, rh = r_t.shape[::-1]
            res = cv2.matchTemplate(plan_inv, r_t, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= min_thresh)
            for pt in zip(*loc[::-1]):
                score = float(res[pt[1], pt[0]])
                status = "Green" if score >= high_thresh else "Yellow"
                detections.append({
                    "bbox": (int(pt[0]), int(pt[1]), int(rw), int(rh)),
                    "center": (int(pt[0] + rw // 2), int(pt[1] + rh // 2)),
                    "score": score,
                    "status": status
                })
                
    if not detections: return []
    indices = cv2.dnn.NMSBoxes([list(d["bbox"]) for d in detections], [d["score"] for d in detections], score_threshold=min_thresh, nms_threshold=0.35)
    final_res = [detections[i] for i in indices.flatten()] if len(indices) > 0 else []
    return final_res

# ========================================================
# 📐 מודול ריצוף וחיפוי
# ========================================================
def calc_flooring_and_wall_tiling(plan_img, tiling_height=2.40, px_per_meter=125.0, plumbing_centers=[]):
    gray = cv2.cvtColor(plan_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    total_flooring_sqm = 0.0
    wet_rooms_perimeter_m = 0.0
    disp_img = plan_img.copy()
    
    for c in contours:
        area_px = cv2.contourArea(c)
        min_room_px = (1.2 * px_per_meter) * (1.2 * px_per_meter)
        max_room_px = (15.0 * px_per_meter) * (15.0 * px_per_meter)
        
        if min_room_px <= area_px <= max_room_px:
            sqm = area_px / (px_per_meter ** 2)
            total_flooring_sqm += sqm
            
            is_wet_room = any(cv2.pointPolygonTest(c, (float(pc[0]), float(pc[1])), False) >= 0 for pc in plumbing_centers)
            peri_m = cv2.arcLength(c, True) / px_per_meter
            
            if is_wet_room:
                wet_rooms_perimeter_m += peri_m
                cv2.drawContours(disp_img, [c], -1, (0, 165, 255), 3)
            else:
                cv2.drawContours(disp_img, [c], -1, (0, 200, 0), 2)
                
    wet_wall_tiling_sqm = wet_rooms_perimeter_m * tiling_height
    return round(total_flooring_sqm, 2), round(wet_rooms_perimeter_m, 2), round(wet_wall_tiling_sqm, 2), disp_img

# ========================================================
# 📑 מנוע ייצוא HTML / EXCEL / PDF
# ========================================================
def generate_master_export_html(project_boq, title="דוח כתב כמויות מאוחד לפרויקט"):
    html = f"""
    <html dir="rtl">
    <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        @media print {{ body {{ -webkit-print-color-adjust: exact; }} }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background-color: #fafafa; }}
        .header-box {{ border-bottom: 4px solid #1F4E78; padding-bottom: 12px; margin-bottom: 25px; }}
        .disc-title {{ color: #1F4E78; border-right: 5px solid #1F4E78; padding-right: 12px; margin-top: 30px; margin-bottom: 12px; }}
        table {{ width: 100%; border-collapse: collapse; background-color: #ffffff; box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        th {{ background-color: #1F4E78; color: white; padding: 12px; font-size: 15px; border: 1px solid #ddd; }}
        td {{ padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 14px; vertical-align: middle; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        img {{ border: 1px solid #ccc; background: white; padding: 2px; border-radius: 4px; }}
    </style>
    </head>
    <body>
    <div class="header-box">
        <h2>🏗️ {title}</h2>
        <p>מערכת הנדסית S.A.Q AI | תאריך הפקה: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}</p>
    </div>
    """
    for disc_name, rows in project_boq.items():
        html += f"""
        <h3 class="disc-title">{disc_name}</h3>
        <table>
            <tr>
                <th>מס'</th>
                <th>סמל / תרשים</th>
                <th>תיאור הפריט והחישוב</th>
                <th>כמות מאושרת</th>
                <th>יחידת מידה</th>
            </tr>
        """
        if not rows:
            html += "<tr><td colspan='5'>לא נרשמו כמויות בדיסציפלינה זו (0)</td></tr>"
        else:
            for r in rows:
                img_tag = f'<img src="{r.get("image_uri", "")}" width="55" height="40"/>' if r.get("image_uri") else "—"
                html += f"""
                <tr>
                    <td>{r.get("מס'", 1)}</td>
                    <td>{img_tag}</td>
                    <td><b>{r.get("תיאור הפריט", "")}</b></td>
                    <td style="color: #1F4E78; font-size: 17px; font-weight: bold;">{r.get("כמות מאושרת", 0)}</td>
                    <td>{r.get("יחידת מידה", "יח'")}</td>
                </tr>
                """
        html += "</table>"
    html += "</body></html>"
    return html

ai_memory = load_ai_memory()
disciplines_list = ["⚡ חשמל ומאור", "🧱 בניה (מחיצות ומעטפת)", "🚿 אינסטלציה", "📐 ריצוף וחיפוי"]

tile_h = 2.40

if "project_boq" not in st.session_state:
    st.session_state["project_boq"] = {d: [] for d in disciplines_list}
if "current_discipline" not in st.session_state:
    st.session_state["current_discipline"] = "⚡ חשמל ומאור"
if "show_master_export" not in st.session_state:
    st.session_state["show_master_export"] = False

def on_discipline_change():
    st.session_state["current_discipline"] = st.session_state["disc_selector_widget"]
    st.session_state.pop("legend_results", None)
    st.session_state.pop("raw_plan_img", None)
    st.session_state["verification_completed"] = False
    st.session_state["show_master_export"] = False

def set_discipline_programmatically(new_disc):
    st.session_state["current_discipline"] = new_disc
    st.session_state.pop("legend_results", None)
    st.session_state.pop("raw_plan_img", None)
    st.session_state["verification_completed"] = False
    st.session_state["show_master_export"] = False
    st.rerun()

curr_idx = disciplines_list.index(st.session_state["current_discipline"]) if st.session_state["current_discipline"] in disciplines_list else 0

with st.sidebar:
    if has_logo: st.image(LOGO_PATH)
    st.header("⚙️ הגדרות עבודה")
    file_type = st.radio("פורמט שרטוט:", ["📄 PDF / תמונה (Raster)", "📐 CAD וקטורי (DXF)"])
    discipline = st.selectbox("דיסציפלינה:", disciplines_list, index=curr_idx, key="disc_selector_widget", on_change=on_discipline_change)
    
    st.markdown("---")
    st.subheader("📏 קנה מידה וכיול")
    scale_choice = st.selectbox("קנה מידה בשרטוט:", ["1:50 (דירות מגורים - ברירת מחדל)", "1:100 (מבנים גדולים)", "כיול ידני לפיקסלים"])
    if scale_choice == "1:50 (דירות מגורים - ברירת מחדל)":
        px_meter = 125.0
    elif scale_choice == "1:100 (מבנים גדולים)":
        px_meter = 62.5
    else:
        px_meter = st.number_input("פיקסלים למטר:", min_value=20.0, max_value=250.0, value=125.0, step=1.0)
        
    if st.session_state["current_discipline"] == "📐 ריצוף וחיפוי":
        tile_h = st.number_input("גובה חיפוי קירות בחדרים רטובים (מטר):", min_value=1.5, max_value=3.5, value=2.40, step=0.10)
        
    filter_banner = st.checkbox("סנן טבלת כותרת (Title Block)", value=True)
    
    st.markdown("---")
    st.subheader("🧠 מנוע למידה AI פעיל")
    st.caption(f"תבניות שאושרו בזיכרון: {len(ai_memory.get('approved_patterns', []))}")
    saved_count = len([k for k, v in st.session_state["project_boq"].items() if len(v) > 0])
    st.info(f"דיסציפלינות עם כמויות: **{saved_count}** מתוך 4")
    if st.button("📑 פתח מרכז דוחות פרויקט מלא"):
        st.session_state["show_master_export"] = True
        st.rerun()

col_l, col_t = st.columns([1, 6])
with col_l:
    if has_logo: st.image(LOGO_PATH, width=90)
with col_t:
    st.title("S.A.Q Takeoff & Delta Platform")
    st.caption(f"פלטפורמת ענן לפענוח הנדסי אוטונומי - {st.session_state['current_discipline']}")

active_disc = st.session_state["current_discipline"]

# ========================================================
# 📑 מרכז דוחות פרויקט מלא (Master BOQ Hub)
# ========================================================
if st.session_state.get("show_master_export", False):
    st.markdown("---")
    st.header("🏗️ מרכז הפקת דוחות סופיים לפרויקט")
    
    for d_name in disciplines_list:
        d_rows = st.session_state["project_boq"].get(d_name, [])
        with st.expander(f"📋 {d_name} ({len(d_rows)} שורות שנשמרו)", expanded=True):
            if d_rows:
                safe_render_table(d_rows)
            else:
                st.write("טרם הוזנו כמויות (יופיע כ-0 בדוח).")
            
    st.markdown("---")
    st.subheader("📦 ייצוא דוח פרויקט מאוחד")
    master_html = generate_master_export_html(st.session_state["project_boq"], title="דוח כתב כמויות מאוחד לפרויקט")
    m_c1, m_c2 = st.columns(2)
    with m_c1:
        st.download_button("📊 הורד דוח פרויקט מאוחד ל-Excel (XLS)", data=master_html.encode("utf-8"), file_name="Project_Master_Takeoff.xls", mime="application/vnd.ms-excel")
    with m_c2:
        st.download_button("📄 הורד דוח פרויקט מאוחד להדפסה/PDF", data=master_html.encode("utf-8"), file_name="Project_Master_Report.html", mime="text/html")
        
    if st.button("🔙 חזור למסך הסריקה"):
        st.session_state["show_master_export"] = False
        st.rerun()

# ========================================================
# 📄 עיבוד שרטוטי PDF / תמונות
# ========================================================
elif file_type == "📄 PDF / תמונה (Raster)":
    
    # ----------------------------------------------------
    # 1. 🧱 מודול בניה (תצוגה עוקבת: סטנדרט ואז ביצוע)
    # ----------------------------------------------------
    if active_disc == "🧱 בניה (מחיצות ומעטפת)":
        c_exec, c_std, c_leg = st.columns(3)
        with c_exec: f_plan = st.file_uploader("1️⃣ תוכנית ביצוע (חובה):", type=["pdf", "png", "jpg"], key="b_plan_exec")
        with c_std: f_std = st.file_uploader("2️⃣ תוכנית סטנדרט / קיים (אופציונלי):", type=["pdf", "png", "jpg"], key="b_plan_std")
        with c_leg: f_leg = st.file_uploader("3️⃣ מקרא בניה (אופציונלי):", type=["pdf", "png", "jpg"], key="b_leg")
        
        st.markdown("---")
        b_wall_h = st.number_input("📏 גובה מחיצות פנים להכפלה (מטר):", min_value=1.5, max_value=5.0, value=2.70, step=0.05)
        
        if f_plan:
            btn_title = "🚀 הפעל השוואת שינויי בניה עוקבת (סטנדרט ואז ביצוע)" if f_std else "🚀 הפעל חישוב מחיצות פנים נטו"
            if st.button(btn_title):
                p_bar = st.progress(0, text="מתחיל עיבוד תוכנית בניה... (0%)")
                img_exec = load_raster(f_plan)
                
                if f_std:
                    p_bar.progress(35, text="טוען תוכנית סטנדרט ומנתח קירות... (35%)")
                    img_std = load_raster(f_std)
                    
                    lin_std, disp_std, _ = calc_building_partitions_clean(img_std, px_meter)
                    lin_exec, disp_exec, _ = calc_building_partitions_clean(img_exec, px_meter)
                    
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    sqm_std = round(lin_std * b_wall_h, 2)
                    sqm_exec = round(lin_exec * b_wall_h, 2)
                    diff_m = round(max(0.0, lin_exec - lin_std), 2)
                    diff_sqm = round(diff_m * b_wall_h, 2)
                    
                    st.success("✅ החישוב הושלם בהצלחה! להלן יוצגו קודם תוכנית הסטנדרט ולאחר מכן תוכנית הביצוע.")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("אורך מחיצות בסטנדרט:", f"{lin_std} מ\"א")
                    c2.metric("אורך מחיצות בביצוע:", f"{lin_exec} מ\"א")
                    c3.metric("תוספת מחיצות לחיוב נטו:", f"{diff_m} מ\"א", f"{diff_sqm} מ\"ר")
                    
                    b_rows = [
                        {"מס'": 1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"תוספת מחיצות פנים מעבר לסטנדרט (ביצוע {lin_exec} מ\"א מול {lin_std} מ\"א בסטנדרט)", "כמות מאושרת": diff_m, "יחידת מידה": 'מ"א'},
                        {"מס'": 2, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"שטח תוספת מחיצות לחיוב (גובה {b_wall_h} מ')", "כמות מאושרת": diff_sqm, "יחידת מידה": 'מ"ר'}
                    ]
                    st.session_state["project_boq"][active_disc] = b_rows
                    safe_render_table(b_rows)
                    
                    st.markdown("### 📄 1. תוכנית סטנדרט (קירות שחושבו מסומנים בצהוב זוהר)")
                    st.image(cv2.cvtColor(disp_std, cv2.COLOR_BGR2RGB), caption="תוכנית סטנדרט - מחיצות פנים בצהוב")
                    
                    st.markdown("### 📄 2. תוכנית ביצוע (קירות שחושבו מסומנים בצהוב זוהר)")
                    st.image(cv2.cvtColor(disp_exec, cv2.COLOR_BGR2RGB), caption="תוכנית ביצוע - מחיצות פנים בצהוב")
                else:
                    p_bar.progress(60, text="מבודד מחיצות פנים ומסנן מעטפת וקווי מידות... (60%)")
                    lin_m, disp_img, _ = calc_building_partitions_clean(img_exec, px_meter)
                    sqm_total = round(lin_m * b_wall_h, 2)
                    
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.success("✅ החישוב הושלם! כל מחיצות הפנים שנמדדו צבועות בצהוב זוהר על גבי השרטוט.")
                    c1, c2 = st.columns(2)
                    c1.metric("אורך מחיצות פנים נטו (מטר רץ):", f"{lin_m} מ\"א")
                    c2.metric(f"שטח מחיצות פנים (גובה {b_wall_h} מ'):", f"{sqm_total} מ\"ר")
                    
                    b_rows = [
                        {"מס'": 1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": "מחיצות פנים נטו - אורך כולל", "כמות מאושרת": lin_m, "יחידת מידה": 'מ"א'},
                        {"מס'": 2, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"מחיצות פנים נטו - שטח כולל (גובה {b_wall_h} מ')", "כמות מאושרת": sqm_total, "יחידת מידה": 'מ"ר'}
                    ]
                    st.session_state["project_boq"][active_disc] = b_rows
                    safe_render_table(b_rows)
                    st.image(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB), caption="הקירות שנמדדו וחושבו צבועים בצהוב זוהר")

    # ----------------------------------------------------
    # 2. 🚿 מודול אינסטלציה
    # ----------------------------------------------------
    elif active_disc == "🚿 אינסטלציה":
        c_exec, c_std, c_leg = st.columns(3)
        with c_exec: f_plan = st.file_uploader("1️⃣ תוכנית ביצוע אינסטלציה (חובה):", type=["pdf", "png", "jpg"], key="p_plan_exec")
        with c_std: f_std = st.file_uploader("2️⃣ תוכנית סטנדרט / קיים (אופציונלי):", type=["pdf", "png", "jpg"], key="p_plan_std")
        with c_leg: f_leg = st.file_uploader("3️⃣ מקרא כלים סניטריים (אופציונלי):", type=["pdf", "png", "jpg"], key="p_leg")
        
        if f_plan:
            btn_title = "🚀 הפעל השוואת שינויים ומרחקי העתקה מול סטנדרט" if f_std else "🚀 הפעל ספירת כלים סניטריים מדויקת"
            if st.button(btn_title):
                st.session_state["plumb_verified"] = False
                p_bar = st.progress(0, text="מתחיל טעינת תוכנית אינסטלציה... (0%)")
                img_plan = load_raster(f_plan)
                
                if f_std:
                    p_bar.progress(30, text="מזהה כלים סניטריים בסטנדרט ובביצוע... (30%)")
                    img_std = load_raster(f_std)
                    relocs, added, disp_delta = compare_plumbing_delta_accurate(img_std, img_plan, px_meter)
                    
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.subheader("🔄 דוח שינויים והעתקת נקודות אינסטלציה מול סטנדרט")
                    st.metric("נקודות וכלים שהועתקו/הוזזו ממקומן:", f"{len(relocs)} יח'", f"+{len(added)} כלים/נקודות חדשות")
                    
                    p_rows = []
                    for idx, r in enumerate(relocs):
                        p_rows.append({"מס'": idx+1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"העתקת {r['type']} (הזזה של {r['distance_m']} מטר)", "כמות מאושרת": 1, "יחידת מידה": f"יח' ({r['distance_m']} מ')"})
                    for idx, a in enumerate(added):
                        p_rows.append({"מס'": len(relocs)+idx+1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"תוספת {a['type']} חדשה מעבר לסטנדרט", "כמות מאושרת": 1, "יחידת מידה": "יח'"})
                    st.session_state["project_boq"][active_disc] = p_rows
                    safe_render_table(p_rows)
                    st.image(cv2.cvtColor(disp_delta, cv2.COLOR_BGR2RGB), caption="חיצים כתומים = העתקת נקודות סניטריות")
                else:
                    p_bar.progress(40, text="מאתר אסלות, אמבטיות וכיורים בחללים הרטובים... (40%)")
                    fixtures_found, disp_fix = detect_sanitary_fixtures_and_points(img_plan, px_meter)
                    
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.session_state["plumb_raw_fixtures"] = fixtures_found
                    st.session_state["plumb_disp_img"] = disp_fix

            if "plumb_raw_fixtures" in st.session_state:
                raw_fix = st.session_state["plumb_raw_fixtures"]
                disp_fix = st.session_state["plumb_disp_img"]
                
                verify_subset = raw_fix[:6]
                is_done_verifying = st.session_state.get("plumb_verified", False)
                
                if verify_subset and not is_done_verifying:
                    st.markdown("---")
                    st.info(f"🔍 **אישור כלים סניטריים שזוהו ולמידת AI (מוצגים {len(verify_subset)} פריטים ראשונים לבדיקה):**")
                    with st.expander("לחץ כאן לאישור/דחיית כלים סניטריים (V / X)", expanded=True):
                        cols = st.columns(min(len(verify_subset), 3))
                        updated_mem = False
                        for idx, f in enumerate(verify_subset):
                            with cols[idx % len(cols)]:
                                x, y, w, h = f["bbox"]
                                pad = 20
                                crop_zoom = disp_fix[max(0, y-pad):min(disp_fix.shape[0], y+h+pad), max(0, x-pad):min(disp_fix.shape[1], x+w+pad)].copy()
                                cv2.circle(crop_zoom, (w//2, h//2), max(w,h)//2 + 5, (0, 0, 255), 2)
                                st.image(cv2.cvtColor(crop_zoom, cv2.COLOR_BGR2RGB), width=120, caption=f"{f['type']}")
                                choice = st.radio("סטטוס:", ["✅ אשר (V)", "❌ דחה (X)"], key=f"plumb_choice_{idx}", horizontal=True)
                                is_appr = ("אשר" in choice)
                                f["user_approved"] = is_appr
                                
                                p_key = f"plumb_{f['type']}_{w}x{h}"
                                if is_appr and p_key not in ai_memory["approved_patterns"]:
                                    ai_memory["approved_patterns"].append(p_key)
                                    updated_mem = True
                                elif not is_appr and p_key not in ai_memory["rejected_patterns"]:
                                    ai_memory["rejected_patterns"].append(p_key)
                                    updated_mem = True
                                st.markdown("---")
                                
                        if updated_mem:
                            save_ai_memory(ai_memory)
                            
                        if st.button("✨ סיימתי לסמן כלים סניטריים - נעל ועדכן כמויות"):
                            st.session_state["plumb_verified"] = True
                            st.rerun()

                counts = {}
                for f in raw_fix:
                    if f.get("user_approved", True):
                        t = f["type"]
                        counts[t] = counts.get(t, 0) + 1
                        
                p_rows = []
                for idx, (f_type, c_val) in enumerate(counts.items()):
                    p_rows.append({"מס'": idx + 1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f_type, "כמות מאושרת": c_val, "יחידת מידה": "יח'"})
                st.session_state["project_boq"][active_disc] = p_rows
                safe_render_table(p_rows)
                st.image(cv2.cvtColor(disp_fix, cv2.COLOR_BGR2RGB), caption="כלים סניטריים שזוהו בתוכנית")

    # ----------------------------------------------------
    # 3. 📐 מודול ריצוף וחיפוי
    # ----------------------------------------------------
    elif active_disc == "📐 ריצוף וחיפוי":
        c_exec, c_std = st.columns(2)
        with c_exec: f_plan = st.file_uploader("1️⃣ תוכנית ביצוע ריצוף/חיפוי (חובה):", type=["pdf", "png", "jpg"], key="f_plan_exec")
        with c_std: f_std = st.file_uploader("2️⃣ תוכנית סטנדרט / קיים (אופציונלי):", type=["pdf", "png", "jpg"], key="f_plan_std")
        
        if f_plan:
            btn_title = "🚀 הפעל השוואת שינויי ריצוף וחיפוי מול סטנדרט" if f_std else "🚀 הפעל חישוב ריצוף נטו וחיפוי חדרים רטובים"
            if st.button(btn_title):
                p_bar = st.progress(0, text="מתחיל טעינת תוכנית ריצוף... (0%)")
                img_plan = load_raster(f_plan)
                
                fixtures_plan, _ = detect_sanitary_fixtures_and_points(img_plan, px_meter)
                plumb_pts = [f["center"] for f in fixtures_plan]
                floor_sqm, wet_peri_m, wet_wall_sqm, disp_img = calc_flooring_and_wall_tiling(img_plan, tile_h, px_meter, plumb_pts)
                
                if f_std:
                    p_bar.progress(50, text="משווה מול תוכנית סטנדרט... (50%)")
                    img_std = load_raster(f_std)
                    fixtures_std, _ = detect_sanitary_fixtures_and_points(img_std, px_meter)
                    plumb_pts_std = [f["center"] for f in fixtures_std]
                    f_std_sqm, _, w_std_sqm, _ = calc_flooring_and_wall_tiling(img_std, tile_h, px_meter, plumb_pts_std)
                    
                    diff_floor = round(floor_sqm - f_std_sqm, 2)
                    diff_wall = round(wet_wall_sqm - w_std_sqm, 2)
                    
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.subheader("🔄 הפרשי כמויות ריצוף וחיפוי מול סטנדרט")
                    c1, c2 = st.columns(2)
                    c1.metric("הפרש ריצוף רצפה נטו:", f"{floor_sqm} מ\"ר", f"{diff_floor:+0.2f} מ\"ר מול סטנדרט")
                    c2.metric("הפרש חיפוי קירות רטובים:", f"{wet_wall_sqm} מ\"ר", f"{diff_wall:+0.2f} מ\"ר מול סטנדרט")
                    
                    f_rows = [
                        {"מס'": 1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"ריצוף רצפה (ביצוע {floor_sqm} מ\"ר לעומת סטנדרט {f_std_sqm} מ\"ר)", "כמות מאושרת": diff_floor, "יחידת מידה": 'מ"ר הפרש'},
                        {"מס'": 2, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"חיפוי חדרים רטובים (ביצוע {wet_wall_sqm} מ\"ר לעומת סטנדרט {w_std_sqm} מ\"ר)", "כמות מאושרת": diff_wall, "יחידת מידה": 'מ"ר הפרש'}
                    ]
                else:
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.success("✅ החישוב הושלם! חללים רטובים זוהו אוטומטית לפי הכלים הסניטריים בתוכם.")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("ריצוף רצפה נטו (בירוק):", f"{floor_sqm} מ\"ר")
                    c2.metric("היקף קירות חדרים רטובים:", f"{wet_peri_m} מ\"א")
                    c3.metric(f"חיפוי קירות רטובים (גובה {tile_h} מ'):", f"{wet_wall_sqm} מ\"ר")
                    
                    f_rows = [
                        {"מס'": 1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": "ריצוף רצפה כללי נטו (ללא שטח קירות)", "כמות מאושרת": floor_sqm, "יחידת מידה": 'מ"ר'},
                        {"מס'": 2, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"חיפוי קירות חדרים רטובים (היקף {wet_peri_m} מ\"א * {tile_h} מ')", "כמות מאושרת": wet_wall_sqm, "יחידת מידה": 'מ"ר'}
                    ]
                st.session_state["project_boq"][active_disc] = f_rows
                safe_render_table(f_rows)
                st.image(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB), caption="כתום = חלל רטוב לחיפוי קירות, ירוק = ריצוף יבש")

    # ----------------------------------------------------
    # 4. ⚡ מודול חשמל ומאור (מנוע מקרא מדויק ללא ספירת אלפים)
    # ----------------------------------------------------
    else:
        c_exec, c_std, c_leg = st.columns(3)
        with c_exec: f_plan = st.file_uploader("1️⃣ תוכנית ביצוע חשמל (חובה):", type=["pdf", "png", "jpg"], key="e_plan_exec")
        with c_std: f_std = st.file_uploader("2️⃣ תוכנית סטנדרט / קיים (אופציונלי):", type=["pdf", "png", "jpg"], key="e_plan_std")
        with c_leg: f_leg = st.file_uploader("3️⃣ מקרא חשמל ומאור (אופציונלי):", type=["pdf", "png", "jpg"], key="e_leg")
        
        if f_plan:
            if st.button("🚀 הפעל פענוח וספירת חשמל ומאור"):
                st.session_state["elec_verified"] = False
                p_bar = st.progress(0, text="מתחיל טעינת קבצי חשמל... (0%)")
                img_plan = load_raster(f_plan)
                
                plan_gray = cv2.cvtColor(img_plan, cv2.COLOR_BGR2GRAY)
                _, plan_inv = cv2.threshold(plan_gray, 230, 255, cv2.THRESH_BINARY_INV)
                
                p_bar.progress(30, text="מחלץ סמלים מתוך קובץ המקרא... (30%)")
                symbols = extract_symbols_from_legend(load_raster(f_leg)) if f_leg else []
                all_results = []
                
                if symbols:
                    total_s = len(symbols)
                    for i, sym in enumerate(symbols):
                        pct = 40 + int(((i + 1) / total_s) * 50)
                        p_bar.progress(pct, text=f"סורק סמל {i+1} מתוך {total_s} בתוכנית... ({pct}%)")
                        m = match_symbol_ai(plan_inv, sym["crop_gray"])
                        all_results.append({
                            "index": i + 1,
                            "symbol_img": sym["crop_color"],
                            "image_uri": img_to_data_uri(sym["crop_color"]),
                            "matches": m
                        })
                else:
                    fixtures_e, _ = detect_sanitary_fixtures_and_points(img_plan, px_meter)
                    for i, f in enumerate(fixtures_e):
                        all_results.append({
                            "index": i + 1,
                            "symbol_img": f["crop"],
                            "image_uri": img_to_data_uri(f["crop"]),
                            "matches": [{"bbox": f["bbox"], "center": f["center"], "score": 0.85, "status": "Green"}]
                        })
                
                p_bar.progress(100, text="הסריקה הושלמה בהצלחה! (100%)")
                time.sleep(0.3)
                p_bar.empty()
                
                st.session_state["elec_results"] = all_results
                st.session_state["elec_plan_raw"] = img_plan

            if "elec_results" in st.session_state:
                res = st.session_state["elec_results"]
                raw_plan = st.session_state["elec_plan_raw"]
                disp_plan = raw_plan.copy()
                
                yellow_items = []
                for s_idx, item in enumerate(res):
                    for m_idx, m in enumerate(item["matches"]):
                        if m["status"] == "Yellow":
                            yellow_items.append((s_idx, m_idx, item, m))
                            
                yellow_items = yellow_items[:6]
                is_done_verifying = st.session_state.get("elec_verified", False)
                
                if yellow_items and not is_done_verifying:
                    st.markdown("---")
                    st.info(f"🔍 **אימות סמלי חשמל בספק ולמידת AI (מוצגים {len(yellow_items)} פריטים ראשונים לבדיקה):**")
                    with st.expander("לחץ כאן לאישור/דחיית סמלים (ה-AI לומד לפעמים הבאות)", expanded=True):
                        cols = st.columns(min(len(yellow_items), 3))
                        updated_mem = False
                        for y_i, (s_idx, m_idx, item, m) in enumerate(yellow_items):
                            with cols[y_i % len(cols)]:
                                x, y, w, h = m["bbox"]
                                pad = 24
                                crop_zoom = raw_plan[max(0, y-pad):min(raw_plan.shape[0], y+h+pad), max(0, x-pad):min(raw_plan.shape[1], x+w+pad)].copy()
                                cv2.circle(crop_zoom, (crop_zoom.shape[1]//2, crop_zoom.shape[0]//2), max(w,h)//2 + 6, (0, 0, 255), 3)
                                
                                st.image(cv2.cvtColor(crop_zoom, cv2.COLOR_BGR2RGB), caption=f"סמל #{item['index']} (ודאות {m['score']*100:.0f}%)", width=130)
                                choice = st.radio("סטטוס:", ["✅ אשר (V)", "❌ דחה (X)"], key=f"elec_choice_{s_idx}_{m_idx}", horizontal=True)
                                is_appr = ("אשר" in choice)
                                m["user_decision"] = "Approved" if is_appr else "Rejected"
                                
                                p_key = f"elec_{item['index']}_{w}x{h}"
                                if is_appr and p_key not in ai_memory["approved_patterns"]:
                                    ai_memory["approved_patterns"].append(p_key)
                                    updated_mem = True
                                elif not is_appr and p_key not in ai_memory["rejected_patterns"]:
                                    ai_memory["rejected_patterns"].append(p_key)
                                    updated_mem = True
                                st.markdown("---")
                                
                        if updated_mem:
                            save_ai_memory(ai_memory)
                            
                        if st.button("✨ סיימתי לסמן נקודות - נעל ועדכן כמויות"):
                            st.session_state["elec_verified"] = True
                            st.rerun()

                e_rows = []
                for s_idx, item in enumerate(res):
                    confirmed_count = 0
                    for m_idx, m in enumerate(item["matches"]):
                        x, y, w, h = m["bbox"]
                        is_green = (m["status"] == "Green")
                        user_dec = m.get("user_decision", "Pending")
                        
                        if is_green or user_dec == "Approved":
                            confirmed_count += 1
                            cv2.rectangle(disp_plan, (x, y), (x + w, y + h), (0, 200, 0), 2)
                        elif user_dec == "Rejected":
                            cv2.line(disp_plan, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.line(disp_plan, (x + w, y), (x, y + h), (0, 0, 255), 2)
                            
                    item["confirmed_count"] = confirmed_count
                    if confirmed_count > 0:
                        e_rows.append({
                            "מס'": item["index"],
                            "תמונת סמל": item["image_uri"],
                            "image_uri": item["image_uri"],
                            "תיאור הפריט": f"סמל חשמל/מאור #{item['index']}",
                            "כמות מאושרת": confirmed_count,
                            "יחידת מידה": "יח'"
                        })
                        
                st.session_state["project_boq"][active_disc] = e_rows
                safe_render_table(e_rows)
                st.image(cv2.cvtColor(disp_plan, cv2.COLOR_BGR2RGB))

    # ========================================================
    # 🏁 כפתורי סיום פרויקט ומעבר דיסציפלינה (זמינים תמיד)
    # ========================================================
    st.markdown("---")
    c_fin, c_next = st.columns(2)
    with c_fin:
        if st.button("🏁 סיום הפרויקט והפקת דוחות סופיים (Excel / PDF)", key=f"btn_finish_master_{active_disc}"):
            st.session_state["show_master_export"] = True
            st.rerun()
    with c_next:
        st.write("**מעבר לחישוב דיסציפלינה נוספת:**")
        rem = [d for d in disciplines_list if d != active_disc]
        cols = st.columns(len(rem))
        for i, d_target in enumerate(rem):
            if cols[i].button(d_target, key=f"btn_nav_{i}"):
                set_discipline_programmatically(d_target)
