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
from PIL import Image
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
    if cv2_img is None or cv2_img.size == 0:
        return ""
    _, buf = cv2.imencode('.png', cv2_img)
    return f"data:image/png;base64,{base64.b64encode(buf).decode()}"

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

# ========================================================
# 🛡️ מנוע אימות תאימות בין תוכניות (Plan Compatibility Check)
# ========================================================
def check_plans_compatibility(img_a, img_b):
    """בדיקה אוטומטית שהשרטוטים שייכים לאותו פרויקט/חלל ולא מדיסציפלינות סותרות"""
    if img_a is None or img_b is None:
        return True, 100.0
    
    ref_w, ref_h = 500, 500
    gray_a = cv2.cvtColor(cv2.resize(img_a, (ref_w, ref_h)), cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(cv2.resize(img_b, (ref_w, ref_h)), cv2.COLOR_BGR2GRAY)
    
    edges_a = cv2.Canny(gray_a, 50, 150)
    edges_b = cv2.Canny(gray_b, 50, 150)
    
    orb = cv2.ORB_create(nfeatures=400)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    
    struct_corr = cv2.matchTemplate(edges_a, edges_b, cv2.TM_CCOEFF_NORMED)[0][0]
    
    if des_a is None or des_b is None or len(kp_a) < 12 or len(kp_b) < 12:
        is_compat = (struct_corr >= 0.18)
        return is_compat, round(max(0.0, float(struct_corr)) * 100, 1)
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    good_matches = [m for m in matches if m.distance < 60]
    match_ratio = len(good_matches) / float(min(len(kp_a), len(kp_b)))
    
    is_compat = (match_ratio >= 0.12) or (struct_corr >= 0.20)
    score = round(max(match_ratio, struct_corr) * 100, 1)
    return is_compat, score

# ========================================================
# ⚡ מנועי פענוח סמלים (חשמל / אינסטלציה / בניה)
# ========================================================
def extract_symbols_from_legend(legend_img):
    if legend_img is None:
        return []
    gray = cv2.cvtColor(legend_img, cv2.COLOR_BGR2GRAY)
    leg_h = gray.shape[0]
    crop_h = int(leg_h * 0.88)
    work_gray = gray[:crop_h, :]
    work_color = legend_img[:crop_h, :]
    
    _, thresh = cv2.threshold(work_gray, 225, 255, cv2.THRESH_BINARY_INV)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    cleaned = cv2.subtract(thresh, cv2.add(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel), cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)))
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_symbols = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if 14 <= w <= 110 and 14 <= h <= 110 and area > 40:
            if 0.40 <= (w / float(h)) <= 2.4:
                pad = 4
                y1, y2 = max(0, y - pad), min(work_gray.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(work_gray.shape[1], x + w + pad)
                raw_symbols.append({
                    "bbox": (x, y, w, h),
                    "crop_color": work_color[y1:y2, x1:x2],
                    "crop_gray": work_gray[y1:y2, x1:x2],
                    "y_pos": y, "x_pos": x
                })
    raw_symbols.sort(key=lambda s: (s["y_pos"] // 35, s["x_pos"]))
    unique = []
    for sym in raw_symbols:
        if not any(np.hypot(sym["x_pos"] - u["x_pos"], sym["y_pos"] - u["y_pos"]) < 26 for u in unique):
            unique.append(sym)
    return unique[:16]

def auto_discover_plan_symbols(plan_roi, min_dim=15, max_dim=90, match_thresh=0.68):
    gray = cv2.cvtColor(plan_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_dim <= w <= max_dim and min_dim <= h <= max_dim and cv2.contourArea(c) > 40:
            pad = 3
            y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
            candidates.append({"bbox": (x, y, w, h), "crop_gray": gray[y1:y2, x1:x2], "crop_color": plan_roi[y1:y2, x1:x2], "center": (x + w // 2, y + h // 2)})
    clusters = []
    for cand in candidates:
        matched = False
        for cl in clusters:
            rep = cl["rep_gray"]
            if abs(cand["crop_gray"].shape[0] - rep.shape[0]) <= 10 and abs(cand["crop_gray"].shape[1] - rep.shape[1]) <= 10:
                resized = cv2.resize(cand["crop_gray"], (rep.shape[1], rep.shape[0]))
                if cv2.matchTemplate(resized, rep, cv2.TM_CCOEFF_NORMED)[0][0] >= match_thresh:
                    cl["items"].append(cand)
                    matched = True
                    break
        if not matched:
            clusters.append({"rep_gray": cand["crop_gray"], "rep_color": cand["crop_color"], "items": [cand]})
    valid = [cl for cl in clusters if len(cl["items"]) >= 2]
    valid.sort(key=lambda x: len(x["items"]), reverse=True)
    return valid[:14]

def match_symbol_ai(plan_inv, templ_gray, min_thresh=0.62, high_thresh=0.74):
    _, templ_inv = cv2.threshold(templ_gray, 230, 255, cv2.THRESH_BINARY_INV)
    pts = cv2.findNonZero(templ_inv)
    if pts is not None:
        tx, ty, tw, th = cv2.boundingRect(pts)
        if tw > 8 and th > 8:
            templ_inv = templ_inv[ty:ty+th, tx:tx+tw]
    detections = []
    for scale in [0.82, 0.90, 1.0, 1.10, 1.20]:
        sw, sh = int(templ_inv.shape[1] * scale), int(templ_inv.shape[0] * scale)
        if sw >= plan_inv.shape[1] or sh >= plan_inv.shape[0] or sw < 8 or sh < 8:
            continue
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
                detections.append({"bbox": (int(pt[0]), int(pt[1]), int(rw), int(rh)), "center": (int(pt[0] + rw // 2), int(pt[1] + rh // 2)), "score": score, "status": "Green" if score >= high_thresh else "Yellow"})
    if not detections:
        return []
    indices = cv2.dnn.NMSBoxes([list(d["bbox"]) for d in detections], [d["score"] for d in detections], score_threshold=min_thresh, nms_threshold=0.20)
    final_res = [detections[i] for i in indices.flatten()] if len(indices) > 0 else []
    return [r for r in final_res if r["status"] == "Green"] if len(final_res) > 70 else final_res

# ========================================================
# 🧱 מודול בניה (מחיצות ומעטפת)
# ========================================================
def calc_building_partitions(plan_img, wall_height=2.70, px_per_meter=55.0):
    gray = cv2.cvtColor(plan_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_thick = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    
    all_walls = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
    thick_envelope = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_thick)
    interior_walls = cv2.subtract(all_walls, thick_envelope)
    
    contours, _ = cv2.findContours(interior_walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_linear_pixels = 0
    disp_img = plan_img.copy()
    
    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri > 30 and area > 50:
            total_linear_pixels += (peri / 2.0)
            cv2.drawContours(disp_img, [c], -1, (255, 100, 0), 2)
            
    total_linear_meters = total_linear_pixels / px_per_meter
    total_sqm = total_linear_meters * wall_height
    return total_linear_meters, total_sqm, disp_img, thick_envelope

def compare_building_delta(plan_a, plan_b, wall_height=2.70, px_per_meter=55.0):
    _, _, _, env_a = calc_building_partitions(plan_a, wall_height, px_per_meter)
    _, _, _, env_b = calc_building_partitions(plan_b, wall_height, px_per_meter)
    
    h, w = env_a.shape[:2]
    env_b_res = cv2.resize(env_b, (w, h))
    env_diff = cv2.absdiff(env_a, env_b_res)
    envelope_anomaly = np.count_nonzero(env_diff) > (w * h * 0.015)
    
    gray_a = cv2.cvtColor(plan_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(plan_b, cv2.COLOR_BGR2GRAY)
    gray_b_res = cv2.resize(gray_b, (w, h))
    
    _, th_a = cv2.threshold(gray_a, 215, 255, cv2.THRESH_BINARY_INV)
    _, th_b = cv2.threshold(gray_b_res, 215, 255, cv2.THRESH_BINARY_INV)
    
    demolition_mask = cv2.subtract(th_a, th_b)
    new_construction_mask = cv2.subtract(th_b, th_a)
    
    demo_len_m = (np.count_nonzero(demolition_mask) / 12.0) / px_per_meter
    demo_sqm = demo_len_m * wall_height
    
    new_len_m = (np.count_nonzero(new_construction_mask) / 12.0) / px_per_meter
    new_sqm = new_len_m * wall_height
    
    delta_disp = cv2.resize(plan_b, (w, h)).copy()
    delta_disp[demolition_mask > 0] = [0, 0, 255]
    delta_disp[new_construction_mask > 0] = [0, 200, 0]
    
    return demo_len_m, demo_sqm, new_len_m, new_sqm, envelope_anomaly, delta_disp

# ========================================================
# 🚿 מודול אינסטלציה (Delta Tracking מרחקי הזזה)
# ========================================================
def compare_plumbing_delta(matches_a, matches_b, px_per_meter=55.0):
    relocations = []
    added = []
    removed = []
    b_matched = set()
    for ma in matches_a:
        ca = ma["center"]
        best_dist = 999999
        best_mb_idx = -1
        for idx_b, mb in enumerate(matches_b):
            if idx_b in b_matched: continue
            cb = mb["center"]
            dist_px = np.hypot(ca[0] - cb[0], ca[1] - cb[1])
            dist_m = dist_px / px_per_meter
            if 0.30 <= dist_m <= 4.50 and dist_px < best_dist:
                best_dist = dist_px
                best_mb_idx = idx_b
        if best_mb_idx != -1:
            b_matched.add(best_mb_idx)
            relocations.append({"from": ca, "to": matches_b[best_mb_idx]["center"], "distance_m": round(best_dist / px_per_meter, 2)})
        else:
            removed.append(ma)
    for idx_b, mb in enumerate(matches_b):
        if idx_b not in b_matched: added.append(mb)
    return relocations, added, removed

# ========================================================
# 📐 מודול ריצוף וחיפוי (רצפה נטו + קירות חדרים רטובים)
# ========================================================
def calc_flooring_and_wall_tiling(plan_img, tiling_height=2.40, px_per_meter=55.0, plumbing_centers=[]):
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
        if not rows: continue
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
        for r in rows:
            img_tag = f'<img src="{r["image_uri"]}" width="55" height="40"/>' if r.get("image_uri") else "—"
            html += f"""
            <tr>
                <td>{r["מס'"]}</td>
                <td>{img_tag}</td>
                <td><b>{r["תיאור הפריט"]}</b></td>
                <td style="color: #1F4E78; font-size: 17px; font-weight: bold;">{r["כמות מאושרת"]}</td>
                <td>{r["יחידת מידה"]}</td>
            </tr>
            """
        html += "</table>"
    html += "</body></html>"
    return html

ai_memory = load_ai_memory()
disciplines_list = ["⚡ חשמל ומאור", "🧱 בניה (מחיצות ומעטפת)", "🚿 אינסטלציה", "📐 ריצוף וחיפוי"]

# אתחול קבוע למניעת שגיאות NameError
wall_h = 2.70
tile_h = 2.40
px_meter = 55.0

if "project_boq" not in st.session_state: st.session_state["project_boq"] = {}
if "current_discipline" not in st.session_state: st.session_state["current_discipline"] = "⚡ חשמל ומאור"
if "show_master_export" not in st.session_state: st.session_state["show_master_export"] = False

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
    st.subheader("📏 פרמטרים הנדסיים לחישוב")
    px_meter = st.number_input("כיול קנה מידה (פיקסלים למטר):", min_value=20.0, max_value=150.0, value=55.0, step=1.0)
    
    if st.session_state["current_discipline"] == "🧱 בניה (מחיצות ומעטפת)":
        wall_h = st.number_input("גובה מחיצות פנים (מטר):", min_value=2.0, max_value=4.5, value=2.70, step=0.05)
    elif st.session_state["current_discipline"] == "📐 ריצוף וחיפוי":
        tile_h = st.number_input("גובה חיפוי קירות בחדרים רטובים (מטר):", min_value=1.5, max_value=3.5, value=2.40, step=0.10)
        
    filter_banner = st.checkbox("סנן טבלת כותרת (Title Block)", value=True)
    
    st.markdown("---")
    saved_count = len([k for k, v in st.session_state["project_boq"].items() if len(v) > 0])
    st.info(f"דיסציפלינות שנשמרו בפרויקט: **{saved_count}**")
    if saved_count > 0:
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
    saved_discs = {k: v for k, v in st.session_state["project_boq"].items() if len(v) > 0}
    
    if not saved_discs:
        st.warning("טרם נשמרו חישובים בפרויקט.")
    else:
        for d_name, d_rows in saved_discs.items():
            with st.expander(f"📋 {d_name} ({len(d_rows)} שורות)", expanded=True):
                df_d = pd.DataFrame([{"מס'": r["מס'"], "תמונת סמל": r["תמונת סמל"], "תיאור הפריט": r["תיאור הפריט"], "כמות מאושרת": r["כמות מאושרת"], "יחידת מידה": r["יחידת מידה"]} for r in d_rows])
                st.dataframe(df_d, column_config={"תמונת סמל": st.column_config.ImageColumn("סמל גרפי", width="small")})
                
        st.markdown("---")
        st.subheader("📦 ייצוא דוח פרויקט מאוחד (כל הדיסציפלינות בקובץ אחד)")
        master_html = generate_master_export_html(saved_discs, title="דוח כתב כמויות מאוחד לפרויקט")
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.download_button("📊 הורד דוח פרויקט מאוחד מלא ל-Excel (XLS)", data=master_html.encode("utf-8"), file_name="Project_Master_Takeoff.xls", mime="application/vnd.ms-excel")
        with m_c2:
            st.download_button("📄 הורד דוח פרויקט מאוחד מלא להדפסה/PDF", data=master_html.encode("utf-8"), file_name="Project_Master_Report.html", mime="text/html")
            
    if st.button("🔙 חזור למסך הסריקה"):
        st.session_state["show_master_export"] = False
        st.rerun()

# ========================================================
# 📄 עיבוד שרטוטי PDF / תמונות
# ========================================================
elif file_type == "📄 PDF / תמונה (Raster)":
    
    # ----------------------------------------------------
    # 1. 🧱 מודול בניה (מחיצות ומעטפת)
    # ----------------------------------------------------
    if active_disc == "🧱 בניה (מחיצות ומעטפת)":
        c_exec, c_std, c_leg = st.columns(3)
        with c_exec: f_plan = st.file_uploader("1️⃣ תוכנית ביצוע (חובה):", type=["pdf", "png", "jpg"], key="b_plan_exec")
        with c_std: f_std = st.file_uploader("2️⃣ תוכנית סטנדרט / קיים (אופציונלי):", type=["pdf", "png", "jpg"], key="b_plan_std")
        with c_leg: f_leg = st.file_uploader("3️⃣ מקרא בניה (אופציונלי):", type=["pdf", "png", "jpg"], key="b_leg")
        
        if f_plan:
            btn_title = "🚀 הפעל השוואת שינויי בניה והריסה מול סטנדרט" if f_std else "🚀 הפעל חישוב מחיצות פנים נטו"
            if st.button(btn_title):
                p_bar = st.progress(0, text="מתחיל טעינת קבצים... (0%)")
                img_plan = load_raster(f_plan)
                
                if f_std:
                    p_bar.progress(20, text="טוען תוכנית סטנדרט ומבצע בדיקת תאימות... (20%)")
                    img_std = load_raster(f_std)
                    
                    # 🛡️ בדיקת תאימות בין התוכניות
                    is_compat, comp_score = check_plans_compatibility(img_plan, img_std)
                    if not is_compat:
                        p_bar.empty()
                        st.error("❌ **שגיאה: התוכניות שהוזנו אינן תואמות!** זוהה חוסר התאמה מבני בין תוכנית הביצוע לתוכנית הסטנדרט (השרטוטים שייכים לדיסציפלינות שונות או לתוכניות שאינן של אותו החלל). נא לוודא הזנת שרטוטים תואמים.")
                        st.stop()
                        
                    p_bar.progress(50, text="מחשב השוואת הריסה מול בניה חדשה... (50%)")
                    d_len, d_sqm, n_len, n_sqm, anomaly, delta_img = compare_building_delta(img_std, img_plan, wall_h, px_meter)
                    p_bar.progress(85, text="בודק שלמות מעטפת קונסטרוקטיבית... (85%)")
                    time.sleep(0.2)
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    if anomaly:
                        st.error("🚨 **התראת שינוי מעטפת (Envelope Anomaly Alert): זוהה שינוי במעטפת/אלמנט קונסטרוקטיבי!**")
                    else:
                        st.success("🛡️ מעטפת המבנה והאלמנטים הקונסטרוקטיביים נשמרו ללא שינוי.")
                    c1, c2 = st.columns(2)
                    c1.metric("מחיצות להריסה מול סטנדרט:", f"{round(d_sqm, 2)} מ\"ר", f"{round(d_len, 2)} מ\"א")
                    c2.metric("מחיצות חדשות לבניה:", f"{round(n_sqm, 2)} מ\"ר", f"{round(n_len, 2)} מ\"א")
                    b_rows = [
                        {"מס'": 1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": "מחיצות פנים מיועדות להריסה ביחס לסטנדרט", "כמות מאושרת": round(d_sqm, 2), "יחידת מידה": 'מ"ר'},
                        {"מס'": 2, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": "מחיצות פנים חדשות לבנייה", "כמות מאושרת": round(n_sqm, 2), "יחידת מידה": 'מ"ר'}
                    ]
                    st.session_state["project_boq"][active_disc] = b_rows
                    st.image(cv2.cvtColor(delta_img, cv2.COLOR_BGR2RGB), caption="אדום = הריסה, ירוק = בניה חדשה")
                else:
                    p_bar.progress(60, text="מבודד קירות פנים בלבד ומסנן מעטפת וממ\"ד... (60%)")
                    lin_m, sqm, disp_img, _ = calc_building_partitions(img_plan, wall_h, px_meter)
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.success("✅ החישוב הושלם! קירות מעטפת וממ\"ד סוננו אוטומטית.")
                    st.metric("שטח מחיצות פנים לבניה:", f"{round(sqm, 2)} מ\"ר", f"{round(lin_m, 2)} מ\"א")
                    b_rows = [
                        {"מס'": 1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": "מחיצות פנים (בלוק/גבס) - אורך כולל", "כמות מאושרת": round(lin_m, 2), "יחידת מידה": 'מ"א'},
                        {"מס'": 2, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"מחיצות פנים (גובה {wall_h} מ') - שטח כולל", "כמות מאושרת": round(sqm, 2), "יחידת מידה": 'מ"ר'}
                    ]
                    st.session_state["project_boq"][active_disc] = b_rows
                    st.image(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB), caption="שרטוט מחיצות פנים שזוהו (בכחול)")

    # ----------------------------------------------------
    # 2. 🚿 מודול אינסטלציה
    # ----------------------------------------------------
    elif active_disc == "🚿 אינסטלציה":
        c_exec, c_std, c_leg = st.columns(3)
        with c_exec: f_plan = st.file_uploader("1️⃣ תוכנית ביצוע אינסטלציה (חובה):", type=["pdf", "png", "jpg"], key="p_plan_exec")
        with c_std: f_std = st.file_uploader("2️⃣ תוכנית סטנדרט / קיים (אופציונלי):", type=["pdf", "png", "jpg"], key="p_plan_std")
        with c_leg: f_leg = st.file_uploader("3️⃣ מקרא כלים סניטריים (אופציונלי):", type=["pdf", "png", "jpg"], key="p_leg")
        
        if f_plan:
            btn_title = "🚀 הפעל השוואת שינויים ומרחקי העתקה מול סטנדרט" if f_std else "🚀 הפעל ספירת נקודות וכלים סניטריים"
            if st.button(btn_title):
                p_bar = st.progress(0, text="מתחיל טעינת תוכנית אינסטלציה... (0%)")
                img_plan = load_raster(f_plan)
                
                if f_std:
                    p_bar.progress(20, text="טוען תוכנית סטנדרט ובודק תאימות שרטוטים... (20%)")
                    img_std = load_raster(f_std)
                    
                    # 🛡️ בדיקת תאימות
                    is_compat, comp_score = check_plans_compatibility(img_plan, img_std)
                    if not is_compat:
                        p_bar.empty()
                        st.error("❌ **שגיאה: התוכניות שהוזנו אינן תואמות!** זוהה חוסר התאמה מבני בין תוכנית הביצוע לתוכנית הסטנדרט (השרטוטים שייכים לדיסציפלינות שונות או לחללים שונים). נא לוודא הזנת שרטוטים תואמים.")
                        st.stop()
                        
                    cl_a = auto_discover_plan_symbols(img_std)
                    p_bar.progress(50, text="מאתר נקודות בתוכנית הביצוע... (50%)")
                    cl_b = auto_discover_plan_symbols(img_plan)
                    
                    pts_a = [it for cl in cl_a for it in cl["items"]]
                    pts_b = [it for cl in cl_b for it in cl["items"]]
                    
                    p_bar.progress(75, text="מחשב מרחקי העתקה והזזה במטרים... (75%)")
                    relocs, added, removed = compare_plumbing_delta(pts_a, pts_b, px_meter)
                    
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.subheader("🔄 דוח שינויים והעתקת נקודות אינסטלציה מול סטנדרט")
                    st.metric("נקודות שהועתקו/הוזזו ממקומן:", f"{len(relocs)} יח'")
                    p_rows = []
                    for idx, r in enumerate(relocs):
                        p_rows.append({"מס'": idx+1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"העתקת נקודת אינסטלציה (הזזה של {r['distance_m']} מטר)", "כמות מאושרת": 1, "יחידת מידה": f"יח' ({r['distance_m']} מ')"})
                    for idx, a in enumerate(added):
                        p_rows.append({"מס'": len(relocs)+idx+1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": "תוספת נקודת אינסטלציה חדשה מעבר לסטנדרט", "כמות מאושרת": 1, "יחידת מידה": "יח'"})
                    st.session_state["project_boq"][active_disc] = p_rows
                    st.dataframe(pd.DataFrame(p_rows))
                else:
                    p_bar.progress(20, text="מעבד שכבות וסמלי אינסטלציה... (20%)")
                    plan_gray = cv2.cvtColor(img_plan, cv2.COLOR_BGR2GRAY)
                    _, plan_inv = cv2.threshold(plan_gray, 230, 255, cv2.THRESH_BINARY_INV)
                    symbols = extract_symbols_from_legend(load_raster(f_leg)) if f_leg else []
                    p_rows = []
                    disp_plan = img_plan.copy()
                    
                    if symbols:
                        total_s = len(symbols)
                        for i, sym in enumerate(symbols):
                            pct = 25 + int(((i + 1) / total_s) * 65)
                            p_bar.progress(pct, text=f"סורק כלי סניטרי {i+1} מתוך {total_s}... ({pct}%)")
                            m = match_symbol_ai(plan_inv, sym["crop_gray"])
                            for pt in m: cv2.rectangle(disp_plan, (pt["bbox"][0], pt["bbox"][1]), (pt["bbox"][0]+pt["bbox"][2], pt["bbox"][1]+pt["bbox"][3]), (0, 200, 0), 2)
                            p_rows.append({"מס'": i+1, "תמונת סמל": img_to_data_uri(sym["crop_color"]), "image_uri": img_to_data_uri(sym["crop_color"]), "תיאור הפריט": f"כלי סניטרי / נקודה #{i+1}", "כמות מאושרת": len(m), "יחידת מידה": "יח'"})
                    else:
                        p_bar.progress(50, text="מאתר ומקבץ נקודות סניטריות אוטונומית... (50%)")
                        clusters = auto_discover_plan_symbols(img_plan)
                        for i, cl in enumerate(clusters):
                            p_rows.append({"מס'": i+1, "תמונת סמל": img_to_data_uri(cl["rep_color"]), "image_uri": img_to_data_uri(cl["rep_color"]), "תיאור הפריט": f"נקודת אינסטלציה / כלי #{i+1}", "כמות מאושרת": len(cl["items"]), "יחידת מידה": "יח'"})
                    
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.session_state["project_boq"][active_disc] = p_rows
                    st.dataframe(pd.DataFrame(p_rows)[["מס'", "תמונת סמל", "תיאור הפריט", "כמות מאושרת", "יחידת מידה"]], column_config={"תמונת סמל": st.column_config.ImageColumn()})
                    st.image(cv2.cvtColor(disp_plan, cv2.COLOR_BGR2RGB))

    # ----------------------------------------------------
    # 3. 📐 מודול ריצוף וחיפוי (ללא מקרא)
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
                
                if f_std:
                    p_bar.progress(20, text="בודק תאימות שרטוטי ריצוף... (20%)")
                    img_std = load_raster(f_std)
                    
                    # 🛡️ בדיקת תאימות
                    is_compat, comp_score = check_plans_compatibility(img_plan, img_std)
                    if not is_compat:
                        p_bar.empty()
                        st.error("❌ **שגיאה: התוכניות שהוזנו אינן תואמות!** זוהה חוסר התאמה מבני בין תוכנית הביצוע לתוכנית הסטנדרט. נא לוודא הזנת שרטוטים תואמים.")
                        st.stop()
                        
                    plumb_clusters = auto_discover_plan_symbols(img_plan)
                    plumb_pts = [it["center"] for cl in plumb_clusters for it in cl["items"]]
                    floor_sqm, wet_peri_m, wet_wall_sqm, disp_img = calc_flooring_and_wall_tiling(img_plan, tile_h, px_meter, plumb_pts)
                    
                    p_bar.progress(60, text="משווה מול תוכנית סטנדרט... (60%)")
                    plumb_std = auto_discover_plan_symbols(img_std)
                    plumb_pts_std = [it["center"] for cl in plumb_std for it in cl["items"]]
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
                    p_bar.progress(30, text="מזהה כלים סניטריים לאיתור חדרים רטובים... (30%)")
                    plumb_clusters = auto_discover_plan_symbols(img_plan)
                    plumb_pts = [it["center"] for cl in plumb_clusters for it in cl["items"]]
                    
                    p_bar.progress(70, text="מחשב שטחי מצולעי חדרים נטו (ללא קירות)... (70%)")
                    floor_sqm, wet_peri_m, wet_wall_sqm, disp_img = calc_flooring_and_wall_tiling(img_plan, tile_h, px_meter, plumb_pts)
                    
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
                st.image(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB), caption="כתום = חלל רטוב לחיפוי קירות, ירוק = ריצוף יבש")

    # ----------------------------------------------------
    # 4. ⚡ מודול חשמל ומאור
    # ----------------------------------------------------
    else:
        c_exec, c_std, c_leg = st.columns(3)
        with c_exec: f_plan = st.file_uploader("1️⃣ תוכנית ביצוע חשמל (חובה):", type=["pdf", "png", "jpg"], key="e_plan_exec")
        with c_std: f_std = st.file_uploader("2️⃣ תוכנית סטנדרט / קיים (אופציונלי):", type=["pdf", "png", "jpg"], key="e_plan_std")
        with c_leg: f_leg = st.file_uploader("3️⃣ מקרא חשמל ומאור (אופציונלי):", type=["pdf", "png", "jpg"], key="e_leg")
        
        if f_plan:
            if st.button("🚀 הפעל פענוח וספירת חשמל ומאור"):
                p_bar = st.progress(0, text="מתחיל טעינת קבצי חשמל... (0%)")
                img_plan = load_raster(f_plan)
                
                if f_std:
                    p_bar.progress(15, text="בודק תאימות שרטוטים... (15%)")
                    img_std = load_raster(f_std)
                    is_compat, comp_score = check_plans_compatibility(img_plan, img_std)
                    if not is_compat:
                        p_bar.empty()
                        st.error("❌ **שגיאה: התוכניות שהוזנו אינן תואמות!** זוהה חוסר התאמה מבני בין תוכנית הביצוע לתוכנית הסטנדרט (השרטוטים שייכים לדיסציפלינות שונות או לקומות שונות). נא לוודא הזנת שרטוטים תואמים.")
                        st.stop()
                        
                p_bar.progress(25, text="מרנדר שכבות הנדסיות ומסנן רעשים... (25%)")
                plan_gray = cv2.cvtColor(img_plan, cv2.COLOR_BGR2GRAY)
                _, plan_inv = cv2.threshold(plan_gray, 230, 255, cv2.THRESH_BINARY_INV)
                
                p_bar.progress(35, text="מחלץ סמלים מהמקרא... (35%)")
                symbols = extract_symbols_from_legend(load_raster(f_leg)) if f_leg else []
                e_rows = []
                disp_plan = img_plan.copy()
                
                if symbols:
                    total_s = len(symbols)
                    for i, sym in enumerate(symbols):
                        pct = 40 + int(((i + 1) / total_s) * 50)
                        p_bar.progress(pct, text=f"סורק סמל {i+1} מתוך {total_s} בתוכנית... ({pct}%)")
                        m = match_symbol_ai(plan_inv, sym["crop_gray"])
                        for pt in m: cv2.rectangle(disp_plan, (pt["bbox"][0], pt["bbox"][1]), (pt["bbox"][0]+pt["bbox"][2], pt["bbox"][1]+pt["bbox"][3]), (0, 200, 0), 2)
                        e_rows.append({"מס'": i+1, "תמונת סמל": img_to_data_uri(sym["crop_color"]), "image_uri": img_to_data_uri(sym["crop_color"]), "תיאור הפריט": f"סמל חשמל/מאור #{i+1}", "כמות מאושרת": len(m), "יחידת מידה": "יח'"})
                else:
                    p_bar.progress(55, text="מאתר ומקבץ נקודות חשמל באופן אוטונומי... (55%)")
                    clusters = auto_discover_plan_symbols(img_plan)
                    for i, cl in enumerate(clusters):
                        e_rows.append({"מס'": i+1, "תמונת סמל": img_to_data_uri(cl["rep_color"]), "image_uri": img_to_data_uri(cl["rep_color"]), "תיאור הפריט": f"נקודת חשמל #{i+1}", "כמות מאושרת": len(cl["items"]), "יחידת מידה": "יח'"})
                
                p_bar.progress(95, text="מעדכן דוחות ובונה שרטוט סופי... (95%)")
                time.sleep(0.2)
                p_bar.progress(100, text="הסריקה הושלמה בהצלחה! (100%)")
                time.sleep(0.3)
                p_bar.empty()
                
                st.session_state["project_boq"][active_disc] = e_rows
                st.dataframe(pd.DataFrame(e_rows)[["מס'", "תמונת סמל", "תיאור הפריט", "כמות מאושרת", "יחידת מידה"]], column_config={"תמונת סמל": st.column_config.ImageColumn()})
                st.image(cv2.cvtColor(disp_plan, cv2.COLOR_BGR2RGB))

    # כפתורי סיום ומעבר דיסציפלינה
    if active_disc in st.session_state["project_boq"] and len(st.session_state["project_boq"][active_disc]) > 0:
        st.markdown("---")
        st.success(f"🎉 חישוב {active_disc} נשמר בהצלחה בזיכרון הפרויקט!")
        c_fin, c_next = st.columns(2)
        with c_fin:
            if st.button("🏁 סיום הפרויקט והפקת דוחות סופיים (Excel / PDF)"):
                st.session_state["show_master_export"] = True
                st.rerun()
        with c_next:
            st.write("**עבור לחישוב הדיסציפלינה הבאה:**")
            rem = [d for d in disciplines_list if d != active_disc]
            cols = st.columns(len(rem))
            for i, d_target in enumerate(rem):
                if cols[i].button(d_target, key=f"btn_nav_{i}"):
                    set_discipline_programmatically(d_target)
