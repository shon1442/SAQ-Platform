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
# 🛡️ מנוע בדיקת מעטפת חסין עיוותים ופונטים (Scale Invariant)
# ========================================================
def get_outer_envelope_contour(plan_img):
    gray = cv2.cvtColor(plan_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(thresh, k, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def verify_envelope_similarity(plan_a, plan_b):
    cnt_a = get_outer_envelope_contour(plan_a)
    cnt_b = get_outer_envelope_contour(plan_b)
    
    if cnt_a is None or cnt_b is None:
        return True, 0.0, None
        
    area_a = cv2.contourArea(cnt_a)
    area_b = cv2.contourArea(cnt_b)
    if area_a < 1000 or area_b < 1000:
        return True, 0.0, None
        
    shape_diff = cv2.matchShapes(cnt_a, cnt_b, cv2.CONTOURS_MATCH_I1, 0.0)
    
    xa, ya, wa, ha = cv2.boundingRect(cnt_a)
    xb, yb, wb, hb = cv2.boundingRect(cnt_b)
    aspect_a = wa / float(ha + 1e-5)
    aspect_b = wb / float(hb + 1e-5)
    aspect_diff = abs(aspect_a - aspect_b) / max(aspect_a, aspect_b)
    
    # חוסר התאמה מהותי: שינוי גיאומטרי קיצוני בצורת המעטפת או ביחס האורך/רוחב
    is_drastic_mismatch = (shape_diff > 0.60 and aspect_diff > 0.28) or (aspect_diff > 0.45)
    
    h, w = 320, 320
    vis = np.ones((h, w * 2, 3), dtype=np.uint8) * 255
    
    norm_a = (cnt_a.copy() - [xa, ya]) * [w / float(wa + 1e-5), h / float(ha + 1e-5)]
    norm_b = (cnt_b.copy() - [xb, yb]) * [w / float(wb + 1e-5), h / float(hb + 1e-5)]
    
    cv2.drawContours(vis[:, :w], [norm_a.astype(np.int32)], -1, (200, 50, 50), 3)
    cv2.putText(vis, "מעטפת תוכנית סטנדרט", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 0, 0), 2)
    
    cv2.drawContours(vis[:, w:], [norm_b.astype(np.int32)], -1, (50, 180, 50), 3)
    cv2.putText(vis, "מעטפת תוכנית ביצוע", (w + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 0), 2)
    
    return not is_drastic_mismatch, round(shape_diff, 3), vis

# ========================================================
# 🧱 מנוע שלד וקירות פנים (Morphological Centerline Engine)
# ========================================================
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

def extract_interior_and_envelope(plan_img, px_per_meter=55.0):
    gray = cv2.cvtColor(plan_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
    
    k_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_clean)
    
    thick_kernel_size = max(9, int(px_per_meter * 0.20))
    k_thick = cv2.getStructuringElement(cv2.MORPH_RECT, (thick_kernel_size, thick_kernel_size))
    envelope = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_thick)
    
    interior = cv2.subtract(cleaned, envelope)
    k_interior_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    interior_clean = cv2.morphologyEx(interior, cv2.MORPH_OPEN, k_interior_smooth)
    
    return interior_clean, envelope

def calc_building_partitions_linear(plan_img, px_per_meter=55.0):
    interior_clean, envelope = extract_interior_and_envelope(plan_img, px_per_meter)
    skel = get_morphological_skeleton(interior_clean)
    
    linear_pixels = cv2.countNonZero(skel)
    linear_meters = round(linear_pixels / float(px_per_meter), 2)
    
    disp_img = plan_img.copy()
    disp_img[skel > 0] = [255, 100, 0]
    disp_img = cv2.dilate(disp_img, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    
    return linear_meters, disp_img, envelope

def compare_building_delta_linear(plan_std, plan_exec, px_per_meter=55.0):
    interior_std, env_std = extract_interior_and_envelope(plan_std, px_per_meter)
    interior_exec, env_exec = extract_interior_and_envelope(plan_exec, px_per_meter)
    
    h, w = env_std.shape[:2]
    env_exec_res = cv2.resize(env_exec, (w, h))
    interior_exec_res = cv2.resize(interior_exec, (w, h))
    
    env_diff = cv2.absdiff(env_std, env_exec_res)
    k_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    env_diff_clean = cv2.morphologyEx(env_diff, cv2.MORPH_OPEN, k_noise)
    
    anomaly_contours, _ = cv2.findContours(env_diff_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_anomalies = [c for c in anomaly_contours if cv2.contourArea(c) > (px_per_meter * 3)]
    envelope_anomaly = len(significant_anomalies) > 0
    
    demolition_mask = cv2.subtract(interior_std, interior_exec_res)
    new_construction_mask = cv2.subtract(interior_exec_res, interior_std)
    
    skel_demo = get_morphological_skeleton(demolition_mask)
    skel_new = get_morphological_skeleton(new_construction_mask)
    
    demo_meters = round(cv2.countNonZero(skel_demo) / float(px_per_meter), 2)
    new_meters = round(cv2.countNonZero(skel_new) / float(px_per_meter), 2)
    
    delta_disp = cv2.resize(plan_exec, (w, h)).copy()
    delta_disp[skel_demo > 0] = [0, 0, 255]
    delta_disp[skel_new > 0] = [0, 200, 0]
    
    if envelope_anomaly:
        for c in significant_anomalies:
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(delta_disp, (max(0, x - 8), max(0, y - 8)), (min(w, x + bw + 8), min(h, y + bh + 8)), (0, 0, 255), 3)
            cv2.putText(delta_disp, "!חריגת מעטפת", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    return demo_meters, new_meters, envelope_anomaly, delta_disp, len(significant_anomalies)

# ========================================================
# ⚡ פענוח סמלים (חשמל / אינסטלציה / מקרא)
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
    if plan_roi is None or plan_roi.size == 0:
        return []
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
    valid = [cl for cl in clusters if len(cl["items"]) >= 1]
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
# 🚿 מודול אינסטלציה (Delta Tracking)
# ========================================================
def compare_plumbing_delta_smart(plan_a, plan_b, px_per_meter=55.0):
    cl_a = auto_discover_plan_symbols(plan_a)
    cl_b = auto_discover_plan_symbols(plan_b)
    
    pts_a = [it for cl in cl_a for it in cl["items"]]
    pts_b = [it for cl in cl_b for it in cl["items"]]
    
    relocations = []
    added = []
    removed = []
    b_matched = set()
    
    for ma in pts_a:
        ca = ma["center"]
        best_dist = 999999
        best_mb_idx = -1
        for idx_b, mb in enumerate(pts_b):
            if idx_b in b_matched: continue
            cb = mb["center"]
            dist_px = np.hypot(ca[0] - cb[0], ca[1] - cb[1])
            dist_m = dist_px / px_per_meter
            if 0.25 <= dist_m <= 5.0 and dist_px < best_dist:
                best_dist = dist_px
                best_mb_idx = idx_b
        if best_mb_idx != -1:
            b_matched.add(best_mb_idx)
            relocations.append({"from": ca, "to": pts_b[best_mb_idx]["center"], "distance_m": round(best_dist / px_per_meter, 2)})
        else:
            removed.append(ma)
            
    for idx_b, mb in enumerate(pts_b):
        if idx_b not in b_matched:
            added.append(mb)
            
    if len(relocations) == 0 and len(added) == 0:
        h, w = plan_a.shape[:2]
        gray_a = cv2.cvtColor(plan_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(cv2.resize(plan_b, (w, h)), cv2.COLOR_BGR2GRAY)
        _, th_a = cv2.threshold(gray_a, 215, 255, cv2.THRESH_BINARY_INV)
        _, th_b = cv2.threshold(gray_b, 215, 255, cv2.THRESH_BINARY_INV)
        diff = cv2.absdiff(th_a, th_b)
        cnts, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = [c for c in cnts if 50 < cv2.contourArea(c) < 3000]
        for c in valid_cnts[:8]:
            x, y, w_c, h_c = cv2.boundingRect(c)
            added.append({"center": (x + w_c // 2, y + h_c // 2), "bbox": (x, y, w_c, h_c)})
            
    return relocations, added, removed

# ========================================================
# 📐 מודול ריצוף וחיפוי
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
px_meter = 55.0

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
    st.subheader("📏 פרמטרים לכיול")
    px_meter = st.number_input("כיול קנה מידה (פיקסלים למטר):", min_value=20.0, max_value=150.0, value=55.0, step=1.0)
    if st.session_state["current_discipline"] == "📐 ריצוף וחיפוי":
        tile_h = st.number_input("גובה חיפוי קירות בחדרים רטובים (מטר):", min_value=1.5, max_value=3.5, value=2.40, step=0.10)
        
    filter_banner = st.checkbox("סנן טבלת כותרת (Title Block)", value=True)
    
    st.markdown("---")
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
    # 1. 🧱 מודול בניה (מחיצות פנים במ"א והכפלה בגובה)
    # ----------------------------------------------------
    if active_disc == "🧱 בניה (מחיצות ומעטפת)":
        c_exec, c_std, c_leg = st.columns(3)
        with c_exec: f_plan = st.file_uploader("1️⃣ תוכנית ביצוע (חובה):", type=["pdf", "png", "jpg"], key="b_plan_exec")
        with c_std: f_std = st.file_uploader("2️⃣ תוכנית סטנדרט / קיים (אופציונלי):", type=["pdf", "png", "jpg"], key="b_plan_std")
        with c_leg: f_leg = st.file_uploader("3️⃣ מקרא בניה (אופציונלי):", type=["pdf", "png", "jpg"], key="b_leg")
        
        st.markdown("---")
        b_wall_h = st.number_input("📏 בכמה מטר גובה להכפיל את המטר הרץ לקבלת שטח (מ\"ר)?", min_value=1.5, max_value=5.0, value=2.70, step=0.05)
        
        if f_plan:
            btn_title = "🚀 הפעל השוואת שינויי בניה והריסה מול סטנדרט" if f_std else "🚀 הפעל חישוב מחיצות פנים"
            if st.button(btn_title):
                p_bar = st.progress(0, text="מתחיל עיבוד תוכנית בניה... (0%)")
                img_plan = load_raster(f_plan)
                
                if f_std:
                    p_bar.progress(20, text="טוען תוכנית סטנדרט ובודק דמיון מעטפת... (20%)")
                    img_std = load_raster(f_std)
                    
                    is_similar, shape_score, vis_env = verify_envelope_similarity(img_std, img_plan)
                    user_overridden = st.session_state.get(f"override_{active_disc}", False)
                    
                    # בדיקת שוני מהותי במעטפת ושאלת המשתמש
                    if not is_similar and not user_overridden:
                        p_bar.empty()
                        st.warning(f"⚠️ **זוהה שוני מהותי במעטפת ובצורת הדירה בין שתי התוכניות (ציון שוני: {shape_score}).** ייתכן שהוזנו תוכניות של דירות שונות לחלוטין.")
                        if vis_env is not None:
                            st.image(cv2.cvtColor(vis_env, cv2.COLOR_BGR2RGB), caption="השוואת מעטפת חיצונית מנורמלת")
                            
                        st.write("❓ **כיצד תרצה להמשיך?**")
                        c_opt1, c_opt2 = st.columns(2)
                        with c_opt1:
                            if st.button("✅ המשך בחישוב מחיצות פנים בכל זאת", key="btn_override_env"):
                                st.session_state[f"override_{active_disc}"] = True
                                st.rerun()
                        with c_opt2:
                            st.info("🛑 באפשרותך להחליף את אחת התוכניות מעלה להשוואת אותה הדירה.")
                        st.stop()
                    
                    st.session_state[f"override_{active_disc}"] = False
                    p_bar.progress(45, text="מבודד מחיצות פנים ומסנן מעטפת... (45%)")
                    demo_m, new_m, anomaly, delta_img, num_anomalies = compare_building_delta_linear(img_std, img_plan, px_meter)
                    
                    demo_sqm = round(demo_m * b_wall_h, 2)
                    new_sqm = round(new_m * b_wall_h, 2)
                    
                    p_bar.progress(85, text="מעדכן נתונים ומסמן חריגות מעטפת... (85%)")
                    time.sleep(0.2)
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.success("✅ **חישוב מחיצות פנים הושלם בהצלחה (קירות מעטפת, עמודים וממ\"ד סוננו אוטומטית):**")
                    c1, c2 = st.columns(2)
                    c1.metric("מחיצות פנים להריסה:", f"{demo_m} מ\"א", f"{demo_sqm} מ\"ר (לפי גובה {b_wall_h} מ')")
                    c2.metric("מחיצות פנים חדשות לבניה:", f"{new_m} מ\"א", f"{new_sqm} מ\"ר (לפי גובה {b_wall_h} מ')")
                    
                    if anomaly:
                        st.error(f"🚨 **התראת שינוי מעטפת (Envelope Anomaly Alert): אותרו {num_anomalies} שינויים באלמנט מעטפת / עמוד קונסטרוקטיבי! המיקומים סומנו בריבוע אדום על גבי השרטוט להלן.**")
                    else:
                        st.info("🛡️ מעטפת המבנה והאלמנטים הקונסטרוקטיביים נשמרו ללא שינוי.")
                        
                    b_rows = [
                        {"מס'": 1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"מחיצות פנים להריסה (אורך {demo_m} מ\"א * גובה {b_wall_h} מ')", "כמות מאושרת": demo_sqm, "יחידת מידה": 'מ"ר'},
                        {"מס'": 2, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"מחיצות פנים חדשות לבנייה (אורך {new_m} מ\"א * גובה {b_wall_h} מ')", "כמות מאושרת": new_sqm, "יחידת מידה": 'מ"ר'}
                    ]
                    st.session_state["project_boq"][active_disc] = b_rows
                    safe_render_table(b_rows)
                    st.image(cv2.cvtColor(delta_img, cv2.COLOR_BGR2RGB), caption="מפת שינויים: אדום = הריסה, ירוק = בניה, ריבוע אדום מודגש = חריגת מעטפת")
                else:
                    p_bar.progress(60, text="מסנן קירות מעטפת, ממ\"ד וקווי מידה... (60%)")
                    lin_m, disp_img, _ = calc_building_partitions_linear(img_plan, px_meter)
                    sqm_total = round(lin_m * b_wall_h, 2)
                    
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.success("✅ החישוב הושלם! קירות מעטפת, ממ\"ד וקווי מידה סוננו אוטומטית.")
                    c1, c2 = st.columns(2)
                    c1.metric("אורך מחיצות פנים נטו:", f"{lin_m} מ\"א")
                    c2.metric(f"שטח מחיצות פנים (גובה {b_wall_h} מ'):", f"{sqm_total} מ\"ר")
                    
                    b_rows = [
                        {"מס'": 1, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": "מחיצות פנים נטו - אורך כולל", "כמות מאושרת": lin_m, "יחידת מידה": 'מ"א'},
                        {"מס'": 2, "תמונת סמל": "", "image_uri": "", "תיאור הפריט": f"מחיצות פנים נטו - שטח כולל (גובה {b_wall_h} מ')", "כמות מאושרת": sqm_total, "יחידת מידה": 'מ"ר'}
                    ]
                    st.session_state["project_boq"][active_disc] = b_rows
                    safe_render_table(b_rows)
                    st.image(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB), caption="שרטוט מחיצות פנים שנמדדו (בכחול)")

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
                    p_bar.progress(20, text="טוען תוכנית סטנדרט ובודק דמיון מעטפת... (20%)")
                    img_std = load_raster(f_std)
                    
                    is_similar, shape_score, vis_env = verify_envelope_similarity(img_std, img_plan)
                    user_overridden = st.session_state.get(f"override_{active_disc}", False)
                    
                    if not is_similar and not user_overridden:
                        p_bar.empty()
                        st.warning(f"⚠️ **זוהה שוני מהותי במעטפת ובצורת הדירה בין שתי התוכניות (ציון שוני: {shape_score}).**")
                        if vis_env is not None:
                            st.image(cv2.cvtColor(vis_env, cv2.COLOR_BGR2RGB), caption="השוואת מעטפת חיצונית מנורמלת")
                        c_opt1, c_opt2 = st.columns(2)
                        with c_opt1:
                            if st.button("✅ המשך בחישוב אינסטלציה בכל זאת", key="btn_override_env_p"):
                                st.session_state[f"override_{active_disc}"] = True
                                st.rerun()
                        with c_opt2:
                            st.info("🛑 באפשרותך להחליף תוכנית מעלה.")
                        st.stop()
                        
                    st.session_state[f"override_{active_disc}"] = False
                    p_bar.progress(60, text="מחשב העתקות נקודות ומרחקי הזזה במטרים... (60%)")
                    relocs, added, removed = compare_plumbing_delta_smart(img_std, img_plan, px_meter)
                    
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.subheader("🔄 דוח שינויים והעתקת נקודות אינסטלציה מול סטנדרט")
                    st.metric("נקודות שהועתקו/הוזזו ממקומן:", f"{len(relocs)} יח'", f"+{len(added)} נקודות חדשות")
                    
                    p_rows = []
                    for idx, r in enumerate(relocs):
                        p_rows.append({
                            "מס'": idx+1, "תמונת סמל": "", "image_uri": "",
                            "תיאור הפריט": f"העתקת נקודת אינסטלציה (הזזה של {r['distance_m']} מטר)",
                            "כמות מאושרת": 1, "יחידת מידה": f"יח' ({r['distance_m']} מ')"
                        })
                    for idx, a in enumerate(added):
                        p_rows.append({
                            "מס'": len(relocs)+idx+1, "תמונת סמל": "", "image_uri": "",
                            "תיאור הפריט": "תוספת נקודת אינסטלציה חדשה מעבר לסטנדרט",
                            "כמות מאושרת": 1, "יחידת מידה": "יח'"
                        })
                    st.session_state["project_boq"][active_disc] = p_rows
                    safe_render_table(p_rows)
                else:
                    p_bar.progress(30, text="מעבד שכבות וסמלי אינסטלציה... (30%)")
                    plan_gray = cv2.cvtColor(img_plan, cv2.COLOR_BGR2GRAY)
                    _, plan_inv = cv2.threshold(plan_gray, 230, 255, cv2.THRESH_BINARY_INV)
                    symbols = extract_symbols_from_legend(load_raster(f_leg)) if f_leg else []
                    p_rows = []
                    disp_plan = img_plan.copy()
                    
                    if symbols:
                        total_s = len(symbols)
                        for i, sym in enumerate(symbols):
                            pct = 35 + int(((i + 1) / total_s) * 55)
                            p_bar.progress(pct, text=f"סורק כלי סניטרי {i+1} מתוך {total_s}... ({pct}%)")
                            m = match_symbol_ai(plan_inv, sym["crop_gray"])
                            for pt in m:
                                cv2.rectangle(disp_plan, (pt["bbox"][0], pt["bbox"][1]), (pt["bbox"][0]+pt["bbox"][2], pt["bbox"][1]+pt["bbox"][3]), (0, 200, 0), 2)
                            p_rows.append({
                                "מס'": i+1,
                                "תמונת סמל": img_to_data_uri(sym["crop_color"]),
                                "image_uri": img_to_data_uri(sym["crop_color"]),
                                "תיאור הפריט": f"כלי סניטרי / נקודה #{i+1}",
                                "כמות מאושרת": len(m),
                                "יחידת מידה": "יח'"
                            })
                    else:
                        p_bar.progress(60, text="מאתר ומקבץ נקודות סניטריות אוטונומית... (60%)")
                        clusters = auto_discover_plan_symbols(img_plan)
                        for i, cl in enumerate(clusters):
                            p_rows.append({
                                "מס'": i+1,
                                "תמונת סמל": img_to_data_uri(cl["rep_color"]),
                                "image_uri": img_to_data_uri(cl["rep_color"]),
                                "תיאור הפריט": f"נקודת אינסטלציה / כלי #{i+1}",
                                "כמות מאושרת": len(cl["items"]),
                                "יחידת מידה": "יח'"
                            })
                    
                    p_bar.progress(100, text="החישוב הושלם בהצלחה! (100%)")
                    time.sleep(0.3)
                    p_bar.empty()
                    
                    st.session_state["project_boq"][active_disc] = p_rows
                    safe_render_table(p_rows)
                    st.image(cv2.cvtColor(disp_plan, cv2.COLOR_BGR2RGB))

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
                
                if f_std:
                    p_bar.progress(30, text="מחשב שטחי ביצוע מול סטנדרט... (30%)")
                    img_std = load_raster(f_std)
                    
                    plumb_clusters = auto_discover_plan_symbols(img_plan)
                    plumb_pts = [it["center"] for cl in plumb_clusters for it in cl["items"]]
                    floor_sqm, wet_peri_m, wet_wall_sqm, disp_img = calc_flooring_and_wall_tiling(img_plan, tile_h, px_meter, plumb_pts)
                    
                    p_bar.progress(65, text="משווה מול תוכנית סטנדרט... (65%)")
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
                    p_bar.progress(40, text="מזהה כלים סניטריים לאיתור חדרים רטובים... (40%)")
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
                safe_render_table(f_rows)
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
                        for pt in m:
                            cv2.rectangle(disp_plan, (pt["bbox"][0], pt["bbox"][1]), (pt["bbox"][0]+pt["bbox"][2], pt["bbox"][1]+pt["bbox"][3]), (0, 200, 0), 2)
                        e_rows.append({
                            "מס'": i+1,
                            "תמונת סמל": img_to_data_uri(sym["crop_color"]),
                            "image_uri": img_to_data_uri(sym["crop_color"]),
                            "תיאור הפריט": f"סמל חשמל/מאור #{i+1}",
                            "כמות מאושרת": len(m),
                            "יחידת מידה": "יח'"
                        })
                else:
                    p_bar.progress(55, text="מאתר ומקבץ נקודות חשמל באופן אוטונומי... (55%)")
                    clusters = auto_discover_plan_symbols(img_plan)
                    for i, cl in enumerate(clusters):
                        e_rows.append({
                            "מס'": i+1,
                            "תמונת סמל": img_to_data_uri(cl["rep_color"]),
                            "image_uri": img_to_data_uri(cl["rep_color"]),
                            "תיאור הפריט": f"נקודת חשמל #{i+1}",
                            "כמות מאושרת": len(cl["items"]),
                            "יחידת מידה": "יח'"
                        })
                
                p_bar.progress(100, text="הסריקה הושלמה בהצלחה! (100%)")
                time.sleep(0.3)
                p_bar.empty()
                
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
