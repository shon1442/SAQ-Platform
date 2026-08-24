import streamlit as st
import os
import io
import json
import base64
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

st.set_page_config(page_title="S.A.Q - Autonomous AI Takeoff", layout="wide", page_icon=app_icon)

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

def extract_symbols_and_text_from_legend(legend_img):
    gray = cv2.cvtColor(legend_img, cv2.COLOR_BGR2GRAY)
    leg_h = gray.shape[0]
    crop_h = int(leg_h * 0.88)
    work_gray = gray[:crop_h, :]
    work_color = legend_img[:crop_h, :]
    
    _, thresh = cv2.threshold(work_gray, 225, 255, cv2.THRESH_BINARY_INV)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
    cleaned = cv2.subtract(thresh, cv2.add(h_lines, v_lines))
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_symbols = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if 14 <= w <= 110 and 14 <= h <= 110 and area > 40:
            aspect = w / float(h)
            if 0.40 <= aspect <= 2.4:
                pad = 4
                y1, y2 = max(0, y - pad), min(work_gray.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(work_gray.shape[1], x + w + pad)
                c_gray = work_gray[y1:y2, x1:x2]
                c_color = work_color[y1:y2, x1:x2]
                
                raw_symbols.append({
                    "bbox": (x, y, w, h),
                    "crop_color": c_color,
                    "crop_gray": c_gray,
                    "y_pos": y,
                    "x_pos": x
                })
                
    raw_symbols.sort(key=lambda s: (s["y_pos"] // 35, s["x_pos"]))
    unique_symbols = []
    for sym in raw_symbols:
        is_dup = False
        for u in unique_symbols:
            if np.hypot(sym["x_pos"] - u["x_pos"], sym["y_pos"] - u["y_pos"]) < 26:
                is_dup = True
                break
        if not is_dup:
            unique_symbols.append(sym)
    return unique_symbols[:16]

def auto_discover_plan_symbols(plan_roi, min_dim=15, max_dim=90, match_thresh=0.68):
    gray = cv2.cvtColor(plan_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_dim <= w <= max_dim and min_dim <= h <= max_dim and cv2.contourArea(c) > 40:
            aspect = w / float(h)
            if 0.45 <= aspect <= 2.2:
                pad = 3
                y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
                candidates.append({
                    "bbox": (x, y, w, h),
                    "crop_gray": gray[y1:y2, x1:x2],
                    "crop_color": plan_roi[y1:y2, x1:x2],
                    "center": (x + w // 2, y + h // 2)
                })
                
    clusters = []
    for cand in candidates:
        matched = False
        c_crop = cand["crop_gray"]
        for cl in clusters:
            rep = cl["rep_gray"]
            if abs(c_crop.shape[0] - rep.shape[0]) > 10 or abs(c_crop.shape[1] - rep.shape[1]) > 10:
                continue
            r_h, r_w = rep.shape[:2]
            resized_c = cv2.resize(c_crop, (r_w, r_h))
            res = cv2.matchTemplate(resized_c, rep, cv2.TM_CCOEFF_NORMED)
            if res[0][0] >= match_thresh:
                cl["items"].append(cand)
                matched = True
                break
        if not matched:
            clusters.append({
                "rep_gray": c_crop,
                "rep_color": cand["crop_color"],
                "items": [cand]
            })
            
    valid_clusters = [cl for cl in clusters if len(cl["items"]) >= 2]
    valid_clusters.sort(key=lambda x: len(x["items"]), reverse=True)
    return valid_clusters[:14]

def match_symbol_ai(plan_inv, templ_gray, min_thresh=0.62, high_thresh=0.74):
    _, templ_inv = cv2.threshold(templ_gray, 230, 255, cv2.THRESH_BINARY_INV)
    pts = cv2.findNonZero(templ_inv)
    if pts is not None:
        tx, ty, tw, th = cv2.boundingRect(pts)
        if tw > 8 and th > 8:
            templ_inv = templ_inv[ty:ty+th, tx:tx+tw]
            
    detections = []
    scales = [0.82, 0.90, 1.0, 1.10, 1.20]
    for scale in scales:
        sw = int(templ_inv.shape[1] * scale)
        sh = int(templ_inv.shape[0] * scale)
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
                status = "Green" if score >= high_thresh else "Yellow"
                detections.append({
                    "bbox": (int(pt[0]), int(pt[1]), int(rw), int(rh)),
                    "center": (int(pt[0] + rw // 2), int(pt[1] + rh // 2)),
                    "score": score,
                    "status": status
                })
                
    if not detections:
        return []
    boxes = [list(d["bbox"]) for d in detections]
    scores = [d["score"] for d in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=min_thresh, nms_threshold=0.20)
    final_res = [detections[i] for i in indices.flatten()] if len(indices) > 0 else []
    if len(final_res) > 70:
        return [r for r in final_res if r["status"] == "Green"]
    return final_res

def generate_export_html(boq_rows, title="דוח כתב כמויות"):
    html = f"""
    <html dir="rtl">
    <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        @media print {{ body {{ -webkit-print-color-adjust: exact; }} }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background-color: #fafafa; }}
        .header-box {{ border-bottom: 3px solid #1F4E78; padding-bottom: 10px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; background-color: #ffffff; box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 25px; }}
        th {{ background-color: #1F4E78; color: white; padding: 12px; font-size: 15px; border: 1px solid #ddd; }}
        td {{ padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 14px; vertical-align: middle; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        img {{ border: 1px solid #ccc; background: white; padding: 2px; border-radius: 4px; }}
    </style>
    </head>
    <body>
    <div class="header-box">
        <h2>📋 {title} - S.A.Q AI Platform</h2>
        <p>תאריך הפקה: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}</p>
    </div>
    <table>
        <tr>
            <th>מס'</th>
            <th>סמל / סימון גרפי</th>
            <th>תיאור הפריט</th>
            <th>כמות מאושרת</th>
            <th>יחידת מידה</th>
        </tr>
    """
    for r in boq_rows:
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
    html += "</table></body></html>"
    return html

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
        if not rows:
            continue
        total_items = sum(r["כמות מאושרת"] for r in rows)
        html += f"""
        <h3 class="disc-title">{disc_name} (סה"כ: {total_items} יח')</h3>
        <table>
            <tr>
                <th>מס'</th>
                <th>סמל גרפי</th>
                <th>תיאור הפריט</th>
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

# אתחול זיכרון פרויקטלי
if "project_boq" not in st.session_state:
    st.session_state["project_boq"] = {}
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
    if has_logo:
        st.image(LOGO_PATH)
    st.header("⚙️ הגדרות עבודה")
    file_type = st.radio("פורמט שרטוט:", ["📄 PDF / תמונה (Raster)", "📐 CAD וקטורי (DXF)"])
    mode = st.radio("מצב פעולה:", ["ספירה מתוכנית בודדת", "השוואת שינויים (Delta)"])
    
    discipline = st.selectbox(
        "דיסציפלינה:",
        disciplines_list,
        index=curr_idx,
        key="disc_selector_widget",
        on_change=on_discipline_change
    )
    
    st.markdown("---")
    st.subheader("📁 סטטוס פרויקט מצטבר")
    saved_count = len([k for k, v in st.session_state["project_boq"].items() if len(v) > 0])
    st.info(f"דיסציפלינות שנשמרו בפרויקט: **{saved_count}**")
    if saved_count > 0:
        if st.button("📑 פתח מרכז דוחות פרויקט מלא"):
            st.session_state["show_master_export"] = True
            st.rerun()
            
    st.markdown("---")
    st.subheader("🧠 מנוע למידה AI וכיול")
    st.caption(f"זיכרון דפוסים פעיל: {len(ai_memory.get('approved_patterns', []))} אישורים שמורים.")
    high_sens = st.slider("סף ודאי אוטומטי (Green %):", min_value=65, max_value=85, value=75, step=1)
    min_sens = st.slider("סף שאלת משתמש (Yellow %):", min_value=50, max_value=68, value=60, step=1)
    filter_banner = st.checkbox("סנן טבלת כותרת (Title Block)", value=True)

col_l, col_t = st.columns([1, 6])
with col_l:
    if has_logo:
        st.image(LOGO_PATH, width=90)
with col_t:
    st.title("S.A.Q Takeoff & Delta Platform")
    st.caption(f"פלטפורמת ענן לפענוח הנדסי אוטונומי - {st.session_state['current_discipline']}")

active_disc = st.session_state["current_discipline"]

# ========================================================
# 📑 תצוגת מרכז דוחות פרויקט מלא (Master BOQ Hub)
# ========================================================
if st.session_state.get("show_master_export", False):
    st.markdown("---")
    st.header("🏗️ מרכז הפקת דוחות סופיים לפרויקט")
    st.caption("כל החישובים שנשמרו בפרויקט מרוכזים להלן. באפשרותך לייצא דוח מאוחד או דוחות נפרדים ב-Excel וב-PDF.")
    
    saved_discs = {k: v for k, v in st.session_state["project_boq"].items() if len(v) > 0}
    
    if not saved_discs:
        st.warning("טרם נשמרו חישובים בפרויקט. בצע סריקה ואישור ראשוני.")
    else:
        for d_name, d_rows in saved_discs.items():
            with st.expander(f"📋 {d_name} ({len(d_rows)} פריטים)", expanded=True):
                df_d = pd.DataFrame([{"מס'": r["מס'"], "תמונת סמל": r["תמונת סמל"], "תיאור הפריט": r["תיאור הפריט"], "כמות מאושרת": r["כמות מאושרת"], "יחידת מידה": r["יחידת מידה"]} for r in d_rows])
                st.dataframe(df_d, column_config={"תמונת סמל": st.column_config.ImageColumn("סמל גרפי", width="small")})
                
                c_d1, c_d2 = st.columns(2)
                h_d = generate_export_html(d_rows, title=f"כתב כמויות - {d_name}")
                with c_d1:
                    st.download_button(f"📊 הורד {d_name} (Excel)", data=h_d.encode("utf-8"), file_name=f"Takeoff_{d_name}.xls", mime="application/vnd.ms-excel", key=f"dl_xls_{d_name}")
                with c_d2:
                    st.download_button(f"📄 הורד {d_name} (PDF)", data=h_d.encode("utf-8"), file_name=f"Report_{d_name}.html", mime="text/html", key=f"dl_pdf_{d_name}")

        st.markdown("---")
        st.subheader("📦 ייצוא דוח פרויקט מאוחד (כל הדיסציפלינות בקובץ אחד)")
        master_html = generate_master_export_html(saved_discs, title="דוח כתב כמויות מאוחד לפרויקט")
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.download_button("📊 הורד דוח פרויקט מאוחד מלא ל-Excel (XLS)", data=master_html.encode("utf-8"), file_name="Project_Master_Takeoff.xls", mime="application/vnd.ms-excel")
        with m_c2:
            st.download_button("📄 הורד דוח פרויקט מאוחד מלא להדפסה/PDF", data=master_html.encode("utf-8"), file_name="Project_Master_Report.html", mime="text/html")
            
    if st.button("🔙 חזור למסך הסריקה והחישוב"):
        st.session_state["show_master_export"] = False
        st.rerun()

# ========================================================
# 📄 תצוגת סריקה וחישוב דיסציפלינה
# ========================================================
elif file_type == "📄 PDF / תמונה (Raster)":
    if mode == "ספירה מתוכנית בודדת":
        c_p, c_l = st.columns(2)
        with c_p:
            f_plan = st.file_uploader("1️⃣ העלה שרטוט תוכנית (חובה):", type=["pdf", "png", "jpg", "jpeg"], key=f"plan_in_{active_disc}")
        with c_l:
            f_legend = st.file_uploader("2️⃣ העלה קובץ מקרא (אופציונלי):", type=["pdf", "png", "jpg", "jpeg"], key=f"leg_in_{active_disc}")
            
        if f_plan:
            btn_text = "🚀 הפעל פענוח מקרא וספירה מדויקת" if f_legend else "🚀 הפעל סריקה וספירה אוטונומית (ללא מקרא)"
            if st.button(btn_text):
                st.session_state["verification_completed"] = False
                img_plan = load_raster(f_plan, scale=1.4)
                
                if img_plan is None:
                    st.error("שגיאה בטעינת קובץ התוכנית.")
                else:
                    plan_gray = cv2.cvtColor(img_plan, cv2.COLOR_BGR2GRAY)
                    h_p, w_p = plan_gray.shape[:2]
                    active_h = int(h_p * 0.82) if filter_banner else h_p
                    plan_roi = plan_gray[:active_h, :]
                    _, plan_inv = cv2.threshold(plan_roi, 230, 255, cv2.THRESH_BINARY_INV)
                    
                    all_results = []
                    
                    if f_legend:
                        img_leg = load_raster(f_legend, scale=1.4)
                        symbols_found = extract_symbols_and_text_from_legend(img_leg)
                        if not symbols_found:
                            st.warning("⚠️ לא אותרו סמלים מוגדרים במקרא. עובר לסריקה עצמאית בתוכנית...")
                            f_legend = None
                        else:
                            progress_bar = st.progress(0, text="סורק ומפענח לפי המקרא...")
                            total_syms = len(symbols_found)
                            for i, sym in enumerate(symbols_found):
                                progress_bar.progress((i + 1) / total_syms, text=f"סורק סמל {i+1} מתוך {total_syms}...")
                                matches = match_symbol_ai(
                                    plan_inv, 
                                    sym["crop_gray"], 
                                    min_thresh=(min_sens / 100.0),
                                    high_thresh=(high_sens / 100.0)
                                )
                                all_results.append({
                                    "index": i + 1,
                                    "symbol_img": sym["crop_color"],
                                    "image_uri": img_to_data_uri(sym["crop_color"]),
                                    "matches": matches
                                })
                            progress_bar.empty()
                    
                    if not f_legend:
                        progress_bar = st.progress(0, text="מאתר ומקבץ סמלים עצמאית...")
                        clusters = auto_discover_plan_symbols(img_plan[:active_h, :], match_thresh=(high_sens / 100.0))
                        for i, cl in enumerate(clusters):
                            matches = []
                            for item in cl["items"]:
                                matches.append({
                                    "bbox": item["bbox"],
                                    "center": item["center"],
                                    "score": 0.85,
                                    "status": "Green"
                                })
                            all_results.append({
                                "index": i + 1,
                                "symbol_img": cl["rep_color"],
                                "image_uri": img_to_data_uri(cl["rep_color"]),
                                "matches": matches
                            })
                        progress_bar.empty()
                        
                    st.session_state["legend_results"] = all_results
                    st.session_state["raw_plan_img"] = img_plan

            if "legend_results" in st.session_state:
                res = st.session_state["legend_results"]
                raw_plan = st.session_state["raw_plan_img"]
                disp_plan = raw_plan.copy()
                
                if filter_banner:
                    h_cut = int(raw_plan.shape[0] * 0.82)
                    cv2.line(disp_plan, (0, h_cut), (raw_plan.shape[1], h_cut), (180, 180, 180), 2)
                    cv2.putText(disp_plan, "Filtered Title Block Area", (20, h_cut + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 2)
                
                yellow_items = []
                for s_idx, item in enumerate(res):
                    for m_idx, m in enumerate(item["matches"]):
                        if m["status"] == "Yellow":
                            yellow_items.append((s_idx, m_idx, item, m))
                
                yellow_items = yellow_items[:8]
                is_done_verifying = st.session_state.get("verification_completed", False)
                
                if yellow_items and not is_done_verifying:
                    st.markdown("---")
                    st.info(f"🔍 **אימות נקודות בספק ולמידת AI ({len(yellow_items)} נקודות לבדיקה):**")
                    with st.expander("לחץ כאן לבדיקה ואישור/דחיית נקודות", expanded=True):
                        cols = st.columns(min(len(yellow_items), 3))
                        updated_memory = False
                        for y_i, (s_idx, m_idx, item, m) in enumerate(yellow_items):
                            with cols[y_i % len(cols)]:
                                x, y, w, h = m["bbox"]
                                pad = 24
                                y1, y2 = max(0, y - pad), min(raw_plan.shape[0], y + h + pad)
                                x1, x2 = max(0, x - pad), min(raw_plan.shape[1], x + w + pad)
                                crop_zoom = raw_plan[y1:y2, x1:x2].copy()
                                cv2.rectangle(crop_zoom, (x - x1, y - y1), (x - x1 + w, y - y1 + h), (0, 165, 255), 2)
                                
                                st.image(cv2.cvtColor(crop_zoom, cv2.COLOR_BGR2RGB), caption=f"סמל #{item['index']} (ודאות {m['score']*100:.0f}%)", width=140)
                                choice = st.radio(
                                    "סטטוס:",
                                    ["✅ אשר נקודה (V)", "❌ דחה נקודה (X)"],
                                    key=f"status_choice_{active_disc}_{s_idx}_{m_idx}",
                                    horizontal=True
                                )
                                is_appr = ("אשר" in choice)
                                m["user_decision"] = "Approved" if is_appr else "Rejected"
                                
                                pattern_key = f"s_{item['index']}_{m['bbox'][2]}x{m['bbox'][3]}"
                                if is_appr and pattern_key not in ai_memory["approved_patterns"]:
                                    ai_memory["approved_patterns"].append(pattern_key)
                                    updated_memory = True
                                elif not is_appr and pattern_key not in ai_memory["rejected_patterns"]:
                                    ai_memory["rejected_patterns"].append(pattern_key)
                                    updated_memory = True
                                st.markdown("---")
                                
                        if updated_memory:
                            save_ai_memory(ai_memory)
                            
                        if st.button("✨ סיימתי לסמן - נעל סימונים ונקה תצוגה"):
                            st.session_state["verification_completed"] = True
                            st.rerun()
                
                # חישוב כמויות וציור שרטוט
                boq_rows = []
                for s_idx, item in enumerate(res):
                    confirmed_count = 0
                    for m_idx, m in enumerate(item["matches"]):
                        x, y, w, h = m["bbox"]
                        is_green = (m["status"] == "Green")
                        user_dec = m.get("user_decision", "Pending")
                        
                        if is_green or user_dec == "Approved":
                            confirmed_count += 1
                            cv2.rectangle(disp_plan, (x, y), (x + w, y + h), (0, 200, 0), 3)
                            cv2.putText(disp_plan, f"#{item['index']} V", (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                        elif user_dec == "Rejected":
                            cv2.line(disp_plan, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.line(disp_plan, (x + w, y), (x, y + h), (0, 0, 255), 2)
                            cv2.putText(disp_plan, "X", (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.rectangle(disp_plan, (x, y), (x + w, y + h), (0, 165, 255), 2)
                            cv2.putText(disp_plan, "?", (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                            
                    item["confirmed_count"] = confirmed_count
                    
                    if confirmed_count > 0:
                        boq_rows.append({
                            "מס'": item["index"],
                            "תמונת סמל": item["image_uri"],
                            "symbol_img": item["symbol_img"],
                            "image_uri": item["image_uri"],
                            "תיאור הפריט": f"סמל {active_disc} #{item['index']}",
                            "כמות מאושרת": confirmed_count,
                            "יחידת מידה": "יח'"
                        })

                # שמירה לזיכרון הפרויקט המצטבר
                st.session_state["project_boq"][active_disc] = boq_rows

                st.markdown("---")
                st.subheader(f"📊 ריכוז סופי לכתב כמויות ({active_disc})")
                
                if boq_rows:
                    df_preview = pd.DataFrame([
                        {
                            "מס'": r["מס'"],
                            "תמונת סמל": r["תמונת סמל"],
                            "תיאור הפריט": r["תיאור הפריט"],
                            "כמות מאושרת": r["כמות מאושרת"],
                            "יחידת מידה": r["יחידת מידה"]
                        }
                        for r in boq_rows
                    ])
                    
                    st.dataframe(
                        df_preview,
                        column_config={
                            "תמונת סמל": st.column_config.ImageColumn("סמל גרפי", width="small")
                        }
                    )
                    
                    st.subheader("📥 ייצוא כתב כמויות (דיסציפלינה זו בלבד)")
                    exp_c1, exp_c2 = st.columns(2)
                    html_report = generate_export_html(boq_rows, title=f"כתב כמויות - {active_disc}")
                    
                    with exp_c1:
                        st.download_button(
                            "📊 הורד כתב כמויות ל-Excel (XLS)",
                            data=html_report.encode("utf-8"),
                            file_name=f"Takeoff_{active_disc}.xls",
                            mime="application/vnd.ms-excel",
                            key=f"dl_single_xls_{active_disc}"
                        )
                    with exp_c2:
                        st.download_button(
                            "📄 הורד דוח PDF מעוצב להדפסה (HTML/PDF)",
                            data=html_report.encode("utf-8"),
                            file_name=f"Report_{active_disc}.html",
                            mime="text/html",
                            key=f"dl_single_pdf_{active_disc}"
                        )
                else:
                    st.warning("לא אותרו כמויות מאושרות לדיסציפלינה זו.")

                st.subheader("🗺️ תוכנית עם סימוני הפריטים המאושרים:")
                st.image(cv2.cvtColor(disp_plan, cv2.COLOR_BGR2RGB))
                
                # --- שלב ההחלטה: סיום פרויקט או מעבר הלאה ---
                st.markdown("---")
                st.success(f"🎉 חישוב {active_disc} נשמר בהצלחה בפרויקט!")
                st.markdown("### ❓ מה ברצונך לעשות כעת?")
                
                step_c1, step_c2 = st.columns(2)
                with step_c1:
                    if st.button("🏁 סיום החישוב והפקת דוחות סופיים לפרויקט", key="btn_finish_all"):
                        st.session_state["show_master_export"] = True
                        st.rerun()
                with step_c2:
                    st.write("**או המשך לחישוב הבא:**")
                    n1, n2, n3 = st.columns(3)
                    remaining = [d for d in disciplines_list if d != active_disc]
                    with n1:
                        if st.button(remaining[0], key="btn_next_1"):
                            set_discipline_programmatically(remaining[0])
                    with n2:
                        if st.button(remaining[1], key="btn_next_2"):
                            set_discipline_programmatically(remaining[1])
                    with n3:
                        if st.button(remaining[2], key="btn_next_3"):
                            set_discipline_programmatically(remaining[2])

        else:
            st.info("💡 אנא העלה שרטוט תוכנית (וקובץ מקרא אם קיים) להפעלת הסריקה.")

# ========================================================
# 📐 CAD וקטורי (DXF)
# ========================================================
else:
    scale_val = 1.0
    if mode == "ספירה מתוכנית בודדת":
        cad_file = st.file_uploader("העלה שרטוט CAD (DXF):", type=["dxf"])
        if cad_file and HAS_VECTOR_ENGINE:
            parser = DXFVectorParser(cad_file, unit_scale_to_meter=scale_val)
            layers = [l["name"] for l in parser.get_layers_summary() if l["entity_count"] > 0]
            if active_disc == "⚡ חשמל ומאור":
                st.subheader("⚡ ספירת סמלי חשמל מבוססת בלוקים (100% דיוק וקטורי)")
                sel_layers = st.multiselect("שכבות חשמל:", layers, default=layers[:3] if layers else [])
                blocks = parser.extract_blocks(sel_layers)
                if blocks:
                    df = pd.DataFrame(blocks)
                    summary = df.groupby(["name", "cardinal_rotation"]).size().reset_index(name="כמות")
                    st.dataframe(summary)
                    df["אושר"] = True
                    edited = st.data_editor(df[["name", "layer", "x", "y", "rotation_deg", "cardinal_rotation", "אושר"]])
                    csv_out = summary.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 ייצא ל-Excel (CSV)", data=csv_out, file_name="Electrical_BOQ.csv", mime="text/csv")
