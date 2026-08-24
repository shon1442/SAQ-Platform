import streamlit as st
import os
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pypdfium2 as pdfium
import shapely.geometry as sg
from saq_vector_engine import DXFVectorParser, compare_vector_delta

LOGO_PATH = "logo.png.png" if os.path.exists("logo.png.png") else "logo.png"
has_logo = os.path.exists(LOGO_PATH)
app_icon = Image.open(LOGO_PATH) if has_logo else "📐"

st.set_page_config(page_title="S.A.Q - Takeoff & Vector CAD Platform", layout="wide", page_icon=app_icon)

def load_raster(file):
    """טעינת PDF/תמונה לרזולוציית עבודה הנדסית"""
    if file.name.lower().endswith(".pdf"):
        pdf = pdfium.PdfDocument(file.read())
        bitmap = pdf.get_page(0).render(scale=2.0)
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

def extract_symbols_from_legend(legend_img, min_size=16, max_size=120):
    """חילוץ ופירוק אוטומטי של כל הסמלים הבודדים מתוך עמוד המקרא"""
    gray = cv2.cvtColor(legend_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # חיפוש רכיבים גרפיים במקרא
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_symbols = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # סינון לפי ממדי סמל הנדסי טיפוסי (מונע לכידת קווי טבלה או אותיות בודדות)
        if min_size <= w <= max_size and min_size <= h <= max_size:
            aspect = w / float(h)
            if 0.45 <= aspect <= 2.2:
                pad = 3
                y1, y2 = max(0, y - pad), min(legend_img.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(legend_img.shape[1], x + w + pad)
                crop_color = legend_img[y1:y2, x1:x2]
                crop_gray = gray[y1:y2, x1:x2]
                raw_symbols.append({
                    "bbox": (x, y, w, h),
                    "crop_color": crop_color,
                    "crop_gray": crop_gray,
                    "y_pos": y
                })
                
    # סידור הסמלים לפי סדר הופעתם מלמעלה למטה במקרא
    raw_symbols.sort(key=lambda s: s["y_pos"])
    
    # ניקוי כפילויות סמוכות
    unique_symbols = []
    for sym in raw_symbols:
        is_dup = False
        for u in unique_symbols:
            if abs(sym["y_pos"] - u["y_pos"]) < 10 and abs(sym["bbox"][0] - u["bbox"][0]) < 10:
                is_dup = True
                break
        if not is_dup:
            unique_symbols.append(sym)
            
    return unique_symbols

def match_symbol_on_plan(plan_gray, plan_edges, templ_gray, threshold=0.38):
    """סריקת סמל מקרא בתוכנית ב-5 קני מידה ו-4 זוויות סיבוב"""
    _, thresh_t = cv2.threshold(templ_gray, 230, 255, cv2.THRESH_BINARY_INV)
    pts = cv2.findNonZero(thresh_t)
    if pts is not None:
        tx, ty, tw, th = cv2.boundingRect(pts)
        if tw > 8 and th > 8:
            templ_gray = templ_gray[ty:ty+th, tx:tx+tw]
            
    templ_edges = cv2.Canny(templ_gray, 50, 150)
    detections = []
    
    scales = [0.6, 0.8, 1.0, 1.2, 1.4]
    for scale in scales:
        sw = int(templ_edges.shape[1] * scale)
        sh = int(templ_edges.shape[0] * scale)
        if sw >= plan_edges.shape[1] or sh >= plan_edges.shape[0] or sw < 8 or sh < 8:
            continue
        resized_t = cv2.resize(templ_edges, (sw, sh))
        
        for rot in [0, 90, 180, 270]:
            if rot == 90: r_t = cv2.rotate(resized_t, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180: r_t = cv2.rotate(resized_t, cv2.ROTATE_180)
            elif rot == 270: r_t = cv2.rotate(resized_t, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else: r_t = resized_t
            
            rw, rh = r_t.shape[::-1]
            res = cv2.matchTemplate(plan_edges, r_t, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                detections.append({
                    "bbox": (int(pt[0]), int(pt[1]), int(rw), int(rh)),
                    "center": (int(pt[0] + rw // 2), int(pt[1] + rh // 2)),
                    "score": float(res[pt[1], pt[0]])
                })
                
    if not detections:
        return []
        
    boxes = [list(d["bbox"]) for d in detections]
    scores = [d["score"] for d in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=threshold, nms_threshold=0.25)
    return [detections[i] for i in indices.flatten()] if len(indices) > 0 else []

with st.sidebar:
    if has_logo:
        st.image(LOGO_PATH, use_container_width=True)
    st.header("⚙️ הגדרות עבודה")
    file_type = st.radio("פורמט שרטוט:", ["📄 PDF / תמונה (Raster)", "📐 CAD וקטורי (DXF)"])
    mode = st.radio("מצב פעולה:", ["ספירה מתוכנית בודדת", "השוואת שינויים (Delta)"])
    discipline = st.selectbox("דיסציפלינה:", ["⚡ חשמל ומאור", "🧱 בניה (מחיצות ומעטפת)", "🚿 אינסטלציה", "📐 ריצוף וחיפוי"])
    st.markdown("---")
    st.subheader("📏 רגישות התאמה")
    scan_sens = st.slider("רגישות סריקה (%):", min_value=20, max_value=80, value=36, step=2)
    thresh_val = scan_sens / 100.0

col_l, col_t = st.columns([1, 6])
with col_l:
    if has_logo:
        st.image(LOGO_PATH, width=90)
with col_t:
    st.title("S.A.Q Takeoff & Delta Platform")
    st.caption("פלטפורמת ענן לפענוח הנדסי, ספירת כמויות והשוואת שרטוטים אוטומטית")

# ========================================================
# 📄 נתיב PDF ורסטר (Legend-to-Plan Matcher)
# ========================================================
if file_type == "📄 PDF / תמונה (Raster)":
    if mode == "ספירה מתוכנית בודדת":
        c_p, c_l = st.columns(2)
        with c_p:
            f_plan = st.file_uploader("1️⃣ העלה שרטוט תוכנית (PDF / תמונה):", type=["pdf", "png", "jpg", "jpeg"], key="plan_in")
        with c_l:
            f_legend = st.file_uploader("2️⃣ העלה קובץ מקרא (PDF / תמונה):", type=["pdf", "png", "jpg", "jpeg"], key="leg_in")
            
        if f_plan and f_legend:
            if st.button("🚀 הפעל פענוח מקרא וספירה אוטומטית בתוכנית"):
                with st.spinner("מחלץ סמלים מהמקרא ומבצע התאמות על כלל התוכנית..."):
                    img_plan = load_raster(f_plan)
                    img_leg = load_raster(f_legend)
                    
                    # 1. חילוץ סמלים מהמקרא
                    symbols_found = extract_symbols_from_legend(img_leg)
                    
                    if not symbols_found:
                        st.error("לא אותרו סמלים מוגדרים בקובץ המקרא. ודא שהקובץ ברור.")
                    else:
                        plan_gray = cv2.cvtColor(img_plan, cv2.COLOR_BGR2GRAY)
                        plan_edges = cv2.Canny(plan_gray, 50, 150)
                        
                        all_results = []
                        disp_plan = img_plan.copy()
                        
                        # צבעים לסימון סמלים שונים
                        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                        
                        for i, sym in enumerate(symbols_found):
                            matches = match_symbol_on_plan(plan_gray, plan_edges, sym["crop_gray"], threshold=thresh_val)
                            count = len(matches)
                            
                            color = colors[i % len(colors)]
                            for m in matches:
                                x, y, w, h = m["bbox"]
                                cv2.rectangle(disp_plan, (x, y), (x + w, y + h), color, 3)
                                cv2.putText(disp_plan, f"S{i+1}", (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                            all_results.append({
                                "index": i + 1,
                                "symbol_img": sym["crop_color"],
                                "count": count,
                                "matches": matches
                            })
                            
                        st.session_state["legend_results"] = all_results
                        st.session_state["disp_plan"] = disp_plan

            if "legend_results" in st.session_state:
                res = st.session_state["legend_results"]
                st.success(f"הסריקה הושלמה! פוענחו {len(res)} סוגי סמלים מתוך המקרא.")
                
                st.subheader("📋 פירוט ספירת כמויות לפי סמלי מקרא")
                boq_rows = []
                
                for item in res:
                    c1, c2, c3 = st.columns([1, 2, 2])
                    with c1:
                        if item["symbol_img"].size > 0:
                            st.image(cv2.cvtColor(item["symbol_img"], cv2.COLOR_BGR2RGB), width=65, caption=f"סמל S{item['index']}")
                    with c2:
                        s_desc = st.text_input(f"תיאור סמל S{item['index']}:", value=f"סמל חשמל S{item['index']}", key=f"desc_{item['index']}")
                        is_inc = st.checkbox("כלול בכתב כמויות", value=(item["count"] > 0), key=f"inc_leg_{item['index']}")
                    with c3:
                        st.metric("כמות שנספרה:", f"{item['count']} יח'")
                    
                    if is_inc:
                        boq_rows.append({"מס'": item["index"], "תיאור הפריט": s_desc, "כמות": item["count"], "יחידת מידה": "יח'"})
                    st.markdown("---")
                    
                st.subheader("🗺️ תוכנית עם מיקומי כל הסמלים שנספרו:")
                st.image(cv2.cvtColor(st.session_state["disp_plan"], cv2.COLOR_BGR2RGB), use_container_width=True)
                
                if boq_rows:
                    df_boq = pd.DataFrame(boq_rows)
                    st.subheader("📊 ריכוז סופי לכתב כמויות")
                    st.dataframe(df_boq, use_container_width=True)
                    
                    out_io = io.BytesIO()
                    with pd.ExcelWriter(out_io, engine="openpyxl") as writer:
                        df_boq.to_excel(writer, index=False, sheet_name="כתב כמויות חשמל")
                    st.download_button("📥 ייצא כתב כמויות ל-Excel", data=out_io.getvalue(), file_name="Legend_Takeoff_BOQ.xlsx")
        else:
            st.info("💡 אנא העלה את קובץ התוכנית ואת קובץ המקרא להפעלת הסריקה המשולבת.")

# ========================================================
# 📐 נתיב CAD וקטורי (DXF)
# ========================================================
else:
    scale_val = 1.0
    if mode == "ספירה מתוכנית בודדת":
        cad_file = st.file_uploader("העלה שרטוט CAD (DXF):", type=["dxf"])
        if cad_file:
            parser = DXFVectorParser(cad_file, unit_scale_to_meter=scale_val)
            layers = [l["name"] for l in parser.get_layers_summary() if l["entity_count"] > 0]
            if discipline == "⚡ חשמל ומאור":
                st.subheader("⚡ ספירת סמלי חשמל מבוססת בלוקים")
                sel_layers = st.multiselect("שכבות חשמל:", layers, default=layers[:3] if layers else [])
                blocks = parser.extract_blocks(sel_layers)
                if blocks:
                    df = pd.DataFrame(blocks)
                    summary = df.groupby(["name", "cardinal_rotation"]).size().reset_index(name="כמות")
                    st.dataframe(summary, use_container_width=True)
                    df["אושר"] = True
                    edited = st.data_editor(df[["name", "layer", "x", "y", "rotation_deg", "cardinal_rotation", "אושר"]], use_container_width=True)
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine="openpyxl") as w:
                        summary.to_excel(w, sheet_name="ריכוז", index=False)
                        edited.to_excel(w, sheet_name="פירוט", index=False)
                    st.download_button("📥 ייצא ל-Excel", data=out.getvalue(), file_name="Electrical_BOQ.xlsx")
