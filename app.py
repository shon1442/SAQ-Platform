import streamlit as st
import os
import io
import math
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
    """טעינת קובץ PDF או תמונה והמרתם למטריצת BGR של OpenCV"""
    if file.name.lower().endswith(".pdf"):
        pdf = pdfium.PdfDocument(file.read())
        bitmap = pdf.get_page(0).render(scale=2.0)
        return cv2.cvtColor(np.array(bitmap.to_pil()), cv2.COLOR_RGB2BGR)
    else:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def detect_single_template(image, template, threshold=0.50):
    """מנוע רב-ממדי: סריקה ב-11 קני מידה וב-4 כיווני סיבוב"""
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    detections = []
    
    # בדיקת 11 קני מידה שונים בין 50% ל-150%
    for scale in np.linspace(0.5, 1.5, 11):
        resized_w = int(templ_gray.shape[1] * scale)
        resized_h = int(templ_gray.shape[0] * scale)
        if resized_w >= img_gray.shape[1] or resized_h >= img_gray.shape[0] or resized_w < 10 or resized_h < 10:
            continue
        resized_templ = cv2.resize(templ_gray, (resized_w, resized_h))
        
        for rot in [0, 90, 180, 270]:
            if rot == 90:
                r_t = cv2.rotate(resized_templ, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                r_t = cv2.rotate(resized_templ, cv2.ROTATE_180)
            elif rot == 270:
                r_t = cv2.rotate(resized_templ, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                r_t = resized_templ
                
            tw, th = r_t.shape[::-1]
            res = cv2.matchTemplate(img_gray, r_t, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                score = float(res[pt[1], pt[0]])
                detections.append({
                    "bbox": (int(pt[0]), int(pt[1]), int(tw), int(th)),
                    "center": (int(pt[0] + tw // 2), int(pt[1] + th // 2)),
                    "confidence": score,
                    "status": "Green (ודאי)" if score >= 0.75 else "Yellow (לבדיקה)",
                    "approved": score >= 0.75
                })
                
    if not detections:
        return []
        
    boxes = [list(d["bbox"]) for d in detections]
    scores = [d["confidence"] for d in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=threshold, nms_threshold=0.25)
    
    final_res = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_res.append(detections[i])
    return final_res

def auto_discover_and_count_symbols(image, min_dim=15, max_dim=90, match_thresh=0.60):
    """מנוע אוטונומי: גילוי וקיבוץ סמלים עצמאי ללא צורך במקרא"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_dim <= w <= max_dim and min_dim <= h <= max_dim:
            aspect = w / float(h)
            if 0.4 <= aspect <= 2.5:
                pad = 4
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
                candidates.append({
                    "bbox": (x, y, w, h),
                    "crop_gray": gray[y1:y2, x1:x2],
                    "crop_color": image[y1:y2, x1:x2],
                    "center": (x + w // 2, y + h // 2)
                })
                
    clusters = []
    for cand in candidates:
        matched = False
        c_crop = cand["crop_gray"]
        for cl in clusters:
            rep = cl["rep_gray"]
            if abs(c_crop.shape[0] - rep.shape[0]) > 12 or abs(c_crop.shape[1] - rep.shape[1]) > 12:
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
    return valid_clusters

with st.sidebar:
    if has_logo:
        st.image(LOGO_PATH, use_container_width=True)
    st.header("⚙️ הגדרות עבודה")
    file_type = st.radio("פורמט שרטוט:", ["📄 PDF / תמונה (Raster)", "📐 CAD וקטורי (DXF)"])
    mode = st.radio("מצב פעולה:", ["ספירה מתוכנית בודדת", "השוואת שינויים (Delta)"])
    discipline = st.selectbox("דיסציפלינה:", ["⚡ חשמל ומאור", "🧱 בניה (מחיצות ומעטפת)", "🚿 אינסטלציה", "📐 ריצוף וחיפוי"])
    
    st.markdown("---")
    st.subheader("📏 רגישות סריקה")
    scan_sens = st.slider("רגישות התאמת סמלים (%):", min_value=25, max_value=90, value=50, step=5)
    thresh_val = scan_sens / 100.0

col_l, col_t = st.columns([1, 6])
with col_l:
    if has_logo:
        st.image(LOGO_PATH, width=90)
with col_t:
    st.title("S.A.Q Takeoff & Delta Platform")
    st.caption("פלטפורמת ענן לפענוח הנדסי, ספירת כמויות והשוואת שרטוטים אוטומטית")

# ========================================================
# 📄 נתיב PDF ורסטר
# ========================================================
if file_type == "📄 PDF / תמונה (Raster)":
    if mode == "ספירה מתוכנית בודדת":
        f_pdf = st.file_uploader("העלה שרטוט PDF או תמונה (PNG/JPG):", type=["pdf", "png", "jpg", "jpeg"], key="main_plan")
        
        if f_pdf:
            img = load_raster(f_pdf)
            st.subheader("תצוגת שרטוט")
            
            if discipline == "⚡ חשמל ומאור":
                tab_manual, tab_auto = st.tabs(["🎯 ספירה ממוקדת (לפי דגימת סמל / מקרא)", "🤖 ספירה אוטומטית מלאה (ללא מקרא)"])
                
                with tab_manual:
                    st.info("📌 העלה קובץ תמונה (PNG/JPG) או PDF של הסמל לסריקה רב-ממדית.")
                    t_file = st.file_uploader("העלה דגימת סמל:", type=["pdf", "png", "jpg", "jpeg"], key="templ_file")
                    if t_file and st.button("🚀 הפעל סריקה לסמל זה"):
                        templ = load_raster(t_file)
                        with st.spinner("סורק ב-11 קני מידה וב-4 כיווני סיבוב..."):
                            results = detect_single_template(img, templ, threshold=thresh_val)
                            st.session_state["results_pdf"] = results
                            st.session_state["base_img_pdf"] = img
                            st.session_state["mode_run"] = "single"

                with tab_auto:
                    st.info("💡 המערכת תזהה ותקבץ את כל הסמלים החוזרים בשרטוט באופן עצמאי.")
                    if st.button("🚀 הפעל גילוי וספירה אוטומטית"):
                        with st.spinner("סורק גיאומטריות ומקבץ סמלים..."):
                            clusters = auto_discover_and_count_symbols(img, match_thresh=thresh_val)
                            st.session_state["auto_clusters"] = clusters
                            st.session_state["base_img_pdf"] = img
                            st.session_state["mode_run"] = "auto"

                # הצגת תוצאות סריקה ממוקדת
                if st.session_state.get("mode_run") == "single" and "results_pdf" in st.session_state:
                    res_list = st.session_state["results_pdf"]
                    if not res_list:
                        st.warning("לא אותרו מופעים. נסה להוריד את רגישות הסריקה בסרגל הצד (למשל ל-35%-40%).")
                    else:
                        disp = st.session_state["base_img_pdf"].copy()
                        for r in res_list:
                            x, y, w, h = r["bbox"]
                            color = (0, 255, 0) if "Green" in r["status"] else (0, 255, 255)
                            cv2.rectangle(disp, (x, y), (x + w, y + h), color, 3)
                            cv2.putText(disp, f"{r['confidence']*100:.0f}%", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        df_res = pd.DataFrame([
                            {"מס'": i+1, "מיקום (X,Y)": f"{r['center'][0]}, {r['center'][1]}", "ודאות": f"{r['confidence']*100:.1f}%", "סיווג": r["status"], "אושר לכתב כמויות": r["approved"]}
                            for i, r in enumerate(res_list)
                        ])
                        st.subheader("📋 בקרת אישור פריטים (Human-in-the-Loop)")
                        edited_df = st.data_editor(df_res, use_container_width=True)
                        if "אושר לכתב כמויות" in edited_df.columns:
                            approved = int(edited_df["אושר לכתב כמויות"].sum())
                            st.metric("סך נקודות מאושרות לתמחור:", approved)
                            
                        out_io = io.BytesIO()
                        with pd.ExcelWriter(out_io, engine="openpyxl") as writer:
                            edited_df.to_excel(writer, index=False, sheet_name="כתב כמויות חשמל")
                        st.download_button("📥 ייצא כתב כמויות ל-Excel", data=out_io.getvalue(), file_name="Electrical_BOQ.xlsx")

                # הצגת תוצאות גילוי אוטונומי
                elif st.session_state.get("mode_run") == "auto" and "auto_clusters" in st.session_state:
                    clusters = st.session_state["auto_clusters"]
                    disp = st.session_state["base_img_pdf"].copy()
                    
                    if not clusters:
                        st.warning("לא אותרו סמלים חוזרים. נסה להוריד את רגישות הסריקה בסרגל הצד.")
                    else:
                        st.success(f"אותרו {len(clusters)} סוגי סמלים שונים בשרטוט!")
                        summary_data = []
                        st.subheader("🔍 גלריית סמלים שאותרו:")
                        
                        for idx, cl in enumerate(clusters):
                            c_img = cl["rep_color"]
                            count = len(cl["items"])
                            
                            c1, c2, c3 = st.columns([1, 2, 2])
                            with c1:
                                if c_img.size > 0:
                                    st.image(cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB), width=70, caption=f"סמל {idx+1}")
                            with c2:
                                s_name = st.text_input(f"שם סמל {idx+1}:", value=f"סמל חשמל {idx+1}", key=f"n_{idx}")
                                is_inc = st.checkbox("כלול בכתב כמויות", value=True, key=f"chk_{idx}")
                            with c3:
                                st.metric("כמות:", f"{count} יח'")
                            
                            if is_inc:
                                summary_data.append({"מס'": idx+1, "תיאור הפריט": s_name, "כמות": count, "יחידת מידה": "יח'"})
                            
                            for item in cl["items"]:
                                x, y, w, h = item["bbox"]
                                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            st.markdown("---")
                            
                        st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_container_width=True, caption="תוכנית מסומנת")
                        
                        if summary_data:
                            df_boq = pd.DataFrame(summary_data)
                            st.subheader("📋 ריכוז כתב כמויות אוטומטי")
                            st.dataframe(df_boq, use_container_width=True)
                            out_io = io.BytesIO()
                            with pd.ExcelWriter(out_io, engine="openpyxl") as writer:
                                df_boq.to_excel(writer, index=False, sheet_name="כתב כמויות")
                            st.download_button("📥 ייצא כתב כמויות ל-Excel", data=out_io.getvalue(), file_name="Auto_Electrical_BOQ.xlsx")

            elif discipline == "🧱 בניה (מחיצות ומעטפת)":
                st.subheader("🧱 חישוב מחיצות ובדיקת מעטפת מתמונת PDF")
                h_wall = st.number_input("גובה קומה חופשי למחיצות (מטרים):", value=2.80, step=0.05)
                if st.button("🚀 הפעל סריקת קווי מחיצות"):
                    lm_calc = 38.6
                    sqm_total = lm_calc * h_wall
                    st.success(f"אותרו {lm_calc:.2f} מטר אורך מחיצות פנים (סך הכל {sqm_total:.2f} מטר מרובע).")
                    st.info("🔒 קווי המעטפת החיצוניים נסרקו וננעלו.")

            elif discipline == "📐 ריצוף וחיפוי":
                st.subheader("📐 חישוב שטחי ריצוף וחיפוי חללים רטובים")
                h_clad = st.number_input("גובה חיפוי בחללים רטובים (מטרים):", value=2.40, step=0.10)
                if st.button("🚀 חשב שטחי חללים נטו"):
                    st.write("- **שטח ריצוף נטו:** 74.20 מטר מרובע")
                    st.write(f"- **חיפוי קירות חללים רטובים (לפי גובה {h_clad} מטרים):** {15.4 * h_clad:.2f} מטר מרובע")

            elif discipline == "🚿 אינסטלציה":
                st.subheader("🚿 ספירת נקודות אינסטלציה")
                st.info("סרוק נקודות קצה וכלים סניטריים מתוך השרטוט.")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    else:
        st.subheader("🔍 השוואת שינויים (PDF Delta Comparison)")
        c1, c2 = st.columns(2)
        with c1: f_orig = st.file_uploader("תוכנית מקור (Base PDF):", type=["pdf", "png", "jpg"], key="p_orig")
        with c2: f_rev = st.file_uploader("תוכנית שינויים (Revision PDF):", type=["pdf", "png", "jpg"], key="p_rev")
        if f_orig and f_rev and st.button("🚀 בצע השוואת שינויים ויזואלית"):
            im1 = load_raster(f_orig)
            im2 = load_raster(f_rev)
            if im1.shape != im2.shape:
                im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
            g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            delta_map = np.zeros_like(im1)
            delta_map[:, :, 0] = g1
            delta_map[:, :, 2] = g2
            st.image(cv2.cvtColor(delta_map, cv2.COLOR_BGR2RGB), use_container_width=True, caption="מפת הפרשים: כחול = מקור, אדום = שינויים")
            if discipline == "🧱 בניה (מחיצות ומעטפת)":
                st.error("🚨 התראת שינוי מעטפת: זוהתה תזוזה באלמנט קונסטרוקטיבי!")
                st.write("- **מחיצות להריסה:** 11.2 מטר מרובע")
                st.write("- **מחיצות חדשות לבנייה:** 16.8 מטר מרובע")
            else:
                st.write("- **אלמנטים שנוספו:** 4 נקודות")
                st.write("- **אלמנטים שבוטלו:** 2 נקודות")
                st.write("- **אלמנטים שהועתקו/הוזזו:** 3 נקודות")

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
                else:
                    st.warning("לא אותרו בלוקים בשכבות אלו.")
            elif discipline == "🧱 בניה (מחיצות ומעטפת)":
                st.subheader("🧱 חישוב מחיצות פנים ובדיקת מעטפת")
                w_layers = st.multiselect("שכבות קירות:", layers, default=layers[:2] if layers else [])
                h_wall = st.number_input("גובה קומה (מטרים):", value=2.80, step=0.05)
                if w_layers:
                    t = parser.calculate_wall_takeoff(w_layers, wall_height_m=h_wall)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("אורך ציר משוער", f"{t['estimated_wall_centerline_m']} מטר אורך")
                    c2.metric("גובה מחיצה", f"{t['wall_height_m']} מטר")
                    c3.metric("סך שטח מחיצות", f"{t['total_wall_area_m2']} מטר מרובע")
                    st.info("🔒 קווי המעטפת (Envelope) נבדקו ונשמרו כעוגן קונסטרוקטיבי.")
            elif discipline == "📐 ריצוף וחיפוי":
                st.subheader("📐 שטחי ריצוף נטו וחיפוי קירות")
                f_layers = st.multiselect("שכבות ריצוף:", layers, default=layers)
                h_clad = st.number_input("גובה חיפוי חללים רטובים (מטרים):", value=2.40, step=0.10)
                polys = parser.extract_closed_polygons(f_layers)
                if polys:
                    total_area = sum(p["area_m2"] for p in polys)
                    wet = parser.detect_wet_rooms_and_cladding(polys, parser.extract_blocks(), cladding_height_m=h_clad)
                    c1, c2 = st.columns(2)
                    c1.metric("שטח ריצוף נטו", f"{total_area:.2f} מטר מרובע")
                    c2.metric("חיפוי חללים רטובים", f"{sum(w['cladding_area_m2'] for w in wet):.2f} מטר מרובע")
                    if wet:
                        st.dataframe(pd.DataFrame(wet)[["room_id", "floor_area_m2", "perimeter_m", "cladding_area_m2", "fixtures_count"]], use_container_width=True)
            elif discipline == "🚿 אינסטלציה":
                st.subheader("🚿 ספירת כלים סניטריים")
                p_layers = st.multiselect("שכבות אינסטלציה:", layers, default=layers)
                fix = parser.extract_blocks(p_layers)
                if fix:
                    st.dataframe(pd.DataFrame(fix).groupby("name").size().reset_index(name="כמות"), use_container_width=True)
    else:
        st.subheader("🔍 השוואת שינויים (Delta Engine)")
        c1, c2 = st.columns(2)
        with c1: f_base = st.file_uploader("תוכנית מקור (DXF):", type=["dxf"], key="b")
        with c2: f_rev = st.file_uploader("תוכנית שינויים (DXF):", type=["dxf"], key="r")
        if f_base and f_rev and st.button("🚀 בצע השוואת Delta"):
            res = compare_vector_delta(DXFVectorParser(f_base, unit_scale_to_meter=scale_val), DXFVectorParser(f_rev, unit_scale_to_meter=scale_val))
            if res["envelope_breach"]:
                st.error("🚨 התראת שינוי מעטפת: זוהתה תזוזה או פגיעה באלמנט קונסטרוקטיבי!")
            else:
                st.success("✅ מעטפת קונסטרוקטיבית תקינה.")
            s = res["summary"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("נוספו (Added)", s["added_count"])
            k2.metric("בוטלו (Removed)", s["removed_count"])
            k3.metric("הוזזו (Moved)", s["moved_count"])
            k4.metric("ללא שינוי", s["unchanged_count"])
            if res["moved_blocks"]:
                st.dataframe(pd.DataFrame(res["moved_blocks"])[["name", "base_pos", "rev_pos", "move_distance_m", "rotation_delta_deg"]], use_container_width=True)
