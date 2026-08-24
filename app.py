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
    leg_h, leg_w = gray.shape[:2]
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
        if 16 <= w <= 110 and 16 <= h <= 110 and area > 45:
            aspect = w / float(h)
            if 0.45 <= aspect <= 2.2:
                pad = 4
                y1, y2 = max(0, y - pad), min(work_gray.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(work_gray.shape[1], x + w + pad)
                c_gray = work_gray[y1:y2, x1:x2]
                c_color = work_color[y1:y2, x1:x2]
                
                text_crop_x1 = max(0, x - 260) if x > 260 else min(work_gray.shape[1], x + w)
                text_crop_x2 = x if x > 260 else min(work_gray.shape[1], x + w + 260)
                text_crop = work_color[y1:y2, text_crop_x1:text_crop_x2]
                
                raw_symbols.append({
                    "bbox": (x, y, w, h),
                    "crop_color": c_color,
                    "crop_gray": c_gray,
                    "text_crop": text_crop,
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

def match_symbol_ai(plan_inv, templ_gray, min_thresh=0.64, high_thresh=0.76):
    _, templ_inv = cv2.threshold(templ_gray, 230, 255, cv2.THRESH_BINARY_INV)
    pts = cv2.findNonZero(templ_inv)
    if pts is not None:
        tx, ty, tw, th = cv2.boundingRect(pts)
        if tw > 8 and th > 8:
            templ_inv = templ_inv[ty:ty+th, tx:tx+tw]
            
    detections = []
    scales = [0.88, 0.95, 1.0, 1.05, 1.12]
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

def generate_html_boq(boq_rows):
    html = """
    <html dir="rtl">
    <head>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #fafafa; }
        table { width: 100%; border-collapse: collapse; background-color: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        th { background-color: #1F4E78; color: white; padding: 12px; font-size: 16px; border: 1px solid #ddd; }
        td { padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 15px; vertical-align: middle; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        img { border: 1px solid #ccc; background: white; padding: 3px; border-radius: 4px; }
    </style>
    </head>
    <body>
    <h2>📋 כתב כמויות חשמל - S.A.Q Takeoff</h2>
    <table>
        <tr>
            <th>מס'</th>
            <th>תמונת סמל</th>
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
            <td style="color: #1F4E78; font-size: 18px; font-weight: bold;">{r["כמות מאושרת"]}</td>
            <td>{r["יחידת מידה"]}</td>
        </tr>
        """
    html += "</table></body></html>"
    return html

ai_memory = load_ai_memory()

with st.sidebar:
    if has_logo:
        st.image(LOGO_PATH)
    st.header("⚙️ הגדרות עבודה")
    file_type = st.radio("פורמט שרטוט:", ["📄 PDF / תמונה (Raster)", "📐 CAD וקטורי (DXF)"])
    mode = st.radio("מצב פעולה:", ["ספירה מתוכנית בודדת", "השוואת שינויים (Delta)"])
    discipline = st.selectbox("דיסציפלינה:", ["⚡ חשמל ומאור", "🧱 בניה (מחיצות ומעטפת)", "🚿 אינסטלציה", "📐 ריצוף וחיפוי"])
    st.markdown("---")
    st.subheader("🧠 מנוע למידה AI וכיול")
    st.caption(f"זיכרון דפוסים פעיל: {len(ai_memory.get('approved_patterns', []))} אישורים שמורים.")
    high_sens = st.slider("סף ודאי אוטומטי (Green %):", min_value=65, max_value=85, value=75, step=1)
    min_sens = st.slider("סף שאלת משתמש (Yellow %):", min_value=50, max_value=68, value=62, step=1)
    filter_banner = st.checkbox("סנן טבלת כותרת (Title Block)", value=True)

col_l, col_t = st.columns([1, 6])
with col_l:
    if has_logo:
        st.image(LOGO_PATH, width=90)
with col_t:
    st.title("S.A.Q Takeoff & Delta Platform")
    st.caption("פלטפורמת ענן עם זיהוי שמות מקרא, תמונות סמלים וספירה אוטומטית")

if file_type == "📄 PDF / תמונה (Raster)":
    if mode == "ספירה מתוכנית בודדת":
        c_p, c_l = st.columns(2)
        with c_p:
            f_plan = st.file_uploader("1️⃣ העלה שרטוט תוכנית (PDF / תמונה):", type=["pdf", "png", "jpg", "jpeg"], key="plan_in")
        with c_l:
            f_legend = st.file_uploader("2️⃣ העלה קובץ מקרא (PDF / תמונה):", type=["pdf", "png", "jpg", "jpeg"], key="leg_in")
            
        if f_plan and f_legend:
            if st.button("🚀 הפעל פענוח מקרא וספירה מדויקת בתוכנית"):
                img_plan = load_raster(f_plan, scale=1.4)
                img_leg = load_raster(f_legend, scale=1.4)
                
                if img_plan is None or img_leg is None:
                    st.error("שגיאה בטעינת הקבצים. ודא שהקובץ תקין.")
                else:
                    symbols_found = extract_symbols_and_text_from_legend(img_leg)
                    if not symbols_found:
                        st.warning("⚠️ לא אותרו סמלים הנדסיים מבודדים במקרא.")
                    else:
                        progress_bar = st.progress(0, text="סורק ומפענח סמלי חשמל ותאורה...")
                        plan_gray = cv2.cvtColor(img_plan, cv2.COLOR_BGR2GRAY)
                        h_p, w_p = plan_gray.shape[:2]
                        active_h = int(h_p * 0.82) if filter_banner else h_p
                        plan_roi = plan_gray[:active_h, :]
                        _, plan_inv = cv2.threshold(plan_roi, 230, 255, cv2.THRESH_BINARY_INV)
                        
                        all_results = []
                        total_syms = len(symbols_found)
                        for i, sym in enumerate(symbols_found):
                            progress_bar.progress((i + 1) / total_syms, text=f"סורק סמל/תאורה {i+1} מתוך {total_syms}...")
                            matches = match_symbol_ai(
                                plan_inv, 
                                sym["crop_gray"], 
                                min_thresh=(min_sens / 100.0),
                                high_thresh=(high_sens / 100.0)
                            )
                            all_results.append({
                                "index": i + 1,
                                "symbol_img": sym["crop_color"],
                                "text_crop": sym["text_crop"],
                                "image_uri": img_to_data_uri(sym["crop_color"]),
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
                
                if yellow_items:
                    st.markdown("---")
                    st.info(f"🔍 **אימות נקודות בספק ולמידת AI ({len(yellow_items)} נקודות לבדיקה):**")
                    with st.expander("לחץ כאן לבדיקה ואישור/דחיית נקודות (V / X)", expanded=True):
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
                                    key=f"status_choice_{s_idx}_{m_idx}",
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

                st.markdown("---")
                st.subheader("📋 פירוט סמלים שנלמדו מהמקרא")
                for item in res:
                    c1, c2, c3 = st.columns([1.5, 2.5, 1.5])
                    with c1:
                        if item["symbol_img"].size > 0:
                            st.image(cv2.cvtColor(item["symbol_img"], cv2.COLOR_BGR2RGB), width=85, caption=f"סמל #{item['index']}")
                    with c2:
                        if item["text_crop"].size > 0:
                            st.caption("חיתוך תיאור מהמקרא:")
                            st.image(cv2.cvtColor(item["text_crop"], cv2.COLOR_BGR2RGB), width=200)
                        s_desc = st.text_input(f"תיאור פריט #{item['index']}:", value=f"סמל חשמל/תאורה #{item['index']}", key=f"desc_{item['index']}")
                        is_inc = st.checkbox("כלול בכתב כמויות", value=(item["confirmed_count"] > 0), key=f"inc_leg_{item['index']}")
                    with c3:
                        st.metric("כמות מאושרת:", f"{item['confirmed_count']} יח'")
                    
                    if is_inc:
                        boq_rows.append({
                            "מס'": item["index"],
                            "תמונת סמל": item["image_uri"],
                            "symbol_img": item["symbol_img"],
                            "image_uri": item["image_uri"],
                            "תיאור הפריט": s_desc,
                            "כמות מאושרת": item["confirmed_count"],
                            "יחידת מידה": "יח'"
                        })
                    st.markdown("---")
                    
                st.subheader("🗺️ תוכנית עם סימוני הפריטים המאושרים:")
                st.image(cv2.cvtColor(disp_plan, cv2.COLOR_BGR2RGB))
                
                if boq_rows:
                    st.subheader("📊 ריכוז סופי לכתב כמויות (עם תמונות סמלים)")
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
                    
                    html_report = generate_html_boq(boq_rows)
                    st.download_button(
                        "📥 ייצא דוח כתב כמויות כולל תמונות סמלים (Excel / Web)",
                        data=html_report.encode("utf-8"),
                        file_name="SAQ_AI_Electrical_Takeoff.xls",
                        mime="application/vnd.ms-excel"
                    )
        else:
            st.info("💡 אנא העלה את קובץ התוכנית ואת קובץ המקרא להפעלת הסריקה.")

else:
    scale_val = 1.0
    if mode == "ספירה מתוכנית בודדת":
        cad_file = st.file_uploader("העלה שרטוט CAD (DXF):", type=["dxf"])
        if cad_file and HAS_VECTOR_ENGINE:
            parser = DXFVectorParser(cad_file, unit_scale_to_meter=scale_val)
            layers = [l["name"] for l in parser.get_layers_summary() if l["entity_count"] > 0]
            if discipline == "⚡ חשמל ומאור":
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
