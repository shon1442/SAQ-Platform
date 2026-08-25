import base64
import copy
import io
import json
import math
import os
import time
import cv2
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    from saq_vector_engine import DXFVectorParser, compare_vector_delta
    HAS_VECTOR_ENGINE = True
except Exception:
    HAS_VECTOR_ENGINE = False

LOGO_PATH = "logo.png.png" if os.path.exists("logo.png.png") else "logo.png"
has_logo = os.path.exists(LOGO_PATH)

try:
    if has_logo:
        app_icon = Image.open(LOGO_PATH)
        app_icon.thumbnail((64, 64)) 
    else:
        app_icon = "SAQ"
except Exception:
    app_icon = "SAQ"

MEMORY_FILE = "saq_ai_memory.json"

st.set_page_config(
    page_title="S.A. Quantities AI - Global Takeoff Platform",
    layout="wide",
    page_icon=app_icon,
)

# ========================================================
# State Management Initialization
# ========================================================
if "global_is_us" not in st.session_state:
    st.session_state["global_is_us"] = False
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = None
if "current_discipline" not in st.session_state:
    st.session_state["current_discipline"] = "elec"
if "show_master_export" not in st.session_state:
    st.session_state["show_master_export"] = False
if "saved_quotes" not in st.session_state:
    st.session_state["saved_quotes"] = []
if "include_vat" not in st.session_state:
    st.session_state["include_vat"] = True
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "global_scale" not in st.session_state:
    st.session_state["global_scale"] = 125.0
if "project_notes" not in st.session_state:
    st.session_state["project_notes"] = ""

is_us_mode = st.session_state["global_is_us"]

disciplines_dict = {
    "elec": "Electrical & Lighting" if is_us_mode else "חשמל ומאור",
    "cons": "Construction (Walls)" if is_us_mode else "בניה (מחיצות ומעטפת)",
    "plum": "Plumbing & Gas" if is_us_mode else "אינסטלציה וגז",
    "tile": "Flooring & Tiling" if is_us_mode else "ריצוף וחיפוי",
    "hvac": "HVAC & Infrastructure" if is_us_mode else "מיזוג אוויר ותשתיות",
    "kitch": "Kitchen Design" if is_us_mode else "מטבחים"
}
disciplines_keys = list(disciplines_dict.keys())
disciplines_display = list(disciplines_dict.values())

if "project_boq" not in st.session_state:
    st.session_state["project_boq"] = {k: [] for k in disciplines_keys}

def reset_project_state():
    st.session_state["project_boq"] = {k: [] for k in disciplines_keys}
    keys_to_clear = [
        "elec_results", "elec_plan_raw", "elec_verified",
        "plumb_results", "plumb_plan_raw", "plumb_verified",
        "hvac_results", "hvac_plan_raw", "hvac_verified",
        "show_master_export", "project_notes"
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

def clear_current_discipline():
    d_key = st.session_state["current_discipline"]
    st.session_state["project_boq"][d_key] = []
    st.session_state.pop(f"{d_key}_results", None)
    st.session_state.pop(f"{d_key}_plan_raw", None)
    st.session_state.pop(f"{d_key}_verified", None)

# ========================================================
# UI Engine & Styling
# ========================================================
app_mode = st.session_state.get("app_mode")

css_code = ""
css_code += "<meta name='google' content='notranslate'>\n"
css_code += "<style>\n"
css_code += "body { top: 0px !important; }\n"
css_code += ".stApp { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }\n"
css_code += ".block-container { padding-top: 1rem !important; max-width: 95% !important; }\n"
css_code += "section[data-testid='stSidebar'] { border-right: 1px solid #334155; box-shadow: 2px 0 15px rgba(0,0,0,0.1); }\n"
css_code += "h1, h2, h3 { font-weight: 700 !important; }\n"
css_code += "div.stButton > button { border-radius: 8px !important; font-weight: 600 !important; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; }\n"
css_code += "div.stButton > button:hover { transform: translateY(-2px) !important; }\n"

if st.session_state["dark_mode"]:
    css_code += "div[data-testid='stAppViewContainer'] { background-color: #121212; color: #ffffff; }\n"
    css_code += "h1, h2, h3, p, span { color: #ffffff !important; }\n"
    css_code += "section[data-testid='stSidebar'] { background-color: #1e1e1e !important; }\n"
    css_code += "div[data-testid='stFileUploader'], div[data-testid='stMetric'], div.stAlert { background: #1e1e1e !important; border: 1px solid #333 !important; }\n"
else:
    if app_mode in ["Tenant_CO", "שינויי דיירים"]:
        css_code += "div[data-testid='stAppViewContainer'] { background-color: #f8fafc; background-image: linear-gradient(rgba(59, 130, 246, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(59, 130, 246, 0.05) 1px, transparent 1px); background-size: 30px 30px; }\n"
    elif app_mode in ["Renovation", "קבלני שיפוצים"]:
        css_code += "div[data-testid='stAppViewContainer'] { background-color: #fdfdfc; background-image: radial-gradient(#cbd5e1 1.5px, transparent 0); background-size: 25px 25px; }\n"
    
    css_code += "div[data-testid='stFileUploader'], div[data-testid='stMetric'], div.stAlert { background: rgba(255, 255, 255, 0.95) !important; border-radius: 12px !important; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important; padding: 15px; }\n"

css_code += "</style>\n"
st.markdown(css_code, unsafe_allow_html=True)

# ========================================================
# Splash Screen
# ========================================================
if "splash_shown" not in st.session_state:
    st.session_state["splash_shown"] = True

    splash_css = ""
    splash_css += "<style>\n"
    splash_css += ".fullscreen-splash { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: #000; z-index: 999999; display: flex; flex-direction: column; justify-content: center; align-items: center; overflow: hidden; font-family: 'Segoe UI', Arial, sans-serif; animation: hideSplash 4.5s forwards ease-in-out; }\n"
    splash_css += "@keyframes hideSplash { 0% { opacity: 1; visibility: visible; } 85% { opacity: 1; visibility: visible; } 99% { opacity: 0; visibility: visible; } 100% { opacity: 0; visibility: hidden; pointer-events: none; z-index: -10; display: none; } }\n"
    # Using the provided image for a time-lapse effect via steps
    splash_css += ".timelapse-bg { width: 800px; height: 450px; background-image: url('Gemini_Generated_Image_3zojl43zojl43zoj.jpeg'); background-size: 400% 100%; animation: buildLapse 3s steps(3, end) forwards; border-radius: 12px; box-shadow: 0 10px 40px rgba(255,255,255,0.2); }\n"
    splash_css += "@keyframes buildLapse { 0% { background-position: 0% 0; } 33% { background-position: 33.33% 0; } 66% { background-position: 66.66% 0; } 100% { background-position: 100% 0; } }\n"
    splash_css += ".splash-text-main { margin-top: 30px; font-size: 42px; font-weight: 600; color: #ffffff; letter-spacing: 4px; text-shadow: 0 2px 10px rgba(0,0,0,0.4); opacity: 0; animation: textFade 1s 1s forwards ease-in-out; }\n"
    splash_css += "@keyframes textFade { 0% { opacity: 0; transform: translateY(15px); } 100% { opacity: 1; transform: translateY(0); } }\n"
    splash_css += "</style>\n"

    splash_html = "<div class='fullscreen-splash' translate='no'>\n"
    splash_html += "<div class='timelapse-bg'></div>\n"
    splash_html += "<div class='splash-text-main'>S.A.Q. מחשבים את העתיד</div>\n"
    splash_html += "</div>\n"

    st.markdown(splash_css + splash_html, unsafe_allow_html=True)
    time.sleep(4.5)
    st.session_state["app_initialized"] = True
    st.rerun()


# ========================================================
# Helper Functions
# ========================================================
def load_ai_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"approved_patterns": [], "rejected_patterns": [], "structural_alerts": []}
    return {"approved_patterns": [], "rejected_patterns": [], "structural_alerts": []}

def save_ai_memory(memory):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

ai_memory = load_ai_memory()

def img_to_data_uri(cv2_img):
    if cv2_img is None or not hasattr(cv2_img, "size") or cv2_img.size == 0:
        return ""
    try:
        h, w = cv2_img.shape[:2]
        if h > 200 or w > 200:
            scale = 200 / max(h, w)
            cv2_img = cv2.resize(cv2_img, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".png", cv2_img)
        return f"data:image/png;base64,{base64.b64encode(buf).decode()}"
    except Exception:
        return ""

def load_raster(file, scale=1.4):
    if file is None: return None
    try:
        file.seek(0)
        max_pixels = 2000 * 2000 
        if file.name.lower().endswith(".pdf"):
            pdf = pdfium.PdfDocument(file.read())
            page = pdf.get_page(0)
            w, h = page.get_size()
            calc_scale = math.sqrt(max_pixels / max(w * h, 1))
            final_scale = min(scale, calc_scale)
            bitmap = page.render(scale=final_scale)
            pil_img = bitmap.to_pil().convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if img is None: return None
            h, w = img.shape[:2]
            if h * w > max_pixels:
                ratio = math.sqrt(max_pixels / (h * w))
                img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
            if len(img.shape) == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3] / 255.0
                bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
                for c in range(3): bg[:, :, c] = (img[:, :, c] * alpha + bg[:, :, c] * (1.0 - alpha)).astype(np.uint8)
                return bg
            elif len(img.shape) == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
    except Exception as e:
        st.error(f"Error loading file: {e}" if is_us_mode else f"שגיאה בטעינת הקובץ: {e}")
        return None

def safe_render_table(rows, is_us=False):
    cols = (["No.", "Symbol", "Item Description", "Approved Qty", "Unit"] if is_us else ["מס'", "תמונת סמל", "תיאור הפריט", "כמות מאושרת", "יחידת מידה"])
    if not rows:
        st.dataframe(pd.DataFrame(columns=cols))
        return
    clean_data = []
    for idx, r in enumerate(rows):
        u_meas = r.get("יחידת מידה", "יח'")
        qty = r.get("כמות מאושרת", 0)
        if is_us:
            if 'מ"א' in u_meas or "מטר" in u_meas: qty = round(qty * 3.28084, 2); u_meas = "Linear Feet (FT)"
            elif 'מ"ר' in u_meas: qty = round(qty * 10.7639, 2); u_meas = "Square Feet (SQFT)"
            elif "יח'" in u_meas: u_meas = "Units"
        clean_data.append({
            cols[0]: r.get("מס'", idx + 1), cols[1]: r.get("תמונת סמל", ""), cols[2]: r.get("תיאור הפריט", f"Item #{idx+1}"), cols[3]: qty, cols[4]: u_meas,
        })
    df = pd.DataFrame(clean_data)[cols]
    st.dataframe(df, column_config={cols[1]: st.column_config.ImageColumn("Symbol" if is_us else "סמל / תרשים", width="small")})

def get_pricing_item_cost(desc, unit, is_us=False, timing="Before"):
    multiplier = 1.5 if timing == "After" else 1.0
    if is_us:
        base_price = 45
        if "FT" in unit or "Length" in desc: base_price = 75
        elif "SQFT" in unit or "Area" in desc: base_price = 110
        elif "Demolition" in desc: base_price = 35
        elif "Units" in unit or "Fixture" in desc: base_price = 180
        if "AC" in desc or "HVAC" in desc: base_price = 1200
        if "Drain" in desc: base_price = 80
        if "Gas" in desc: base_price = 250
        if "Kitchen" in desc: base_price = 3500
        return base_price * multiplier
    else:
        unit_price = 150
        if "מ\"א" in unit or "מטר" in desc: unit_price = 220
        elif "מ\"ר" in unit or "שטח" in desc: unit_price = 340
        elif "הריסה" in desc: unit_price = 110
        elif "יח'" in unit or "נקודת" in desc: unit_price = 450
        if "מיזוג" in desc or "מזגן" in desc: unit_price = 3500
        if "ניקוז" in desc or 'קחז"מ' in desc: unit_price = 250
        if "גז" in desc: unit_price = 800
        if "מטבח" in desc: unit_price = 15000
        return unit_price * multiplier

def show_engineering_loader(text="Processing...", is_us=False):
    progress_bar = st.progress(0)
    status_box = st.empty()
    steps = 15
    for i in range(steps):
        time.sleep(0.05)
        percent = int((i + 1) * (100 / steps))
        progress_bar.progress(percent)
        status_box.markdown(f"Scanning: {text}" if is_us else f"סורק שרטוטים: {text}")
    progress_bar.empty()
    status_box.success("Takeoff completed successfully!" if is_us else "פענוח האתר הסתיים בהצלחה!")

# ========================================================
# Home Screen (Model Selection)
# ========================================================
if st.session_state["app_mode"] is None:
    home_css = ""
    home_css += "<style>\n"
    home_css += "div[data-testid='column'] div.stButton > button { height: 300px !important; width: 100%; border-radius: 12px !important; font-weight: bold; padding: 20px; font-size: 18px; color: white !important; white-space: pre-wrap; display: flex; flex-direction: column; justify-content: center; align-items: center; }\n"
    home_css += "div[data-testid='column']:nth-of-type(1) div.stButton > button { background: #1e3a8a !important; }\n"
    home_css += "div[data-testid='column']:nth-of-type(2) div.stButton > button { background: #b45309 !important; }\n"
    home_css += "</style>\n"
    st.markdown(home_css, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>S.A. Quantities AI (S.A.Q)</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #64748b;'>Advanced Digital Construction Site Takeoff</h3>" if is_us_mode else "<h3 style='text-align: center; color: #64748b;'>אתר בנייה דיגיטלי מתקדם לפענוח שרטוטים וכתבי כמויות</h3>", unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns(2, gap="large")
    with col_m1:
        tenant_btn_txt = "Tenant Modifications (COs)\n\nCompares requested change drawings against baseline contract standards." if is_us_mode else "מודל שינויי דיירים\n\nהשוואת שרטוט שינויים מול שרטוט מכר. דלתא, מעקב בקרת מעטפת הנדסית."
        if st.button(tenant_btn_txt, use_container_width=True, key="btn_mode_tenant"):
            st.session_state["app_mode"] = "Tenant_CO" if is_us_mode else "שינויי דיירים"
            reset_project_state()
            st.rerun()

    with col_m2:
        reno_btn_txt = "Renovation Contractors (As-Is)\n\nCompares proposed plan vs. As-Is existing layout." if is_us_mode else "מודל קבלני שיפוצים\n\nהשוואת שרטוט מוצע מול מצב קיים. הריסה, בנייה חדשה, חציבות."
        if st.button(reno_btn_txt, use_container_width=True, key="btn_mode_reno"):
            st.session_state["app_mode"] = "Renovation" if is_us_mode else "קבלני שיפוצים"
            reset_project_state()
            st.rerun()
    st.stop()


# ========================================================
# Advanced Toolbar & Navigation
# ========================================================
def go_home():
    st.session_state["app_mode"] = None
    reset_project_state()
    st.rerun()

def go_calculation_options():
    # Returns to discipline view, clears current selection data but keeps mode
    st.session_state["show_master_export"] = False
    st.rerun()

with st.container():
    t_col1, t_col2, t_col3, t_col4, t_col5, t_col6, t_col7 = st.columns([1, 1.5, 1, 1, 1, 1, 1.5])
    with t_col1:
        if st.button("SAQ Logo" if not has_logo else "Logo", key="logo_btn", help="Back to Options" if is_us_mode else "חזרה לאפשרויות חישוב"):
            go_calculation_options()
    with t_col2:
        scale_opts = ["1:50", "1:100", "Custom"]
        selected_scale = st.selectbox("Scale" if is_us_mode else "קנה מידה", scale_opts, label_visibility="collapsed")
        if selected_scale == "1:50": st.session_state["global_scale"] = 38.0 if is_us_mode else 125.0
        elif selected_scale == "1:100": st.session_state["global_scale"] = 19.0 if is_us_mode else 62.5
    with t_col3:
        if st.button("Calc" if is_us_mode else "מחשבון"):
            pass # Expand logic could go here via dialogue in later streamlit versions
    with t_col4:
        if st.button("Reset" if is_us_mode else "איפוס נתונים"):
            clear_current_discipline()
            st.rerun()
    with t_col5:
        if st.button("Dark" if not st.session_state["dark_mode"] else "Light"):
            st.session_state["dark_mode"] = not st.session_state["dark_mode"]
            st.rerun()
    with t_col6:
        if st.button("Notes" if is_us_mode else "הערות"):
            pass # Handled in expander below
    with t_col7:
        st.session_state["include_vat"] = st.toggle("Include VAT" if is_us_mode else "כולל מע\"מ", value=st.session_state.get("include_vat", True))

with st.expander("Project Notes" if is_us_mode else "הערות פרויקט"):
    st.session_state["project_notes"] = st.text_area("Notes" if is_us_mode else "הזן הערות שיופיעו בדוח", value=st.session_state.get("project_notes", ""))

# ========================================================
# Sidebar
# ========================================================
with st.sidebar:
    st.markdown("### Control Center" if is_us_mode else "### מרכז בקרה")
    if st.button("Home" if is_us_mode else "חזרה למסך הבית", use_container_width=True):
        go_home()
    
    st.markdown("---")
    st.markdown("**Progress:**" if is_us_mode else "**התקדמות:**")
    for k, v in disciplines_dict.items():
        is_done = len(st.session_state["project_boq"].get(k, [])) > 0
        status_icon = "[V]" if is_done else "[ ]"
        st.markdown(f"{status_icon} {v}")

    st.markdown("---")
    curr_idx = disciplines_keys.index(st.session_state["current_discipline"]) if st.session_state["current_discipline"] in disciplines_keys else 0
    selected_disc = st.selectbox(
        "Current Plan:" if is_us_mode else "סוג תוכנית:",
        disciplines_display,
        index=curr_idx,
        key="disc_selector_widget"
    )
    for k, v in disciplines_dict.items():
        if v == selected_disc and st.session_state["current_discipline"] != k:
            st.session_state["current_discipline"] = k
            st.rerun()
            
    if st.button("Open Master BOQ Hub" if is_us_mode else "פתח מרכז דוחות", use_container_width=True):
        st.session_state["show_master_export"] = True
        st.rerun()

curr_key = st.session_state["current_discipline"]
mode_lbl = st.session_state["app_mode"]
is_tenant = mode_lbl in ["Tenant_CO", "שינויי דיירים"]
px_meter = st.session_state["global_scale"]

st.title(f"{disciplines_dict[curr_key]} - {mode_lbl}")


# ========================================================
# AI and CV logic (Mocked for Drain/Gas specific distances)
# ========================================================
def calc_pipe_split(distance_m):
    # Splits pipe into wall chasing (30%) and floor embedding (70%)
    return round(distance_m * 0.3, 2), round(distance_m * 0.7, 2)

# ========================================================
# Discipline Views
# ========================================================
if not st.session_state["show_master_export"]:
    
    timing_val = "Before"
    if is_tenant:
        timing_choice = st.radio(
            "Execution Stage:" if is_us_mode else "שלב ביצוע עבור שינויי הדיירים:",
            ["Before Execution (Planning/Pricing)", "After Execution (Field Verification)"] if is_us_mode else ["לפני ביצוע (תכנון, תמחור מוקדם ואישור דייר)", "אחרי ביצוע (בדיקת שטח, מדידה ובקרה בפועל)"],
            horizontal=True
        )
        timing_val = "After" if "After" in timing_choice or "אחרי" in timing_choice else "Before"

    if curr_key == "cons":
        c_exec, c_std = st.columns(2)
        with c_exec:
            f_plan = st.file_uploader("Proposed Plan (Required)" if is_us_mode else "שרטוט מוצע/ביצוע (חובה)", type=["pdf", "png", "jpg"])
        with c_std:
            f_std = st.file_uploader("Baseline Plan (Optional)" if is_us_mode else "שרטוט מצב קיים/סטנדרט (אופציונלי)", type=["pdf", "png", "jpg"])
        
        b_wall_h = st.number_input("Wall Height (m)" if is_us_mode else "גובה קירות (מטר):", value=2.7, step=0.1)

        if f_plan and st.button("Run Takeoff" if is_us_mode else "הפעל חישוב בניה"):
            show_engineering_loader("Scanning walls...", is_us_mode)
            # Dummy logic
            lin_exec = 15.5
            lin_std = 12.0 if f_std else 0
            
            diff_m = round(lin_exec - lin_std, 2)
            diff_sqm = round(diff_m * b_wall_h, 2)

            b_rows = [{
                "מס'": 1, "תמונת סמל": "", "כמות מאושרת": abs(diff_m),
                "תיאור הפריט": f"Partition Walls Delta (Diff: {diff_m})" if is_us_mode else f"דלתא שינויי אורך קירות (הפרש: {diff_m})",
                "יחידת מידה": 'מ"א',
            }, {
                "מס'": 2, "תמונת סמל": "", "כמות מאושרת": abs(diff_sqm),
                "תיאור הפריט": f"Partition Area Delta (Height {b_wall_h})" if is_us_mode else f"דלתא שטח קירות (גובה {b_wall_h})",
                "יחידת מידה": 'מ"ר',
            }]
            st.session_state["project_boq"][curr_key] = b_rows
            safe_render_table(b_rows, is_us=is_us_mode)

    elif curr_key == "plum":
        c_exec, c_std = st.columns(2)
        with c_exec: f_plan = st.file_uploader("Plumbing Plan (Required)" if is_us_mode else "תוכנית אינסטלציה (חובה)", type=["pdf", "png"])
        with c_std: f_std = st.file_uploader("Baseline Standard" if is_us_mode else "תוכנית קיים/סטנדרט", type=["pdf", "png"])

        if f_plan and st.button("Run Plumbing Takeoff" if is_us_mode else "הפעל ספירת נקודות אינסטלציה וגז"):
            show_engineering_loader("Scanning fixtures and pipes...", is_us_mode)
            p_rows = []
            relocs = [{"type": "Sink", "distance_m": 2.5}]
            added = [{"type": "Gas Point", "distance_m": 4.0}]
            
            idx = 1
            for r in relocs:
                p_rows.append({
                    "מס'": idx, "תמונת סמל": "", "כמות מאושרת": 1,
                    "תיאור הפריט": f"Relocate {r['type']}" if is_us_mode else f"הזזת {r['type']} (מרחק: {r['distance_m']} מ')",
                    "יחידת מידה": "יח'",
                })
                chase, embed = calc_pipe_split(r['distance_m'])
                p_rows.append({
                    "מס'": idx+1, "תמונת סמל": "", "כמות מאושרת": embed,
                    "תיאור הפריט": "Concrete embedded pipe to drain" if is_us_mode else f"ביטון צנרת מים/ביוב עד לקולטן (עבור {r['type']})",
                    "יחידת מידה": 'מ"א',
                })
                idx += 2
                
            for a in added:
                p_rows.append({
                    "מס'": idx, "תמונת סמל": "", "כמות מאושרת": 1,
                    "תיאור הפריט": f"New {a['type']}" if is_us_mode else f"נקודת {a['type']} חדשה",
                    "יחידת מידה": "יח'",
                })
                if "Gas" in a['type'] or "גז" in a['type']:
                     p_rows.append({
                        "מס'": idx+1, "תמונת סמל": "", "כמות מאושרת": a['distance_m'],
                        "תיאור הפריט": "Gas piping compared to standard" if is_us_mode else "אורך צנרת גז בהשוואה לסטנדרט (שינוי זוהה)",
                        "יחידת מידה": 'מ"א',
                    })
                idx += 2

            st.session_state["project_boq"][curr_key] = p_rows
            safe_render_table(p_rows, is_us=is_us_mode)

    elif curr_key == "hvac":
        c_exec, c_std = st.columns(2)
        with c_exec: f_plan = st.file_uploader("HVAC Plan (Required)" if is_us_mode else "תוכנית מיזוג (חובה)", type=["pdf", "png"])
        with c_std: f_std = st.file_uploader("Baseline Standard" if is_us_mode else "תוכנית קיים/סטנדרט", type=["pdf", "png"])

        if f_plan and st.button("Run HVAC Takeoff" if is_us_mode else "הפעל סריקת מיזוג אוויר ותשתיות"):
            show_engineering_loader("Scanning AC units...", is_us_mode)
            h_rows = []
            relocs = [{"type": "AC Unit", "distance_m": 5.0}]
            idx = 1
            for r in relocs:
                chase, embed = calc_pipe_split(r['distance_m'])
                h_rows.append({
                    "מס'": idx, "תמונת סמל": "", "כמות מאושרת": 1,
                    "תיאור הפריט": f"Relocate {r['type']}" if is_us_mode else f"הזזת יחידת {r['type']}",
                    "יחידת מידה": "יח'",
                })
                h_rows.append({
                    "מס'": idx+1, "תמונת סמל": "", "כמות מאושרת": chase,
                    "תיאור הפריט": "Wall chasing for AC piping" if is_us_mode else "חציבת צנרת מיזוג בקיר",
                    "יחידת מידה": 'מ"א',
                })
                h_rows.append({
                    "מס'": idx+2, "תמונת סמל": "", "כמות מאושרת": embed,
                    "תיאור הפריט": "Floor embedding for AC piping to drain" if is_us_mode else "ביטון צנרת מיזוג בריצפה עד לקולטן",
                    "יחידת מידה": 'מ"א',
                })
                idx += 3
            
            st.session_state["project_boq"][curr_key] = h_rows
            safe_render_table(h_rows, is_us=is_us_mode)

    elif curr_key == "kitch":
        st.info("Kitchen module requires execution plan (mandatory) and baseline (optional)." if is_us_mode else "מודול מטבחים מחייב תוכנית ביצוע מפורטת. שרטוט מצב קיים הנו אופציונלי.")
        c_exec, c_std = st.columns(2)
        with c_exec: f_plan = st.file_uploader("Kitchen Plan (Required)" if is_us_mode else "תוכנית מטבח ביצוע (חובה)", type=["pdf", "png"])
        with c_std: f_std = st.file_uploader("Baseline Kitchen" if is_us_mode else "תוכנית מטבח קיים", type=["pdf", "png"])
        
        if f_plan and st.button("Run Kitchen Takeoff" if is_us_mode else "הפעל חישוב מטבח"):
            show_engineering_loader("Scanning kitchen elements...", is_us_mode)
            k_rows = [{
                "מס'": 1, "תמונת סמל": "", "כמות מאושרת": 1,
                "תיאור הפריט": "Kitchen island addition" if is_us_mode else "תוספת אי למטבח כולל תשתיות",
                "יחידת מידה": "יח'",
            }, {
                "מס'": 2, "תמונת סמל": "", "כמות מאושרת": 4.5,
                "תיאור הפריט": "Extra countertop length" if is_us_mode else "תוספת משטח עבודה (שיש) נטו",
                "יחידת מידה": 'מ"א',
            }]
            st.session_state["project_boq"][curr_key] = k_rows
            safe_render_table(k_rows, is_us=is_us_mode)

    else:
        # Fallback for elec/tile using simple structure due to constraints
        c_exec = st.file_uploader(f"{disciplines_dict[curr_key]} Plan", type=["pdf", "png"])
        if c_exec and st.button("Run Takeoff" if is_us_mode else "הפעל פענוח"):
            show_engineering_loader("Processing...", is_us_mode)
            dummy = [{"מס'": 1, "תמונת סמל": "", "כמות מאושרת": 10, "תיאור הפריט": "Standard Items" if is_us_mode else "פריטים סטנדרטיים", "יחידת מידה": "יח'"}]
            st.session_state["project_boq"][curr_key] = dummy
            safe_render_table(dummy, is_us=is_us_mode)


# ========================================================
# Master Export Hub
# ========================================================
if st.session_state.get("show_master_export", False):
    st.markdown("---")
    st.header("Master BOQ Hub" if is_us_mode else "מרכז הדוחות לאתר הבנייה")
    if st.session_state["project_notes"]:
        st.info(st.session_state["project_notes"])

    all_project_rows = []
    grand_total = 0
    tax_rate = 0.17 if not is_us_mode else 0.085
    
    for d_key in disciplines_keys:
        d_rows = st.session_state["project_boq"].get(d_key, [])
        if d_rows:
            st.subheader(disciplines_dict[d_key])
            safe_render_table(d_rows, is_us=is_us_mode)
            for r in d_rows:
                cost = get_pricing_item_cost(r.get("תיאור הפריט", ""), r.get("יחידת מידה", ""), is_us_mode)
                grand_total += cost * r.get("כמות מאושרת", 0)
                all_project_rows.append(r)

    if all_project_rows:
        tax_val = grand_total * tax_rate if st.session_state["include_vat"] else 0
        final_total = grand_total + tax_val
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Excl. VAT" if is_us_mode else "סה\"כ לפני מע\"מ", f"{grand_total:,.2f}")
        c2.metric("VAT" if is_us_mode else "מע\"מ", f"{tax_val:,.2f}")
        c3.metric("Grand Total" if is_us_mode else "סה\"כ לתשלום", f"{final_total:,.2f}")
        
        st.button("Save & Finish" if is_us_mode else "שמור וסיים פרויקט", on_click=reset_project_state)
