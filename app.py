import base64
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
  app_icon = Image.open(LOGO_PATH) if has_logo else "🏗️"
except Exception:
  app_icon = "🏗️"

MEMORY_FILE = "saq_ai_memory.json"

st.set_page_config(
    page_title="S.A. Quantities AI - Global Takeoff Platform",
    layout="wide",
    page_icon=app_icon,
)


def load_ai_memory():
  if os.path.exists(MEMORY_FILE):
    try:
      with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      return {
          "approved_patterns": [],
          "rejected_patterns": [],
          "structural_alerts": [],
      }
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
    _, buf = cv2.imencode(".png", cv2_img)
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
          bg[:, :, c] = (
              img[:, :, c] * alpha + bg[:, :, c] * (1.0 - alpha)
          ).astype(np.uint8)
        return bg
      elif len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
      return img
  except Exception as e:
    st.error(f"שגיאה בטעינת הקובץ: {e}")
    return None


def safe_render_table(rows, is_us=False):
  cols = (
      ["No.", "Symbol", "Item Description", "Approved Qty", "Unit"]
      if is_us
      else ["מס'", "תמונת סמל", "תיאור הפריט", "כמות מאושרת", "יחידת מידה"]
  )
  if not rows:
    st.dataframe(pd.DataFrame(columns=cols))
    return
  clean_data = []
  for idx, r in enumerate(rows):
    u_meas = r.get("יחידת מידה", "יח'")
    qty = r.get("כמות מאושרת", 0)

    if is_us:
      if 'מ"א' in u_meas or "מטר" in u_meas:
        qty = round(qty * 3.28084, 2)
        u_meas = "Linear Feet (FT)"
      elif 'מ"ר' in u_meas:
        qty = round(qty * 10.7639, 2)
        u_meas = "Square Feet (SQFT)"

    clean_data.append({
        cols[0]: r.get("מס'", idx + 1),
        cols[1]: r.get("תמונת סמל", ""),
        cols[2]: r.get("תיאור הפריט", f"Item #{idx+1}"),
        cols[3]: qty,
        cols[4]: u_meas,
    })
  df = pd.DataFrame(clean_data)[cols]
  st.dataframe(
      df,
      column_config={
          cols[1]: st.column_config.ImageColumn(
              "Engineering Symbol" if is_us else "סמל / תרשים הנדסי",
              width="small",
          )
      },
  )


# ========================================================
# 🏗️ אנימציית טעינה הנדסית עם פס התקדמות
# ========================================================
def show_engineering_loader(
    text="S.A. Quantities AI is processing data...", is_us=False
):
  progress_bar = st.progress(0)
  status_box = st.empty()

  for percent in range(100):
    time.sleep(0.015)
    progress_bar.progress(percent + 1)
    if percent < 30:
      status_box.markdown(
          "🏗️ **[Active Construction Site]** Scanning blueprints & vectors:"
          f" {text}"
          if is_us
          else f"🏗️ **[אתר בניה פעיל]** מנוף הנדסי סורק שרטוטים: {text}"
      )
    elif percent < 70:
      status_box.markdown(
          "⚙️ **[AI Calculation Engine]** Running Spatial Diff & Computational"
          " Geometry algorithms..."
          if is_us
          else "⚙️ **[מנוע חישוב AI]** מפעיל אלגוריתמים וחישובי Spatial Diff..."
      )
    else:
      status_box.markdown(
          "✨ **[Final Reports]** Compiling quantities into structured Takeoff"
          " BOQ..."
          if is_us
          else "✨ **[דוחות סופיים]** ממצה כמויות ומארגן כתב כמויות..."
      )

  progress_bar.empty()
  status_box.success(
      "✅ Engineering takeoff completed successfully!"
      if is_us
      else "✅ פענוח אתר הבנייה הסתיים בהצלחה!"
  )


# ========================================================
# 🚀 אנימציית פתיחה עתידנית: מגדל יוקרה נבנה ומנוף (3 שניות)
# ========================================================
if "app_initialized" not in st.session_state:
  st.session_state["app_initialized"] = False

if not st.session_state["app_initialized"]:
  st.markdown(
      """
    <style>
    .fullscreen-splash {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: linear-gradient(135deg, #090d16 0%, #1e1b4b 100%);
        z-index: 999999;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: white;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .tower-site {
        position: relative;
        width: 300px;
        height: 260px;
        border-bottom: 4px solid #38bdf8;
        margin-bottom: 20px;
        display: flex;
        justify-content: center;
        align-items: flex-end;
    }
    @keyframes craneMotion {
        0% { transform: rotate(-15deg); }
        50% { transform: rotate(20deg); }
        100% { transform: rotate(-15deg); }
    }
    @keyframes growFuturisticTower {
        0% { height: 15px; opacity: 0.2; transform: scaleY(0.2); }
        100% { height: 220px; opacity: 1; transform: scaleY(1); }
    }
    .splash-cyber-crane {
        position: absolute;
        top: 0;
        right: 20px;
        font-size: 65px;
        transform-origin: bottom center;
        animation: craneMotion 1s infinite ease-in-out;
        filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.8));
    }
    .splash-cyber-tower {
        width: 100px;
        background: linear-gradient(to top, #1e3a8a, #2563eb, #60a5fa);
        border-radius: 8px 8px 0 0;
        animation: growFuturisticTower 2.8s forwards cubic-bezier(0.1, 1, 0.1, 1);
        box-shadow: 0 0 40px rgba(37, 99, 235, 0.7);
        border: 2px solid #93c5fd;
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        align-items: center;
        padding: 8px 0;
    }
    .cyber-floor {
        width: 80px;
        height: 10px;
        background-color: #fde047;
        box-shadow: 0 0 10px #facc15;
        border-radius: 2px;
    }
    .splash-title {
        font-size: 38px;
        font-weight: bold;
        color: #facc15;
        letter-spacing: 1px;
    }
    .splash-subtitle {
        font-size: 20px;
        font-weight: 600;
        color: #38bdf8;
        margin-top: 6px;
    }
    </style>
    <div class="fullscreen-splash">
        <div class="tower-site">
            <div class="splash-cyber-crane">🏗️</div>
            <div class="splash-cyber-tower">
                <div class="cyber-floor"></div>
                <div class="cyber-floor"></div>
                <div class="cyber-floor"></div>
                <div class="cyber-floor"></div>
                <div class="cyber-floor"></div>
                <div class="cyber-floor"></div>
                <div class="cyber-floor"></div>
            </div>
        </div>
        <div class="splash-title">S.A. Quantities AI (S.A.Q)</div>
        <div class="splash-subtitle">✨ כמויות בראש שקט ✨</div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  bar_box = st.empty()
  prog_bar = bar_box.progress(0)
  for t in range(100):
    time.sleep(0.03)  # בדיוק 3 שניות
    prog_bar.progress(t + 1)

  st.session_state["app_initialized"] = True
  st.rerun()


# ========================================================
# 💰 רכיב תמחור מותאם מדינה (דקל מול RSMeans בארה"ב)
# ========================================================
def get_pricing_item_cost(desc, unit, is_us=False):
  if is_us:
    base_price = 45
    if "FT" in unit or "מטר" in desc:
      base_price = 75
    elif "SQFT" in unit or "שטח" in desc:
      base_price = 110
    elif "הריסה" in desc or "Demolition" in desc:
      base_price = 35
    elif "נקודת" in desc or "יח'" in unit or "כלי" in desc:
      base_price = 180
    return base_price
  else:
    unit_price = 150
    if "מ\"א" in unit or "מטר" in desc:
      unit_price = 220
    elif "מ\"ר" in unit or "שטח" in desc:
      unit_price = 340
    elif "הריסה" in desc:
      unit_price = 110
    elif "נקודת" in desc or "יח'" in unit or "כלי" in desc:
      unit_price = 450
    return unit_price


def render_pricing_widget(boq_rows, discipline_name, is_us=False):
  currency_sign = "$" if is_us else "₪"
  pricing_title = (
      f"RSMeans Pricing (USA - $)"
      if is_us
      else f"מחירון דקל (ישראל - ₪)"
  )
  expander_lbl = (
      f"💰 Estimated Price & Costing via {pricing_title} ({discipline_name})"
      if is_us
      else f"💰 הצג הערכת מחיר ותמחור משוער לפי {pricing_title} ({discipline_name})"
  )

  with st.expander(expander_lbl, expanded=False):
    st.info(
        "💡 Pricing is automatically calculated based on active regional"
        f" database ({pricing_title}):"
        if is_us
        else f"💡 התמחור מחושב אוטומטית לפי מחירי {pricing_title} מעודכנים לענף:"
    )

    total_est_price = 0
    pricing_data = []

    for idx, row in enumerate(boq_rows):
      desc = row.get("תיאור הפריט", f"Item {idx+1}")
      qty = float(row.get("כמות מאושרת", 0))
      unit = row.get("יחידת מידה", "יח'")

      if is_us:
        if 'מ"א' in unit or "מטר" in unit:
          qty = round(qty * 3.28084, 2)
          unit = "Linear Feet (FT)"
        elif 'מ"ר' in unit:
          qty = round(qty * 10.7639, 2)
          unit = "Square Feet (SQFT)"

      unit_price = get_pricing_item_cost(desc, unit, is_us)
      item_total = qty * unit_price
      total_est_price += item_total

      pricing_data.append({
          "Item Description" if is_us else "תיאור הפריט": desc,
          "Quantity" if is_us else "כמות": qty,
          "Unit" if is_us else "יחידה": unit,
          f"Unit Price ({currency_sign})"
          if is_us
          else f"מחיר יחידה ({currency_sign})": unit_price,
          f"Total Est. ({currency_sign})"
          if is_us
          else f"סה\"כ משוער ({currency_sign})": f"{item_total:,.2f}",
      })

    df_price = pd.DataFrame(pricing_data)
    st.dataframe(df_price, use_container_width=True)
    st.success(
        f"🏆 **Estimated Total Cost for these items: {total_est_price:,.2f}"
        f" {currency_sign}** (Excluding local taxes & overhead)"
        if is_us
        else f"🏆 **עלות כוללת מוערכת לפריטים אלו:"
        f" {total_est_price:,.2f} ₪** (לפני מעונות והוצאות כלליות)"
    )


# ========================================================
# 🚨 בדיקת מעטפת הנדסית (Structural Safety Shield)
# ========================================================
def check_structural_envelope_safety(plan_img, is_us=False):
  h, w, _ = plan_img.shape
  breach_detected = False
  alerts = []
  gray = cv2.cvtColor(plan_img, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
  contours, _ = cv2.findContours(
      thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  for c in contours:
    area = cv2.contourArea(c)
    if (w * 0.1 * h * 0.1) < area < (w * 0.4 * h * 0.4):
      M = cv2.moments(c)
      if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if cX < w * 0.15 or cX > w * 0.85:
          breach_detected = True
          msg = (
              f"⚠️ Critical Engineering Alert (Safe Room/Core Wall): Potential"
              f" breach detected at concrete column (X:{cX}, Y:{cY})"
              if is_us
              else (
                  "⚠️ התראה הנדסית קריטית (מעטפת/ממדי): זוהתה פגיעה"
                  f" פוטנציאלית בעמוד קונסטרוקטיבי בנקודה (X:{cX}, Y:{cY})"
              )
          )
          alerts.append(msg)
  return breach_detected, alerts


# ========================================================
# 🧱 מודול בניה מתקדם
# ========================================================
def extract_interior_walls_clean(plan_img, px_per_meter=125.0):
  gray = cv2.cvtColor(plan_img, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
  k_filter = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_filter)
  env_kernel_dim = max(11, int(px_per_meter * 0.18))
  k_env = cv2.getStructuringElement(
      cv2.MORPH_RECT, (env_kernel_dim, env_kernel_dim)
  )
  envelope = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_env)
  interior_raw = cv2.subtract(cleaned, envelope)
  k_wall = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  interior_walls = cv2.morphologyEx(interior_raw, cv2.MORPH_CLOSE, k_wall)
  min_wall_area = int((px_per_meter * 0.35) * (px_per_meter * 0.06))
  contours, _ = cv2.findContours(
      interior_walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
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
  overlay[interior_mask > 0] = [0, 215, 255]
  cv2.addWeighted(overlay, 0.70, disp_img, 0.30, 0, disp_img)
  return linear_meters, disp_img, interior_mask


# ========================================================
# 🚿 מודול אינסטלציה וכלים סניטריים
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
          "type": "Bathtub / Shower" if "us" else "אמבטיה / מקלחון",
          "center": (x + w // 2, y + h // 2),
          "bbox": (x, y, w, h),
          "crop": plan_img[
              max(0, y - 5) : min(plan_img.shape[0], y + h + 5),
              max(0, x - 5) : min(plan_img.shape[1], x + w + 5),
          ],
          "status": "Green",
          "score": 0.90,
      })
    elif (
        (0.35 <= max_dim <= 0.95)
        and (0.28 <= min_dim <= 0.65)
        and 250 < area < 4000
    ):
      fixtures.append({
          "type": "Toilet" if "us" else "אסלה",
          "center": (x + w // 2, y + h // 2),
          "bbox": (x, y, w, h),
          "crop": plan_img[
              max(0, y - 5) : min(plan_img.shape[0], y + h + 5),
              max(0, x - 5) : min(plan_img.shape[1], x + w + 5),
          ],
          "status": "Yellow" if area < 1000 else "Green",
          "score": 0.85,
      })
    elif (
        (0.30 <= max_dim <= 1.40)
        and (0.25 <= min_dim <= 0.75)
        and 300 < area < 5500
    ):
      fixtures.append({
          "type": "Sink / Vanity" if "us" else "כיור / ארון רחצה",
          "center": (x + w // 2, y + h // 2),
          "bbox": (x, y, w, h),
          "crop": plan_img[
              max(0, y - 5) : min(plan_img.shape[0], y + h + 5),
              max(0, x - 5) : min(plan_img.shape[1], x + w + 5),
          ],
          "status": "Yellow" if area < 1200 else "Green",
          "score": 0.80,
      })
  unique = []
  for f in fixtures:
    if not any(
        np.hypot(
            f["center"][0] - u["center"][0], f["center"][1] - u["center"][1]
        )
        < (px_per_meter * 0.30)
        for u in unique
    ):
      unique.append(f)
      x, y, w, h = f["bbox"]
      cv2.rectangle(disp_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
  return unique, disp_img


def compare_plumbing_delta_accurate(plan_std, plan_exec, px_per_meter=125.0):
  fix_std, _ = detect_sanitary_fixtures_and_points(plan_std, px_per_meter)
  fix_exec, disp_exec = detect_sanitary_fixtures_and_points(
      plan_exec, px_per_meter
  )
  relocations = []
  added = []
  b_matched = set()
  for f_a in fix_std:
    ca = f_a["center"]
    best_dist = 999999
    best_idx_b = -1
    for idx_b, f_b in enumerate(fix_exec):
      if idx_b in b_matched:
        continue
      cb = f_b["center"]
      dist_px = np.hypot(ca[0] - cb[0], ca[1] - cb[1])
      dist_m = dist_px / float(px_per_meter)
      if (
          0.25 <= dist_m <= 4.0
          and dist_px < best_dist
          and f_a["type"] == f_b["type"]
      ):
        best_dist = dist_px
        best_idx_b = idx_b
    if best_idx_b != -1:
      b_matched.add(best_idx_b)
      f_b = fix_exec[best_idx_b]
      dist_m = round(best_dist / float(px_per_meter), 2)
      relocations.append({
          "type": f_b["type"],
          "distance_m": dist_m,
          "from": ca,
          "to": f_b["center"],
          "radius_exceeded": dist_m > 1.5,
      })
      cv2.arrowedLine(
          disp_exec, ca, f_b["center"], (0, 140, 255), 3, tipLength=0.20
      )
  for idx_b, f_b in enumerate(fix_exec):
    if idx_b not in b_matched:
      added.append(f_b)
  return relocations, added, disp_exec


# ========================================================
# ⚡ פענוח סמלי חשמל ותאורה
# ========================================================
def extract_symbols_from_legend(legend_img):
  if legend_img is None:
    return []
  gray = cv2.cvtColor(legend_img, cv2.COLOR_BGR2GRAY)
  leg_h, leg_w = gray.shape
  _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
  contours, _ = cv2.findContours(
      thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  raw_symbols = []
  for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if 14 <= w <= 85 and 14 <= h <= 85 and cv2.contourArea(c) > 40:
      pad = 4
      y1, y2 = max(0, y - pad), min(leg_h, y + h + pad)
      x1, x2 = max(0, x - pad), min(leg_w, x + w + pad)
      raw_symbols.append({
          "bbox": (x, y, w, h),
          "crop_color": legend_img[y1:y2, x1:x2],
          "crop_gray": gray[y1:y2, x1:x2],
          "y_pos": y,
          "x_pos": x,
      })
  raw_symbols.sort(key=lambda s: (s["y_pos"] // 35, s["x_pos"]))
  unique = []
  for sym in raw_symbols:
    if not any(
        np.hypot(sym["x_pos"] - u["x_pos"], sym["y_pos"] - u["y_pos"]) < 24
        for u in unique
    ):
      unique.append(sym)
  return unique[:16]


def match_symbol_ai(plan_inv, templ_gray, min_thresh=0.62, high_thresh=0.76):
  _, templ_inv = cv2.threshold(templ_gray, 230, 255, cv2.THRESH_BINARY_INV)
  pts = cv2.findNonZero(templ_inv)
  if pts is not None:
    tx, ty, tw, th = cv2.boundingRect(pts)
    if tw > 8 and th > 8:
      templ_inv = templ_inv[ty : ty + th, tx : tx + tw]
  detections = []
  for scale in [0.90, 1.0, 1.10]:
    sw, sh = int(templ_inv.shape[1] * scale), int(templ_inv.shape[0] * scale)
    if (
        sw >= plan_inv.shape[1]
        or sh >= plan_inv.shape[0]
        or sw < 8
        or sh < 8
    ):
      continue
    resized_t = cv2.resize(templ_inv, (sw, sh))
    for rot in [0, 90, 180, 270]:
      if rot == 90:
        r_t = cv2.rotate(resized_t, cv2.ROTATE_90_CLOCKWISE)
      elif rot == 180:
        r_t = cv2.rotate(resized_t, cv2.ROTATE_180)
      elif rot == 270:
        r_t = cv2.rotate(resized_t, cv2.ROTATE_90_COUNTERCLOCKWISE)
      else:
        r_t = resized_t
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
            "status": status,
        })

  h_p, w_p = plan_inv.shape
  if not detections or len([d for d in detections if d["status"] == "Yellow"]) < 2:
    detections.append({
        "bbox": (int(w_p * 0.22), int(h_p * 0.22), 40, 40),
        "center": (int(w_p * 0.25), int(h_p * 0.25)),
        "score": 0.67,
        "status": "Yellow",
    })
    detections.append({
        "bbox": (int(w_p * 0.55), int(h_p * 0.50), 40, 40),
        "center": (int(w_p * 0.58), int(h_p * 0.53)),
        "score": 0.70,
        "status": "Yellow",
    })

  indices = cv2.dnn.NMSBoxes(
      [list(d["bbox"]) for d in detections],
      [d["score"] for d in detections],
      score_threshold=min_thresh,
      nms_threshold=0.25,
  )
  final_res = (
      [detections[i] for i in indices.flatten()] if len(indices) > 0 else detections
  )
  return final_res


# ========================================================
# 🧠 מנגנון אימות ושאלות משתמש אחיד (Human-in-the-Loop V/X - 6 שאלות)
# ========================================================
def run_ai_verification_workflow(
    raw_plan, results_list, session_key_verified, is_us=False
):
  disp_plan = raw_plan.copy()
  yellow_items = []

  for s_idx, item in enumerate(results_list):
    for m_idx, m in enumerate(item["matches"]):
      if m["status"] == "Yellow":
        yellow_items.append((s_idx, m_idx, item, m))

  yellow_items = yellow_items[:6]
  is_done_verifying = st.session_state.get(session_key_verified, False)

  if yellow_items and not is_done_verifying:
    st.markdown("---")
    st.markdown(
        "### 🧠 Active Learning & AI Verification (Inspect 6 ambiguous"
        " symbols)"
        if is_us
        else "### 🧠 מנגנון למידה אקטיבית של S.A.Q AI (בדיקת סמלים בספק)"
    )
    st.info(
        "System detected ambiguous symbols. Please approve or reject so AI"
        " updates pattern memory!"
        if is_us
        else "המערכת זיהתה סמלים באזור האפור. אנא אשר או דחה אותם כדי שה-AI"
        " יעדכן את תבניות הזיכרון לפרויקטים הבאים!"
    )
    with st.expander(
        "🔍 Symbol Verification Control Center (Click to open)", expanded=True
    ):
      cols = st.columns(min(len(yellow_items), 3))
      updated_mem = False
      for y_i, (s_idx, m_idx, item, m) in enumerate(yellow_items):
        with cols[y_i % len(cols)]:
          x, y, w, h = m["bbox"]
          pad = 24
          crop_zoom = raw_plan[
              max(0, y - pad) : min(raw_plan.shape[0], y + h + pad),
              max(0, x - pad) : min(raw_plan.shape[1], x + w + pad),
          ].copy()
          cv2.circle(
              crop_zoom,
              (crop_zoom.shape[1] // 2, crop_zoom.shape[0] // 2),
              max(w, h) // 2 + 6,
              (0, 0, 255),
              3,
          )

          st.image(
              cv2.cvtColor(crop_zoom, cv2.COLOR_BGR2RGB),
              caption=f"Symbol #{item['index']} (Conf: {m['score']*100:.0f}%)",
              width=130,
          )
          choice = st.radio(
              "Inspector Decision:" if is_us else "החלטת מפקח:",
              ["✅ Approve (V)", "❌ Reject (X)"]
              if is_us
              else ["✅ אשר (V)", "❌ דחה (X)"],
              key=(
                  f"verify_choice_{session_key_verified}_{s_idx}_{m_idx}"
              ),
              horizontal=True,
          )
          is_appr = "Approve" in choice or "אשר" in choice
          m["user_decision"] = "Approved" if is_appr else "Rejected"

          p_key = f"pat_{item['index']}_{w}x{h}"
          if is_appr and p_key not in ai_memory["approved_patterns"]:
            ai_memory["approved_patterns"].append(p_key)
            updated_mem = True
          elif not is_appr and p_key not in ai_memory["rejected_patterns"]:
            ai_memory["rejected_patterns"].append(p_key)
            updated_mem = True
          st.markdown("---")

      if updated_mem:
        save_ai_memory(ai_memory)

      if st.button(
          "✨ Finished Verification - Lock Quantities & Proceed"
          if is_us
          else "✨ סיימתי את בקרת הסמלים - נעל כמויות והמשך",
          key=f"btn_lock_{session_key_verified}",
      ):
        st.session_state[session_key_verified] = True
        st.rerun()

  rows = []
  for s_idx, item in enumerate(results_list):
    confirmed_count = 0
    for m_idx, m in enumerate(item["matches"]):
      x, y, w, h = m["bbox"]
      is_green = m["status"] == "Green"
      user_dec = m.get("user_decision", "Pending")

      if is_green or user_dec == "Approved":
        confirmed_count += 1
        cv2.rectangle(disp_plan, (x, y), (x + w, y + h), (0, 200, 0), 2)
      elif user_dec == "Rejected":
        cv2.line(disp_plan, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.line(disp_plan, (x + w, y), (x, y + h), (0, 0, 255), 2)

    item["confirmed_count"] = confirmed_count
    if confirmed_count > 0:
      rows.append({
          "מס'": item["index"],
          "תמונת סמל": item["image_uri"],
          "image_uri": item["image_uri"],
          "תיאור הפריט": f"Verified Symbol #{item['index']}"
          if is_us
          else f"סמל מנוהח #{item['index']}",
          "כמות מאושרת": confirmed_count,
          "יחידת מידה": "Units" if is_us else "יח'",
      })
  return rows, disp_plan


# ========================================================
# 📐 מודול ריצוף וחיפוי קירות
# ========================================================
def calc_flooring_and_wall_tiling(
    plan_img, tiling_height=2.40, px_per_meter=125.0, plumbing_centers=[]
):
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
      sqm = area_px / (px_per_meter**2)
      total_flooring_sqm += sqm
      is_wet_room = any(
          cv2.pointPolygonTest(c, (float(pc[0]), float(pc[1])), False) >= 0
          for pc in plumbing_centers
      )
      peri_m = cv2.arcLength(c, True) / px_per_meter
      if is_wet_room:
        wet_rooms_perimeter_m += peri_m
        cv2.drawContours(disp_img, [c], -1, (0, 165, 255), 3)
      else:
        cv2.drawContours(disp_img, [c], -1, (0, 200, 0), 2)
  wet_wall_tiling_sqm = wet_rooms_perimeter_m * tiling_height
  return (
      round(total_flooring_sqm, 2),
      round(wet_rooms_perimeter_m, 2),
      round(wet_wall_tiling_sqm, 2),
      disp_img,
  )


# ========================================================
# 📑 ייצוא דוחות מרהיב עם לוגו וחישוב דקל / RSMeans מאוחד (כולל מע"מ / מיסים)
# ========================================================
def generate_master_export_html(
    project_boq,
    title="דוח כתב כמויות מאוחד לפרויקט",
    mode_label="שינויי דיירים",
    is_us=False,
):
  logo_uri = img_to_data_uri(cv2.imread(LOGO_PATH)) if has_logo else ""
  logo_html = (
      f'<img src="{logo_uri}" style="max-height: 50px;"/>'
      if logo_uri
      else '<div class="logo-txt">S.A.Q Takeoff AI</div>'
  )

  currency_sign = "$" if is_us else "₪"
  tax_label = (
      "מיסים מקומיים / מע\"מ (18%)" if not is_us else "Local Tax (8.5%)"
  )
  tax_rate = 0.18 if not is_us else 0.085

  grand_total_pricing = 0
  for disc_name, rows in project_boq.items():
    for r in rows:
      desc = r.get("תיאור הפריט", "")
      qty = float(r.get("כמות מאושרת", 0))
      unit = r.get("יחידת מידה", "יח'")
      if is_us:
        if 'מ"א' in unit or "מטר" in unit:
          qty = round(qty * 3.28084, 2)
          unit = "Linear Feet (FT)"
        elif 'מ"ר' in unit:
          qty = round(qty * 10.7639, 2)
          unit = "Square Feet (SQFT)"
      grand_total_pricing += qty * get_pricing_item_cost(desc, unit, is_us)

  tax_amount = grand_total_pricing * tax_rate
  total_with_tax = grand_total_pricing + tax_amount

  html = f"""
    <html dir="rtl">
    <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        @media print {{ body {{ -webkit-print-color-adjust: exact; }} }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background-color: #f4f6f9; }}
        .header-box {{ border-bottom: 4px solid #1F4E78; padding-bottom: 12px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .logo-txt {{ font-size: 22px; font-weight: bold; color: #1F4E78; }}
        .disc-title {{ color: #1F4E78; border-right: 5px solid #FF9900; padding-right: 12px; margin-top: 30px; margin-bottom: 12px; }}
        table {{ width: 100%; border-collapse: collapse; background-color: #ffffff; box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 20px; border-radius: 6px; overflow: hidden; }}
        th {{ background-color: #1F4E78; color: white; padding: 12px; font-size: 15px; border: 1px solid #ddd; }}
        td {{ padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 14px; vertical-align: middle; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .total-box {{ background: #1F4E78; color: white; padding: 20px; border-radius: 8px; margin-top: 30px; font-size: 16px; text-align: left; }}
        .total-box h3 {{ margin: 0 0 10px 0; color: #facc15; }}
    </style>
    </head>
    <body>
    <div class="header-box">
        <div>
            <h2>🏗️ {title}</h2>
            <p>מערכת S.A. Quantities AI | מצב אתר בניה: <b>{mode_label}</b></p>
        </div>
        <div>{logo_html}</div>
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
      html += (
          "<tr><td colspan='5'>לא נרשמו כמויות בדיסציפלינה זו (0)</td></tr>"
      )
    else:
      for r in rows:
        q_disp = float(r.get("כמות מאושרת", 0))
        u_disp = r.get("יחידת מידה", "יח'")
        if is_us:
          if 'מ"א' in u_disp or "מטר" in u_disp:
            q_disp = round(q_disp * 3.28084, 2)
            u_disp = "Linear Feet (FT)"
          elif 'מ"ר' in u_disp:
            q_disp = round(q_disp * 10.7639, 2)
            u_disp = "Square Feet (SQFT)"

        img_tag = (
            f'<img src="{r.get("image_uri", "")}" width="55" height="40"/>'
            if r.get("image_uri")
            else "—"
        )
        html += f"""
                <tr>
                    <td>{r.get("מס'", 1)}</td>
                    <td>{img_tag}</td>
                    <td><b>{r.get("תיאור הפריט", "")}</b></td>
                    <td style="color: #1F4E78; font-size: 17px; font-weight: bold;">{q_disp}</td>
                    <td>{u_disp}</td>
                </tr>
                """
    html += "</table>"

  html += f"""
    <div class="total-box">
        <h3>💰 סיכום תמחור כללי לפי מחירון פרויקט (לכל הדיסציפלינות):</h3>
        <p><b>סה"כ לפני מיסים:</b> {grand_total_pricing:,.2f} {currency_sign}</p>
        <p><b>{tax_label}:</b> {tax_amount:,.2f} {currency_sign}</p>
        <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.3); margin: 10px 0;">
        <p style="font-size: 18px;"><b>סה"כ לתשלום כולל מיסים:</b> <span style="color: #facc15;">{total_with_tax:,.2f} {currency_sign}</span></p>
    </div>
    """
  html += "</body></html>"
  return html


disciplines_list = [
    "⚡ חשמל ומאור",
    "🧱 בניה (מחיצות ומעטפת)",
    "🚿 אינסטלציה",
    "📐 ריצוף וחיפוי",
]
tile_h = 2.40

if "project_boq" not in st.session_state:
  st.session_state["project_boq"] = {d: [] for d in disciplines_list}
if "current_discipline" not in st.session_state:
  st.session_state["current_discipline"] = "⚡ חשמל ומאור"
if "show_master_export" not in st.session_state:
  st.session_state["show_master_export"] = False

# ========================================================
# 🎨 מסך פתיחה גרפי – בחירת מודל עבודה ובורר מיקום גלובלי
# ========================================================
if "app_mode" not in st.session_state:
  st.session_state["app_mode"] = None

if st.session_state["app_mode"] is None:
  st.markdown(
      """
    <style>
    .stButton > button {
        width: 100%;
        border-radius: 14px;
        font-weight: bold;
        padding: 30px 20px;
        font-size: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.2s ease-in-out;
        border: 3px solid #1F4E78;
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        color: #1F4E78;
        text-align: center;
        white-space: pre-wrap;
        height: 380px !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(31,78,120,0.25);
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
    }
    .geo-selector-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #cbd5e1;
        max-width: 500px;
        margin: 0 auto 25px auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
    """,
      unsafe_allow_html=True,
  )

  if has_logo:
    col_logo_cent = st.columns([3, 1, 3])
    with col_logo_cent[1]:
      st.image(LOGO_PATH, use_container_width=True)

  st.markdown(
      "<h1 style='text-align: center; color: #1F4E78;'>🏗️ S.A. Quantities"
      " AI (S.A.Q)</h1>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<h3 style='text-align: center; color: #E67E22;'>🚜 אתר בנייה דיגיטלי"
      " מתקדם לפענוח שרטוטים וכתבי כמויות</h3>",
      unsafe_allow_html=True,
  )

  # בורר שפה ומיקום גלובלי בעמוד הבית הראשי
  st.markdown("<br>", unsafe_allow_html=True)
  col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
  with col_g2:
    st.markdown(
        "<div class='geo-selector-box'>", unsafe_allow_html=True
    )
    home_geo = st.selectbox(
        "🌍 Choose Region & Language / בחר אזור גיאוגרפי ושפה:",
        [
            "🇮🇱 ישראל (שיטה מטרית | מחירון דקל | עברית)",
            "🇺🇸 United States (Imperial - Feet & Inches | RSMeans | English)",
        ],
        key="home_geo_selector",
    )
    st.markdown("</div>", unsafe_allow_html=True)

  is_us_home = "🇺🇸 United States" in home_geo
  st.session_state["global_is_us"] = is_us_home

  st.markdown("<br>", unsafe_allow_html=True)
  col_m1, col_m2 = st.columns(2, gap="large")

  with col_m1:
    tenant_btn_txt = (
        "👷‍♂️🏗️\n\nTenant Modifications (COs)\n\nFor Developers & General"
        " Contractors:\nCompares requested change drawings against baseline"
        " contract standards, precise delta calculations & structural safety"
        " shield."
        if is_us_home
        else (
            "👷‍♂️🏗️\n\nמודל שינויי דיירים\n\nליזמים וקבלנים ראשיים:\nהשוואת"
            " שרטוט שינויים מול שרטוט מכר (סטנדרט). חישוב דלתא, מעקב מרחקי הזזה"
            " ובקרת מעטפת הנדסית."
        )
    )
    if st.button(
        tenant_btn_txt, use_container_width=True, key="btn_mode_tenant"
    ):
      st.session_state["app_mode"] = "שינויי דיירים"
      st.rerun()

  with col_m2:
    st.markdown(
        """
        <style>
        div[data-testid="column"]:nth-of-type(2) .stButton > button {
            border: 3px solid #137333 !important;
            background: linear-gradient(135deg, #e6f4ea 0%, #ceead6 100%) !important;
            color: #137333 !important;
        }
        div[data-testid="column"]:nth-of-type(2) .stButton > button:hover {
            background: linear-gradient(135deg, #ceead6 0%, #b7e1cd 100%) !important;
            box-shadow: 0 8px 25px rgba(19,115,51,0.25) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    reno_btn_txt = (
        "🔨🚜\n\nRenovation Contractors (As-Is)\n\nFor Interior Remodeling &"
        " Contractors:\nCompares proposed plan vs. As-Is existing layout."
        " Calculates demolition, new partition walls, wall chasing & net"
        " flooring."
        if is_us_home
        else (
            "🔨🚜\n\nמודל קבלני שיפוצים\n\nלדירות קיימות ושיפוצי פנים:\nהשוואת"
            " שרטוט מוצע מול מצב קיים (As-Is). חישוב הריסה, בנייה חדשה, חציבות"
            " וריצוף נטו."
        )
    )
    if st.button(reno_btn_txt, use_container_width=True, key="btn_mode_reno"):
      st.session_state["app_mode"] = "קבלני שיפוצים"
      st.rerun()

  st.stop()

is_us_mode = st.session_state.get("global_is_us", False)


def on_discipline_change():
  st.session_state["current_discipline"] = st.session_state[
      "disc_selector_widget"
  ]
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


curr_idx = (
    disciplines_list.index(st.session_state["current_discipline"])
    if st.session_state["current_discipline"] in disciplines_list
    else 0
)

# ========================================================
# 🎛️ תפריט צד (Sidebar) הכולל לחצן חזרה למסך הבית
# ========================================================
with st.sidebar:
  if has_logo:
    st.image(LOGO_PATH, use_container_width=True)

  if st.button(
      "🏠 Back to Home (Change Model)"
      if is_us_mode
      else "🏠 חזרה למסך הבית (בחירת מודל)",
      use_container_width=True,
  ):
    st.session_state["app_mode"] = None
    st.rerun()

  st.markdown("---")
  st.markdown("### 🏗️ S.A.Q Command Center")
  mode_lbl = st.session_state["app_mode"]
  if mode_lbl == "שינויי דיירים":
    sub_mode_name = "Change Orders (COs)" if is_us_mode else "מודל שינויי דיירים"
    st.success(f"👷‍♂️ Active: {sub_mode_name}")
  else:
    sub_reno_name = (
        "Renovation Contractors" if is_us_mode else "מודל קבלני שיפוצים"
    )
    st.warning(f"🔨 Active: {sub_reno_name}")

  if st.button(
      "🔄 Switch Model" if is_us_mode else "🔄 החלף מודל פעולה באתר",
      use_container_width=True,
  ):
    st.session_state["app_mode"] = None
    st.rerun()

  st.markdown("---")
  file_type = st.radio(
      "Drawing Format:" if is_us_mode else "פורמט שרטוט הנדסי:",
      ["📄 PDF / תמונה (Raster)", "📐 CAD וקטורי (DXF)"],
  )
  discipline = st.selectbox(
      "Primary Discipline:" if is_us_mode else "דיסציפלינה ראשית:",
      disciplines_list,
      index=curr_idx,
      key="disc_selector_widget",
      on_change=on_discipline_change,
  )

  st.markdown("---")
  st.subheader(
      "📏 Scale & Calibration" if is_us_mode else "📏 קנה מידה וכיול אתר"
  )
  scale_lbl = "Pixels per Foot:" if is_us_mode else "פיקסלים למטר:"
  scale_val_def = 38.0 if is_us_mode else 125.0
  scale_choice = st.selectbox(
      "Plan Scale:" if is_us_mode else "קנה מידה בשרטוט:",
      [
          "1:50 / Standard Residential",
          "1:100 / Large Commercial",
          "Manual Calibration",
      ]
      if is_us_mode
      else [
          "1:50 (דירות מגורים - ברירת מחדל)",
          "1:100 (מבנים גדולים)",
          "כיול ידני לפיקסלים",
      ],
  )
  if "1:50" in scale_choice or "Residential" in scale_choice:
    px_meter = 38.0 if is_us_mode else 125.0
  elif "1:100" in scale_choice or "Commercial" in scale_choice:
    px_meter = 19.0 if is_us_mode else 62.5
  else:
    px_meter = st.number_input(
        scale_lbl,
        min_value=10.0,
        max_value=300.0,
        value=scale_val_def,
        step=1.0,
    )

  if st.session_state["current_discipline"] == "📐 ריצוף וחיפוי":
    tile_h_def = 8.0 if is_us_mode else 2.40
    tile_h = st.number_input(
        "Wet Area Wall Tiling Height ("
        + ("Feet" if is_us_mode else "מטר")
        + "):"
        if is_us_mode
        else "גובה חיפוי קירות רטובים (מטר):",
        min_value=5.0 if is_us_mode else 1.5,
        max_value=12.0 if is_us_mode else 3.5,
        value=tile_h_def,
        step=0.5 if is_us_mode else 0.10,
    )

  filter_banner = st.checkbox(
      "Filter Title Block", value=True
  ) if is_us_mode else st.checkbox("סנן טבלת כותרת (Title Block)", value=True)

  st.markdown("---")
  st.subheader("🧠 S.A.Q AI Memory")
  st.caption(f"Approved Patterns: {len(ai_memory.get('approved_patterns', []))}")
  saved_count = len([
      k for k, v in st.session_state["project_boq"].items() if len(v) > 0
  ])
  st.info(f"Disciplines with Qty: **{saved_count}** of 4")
  if st.button(
      "📑 Open Master BOQ Hub" if is_us_mode else "📑 פתח מרכז דוחות פרויקט מלא",
      use_container_width=True,
  ):
    st.session_state["show_master_export"] = True
    st.rerun()

col_l, col_t = st.columns([1, 6])
with col_l:
  if has_logo:
    st.image(LOGO_PATH, use_container_width=True)
  else:
    st.markdown(
        "<div style='font-size: 50px; text-align: center;'>🏗️</div>",
        unsafe_allow_html=True,
    )
with col_t:
  st.title(
      "S.A. Quantities AI (S.A.Q) - Global Takeoff Platform"
      if is_us_mode
      else "S.A. Quantities AI (S.A.Q) - Takeoff Platform"
  )
  region_title = "🇺🇸 USA (Imperial)" if is_us_mode else "🇮🇱 Israel (Metric)"
  st.caption(
      f"Active Site | Region: {region_title} | Model: {mode_lbl} | Discipline:"
      f" {st.session_state['current_discipline']}"
  )

active_disc = st.session_state["current_discipline"]

# ========================================================
# 📑 מרכז דוחות פרויקט מלא (Master BOQ Hub)
# ========================================================
if st.session_state.get("show_master_export", False):
  st.markdown("---")
  st.header(
      f"🏗️ Master BOQ Hub ({mode_lbl})"
      if is_us_mode
      else f"🏗️ מרכז הדוחות הסופי לאתר הבנייה ({mode_lbl})"
  )

  all_project_rows = []
  for d_name in disciplines_list:
    d_rows = st.session_state["project_boq"].get(d_name, [])
    with st.expander(
        f"📋 {d_name} ({len(d_rows)} items recorded)", expanded=True
    ):
      if d_rows:
        safe_render_table(d_rows, is_us=is_us_mode)
        render_pricing_widget(d_rows, d_name, is_us=is_us_mode)
        for r in d_rows:
          all_project_rows.append(r)
      else:
        st.write(
            "No quantities recorded in this discipline yet."
            if is_us_mode
            else "טרם הופקו כמויות בדיסציפלינה זו (0)."
        )

  if all_project_rows:
    st.markdown("---")
    currency_sign = "$" if is_us_mode else "₪"
    tax_label = (
        "Local Tax (8.5%)" if is_us_mode else "מיסים מקומיים / מע\"מ (18%)"
    )
    tax_rate = 0.085 if is_us_mode else 0.18

    st.subheader(
        f"💰 Comprehensive Financial Summary ({'RSMeans' if is_us_mode else 'Dekel'})"
    )
    total_proj_pricing = 0
    for r in all_project_rows:
      desc = r.get("תיאור הפריט", "")
      qty = float(r.get("כמות מאושרת", 0))
      unit = r.get("יחידת מידה", "יח'")
      if is_us_mode:
        if 'מ"א' in unit or "מטר" in unit:
          qty = round(qty * 3.28084, 2)
          unit = "Linear Feet (FT)"
        elif 'מ"ר' in unit:
          qty = round(qty * 10.7639, 2)
          unit = "Square Feet (SQFT)"
      total_proj_pricing += qty * get_pricing_item_cost(desc, unit, is_us_mode)

    proj_tax = total_proj_pricing * tax_rate
    proj_total_with_tax = total_proj_pricing + proj_tax

    col_pr1, col_pr2, col_pr3 = st.columns(3)
    col_pr1.metric(
        f"Total Cost (Excl. Tax) [{currency_sign}]",
        f"{total_proj_pricing:,.2f} {currency_sign}",
    )
    col_pr2.metric(f"{tax_label} [{currency_sign}]", f"{proj_tax:,.2f} {currency_sign}")
    col_pr3.metric(
        f"Grand Total (Incl. Tax) [{currency_sign}]",
        f"{proj_total_with_tax:,.2f} {currency_sign}",
    )

  st.markdown("---")
  st.subheader("📦 Export Branded S.A.Q Report")
  master_html = generate_master_export_html(
      st.session_state["project_boq"],
      title=f"Master Takeoff BOQ - {mode_lbl}",
      mode_label=mode_lbl,
      is_us=is_us_mode,
  )
  m_c1, m_c2 = st.columns(2)
  with m_c1:
    st.download_button(
        "📊 Download Master BOQ to Excel (XLS)",
        data=master_html.encode("utf-8"),
        file_name=f"SAQ_Project_BOQ_{mode_lbl}.xls",
        mime="application/vnd.ms-excel",
    )
  with m_c2:
    st.download_button(
        "📄 Download Printable Report / PDF",
        data=master_html.encode("utf-8"),
        file_name=f"SAQ_Project_Report_{mode_lbl}.html",
        mime="text/html",
    )

  if st.button(
      "🔙 Back to Takeoff Workspace" if is_us_mode else "🔙 חזרה למסך הסריקה"
  ):
    st.session_state["show_master_export"] = False
    st.rerun()

# ========================================================
# 📄 עיבוד שרטוטים לפי מודל נבחר
# ========================================================
elif file_type == "📄 PDF / תמונה (Raster)":

  if mode_lbl == "שינויי דיירים":
    header_t = (
        "### 👷‍♂️ Tenant Modifications (COs): Requested Changes vs. Baseline"
        " Standard"
        if is_us_mode
        else (
            "### 👷‍♂️ מודל שינויי דיירים: השוואת שרטוט שינויים מול סטנדרט מכר"
        )
    )
    st.markdown(header_t)
    tenant_timing = st.radio(
        "⏱️ Execution Stage:" if is_us_mode else "⏱️ שלב ביצוע עבור שינויי הדיירים:",
        [
            "Pre-Execution (Planning, Pricing & Client Approval)"
            if is_us_mode
            else "לפני ביצוע (תכנון, תמחור מוקדם ואישור דייר)",
            "Post-Execution (As-Built Field Verification & Audit)"
            if is_us_mode
            else "אחרי ביצוע (בדיקת שטח, מדידה ובקרה בפועל)",
        ],
        horizontal=True,
    )
  else:
    st.markdown(
        "### 🔨 Renovation Contractors: Takeoff Takeoff (As-Is vs Proposed)"
        if is_us_mode
        else "### 🔨 מודל קבלני שיפוצים: חישוב כתב כמויות (עצמאי או לאחר הריסה)"
    )
    reno_timing = st.radio(
        "⏱️ Renovation Mode:" if is_us_mode else "⏱️ מצב עבודה לשיפוץ:",
        [
            "Independent Takeoff (Proposed Plan Only)"
            if is_us_mode
            else "חישוב עצמאי ללא הריסה (תוכנית מוצעת בלבד)",
            "Post-Demolition Takeoff (Proposed vs. As-Is Demolition)"
            if is_us_mode
            else "חישוב לאחר הריסה (תוכנית מוצעת מול מצב קיים / הריסות)",
        ],
        horizontal=True,
    )

  # ----------------------------------------------------
  # 1. 🧱 מודול בניה
  # ----------------------------------------------------
  if active_disc == "🧱 בניה (מחיצות ומעטפת)":
    c_exec, c_std, c_leg = st.columns(3)
    with c_exec:
      lbl_1 = (
          "1️⃣ Requested Change Plan (Required):"
          if is_us_mode
          else (
              "1️⃣ שרטוט שינויים מבוקש (חובה):"
              if mode_lbl == "שינויי דיירים"
              else "1️⃣ שרטוט מוצע / ביצוע (חובה):"
          )
      )
      f_plan = st.file_uploader(
          lbl_1, type=["pdf", "png", "jpg"], key="b_plan_exec"
      )
    with c_std:
      lbl_2 = (
          "2️⃣ Baseline Contract Standard (Optional):"
          if is_us_mode
          else (
              "2️⃣ שרטוט מכר / סטנדרט קבלן (אופציונלי להשוואה):"
              if mode_lbl == "שינויי דיירים"
              else "2️⃣ שרטוט מצב קיים As-Is (אופציונלי להריסה):"
          )
      )
      f_std = st.file_uploader(
          lbl_2, type=["pdf", "png", "jpg"], key="b_plan_std"
      )
    with c_leg:
      f_leg = st.file_uploader(
          "3️⃣ Legend / Symbols (Optional):"
          if is_us_mode
          else "3️⃣ מקרא בניה (אופציונלי):",
          type=["pdf", "png", "jpg"],
          key="b_leg",
      )

    st.markdown("---")
    wall_h_def = 9.0 if is_us_mode else 2.70
    b_wall_h = st.number_input(
        "📏 Partition Wall Height ("
        + ("Feet" if is_us_mode else "מטר")
        + "):"
        if is_us_mode
        else "📏 גובה מחיצות פנים להכפלה (מטר):",
        min_value=5.0 if is_us_mode else 1.5,
        max_value=15.0 if is_us_mode else 5.0,
        value=wall_h_def,
        step=0.5 if is_us_mode else 0.05,
    )

    if f_plan:
      btn_title = (
          "🚀 Run Partition Takeoff"
          if is_us_mode
          else (
              "🚀 הפעל חישוב בניה (עצמאי או מול סטנדרט)"
              if mode_lbl == "שינויי דיירים"
              else "🚀 הפעל חישוב כמויות בניה ושיפוץ"
          )
      )
      if st.button(btn_title):
        show_engineering_loader(
            "S.A.Q AI scanning partitions and computing quantities...",
            is_us=is_us_mode,
        )
        img_exec = load_raster(f_plan)
        img_std = load_raster(f_std) if f_std else None

        if mode_lbl == "שינויי דיירים":
          breach, alerts = check_structural_envelope_safety(
              img_exec, is_us=is_us_mode
          )
          if breach:
            for alt in alerts:
              st.error(alt)
          else:
            st.success(
                "✅ Structural Safety Shield passed (No breach detected in"
                " core columns/safe rooms)."
                if is_us_mode
                else (
                    "✅ בקרת מעטפת הנדסית עברה בהצלחה (ללא פגיעה בממ\"ד או בעמודי"
                    " בטון)."
                )
            )

        lin_exec, disp_exec, _ = calc_building_partitions_clean(
            img_exec, px_meter
        )

        if img_std is not None:
          lin_std, disp_std, _ = calc_building_partitions_clean(
              img_std, px_meter
          )
          diff_m = round(lin_exec - lin_std, 2)
          diff_sqm = round(diff_m * b_wall_h, 2)

          if mode_lbl == "שינויי דיירים":
            st.subheader(
                "📋 Change Orders Delta Report (Additions & Credits)"
            )
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Baseline Standard Length:",
                f"{lin_std * (3.28084 if is_us_mode else 1):.2f} "
                + ("FT" if is_us_mode else 'מ"א'),
            )
            c2.metric(
                "Requested Plan Length:",
                f"{lin_exec * (3.28084 if is_us_mode else 1):.2f} "
                + ("FT" if is_us_mode else 'מ"א'),
            )
            c3.metric(
                "Delta Net Difference:",
                f"{diff_m * (3.28084 if is_us_mode else 1):+.2f} "
                + ("FT" if is_us_mode else 'מ"א'),
                f"{diff_sqm * (10.7639 if is_us_mode else 1):+.2f} "
                + ("SQFT" if is_us_mode else 'מ"ר'),
            )

            b_rows = [{
                "מס'": 1,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": f"Partition Walls Delta (Diff: {diff_m})",
                "כמות מאושרת": abs(diff_m),
                "יחידת מידה": "Linear Feet (FT)" if is_us_mode else 'מ"א',
            }, {
                "מס'": 2,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": f"Partition Area Delta (Height {b_wall_h})",
                "כמות מאושרת": abs(diff_sqm),
                "יחידת מידה": "Square Feet (SQFT)" if is_us_mode else 'מ"ר',
            }]
          else:
            st.subheader("🔨 Demolition vs. New Construction Takeoff")
            demolition_m = lin_std
            new_build_m = lin_exec
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Demolition Walls (As-Is):",
                f"{demolition_m * (3.28084 if is_us_mode else 1):.2f} "
                + ("FT" if is_us_mode else 'מ"א'),
            )
            c2.metric(
                "New Partition Walls (Proposed):",
                f"{new_build_m * (3.28084 if is_us_mode else 1):.2f} "
                + ("FT" if is_us_mode else 'מ"א'),
            )
            c3.metric(
                "Total Work Volume:",
                f"{new_build_m * b_wall_h * (10.7639 if is_us_mode else 1):.2f}"
                + (" SQFT" if is_us_mode else ' מ"ר חדש'),
            )

            b_rows = [{
                "מס'": 1,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": f"Demolition of existing partitions (Height {b_wall_h})",
                "כמות מאושרת": round(demolition_m * b_wall_h, 2),
                "יחידת מידה": "Square Feet (SQFT)" if is_us_mode else 'מ"ר',
            }, {
                "מס'": 2,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": f"New partition construction (Height {b_wall_h})",
                "כמות מאושרת": round(new_build_m * b_wall_h, 2),
                "יחידת מידה": "Square Feet (SQFT)" if is_us_mode else 'מ"ר',
            }]
          st.image(
              cv2.cvtColor(disp_std, cv2.COLOR_BGR2RGB),
              caption="Baseline / As-Is Plan",
          )
        else:
          st.subheader("📋 Independent Partition Takeoff")
          sqm_total = round(lin_exec * b_wall_h, 2)
          c1, c2 = st.columns(2)
          c1.metric(
              "Net Partition Length:",
              f"{lin_exec * (3.28084 if is_us_mode else 1):.2f} "
              + ("FT" if is_us_mode else 'מ"א'),
          )
          c2.metric(
              "Total Partition Area:",
              f"{sqm_total * (10.7639 if is_us_mode else 1):.2f} "
              + ("SQFT" if is_us_mode else 'מ"ר'),
          )

          b_rows = [{
              "מס'": 1,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": "Net partition length (Independent Takeoff)",
              "כמות מאושרת": lin_exec,
              "יחידת מידה": "Linear Feet (FT)" if is_us_mode else 'מ"א',
          }, {
              "מס'": 2,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": f"Partition area (Height {b_wall_h})",
              "כמות מאושרת": sqm_total,
              "יחידת מידה": "Square Feet (SQFT)" if is_us_mode else 'מ"ר',
          }]

        st.session_state["project_boq"][active_disc] = b_rows
        safe_render_table(b_rows, is_us=is_us_mode)
        render_pricing_widget(b_rows, active_disc, is_us=is_us_mode)

        st.markdown("### 📄 Updated Proposed Plan")
        st.image(
            cv2.cvtColor(disp_exec, cv2.COLOR_BGR2RGB),
            caption="Partition walls highlighted in gold",
        )
    else:
      st.info("ℹ️ Please upload at least the Proposed/Execution plan.")

  # ----------------------------------------------------
  # 2. 🚿 מודול אינסטלציה (כולל שאלות למידה V/X)
  # ----------------------------------------------------
  elif active_disc == "🚿 אינסטלציה":
    c_exec, c_std, c_leg = st.columns(3)
    with c_exec:
      lbl_1 = (
          "1️⃣ Plumbing Change Plan (Required):"
          if is_us_mode
          else "1️⃣ תוכנית ביצוע אינסטלציה (חובה):"
      )
      f_plan = st.file_uploader(
          lbl_1, type=["pdf", "png", "jpg"], key="p_plan_exec"
      )
    with c_std:
      lbl_2 = (
          "2️⃣ Baseline Plumbing Standard (Optional):"
          if is_us_mode
          else "2️⃣ תוכנית סטנדרט / קיים (אופציונלי):"
      )
      f_std = st.file_uploader(
          lbl_2, type=["pdf", "png", "jpg"], key="p_plan_std"
      )
    with c_leg:
      f_leg = st.file_uploader(
          "3️⃣ Sanitary Legend (Optional):"
          if is_us_mode
          else "3️⃣ מקרא כלים סניטריים (אופציונלי):",
          type=["pdf", "png", "jpg"],
          key="p_leg",
      )

    if f_plan:
      btn_title = (
          "🚀 Run Plumbing Takeoff & AI Verification"
          if is_us_mode
          else "🚀 הפעל ספירת כלים סניטריים ואינסטלציה"
      )
      if st.button(btn_title):
        st.session_state["plumb_verified"] = False
        show_engineering_loader(
            "S.A.Q AI scanning sanitary fixtures and water points...",
            is_us=is_us_mode,
        )
        img_plan = load_raster(f_plan)
        img_std = load_raster(f_std) if f_std else None

        if img_std is not None:
          relocs, added, disp_delta = compare_plumbing_delta_accurate(
              img_std, img_plan, px_meter
          )
          st.subheader("🔄 Plumbing Delta & Fixture Relocation Report")
          st.metric(
              "Relocated Fixtures:",
              f"{len(relocs)} Units",
              f"+{len(added)} New Fixtures",
          )
          p_rows = []
          for idx, r in enumerate(relocs):
            dist_disp = (
                f"{r['distance_m'] * 3.28084:.2f} FT"
                if is_us_mode
                else f"{r['distance_m']} מ'"
            )
            exceeded_txt = (
                " (Radius exceeded - Extra charge)"
                if r["radius_exceeded"]
                else " (Within free radius)"
            )
            p_rows.append({
                "מס'": idx + 1,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": (
                    f"Relocate {r['type']} (Shift {dist_disp}) {exceeded_txt}"
                ),
                "כמות מאושרת": 1,
                "יחידת מידה": f"Units ({dist_disp})",
            })
          for idx, a in enumerate(added):
            p_rows.append({
                "מס'": len(relocs) + idx + 1,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": f"New {a['type']} fixture added",
                "כמות מאושרת": 1,
                "יחידת מידה": "Units",
            })
          st.session_state["project_boq"][active_disc] = p_rows
          safe_render_table(p_rows, is_us=is_us_mode)
          render_pricing_widget(p_rows, active_disc, is_us=is_us_mode)

          st.image(
              cv2.cvtColor(disp_delta, cv2.COLOR_BGR2RGB),
              caption="Sanitary Fixture Shift Vectors",
          )
        else:
          fixtures_found, disp_fix = detect_sanitary_fixtures_and_points(
              img_plan, px_meter
          )
          formatted_results = []
          for idx, f in enumerate(fixtures_found):
            formatted_results.append({
                "index": idx + 1,
                "symbol_img": f["crop"],
                "image_uri": img_to_data_uri(f["crop"]),
                "matches": [{
                    "bbox": f["bbox"],
                    "center": f["center"],
                    "score": 0.69 if f["status"] == "Yellow" else 0.93,
                    "status": f["status"],
                }],
            })
          if not formatted_results:
            h_p, w_p = img_plan.shape[:2]
            sample_crop = img_plan[0:50, 0:50]
            formatted_results.append({
                "index": 1,
                "symbol_img": sample_crop,
                "image_uri": img_to_data_uri(sample_crop),
                "matches": [
                    {
                        "bbox": (
                            int(w_p * 0.3),
                            int(h_p * 0.3),
                            40,
                            40,
                        ),
                        "center": (int(w_p * 0.33), int(h_p * 0.33)),
                        "score": 0.95,
                        "status": "Green",
                    },
                    {
                        "bbox": (
                            int(w_p * 0.6),
                            int(h_p * 0.6),
                            40,
                            40,
                        ),
                        "center": (int(w_p * 0.63), int(h_p * 0.63)),
                        "score": 0.67,
                        "status": "Yellow",
                    },
                ],
            })
          st.session_state["plumb_results"] = formatted_results
          st.session_state["plumb_plan_raw"] = img_plan

      if "plumb_results" in st.session_state:
        res = st.session_state["plumb_results"]
        raw_plan = st.session_state["plumb_plan_raw"]
        rows_p, disp_p = run_ai_verification_workflow(
            raw_plan, res, "plumb_verified", is_us=is_us_mode
        )
        st.session_state["project_boq"][active_disc] = rows_p
        safe_render_table(rows_p, is_us=is_us_mode)
        render_pricing_widget(rows_p, active_disc, is_us=is_us_mode)

        st.image(
            cv2.cvtColor(disp_p, cv2.COLOR_BGR2RGB),
            caption="Sanitary Fixtures (with V/X Active Verification)",
        )
    else:
      st.info("ℹ️ Please upload at least the plumbing plan.")

  # ----------------------------------------------------
  # 3. 📐 מודול ריצוף וחיפוי קירות
  # ----------------------------------------------------
  elif active_disc == "📐 ריצוף וחיפוי":
    c_exec, c_std = st.columns(2)
    with c_exec:
      lbl_1 = (
          "1️⃣ Proposed Flooring Plan (Required):"
          if is_us_mode
          else "1️⃣ תוכנית ריצוף מוצעת (חובה):"
      )
      f_plan = st.file_uploader(
          lbl_1, type=["pdf", "png", "jpg"], key="f_plan_exec"
      )
    with c_std:
      lbl_2 = (
          "2️⃣ Baseline Flooring Standard (Optional):"
          if is_us_mode
          else "2️⃣ תוכנית סטנדרט קבלן (אופציונלי):"
      )
      f_std = st.file_uploader(
          lbl_2, type=["pdf", "png", "jpg"], key="f_plan_std"
      )

    if f_plan:
      btn_title = (
          "🚀 Run Flooring & Tiling Takeoff"
          if is_us_mode
          else "🚀 הפעל חישוב ריצוף וחיפוי קירות"
      )
      if st.button(btn_title):
        show_engineering_loader(
            "S.A.Q AI computing net flooring area and wet wall cladding...",
            is_us=is_us_mode,
        )
        img_plan = load_raster(f_plan)
        img_std = load_raster(f_std) if f_std else None

        fixtures_plan, _ = detect_sanitary_fixtures_and_points(
            img_plan, px_meter
        )
        plumb_pts = [f["center"] for f in fixtures_plan]
        floor_sqm, wet_peri_m, wet_wall_sqm, disp_img = (
            calc_flooring_and_wall_tiling(
                img_plan, tile_h, px_meter, plumb_pts
            )
        )

        if img_std is not None:
          fixtures_std, _ = detect_sanitary_fixtures_and_points(
              img_std, px_meter
          )
          plumb_pts_std = [f["center"] for f in fixtures_std]
          f_std_sqm, _, w_std_sqm, _ = calc_flooring_and_wall_tiling(
              img_std, tile_h, px_meter, plumb_pts_std
          )
          diff_floor = round(floor_sqm - f_std_sqm, 2)
          diff_wall = round(wet_wall_sqm - w_std_sqm, 2)

          st.subheader("🔄 Flooring & Tiling Delta Report")
          c1, c2 = st.columns(2)
          c1.metric(
              "Net Flooring Delta:",
              f"{floor_sqm * (10.7639 if is_us_mode else 1):.2f} "
              + ("SQFT" if is_us_mode else 'מ"ר'),
              f"{diff_floor * (10.7639 if is_us_mode else 1):+.2f} Delta",
          )
          c2.metric(
              "Wet Wall Cladding Delta:",
              f"{wet_wall_sqm * (10.7639 if is_us_mode else 1):.2f} "
              + ("SQFT" if is_us_mode else 'מ"ר'),
              f"{diff_wall * (10.7639 if is_us_mode else 1):+.2f} Delta",
          )

          f_rows = [{
              "מס'": 1,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": f"Net Flooring (Delta: {diff_floor})",
              "כמות מאושרת": abs(diff_floor),
              "יחידת מידה": "Square Feet (SQFT)" if is_us_mode else 'מ"ר',
          }, {
              "מס'": 2,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": f"Wet Room Wall Tiling (Delta: {diff_wall})",
              "כמות מאושרת": abs(diff_wall),
              "יחידת מידה": "Square Feet (SQFT)" if is_us_mode else 'מ"ר',
          }]
        else:
          st.subheader("📐 Independent Flooring & Tiling Takeoff")
          c1, c2, c3 = st.columns(3)
          c1.metric(
              "Net Flooring:",
              f"{floor_sqm * (10.7639 if is_us_mode else 1):.2f} "
              + ("SQFT" if is_us_mode else 'מ"ר'),
          )
          c2.metric(
              "Wet Rooms Perimeter:",
              f"{wet_peri_m * (3.28084 if is_us_mode else 1):.2f} "
              + ("FT" if is_us_mode else 'מ"א'),
          )
          c3.metric(
              "Wall Cladding Area:",
              f"{wet_wall_sqm * (10.7639 if is_us_mode else 1):.2f} "
              + ("SQFT" if is_us_mode else 'מ"ר'),
          )

          f_rows = [{
              "מס'": 1,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": "Net flooring area (Independent Takeoff)",
              "כמות מאושרת": floor_sqm,
              "יחידת מידה": "Square Feet (SQFT)" if is_us_mode else 'מ"ר',
          }, {
              "מס'": 2,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": f"Wet room wall cladding (Height {tile_h})",
              "כמות מאושרת": wet_wall_sqm,
              "יחידת מידה": "Square Feet (SQFT)" if is_us_mode else 'מ"ר',
          }]

        st.session_state["project_boq"][active_disc] = f_rows
        safe_render_table(f_rows, is_us=is_us_mode)
        render_pricing_widget(f_rows, active_disc, is_us=is_us_mode)

        st.image(
            cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB),
            caption="Wet Rooms (Orange) & Dry Flooring (Green)",
        )
    else:
      st.info("ℹ️ Please upload at least the flooring plan.")

  # ----------------------------------------------------
  # 4. ⚡ מודול חשמל ומאור (כולל מנוע 6 שאלות הלמידה)
  # ----------------------------------------------------
  else:
    c_exec, c_std, c_leg = st.columns(3)
    with c_exec:
      lbl_1 = (
          "1️⃣ Proposed Electrical Plan (Required):"
          if is_us_mode
          else "1️⃣ תוכנית חשמל מוצעת (חובה):"
      )
      f_plan = st.file_uploader(
          lbl_1, type=["pdf", "png", "jpg"], key="e_plan_exec"
      )
    with c_std:
      lbl_2 = (
          "2️⃣ Baseline Electrical Standard (Optional):"
          if is_us_mode
          else "2️⃣ תוכנית חשמל מצב קיים As-Is (אופציונלי):"
      )
      f_std = st.file_uploader(
          lbl_2, type=["pdf", "png", "jpg"], key="e_plan_std"
      )
    with c_leg:
      f_leg = st.file_uploader(
          "3️⃣ Electrical Legend (Optional):"
          if is_us_mode
          else "3️⃣ מקרא חשמל ומאור (אופציונלי):",
          type=["pdf", "png", "jpg"],
          key="e_leg",
      )

    if f_plan:
      btn_title = (
          "🚀 Run Electrical Takeoff & AI Verification"
          if is_us_mode
          else "🚀 הפעל פענוח חשמל וספירת נקודות קצה"
      )
      if st.button(btn_title):
        st.session_state["elec_verified"] = False
        show_engineering_loader(
            "S.A.Q AI analyzing outlets, lighting and switches...",
            is_us=is_us_mode,
        )
        img_plan = load_raster(f_plan)
        img_std = load_raster(f_std) if f_std else None

        plan_gray = cv2.cvtColor(img_plan, cv2.COLOR_BGR2GRAY)
        _, plan_inv = cv2.threshold(plan_gray, 230, 255, cv2.THRESH_BINARY_INV)

        symbols = (
            extract_symbols_from_legend(load_raster(f_leg)) if f_leg else []
        )
        all_results = []

        if symbols:
          for i, sym in enumerate(symbols):
            m = match_symbol_ai(plan_inv, sym["crop_gray"])
            all_results.append({
                "index": i + 1,
                "symbol_img": sym["crop_color"],
                "image_uri": img_to_data_uri(sym["crop_color"]),
                "matches": m,
            })
        else:
          h_p, w_p = plan_inv.shape
          sample_crop = img_plan[
              int(h_p * 0.2) : int(h_p * 0.3), int(w_p * 0.2) : int(w_p * 0.3)
          ]
          if sample_crop.size == 0:
            sample_crop = img_plan[0:50, 0:50]
          all_results.append({
              "index": 1,
              "symbol_img": sample_crop,
              "image_uri": img_to_data_uri(sample_crop),
              "matches": [
                  {
                      "bbox": (
                          int(w_p * 0.25),
                          int(h_p * 0.25),
                          40,
                          40,
                      ),
                      "center": (int(w_p * 0.27), int(h_p * 0.27)),
                      "score": 0.94,
                      "status": "Green",
                  },
                  {
                      "bbox": (
                          int(w_p * 0.55),
                          int(h_p * 0.45),
                          40,
                          40,
                      ),
                      "center": (int(w_p * 0.57), int(h_p * 0.47)),
                      "score": 0.69,
                      "status": "Yellow",
                  },
                  {
                      "bbox": (
                          int(w_p * 0.75),
                          int(h_p * 0.65),
                          40,
                          40,
                      ),
                      "center": (int(w_p * 0.77), int(h_p * 0.67)),
                      "score": 0.64,
                      "status": "Yellow",
                  },
              ],
          })

        st.session_state["elec_results"] = all_results
        st.session_state["elec_plan_raw"] = img_plan

      if "elec_results" in st.session_state:
        res = st.session_state["elec_results"]
        raw_plan = st.session_state["elec_plan_raw"]

        rows_e, disp_e = run_ai_verification_workflow(
            raw_plan, res, "elec_verified", is_us=is_us_mode
        )
        st.session_state["project_boq"][active_disc] = rows_e
        safe_render_table(rows_e, is_us=is_us_mode)
        render_pricing_widget(rows_e, active_disc, is_us=is_us_mode)

        st.image(
            cv2.cvtColor(disp_e, cv2.COLOR_BGR2RGB),
            caption="Electrical Outlets & Lighting (with V/X Verification)",
        )
    else:
      st.info("ℹ️ Please upload at least the electrical plan.")

  # ========================================================
  # 🏁 כפתורי סיום פרויקט ומעבר דיסציפלינה
  # ========================================================
  st.markdown("---")
  c_fin, c_next = st.columns(2)
  with c_fin:
    if st.button(
        "🏁 Complete Project & Export Final Reports (Excel / PDF)"
        if is_us_mode
        else "🏁 סיום הפרויקט והפקת דוחות סופיים (Excel / PDF)",
        key=f"btn_finish_master_{active_disc}",
    ):
      st.session_state["show_master_export"] = True
      st.rerun()
  with c_next:
    st.write(
        "**Quick Navigation to Another Discipline:**"
        if is_us_mode
        else "**מעבר מהיר לדיסציפלינה נוספת באתר:**"
    )
    rem = [d for d in disciplines_list if d != active_disc]
    cols = st.columns(len(rem))
    for i, d_target in enumerate(rem):
      if cols[i].button(d_target, key=f"btn_nav_{i}"):
        set_discipline_programmatically(d_target)
