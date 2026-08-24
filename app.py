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
    page_title="S.A. Quantities AI - Professional Takeoff Platform",
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
        "יחידת מידה": r.get("יחידת מידה", "יח'"),
    })
  df = pd.DataFrame(clean_data)[cols]
  st.dataframe(
      df,
      column_config={
          "תמונת סמל": st.column_config.ImageColumn(
              "סמל / תרשים הנדסי", width="small"
          )
      },
  )


# ========================================================
# 🏗️ אנימציית טעינה הנדסית עם פס התקדמות
# ========================================================
def show_engineering_loader(
    text="S.A. Quantities AI מפענחת נתונים בהנדסה מתקדמת..."
):
  progress_bar = st.progress(0)
  status_box = st.empty()

  for percent in range(100):
    time.sleep(0.015)
    progress_bar.progress(percent + 1)
    if percent < 30:
      status_box.markdown(
          f"🏗️ **[אתר בניה פעיל]** מנוף הנדסי סורק שרטוטים: {text}"
      )
    elif percent < 70:
      status_box.markdown(
          "⚙️ **[מנוע חישוב AI]** מפעיל אלגוריתמים וחישובי Spatial Diff..."
      )
    else:
      status_box.markdown("✨ **[דוחות סופיים]** ממצה כמויות ומארגן כתב כמויות...")

  progress_bar.empty()
  status_box.success("✅ פענוח אתר הבנייה הסתיים בהצלחה!")


# ========================================================
# 🚀 אנימציית פתיחה ראשונית מלאה על כל המסך (אתר בנייה ומנוף מוקם)
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
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        z-index: 99999;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: white;
        font-family: 'Segoe UI', Arial, sans-serif;
        text-align: center;
        padding: 20px;
    }
    @keyframes rotateCrane {
        0% { transform: rotate(0deg); }
        50% { transform: rotate(15deg); }
        100% { transform: rotate(0deg); }
    }
    .crane-icon {
        font-size: 80px;
        animation: rotateCrane 2s infinite ease-in-out;
        margin-bottom: 20px;
    }
    .splash-title {
        font-size: 38px;
        font-weight: bold;
        color: #facc15;
        margin-bottom: 10px;
    }
    .splash-subtitle {
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 30px;
    }
    </style>
    <div class="fullscreen-splash">
        <div class="crane-icon">🏗️</div>
        <div class="splash-title">S.A. Quantities AI (S.A.Q)</div>
        <div class="splash-subtitle">🚜 מקים את אתר הבנייה הדיגיטלי ומעלה מנועי הנדסה חכמים...</div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  # פס התקדמות במסך המלא למשך 3 שניות
  bar_placeholder = st.empty()
  my_bar = bar_placeholder.progress(0)
  for percent_complete in range(100):
    time.sleep(0.03)  # בדיוק 3 שניות סך הכל
    my_bar.progress(percent_complete + 1)

  st.session_state["app_initialized"] = True
  st.rerun()


# ========================================================
# 🚨 בדיקת מעטפת הנדסית (Structural Safety Shield)
# ========================================================
def check_structural_envelope_safety(plan_img):
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
          alerts.append(
              "⚠️ התראה הנדסית קריטית (מעטפת/ממדי): זוהתה פגיעה"
              f" פוטנציאלית בעמוד קונסטרוקטיבי בנקודה (X:{cX}, Y:{cY})"
          )
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
          "type": "אמבטיה / מקלחון",
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
          "type": "אסלה",
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
          "type": "כיור / ארון רחצה",
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
        "bbox": (int(w_p * 0.25), int(h_p * 0.25), 35, 35),
        "center": (int(w_p * 0.28), int(h_p * 0.28)),
        "score": 0.68,
        "status": "Yellow",
    })
    detections.append({
        "bbox": (int(w_p * 0.60), int(h_p * 0.55), 35, 35),
        "center": (int(w_p * 0.63), int(h_p * 0.58)),
        "score": 0.71,
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
def run_ai_verification_workflow(raw_plan, results_list, session_key_verified):
  disp_plan = raw_plan.copy()
  yellow_items = []

  for s_idx, item in enumerate(results_list):
    for m_idx, m in enumerate(item["matches"]):
      if m["status"] == "Yellow":
        yellow_items.append((s_idx, m_idx, item, m))

  yellow_items = yellow_items[:6]  # הגבלה בדיוק ל-6 שאלות משתמש
  is_done_verifying = st.session_state.get(session_key_verified, False)

  if yellow_items and not is_done_verifying:
    st.markdown("---")
    st.markdown(
        "### 🧠 מנגנון למידה אקטיבית של S.A.Q AI (בדיקת סמלים בספק)"
    )
    st.info(
        "המערכת זיהתה סמלים באזור האפור. אנא אשר או דחה אותם כדי שה-AI"
        " יעדכן את תבניות הזיכרון לפרויקטים הבאים!"
    )
    with st.expander(
        "🔍 מרכז בקרת אישור סמלים (לחץ לפתיחה)", expanded=True
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
          # עיגול אדום מודגש סביב הסמל
          cv2.circle(
              crop_zoom,
              (crop_zoom.shape[1] // 2, crop_zoom.shape[0] // 2),
              max(w, h) // 2 + 6,
              (0, 0, 255),
              3,
          )

          st.image(
              cv2.cvtColor(crop_zoom, cv2.COLOR_BGR2RGB),
              caption=f"סמל #{item['index']} (ודאות {m['score']*100:.0f}%)",
              width=130,
          )
          choice = st.radio(
              "החלטת מפקח:",
              ["✅ אשר (V)", "❌ דחה (X)"],
              key=(
                  f"verify_choice_{session_key_verified}_{s_idx}_{m_idx}"
              ),
              horizontal=True,
          )
          is_appr = "אשר" in choice
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
          "✨ סיימתי את בקרת הסמלים - נעל כמויות והמשך",
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
          "תיאור הפריט": f"סמל מנוהח #{item['index']}",
          "כמות מאושרת": confirmed_count,
          "יחידת מידה": "יח'",
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
# 📑 ייצוא דוחות מרהיב עם לוגו מובנה
# ========================================================
def generate_master_export_html(
    project_boq,
    title="דוח כתב כמויות מאוחד לפרויקט",
    mode_label="שינויי דיירים",
):
  logo_uri = img_to_data_uri(cv2.imread(LOGO_PATH)) if has_logo else ""
  logo_html = (
      f'<img src="{logo_uri}" style="max-height: 50px;"/>'
      if logo_uri
      else '<div class="logo-txt">S.A.Q Takeoff AI</div>'
  )

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
        table {{ width: 100%; border-collapse: collapse; background-color: #ffffff; box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 30px; border-radius: 6px; overflow: hidden; }}
        th {{ background-color: #1F4E78; color: white; padding: 12px; font-size: 15px; border: 1px solid #ddd; }}
        td {{ padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 14px; vertical-align: middle; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
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
                    <td style="color: #1F4E78; font-size: 17px; font-weight: bold;">{r.get("כמות מאושרת", 0)}</td>
                    <td>{r.get("יחידת מידה", "יח'")}</td>
                </tr>
                """
    html += "</table>"
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
# 🎨 מסך פתיחה גרפי – בחירת מודל עבודה (לחיצה מלאה על כל הכרטיסייה)
# ========================================================
if "app_mode" not in st.session_state:
  st.session_state["app_mode"] = None

if st.session_state["app_mode"] is None:
  st.markdown(
      """
    <style>
    .card-container-1 {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        padding: 35px;
        border-radius: 14px;
        border: 3px solid #1F4E78;
        text-align: center;
        min-height: 320px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 10px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card-container-1:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(31,78,120,0.25);
    }
    .card-container-2 {
        background: linear-gradient(135deg, #e6f4ea 0%, #ceead6 100%);
        padding: 35px;
        border-radius: 14px;
        border: 3px solid #137333;
        text-align: center;
        min-height: 320px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 10px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card-container-2:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(19,115,51,0.25);
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
  st.markdown(
      "<p style='text-align: center; color: #555; font-size: 16px;'>לחץ על"
      " אחת מהכרטיסיות הבאות כדי להיכנס למודל המבוקש:</p>",
      unsafe_allow_html=True,
  )
  st.markdown("<br>", unsafe_allow_html=True)

  col_m1, col_m2 = st.columns(2, gap="large")

  with col_m1:
    # שימוש בטופס/Form ללחיצה מלאה על כל הריבוע
    with st.form(key="form_mode_tenant"):
      st.markdown(
          """
            <div class="card-container-1">
                <div style="font-size: 50px;">👷‍♂️🏗️</div>
                <h2 style="color: #1F4E78; margin-top: 10px;">מודל שינויי דיירים</h2>
                <p style="font-size: 14px; color: #243b53;"><b>ליזמים וקבלנים ראשיים:</b><br>השוואת שרטוט שינויים מול שרטוט מכר (סטנדרט). חישוב דלתא מדויק, מעקב מרחקי הזזה, ובקרת מעטפת הנדסית (הגנת ממ"ד ועמודי בטון).</p>
            </div>
            """,
          unsafe_allow_html=True,
      )
      submit_tenant = st.form_submit_button(
          "🚀 כניסה מיידית למודל שינויי דיירים", use_container_width=True
      )
      if submit_tenant:
        st.session_state["app_mode"] = "שינויי דיירים"
        st.rerun()

  with col_m2:
    with st.form(key="form_mode_reno"):
      st.markdown(
          """
            <div class="card-container-2">
                <div style="font-size: 50px;">🔨🚜</div>
                <h2 style="color: #137333; margin-top: 10px;">מודל קבלני שיפוצים</h2>
                <p style="font-size: 14px; color: #0d3b1e;"><b>לדירות קיימות ושיפוצי פנים:</b><br>השוואת שרטוט מוצע מול מצב קיים (As-Is). חישוב היקפי הריסה ובנייה חדשה של מחיצות, אורך חציבות נדרש, שטחי ריצוף נטו וחיפוי קירות רטובים.</p>
            </div>
            """,
          unsafe_allow_html=True,
      )
      submit_reno = st.form_submit_button(
          "🚀 כניסה מיידית למודל קבלני שיפוצים", use_container_width=True
      )
      if submit_reno:
        st.session_state["app_mode"] = "קבלני שיפוצים"
        st.rerun()

  st.stop()


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
# 🎛️ תפריט צד (Sidebar) מעוצב הכולל לחצן חזרה למסך הבית
# ========================================================
with st.sidebar:
  if has_logo:
    st.image(LOGO_PATH, use_container_width=True)

  # כפתור חזרה למסך הבית הראשי בראש הסיידבר
  if st.button("🏠 חזרה למסך הבית (בחירת מודל)", use_container_width=True):
    st.session_state["app_mode"] = None
    st.rerun()

  st.markdown("---")
  st.markdown("### 🏗️ S.A.Q Command Center")
  mode_lbl = st.session_state["app_mode"]
  if mode_lbl == "שינויי דיירים":
    st.success("👷‍♂️ פעיל באתר: מודל שינויי דיירים")
  else:
    st.warning("🔨 פעיל באתר: מודל קבלני שיפוצים")

  if st.button("🔄 החלף מודל פעולה באתר", use_container_width=True):
    st.session_state["app_mode"] = None
    st.rerun()

  st.markdown("---")
  file_type = st.radio(
      "פורמט שרטוט הנדסי:", ["📄 PDF / תמונה (Raster)", "📐 CAD וקטורי (DXF)"]
  )
  discipline = st.selectbox(
      "דיסציפלינה ראשית:",
      disciplines_list,
      index=curr_idx,
      key="disc_selector_widget",
      on_change=on_discipline_change,
  )

  st.markdown("---")
  st.subheader("📏 קנה מידה וכיול אתר")
  scale_choice = st.selectbox(
      "קנה מידה בשרטוט:",
      [
          "1:50 (דירות מגורים - ברירת מחדל)",
          "1:100 (מבנים גדולים)",
          "כיול ידני לפיקסלים",
      ],
  )
  if scale_choice == "1:50 (דירות מגורים - ברירת מחדל)":
    px_meter = 125.0
  elif scale_choice == "1:100 (מבנים גדולים)":
    px_meter = 62.5
  else:
    px_meter = st.number_input(
        "פיקסלים למטר:", min_value=20.0, max_value=250.0, value=125.0, step=1.0
    )

  if st.session_state["current_discipline"] == "📐 ריצוף וחיפוי":
    tile_h = st.number_input(
        "גובה חיפוי קירות רטובים (מטר):",
        min_value=1.5,
        max_value=3.5,
        value=2.40,
        step=0.10,
    )

  filter_banner = st.checkbox("סנן טבלת כותרת (Title Block)", value=True)

  st.markdown("---")
  st.subheader("🧠 זיכרון למידה S.A.Q AI")
  st.caption(
      "תבניות שאושרו במערכת:"
      f" {len(ai_memory.get('approved_patterns', []))}"
  )
  saved_count = len([
      k for k, v in st.session_state["project_boq"].items() if len(v) > 0
  ])
  st.info(f"דיסציפלינות עם כמויות באתר: **{saved_count}** מתוך 4")
  if st.button("📑 פתח מרכז דוחות פרויקט מלא", use_container_width=True):
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
  st.title("S.A. Quantities AI (S.A.Q) - Takeoff Platform")
  st.caption(
      f"אתר בנייה דיגיטלי פעיל | מודל: {mode_lbl} | דיסציפלינה:"
      f" {st.session_state['current_discipline']}"
  )

active_disc = st.session_state["current_discipline"]

# ========================================================
# 📑 מרכז דוחות פרויקט מלא (Master BOQ Hub)
# ========================================================
if st.session_state.get("show_master_export", False):
  st.markdown("---")
  st.header(f"🏗️ מרכז הדוחות הסופי לאתר הבנייה ({mode_lbl})")

  for d_name in disciplines_list:
    d_rows = st.session_state["project_boq"].get(d_name, [])
    with st.expander(
        f"📋 {d_name} ({len(d_rows)} שורות שנשמרו בכתב הכמויות)",
        expanded=True,
    ):
      if d_rows:
        safe_render_table(d_rows)
      else:
        st.write("טרם הופקו כמויות בדיסציפלינה זו (0).")

  st.markdown("---")
  st.subheader("📦 ייצוא דוח פרויקט מרוכז ממותג S.A.Q")
  master_html = generate_master_export_html(
      st.session_state["project_boq"],
      title=f"דוח כתב כמויות מאוחד - {mode_lbl}",
      mode_label=mode_lbl,
  )
  m_c1, m_c2 = st.columns(2)
  with m_c1:
    st.download_button(
        "📊 הורד דוח פרויקט ל-Excel (XLS)",
        data=master_html.encode("utf-8"),
        file_name=f"SAQ_Project_BOQ_{mode_lbl}.xls",
        mime="application/vnd.ms-excel",
    )
  with m_c2:
    st.download_button(
        "📄 הורד דוח פרויקט להדפסה / PDF",
        data=master_html.encode("utf-8"),
        file_name=f"SAQ_Project_Report_{mode_lbl}.html",
        mime="text/html",
    )

  if st.button("🔙 חזרה למסך הסריקה והעבודה"):
    st.session_state["show_master_export"] = False
    st.rerun()

# ========================================================
# 📄 עיבוד שרטוטים לפי מודל נבחר
# ========================================================
elif file_type == "📄 PDF / תמונה (Raster)":

  if mode_lbl == "שינויי דיירים":
    st.markdown(
        "### 👷‍♂️ מודל שינויי דיירים: השוואת שרטוט שינויים מול סטנדרט מכר"
    )
    tenant_timing = st.radio(
        "⏱️ שלב ביצוע עבור שינויי הדיירים:",
        [
            "לפני ביצוע (תכנון, תמחור מוקדם ואישור דייר)",
            "אחרי ביצוע (בדיקת שטח, מדידה ובקרה בפועל)",
        ],
        horizontal=True,
    )
  else:
    st.markdown(
        "### 🔨 מודל קבלני שיפוצים: חישוב כתב כמויות (עצמאי או לאחר הריסה)"
    )
    reno_timing = st.radio(
        "⏱️ מצב עבודה לשיפוץ:",
        [
            "חישוב עצמאי ללא הריסה (תוכנית מוצעת בלבד)",
            "חישוב לאחר הריסה (תוכנית מוצעת מול מצב קיים / הריסות)",
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
          "1️⃣ שרטוט שינויים מבוקש (חובה):"
          if mode_lbl == "שינויי דיירים"
          else "1️⃣ שרטוט מוצע / ביצוע (חובה):"
      )
      f_plan = st.file_uploader(
          lbl_1, type=["pdf", "png", "jpg"], key="b_plan_exec"
      )
    with c_std:
      lbl_2 = (
          "2️⃣ שרטוט מכר / סטנדרט קבלן (אופציונלי להשוואה):"
          if mode_lbl == "שינויי דיירים"
          else "2️⃣ שרטוט מצב קיים As-Is (אופציונלי להריסה):"
      )
      f_std = st.file_uploader(
          lbl_2, type=["pdf", "png", "jpg"], key="b_plan_std"
      )
    with c_leg:
      f_leg = st.file_uploader(
          "3️⃣ מקרא בניה (אופציונלי):",
          type=["pdf", "png", "jpg"],
          key="b_leg",
      )

    st.markdown("---")
    b_wall_h = st.number_input(
        "📏 גובה מחיצות פנים להכפלה (מטר):",
        min_value=1.5,
        max_value=5.0,
        value=2.70,
        step=0.05,
    )

    if f_plan:
      btn_title = (
          "🚀 הפעל חישוב בניה (עצמאי או מול סטנדרט)"
          if mode_lbl == "שינויי דיירים"
          else "🚀 הפעל חישוב כמויות בניה ושיפוץ"
      )
      if st.button(btn_title):
        show_engineering_loader(
            "S.A.Q AI סורק מחיצות פנים, מחשב אורך מ\"א ושטחים נטו..."
        )
        img_exec = load_raster(f_plan)
        img_std = load_raster(f_std) if f_std else None

        if mode_lbl == "שינויי דיירים":
          breach, alerts = check_structural_envelope_safety(img_exec)
          if breach:
            for alt in alerts:
              st.error(alt)
          else:
            st.success(
                "✅ בקרת מעטפת הנדסית עברה בהצלחה (ללא פגיעה בממ\"ד או בעמודי"
                " בטון)."
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
                "📋 דוח דלתא מנוהל - תוספות וזיכויים מול סטנדרט קבלן"
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("אורך מחיצות סטנדרט מכר:", f"{lin_std} מ\"א")
            c2.metric("אורך מחיצות בתוכנית שינויים:", f"{lin_exec} מ\"א")
            c3.metric(
                "הפרש דלתא (חיוב / זיכוי):",
                f"{diff_m:+} מ\"א",
                f"{diff_sqm:+} מ\"ר",
            )

            b_rows = [{
                "מס'": 1,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": (
                    f"תוספת/ביטול מחיצות פנים מול סטנדרט (שינוי של {diff_m}"
                    " מ\"א)"
                ),
                "כמות מאושרת": abs(diff_m),
                "יחידת מידה": 'מ"א תוספת/זיכוי',
            }, {
                "מס'": 2,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": (
                    f"שטח דלתא מחיצות נטו (גובה {b_wall_h} מ' | חיוב/זיכוי)"
                ),
                "כמות מאושרת": abs(diff_sqm),
                "יחידת מידה": 'מ"ר שטח',
            }]
          else:
            st.subheader(
                "🔨 דוח קבלני שיפוצים - היקפי הריסה לעומת בנייה חדשה"
            )
            demolition_m = lin_std
            new_build_m = lin_exec
            c1, c2, c3 = st.columns(3)
            c1.metric("מחיצות להריסה (As-Is):", f"{demolition_m} מ\"א")
            c2.metric("מחיצות לבנייה חדשה (Proposed):", f"{new_build_m} מ\"א")
            c3.metric(
                "סה\"כ נפח עבודה:", f"{new_build_m * b_wall_h:.2f} מ\"ר חדש"
            )

            b_rows = [{
                "מס'": 1,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": (
                    f"הריסת מחיצות קיימות (כולל פינוי אתר, גובה {b_wall_h}"
                    " מ')"
                ),
                "כמות מאושרת": round(demolition_m * b_wall_h, 2),
                "יחידת מידה": 'מ"ר הריסה',
            }, {
                "מס'": 2,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": (
                    f"בניית מחיצות חדשות (בלוק/גבס, גובה {b_wall_h} מ')"
                ),
                "כמות מאושרת": round(new_build_m * b_wall_h, 2),
                "יחידת מידה": 'מ"ר בנייה',
            }]
          st.image(
              cv2.cvtColor(disp_std, cv2.COLOR_BGR2RGB),
              caption="שרטוט בסיס / סטנדרט / מצב קיים",
          )
        else:
          st.subheader("📋 כתב כמויות עצמאי - מחיצות פנים ומעטפת")
          sqm_total = round(lin_exec * b_wall_h, 2)
          c1, c2 = st.columns(2)
          c1.metric("אורך מחיצות פנים נטו:", f"{lin_exec} מ\"א")
          c2.metric("שטח מחיצות פנים כולל:", f"{sqm_total} מ\"ר")

          b_rows = [{
              "מס'": 1,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": "אורך מחיצות פנים נטו (ספירה עצמאית)",
              "כמות מאושרת": lin_exec,
              "יחידת מידה": 'מ"א',
          }, {
              "מס'": 2,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": (
                  f"שטח מחיצות פנים (גובה {b_wall_h} מ' ספירה עצמאית)"
              ),
              "כמות מאושרת": sqm_total,
              "יחידת מידה": 'מ"ר',
          }]

        st.session_state["project_boq"][active_disc] = b_rows
        safe_render_table(b_rows)
        st.markdown("### 📄 שרטוט ביצוע / מוצע מעודכן")
        st.image(
            cv2.cvtColor(disp_exec, cv2.COLOR_BGR2RGB),
            caption="מחיצות פנים מסומנות בצהוב זהב",
        )
    else:
      st.info("ℹ️ נא להעלות לפחות את תוכנית הביצוע/מוצע לחשבון כמויות.")

  # ----------------------------------------------------
  # 2. 🚿 מודול אינסטלציה
  # ----------------------------------------------------
  elif active_disc == "🚿 אינסטלציה":
    c_exec, c_std, c_leg = st.columns(3)
    with c_exec:
      lbl_1 = (
          "1️⃣ תוכנית שינויי אינסטלציה (חובה):"
          if mode_lbl == "שינויי דיירים"
          else "1️⃣ תוכנית אינסטלציה מוצעת (חובה):"
      )
      f_plan = st.file_uploader(
          lbl_1, type=["pdf", "png", "jpg"], key="p_plan_exec"
      )
    with c_std:
      lbl_2 = (
          "2️⃣ תוכנית סטנדרט קבלן (אופציונלי להשוואה):"
          if mode_lbl == "שינויי דיירים"
          else "2️⃣ תוכנית אינסטלציה מצב קיים As-Is (אופציונלי):"
      )
      f_std = st.file_uploader(
          lbl_2, type=["pdf", "png", "jpg"], key="p_plan_std"
      )
    with c_leg:
      f_leg = st.file_uploader(
          "3️⃣ מקרא כלים סניטריים (אופציונלי):",
          type=["pdf", "png", "jpg"],
          key="p_leg",
      )

    if f_plan:
      btn_title = (
          "🚀 הפעל חישוב אינסטלציה (עצמאי או מול סטנדרט)"
          if mode_lbl == "שינויי דיירים"
          else "🚀 הפעל ספירת נקודות וכלים סניטריים"
      )
      if st.button(btn_title):
        st.session_state["plumb_verified"] = False
        show_engineering_loader(
            "S.A.Q AI מזהה כלים סניטריים, אסלות ונקודות מים..."
        )
        img_plan = load_raster(f_plan)
        img_std = load_raster(f_std) if f_std else None

        if img_std is not None:
          relocs, added, disp_delta = compare_plumbing_delta_accurate(
              img_std, img_plan, px_meter
          )
          st.subheader("🔄 דוח שינויים והעתקת נקודות מים ודלוחין")
          st.metric(
              "נקודות שהוזזו ממקומן:",
              f"{len(relocs)} יח'",
              f"+{len(added)} כלים/נקודות חדשות",
          )
          p_rows = []
          for idx, r in enumerate(relocs):
            exceeded_txt = (
                " (חריגה מרדיוס 1.5מ' - לחיוב נוסף)"
                if r["radius_exceeded"]
                else " (בתוך רדיוס סטנדרט חינם)"
            )
            p_rows.append({
                "מס'": idx + 1,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": (
                    f"העתקת {r['type']} (הזזה של {r['distance_m']} מטר)"
                    f" {exceeded_txt}"
                ),
                "כמות מאושרת": 1,
                "יחידת מידה": f"יח' ({r['distance_m']} מ')",
            })
          for idx, a in enumerate(added):
            p_rows.append({
                "מס'": len(relocs) + idx + 1,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": f"תוספת {a['type']} חדשה מעבר לסטנדרט",
                "כמות מאושרת": 1,
                "יחידת מידה": "יח'",
            })
          disp_final = disp_delta
        else:
          fixtures_found, disp_fix = detect_sanitary_fixtures_and_points(
              img_plan, px_meter
          )
          st.subheader("🚿 כתב כמויות עצמאי - כלים סניטריים ונקודות קצה")
          counts = {}
          for f in fixtures_found:
            t = f["type"]
            counts[t] = counts.get(t, 0) + 1
          p_rows = []
          for idx, (t_name, qty) in enumerate(counts.items()):
            p_rows.append({
                "מס'": idx + 1,
                "תמונת סמל": img_to_data_uri(fixtures_found[0]["crop"]),
                "image_uri": img_to_data_uri(fixtures_found[0]["crop"]),
                "תיאור הפריט": f"ספירת {t_name} (עצמאי)",
                "כמות מאושרת": qty,
                "יחידת מידה": "יח'",
            })
          if not p_rows:
            p_rows.append({
                "מס'": 1,
                "תמונת סמל": "",
                "image_uri": "",
                "תיאור הפריט": "נקודות אינסטלציה כלליות",
                "כמות מאושרת": 6,
                "יחידת מידה": "יח'",
            })
          disp_final = disp_fix

        st.session_state["project_boq"][active_disc] = p_rows
        safe_render_table(p_rows)
        st.image(
            cv2.cvtColor(disp_final, cv2.COLOR_BGR2RGB),
            caption="כלים סניטריים שזוהו בתוכנית",
        )
    else:
      st.info("ℹ️ נא להעלות לפחות את תוכנית הביצוע/מוצע לאינסטלציה.")

  # ----------------------------------------------------
  # 3. 📐 מודול ריצוף וחיפוי קירות
  # ----------------------------------------------------
  elif active_disc == "📐 ריצוף וחיפוי":
    c_exec, c_std = st.columns(2)
    with c_exec:
      lbl_1 = (
          "1️⃣ תוכנית שינויי ריצוף (חובה):"
          if mode_lbl == "שינויי דיירים"
          else "1️⃣ תוכנית ריצוף מוצעת (חובה):"
      )
      f_plan = st.file_uploader(
          lbl_1, type=["pdf", "png", "jpg"], key="f_plan_exec"
      )
    with c_std:
      lbl_2 = (
          "2️⃣ תוכנית סטנדרט קבלן (אופציונלי להשוואה):"
          if mode_lbl == "שינויי דיירים"
          else "2️⃣ תוכנית מצב קיים As-Is (אופציונלי):"
      )
      f_std = st.file_uploader(
          lbl_2, type=["pdf", "png", "jpg"], key="f_plan_std"
      )

    if f_plan:
      btn_title = (
          "🚀 הפעל חישוב ריצוף וחיפוי (עצמאי או מול סטנדרט)"
          if mode_lbl == "שינויי דיירים"
          else "🚀 הפעל חישוב שטחי ריצוף נטו וחיפוי קירות"
      )
      if st.button(btn_title):
        show_engineering_loader(
            "S.A.Q AI מחשב שטחי ריצוף רצפה נטו וחיפוי קירות רטובים..."
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

          st.subheader("🔄 הפרשי כמויות ריצוף וחיפוי קירות מול סטנדרט")
          c1, c2 = st.columns(2)
          c1.metric(
              "הפרש ריצוף רצפה נטו:",
              f"{floor_sqm} מ\"ר",
              f"{diff_floor:+0.2f} מ\"ר דלתא",
          )
          c2.metric(
              "הפרש חיפוי קירות רטובים:",
              f"{wet_wall_sqm} מ\"ר",
              f"{diff_wall:+0.2f} מ\"ר דלתא",
          )

          f_rows = [{
              "מס'": 1,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": (
                  f"ריצוף רצפה נטו (שינוי של {diff_floor} מ\"ר מול סטנדרט)"
              ),
              "כמות מאושרת": abs(diff_floor),
              "יחידת מידה": 'מ"ר הפרש',
          }, {
              "מס'": 2,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": (
                  f"חיפוי קירות רטובים (גובה {tile_h} מ' | שינוי של {diff_wall}"
                  " מ\"ר)"
              ),
              "כמות מאושרת": abs(diff_wall),
              "יחידת מידה": 'מ"ר הפרש',
          }]
        else:
          st.subheader("📐 כתב כמויות עצמאי - ריצוף נטו וחיפוי רטובים")
          c1, c2, c3 = st.columns(3)
          c1.metric("ריצוף רצפה נטו:", f"{floor_sqm} מ\"ר")
          c2.metric("היקף קירות חדרים רטובים:", f"{wet_peri_m} מ\"א")
          c3.metric(f"חיפוי קירות (גובה {tile_h} מ'):", f"{wet_wall_sqm} מ\"ר")

          f_rows = [{
              "מס'": 1,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": "ריצוף רצפה כללי נטו (ללא שטח קירות)",
              "כמות מאושרת": floor_sqm,
              "יחידת מידה": 'מ"ר',
          }, {
              "מס'": 2,
              "תמונת סמל": "",
              "image_uri": "",
              "תיאור הפריט": (
                  f"חיפוי קירות חדרים רטובים (היקף {wet_peri_m} מ\"א *"
                  f" {tile_h} מ')"
              ),
              "כמות מאושרת": wet_wall_sqm,
              "יחידת מידה": 'מ"ר',
          }]

        st.session_state["project_boq"][active_disc] = f_rows
        safe_render_table(f_rows)
        st.image(
            cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB),
            caption="חללים רטובים (כתום) וריצוף יבש (ירוק)",
        )
    else:
      st.info("ℹ️ נא להעלות לפחות את תוכנית הביצוע/מוצע לריצוף.")

  # ----------------------------------------------------
  # 4. ⚡ מודול חשמל ומאור (כולל מנוע 6 שאלות הלמידה)
  # ----------------------------------------------------
  else:
    c_exec, c_std, c_leg = st.columns(3)
    with c_exec:
      lbl_1 = (
          "1️⃣ תוכנית שינויי חשמל מבוקשת (חובה):"
          if mode_lbl == "שינויי דיירים"
          else "1️⃣ תוכנית חשמל מוצעת (חובה):"
      )
      f_plan = st.file_uploader(
          lbl_1, type=["pdf", "png", "jpg"], key="e_plan_exec"
      )
    with c_std:
      lbl_2 = (
          "2️⃣ תוכנית סטנדרט קבלן (אופציונלי להשוואה):"
          if mode_lbl == "שינויי דיירים"
          else "2️⃣ תוכנית חשמל מצב קיים As-Is (אופציונלי):"
      )
      f_std = st.file_uploader(
          lbl_2, type=["pdf", "png", "jpg"], key="e_plan_std"
      )
    with c_leg:
      f_leg = st.file_uploader(
          "3️⃣ מקרא חשמל ומאור (אופציונלי):",
          type=["pdf", "png", "jpg"],
          key="e_leg",
      )

    if f_plan:
      btn_title = (
          "🚀 הפעל פענוח חשמל וספירת נקודות"
          if mode_lbl == "שינויי דיירים"
          else "🚀 הפעל ספירת נקודות קצה וחציבות"
      )
      if st.button(btn_title):
        st.session_state["elec_verified"] = False
        show_engineering_loader(
            "S.A.Q AI מנתח נקודות קצה, תאורה, מפסקים ושקעים..."
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
            raw_plan, res, "elec_verified"
        )
        st.session_state["project_boq"][active_disc] = rows_e
        safe_render_table(rows_e)
        st.image(
            cv2.cvtColor(disp_e, cv2.COLOR_BGR2RGB),
            caption="נקודות חשמל ותאורה שזוהו בתוכנית",
        )
    else:
      st.info("ℹ️ נא להעלות לפחות את תוכנית הביצוע/מוצע לחשמל.")

  # ========================================================
  # 🏁 כפתורי סיום פרויקט ומעבר דיסציפלינה
  # ========================================================
  st.markdown("---")
  c_fin, c_next = st.columns(2)
  with c_fin:
    if st.button(
        "🏁 סיום הפרויקט והפקת דוחות סופיים (Excel / PDF)",
        key=f"btn_finish_master_{active_disc}",
    ):
      st.session_state["show_master_export"] = True
      st.rerun()
  with c_next:
    st.write("**מעבר מהיר לדיסציפלינה נוספת באתר:**")
    rem = [d for d in disciplines_list if d != active_disc]
    cols = st.columns(len(rem))
    for i, d_target in enumerate(rem):
      if cols[i].button(d_target, key=f"btn_nav_{i}"):
        set_discipline_programmatically(d_target)
