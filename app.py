import json
import os
import time
import pandas as pd
import streamlit as st

# הגדרת תצורת העמוד
st.set_page_config(
    page_title="S.A. Quantities AI (S.A.Q)",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# קובץ זיכרון לומד למערכת ה-AI
MEMORY_FILE = "saq_ai_memory.json"


def load_memory():
  if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
      return json.load(f)
  return {"verified_symbols": {}, "rules": []}


def save_memory(data):
  with open(MEMORY_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


# עיצוב מותאם אישית (CSS) - פלטת צבעים הנדסית
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stSidebar {
        background-color: #1e293b;
    }
    h1, h2, h3 {
        color: #facc15 !important;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #facc15;
    }
    .stButton>button {
        background-color: #d97706;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #b45309;
    }
    .alert-box {
        padding: 10px;
        background-color: #7f1d1d;
        color: #fca5a5;
        border-radius: 5px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .success-box {
        padding: 10px;
        background-color: #064e3b;
        color: #6ee7b7;
        border-radius: 5px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# כותרת ראשית לפלטפורמה
st.title("🏗️ S.A. Quantities AI (S.A.Q)")
st.markdown(
    "**פלטפורמת ענן מבוססת AI לפענוח הנדסי אוטונומי של תוכניות בנייה וכתבי כמויות**"
)

# תפריט צד - בחירת מודל עבודה
st.sidebar.header("⚙️ בחר מצב עבודה (Mode)")
app_mode = st.sidebar.radio(
    "בחר קהל יעד ותהליך עבודה:",
    [
        "👷‍♂️ שדרוג ושינויי דיירים (מול סטנדרט קבלן)",
        "🔨 קבלני שיפוצים (מול מצב קיים / As-Is)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 בחר דיסציפלינה ניתוחית")
discipline = st.sidebar.selectbox(
    "בחר מערכת לפענוח:",
    [
        "⚡ חשמל ומאור",
        "🧱 בניה (מחיצות ומעטפת)",
        "🚿 אינסטלציה",
        "📐 ריצוף וחיפוי קירות",
    ],
)

# העלאת קבצים בהתאם למודל
st.sidebar.markdown("---")
st.sidebar.subheader("📂 העלאת תוכניות לעיבוד")

if "שינויי דיירים" in app_mode:
  file_standard = st.sidebar.file_uploader(
        "העלה שרטוט מכר (סטנדרט קבלן) - PDF/DWG", type=["pdf", "dwg", "png", "jpg"]
  )
  file_mod = st.sidebar.file_uploader(
        "העלה שרטוט שינויים מבוקש - PDF/DWG", type=["pdf", "dwg", "png", "jpg"]
  )
else:
  file_asis = st.sidebar.file_uploader(
        "העלה שרטוט מצב קיים (As-Is) - PDF/DWG",
        type=["pdf", "dwg", "png", "jpg"],
  )
  file_proposed = st.sidebar.file_uploader(
        "העלה שרטוט מוצע (ביצוע) - PDF/DWG", type=["pdf", "dwg", "png", "jpg"]
  )

# כפתור הפעלה ראשי עם אנימציית מנוף הנדסי
if st.sidebar.button("הפעל מנוע חישוב AI 🚀"):
  if ("שינויי דיירים" in app_mode and not (file_standard and file_mod)) or (
      "קבלני שיפוצים" in app_mode and not (file_asis and file_proposed)
  ):
    st.error(
        "⚠️ נא להעלות את שני קבצי התוכנית הנדרשים להשוואה לפני הפעלת המנוע."
    )
  else:
    with st.spinner(
        "🏗️ מנוף הנדסי פעיל: סורק שרטוטים, מחלץ סמלים ומבצע חישובי דלתא..."
    ):
      time.sleep(3)  דמיות זמן עיבוד סריקה
    st.success("✅ הניתוח ההנדסי הושלם בהצלחה!")

# אזור עבודה מרכזי לפי המודל הנבחר
if "שינויי דיירים" in app_mode:
  st.header("👷‍♂️ דוח שינויי דיירים (תוספות וזיכויים מול יזם)")

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("תוספות חשמל נטו", "+12 נקודות", "+2,400 ₪")
  col2.metric("זיכויים מביטולים", "-3 נקודות", "-600 ₪")
  col3.metric("הזזת נקודות (חריגה)", "2 נקודות", "מעבר לרדיוס 1.5מ'")
  col4.metric("סה\"כ לתשלום / זיכוי דייר", "+1,800 ₪", "דלתא סופית")

  st.markdown("---")
  st.subheader("🚨 בקרה והגנה על מעטפת הנדסית")
  st.markdown(
      '<div class="alert-box">⚠️ התראה הנדסית: בתוכנית השינויים זוהתה חריגה או ניסיון פגיעה בממ"ד / עמוד בטון מרכזי בציר 4. נדרש אישור מהנדס קונסטרוקציה.</div>',
      unsafe_allow_html=True,
  )

  st.subheader("📋 טבלת ריכוז שינויי דיירים ותמחור")
  data_tenant = {
      "אלמנט / פריט": [
          "נקודת מאור כפולה",
          "שקע כוח 16A",
          "הכנה למזגן מיני מרכזי",
          "הזזת נקודת אינסטלציה",
          "ביטול נקודת תקשורת",
      ],
      "סטנדרט קבלן": [
          "כלול (מפרט)",
          "כלול (מפרט)",
          "כלול (נקודה ראשית)",
          "במיקום מקורי",
          "קיים",
      ],
      "תוכנית שינויים": [
          "נוסף (2 יח')",
          "נוסף (4 יח')",
          "הסטה והגדלה",
          "הוזז ב-2.4 מטר",
          "בוטל",
      ],
      "סטטוס דלתא": [
          "תוספת תשלום",
          "תוספת תשלום",
          "חריגת מרחק",
          "חריגה מעבר לסטנדרט",
          "זיכוי לדייר",
      ],
      "עלות מוערכת (₪)": [500, 800, 1500, 450, -150],
  }
  df_tenant = pd.DataFrame(data_tenant)
  st.dataframe(df_tenant, use_container_width=True)

else:
  st.header("🔨 דוח קבלני שיפוצים (השוואת מוצע מול מצב קיים As-Is)")

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("הריסת מחיצות", "34 מ\"ר", "בלוק + טיח")
  col2.metric("בניית מחיצות חדשות", "28 מ\"ר", "גבס / בלוק איטונג")
  col3.metric("אורך חציבות לקווים", "45 מ\"ר", "חשמל ואינסטלציה")
  col4.metric("שטח ריצוף נטו", "85 מ\"ר", "אחרי קיזוז מחיצות")

  st.markdown("---")
  st.subheader("📋 כתב כמויות (Takeoff) מלא להצעת מחיר")
  data_reno = {
      "סעיף עבודה": [
          "הריסת קירות פנים (כולל פינוי)",
          "בניית קירות בלוקים 10 ס\"מ",
          "חציבות ותשתיות קו חשמל חדש",
          "התקנת נקודות אינסטלציה חדשות",
          "ריצוף מרצפות פורצלן 80X80",
          "חיפוי קירות חדר רטוב (גובה 2.2מ')",
      ],
      "יחידת מידה": ["מ\"ר", "מ\"ר", "מ\"א", "נקודה", "מ\"ר נטו", "מ\"ר קירות"],
      "כמות מחושבת": [34.0, 28.0, 45.0, 12.0, 85.0, 42.0],
      "הערות הנדסיות מה-AI": [
          "פינוי אתר מוגדר",
          "כולל אשפרה",
          "כולל השחלת חוטי 2.5 ממ\"ר",
          "כולל בדיקת לחץ מים",
          "כולל פנלים",
          "כולל איטום משחתי",
      ],
  }
  df_reno = pd.DataFrame(data_reno)
  st.dataframe(df_reno, use_container_width=True)

# אזור Human-in-the-Loop אימות סמלים V/X
st.markdown("---")
st.subheader("🔍 מנגנון אימות אנושי חכם (Human-in-the-Loop V/X)")
st.write(
    "המערכת זיהתה סמלים גבוליים שדורשים אימות מהיר לפני הפקת הדוח הסופי:"
)

c1, c2 = st.columns([2, 1])
with c1:
  st.info("❓ שאלה 1 מתוך 3: האם הסימון המסומן בעיגול האדום הוא שקע כוח או מאור?")
  st.image(
      "https://images.unsplash.com/photo-1581092160607-ee22621dd758?w=400&auto=format&fit=crop&q=60",
      width=300,
  )
with c2:
  st.write("בחר החלטה:")
  if st.button("✔️ מאשר כ-שקע כוח"):
  memory = load_memory()
  memory["verified_symbols"]["symbol_1"] = "power_socket"
  save_memory(memory)
  st.success("עודכן ונשמר בזיכרון הלומד (saq_ai_memory.json)!")
  if st.button("❌ דחה / סמן אחרת"):
  st.warning("הסממן סומן לבדיקה ידנית נוספת.")

# ייצוא דוחות סופי
st.markdown("---")
st.subheader("📥 ייצוא דוחות מערכת רשמיים")
col_exp1, col_exp2 = st.columns(2)
with col_exp1:
  if st.button("📊 הורד דוח מרוכז ל-Excel"):
  st.success("קובץ ה-Excel הופק בהצלחה עם לוגו S.A. Quantities AI!")
with col_exp2:
  if st.button("📄 הפק דוח HTML-PDF מעוצב"):
  st.success("דוח ה-PDF ההנדסי מוכן להפצה ללקוח/יזם!")
    
