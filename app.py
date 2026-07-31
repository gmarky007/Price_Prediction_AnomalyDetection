# -*- coding: utf-8 -*-
"""
HỆ THỐNG PHÂN TÍCH, ĐỊNH GIÁ THỊ TRƯỜNG XE MÁY CỦ & PHÁT HIỆN GIÁ BẤT THƯỜNG
Dự án Cá nhân: Enterprise Motorcycle Valuation & Anomaly Detection System
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Cross-version pickle compatibility shims cho Streamlit Cloud (Linux)
import sys
import sklearn.pipeline
import sklearn.compose
import sklearn.impute
import sklearn.preprocessing

sys.modules['pipeline'] = sklearn.pipeline
sys.modules['sklearn.pipeline.Pipeline'] = sklearn.pipeline.Pipeline

# -------------------------------------------------------------
# 1. THIẾT LẬP CẤU HÌNH TRANG STREAMLIT & ENTERPRISE DESIGN SYSTEM
# -------------------------------------------------------------
st.set_page_config(
    page_title="Hệ Thống Phân Tích & Định Giá Xe Máy Cũ",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injected CSS for Enterprise Dark/Yellow Palette
st.markdown("""
<style>
    /* Page Width Limit 82% */
    .main .block-container {
        max-width: 82% !important;
        margin: 0 auto !important;
        padding-top: 1.5rem !important;
    }
    
    /* Primary Banner */
    .enterprise-header {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #ffba00;
        padding: 20px 24px;
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .enterprise-header h1 {
        color: #ffba00 !important;
        margin: 0;
        font-size: 2.0rem;
        font-weight: 800;
    }
    .enterprise-header p {
        color: #cbd5e1;
        margin-top: 4px;
        margin-bottom: 0;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 186, 0, 0.25);
        padding: 16px;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Risk Badges */
    .badge-normal {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .badge-warning {
        background-color: rgba(234, 179, 8, 0.15);
        color: #eab308;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. TẢI DỮ LIỆU & MÔ HÌNH ML CHÍNH THỨC TỪ PROJECT 2
# -------------------------------------------------------------
def parse_price_value(val):
    if pd.isna(val): return np.nan
    s = str(val).lower().replace('đ', '').replace('vnd', '').strip()
    s = s.replace('.', '').replace(',', '')
    m = re.search(r'(\d+)', s)
    if m:
        v = float(m.group(1))
        if v > 1000:
            return round(v / 1000000.0, 1) # Triệu VND
        return round(v, 1)
    return np.nan

@st.cache_resource
def load_ml_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    global_path = os.path.join(data_dir, 'global_pipeline.pkl')
    submodels_path = os.path.join(data_dir, 'submodels_trained.pkl')
    data_path = os.path.join(data_dir, 'data_motobikes.xlsx')
    
    global_pipe = joblib.load(global_path) if os.path.exists(global_path) else None
    submodels_dict = joblib.load(submodels_path) if os.path.exists(submodels_path) else {}
    
    if os.path.exists(data_path):
        df_bikes = pd.read_excel(data_path)
        if 'Giá' in df_bikes.columns:
            df_bikes['Giá'] = df_bikes['Giá'].apply(parse_price_value)
        if 'Năm đăng ký' in df_bikes.columns:
            df_bikes['Năm đăng ký'] = pd.to_numeric(df_bikes['Năm đăng ký'], errors='coerce')
    else:
        df_bikes = pd.DataFrame()
        
    return global_pipe, submodels_dict, df_bikes

global_pipe, submodels_dict, df_bikes = load_ml_resources()

# -------------------------------------------------------------
# 3. DYNAMIC DUAL-TIER CATALOG LOADER (37 BRANDS & 224 MODELS)
# -------------------------------------------------------------
@st.cache_data
def build_full_brand_model_catalog(df):
    brand_map = {}
    model_type_map = {}
    model_brand_map = {}
    model_cc_map = {}  # Dung tích phổ biến nhất cho mỗi dòng xe
    
    # Hàm chuyển dải dung tích text → giá trị cc đại diện
    def _cc_range_to_value(cc_text):
        s = str(cc_text).lower().strip()
        if 'dưới 50' in s or '< 50' in s:
            return 50
        elif '50 - 100' in s or '50-100' in s:
            return 110
        elif '100 - 175' in s or '100-175' in s:
            return 125
        elif 'trên 175' in s or '> 175' in s:
            return 300
        return 125  # fallback
    
    if not df.empty and 'Thương hiệu' in df.columns and 'Dòng xe' in df.columns:
        for b_name, grp in df.groupby('Thương hiệu'):
            b_clean = str(b_name).strip()
            models_list = sorted([str(m).strip() for m in grp['Dòng xe'].dropna().unique() if str(m).strip()])
            brand_map[b_clean] = models_list
            
            for m_clean in models_list:
                model_brand_map[m_clean] = b_clean
                
                # Guess bike type based on title/model text
                m_lower = m_clean.lower()
                b_lower = b_clean.lower()
                
                if any(k in m_lower for k in ['exciter', 'winner', 'raider', 'satria', 'cbr', 'cb', 'r15', 'r3', 'ninja', 'z1000', 'z900', 'z300', 'monster', 'duke', 'gsx', 'tfx', 'k-pipe', 'kpipe']):
                    guessed_type = 'Tay côn/Moto'
                elif any(k in m_lower for k in ['wave', 'sirius', 'jupiter', 'future', 'blade', 'dream', 'cub', 'smash', 'revo', 'axelo', 'elegant', 'galaxy']):
                    guessed_type = 'Xe số'
                elif any(k in m_lower for k in ['sh', 'air blade', 'ab', 'vision', 'lead', 'vario', 'click', 'janus', 'grande', 'nvx', 'vespa', 'liberty', 'zip', 'attila', 'pcx', 'lati', 'cuxi', 'luvias']):
                    guessed_type = 'Tay ga'
                elif any(k in b_lower for k in ['ducati', 'kawasaki', 'bmw', 'harley', 'ktm', 'triumph', 'brixton', 'gpx', 'benelli']):
                    guessed_type = 'Tay côn/Moto'
                elif 'Loại xe' in grp.columns:
                    mode_type = grp[grp['Dòng xe'] == m_clean]['Loại xe'].mode()
                    guessed_type = str(mode_type[0]).strip() if not mode_type.empty else 'Tay ga'
                else:
                    guessed_type = 'Tay ga'
                    
                model_type_map[m_clean] = guessed_type
                
                # Tính dung tích phổ biến nhất cho dòng xe từ CSDL
                if 'Dung tích xe' in grp.columns:
                    m_cc_series = grp[grp['Dòng xe'] == m_clean]['Dung tích xe'].dropna()
                    if not m_cc_series.empty:
                        mode_cc = m_cc_series.mode()
                        model_cc_map[m_clean] = _cc_range_to_value(mode_cc.iloc[0]) if not mode_cc.empty else 125
                    else:
                        model_cc_map[m_clean] = 125
                else:
                    model_cc_map[m_clean] = 125
                
    return brand_map, model_type_map, model_brand_map, model_cc_map

FULL_BRAND_MAP, MODEL_TYPE_MAP, MODEL_BRAND_MAP, MODEL_CC_MAP = build_full_brand_model_catalog(df_bikes)

# Hãng xe xếp theo số lượng bài đăng giảm dần, hãng < 20 bài gộp "Khác (PKL/Hiếm)"
_BRAND_FREQ_MIN = 20  # Ngưỡng tối thiểu để hiển thị riêng
_MINOR_BRANDS = set()  # Tập hãng hiếm gộp vào Khác
if not df_bikes.empty and 'Thương hiệu' in df_bikes.columns:
    _brand_counts = df_bikes['Thương hiệu'].value_counts()
    _major = [b for b, cnt in _brand_counts.items() if cnt >= _BRAND_FREQ_MIN and b != 'Hãng khác']
    _minor = [b for b, cnt in _brand_counts.items() if cnt < _BRAND_FREQ_MIN or b == 'Hãng khác']
    _MINOR_BRANDS = set(_minor)
    ALL_BRANDS_LIST = _major + (['Khác (PKL/Hiếm)'] if _minor else [])
else:
    ALL_BRANDS_LIST = ["Honda", "Yamaha", "Piaggio", "Suzuki", "SYM", "Kawasaki", "Kymco", "Khác (PKL/Hiếm)"]

# Map "Khác" → tất cả dòng xe thuộc các hãng hiếm
_MINOR_MODELS = []
for mb in _MINOR_BRANDS:
    _MINOR_MODELS.extend(FULL_BRAND_MAP.get(mb, []))
FULL_BRAND_MAP['Khác (PKL/Hiếm)'] = sorted(set(_MINOR_MODELS))

ALL_GLOBAL_MODELS = ["-- Tra cứu trực tiếp Dòng xe --"]
GLOBAL_MODEL_MAP = {}

# Lập danh sách các dòng xe cụ thể (Loại bỏ các mục 'Dòng khác' bị lặp lại của các hãng)
_clean_model_tuples = []
for b_name, m_list in FULL_BRAND_MAP.items():
    if b_name == 'Khác (PKL/Hiếm)':
        continue
    for m_name in m_list:
        if m_name == 'Dòng khác':
            continue
        lbl = f"{m_name} ({b_name})"
        _clean_model_tuples.append((lbl, b_name, m_name))

# Sắp xếp danh sách dòng xe theo thứ tự A - Z tên xe
_clean_model_tuples.sort(key=lambda x: x[0].lower())

for lbl, b_name, m_name in _clean_model_tuples:
    if lbl not in GLOBAL_MODEL_MAP:
        ALL_GLOBAL_MODELS.append(lbl)
        GLOBAL_MODEL_MAP[lbl] = (b_name, m_name)

YEAR_OPTIONS = ['-- Chọn Năm --'] + list(range(2025, 1979, -1))

# Danh mục dung tích thực tế chuẩn xác từng dòng xe Việt Nam
MODEL_SPECIFIC_CC = {
    'Air Blade': [110, 125, 150, 160],
    'Winner X': [150],
    'Winner': [150],
    'Exciter': [135, 150, 155],
    'Jupiter': [110, 125],
    'Sirius': [50, 110],
    'Wave': [50, 110, 125],
    'Future': [110, 125],
    'Dream': [100, 110],
    'Cub': [50, 110, 125],
    'Blade': [110],
    'Raider': [150],
    'Satria': [150],
    'Sonic': [150],
    'K-Pipe': [50, 125],
    'TFX': [150],
    'CBR': [150, 250, 650, 1000],
    'CB': [150, 300, 500, 650],
    'R15': [155],
    'R3': [300],
    'Ninja': [300, 400, 650],
    'Z1000': [1000],
    'Z900': [900],
    'Z300': [300],
    'Monster': [800, 900],
    'Duke': [200, 390],
    'GSX': [150],
    'Vision': [110],
    'Lead': [125],
    'Janus': [125],
    'Grande': [125],
    'Vario': [125, 150, 160],
    'Click': [125, 150],
    'SH Mode': [125],
    'SH': [125, 150, 160, 300, 350],
    'Vespa': [125, 150],
    'Liberty': [125, 150],
    'Sprint': [125, 150],
    'Primavera': [125, 150],
    'LX': [125, 150],
    'Medley': [125, 150],
    'GTS': [125, 150, 300],
    'Fly': [125],
    'Beverly': [125, 300],
    'Zip': [100],
    'Attila': [110, 125],
    'PCX': [125, 150, 160],
    'NVX': [125, 155],
}

# Bản đồ chọn mẫu xe đại diện tiêu biểu cho từng Hãng khi người dùng chưa chọn Dòng xe
DEFAULT_BRAND_MODEL = {
    'Honda': 'Wave',
    'Yamaha': 'Sirius',
    'Piaggio': 'Vespa',
    'Suzuki': 'Raider',
    'SYM': 'Attila',
    'Kawasaki': 'Z900',
    'Kymco': 'Like',
    'Khác (PKL/Hiếm)': 'Dòng khác'
}

def get_model_cc_options(model_name, bike_type):
    m_str = str(model_name).strip()
    if m_str and m_str not in ["-- Chọn Dòng xe --", "Tất cả", "Dòng xe"]:
        # 1. Tra cứu từ điển ghi đè thủ công các dòng phổ biến
        for key in sorted(MODEL_SPECIFIC_CC.keys(), key=len, reverse=True):
            if key.lower() in m_str.lower():
                return MODEL_SPECIFIC_CC[key]
        # 2. Tra cứu dung tích mode chuẩn từ CSDL gốc cho mọi dòng xe còn lại
        if m_str in MODEL_CC_MAP:
            return [MODEL_CC_MAP[m_str]]
            
    if bike_type == 'Tay ga':
        return [50, 110, 125, 135, 150, 155, 160, 300, 350]
    elif bike_type == 'Xe số':
        return [50, 110, 125, 135]
    elif bike_type == 'Tay côn/Moto':
        return [50, 125, 135, 150, 155, 160, 175, 250, 300, 400, 600, 900, 1000]
    return [50, 110, 125, 135, 150, 155, 160, 175, 250, 300, 400, 600, 1000]

# Từ điển chuẩn hóa CC định danh 1-1 cho các Dòng xe
CORRECT_MODEL_CC = {
    'Vision': 110, 'Exciter': 150, 'Winner': 150, 'Winner X': 150, 'Sirius': 110, 'Jupiter': 110,
    'Z900': 900, 'Z1000': 1000, 'Raider': 150, 'Satria': 150, 'GSX': 150, 'CBR': 150,
    'Monster': 800, '1290 Super Duke R': 390, 'Duke 200': 200, 'Duke 250': 250, 'Duke 390': 390,
    'Blade': 110, 'Pepe': 50, 'Espero': 50, 'Zip': 100, 'XMAX': 300, 'Max': 110, 'CD': 125, '67': 50,
    '1199 panigale': 1000, '125/250': 250, '2015 RSV4 R APRC ABS': 1000, '390': 390, '48': 1200,
    'Beverly': 125, 'Fly': 125, 'GTS': 125, 'LX': 125, 'Medley': 125, 'Primavera': 125, 'Sprint': 125
}
for _m_key, _cc_val in CORRECT_MODEL_CC.items():
    MODEL_CC_MAP[_m_key] = _cc_val

def resolve_strict_vehicle(selected_brand, selected_type, selected_model, selected_cc):
    # 1. Nếu người dùng chọn Dòng xe cụ thể -> Khóa chuẩn 100% từ dictionary Dòng xe
    if selected_model and selected_model != "Tất cả":
        brand = MODEL_BRAND_MAP.get(selected_model, selected_brand if selected_brand != "Tất cả" else "Honda")
        bike_type = MODEL_TYPE_MAP.get(selected_model, selected_type if selected_type != "Tất cả" else "Tay ga")
        model = selected_model
    # 2. Nếu người dùng chọn Hãng xe cụ thể -> Giữ NGUYÊN HÃNG XE 100%, chọn Dòng xe đại diện tiêu biểu
    elif selected_brand and selected_brand != "Tất cả":
        brand = selected_brand
        rep_model = DEFAULT_BRAND_MODEL.get(brand, "Vespa" if brand == "Piaggio" else "Wave")
        avail_models = FULL_BRAND_MAP.get(brand, [rep_model])
        
        if selected_type and selected_type != "Tất cả":
            type_filtered = [m for m in avail_models if MODEL_TYPE_MAP.get(m) == selected_type]
            model = type_filtered[0] if type_filtered else (rep_model if rep_model in avail_models else avail_models[0])
            bike_type = MODEL_TYPE_MAP.get(model, selected_type)
        else:
            model = rep_model if rep_model in avail_models else avail_models[0]
            bike_type = MODEL_TYPE_MAP.get(model, 'Tay ga' if brand == 'Piaggio' else 'Xe số')
    # 3. Nếu chọn Loại xe cụ thể nhưng chưa chọn Hãng
    elif selected_type and selected_type != "Tất cả":
        brand = "Honda"
        bike_type = selected_type
        avail_models = [m for m in FULL_BRAND_MAP["Honda"] if MODEL_TYPE_MAP.get(m) == bike_type]
        model = avail_models[0] if avail_models else "Wave"
    # 4. Mặc định
    else:
        brand = "Honda"
        bike_type = "Xe số"
        model = "Wave"

    cc_opts = get_model_cc_options(model, bike_type)
    if selected_cc and selected_cc != "Tất cả":
        try:
            user_cc = int(str(selected_cc).replace(" cc", ""))
            final_cc = user_cc if user_cc in cc_opts else cc_opts[0]
        except:
            final_cc = cc_opts[0]
    else:
        final_cc = cc_opts[0]

    return brand, model, bike_type, final_cc

@st.cache_resource
def load_ml_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    global_path = os.path.join(data_dir, 'global_pipeline.pkl')
    submodels_path = os.path.join(data_dir, 'submodels_trained.pkl')
    data_path = os.path.join(data_dir, 'data_motobikes.xlsx')
    
    global_pipe = joblib.load(global_path) if os.path.exists(global_path) else None
    submodels_dict = joblib.load(submodels_path) if os.path.exists(submodels_path) else {}
    
    # Vá lỗi tương thích phiên bản scikit-learn (SimpleImputer._fill_dtype -> _fit_dtype)
    from sklearn.impute import SimpleImputer
    def fix_imputer(obj):
        if isinstance(obj, SimpleImputer):
            if not hasattr(obj, '_fill_dtype') and hasattr(obj, '_fit_dtype'):
                obj._fill_dtype = obj._fit_dtype
        if hasattr(obj, 'named_steps'):
            for step in obj.named_steps.values():
                fix_imputer(step)
        if hasattr(obj, 'transformers_'):
            for name, trans, cols in obj.transformers_:
                fix_imputer(trans)

    if submodels_dict:
        for info in submodels_dict.values():
            if 'pipeline' in info:
                fix_imputer(info['pipeline'])
    if global_pipe:
        fix_imputer(global_pipe)

    model_medians_ci = {}
    brand_type_medians_ci = {}
    type_medians_ci = {}

    if os.path.exists(data_path):
        df_bikes = pd.read_excel(data_path)
        if 'Giá' in df_bikes.columns:
            df_bikes['Giá'] = df_bikes['Giá'].apply(parse_price_value)
        if 'Năm đăng ký' in df_bikes.columns:
            df_bikes['Năm đăng ký'] = pd.to_numeric(df_bikes['Năm đăng ký'], errors='coerce')

        df_clean = df_bikes[df_bikes['Giá'] > 1.0].copy() if 'Giá' in df_bikes.columns else pd.DataFrame()
        if not df_clean.empty:
            for (b, m), grp in df_clean.groupby(['Thương hiệu', 'Dòng xe']):
                k = (str(b).strip().lower(), str(m).strip().lower())
                model_medians_ci[k] = float(grp['Giá'].median())

            for (b, t), grp in df_clean.groupby(['Thương hiệu', 'Loại xe']):
                k = (str(b).strip().lower(), str(t).strip().lower())
                brand_type_medians_ci[k] = float(grp['Giá'].median())

            for t, grp in df_clean.groupby('Loại xe'):
                k = str(t).strip().lower()
                type_medians_ci[k] = float(grp['Giá'].median())
    else:
        df_bikes = pd.DataFrame()
        
    return global_pipe, submodels_dict, df_bikes, model_medians_ci, brand_type_medians_ci, type_medians_ci

global_pipe, submodels_dict, df_bikes, model_medians_ci, brand_type_medians_ci, type_medians_ci = load_ml_resources()

ODO_RANGES = [
    ('-- Chọn --', 15000),
    ('Dưới 1.000 Km (Xe mới / Lướt)', 500),
    ('1.000 - 5.000 Km', 3000),
    ('5.000 - 10.000 Km', 7500),
    ('10.000 - 15.000 Km', 12500),
    ('15.000 - 20.000 Km', 17500),
    ('20.000 - 30.000 Km', 25000),
    ('30.000 - 50.000 Km', 40000),
    ('50.000 - 100.000 Km', 70000),
    ('Trên 100.000 Km', 120000),
]
ODO_LABELS = [r[0] for r in ODO_RANGES]
ODO_VALUES = {r[0]: r[1] for r in ODO_RANGES}

# -------------------------------------------------------------
# 4. BỘ HÀM PHÂN TÍCH CHUẨN HÓA THUẬT NGỮ CHÍNH QUY
# -------------------------------------------------------------

def assign_subgen(row):
    model_s = str(row.get('Dòng_xe', row.get('Dòng xe', ''))).lower()
    title_s = str(row.get('Tiêu_đề', row.get('Tiêu đề', ''))).lower()
    year_val = row.get('Năm_đăng_ký', row.get('Năm đăng ký', 2020))
    try:
        yr = int(year_val)
    except:
        yr = 2020
        
    full_str = f"{model_s} {title_s}"
    
    if 'sh mode' in full_str:
        return 'SH Mode'
    elif 'sh' in full_str:
        if '150' in full_str or '300' in full_str or '350' in full_str:
            if yr >= 2017: return 'SH 150i (Đời >=2017)'
            else: return 'SH 150i (Đời <2017)'
        elif '125' in full_str:
            return 'SH 125i'
        else:
            if yr >= 2017: return 'SH 150i (Đời >=2017)'
            else: return 'SH 150i (Đời <2017)'
    elif 'air blade' in full_str or 'ab' in full_str:
        if '150' in full_str or '160' in full_str: return 'Air Blade 150/160'
        else: return 'Air Blade 125'
    elif 'exciter' in full_str:
        return 'Exciter 150'
    elif 'vision' in full_str:
        return 'Vision'
    elif 'vario' in full_str:
        return 'Vario'
    elif 'janus' in full_str:
        return 'Janus'
    elif 'sirius' in full_str:
        return 'Sirius'
    elif 'wave' in full_str:
        if yr >= 2017: return 'Wave Alpha (Đời >=2017)'
        else: return 'Wave Alpha (Đời <2017)'
        
    return 'Khác'

def map_2tier_model(row):
    brand = str(row.get('Thương hiệu', '')).strip()
    model = str(row.get('Dòng xe', '')).strip()
    return f"{brand} - {model}"

def classify_risk(anomaly_score):
    if anomaly_score <= 25.0:
        return "🟢 Mức Giá Hợp Lý (Phù hợp dải thị trường)"
    elif anomaly_score <= 50.0:
        return "⚠️ Độ Lệch Giá Trung Bình (Cần kiểm tra thêm)"
    elif anomaly_score <= 75.0:
        return "🚨 Độ Lệch Giá Cao (Cần xác minh chất lượng)"
    else:
        return "🔥 Độ Lệch Giá Bất Thường Rất Cao"

def predict_bike_price(input_user, global_pipeline, submodels_trained):
    brand = input_user.get('brand', '')
    model = input_user.get('model', '')
    year = input_user.get('year', 2020)
    odo = input_user.get('odo', 20000)
    bike_type = input_user.get('bike_type', 'Tay ga')
    engine_cc = input_user.get('engine_cc', 110.0)
    condition_text = input_user.get('condition_text', '')
    price_listed = input_user.get('price_listed', None)
    
    tuoi_xe = 2025 - year
    len_mota = len(condition_text.split())
    text_l = condition_text.lower()
    has_zin = int(bool(re.search(r'(zin|nguyên zin|nguyên bản|chưa rớt đầu|máy zin)', text_l)))
    has_cc = int(bool(re.search(r'(chính chủ|sang tên|hợp lệ|ủy quyền|ký giấy)', text_l)))
    has_keng = int(bool(re.search(r'(rất mới|mới keng|bảo quản tốt|lướt|sơn zin|đẹp)', text_l)))
    
    row_dummy = {'Thương hiệu': brand, 'Dòng xe': model}
    dong_xe_pc = map_2tier_model(pd.Series(row_dummy))
    
    row_eval = {'Dòng_xe': model, 'Năm_đăng_ký': year, 'Tiêu_đề': f"{brand} {model}"}
    subgen_name = assign_subgen(pd.Series(row_eval))
    
    if submodels_trained and subgen_name in submodels_trained:
        sub_info = submodels_trained[subgen_name]
        mdl = sub_info['pipeline']
        q25 = sub_info['q25']
        q75 = sub_info['q75']
        
        X_in = pd.DataFrame([{
            'Thương_hiệu': brand,
            'Dòng_xe_phân_cấp': dong_xe_pc,
            'Loại_xe': bike_type,
            'Dung_tich_cc': engine_cc if engine_cc else 110.0,
            'Năm_đăng_ký': year,
            'Tuổi_xe': tuoi_xe,
            'Số_Km': odo,
            'Len_MoTa': len_mota,
            'Has_Zin': has_zin,
            'Has_ChinhChu': has_cc,
            'Has_Moi_Keng': has_keng
        }])
        
        try:
            pred_log = mdl.predict(X_in)[0]
            pred_price = round(float(np.expm1(pred_log)), 1)
        except Exception as e:
            pred_price = round((q25 + q75) / 2.0, 1)
            
        p_min = round(q25, 1)
        p_max = round(q75, 1)
        tier_str = f'Tầng 1 (Submodel: {subgen_name})'
    else:
        # TẦNG 2 DYNAMIC MEDIAN LOOKUP TỪ 7.208 BẢN GHI DỮ LIỆU
        brand_k = str(brand).strip().lower()
        model_k = str(model).strip().lower()
        type_k = str(bike_type).strip().lower()
        
        base_price = model_medians_ci.get((brand_k, model_k))
        if base_price is None or pd.isna(base_price):
            base_price = brand_type_medians_ci.get((brand_k, type_k))
        if base_price is None or pd.isna(base_price):
            base_price = type_medians_ci.get(type_k)
            
        if base_price is None or pd.isna(base_price):
            ecc = engine_cc if engine_cc else 110
            if ecc >= 300: base_price = 95.0
            elif ecc >= 150: base_price = 45.0
            elif bike_type == 'Tay ga': base_price = 32.0
            else: base_price = 18.0
            
        pred_price = round(base_price * (0.93 ** tuoi_xe) * max(0.85, 1.0 - (float(odo)/100000.0)*0.2), 1)
        p_min = round(pred_price * 0.85, 1)
        p_max = round(pred_price * 1.15, 1)
        tier_str = f'Tầng 2 (Trung vị Dòng xe: {base_price:.1f}tr)'

    eval_price_str = 'Chưa cung cấp giá niêm yết'
    anomaly_score = 0.0
    
    if price_listed is not None and price_listed > 0:
        pl = float(price_listed)
        if pl < p_min * 0.8:
            eval_price_str = f'Thấp hơn dải thị trường khuyên dùng (Dưới {p_min} triệu)'
        elif pl > p_max * 1.2:
            eval_price_str = f'Cao hơn dải thị trường khuyên dùng (Trên {p_max} triệu)'
        else:
            eval_price_str = f'Phù hợp dải thị trường khuyên dùng ({p_min} - {p_max} triệu)'
            
        s_res = min(1.0, abs(pl - pred_price) / max(pred_price, 1.0))
        s_b = 0.0
        if pl < p_min: s_b = (p_min - pl) / p_min
        elif pl > p_max: s_b = (pl - p_max) / p_max
        s_b = min(1.0, s_b)
        
        anomaly_score = round(100 * (0.45 * s_res + 0.55 * s_b), 1)
        
    risk_str = classify_risk(anomaly_score)
    
    return {
        'tier': tier_str,
        'pred_price': pred_price,
        'p_min': p_min,
        'p_max': p_max,
        'eval_str': eval_price_str,
        'anomaly_score': anomaly_score,
        'risk_str': risk_str,
        'nlp_info': {
            'has_zin': has_zin,
            'has_cc': has_cc,
            'has_keng': has_keng,
            'len_mota': len_mota
        }
    }

# -------------------------------------------------------------
# 5. GIAO DIỆN CHÍNH (SIDEBAR MODEL LOADER & TAB NAVIGATION)
# -------------------------------------------------------------
def main():
    st.markdown("""
    <div class="enterprise-header">
        <h1>🏍️ HỆ THỐNG PHÂN TÍCH & ĐỊNH GIÁ THỊ TRƯỜNG XE MÁY CỦ</h1>
        <p>Hệ Thống Phân Tích Dữ Liệu Học Máy Tách Biệt Chức Năng ĐỊNH GIÁ THỊ TRƯỜNG & KIỂM TRA ĐỘ LỆCH GIÁ TÍN HIỆU (Price Anomaly Detection)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------
    # SIDEBAR: KHUNG QUẢN LÝ & NẠP MÔ HÌNH ML (MODEL LOADER & MANAGER)
    # -------------------------------------------------------------
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/171/171239.png", width=65)
    st.sidebar.title("📌 Quản Lý Mô Hình ML")
    
    # Model Status Health Card
    num_submodels = len(submodels_dict) if submodels_dict else 0
    st.sidebar.markdown(f"""
    <div style='background-color: #1e293b; border: 1px solid #ffba00; padding: 12px; border-radius: 10px; color: #f8fafc; font-size: 0.85rem;'>
        <h5 style='margin-top:0; color:#ffba00;'>🟢 Trạng Thái Nạp Mô Hình:</h5>
        <p style='margin-bottom:4px;'>• <b>Global Pipeline:</b> Đã nạp thành công</p>
        <p style='margin-bottom:4px;'>• <b>Submodels Chuyên Biệt:</b> {num_submodels} mô hình</p>
        <p style='margin-bottom:0;'>• <b>Cơ Sở Dữ Liệu:</b> 7.208 bài đăng (37 Hãng, 224 Dòng xe)</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar.expander("📋 Danh sách 11 Submodels chuyên biệt"):
        if submodels_dict:
            for k in submodels_dict.keys():
                st.write(f"• `{k}`")
        else:
            st.caption("Chưa nạp submodel")
            
    st.sidebar.markdown("#### 📦 Nạp Mô Hình Tùy Chỉnh")
    up_model_file = st.sidebar.file_uploader("Nạp file mô hình huấn luyện (.pkl / .joblib):", type=["pkl", "joblib"], key="custom_model_uploader")
    if up_model_file is not None:
        st.sidebar.success(f"✅ Đã tiếp nhận file mô hình: {up_model_file.name}")
        
    if st.sidebar.button("🔄 Nạp Lại Mô Hình (Reset Cache)", use_container_width=True):
        st.cache_resource.clear()
        st.sidebar.success("✅ Đã làm mới bộ nhớ cache và nạp lại mô hình!")
        st.rerun()
        
    st.sidebar.divider()
    st.sidebar.markdown("""
    <div style='font-size: 0.82rem; color: #94a3b8; line-height: 1.5;'>
        <b>👤 Tác giả thực hiện:</b> Nguyễn Văn Nam & Lê Văn Lưu<br/>
        <b>🏍️ Ứng dụng:</b> Định Giá Xe Máy Cũ & Phát Hiện Bất Thường<br/>
        <b>🚀 Phiên bản:</b> Enterprise Edition v2.5
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------
    # CREATE 3 SEPARATED TABS WITH FORMAL LABELS
    # -------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "🔮 Định Giá Thị Trường",
        "🚨 Phân Tích Độ Lệch Giá",
        "📁 Kiểm Duyệt Hàng Loạt"
    ])
    
    # -------------------------------------------------------------
    # HELPER FUNCTION TO RENDER CHỢ TỐT XE STYLE FILTER CAPSULES (LAYOUT 2 HÀNG x 5 CỘT CÂN BẰNG 100%)
    # -------------------------------------------------------------
    def render_form_controls(prefix_key):
        # Styling CSS chuẩn Chợ Tốt Xe (Capsule Sáng Căn Giữa 100% + Nút Bấm Không Con Trỏ Gõ Text)
        st.markdown("""
        <style>
            /* Ép lề chiều dọc giữa các hàng khít lại gần nhau */
            div[data-testid="stHorizontalBlock"] {
                margin-bottom: -16px !important;
            }
            div[data-testid="stSelectbox"] {
                margin-bottom: 2px !important;
            }
            /* Căn giữa chữ 100% và gỡ bỏ con trỏ gõ text (caret-color: transparent) */
            div[data-testid="stSelectbox"] > div > div {
                border-radius: 23px !important;
                padding: 6px 12px !important;
                min-height: 46px !important;
                font-weight: 700 !important;
                font-size: 0.95rem !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
                transition: all 0.2s ease !important;
                text-align: center !important;
                cursor: pointer !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
            div[data-testid="stSelectbox"] input {
                caret-color: transparent !important; /* Ẩn hoàn toàn con trỏ gõ văn bản | */
                cursor: pointer !important;
                user-select: none !important;
            }
            div[data-testid="stSelectbox"] * {
                cursor: pointer !important;
            }
            div[data-testid="stSelectbox"] div[data-baseweb="select"] {
                text-align: center !important;
                cursor: pointer !important;
                width: 100% !important;
            }
            div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
                text-align: center !important;
                justify-content: center !important;
                display: flex !important;
                align-items: center !important;
                cursor: pointer !important;
                width: 100% !important;
            }
            div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
                text-align: center !important;
                width: 100% !important;
                display: block !important;
                cursor: pointer !important;
                font-size: 0.95rem !important;
            }
        </style>
        """, unsafe_allow_html=True)

        # Helper function render capsule Nền TRẮNG (#ffffff) có viền/chữ Đỏ (Chưa chọn) / Xanh lá (Đã chọn)
        def render_styled_capsule(col, label, options, key, is_selected, format_func=str):
            border_color = "#22c55e" if is_selected else "#ef4444"
            text_color = "#15803d" if is_selected else "#b91c1c"
            
            st.markdown(f"""
            <style>
                div[data-testid="stSelectbox"]:has(input[id*="{key}"]),
                div[data-testid="stSelectbox"]:has(div[id*="{key}"]) {{
                    width: 100% !important;
                }}
                div[data-testid="stSelectbox"]:has(input[id*="{key}"]) > div > div,
                div[data-testid="stSelectbox"]:has(div[id*="{key}"]) > div > div {{
                    background-color: #ffffff !important;
                    border: 2px solid {border_color} !important;
                    color: {text_color} !important;
                    border-radius: 23px !important;
                    padding: 6px 12px !important;
                    min-height: 46px !important;
                    font-weight: 700 !important;
                    font-size: 0.95rem !important;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
                    text-align: center !important;
                    cursor: pointer !important;
                    width: 100% !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                }}
                div[data-testid="stSelectbox"]:has(input[id*="{key}"]) div[data-baseweb="select"] span,
                div[data-testid="stSelectbox"]:has(div[id*="{key}"]) div[data-baseweb="select"] span {{
                    color: {text_color} !important;
                    font-weight: 700 !important;
                    text-align: center !important;
                    width: 100% !important;
                    display: block !important;
                    cursor: pointer !important;
                    font-size: 0.95rem !important;
                }}
            </style>
            """, unsafe_allow_html=True)
            
            with col:
                return st.selectbox(
                    label,
                    options,
                    index=0,
                    key=key,
                    label_visibility="collapsed",
                    format_func=format_func
                )

        # ===== BỐ CỤC 3 CỘT NHỎ GỌN (GAP=SMALL, ÉP NÚT 100% RỘNG) =====
        key_b = f"sb_brand_{prefix_key}"
        key_t = f"sb_type_{prefix_key}"
        key_m = f"sb_model_{prefix_key}"
        key_c = f"sb_cc_{prefix_key}"

        # TỰ ĐỘNG ĐỒNG BỘ HÃNG XE & LOẠI XE KHI NGƯỜI DÙNG CHỌN DÒNG XE TRƯỚC
        model_in_state = st.session_state.get(key_m)
        if model_in_state and model_in_state not in ["Dòng xe", "Tất cả"]:
            auto_b = MODEL_BRAND_MAP.get(model_in_state)
            auto_t = MODEL_TYPE_MAP.get(model_in_state)
            if auto_b and st.session_state.get(key_b, "Hãng xe") == "Hãng xe":
                st.session_state[key_b] = auto_b
            if auto_t and st.session_state.get(key_t, "Loại xe") == "Loại xe":
                st.session_state[key_t] = auto_t
            
            # Đồng bộ dung tích cc nếu dòng xe chỉ có duy nhất 1 mức cc
            cc_list = get_model_cc_options(model_in_state, auto_t if auto_t else "Tay ga")
            if len(cc_list) == 1:
                st.session_state[key_c] = f"{cc_list[0]} cc"

        r1c1, r1c2, r1c3 = st.columns(3, gap="small")
        
        # 1. HÃNG XE (Capsule Selectbox)
        brand_options = ["Hãng xe"] + ALL_BRANDS_LIST
        is_sel_b = st.session_state.get(key_b, "Hãng xe") != "Hãng xe"
        selected_b_radio = render_styled_capsule(r1c1, "Hãng xe", brand_options, key_b, is_sel_b)
        selected_brand = selected_b_radio if (selected_b_radio and selected_b_radio != "Hãng xe") else None
        
        # 2. LOẠI XE (Capsule Selectbox)
        all_type_order = ["Tay ga", "Xe số", "Tay côn/Moto"]
        if selected_brand:
            brand_models = FULL_BRAND_MAP.get(selected_brand, [])
            available_types = set(MODEL_TYPE_MAP.get(m, 'Tay ga') for m in brand_models)
            filtered_types = [t for t in all_type_order if t in available_types]
        else:
            filtered_types = all_type_order
            
        type_options = ["Loại xe"] + filtered_types
        is_sel_t = st.session_state.get(key_t, "Loại xe") != "Loại xe"
        selected_t_radio = render_styled_capsule(r1c2, "Loại xe", type_options, key_t, is_sel_t)
        selected_type = selected_t_radio if (selected_t_radio and selected_t_radio != "Loại xe") else None

        # 3. DÒNG XE (Capsule Selectbox)
        if selected_brand:
            avail_models = FULL_BRAND_MAP.get(selected_brand, [])
        else:
            avail_models = sorted(list(MODEL_BRAND_MAP.keys()))
            
        if selected_type:
            avail_models = [m for m in avail_models if MODEL_TYPE_MAP.get(m) == selected_type]
            
        model_options = ["Dòng xe"] + avail_models
        is_sel_m = st.session_state.get(key_m, "Dòng xe") != "Dòng xe"
        selected_m_radio = render_styled_capsule(r1c3, "Dòng xe", model_options, key_m, is_sel_m)
        selected_model = selected_m_radio if (selected_m_radio and selected_m_radio != "Dòng xe") else None

        # ===== HÀNG 2: NĂM ĐĂNG KÝ | QUÃNG ĐƯỜNG | DUNG TÍCH =====
        r2c1, r2c2, r2c3 = st.columns(3, gap="small")

        # 4. NĂM ĐĂNG KÝ (Capsule Selectbox)
        y_opts = ["Năm đăng ký"] + [str(y) for y in list(range(2025, 2011, -1))] + ["Trước 2012"]
        key_y = f"sb_year_{prefix_key}"
        is_sel_y = st.session_state.get(key_y, "Năm đăng ký") != "Năm đăng ký"
        selected_y_radio = render_styled_capsule(r2c1, "Năm đăng ký", y_opts, key_y, is_sel_y)

        # 5. QUÃNG ĐƯỜNG (Capsule Selectbox)
        o_opts = ["Quãng đường"] + ODO_LABELS
        key_o = f"sb_odo_{prefix_key}"
        is_sel_o = st.session_state.get(key_o, "Quãng đường") != "Quãng đường"
        selected_o_radio = render_styled_capsule(r2c2, "Quãng đường", o_opts, key_o, is_sel_o)

        # 6. DUNG TÍCH (cc)
        cc_opts_num = get_model_cc_options(selected_model, selected_type)
        if len(cc_opts_num) == 1 and selected_model is not None:
            cc_opts_str = [f"{cc_opts_num[0]} cc"]
            key_c = f"sb_cc_{prefix_key}"
            is_sel_c = True
            selected_c_radio = render_styled_capsule(r2c3, "Dung tích", cc_opts_str, key_c, is_sel_c)
        else:
            cc_opts_str = ["Dung tích"] + [f"{c} cc" for c in cc_opts_num]
            key_c = f"sb_cc_{prefix_key}"
            is_sel_c = st.session_state.get(key_c, "Dung tích") != "Dung tích"
            selected_c_radio = render_styled_capsule(r2c3, "Dung tích", cc_opts_str, key_c, is_sel_c)

        # ===== HÀNG 3: PHÁP LÝ | ĐỘNG CƠ | NGOẠI HÌNH =====
        r3c1, r3c2, r3c3 = st.columns(3, gap="small")

        # 7. PHÁP LÝ (Capsule Selectbox)
        d_opts = ["Pháp lý", "Chính chủ", "Hợp lệ", "Ủy quyền"]
        key_d = f"sb_doc_{prefix_key}"
        is_sel_d = st.session_state.get(key_d, "Pháp lý") != "Pháp lý"
        selected_d_radio = render_styled_capsule(r3c1, "Pháp lý", d_opts, key_d, is_sel_d)

        # 8. ĐỘNG CƠ (Capsule Selectbox)
        e_opts = ["Động cơ", "Nguyên bản", "Bảo dưỡng", "Đại tu"]
        key_e = f"sb_eng_{prefix_key}"
        is_sel_e = st.session_state.get(key_e, "Động cơ") != "Động cơ"
        selected_e_radio = render_styled_capsule(r3c2, "Động cơ", e_opts, key_e, is_sel_e)

        # 9. NGOẠI HÌNH (Capsule Selectbox)
        bg_opts = ["Ngoại hình", "Rất mới", "Trầy nhẹ", "Sơn zin"]
        key_bg = f"sb_body_{prefix_key}"
        is_sel_bg = st.session_state.get(key_bg, "Ngoại hình") != "Ngoại hình"
        selected_bg_radio = render_styled_capsule(r3c3, "Ngoại hình", bg_opts, key_bg, is_sel_bg)

        # NÚT KHỞI TẠO LẠI BỘ LỌC (RESET ALL CAPSULES)
        if st.button("🔄 Khởi Tạo Lại Thông Số", key=f"btn_reset_{prefix_key}", use_container_width=True):
            for k in [key_b, key_t, key_m, key_y, key_o, key_c, key_d, key_e, key_bg]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

        # Đồng bộ nguyên khối bằng resolve_strict_vehicle
        final_brand, final_model, final_type, engine_cc_val = resolve_strict_vehicle(
            selected_brand, selected_type, selected_model, selected_c_radio
        )
        
        # Parsing Year
        if selected_y_radio and selected_y_radio not in ["Năm đăng ký", "Tất cả"]:
            if "Trước" in selected_y_radio:
                year_int = 2011
            else:
                try: year_int = int(selected_y_radio)
                except: year_int = 2020
        else:
            year_int = 2020

        # Parsing ODO
        is_odo_selected = bool(selected_o_radio and selected_o_radio not in ["Quãng đường", "Tất cả"])
        if is_odo_selected:
            odo_val = ODO_VALUES.get(selected_o_radio, 30000)
        else:
            odo_val = 30000

        parts = []
        if selected_d_radio and selected_d_radio not in ["Pháp lý", "Tất cả"]: parts.append(selected_d_radio)
        if selected_e_radio and selected_e_radio not in ["Động cơ", "Tất cả"]: parts.append(selected_e_radio)
        if selected_bg_radio and selected_bg_radio not in ["Ngoại hình", "Tất cả"]: parts.append(selected_bg_radio)
        combined_text = ", ".join(parts) + ("." if parts else "")

        # Cờ xác định xem người dùng đã chọn đủ thông tin cơ bản chưa (Hãng xe, Dòng xe, Năm đăng ký)
        has_basic_info = bool(is_sel_b and is_sel_m and is_sel_y)

        return {
            'brand': final_brand,
            'model': final_model,
            'bike_type': final_type,
            'year': year_int,
            'odo': odo_val,
            'is_type_selected': is_sel_t,
            'is_year_selected': is_sel_y,
            'is_odo_selected': is_sel_o,
            'is_cc_selected': is_sel_c,
            'engine_cc': engine_cc_val,
            'condition_text': combined_text
        }, has_basic_info

    # -------------------------------------------------------------
    # HELPER COMPONENT VẼ THANH TRỰC QUAN DẢI GIÁ THỊ TRƯỜNG CHỢ TỐT XE
    # -------------------------------------------------------------
    def render_market_price_gauge(p_min, p_max, p_current, title="Khoảng giá thị trường", is_listed=False):
        span = max(p_max - p_min, 1.0)
        buffer = span * 0.35
        view_min = max(0.0, p_min - buffer)
        view_max = p_max + buffer
        total_view_range = max(view_max - view_min, 0.1)

        min_pct = max(2, min(98, float(((p_min - view_min) / total_view_range) * 100)))
        max_pct = max(2, min(98, float(((p_max - view_min) / total_view_range) * 100)))
        cur_pct = max(2, min(98, float(((p_current - view_min) / total_view_range) * 100)))

        if is_listed:
            if p_current < p_min:
                pin_color = "#ef4444"
                pin_label = f"{p_current:.2f} tr (Giá rẻ hơn thị trường)"
            elif p_current > p_max:
                pin_color = "#f59e0b"
                pin_label = f"{p_current:.2f} tr (Giá cao hơn thị trường)"
            else:
                pin_color = "#2563eb"
                pin_label = f"{p_current:.2f} tr (Giá niêm yết hợp lý)"
        else:
            pin_color = "#2563eb"
            pin_label = f"{p_current:.2f} tr"

        gauge_html = (
            f"<div style='background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 24px 28px; margin-top: 16px; color: #f8fafc; box-shadow: 0 4px 15px rgba(0,0,0,0.25);'>"
            f"<div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;'>"
            f"<div style='display: flex; align-items: center; gap: 8px;'>"
            f"<span style='color: #f8fafc; font-weight: 800; font-size: 1.25rem;'>{title}</span>"
            f"<span style='color: #94a3b8; font-size: 1.1rem; cursor: pointer;' title='Khoảng giá thị trường khuyên dùng dựa trên dữ liệu giao dịch Chợ Tốt Xe'>ⓘ</span>"
            f"</div>"
            f"<div style='color: #cbd5e1; font-size: 1.0rem; font-weight: 600;'>"
            f"Dải giá khuyên dùng: <b style='color: #38bdf8;'>{p_min:.2f} tr - {p_max:.2f} tr</b>"
            f"</div>"
            f"</div>"
            f"<div style='position: relative; height: 80px; width: 100%; margin-top: 15px;'>"
            f"<div style='position: absolute; top: 48px; left: 0; right: 0; height: 14px; background: rgba(255,255,255,0.12); border-radius: 7px;'></div>"
            f"<div style='position: absolute; top: 48px; left: {min_pct:.1f}%; width: {max(max_pct - min_pct, 4):.1f}%; height: 14px; background: #3b82f6; border-radius: 7px; box-shadow: 0 0 12px rgba(59,130,246,0.6);'></div>"
            f"<div style='position: absolute; top: 0px; left: min(max(calc({cur_pct:.1f}% - 45px), 0px), calc(100% - 90px)); z-index: 10;'>"
            f"<div style='background: {pin_color}; color: #ffffff; padding: 6px 16px; border-radius: 10px; font-size: 1.0rem; font-weight: 800; white-space: nowrap; box-shadow: 0 4px 12px rgba(0,0,0,0.4); text-align: center;'>{pin_label}</div>"
            f"<div style='width: 0; height: 0; border-left: 7px solid transparent; border-right: 7px solid transparent; border-top: 8px solid {pin_color}; margin: 0 auto;'></div>"
            f"</div>"
            f"<div style='position: absolute; top: 42px; left: {cur_pct:.1f}%; width: 3px; height: 22px; background: {pin_color}; transform: translateX(-50%); z-index: 5;'></div>"
            f"</div>"
            f"<div style='position: relative; width: 100%; height: 26px; margin-top: 6px;'>"
            f"<div style='position: absolute; left: {min_pct:.1f}%; transform: translateX(-50%); color: #f8fafc; font-weight: 800; font-size: 1.05rem;'>{p_min:.2f} tr</div>"
            f"<div style='position: absolute; left: {max_pct:.1f}%; transform: translateX(-50%); color: #f8fafc; font-weight: 800; font-size: 1.05rem;'>{p_max:.2f} tr</div>"
            f"</div>"
            f"</div>"
        )
        return gauge_html

    # -------------------------------------------------------------
    # TAB 1: 🔮 ĐỊNH GIÁ THỊ TRƯỜNG (DỰ ĐOÁN THUẦN TÚY - KHÔNG CẦN GIÁ RAO)
    # -------------------------------------------------------------
    with tab1:
        st.markdown("### 🔮 Định Giá Thị Trường Xe Máy Cũ")
        st.caption("Dành cho người mua hoặc người bán tra cứu mức giá thị trường khuyên dùng của phương tiện:")
        
        col_left, col_right = st.columns([3, 2])
        with col_left:
            inputs1, has_basic1 = render_form_controls("tab1")
            
        with col_right:
            b_m_str1 = f"{inputs1['brand']} {inputs1['model']}" if has_basic1 else "Chưa chọn (Cần Hãng xe, Dòng xe & Năm)"
            
            t_sel1 = inputs1.get('is_type_selected', False)
            c_sel1 = inputs1.get('is_cc_selected', False)
            if t_sel1 and c_sel1: type_cc_str1 = f"{inputs1['bike_type']} • {inputs1['engine_cc']}cc"
            elif t_sel1: type_cc_str1 = f"{inputs1['bike_type']} • Chưa chọn CC"
            elif c_sel1: type_cc_str1 = f"Chưa chọn loại xe • {inputs1['engine_cc']}cc"
            else: type_cc_str1 = "Chưa chọn"

            y_sel1 = inputs1.get('is_year_selected', False)
            o_sel1 = inputs1.get('is_odo_selected', False)
            if y_sel1 and o_sel1: year_odo_str1 = f"Đời {inputs1['year']} • {inputs1['odo']:,} Km"
            elif y_sel1: year_odo_str1 = f"Đời {inputs1['year']} • Chưa chọn ODO"
            elif o_sel1: year_odo_str1 = f"Chưa chọn đời xe • {inputs1['odo']:,} Km"
            else: year_odo_str1 = "Chưa chọn"
            
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); border: 1px solid rgba(255,186,0,0.3); border-radius: 12px; padding: 20px; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
                <h4 style='color: #ffba00; margin-top: 0; margin-bottom: 15px; font-weight: 700;'><i class='fa fa-list-alt'></i> Tóm Tắt Thông Số Lựa Chọn</h4>
                <div style='margin-bottom: 10px;'>
                    <span style='color: #cbd5e1; font-size: 0.9rem;'>🏍️ Hãng & Dòng xe:</span><br/>
                    <strong style='color: {"#38bdf8" if has_basic1 else "#f87171"}; font-size: 1.05rem;'>{b_m_str1}</strong>
                </div>
                <div style='margin-bottom: 10px;'>
                    <span style='color: #cbd5e1; font-size: 0.9rem;'>⚙️ Loại xe & Dung tích:</span><br/>
                    <strong style='color: {"#f8fafc" if (t_sel1 or c_sel1) else "#94a3b8"}; font-size: 1.0rem;'>{type_cc_str1}</strong>
                </div>
                <div style='margin-bottom: 15px;'>
                    <span style='color: #cbd5e1; font-size: 0.9rem;'>📅 Năm đăng ký & ODO:</span><br/>
                    <strong style='color: {"#f8fafc" if (y_sel1 or o_sel1) else "#94a3b8"}; font-size: 1.0rem;'>{year_odo_str1}</strong>
                </div>
                <div style='margin-top: auto;'>
                    { "<span style='background: rgba(34, 197, 94, 0.2); color: #22c55e; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid #22c55e;'>✅ Đã chọn đủ thông tin cơ bản</span>" if has_basic1 else "<span style='background: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid #ef4444;'>⚠️ Chưa chọn đủ 3 thông tin cơ bản</span>" }
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        btn_predict1 = st.button("🔮 Thực Hiện Định Giá Xe", type="primary", use_container_width=True, key="btn_predict_tab1")
        
        if not btn_predict1:
            st.markdown("""
            <div style='background: rgba(15, 23, 42, 0.7); border: 1px dashed rgba(56, 189, 248, 0.4); border-radius: 14px; padding: 28px 20px; text-align: center; margin-top: 15px;'>
                <div style='font-size: 2.2rem; margin-bottom: 6px;'>🛵</div>
                <h4 style='color: #38bdf8; font-weight: 700; margin: 0 0 6px 0; font-size: 1.1rem;'>Vui lòng chọn thông số và nhấn nút định giá</h4>
                <p style='color: #94a3b8; font-size: 0.9rem; margin: 0;'>Vui lòng chọn đủ 3 thông tin cơ bản: <b>[Hãng xe]</b>, <b>[Dòng xe]</b> và <b>[Năm đăng ký]</b> ở các thẻ phía trên, sau đó bấm nút <b>🔮 Thực Hiện Định Giá Xe</b>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if not has_basic1:
                st.warning("⚠️ VUI LÒNG CHỌN ĐỦ 3 THÔNG TIN CƠ BẢN: [HÃNG XE], [DÒNG XE] VÀ [NĂM ĐĂNG KÝ] ĐỂ THỰC HIỆN ĐỊNH GIÁ!")
            else:
                # ===== HIỂN THỊ KẾT QUẢ ĐỊNH GIÁ THỊ TRƯỜNG KHI CHỌN ĐỦ THÔNG TIN VÀ BẤM NÚT DỰ ĐOÁN =====
                res1 = predict_bike_price(inputs1, global_pipe, submodels_dict)
                
                st.divider()
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95)); border: 1px solid #ffba00; border-radius: 12px; padding: 16px; margin-top: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
                    <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 186, 0, 0.2); padding-bottom: 8px; margin-bottom: 12px;'>
                        <span style='color: #ffba00; font-weight: 800; font-size: 1.1rem;'>🎯 KẾT QUẢ ĐỊNH GIÁ THỊ TRƯỜNG</span>
                        <span style='background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #38bdf8; padding: 3px 10px; border-radius: 15px; font-weight: 700; font-size: 0.8rem;'>● Đã tính toán</span>
                    </div>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px; text-align: center;'>
                        <div style='background: rgba(255,255,255,0.03); padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);'>
                            <div style='color: #94a3b8; font-size: 0.85rem;'>Phương Tiện Được Chọn</div>
                            <div style='color: #38bdf8; font-weight: 700; font-size: 1.1rem; margin-top: 4px;'>{inputs1['brand']} {inputs1['model']}</div>
                            <div style='color: #cbd5e1; font-size: 0.82rem;'>{inputs1['bike_type']} • {inputs1['engine_cc']}cc • Đời {inputs1['year']}</div>
                        </div>
                        <div style='background: rgba(255,255,255,0.03); padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);'>
                            <div style='color: #94a3b8; font-size: 0.85rem;'>Mô Hình Định Tuyến</div>
                            <div style='color: #ffba00; font-weight: 700; font-size: 1.0rem; margin-top: 6px;'>{res1['tier']}</div>
                            <div style='color: #94a3b8; font-size: 0.8rem; margin-top: 4px;'>Tự động định tuyến submodel</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # BỔ SUNG THANH TRỰC QUAN DẢI GIÁ THỊ TRƯỜNG (CHỢ TỐT XE MARKET PRICE RANGE GAUGE)
                gauge1 = render_market_price_gauge(res1['p_min'], res1['p_max'], res1['pred_price'], title="Khoảng giá thị trường", is_listed=False)
                st.markdown(gauge1, unsafe_allow_html=True)
            
            st.write("")
            st.info("💡 **Ghi chú:** Kết quả định giá được tính toán dựa trên mô hình Học máy trained trên 7.208 tập bản ghi dữ liệu giao dịch xe máy cũ.")

    # -------------------------------------------------------------
    # TAB 2: 🚨 PHÂN TÍCH ĐỘ LỆCH GIÁ (BẮT BUỘC NHẬP GIÁ RAO)
    # -------------------------------------------------------------
    with tab2:
        st.markdown("### 🚨 Phân Tích Độ Lệch Giá & Tín Hiệu Bất Thường Tin Rao")
        st.caption("Dành cho đơn vị kiểm duyệt tin rao hoặc người mua đối chiếu mức giá niêm yết với dải giá thị trường chuẩn:")
        
        col_left, col_right = st.columns([3, 2])
        with col_left:
            inputs2, has_basic2 = render_form_controls("tab2")
            
        with col_right:
            b_m_str2 = f"{inputs2['brand']} {inputs2['model']}" if has_basic2 else "Chưa chọn (Cần Hãng xe, Dòng xe & Năm)"
            
            t_sel2 = inputs2.get('is_type_selected', False)
            c_sel2 = inputs2.get('is_cc_selected', False)
            if t_sel2 and c_sel2: type_cc_str2 = f"{inputs2['bike_type']} • {inputs2['engine_cc']}cc"
            elif t_sel2: type_cc_str2 = f"{inputs2['bike_type']} • Chưa chọn CC"
            elif c_sel2: type_cc_str2 = f"Chưa chọn loại xe • {inputs2['engine_cc']}cc"
            else: type_cc_str2 = "Chưa chọn"

            y_sel2 = inputs2.get('is_year_selected', False)
            o_sel2 = inputs2.get('is_odo_selected', False)
            if y_sel2 and o_sel2: year_odo_str2 = f"Đời {inputs2['year']} • {inputs2['odo']:,} Km"
            elif y_sel2: year_odo_str2 = f"Đời {inputs2['year']} • Chưa chọn ODO"
            elif o_sel2: year_odo_str2 = f"Chưa chọn đời xe • {inputs2['odo']:,} Km"
            else: year_odo_str2 = "Chưa chọn"
            
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); border: 1px solid rgba(255,186,0,0.3); border-radius: 12px; padding: 20px; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
                <h4 style='color: #ffba00; margin-top: 0; margin-bottom: 15px; font-weight: 700;'><i class='fa fa-list-alt'></i> Tóm Tắt Thông Số Lựa Chọn</h4>
                <div style='margin-bottom: 10px;'>
                    <span style='color: #cbd5e1; font-size: 0.9rem;'>🏍️ Hãng & Dòng xe:</span><br/>
                    <strong style='color: {"#38bdf8" if has_basic2 else "#f87171"}; font-size: 1.05rem;'>{b_m_str2}</strong>
                </div>
                <div style='margin-bottom: 10px;'>
                    <span style='color: #cbd5e1; font-size: 0.9rem;'>⚙️ Loại xe & Dung tích:</span><br/>
                    <strong style='color: {"#f8fafc" if has_basic2 else "#94a3b8"}; font-size: 1.0rem;'>{type_cc_str2}</strong>
                </div>
                <div style='margin-bottom: 15px;'>
                    <span style='color: #cbd5e1; font-size: 0.9rem;'>📅 Năm đăng ký & ODO:</span><br/>
                    <strong style='color: {"#f8fafc" if has_basic2 else "#94a3b8"}; font-size: 1.0rem;'>{year_odo_str2}</strong>
                </div>
                <div style='margin-bottom: 15px;'>
                    { "<span style='background: rgba(34, 197, 94, 0.2); color: #22c55e; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid #22c55e;'>✅ Đã chọn đủ thông tin cơ bản</span>" if has_basic2 else "<span style='background: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid #ef4444;'>⚠️ Chưa chọn đủ 3 thông tin cơ bản</span>" }
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # BẮT BUỘC NHẬP GIÁ RAO NIÊM YẾT (Trắng rỗng value=None)
            price_listed_tab2 = st.number_input(
                "8. Mức giá rao niêm yết cần kiểm tra (Triệu VND) *:",
                min_value=0.0,
                max_value=500.0,
                value=None, # TRẮNG RỖNG MẶC ĐỊNH
                step=1.0,
                placeholder="Nhập mức giá rao bán cần đối chiếu (VD: 85.0)...",
                key=f"price_tab2_{inputs2['brand']}_{inputs2['model']}"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        btn_pred2 = st.button("🔎 Phân Tích Độ Lệch Giá Tin Rao", type="primary", use_container_width=True, key="btn_pred_tab2")
        
        if not btn_pred2:
            st.markdown("""
            <div style='background: rgba(15, 23, 42, 0.7); border: 1px dashed rgba(239, 68, 68, 0.4); border-radius: 14px; padding: 28px 20px; text-align: center; margin-top: 15px;'>
                <div style='font-size: 2.2rem; margin-bottom: 6px;'>🚨</div>
                <h4 style='color: #f87171; font-weight: 700; margin: 0 0 6px 0; font-size: 1.1rem;'>Vui lòng nhập Giá rao niêm yết và nhấn nút phân tích</h4>
                <p style='color: #94a3b8; font-size: 0.9rem; margin: 0;'>Vui lòng chọn thông số xe ở trên, nhập <b>Mức giá rao niêm yết (Triệu VND)</b> tại ô số 8, sau đó bấm nút <b>🔎 Phân Tích Độ Lệch Giá Tin Rao</b>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if not has_basic2:
                st.warning("⚠️ VUI LÒNG CHỌN ĐỦ 3 THÔNG TIN CƠ BẢN: [HÃNG XE], [DÒNG XE] VÀ [NĂM ĐĂNG KÝ] ĐỂ THỰC HIỆN ĐỐI CHIẾU!")
            elif price_listed_tab2 is None or price_listed_tab2 <= 0.0:
                st.warning("⚠️ VUI LÒNG NHẬP MỨC GIÁ RAO NIÊM YẾT (TRIỆU VNĐ) ĐỂ THỰC HIỆN ĐỐI CHIẾU MÔ HÌNH!")
            else:
                inputs2['price_listed'] = price_listed_tab2
                res2 = predict_bike_price(inputs2, global_pipe, submodels_dict)
                
                st.divider()
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95)); border: 1px solid #ef4444; border-radius: 12px; padding: 16px; margin-top: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
                    <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(239, 68, 68, 0.2); padding-bottom: 8px; margin-bottom: 12px;'>
                        <span style='color: #f87171; font-weight: 800; font-size: 1.1rem;'>🚨 KẾT QUẢ DỰ ĐOÁN & PHÁT HIỆN GIÁ BẤT THƯỜNG TIN RAO</span>
                        <span style='background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #ef4444; padding: 3px 10px; border-radius: 15px; font-weight: 700; font-size: 0.8rem;'>● Đã phân tích độ lệch</span>
                    </div>
                    <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; text-align: center;'>
                        <div style='background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);'>
                            <div style='color: #94a3b8; font-size: 0.8rem;'>Mức Giá Rao Niêm Yết</div>
                            <div style='color: #38bdf8; font-weight: 800; font-size: 1.6rem; margin: 2px 0;'>{price_listed_tab2} Triệu đ</div>
                            <div style='color: #cbd5e1; font-size: 0.78rem;'>Giá khai báo trên bài đăng</div>
                        </div>
                        <div style='background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(34, 197, 94, 0.3);'>
                            <div style='color: #94a3b8; font-size: 0.8rem;'>Giá Thị Trường Khuyên Dùng</div>
                            <div style='color: #22c55e; font-weight: 800; font-size: 1.6rem; margin: 2px 0;'>{res2['pred_price']} Triệu đ</div>
                            <div style='color: #cbd5e1; font-size: 0.78rem;'>Khoảng dải chuẩn: <b style='color:#ffba00;'>{res2['p_min']} - {res2['p_max']} tr</b></div>
                        </div>
                        <div style='background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);'>
                            <div style='color: #94a3b8; font-size: 0.8rem;'>Chỉ Số Độ Lệch Bất Thường</div>
                            <div style='color: #f87171; font-weight: 800; font-size: 1.6rem; margin: 2px 0;'>{res2['anomaly_score']} / 100</div>
                            <div style='margin-top: 4px;'><span style='background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; padding: 2px 8px; border-radius: 12px; font-weight: 700; font-size: 0.78rem;'>{res2['risk_str']}</span></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.progress(min(1.0, float(res2['anomaly_score']) / 100.0))
                st.info(f"💡 **Đánh giá tổng quan:** {res2['eval_str']}")
                
                # BỔ SUNG THANH TRỰC QUAN DẢI GIÁ THỊ TRƯỜNG CHO TAB 2
                gauge2 = render_market_price_gauge(res2['p_min'], res2['p_max'], price_listed_tab2, title="Đối chiếu Khoảng giá thị trường tin rao", is_listed=True)
                st.markdown(gauge2, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 3: 📁 KIỂM DUYỆT TIN RAO HÀNG LOẠT (IMPORT & EXPORT)
    # -------------------------------------------------------------
    with tab3:
        st.markdown("### 📁 Phân Tích & Kiểm Duyệt Tin Rao Hàng Loạt")
        st.caption("Dành cho đơn vị quản lý nạp tập tin bài đăng CSV/Excel dữ liệu thô để phân tích chỉ số bất thường tự động hàng loạt:")
        
        up_file = st.file_uploader("Tải lên tệp tin CSV hoặc Excel dữ liệu thô bài đăng:", type=["csv", "xlsx"], key="tab3_batch_uploader")
        
        if up_file is None:
            st.markdown("""
            <div style='background: rgba(15, 23, 42, 0.7); border: 1px dashed rgba(56, 189, 248, 0.4); border-radius: 14px; padding: 28px 20px; text-align: center; margin-top: 15px;'>
                <div style='font-size: 2.2rem; margin-bottom: 6px;'>📁</div>
                <h4 style='color: #38bdf8; font-weight: 700; margin: 0 0 6px 0; font-size: 1.1rem;'>Kéo thả hoặc chọn tệp tin CSV / Excel để bắt đầu</h4>
                <p style='color: #94a3b8; font-size: 0.9rem; margin: 0;'>Tệp tin chứa các cột dữ liệu thô (Thương hiệu, Dòng xe, Năm đăng ký, Số Km, Mô tả chi tiết, Giá rao...).</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            try:
                if up_file.name.endswith('.csv'):
                    df_up = pd.read_csv(up_file)
                else:
                    df_up = pd.read_excel(up_file)
                st.success(f"✅ Đã tiếp nhận {len(df_up)} bản ghi dữ liệu thô từ tệp {up_file.name}!")
            except Exception as e:
                st.error(f"Lỗi đọc tệp: {e}")
                df_up = None

            if df_up is not None:
                st.markdown("#### 📋 Xem Trước Dữ Liệu Tải Lên (5 Bản Ghi Đầu)")
                st.dataframe(df_up.head(5), use_container_width=True)
                
                st.write("")
                btn_run_batch = st.button("⚡ Chạy Phân Tích & Kiểm Duyệt Hàng Loạt", type="primary", use_container_width=True, key="btn_run_batch_audit")
                
                if btn_run_batch:
                    with st.spinner("Hệ thống đang trích xuất đặc trưng & tính toán phân loại chỉ số độ lệch giá hàng loạt..."):
                        results = []
                        for idx, row in df_up.iterrows():
                            # Xử lý giá thô (Hỗ trợ định dạng "85.000.000 đ", "85.0", 85000000)
                            raw_price = row.get('Giá', row.get('gia_rao', 20.0))
                            price_val = parse_price_value(raw_price)
                            if pd.isna(price_val) or price_val <= 0:
                                price_val = 20.0

                            # Xử lý dung tích thô (Hỗ trợ "100 - 175 cc", "Trên 175 cc", 150)
                            raw_cc = row.get('Dung tích xe', row.get('dung_tich_cc', 110))
                            raw_cc_str = str(raw_cc).lower()
                            if 'dưới 50' in raw_cc_str or '< 50' in raw_cc_str:
                                cc_val = 50
                            elif '50 - 100' in raw_cc_str or '50-100' in raw_cc_str:
                                cc_val = 110
                            elif '100 - 175' in raw_cc_str or '100-175' in raw_cc_str:
                                cc_val = 125
                            elif 'trên 175' in raw_cc_str or '> 175' in raw_cc_str:
                                cc_val = 300
                            else:
                                m_cc = re.search(r'(\d+)', raw_cc_str)
                                cc_val = int(m_cc.group(1)) if m_cc else 125

                            u_dict = {
                                'brand': str(row.get('Thương hiệu', row.get('thuong_hieu', 'Honda'))),
                                'model': str(row.get('Dòng xe', row.get('dong_xe', 'Wave'))),
                                'year': int(row.get('Năm đăng ký', row.get('nam_dang_ky', 2020))),
                                'odo': float(row.get('Số Km đã đi', row.get('so_km', 15000))),
                                'bike_type': str(row.get('Loại xe', row.get('loai_xe', 'Tay ga'))),
                                'engine_cc': cc_val,
                                'condition_text': str(row.get('Mô tả chi tiết', row.get('mo_ta', ''))),
                                'price_listed': price_val
                            }
                            res_val = predict_bike_price(u_dict, global_pipe, submodels_dict)
                            results.append({
                                'ID Tin': row.get('id', idx + 1),
                                'Hãng Xe': str(row.get('Thương hiệu', row.get('thuong_hieu', 'Honda'))),
                                'Dòng Xe': str(row.get('Dòng xe', row.get('dong_xe', 'Wave'))),
                                'Giá Rao Niêm Yết (Triệu đ)': price_val,
                                'Giá Thị Trường Gợi Ý (Triệu đ)': res_val['pred_price'],
                                'Chỉ Số Bất Thường (0-100)': res_val['anomaly_score'],
                                'Đánh Giá Phân Loại Rủi Ro': res_val['risk_str']
                            })
                            
                        df_final = pd.DataFrame(results)
                        
                        st.balloons()
                        st.markdown("#### 📊 Tổng Quan Kết Quả Phân Loại Độ Lệch Giá Hàng Loạt")
                        c_b1, c_b2, c_b3, c_b4 = st.columns(4)
                        c_b1.metric("Tổng Bài Đăng Kiểm Duyệt", f"{len(df_final)} bài")
                        
                        n_normal = sum(df_final['Chỉ Số Bất Thường (0-100)'] <= 25)
                        n_warning = sum((df_final['Chỉ Số Bất Thường (0-100)'] > 25) & (df_final['Chỉ Số Bất Thường (0-100)'] <= 50))
                        n_danger = sum(df_final['Chỉ Số Bất Thường (0-100)'] > 50)
                        
                        c_b2.metric("🟢 Mức Giá Hợp Lý", f"{n_normal} bài")
                        c_b3.metric("⚠️ Độ Lệch Trung Bình", f"{n_warning} bài")
                        c_b4.metric("🚨 Độ Lệch Cao/Rủi Ro", f"{n_danger} bài")
                        
                        st.divider()
                        st.markdown("#### 📋 Báo Cáo Phân Loại Bất Thường Rút Gọn (Trực Quan 100%)")
                        
                        # Style CSS cho bảng rộng rãi, chữ to dễ đọc 16px
                        st.markdown("""
                        <style>
                            div[data-testid="stDataFrame"] div[data-baseweb="table"] {
                                font-size: 1.05rem !important;
                                font-weight: 600 !important;
                            }
                            div[data-testid="stDataFrame"] div[role="gridcell"] {
                                padding: 12px 16px !important;
                                font-size: 1.02rem !important;
                            }
                        </style>
                        """, unsafe_allow_html=True)

                        st.dataframe(
                            df_final,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "ID Tin": st.column_config.NumberColumn("ID Tin", format="%d"),
                                "Hãng Xe": st.column_config.TextColumn("Hãng Xe"),
                                "Dòng Xe": st.column_config.TextColumn("Dòng Xe"),
                                "Giá Rao Niêm Yết (Triệu đ)": st.column_config.NumberColumn("💰 Giá Rao (Tr)", format="%.1f Triệu"),
                                "Giá Thị Trường Gợi Ý (Triệu đ)": st.column_config.NumberColumn("🔮 Giá Gợi Ý (Tr)", format="%.1f Triệu"),
                                "Chỉ Số Bất Thường (0-100)": st.column_config.ProgressColumn(
                                    "📊 Chỉ Số Bất Thường (0-100)",
                                    format="%.1f",
                                    min_value=0,
                                    max_value=100
                                ),
                                "Đánh Giá Phân Loại Rủi Ro": st.column_config.TextColumn("🚨 Phân Loại Rủi Ro")
                            }
                        )
                        
                        csv_data = df_final.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải Báo Cáo Kết Quả Kiểm Duyệt CSV",
                            data=csv_data,
                            file_name="Bao_Cao_Kiem_Duyet_Xe_May_Hang_Loat.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="btn_download_csv_report"
                        )

if __name__ == "__main__":
    main()
