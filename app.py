"""SoilSense-Ai — minimal dashboard for Transformer sensor screening/imputation."""
from pathlib import Path
import hashlib
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from engine import EXPECTED, load_model, read_csv, validate_csv, analyze

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='SoilSense-Ai | Sensor dashboard', page_icon='🌱', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''
<style>
:root{color-scheme:light}
.stApp{background:#f7faf7;color:#19382e}
.block-container{max-width:1160px;padding-top:1.3rem;padding-bottom:2rem}
[data-testid="stSidebar"], [data-testid="stDecoration"]{display:none}
h1,h2,h3{letter-spacing:-.035em;color:#153f32!important}
h1{font-size:2.15rem!important;margin-bottom:.15rem!important}
[data-testid="stMetric"]{background:#fff;border:1px solid #d8e7df;border-radius:14px;padding:17px 20px;box-shadow:0 2px 8px #1a493208}
[data-testid="stMetricValue"]{color:#174d3b;font-weight:780}
.stButton button[kind="primary"],.stDownloadButton button[kind="primary"]{background:#174d3b!important;border-color:#174d3b!important;border-radius:10px!important;color:#fff!important;font-weight:650!important}
/* Prominent upload panel */
[data-testid="stFileUploader"]{background:#ffffff;border:2px solid #a8cabb;border-radius:18px;padding:12px 14px;box-shadow:0 4px 14px #123f3010}
[data-testid="stFileUploaderDropzone"]{min-height:145px;background:#f0f8f3;border:2px dashed #488970;border-radius:14px;padding:24px 18px}
[data-testid="stFileUploaderDropzone"] button{background:#174d3b!important;color:white!important;border:0!important;font-size:1rem!important;font-weight:750!important;padding:0.65rem 1.15rem!important}
[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzone"] span{font-size:1rem!important}
.upload-heading{font-size:1.6rem;font-weight:850;letter-spacing:-.025em;color:#153f32;margin:0 0 .75rem}

[data-testid="stAlert"]{border-radius:10px}
hr{border-color:#dce8e0}
.smallsub{font-size:1rem;color:#658176;margin-bottom:1.25rem}
</style>
''', unsafe_allow_html=True)

st.title('🌱 SoilSense-Ai')
st.markdown('<div class="smallsub">Upload sensor data. Run an analysis. Explore and download your results.</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model(ROOT/'models'/'soil_transformer.pt')

# Avoid stale results when an uploaded file is replaced with a new file using the same name.
if 'latest' not in st.session_state:
    st.session_state.latest = None

left, right = st.columns([1.45, 1], gap='large')
with left:
    st.markdown('<div class="upload-heading">1 · Upload dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader('CSV file', type='csv', label_visibility='collapsed')
with right:
    st.markdown('**2 · Select analysis**')
    operation = st.selectbox('Analysis', ('Sensor anomaly detection', 'Missing data reconstruction'), label_visibility='collapsed')

with st.expander('Data requirements and options', expanded=False):
    st.caption('Required: a timestamp column, 5-minute spacing, and the ten sensor channels below in compatible units. At least 9 consecutive rows are needed.')
    st.code(', '.join(EXPECTED),language=None)
    st.download_button('Download example CSV', (ROOT/'examples'/'sample_sensor_data.csv').read_bytes(),file_name='soilsense_example.csv',mime='text/csv')
    if operation == 'Sensor anomaly detection':
        threshold_mode=st.selectbox('Anomaly threshold', ['Dataset-relative','Original orchard calibration'], help='Dataset-relative finds the highest mismatch scores in the uploaded file. Original calibration can generate many false alerts on other datasets.')
        percentile=st.slider('Dataset-relative cutoff percentile',90,99,98) if threshold_mode=='Dataset-relative' else 95
    else:
        threshold_mode='Dataset-relative'
        percentile=98

if uploaded is None:
    st.info('Upload a CSV to get started. The example CSV is available under Data requirements and options.')
    st.stop()

try:
    content=uploaded.getvalue()
    if len(content)>15*1024*1024:
        st.error('File exceeds the 15 MB upload limit for this prototype.')
        st.stop()
    from io import BytesIO
    source=read_csv(BytesIO(content))
    probable=[i for i,c in enumerate(source.columns) if 'time' in c.lower() or 'date' in c.lower()]
    time_col=st.selectbox('Timestamp column', source.columns.tolist(), index=probable[0] if probable else 0)
    data,dates=validate_csv(source,time_col)
except Exception as exc:
    st.error(f'Cannot process this CSV: {exc}')
    st.stop()

missing=int(data[EXPECTED].isna().sum().sum())
st.caption(f'**{len(data):,}** rows · **{len(EXPECTED)}** sensors · **{missing:,}** missing values')
file_key=hashlib.sha256(content).hexdigest()
request_key=(file_key,time_col,operation,threshold_mode,percentile)

if st.button('Run analysis',type='primary',use_container_width=True):
    with st.spinner('Running the model on your sensor data…'):
        try:
            model,median,scale=get_model()
            if operation == 'Sensor anomaly detection':
                reference=None
