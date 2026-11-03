"""SoilSense-Ai — minimal dashboard for masked Transformer sensor screening/imputation."""
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
                if threshold_mode=='Original orchard calibration':
                    reference=json.loads((ROOT/'models'/'soil_fault_metrics.json').read_text())['thresholds']['Masked Transformer']
                result,stats=analyze(data,dates,model,median,scale,mode='anomaly',quantile=percentile/100,reference_thresholds=reference)
            else:
                result,stats=analyze(data,dates,model,median,scale,mode='impute')
            st.session_state.latest={'key':request_key,'result':result,'stats':stats}
        except Exception as exc:
            st.error(f'Analysis failed: {exc}')
            st.stop()

saved=st.session_state.latest
if saved is None or saved['key']!=request_key:
    st.caption('Click **Run analysis** to display the dashboard.')
    st.stop()

result,stats=saved['result'],saved['stats']
st.divider()
st.subheader('Results dashboard')

if operation == 'Sensor anomaly detection':
    m1,m2,m3=st.columns(3,gap='medium')
    m1.metric('Anomalous sensor readings',f"{stats['alert_cells']:,}")
    m2.metric('Timestamps with anomalies',f"{stats['alert_timestamps']:,}")
    m3.metric('Analyzed time windows',f"{stats['valid_centers']:,}")
    st.caption('Anomaly labels are screening candidates, not verified instrument faults. Non-evaluable measurements are marked “not evaluated.”')
    sensor=st.selectbox('Sensor to inspect',EXPECTED)
    flag=result[f'{sensor}__candidate_fault'].fillna(False).to_numpy(dtype=bool)
    chart_df=pd.DataFrame({'Time':dates,'Observed':data[sensor].to_numpy(),'Expected':result[f'{sensor}__expected'].to_numpy(),'Anomaly':flag})
    traces=chart_df.melt(id_vars='Time',value_vars=['Observed','Expected'],var_name='Series',value_name='Value').dropna()
    base=alt.Chart(traces).mark_line(strokeWidth=2).encode(
        x=alt.X('Time:T',title='Time'),y=alt.Y('Value:Q',title=sensor,scale=alt.Scale(zero=False)),
        color=alt.Color('Series:N',scale=alt.Scale(domain=['Observed','Expected'],range=['#166b54','#9abbb0'])),tooltip=['Time:T','Series:N',alt.Tooltip('Value:Q',format='.3f')])
    points=alt.Chart(chart_df.loc[chart_df['Anomaly']]).mark_circle(size=75,color='#db5c45',opacity=.92).encode(x='Time:T',y='Observed:Q',tooltip=['Time:T',alt.Tooltip('Observed:Q',format='.3f')])
    st.altair_chart((base+points).properties(height=340).interactive(),use_container_width=True)
    st.caption(f'**{int(flag.sum()):,}** candidate anomalous readings in {sensor}; highlighted in coral.')
    with st.expander('Inspect flagged readings'):
        flagged_rows=result[result['anomaly_label']=='anomaly']
        st.dataframe(flagged_rows[[time_col,'anomaly_label','candidate_fault_count']].head(500),hide_index=True,use_container_width=True)
    download_name='soilsense_anomaly_labels.csv'
    st.markdown('**Download labeled dataset**')
    st.caption('Includes `anomaly_label` (anomaly / regular / not evaluated), plus per-sensor labels and reconstruction scores.')
else:
    m1,m2,m3=st.columns(3,gap='medium')
    m1.metric('Reconstructed values',f"{stats['imputed_cells']:,}")
    m2.metric('Missing values remaining',f"{stats['unfilled_cells']:,}")
    m3.metric('Analyzed time windows',f"{stats['valid_centers']:,}")
    sensor=st.selectbox('Sensor to inspect',EXPECTED)
    reconstructed=result[sensor].to_numpy()
    observed=data[sensor].to_numpy()
    filled=result[f'{sensor}__was_imputed'].to_numpy(dtype=bool)
    graph=pd.DataFrame({'Time':dates,'Original':observed,'Reconstructed series':reconstructed,'Filled':filled})
    traces=graph.melt(id_vars='Time',value_vars=['Original','Reconstructed series'],var_name='Series',value_name='Value').dropna()
    base=alt.Chart(traces).mark_line(strokeWidth=2).encode(
        x=alt.X('Time:T',title='Time'),y=alt.Y('Value:Q',title=sensor,scale=alt.Scale(zero=False)),
        color=alt.Color('Series:N',scale=alt.Scale(domain=['Original','Reconstructed series'],range=['#a5b6b0','#137757'])),tooltip=['Time:T','Series:N',alt.Tooltip('Value:Q',format='.3f')])
    points=alt.Chart(graph.loc[graph['Filled']]).mark_circle(size=70,color='#e4a548').encode(x='Time:T',y=alt.Y('Reconstructed series:Q'),tooltip=['Time:T',alt.Tooltip('Reconstructed series:Q',format='.3f')])
    st.altair_chart((base+points).properties(height=340).interactive(),use_container_width=True)
    st.caption(f'**{int(filled.sum()):,}** values reconstructed for {sensor}, shown in amber. Existing observations are unchanged.')
    download_name='soilsense_reconstructed.csv'
    st.markdown('**Download reconstructed dataset**')
    st.caption('Includes original columns and `__was_imputed` flags indicating the cells reconstructed by the model.')

st.download_button('⬇ Download result CSV',result.to_csv(index=False).encode('utf-8'),file_name=download_name,mime='text/csv',type='primary',use_container_width=True)
with st.expander('Preview output data'):
    st.dataframe(result.head(100),hide_index=True,use_container_width=True)
st.caption('Research prototype trained on one orchard dataset. Expected units and sensor columns are required. Imputation uses the masked Transformer, not a diffusion model. Data are processed on the Streamlit server; avoid sensitive uploads.')
