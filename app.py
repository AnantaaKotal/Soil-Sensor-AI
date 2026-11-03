"""Streamlit analysis service for GitHub-hosted Soil AI-Ready Data site."""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from engine import EXPECTED, load_model, read_csv, validate_csv, analyze

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='SoilSense | Sensor Data Workbench',page_icon='🌱',layout='wide')
st.markdown('''<style>
.block-container{max-width:1180px;padding-top:2.5rem}.hero{padding:1.5rem 1.75rem;border-radius:18px;background:linear-gradient(110deg,#123b34,#285b48);color:#fff;margin-bottom:1.2rem}.hero h1{color:white;margin:0}.hero p{color:#daece3;margin-bottom:0}.stButton>button[kind=primary]{background:#216a51;border:0}
</style>''',unsafe_allow_html=True)
st.markdown('<section class="hero"><h1>🌱 SoilSense · Sensor Data Workbench</h1><p>Explore quality-control candidates and reconstruct missing measurements in multivariate environmental records.</p></section>',unsafe_allow_html=True)
st.warning('Research prototype: trained on one orchard dataset. Alerts are not verified faults. Imputation uses masked Transformer reconstruction, NOT a diffusion model. Check sensor units, calibration and environmental context before using results.',icon='⚠️')

@st.cache_resource(show_spinner='Loading trained sensor model…')
def get_model():
    return load_model(ROOT/'models'/'soil_transformer.pt')

with st.sidebar:
    st.header('Analysis setup')
    operation=st.radio('Choose a tool',['Detect sensor anomalies','Fill missing sensor values'], help='Both tools use the trained cross-sensor masked Transformer.')
    if operation.startswith('Detect'):
        threshold_mode=st.radio('Alert threshold',['Dataset-relative (recommended for exploration)','Original orchard calibration'], help='Dataset-relative thresholds are per sensor. They control the fraction of high residuals selected on the uploaded dataset, not the false-positive rate. Orchard thresholds may have severe distribution shift.')
        if threshold_mode.startswith('Dataset'):
            sensitivity=st.slider('High-residual percentile', 90, 99, 98,help='98 = marks approximately the top 2% of scorable sensor readings for each channel. Not a calibrated confidence estimate.')
        else:
            sensitivity=95
    st.divider()
    st.caption('Expected cadence: 5 minutes, with at least nine successive rows for predictions. The model expects these exact sensor names and compatible units:')
    st.code(', '.join(EXPECTED),language=None)

file=st.file_uploader('Upload a CSV containing a timestamp and the model sensor columns',type=['csv'])
left,right=st.columns([1.3,1])
with left:
    st.markdown('**New here?** Start with the example CSV to check the layout.')
with right:
    st.download_button('Download example CSV',data=(ROOT/'examples'/'sample_sensor_data.csv').read_bytes(),file_name='sample_sensor_data.csv',mime='text/csv',use_container_width=True)
if file is None:
    st.subheader('Two tools, one transparent workflow')
    c1,c2=st.columns(2)
    with c1:
        st.markdown('#### 🔎 Fault screening')
        st.write('Compare each sensor measurement against a prediction reconstructed from other sensors and neighboring timestamps. Download candidate alerts and residual scores for expert review.')
    with c2:
        st.markdown('#### 🧩 Missing-data reconstruction')
        st.write('Fill missing values using cross-sensor context where enough observations exist. Original measurements remain unchanged; reconstructed cells are explicitly flagged.')
    st.info('For genuine diffusion-model imputation, a separate trained diffusion checkpoint and inference code are still needed.')
    st.stop()

try:
    if file.size>15*1024*1024:
        st.error('CSV exceeds the 15 MB size limit for this public prototype.');st.stop()
    data=read_csv(file)
    probable=[x for x in data.columns if 'time' in x.lower() or 'date' in x.lower()]
    time_col=st.selectbox('Timestamp column',data.columns.tolist(), index=(data.columns.tolist().index(probable[0]) if probable else 0))
    clean,dt=validate_csv(data,time_col)
except Exception as e:
    st.error(f'Invalid CSV: {e}'); st.stop()

nrows=len(clean)
missing=int(clean[EXPECTED].isna().sum().sum())
st.caption(f'{nrows:,} rows · {len(EXPECTED)} sensors · {missing:,} missing sensor values · UTC-normalized timestamps')
st.dataframe(clean.head(6),use_container_width=True,hide_index=True)

if st.button('Run analysis', type='primary'):
    with st.spinner('Reconstructing sensor expectations…'):
        try:
            model,median,scale=get_model()
            progress_bar=st.progress(0)
            progress=lambda f: progress_bar.progress(min(100,max(0,int(f*100))))
            if operation.startswith('Detect'):
                thresholds=None
                if threshold_mode.startswith('Original'):
                    thresholds=json.loads((ROOT/'models'/'soil_fault_metrics.json').read_text())['thresholds']['Masked Transformer']
                result,stats=analyze(clean,dt,model,median,scale,mode='anomaly',quantile=sensitivity/100,reference_thresholds=thresholds,progress=progress)
            else:
                result,stats=analyze(clean,dt,model,median,scale,mode='impute',progress=progress)
            st.session_state['result']=result
            st.session_state['stats']=stats
            st.session_state['input_file_name']=file.name
            st.session_state['operation']=operation
            st.session_state['time_col']=time_col
        except Exception as e:
            st.error(f'Analysis failed: {e}');st.stop()

if 'result' in st.session_state and st.session_state.get('input_file_name')==file.name and st.session_state.get('operation')==operation:
    result=st.session_state['result'];stats=st.session_state['stats']
    st.success('Completed analysis on the uploaded CSV.')
    if operation.startswith('Detect'):
        c1,c2,c3=st.columns(3)
        c1.metric('Candidate sensor-cell alerts',f"{stats['alert_cells']:,}")
        c2.metric('Timestamps with alerts',f"{stats['alert_timestamps']:,}")
        c3.metric('Analyzable 5-minute windows',f"{stats['valid_centers']:,}")
        st.caption('These are candidates for manual review, not confirmed faults. Cross-sensor co-movement or changed conditions can also trigger alerts.')
    else:
        a,b,c=st.columns(3)
        a.metric('Imputed sensor values',f"{stats['imputed_cells']:,}")
        b.metric('Remaining missing values',f"{stats['unfilled_cells']:,}")
        c.metric('Analyzable 5-minute windows',f"{stats['valid_centers']:,}")
        st.caption('No extrapolation over timestamp gaps or outside nine-point windows. Predictions require at least three other present sensors at the target time.')
    sensor=st.selectbox('Inspect sensor', EXPECTED)
    plot=pd.DataFrame({'Timestamp':dt,'Observed':clean[sensor]})
    if operation.startswith('Detect'):
        plot['Expected']=result[f'{sensor}__expected']
        st.line_chart(plot.set_index('Timestamp')[['Observed','Expected']],use_container_width=True)
        st.line_chart(pd.DataFrame({'Timestamp':dt,'Reconstruction mismatch':result[f'{sensor}__score']}).set_index('Timestamp'),use_container_width=True)
        selected=result.loc[result[f'{sensor}__candidate_fault'],[time_col,sensor,f'{sensor}__score']]
        st.markdown(f'**{len(selected):,} candidate alerts** for `{sensor}`')
        st.dataframe(selected.head(250),use_container_width=True,hide_index=True)
    else:
        plot['Reconstructed']=result[sensor]
        st.line_chart(plot.set_index('Timestamp')[['Observed','Reconstructed']],use_container_width=True)
        st.caption(f"{int(result[f'{sensor}__was_imputed'].sum())} imputed values in {sensor}.")
    st.download_button('Download analysis CSV',result.to_csv(index=False).encode('utf-8'),file_name='soil_sense_'+('anomaly_flags' if operation.startswith('Detect') else 'imputed')+'.csv',mime='text/csv',type='primary')
    st.dataframe(result.head(100),use_container_width=True,hide_index=True)

st.divider()
st.caption('Research prototype · On-device training not provided · For data reuse, review uncertainty, sensor provenance, unit compatibility, and field conditions. Uploaded data are processed by the Python app server; do not upload sensitive datasets without reviewing its hosting policy.')
