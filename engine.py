"""Experimental soil sensor screening and context-based gap reconstruction.

The model was trained on one orchard dataset. Outputs are NOT validated fault
labels, and inference on new locations/instruments may be out-of-distribution.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
EXPECTED = ['CO2_30','CO2_60','O2_30','O2_60','VWC_30','VWC_60','SoilT_30','SoilT_60','Air_P','ESFD_Flux']
HALF = 4
LENGTH = 2*HALF+1

class MaskedTimeTransformer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj=nn.Linear(d*2,48)
        self.pos=nn.Parameter(torch.zeros(1,LENGTH,48))
        layer=nn.TransformerEncoderLayer(d_model=48,nhead=4,dim_feedforward=96,dropout=.08,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(48), nn.Linear(48,48), nn.GELU(), nn.Linear(48,d))
    def forward(self, a, mask):
        h=self.proj(torch.cat([a*(1-mask),mask],dim=-1))+self.pos
        return self.head(self.enc(h)[:,HALF,:])

def load_model(path=None):
    path=Path(path) if path else ROOT/'models'/'soil_transformer.pt'
    checkpoint=torch.load(path, map_location='cpu', weights_only=True)
    names=list(checkpoint['features'])
    if names != EXPECTED:
        raise ValueError(f'Unexpected features in model checkpoint: {names}')
    model=MaskedTimeTransformer(len(names))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model,np.asarray(checkpoint['median'],dtype=np.float32),np.asarray(checkpoint['scale'],dtype=np.float32)

def read_csv(upload):
    df=pd.read_csv(upload)
    df.columns=[str(s).strip() for s in df.columns]
    return df

def validate_csv(df, time_column):
    missing=[v for v in EXPECTED if v not in df.columns]
    if missing:
        raise ValueError('Missing required sensor columns: '+', '.join(missing))
    if not time_column or time_column not in df:
        raise ValueError('Select an existing timestamp column.')
    if df.empty or len(df)<LENGTH:
        raise ValueError('The file needs at least 9 rows.')
    out=df.copy()
    dt=pd.to_datetime(out[time_column],errors='coerce',utc=True)
    if dt.isna().any():
        raise ValueError('Some timestamps could not be parsed. Use ISO-8601 timestamps.')
    if dt.duplicated().any():
        raise ValueError('Timestamps must be unique. Aggregate duplicate rows first.')
    if not dt.is_monotonic_increasing:
        out=out.iloc[np.argsort(dt.to_numpy())].reset_index(drop=True)
        dt=dt.sort_values().reset_index(drop=True)
    for feature in EXPECTED:
        out[feature]=pd.to_numeric(out[feature],errors='coerce').replace([np.inf,-np.inf],np.nan)
    return out,dt

def infer(model,df,dt,median,scale,progress=None,batch_size=128):
    """Mask entire target channel to avoid leakage, and mask other missing input cells.

    Only infer for valid 9-row, 5-min consecutive windows with >=3 other
    channels observed at the middle timestamp. Returns raw predictions and
    standardized residuals; NaNs indicate unsupported positions.
    """
    values=df[EXPECTED].to_numpy(dtype=np.float32)
    obs=np.isfinite(values)
    normalized=np.clip((np.where(obs,values,median)-median)/scale,-12,12).astype(np.float32)
    normalized[~obs]=0.0
    t=(dt.astype('int64').to_numpy() / 1e9)
    step=np.diff(t)
    valid=[]
    for center in range(HALF,len(df)-HALF):
        if np.all(np.abs(step[center-HALF:center+HALF]-300)<1):
            valid.append(center)
    prediction=np.full(values.shape,np.nan,dtype=np.float32)
    if not valid:
        return prediction,np.full_like(prediction,np.nan),np.zeros(len(df),dtype=bool)
    centers=np.asarray(valid,dtype=np.int32)
    with torch.inference_mode():
        for start in range(0,len(centers),batch_size):
            chunk=centers[start:start+batch_size]
            w=np.stack([normalized[chunk+offset] for offset in range(-HALF,HALF+1)],axis=1)
            known=np.stack([obs[chunk+offset] for offset in range(-HALF,HALF+1)],axis=1)
            # Separate masked prediction for each target channel, all sensors.
            a=np.repeat(w, len(EXPECTED),axis=0)
            m=np.repeat((~known).astype(np.float32),len(EXPECTED),axis=0)
            target=np.tile(np.arange(len(EXPECTED)),len(chunk))
            m[np.arange(len(a)),:,target]=1.0
            y=model(torch.from_numpy(a),torch.from_numpy(m)).numpy()[np.arange(len(a)),target]
            y=y.reshape(len(chunk),len(EXPECTED))*scale+median
            # Need >=3 other observed sensors at center; don't use impossible extrapolations.
            num_other=obs[chunk].sum(axis=1,keepdims=True)-obs[chunk].astype(np.int32)
            y[num_other<3]=np.nan
            prediction[chunk]=y
            if progress:
                progress(min(1.0,(start+len(chunk))/len(centers)))
    residual=np.abs((values-prediction)/scale)
    residual[~obs]=np.nan
    valid_center=np.zeros(len(df),dtype=bool)
    valid_center[centers]=True
    return prediction,residual,valid_center

def analyze(df,dt,model,median,scale,mode='anomaly',quantile=.99,reference_thresholds=None,progress=None):
    preds,scores,valid_centers=infer(model,df,dt,median,scale,progress=progress)
    result=df.copy()
    if mode=='anomaly':
        use_reference=reference_thresholds is not None
        for j,name in enumerate(EXPECTED):
            s=scores[:,j]
            if use_reference:
                limit=float(reference_thresholds[name])
            else:
                observed=s[np.isfinite(s)]
                limit=float(np.quantile(observed,quantile)) if len(observed)>=20 else np.nan
            result[f'{name}__expected']=preds[:,j]
            result[f'{name}__score']=s
            flags=(s>limit) if np.isfinite(limit) else np.zeros_like(s,dtype=bool)
            result[f'{name}__candidate_fault']=flags
            result[f'{name}__label']=np.where(~np.isfinite(s),'not evaluated',np.where(flags,'anomaly','regular'))
        alerts=result[[f'{v}__candidate_fault' for v in EXPECTED]].sum(axis=1)
        result['candidate_fault_count']=alerts
        evaluated=result[[f'{v}__label' for v in EXPECTED]].ne('not evaluated').any(axis=1)
        result['anomaly_label']=np.where(alerts>0,'anomaly',np.where(evaluated,'regular','not evaluated'))
        return result,{'valid_centers':int(valid_centers.sum()),'alert_cells':int(alerts.sum()),'alert_timestamps':int((alerts>0).sum()), 'mode':'reference thresholds' if use_reference else 'dataset-relative scores'}
    if mode=='impute':
        n=0
        for j,name in enumerate(EXPECTED):
            missing=result[name].isna().to_numpy()
            fill=missing & np.isfinite(preds[:,j])
            result[f'{name}__was_imputed']=fill
            result.loc[fill,name]=preds[fill,j]
            n+=int(fill.sum())
        return result,{'valid_centers':int(valid_centers.sum()),'imputed_cells':n,'unfilled_cells':int(result[EXPECTED].isna().sum().sum())}
    raise ValueError('Unsupported mode')
