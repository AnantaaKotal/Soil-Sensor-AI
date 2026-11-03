import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from engine import EXPECTED, load_model, validate_csv, analyze
ROOT=Path(__file__).resolve().parents[1]
class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.median,cls.scale=load_model()
        cls.df=pd.read_csv(ROOT/'examples'/'sample_sensor_data.csv')
        cls.clean,cls.dt=validate_csv(cls.df,'timestamp')
    def test_model_matches_schema(self):
        self.assertEqual(self.clean[EXPECTED].shape[1],10)
    def test_anomaly_outputs(self):
        o,meta=analyze(self.clean,self.dt,self.model,self.median,self.scale,mode='anomaly')
        self.assertTrue(meta['valid_centers']>0)
        self.assertIn('CO2_30__score',o.columns)
        self.assertTrue(np.isfinite(o['CO2_30__score']).sum()>0)
    def test_imputation_only_missing(self):
        o,meta=analyze(self.clean,self.dt,self.model,self.median,self.scale,mode='impute')
        self.assertGreater(meta['imputed_cells'],0)
        orig=self.clean['CO2_30'].notna()
        self.assertTrue(np.allclose(self.clean.loc[orig,'CO2_30'],o.loc[orig,'CO2_30']))
    def test_invalid_missing_channels(self):
        with self.assertRaises(ValueError): validate_csv(self.df.drop(columns=['O2_30']),'timestamp')
if __name__=='__main__':unittest.main()
