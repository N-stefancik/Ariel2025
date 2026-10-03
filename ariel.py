import pandas as pd
import numpy as np
import pandas.api.types
import scipy.stats

from tqdm import tqdm
from pqdm.threads import pqdm
import itertools
import pickle

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

import plotly.express as px

from astropy.stats import sigma_clip
from scipy.signal import savgol_filter

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get('ARIEL_DATA_PATH', str(PROJECT_ROOT / 'data' / 'ariel-data-challenge-2025')))
PREPROCESSED_PATH = Path(os.environ.get('ARIEL_PREPROCESSED_PATH', str(PROJECT_ROOT / 'data' / 'preprocessed_data.npy')))

class Config:
    DATA_PATH = str(DATA_PATH)
    DATASET = "test"

    SCALE = 0.927
    SIGMA = 0.0009
    
    CUT_INF = 39
    CUT_SUP = 250
    
    SENSOR_CONFIG = {
        "AIRS-CH0": {
            "raw_shape": [11250, 32, 356],
            "calibrated_shape": [1, 32, CUT_SUP - CUT_INF],
            "linear_corr_shape": (6, 32, 356),
            "dt_pattern": (0.1, 4.5), 
            "binning": 30
        },
        "FGS1": {
            "raw_shape": [135000, 32, 32],
            "calibrated_shape": [1, 32, 32],
            "linear_corr_shape": (6, 32, 32),
            "dt_pattern": (0.1, 0.1),
            "binning": 30 * 12
        }
    }
    
    MODEL_PHASE_DETECTION_SLICE = slice(0, 186)
    MODEL_OPTIMIZATION_DELTA = 7
    MODEL_POLYNOMIAL_DEGREE = 3
    
    N_JOBS = 4

class SignalProcessor:
    def __init__(self, config):
        self.cfg = config
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.planet_ids = pd.read_csv(f'{self.cfg.DATA_PATH}/{self.cfg.DATASET}_star_info.csv', index_col='planet_id').index.astype(int)

    def _apply_linear_corr(self, linear_corr, signal):
        linear_corr_flipped = np.flip(linear_corr, axis=0)
        corrected_signal = signal.copy()
        
        for x, y in itertools.product(range(signal.shape[1]), range(signal.shape[2])):
            poly = np.poly1d(linear_corr_flipped[:, x, y])
            corrected_signal[:, x, y] = poly(corrected_signal[:, x, y])
            
        return corrected_signal

    def _calibrate_single_signal(self, planet_id, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]

        signal = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_signal_0.parquet").to_numpy()
        dark = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dark.parquet").to_numpy()
        dead = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dead.parquet").to_numpy()
        flat = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/flat.parquet").to_numpy()
        linear_corr = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/linear_corr.parquet").values.astype(np.float64).reshape(sensor_cfg["linear_corr_shape"])

        signal = signal.reshape(sensor_cfg["raw_shape"])
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset

        hot = sigma_clip(dark, sigma=5, maxiters=5).mask

        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            hot = hot[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
        
        base_dt, increment = sensor_cfg["dt_pattern"]
        dt = np.ones(len(signal)) * base_dt
        dt[1::2] += increment
        
        signal = signal.clip(0)
        signal = self._apply_linear_corr(linear_corr, signal)
        signal -= dark * dt[:, np.newaxis, np.newaxis]
        
        flat = flat.reshape(sensor_cfg["calibrated_shape"])
        flat[dead.reshape(sensor_cfg["calibrated_shape"])] = np.nan
        flat[hot.reshape(sensor_cfg["calibrated_shape"])] = np.nan
        
        signal = signal / flat
        
        return signal

    def _preprocess_calibrated_signal(self, calibrated_signal, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]
        binning = sensor_cfg["binning"]

        if sensor == "AIRS-CH0":
            signal_roi = calibrated_signal[:, 10:22, :]
        elif sensor == "FGS1":
            signal_roi = calibrated_signal[:, 10:22, 10:22]
            signal_roi = signal_roi.reshape(signal_roi.shape[0], -1)
        
        mean_signal = np.nanmean(signal_roi, axis=1)

        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        n_bins = cds_signal.shape[0] // binning
        binned = np.array([
            cds_signal[j*binning : (j+1)*binning].mean(axis=0) 
            for j in range(n_bins)
        ])

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))
            
        return binned

    def _process_planet_sensor(self, args):
        planet_id, sensor = args['planet_id'], args['sensor']
        calibrated = self._calibrate_single_signal(planet_id, sensor)
        preprocessed = self._preprocess_calibrated_signal(calibrated, sensor)
        return preprocessed

    def process_all_data(self):
        args_fgs1 = [dict(planet_id=planet_id, sensor="FGS1") for planet_id in self.planet_ids]
        preprocessed_fgs1 = pqdm(args_fgs1, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        args_airs_ch0 = [dict(planet_id=planet_id, sensor="AIRS-CH0") for planet_id in self.planet_ids]
        preprocessed_airs_ch0 = pqdm(args_airs_ch0, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        preprocessed_signal = np.concatenate(
            [np.stack(preprocessed_fgs1), np.stack(preprocessed_airs_ch0)], axis=2
        )
        return preprocessed_signal

class TransitModel:
    def __init__(self, config):
        self.cfg = config

    def _phase_detector(self, signal):
        search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = np.argmin(signal[search_slice]) + search_slice.start
    
        signal1 = signal[:min_index]
        signal2 = signal[min_index:]
    
        grad1 = np.abs(np.gradient(signal1))
        phase1 = np.argmax(grad1)
    
        grad2 = np.abs(np.gradient(signal2))
        phase2 = np.argmax(grad2) + min_index
    
        return phase1, phase2
    
    def _objective_function(self, s, signal, phase1, phase2, middle):
        delta = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE
        off=7

        middle_point = (np.floor((phase2-phase1)/2)+phase1).astype(int)
        half_size = (np.floor(middle*(phase2-phase1)/2)).astype(int)
        #print(phase1, middle_point-half_size, middle_point, middle_point+half_size, phase2)


        begin = phase1-off
        end = phase2+off
        '''
        # Handle edge cases
        if begin < 1:
            begin = 1
            begin_pad = np.repeat(signal[0], 20)
        else:
            begin_pad = np.array([])
        
        if end > 186:
            end = 186
            end_pad = np.repeat(signal[186], 20)
        else:
            end_pad = np.array([])
        
        # Construct y
        y = np.concatenate([
            begin_pad,
            signal[:begin],
            signal[middle_point - half_size : middle_point + half_size] * (1 + s),
            signal[end:],
            end_pad
        ])
        '''
        
      
        if begin<1:
            begin=1
        if end>186:
            end=186
        # Construct y
        y = np.concatenate([
            signal[:begin],
            signal[middle_point - half_size : middle_point + half_size] * (1 + s),
            signal[end:]
        ])

        
        x = np.arange(len(y))

        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        error = np.abs(poly(x) - y).mean()
        
        return error

    def predict(self, single_preprocessed_signal, middle):
        # Step 1: Reduce to 1D signal
        signal_1d = single_preprocessed_signal[:, :].mean(axis=1)
    
        # Step 2: Denoise for detection and fitting
        signal_1d_detection = savgol_filter(signal_1d, 31, 2)
        signal_1d = savgol_filter(signal_1d, 11, 3)
    
        # Step 3: Detect phases
        phase1, phase2 = self._phase_detector(signal_1d_detection)
        
        
        # Step 4: Check for edge/cut-off cases
        if phase1 < 15 or phase2 > 174:
            # Fallback to min-max depth estimation
            depth_estimate = (signal_1d.max() - signal_1d.min())/(signal_1d.max())
            return depth_estimate
        
        # Step 5: Clamp for optimization bounds
        phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
        phase2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)
    
        # Step 6: Optimize
        result = minimize(
            fun=self._objective_function,
            x0=[0.0001],
            args=(signal_1d, phase1, phase2, middle),
            method="Nelder-Mead"
        )
       
        
        return result.x[0]


    def predict_all(self, preprocessed_signals, middle):
        predictions = [
            self.predict(preprocessed_signal, middle)
            for preprocessed_signal in tqdm(preprocessed_signals)
        ]
        return np.array(predictions) * self.cfg.SCALE

class SubmissionGenerator:
    def __init__(self, config):
        self.cfg = config
        self.sample_submission = pd.read_csv(f"{self.cfg.DATA_PATH}/sample_submission.csv", index_col="planet_id")
        

    def create(self, predictions):
        planet_ids = self.sample_submission.index
        repeated_predictions = np.repeat(predictions, self.sample_submission.shape[1] // 2).reshape(len(predictions), -1)
        repeated_predictions = repeated_predictions.clip(0)
        
        sigmas = np.ones_like(repeated_predictions) * self.cfg.SIGMA

        submission_df = pd.DataFrame(
            np.concatenate([repeated_predictions, sigmas], axis=1),
            columns=self.sample_submission.columns,
            index=planet_ids
        )
        
        # The runner chooses the destination.
        return submission_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict Ariel transit depths.')
    parser.add_argument('--data-path', type=Path, default=DATA_PATH)
    parser.add_argument('--preprocessed', type=Path)
    parser.add_argument('--dataset', choices=('train', 'test'), default='test')
    parser.add_argument('--middle', type=float, default=0.45)
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
    parser.add_argument('--submission', action='store_true')
    args = parser.parse_args()
    config = Config()
    config.DATA_PATH = str(args.data_path)
    config.DATASET = args.dataset
    signals = np.load(args.preprocessed) if args.preprocessed else SignalProcessor(config).process_all_data()
    if signals.ndim != 3 or signals.shape[0] == 0 or signals.shape[1] < 31 or signals.shape[2] == 0 or not np.isfinite(signals).all():
        parser.error('Expected finite signals shaped (planets, time steps >= 31, channels).')
    predictions = TransitModel(config).predict_all(signals, args.middle)
    if args.submission:
        generator = SubmissionGenerator(config)
        if len(predictions) != len(generator.sample_submission):
            parser.error('Prediction count must match sample_submission.csv; planet order must also match.')
        frame = generator.create(predictions)
    else:
        frame = pd.DataFrame({'array_index': np.arange(len(predictions)), 'transit_depth': predictions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=args.submission)
    print(f'Saved {len(predictions)} predictions to {args.output}')
