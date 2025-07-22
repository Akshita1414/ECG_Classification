import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from copy import deepcopy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import wfdb
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis, ttest_rel
import warnings
warnings.filterwarnings('ignore')

# XGBoost import with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using Random Forest as fallback")

# Optional GNNExplainer import - graceful fallback if not available
try:
    from torch_geometric.explain import Explainer, GNNExplainer
    GNNEXPLAINER_AVAILABLE = True
    print("✅ GNNExplainer available for enhanced explainability - FAST MODE")
except ImportError:
    GNNEXPLAINER_AVAILABLE = False
    print("⚠️ GNNExplainer not available, using attention-based explainability only")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class MITBIHDataLoader:
    """Load and preprocess real MIT-BIH Arrhythmia dataset"""

    def __init__(self, data_dir='./mitbih_data'):
        self.data_dir = data_dir
        self.records = [
            '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
            '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
            '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
            '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
            '222', '223', '228', '230', '231', '232', '233', '234'
        ]

        # AAMI standard arrhythmia classification
        self.beat_classes = {
            # Normal beats
            'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
            # Supraventricular ectopic beats
            'A': 1, 'a': 1, 'J': 1, 'S': 1,
            # Ventricular ectopic beats
            'V': 2, 'E': 2,
            # Fusion beats
            'F': 3,
            # Unclassifiable beats
            '/': 4, 'f': 4, 'Q': 4
        }

        self.class_names = ['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown']

    def check_record_exists(self, record_name):
        """Check if a MIT-BIH record already exists locally"""
        # MIT-BIH records consist of .dat, .hea, and .atr files
        required_extensions = ['.dat', '.hea', '.atr']
        
        for ext in required_extensions:
            file_path = os.path.join(self.data_dir, record_name + ext)
            if not os.path.exists(file_path):
                return False
        return True

    def download_data(self, max_records=None, priority_records=None):
        """Download MIT-BIH data from PhysioNet with smart file checking"""
        print("Setting up MIT-BIH dataset...")

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Default priority records (known to have good quality and diverse arrhythmias)
        if priority_records is None:
            priority_records = [
                # Normal and common arrhythmias 
                '100', '101', '103', '105', '106', '107', '108', '109', '111', '112', 
                '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
                # More complex arrhythmias   
                '102', '104', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
                '213', '214', '215', '217', '219', '220', '221', '222', '223', '228',
                '230', '231', '232', '233', '234'
            ]

        # Determine records to check/download
        if max_records is None:
            records_to_process = priority_records  # Process all available
            print(f"Checking ALL {len(priority_records)} MIT-BIH records...")
        else:
            records_to_process = priority_records[:max_records]
            print(f"Checking {len(records_to_process)} MIT-BIH records (limited by max_records={max_records})...")

        # Check which files already exist vs need downloading
        existing_records = []
        missing_records = []
        
        print("📁 Checking existing files...")
        for record in records_to_process:
            if self.check_record_exists(record):
                existing_records.append(record)
            else:
                missing_records.append(record)
        
        print(f"✅ Found {len(existing_records)} records already downloaded")
        print(f"📥 Need to download {len(missing_records)} records")
        
        if existing_records:
            print(f"   Existing: {existing_records[:10]}{'...' if len(existing_records) > 10 else ''}")
        
        if missing_records:
            print(f"   Missing: {missing_records[:10]}{'...' if len(missing_records) > 10 else ''}")

        # Only download missing files
        if not missing_records:
            print("🎉 All required files already exist! Skipping download.")
            estimated_beats = len(existing_records) * 2000
            print(f"📊 Estimated total beats available: ~{estimated_beats:,}")
            return len(existing_records) > 0

        # Download missing files
        print(f"\n📥 Downloading {len(missing_records)} missing records...")
        try:
            successful_downloads = 0
            failed_downloads = []

            for i, record in enumerate(missing_records, 1):
                try:
                    print(f"Downloading record {record} ({i}/{len(missing_records)})...")
                    wfdb.dl_database('mitdb', self.data_dir, records=[record])
                    
                    # Verify download was successful
                    if self.check_record_exists(record):
                        successful_downloads += 1
                        print(f"  ✅ {record} downloaded successfully")
                    else:
                        print(f"  ❌ {record} download incomplete")
                        failed_downloads.append(record)
                        
                except Exception as e:
                    print(f"  ❌ Error downloading {record}: {e}")
                    failed_downloads.append(record)
                    continue

            # Summary
            total_available = len(existing_records) + successful_downloads
            print(f"\n📊 Data setup completed!")
            print(f"✅ Records already available: {len(existing_records)}")
            print(f"✅ Records newly downloaded: {successful_downloads}")
            print(f"✅ Total records available: {total_available}")
            
            if failed_downloads:
                print(f"❌ Failed downloads: {failed_downloads}")
            
            # Calculate estimated dataset size
            estimated_beats = total_available * 2000  # ~2000 beats per record average
            print(f"📊 Estimated total beats available: ~{estimated_beats:,}")
            
            return total_available > 0

        except Exception as e:
            print(f"Error during download process: {e}")
            # Still return True if we have existing files
            if existing_records:
                print(f"Using {len(existing_records)} existing records...")
                return True
            return False

    def load_record(self, record_name):
        """Load a single MIT-BIH record"""
        try:
            # Read the record
            record_path = os.path.join(self.data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            # Get ECG signal (use MLII lead if available, otherwise first lead)
            if 'MLII' in record.sig_name:
                signal_idx = record.sig_name.index('MLII')
            else:
                signal_idx = 0

            ecg_signal = record.p_signal[:, signal_idx]

            # Get annotations
            beat_locations = annotation.sample
            beat_types = annotation.symbol

            return ecg_signal, beat_locations, beat_types, record.fs

        except Exception as e:
            print(f"Error loading record {record_name}: {e}")
            return None, None, None, None

    def segment_beats(self, ecg_signal, beat_locations, beat_types, fs=360, window_size=360):
        """Segment ECG signal into individual beats"""
        beats = []
        labels = []

        half_window = window_size // 2

        for i, (location, beat_type) in enumerate(zip(beat_locations, beat_types)):
            # Skip if beat type not in our classification
            if beat_type not in self.beat_classes:
                continue

            # Define beat window
            start_idx = max(0, location - half_window)
            end_idx = min(len(ecg_signal), location + half_window)

            # Extract beat segment
            beat_segment = ecg_signal[start_idx:end_idx]

            # Pad or truncate to fixed size
            if len(beat_segment) < window_size:
                # Pad with zeros
                beat_segment = np.pad(beat_segment, (0, window_size - len(beat_segment)), 'constant')
            elif len(beat_segment) > window_size:
                # Truncate
                beat_segment = beat_segment[:window_size]

            beats.append(beat_segment)
            labels.append(self.beat_classes[beat_type])

        return np.array(beats), np.array(labels)

    def get_available_records_summary(self):
        """Get summary of locally available records"""
        available_records = []
        
        for record in self.records:
            if self.check_record_exists(record):
                available_records.append(record)
        
        return available_records

    def load_dataset(self, max_records=None, beats_per_class=500, use_full_dataset=True):
        """Load complete MIT-BIH dataset with configurable size and smart file management"""
        print("Loading MIT-BIH Arrhythmia dataset...")

        # Configure dataset size based on use_full_dataset flag
        if use_full_dataset:
            if max_records is None:
                max_records = len(self.records)  # Use all 48 records
                print("🔬 FULL DATASET MODE: Using all available MIT-BIH records")
            download_max = None  # Download all available
        else:
            if max_records is None:
                max_records = 10  # Default to 10 for faster testing
                print("⚡ FAST MODE: Using subset of records for quick testing")
            download_max = max_records

        # Check what's already available before downloading
        available_before = self.get_available_records_summary()
        print(f"📁 Local files status: {len(available_before)}/48 records already available")

        # Try to download/check data first
        download_success = self.download_data(max_records=download_max)
        
        if not download_success:
            print("⚠️ No data available for loading (download failed and no local files)")
            return np.array([]), np.array([]), []

        # Check what's available after download attempt
        available_after = self.get_available_records_summary()
        newly_downloaded = len(available_after) - len(available_before)
        
        if newly_downloaded > 0:
            print(f"📥 Successfully downloaded {newly_downloaded} new records")
        
        print(f"📊 Total records available for processing: {len(available_after)}")

        all_beats = []
        all_labels = []
        records_loaded = 0
        records_processed = []
        records_skipped = []

        # Class counters for balancing
        class_counts = {i: 0 for i in range(len(self.class_names))}
        
        # Track beats per record for statistics
        beats_per_record = {}

        # Process available records up to max_records limit
        available_records = available_after[:max_records] if max_records else available_after

        for record in available_records:
            print(f"Processing record {record} ({records_loaded + 1}/{len(available_records)})...")

            # Load record
            ecg_signal, beat_locations, beat_types, fs = self.load_record(record)

            if ecg_signal is None:
                print(f"  ❌ Skipped {record} (failed to load)")
                records_skipped.append(record)
                continue

            # Segment beats
            beats, labels = self.segment_beats(ecg_signal, beat_locations, beat_types, fs)

            if len(beats) == 0:
                print(f"  ❌ Skipped {record} (no valid beats found)")
                records_skipped.append(record)
                continue

            # Track beats found in this record
            beats_per_record[record] = len(beats)
            record_class_dist = np.bincount(labels, minlength=len(self.class_names))

            # Add beats with class balancing (if beats_per_class is specified)
            added_beats = 0
            if beats_per_class is None:
                # Add all beats without limit
                all_beats.extend(beats)
                all_labels.extend(labels)
                for label in labels:
                    class_counts[label] += 1
                added_beats = len(beats)
            else:
                # Add beats with class balancing
                for beat, label in zip(beats, labels):
                    if class_counts[label] < beats_per_class:
                        all_beats.append(beat)
                        all_labels.append(label)
                        class_counts[label] += 1
                        added_beats += 1

            records_loaded += 1
            records_processed.append(record)
            print(f"  ✅ Record {record}: {len(beats)} beats found, {added_beats} added")

        # Convert to numpy arrays
        all_beats = np.array(all_beats)
        all_labels = np.array(all_labels)

        # Fix class labels - ensure they are contiguous and start from 0
        unique_labels = np.unique(all_labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

        # Remap labels
        remapped_labels = np.array([label_mapping[label] for label in all_labels])

        # Update class names based on actual classes present
        present_classes = [self.class_names[label] for label in unique_labels]

        print(f"\n{'='*60}")
        print(f"📊 DATASET LOADING COMPLETE")
        print(f"{'='*60}")
        print(f"📁 Files status:")
        print(f"   Available locally: {len(available_after)} records")
        print(f"   Newly downloaded: {newly_downloaded} records") 
        print(f"   Requested for processing: {len(available_records)} records")
        print(f"   Successfully processed: {records_loaded} records")
        if records_skipped:
            print(f"   Skipped (errors): {len(records_skipped)} records")
        print(f"📊 Data extracted:")
        print(f"   Total beats: {len(all_beats):,}")
        print(f"   Classes present: {present_classes}")
        print(f"   Final class distribution: {dict(zip(present_classes, np.bincount(remapped_labels)))}")
        
        if beats_per_record:
            avg_beats_per_record = np.mean(list(beats_per_record.values()))
            print(f"   Average beats per record: {avg_beats_per_record:.1f}")
            coverage = len(available_after) / 48 * 100
            print(f"📏 Dataset coverage: {coverage:.1f}% of full MIT-BIH database ({len(available_after)}/48 records)")

        # Data quality check
        if len(all_beats) < 100:
            print(f"⚠️  WARNING: Very small dataset ({len(all_beats)} beats). Consider:")
            print(f"   - Increasing max_records (currently {max_records})")
            print(f"   - Increasing beats_per_class (currently {beats_per_class})")
            print(f"   - Setting use_full_dataset=True for complete dataset")

        return all_beats, remapped_labels, present_classes

    def create_synthetic_fallback(self, n_samples=1000):
        """Create synthetic data if real data is not available"""
        print("Creating synthetic MIT-BIH style data as fallback...")

        signals = []
        labels = []

        # Only generate for classes that actually exist (0, 1, 2, 3)
        active_classes = 4  # Normal, SVEB, VEB, Fusion
        samples_per_class = n_samples // active_classes

        for class_idx in range(active_classes):
            for _ in range(samples_per_class):
                beat = self._generate_beat_by_class(class_idx)
                # Add realistic noise
                beat += np.random.normal(0, 0.05, len(beat))
                signals.append(beat)
                labels.append(class_idx)

        return np.array(signals), np.array(labels), ['Normal', 'SVEB', 'VEB', 'Fusion']

    def _generate_beat_by_class(self, class_idx, length=360):
        """Generate synthetic beat based on class type"""
        t = np.linspace(0, 1, length)

        if class_idx == 0:  # Normal
            return self._normal_beat(t)
        elif class_idx == 1:  # SVEB
            return self._sveb_beat(t)
        elif class_idx == 2:  # VEB
            return self._veb_beat(t)
        elif class_idx == 3:  # Fusion
            return self._fusion_beat(t)
        else:  # Unknown
            return self._unknown_beat(t)

    def _normal_beat(self, t):
        """Generate normal sinus beat"""
        # P wave
        p = 0.1 * np.exp(-((t - 0.2) * 15)**2)
        # QRS complex
        qrs = (0.8 * np.exp(-((t - 0.5) * 40)**2) -
               0.2 * np.exp(-((t - 0.48) * 80)**2) -
               0.2 * np.exp(-((t - 0.52) * 80)**2))
        # T wave
        t_wave = 0.3 * np.exp(-((t - 0.75) * 12)**2)
        return p + qrs + t_wave

    def _sveb_beat(self, t):
        """Generate supraventricular ectopic beat"""
        # Early, abnormal P wave
        p = 0.15 * np.exp(-((t - 0.15) * 20)**2)
        # Normal-ish QRS but slightly different
        qrs = (0.7 * np.exp(-((t - 0.45) * 35)**2) -
               0.15 * np.exp(-((t - 0.43) * 70)**2) -
               0.15 * np.exp(-((t - 0.47) * 70)**2))
        # Different T wave
        t_wave = 0.25 * np.exp(-((t - 0.7) * 10)**2)
        return p + qrs + t_wave

    def _veb_beat(self, t):
        """Generate ventricular ectopic beat"""
        # No P wave
        # Wide, bizarre QRS
        qrs = (1.0 * np.exp(-((t - 0.4) * 15)**2) -
               0.3 * np.exp(-((t - 0.35) * 25)**2) +
               0.4 * np.exp(-((t - 0.55) * 20)**2))
        # Inverted T wave
        t_wave = -0.2 * np.exp(-((t - 0.8) * 8)**2)
        return qrs + t_wave

    def _fusion_beat(self, t):
        """Generate fusion beat"""
        normal = self._normal_beat(t)
        veb = self._veb_beat(t)
        return 0.6 * normal + 0.4 * veb

    def _unknown_beat(self, t):
        """Generate unclassifiable beat"""
        # Irregular, noisy pattern
        noise = np.random.random(len(t)) * 0.3
        base = 0.5 * np.exp(-((t - 0.5) * 10)**2)
        return base + noise

class ECGFeatureExtractor:
    """Extract comprehensive features from ECG beats"""

    def __init__(self, fs=360):
        self.fs = fs

    def extract_all_features(self, beat):
        """Extract all types of features from an ECG beat"""
        features = {}

        # Morphological features
        features.update(self.extract_morphological_features(beat))

        # Statistical features
        features.update(self.extract_statistical_features(beat))

        # Frequency domain features
        features.update(self.extract_frequency_features(beat))

        # Wavelet-based features
        features.update(self.extract_wavelet_features(beat))

        # RR interval features (simulated)
        features.update(self.extract_rr_features(beat))

        return features

    def extract_morphological_features(self, beat):
        """Extract morphological features"""
        features = {}

        # Find R peak
        r_peak_idx = np.argmax(np.abs(beat))
        features['r_peak_amplitude'] = beat[r_peak_idx]
        features['r_peak_position'] = r_peak_idx / len(beat)

        # QRS width estimation
        qrs_start = max(0, r_peak_idx - 40)
        qrs_end = min(len(beat), r_peak_idx + 40)
        qrs_region = beat[qrs_start:qrs_end]

        # Find QRS boundaries (simplified)
        threshold = 0.1 * np.max(np.abs(qrs_region))
        qrs_width = np.sum(np.abs(qrs_region) > threshold)
        features['qrs_width'] = qrs_width

        # P and T wave features (simplified detection)
        pre_qrs = beat[:max(0, r_peak_idx - 50)]
        post_qrs = beat[min(len(beat), r_peak_idx + 50):]

        if len(pre_qrs) > 0:
            features['p_wave_amplitude'] = np.max(np.abs(pre_qrs))
        else:
            features['p_wave_amplitude'] = 0

        if len(post_qrs) > 0:
            features['t_wave_amplitude'] = np.max(np.abs(post_qrs))
        else:
            features['t_wave_amplitude'] = 0

        # Beat area and energy
        features['beat_area'] = np.trapz(np.abs(beat))
        features['beat_energy'] = np.sum(beat**2)

        return features

    def extract_statistical_features(self, beat):
        """Extract statistical features"""
        features = {}

        features['mean'] = np.mean(beat)
        features['std'] = np.std(beat)
        features['var'] = np.var(beat)
        features['skewness'] = skew(beat)
        features['kurtosis_stat'] = kurtosis(beat)
        features['rms'] = np.sqrt(np.mean(beat**2))
        features['max_val'] = np.max(beat)
        features['min_val'] = np.min(beat)
        features['range_val'] = features['max_val'] - features['min_val']
        features['median'] = np.median(beat)
        features['mad'] = np.median(np.abs(beat - features['median']))  # Median absolute deviation

        # Percentiles
        features['q25'] = np.percentile(beat, 25)
        features['q75'] = np.percentile(beat, 75)
        features['iqr'] = features['q75'] - features['q25']

        return features

    def extract_frequency_features(self, beat):
        """Extract frequency domain features"""
        features = {}

        # FFT
        fft_beat = np.fft.fft(beat)
        freqs = np.fft.fftfreq(len(beat), 1/self.fs)
        power = np.abs(fft_beat)**2

        # Frequency bands
        low_freq = (freqs >= 0.5) & (freqs < 15)
        mid_freq = (freqs >= 15) & (freqs < 40)
        high_freq = (freqs >= 40) & (freqs < 100)

        features['low_freq_power'] = np.sum(power[low_freq])
        features['mid_freq_power'] = np.sum(power[mid_freq])
        features['high_freq_power'] = np.sum(power[high_freq])
        features['total_power'] = np.sum(power)

        # Spectral characteristics
        if features['total_power'] > 0:
            features['spectral_centroid'] = np.sum(freqs[:len(freqs)//2] * power[:len(power)//2]) / np.sum(power[:len(power)//2])
        else:
            features['spectral_centroid'] = 0

        # Dominant frequency
        features['dominant_freq'] = freqs[np.argmax(power[:len(power)//2])]

        return features

    def extract_wavelet_features(self, beat):
        """Extract wavelet-inspired features using filtering"""
        features = {}

        def bandpass_filter(data, lowcut, highcut, fs, order=4):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, data)

        # Decompose into frequency bands (simulating wavelet decomposition)
        try:
            d1 = bandpass_filter(beat, 64, 128, self.fs)  # High frequency details
            d2 = bandpass_filter(beat, 32, 64, self.fs)
            d3 = bandpass_filter(beat, 16, 32, self.fs)
            d4 = bandpass_filter(beat, 8, 16, self.fs)
            d5 = bandpass_filter(beat, 4, 8, self.fs)
            a5 = bandpass_filter(beat, 0.5, 4, self.fs)    # Low frequency approximation

            coeffs = [d1, d2, d3, d4, d5, a5]

            for i, coeff in enumerate(coeffs, 1):
                features[f'wavelet_energy_{i}'] = np.sum(coeff**2)
                features[f'wavelet_std_{i}'] = np.std(coeff)
                features[f'wavelet_mean_{i}'] = np.mean(coeff)
                features[f'wavelet_max_{i}'] = np.max(np.abs(coeff))

        except Exception as e:
            # Fallback if filtering fails
            for i in range(1, 7):
                features[f'wavelet_energy_{i}'] = 0
                features[f'wavelet_std_{i}'] = 0
                features[f'wavelet_mean_{i}'] = 0
                features[f'wavelet_max_{i}'] = 0

        return features

    def extract_rr_features(self, beat):
        """Extract RR interval related features (simplified for single beat)"""
        features = {}

        # Since we're working with single beats, we'll extract beat-to-beat variation features
        # This is a simplified representation

        # Beat duration (fixed for our case)
        features['beat_duration'] = len(beat) / self.fs

        # Estimate heart rate from beat characteristics
        # This is a rough estimation based on beat morphology
        r_peak_prominence = np.max(beat) - np.min(beat)
        features['estimated_hr'] = 60.0  # Default, would be calculated from actual RR intervals

        # Beat interval variability (simulated)
        features['rr_variability'] = np.std(beat) / (np.mean(beat) + 1e-8)

        return features

class OptimizedGNNFeatureSelector(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_heads=4, dropout=0.4):
        super(OptimizedGNNFeatureSelector, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.input_proj = nn.Linear(1, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, concat=False)

        # Deep attention network
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None, return_attention=True):
        x = self.input_proj(x)
        h1 = F.relu(self.gat1(x, edge_index))
        h1 = self.dropout(h1)
        h2 = F.relu(self.gat2(h1, edge_index))
        h_combined = h1 + h2

        attention_scores = None
        if return_attention:
            attention_scores = torch.sigmoid(self.feature_attention(h_combined)).squeeze(-1)

        if batch is not None:
            graph_embeddings = global_mean_pool(h2, batch)
        else:
            graph_embeddings = torch.mean(h2, dim=0, keepdim=True)

        out = self.classifier(graph_embeddings)

        if return_attention and attention_scores is not None:
            if batch is not None:
                batch_size = batch.max().item() + 1
                attention_per_graph = []
                for i in range(batch_size):
                    mask = (batch == i)
                    graph_attention = attention_scores[mask]
                    if len(graph_attention) == self.num_features:
                        attention_per_graph.append(graph_attention)
                    elif len(graph_attention) < self.num_features:
                        padded = torch.zeros(self.num_features, device=graph_attention.device)
                        padded[:len(graph_attention)] = graph_attention
                        attention_per_graph.append(padded)
                    else:
                        attention_per_graph.append(graph_attention[:self.num_features])
                final_attention = torch.stack(attention_per_graph).mean(dim=0)
                return out, final_attention
            else:
                if len(attention_scores) == self.num_features:
                    return out, attention_scores
                else:
                    padded = torch.zeros(self.num_features, device=attention_scores.device)
                    padded[:len(attention_scores)] = attention_scores[:self.num_features]
                    return out, padded

        return out

class OptimizedGNNFeatureSelector(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_heads=4, dropout=0.4):
        super(OptimizedGNNFeatureSelector, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.input_proj = nn.Linear(1, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, concat=False)

        # Deep attention network
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None, return_attention=True):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.input_proj(x)
        h1 = F.relu(self.gat1(x, edge_index))
        h1 = self.dropout(h1)
        h2 = F.relu(self.gat2(h1, edge_index))
        h_combined = h2

        attention_scores = None
        if return_attention:
            attention_scores = torch.sigmoid(self.feature_attention(h_combined)).squeeze(-1)

        if batch is not None:
            graph_embeddings = global_mean_pool(h2, batch)
        else:
            graph_embeddings = torch.mean(h2, dim=0, keepdim=True)

        out = self.classifier(graph_embeddings)

        if return_attention and attention_scores is not None:
            if batch is not None:
                batch_size = batch.max().item() + 1
                attention_per_graph = []
                for i in range(batch_size):
                    mask = (batch == i)
                    graph_attention = attention_scores[mask]
                    if len(graph_attention) == self.num_features:
                        attention_per_graph.append(graph_attention)
                    elif len(graph_attention) < self.num_features:
                        padded = torch.zeros(self.num_features, device=graph_attention.device)
                        padded[:len(graph_attention)] = graph_attention
                        attention_per_graph.append(padded)
                    else:
                        attention_per_graph.append(graph_attention[:self.num_features])
                final_attention = torch.stack(attention_per_graph).mean(dim=0)
                return out, final_attention
            else:
                if len(attention_scores) == self.num_features:
                    return out, attention_scores
                else:
                    padded = torch.zeros(self.num_features, device=attention_scores.device)
                    padded[:len(attention_scores)] = attention_scores[:self.num_features]
                    return out, padded

        return out

# Modify graph creation to use cosine similarity
class GEFAApproach:
    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
    def create_feature_graph(self, X, y=None, method='cosine', threshold=0.4):
        print(f"🕸️ Creating feature graph using {method}...")

        if method == 'mutual_info':
            if y is None:
                raise ValueError("y must be provided for mutual_info graph.")
            mi = mutual_info_classif(X, y, discrete_features='auto')
            adj_matrix = np.outer(mi, mi)
            adj_matrix = (adj_matrix > np.percentile(adj_matrix, 75)).astype(float)
            np.fill_diagonal(adj_matrix, 0)

        elif method == 'cosine':
            sim_matrix = cosine_similarity(X.T)
            adj_matrix = (sim_matrix > threshold).astype(float)
            np.fill_diagonal(adj_matrix, 0)
    
        else:
            raise ValueError("Unsupported method")

        edge_indices = np.where(adj_matrix)
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        return edge_index, adj_matrix
  
    def train(self, X, y, feature_names, graph_method='cosine', epochs=180):
        print(f"🔧 Starting GEFA training...")
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        edge_index, adj_matrix = self.create_feature_graph(X_scaled, method='cosine', threshold=0.4)


  
        self.model = OptimizedGNNFeatureSelector(X.shape[1], len(np.unique(y)), hidden_dim=64, num_heads=4, dropout=0.4)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
  
        node_features_list = [torch.FloatTensor(sample).unsqueeze(-1) for sample in X_scaled]
        labels_list = list(y)
        self.model.train()
        batch_size = 64
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        for epoch in range(epochs):
            total_loss = 0
            indices = torch.randperm(len(node_features_list))
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_data = []
                batch_labels = []
                for idx in batch_indices:
                    data = Data(x=node_features_list[idx], edge_index=edge_index)
                    batch_data.append(data)
                    batch_labels.append(labels_list[idx])
  
                batch = Batch.from_data_list(batch_data)
                labels_tensor = torch.LongTensor(batch_labels)
  
                optimizer.zero_grad()
                output, attention_scores = self.model(batch.x, batch.edge_index, batch.batch, return_attention=True)
                loss = criterion(output, labels_tensor)
  
                if attention_scores is not None:
                    l1 = torch.mean(torch.abs(attention_scores))
                    var = -torch.var(attention_scores)
                    loss += 0.05 * l1 + 0.03 * var
  
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss/len(node_features_list)
            print(f"Epoch {epoch}/{epochs}, Avg Loss: {avg_loss:.4f}")
            if epoch > 100:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"⏹️ Early stopping at epoch {epoch}")
                        break

        return adj_matrix
            
  
    def get_feature_importance(self, X, y= None, graph_method='cosine'):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        edge_index, _ = self.create_feature_graph(X_scaled, y=y,  method= graph_method)

  

  
        all_scores = []
        for i in range(0, len(X_scaled), 64):
            batch_samples = X_scaled[i:i+64]
            data_list = [Data(x=torch.FloatTensor(s).unsqueeze(-1), edge_index=edge_index) for s in batch_samples]
            batch = Batch.from_data_list(data_list)
            with torch.no_grad():
                _, attention = self.model(batch.x, batch.edge_index, batch.batch, return_attention=True)
                all_scores.append(attention.numpy())
  
        return np.mean(np.stack(all_scores), axis=0)
    def get_enhanced_feature_importance(self, X, y, use_gnn_explainer=True, graph_method='cosine'):
        print("⚙️ Computing attention-based + Ridge hybrid importance...")
        attention_importance = self.get_feature_importance(X=X, y=y, graph_method=graph_method)

    
        lasso = LassoCV(cv=5, random_state=42).fit(X, y)
        lasso_scores = np.abs(lasso.coef_)
        lasso_scores /= (np.max(lasso_scores) + 1e-8)
        hybrid_scores = 0.7 * attention_importance + 0.3 * lasso_scores


        print("✅ Hybrid GEFA+Lasso importance computed")
        return hybrid_scores, attention_importance, None



    def select_features(self, X, y, feature_names, k=35):
        """Select top-k features using GEFA"""
        print(f"\n=== GEFA Feature Selection ===")

    # Train model
        adj_matrix = self.train(X, y, feature_names, graph_method='cosine')


    # Get enhanced feature importance
        combined_importance, attention_importance, gnn_importance = self.get_enhanced_feature_importance(X=X, y=y, graph_method = 'cosine')


    # Select top-k features
        top_k_indices = np.argsort(combined_importance)[-k:][::-1]
        selected_features = [feature_names[i] for i in top_k_indices]

        print(f"🏆 Selected top-{k} features: {selected_features}")
        print(f"🔢 Importance scores: {combined_importance[top_k_indices]}")
        print(f"🧠 GNNExplainer contribution: ❌ Skipped (attention only)")

        return top_k_indices, selected_features, combined_importance, adj_matrix


# Feature Selection Methods for Comparison
class LinearFeatureSelectors:
    """Linear feature selection methods for comparison with GEFA"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def lasso_selection(self, X, y, feature_names, k=15):
        """Lasso feature selection with cross-validation"""
        print(f"\n=== Lasso Feature Selection ===")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Use LassoCV for automatic alpha selection
        lasso_cv = LassoCV(cv=5, random_state=42, max_iter=1000)
        lasso_cv.fit(X_scaled, y)
        
        # Get feature importance (absolute coefficients)
        importance = np.abs(lasso_cv.coef_)
        top_k_indices = np.argsort(importance)[-k:][::-1]
        selected_features = [feature_names[i] for i in top_k_indices]
        
        print(f"Best alpha: {lasso_cv.alpha_:.4f}")
        print(f"Selected features: {selected_features}")
        print(f"Importance scores: {importance[top_k_indices]}")
        
        return top_k_indices, selected_features, importance
    
    def ridge_selection(self, X, y, feature_names, k=15):
        """Ridge feature selection with cross-validation"""
        print(f"\n=== Ridge Feature Selection ===")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Use RidgeCV for automatic alpha selection
        ridge_cv = RidgeCV(cv=5)
        ridge_cv.fit(X_scaled, y)
        
        # Get feature importance (absolute coefficients)
        importance = np.abs(ridge_cv.coef_)
        top_k_indices = np.argsort(importance)[-k:][::-1]
        selected_features = [feature_names[i] for i in top_k_indices]
        
        print(f"Best alpha: {ridge_cv.alpha_:.4f}")
        print(f"Selected features: {selected_features}")
        print(f"Importance scores: {importance[top_k_indices]}")
        
        return top_k_indices, selected_features, importance
    
    def elastic_net_selection(self, X, y, feature_names, k=15):
        """Elastic Net feature selection with cross-validation"""
        print(f"\n=== Elastic Net Feature Selection ===")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Use ElasticNetCV for automatic parameter selection
        elastic_cv = ElasticNetCV(cv=5, random_state=42, max_iter=1000)
        elastic_cv.fit(X_scaled, y)
        
        # Get feature importance (absolute coefficients)
        importance = np.abs(elastic_cv.coef_)
        top_k_indices = np.argsort(importance)[-k:][::-1]
        selected_features = [feature_names[i] for i in top_k_indices]
        
        print(f"Best alpha: {elastic_cv.alpha_:.4f}, Best l1_ratio: {elastic_cv.l1_ratio_:.4f}")
        print(f"Selected features: {selected_features}")
        print(f"Importance scores: {importance[top_k_indices]}")
        
        return top_k_indices, selected_features, importance

def analyze_graph_topology(adj_matrix, feature_names):
    """Analyze the learned graph topology with enhanced insights"""
    print("\n=== GEFA Feature Graph Topology Analysis ===")
    
    if adj_matrix is None:
        print("❌ No adjacency matrix available")
        return None
    
    # Calculate node degrees
    node_degrees = np.sum(adj_matrix, axis=1)
    total_edges = np.sum(adj_matrix)
    n_features = len(feature_names)
    
    print(f"📊 Graph Statistics:")
    print(f"   Total features (nodes): {n_features}")
    print(f"   Total connections (edges): {total_edges:.0f}")
    print(f"   Possible connections: {n_features * (n_features - 1)}")
    
    # Graph density
    if n_features > 1:
        total_possible_edges = n_features * (n_features - 1)
        density = total_edges / total_possible_edges
        print(f"   Graph density: {density:.4f} ({density*100:.2f}%)")
    else:
        density = 0
        print(f"   Graph density: 0 (insufficient features)")
    
    # Node degree statistics
    print(f"📈 Node Degree Statistics:")
    print(f"   Average degree: {np.mean(node_degrees):.2f}")
    print(f"   Max degree: {np.max(node_degrees):.0f}")
    print(f"   Min degree: {np.min(node_degrees):.0f}")
    print(f"   Std deviation: {np.std(node_degrees):.2f}")
    
    # Find most connected features
    most_connected = np.argsort(node_degrees)[-10:][::-1]
    
    print(f"🔗 Most Connected Features:")
    for i, idx in enumerate(most_connected, 1):
        if node_degrees[idx] > 0:
            print(f"   {i:2d}. {feature_names[idx]:<25} ({node_degrees[idx]:.0f} connections)")
    
    # Find least connected features
    least_connected = np.argsort(node_degrees)[:5]
    isolated_features = [idx for idx in least_connected if node_degrees[idx] == 0]
    
    if isolated_features:
        print(f"🔇 Isolated Features (no connections):")
        for idx in isolated_features:
            print(f"      {feature_names[idx]}")
    
    # Community analysis (simple clustering based on connectivity)
    print(f"🌐 Graph Connectivity:")
    if density > 0.5:
        print(f"   Highly connected graph - features are strongly interdependent")
    elif density > 0.1:
        print(f"   Moderately connected graph - selective feature relationships")
    else:
        print(f"   Sparsely connected graph - features are mostly independent")
    
    return {
        'node_degrees': node_degrees,
        'most_connected': most_connected,
        'density': density,
        'total_edges': total_edges,
        'avg_degree': np.mean(node_degrees),
        'isolated_features': len([idx for idx in range(n_features) if node_degrees[idx] == 0])
    }

def evaluate_feature_selection(X, y, selected_indices, method_name, class_names, cv_folds=5):
    """Enhanced evaluation with RF, SVM, and XGBoost"""
    print(f"\n=== Enhanced Evaluation: {method_name} ===")

    # Select features
    X_selected = X[:, selected_indices]

    # Split data for detailed analysis
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test three classifiers with cross-validation
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        classifiers['XGBoost'] = xgb.XGBClassifier(
            random_state=42, 
            eval_metric='mlogloss',
            verbosity=0  # Suppress XGBoost warnings
        )

    results = {}
    cv_results = {}
    
    for clf_name, clf in classifiers.items():
        # Cross-validation scores
        cv_scores = cross_val_score(clf, X_selected, y,
                            cv=cv_folds, scoring='accuracy')   

        
        cv_results[clf_name] = {
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'scores': cv_scores
        }
        
        # Train and evaluate on hold-out test set
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        results[clf_name] = accuracy

        print(f"{clf_name} Test Accuracy: {accuracy:.4f}")
        print(f"{clf_name} CV Accuracy: {cv_results[clf_name]['mean']:.4f} ± {cv_results[clf_name]['std']:.4f}")

        # Detailed classification report for best classifier
        if clf_name == 'Random Forest':
            print(f"\nDetailed Classification Report ({clf_name}):")

            # Get unique labels in test set
            unique_test_labels = sorted(set(y_test))

            # Create target names for only the classes present in test set
            test_class_names = [class_names[i] for i in unique_test_labels]

            print(classification_report(y_test, y_pred,
                                      labels=unique_test_labels,
                                      target_names=test_class_names,
                                      zero_division=0))

    return results, cv_results

def get_dataset_config_recommendations():
    """Provide dataset configuration recommendations based on use case"""
    
    configs = {
        'quick_test': {
            'use_full_dataset': False,
            'max_records': 5,
            'beats_per_class': 200,
            'description': 'Quick testing (5 records, ~2k beats, ~2 minutes)',
            'use_case': 'Code testing, debugging, initial development'
        },
        
        'medium_development': {
            'use_full_dataset': False,
            'max_records': 15,
            'beats_per_class': 500,
            'description': 'Medium development (15 records, ~7k beats, ~5 minutes)',
            'use_case': 'Method development, hyperparameter tuning'
        },
        
        'research_validation': {
            'use_full_dataset': True,
            'max_records': 30,
            'beats_per_class': 1000,
            'description': 'Research validation (30 records, ~30k beats, ~15 minutes)',
            'use_case': 'Method validation, paper results'
        },
        
        'full_benchmark': {
            'use_full_dataset': True,
            'max_records': None,  # All 48 records
            'beats_per_class': None,  # No limit
            'description': 'Full benchmark (48 records, 100k+ beats, ~60 minutes)',
            'use_case': 'Final benchmarking, clinical validation, publication'
        }
    }
    
    return configs

def print_dataset_recommendations():
    """Print dataset configuration recommendations"""
    configs = get_dataset_config_recommendations()
    
    print("📚 DATASET SIZE RECOMMENDATIONS:")
    print("=" * 50)
    
    for config_name, config in configs.items():
        print(f"\n🔹 {config_name.upper().replace('_', ' ')}:")
        print(f"   Description: {config['description']}")
        print(f"   Use case: {config['use_case']}")
        print(f"   Config: use_full_dataset={config['use_full_dataset']}")
        print(f"           max_records={config['max_records']}")
        print(f"           beats_per_class={config['beats_per_class']}")
    
    print(f"\n💡 To use a configuration, update DATASET_CONFIG in main() function")

def main():
    """Main execution function with FIXED f-string formatting"""
    print("🫀 GNN-Based Feature Selection for MIT-BIH ECG Classification")
    print("⚡ GEFA vs Linear Methods Comparison - FIXED VERSION")
    print("=" * 80)
    
    # Print dataset recommendations
    print_dataset_recommendations()
    
    # Dataset configuration
    DATASET_CONFIG = get_dataset_config_recommendations()['research_validation']  
    
    DATASET_CONFIG['min_class_samples'] = 30
    
    selected_description = DATASET_CONFIG['description']
    selected_config_name = next(
        (k for k, v in get_dataset_config_recommendations().items() if v['description'] == selected_description),
      'custom_config')
    print(f"\n📋 SELECTED CONFIGURATION: {selected_config_name}")

    print(f"   Full dataset mode: {DATASET_CONFIG['use_full_dataset']}")
    print(f"   Max records: {DATASET_CONFIG['max_records'] or 'ALL (48 records)'}")
    print(f"   Beats per class: {DATASET_CONFIG['beats_per_class'] or 'NO LIMIT'}")
    print(f"   Min class samples: {DATASET_CONFIG['min_class_samples']}")
    print(f"🔬 GNNExplainer: {'Available (ULTRA-FAST MODE)' if GNNEXPLAINER_AVAILABLE else 'Not Available'}")
    print(f"🚀 XGBoost: {'Available' if XGBOOST_AVAILABLE else 'Using Random Forest fallback'}")

    # Initialize data loader
    data_loader = MITBIHDataLoader()

    # Try to load real MIT-BIH data
    try:
        print("Attempting to load real MIT-BIH dataset...")
        beats, labels, class_names = data_loader.load_dataset(
            max_records=DATASET_CONFIG['max_records'],
            beats_per_class=DATASET_CONFIG['beats_per_class'],
            use_full_dataset=DATASET_CONFIG['use_full_dataset']
        )

        if len(beats) == 0:
            raise Exception("No data loaded")

        print(f"✅ Real MIT-BIH data loaded successfully!")

    except Exception as e:
        print(f"❌ Could not load real MIT-BIH data: {e}")
        print("Using synthetic fallback data...")
        
        # Scale synthetic data size based on configuration
        if DATASET_CONFIG['use_full_dataset']:
            if DATASET_CONFIG['max_records'] is None:
                synthetic_size = 5000
            else:
                synthetic_size = DATASET_CONFIG['max_records'] * 100
        else:
            synthetic_size = 1500
            
        beats, labels, class_names = data_loader.create_synthetic_fallback(n_samples=synthetic_size)
        print(f"Generated {len(beats)} synthetic beats for testing")

    print(f"Dataset: {len(beats)} beats, {len(np.unique(labels))} classes")
    print(f"Class distribution: {np.bincount(labels)}")

    # Check for class imbalance and handle it
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Unique labels: {unique_labels}")
    print(f"Label counts: {counts}")

    # Remove classes with too few samples
    min_samples = DATASET_CONFIG['min_class_samples']
    valid_classes = unique_labels[counts >= min_samples]

    if len(valid_classes) < len(unique_labels):
        print(f"Removing classes with less than {min_samples} samples...")
        
        valid_mask = np.isin(labels, valid_classes)
        beats = beats[valid_mask]
        labels = labels[valid_mask]

        # Remap labels to be contiguous
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_classes)}
        labels = np.array([label_mapping[label] for label in labels])
        class_names = [class_names[i] for i in valid_classes]

        print(f"After filtering: {len(beats)} beats, {len(np.unique(labels))} classes")
        print(f"Final class distribution: {np.bincount(labels)}")
        print(f"Final classes: {class_names}")

    # Extract features from beats
    print("\nExtracting comprehensive ECG features...")
    feature_extractor = ECGFeatureExtractor()

    all_features = []
    feature_names = None

    for i, beat in enumerate(beats):
        if i % 500 == 0:
            print(f"Processing beat {i+1}/{len(beats)}")

        # Extract all features
        features_dict = feature_extractor.extract_all_features(beat)

        if feature_names is None:
            feature_names = list(features_dict.keys())

        all_features.append(list(features_dict.values()))

    # Convert to numpy array
    X = np.array(all_features)
    y = labels

    print(f"\nFeature extraction complete!")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Total features extracted: {len(feature_names)}")

    # Handle any NaN or infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Initialize approaches
    num_features = X.shape[1]
    num_classes = len(np.unique(y))

    print(f"\nInitializing feature selection approaches...")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    
    # Test PyTorch availability for GEFA
    try:
        test_tensor = torch.FloatTensor([1.0, 2.0, 3.0])
        print(f"✅ PyTorch working correctly")
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        print("GEFA will likely fail - continuing with linear methods only")
    
    # Test PyTorch Geometric availability
    try:
        from torch_geometric.data import Data
        test_data = Data(x=torch.FloatTensor([[1], [2]]), edge_index=torch.LongTensor([[0], [1]]))
        print(f"✅ PyTorch Geometric working correctly")
    except Exception as e:
        print(f"❌ PyTorch Geometric error: {e}")
        print("GEFA will likely fail - continuing with linear methods only")

    # Feature selection with all approaches
    k = min(20, num_features)  # Select top 20 features
    
    all_methods_results = []
    
    # Store results for detailed comparison
    gefa_indices, gefa_features, gefa_scores, adj_matrix = None, None, None, None
    lasso_indices, lasso_features, lasso_scores = None, None, None
    ridge_indices, ridge_features, ridge_scores = None, None, None
    elastic_indices, elastic_features, elastic_scores = None, None, None
    rf_indices, rf_features, rf_importance = None, None, None
    
    # 1. GEFA approach 
    print(f"\n{'='*50}")
    print(f"🧠 Testing GEFA (Graph Neural Network) Feature Selection")
    print(f"{'='*50}")
    
    try:
        print("Initializing GEFA...")
        gefa = GEFAApproach(num_features, num_classes)
        print("GEFA initialized, starting feature selection...")
        
        gefa_indices, gefa_features, gefa_scores, adj_matrix = gefa.select_features(
            X, y, feature_names, k=25
        )
        # After selection
        

        
        print(f"📊 GEFA Feature Graph Analysis:")
        print(f"   Graph edges: {np.sum(adj_matrix):.0f}")
        print(f"   Graph density: {np.sum(adj_matrix) / (len(feature_names) * (len(feature_names) - 1)):.4f}")
        
        # Analyze graph topology
        topology_analysis = analyze_graph_topology(adj_matrix, feature_names)
        
        # Enhanced evaluation
        print("Evaluating GEFA performance...")
        gefa_results, gefa_cv = evaluate_feature_selection(
            X, y, gefa_indices, "GEFA", class_names
        )
        
        all_methods_results.append(("GEFA", gefa_results, gefa_cv))
        print(f"✅ GEFA completed successfully!")
        
    except Exception as e:
        print(f"❌ Error with GEFA: {e}")
        import traceback
        print(f"Full error traceback:")
        traceback.print_exc()
        print("Continuing with other methods...")

    # 2. Linear Methods Comparison
    print(f"\n{'='*60}")
    print(f"📏 Testing Linear Regularization Methods")
    print(f"{'='*60}")
    
    try:
        linear_selectors = LinearFeatureSelectors()
        
        # Lasso
        print(f"\n--- Lasso (L1 Regularization) ---")
        lasso_indices, lasso_features, lasso_scores = linear_selectors.lasso_selection(X, y, feature_names, k=k)
        lasso_results, lasso_cv = evaluate_feature_selection(X, y, lasso_indices, "Lasso", class_names)
        all_methods_results.append(("Lasso", lasso_results, lasso_cv))

        # Ridge
        print(f"\n--- Ridge (L2 Regularization) ---")
        ridge_indices, ridge_features, ridge_scores = linear_selectors.ridge_selection(X, y, feature_names, k=k)
        ridge_results, ridge_cv = evaluate_feature_selection(X, y, ridge_indices, "Ridge", class_names)
        all_methods_results.append(("Ridge", ridge_results, ridge_cv))

        # Elastic Net
        print(f"\n--- Elastic Net (L1 + L2 Regularization) ---")
        elastic_indices, elastic_features, elastic_scores = linear_selectors.elastic_net_selection(X, y, feature_names, k=k)
        elastic_results, elastic_cv = evaluate_feature_selection(X, y, elastic_indices, "Elastic Net", class_names)
        all_methods_results.append(("Elastic Net", elastic_results, elastic_cv))
        
        print(f"✅ All linear methods completed!")
        
    except Exception as e:
        print(f"❌ Error with linear methods: {e}")

    # 3. Baseline: Random Forest feature importance
    print(f"\n{'='*50}")
    print("🌳 Baseline: Random Forest Feature Importance")
    print(f"{'='*50}")
    
    try:
        rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_baseline.fit(X, y)
        rf_importance = rf_baseline.feature_importances_
        rf_indices = np.argsort(rf_importance)[-k:][::-1]
        rf_features = [feature_names[i] for i in rf_indices]
        print(f"Selected features: {rf_features}")

        baseline_results, baseline_cv = evaluate_feature_selection(X, y, rf_indices, "RF Baseline", class_names)
        all_methods_results.append(("RF Baseline", baseline_results, baseline_cv))
        print(f"✅ Random Forest baseline completed!")
    except Exception as e:
        print(f"❌ Error with RF Baseline: {e}")

    # Compare results with focus on GEFA vs Linear Methods
    print(f"\n{'='*100}")
    print("🏆 FEATURE SELECTION COMPARISON RESULTS")
    print(f"{'='*100}")
    
    # Check if we have any results to compare
    if not all_methods_results:
        print("❌ No methods completed successfully!")
        return
    
    print(f"✅ Successfully completed {len(all_methods_results)} methods")
    
    # Print results table
    if XGBOOST_AVAILABLE:
        print(f"{'Method':<15} {'RF Test':<10} {'RF CV':<12} {'SVM Test':<10} {'SVM CV':<12} {'XGB Test':<10} {'XGB CV':<12} {'Avg Score':<12}")
        print("-" * 105)
    else:
        print(f"{'Method':<15} {'RF Test':<10} {'RF CV':<12} {'SVM Test':<10} {'SVM CV':<12} {'Avg Score':<12}")
        print("-" * 75)

    best_scores = []
    gefa_performance = None
    
    for method_name, results, cv_results in all_methods_results:
        if results and cv_results:
            rf_test = results.get('Random Forest', 0)
            rf_cv = cv_results.get('Random Forest', {}).get('mean', 0)
            svm_test = results.get('SVM', 0)
            svm_cv = cv_results.get('SVM', {}).get('mean', 0)
            
            if XGBOOST_AVAILABLE:
                xgb_test = results.get('XGBoost', 0)
                xgb_cv = cv_results.get('XGBoost', {}).get('mean', 0)
                avg_score = np.mean([rf_cv, svm_cv, xgb_cv])
                
                # Highlight GEFA results
                if method_name == "GEFA":
                    print(f"🧠 {method_name:<12} {rf_test:<10.4f} {rf_cv:<12.4f} {svm_test:<10.4f} {svm_cv:<12.4f} {xgb_test:<10.4f} {xgb_cv:<12.4f} {avg_score:<12.4f} ⭐")
                    gefa_performance = avg_score
                else:
                    print(f"   {method_name:<12} {rf_test:<10.4f} {rf_cv:<12.4f} {svm_test:<10.4f} {svm_cv:<12.4f} {xgb_test:<10.4f} {xgb_cv:<12.4f} {avg_score:<12.4f}")
            else:
                avg_score = np.mean([rf_cv, svm_cv])
                
                # Highlight GEFA results
                if method_name == "GEFA":
                    print(f"🧠 {method_name:<12} {rf_test:<10.4f} {rf_cv:<12.4f} {svm_test:<10.4f} {svm_cv:<12.4f} {avg_score:<12.4f} ⭐")
                    gefa_performance = avg_score
                else:
                    print(f"   {method_name:<12} {rf_test:<10.4f} {rf_cv:<12.4f} {svm_test:<10.4f} {svm_cv:<12.4f} {avg_score:<12.4f}")
            
            best_scores.append(avg_score)
        else:
            best_scores.append(0)
            print(f"❌ {method_name:<12} Failed to complete")

    # Find best method and compare with GEFA
    if best_scores:
        best_method_idx = np.argmax(best_scores)
        best_method = all_methods_results[best_method_idx][0]
        best_score = best_scores[best_method_idx]
        
        print(f"\n🏆 BEST OVERALL METHOD: {best_method} (Score: {best_score:.4f})")
        
        # GEFA vs Linear Methods Analysis
        if gefa_performance is not None:
            linear_methods = [name for name, _, _ in all_methods_results if name in ['Lasso', 'Ridge', 'Elastic Net']]
            linear_scores = [score for (name, _, _), score in zip(all_methods_results, best_scores) if name in linear_methods]
            
            if linear_scores:
                best_linear_score = max(linear_scores)
                best_linear_method = linear_methods[linear_scores.index(best_linear_score)]
                
                print(f"\n🎯 GEFA vs BEST LINEAR METHOD:")
                print(f"   🧠 GEFA Score: {gefa_performance:.4f}")
                print(f"   📏 Best Linear ({best_linear_method}): {best_linear_score:.4f}")
                improvement = ((gefa_performance - best_linear_score) / best_linear_score) * 100 if best_linear_score > 0 else 0
                if improvement > 0:
                    print(f"   ✅ GEFA is {improvement:.2f}% better than best linear method!")
                else:
                    print(f"   ❌ GEFA is {abs(improvement):.2f}% worse than best linear method")
        else:
            print(f"\n⚠️ GEFA did not complete successfully - only linear methods available for comparison")
            linear_methods = [name for name, _, _ in all_methods_results if name in ['Lasso', 'Ridge', 'Elastic Net']]
            if linear_methods:
                print(f"📏 Available linear methods: {', '.join(linear_methods)}")
    else:
        best_method = "N/A"
        best_score = 0
        print(f"\n❌ No methods completed successfully!")

    # Enhanced visualization with GEFA Feature Graph
    print(f"\n📊 Creating comprehensive visualization...")
    
    # Skip visualization if no methods completed
    if not all_methods_results:
        print("⚠️ No methods completed successfully - skipping visualization")
        print("🔍 Check the error messages above to debug the issues")
        return
    
    try:
        plt.figure(figsize=(18, 12))
        
        # Plot 1: Sample ECG beats by class
        plt.subplot(3, 4, 1)
        for class_idx in range(min(len(class_names), num_classes)):
            class_beats = beats[labels == class_idx]
            if len(class_beats) > 0:
                plt.plot(class_beats[0], label=f'{class_names[class_idx]}', alpha=0.8)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('Sample ECG Beats by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: GEFA Feature Graph Visualization
        plt.subplot(3, 4, 2)
        if adj_matrix is not None and gefa_indices is not None:
            try:
                print("   Creating GEFA feature graph visualization...")
                # Create subgraph of selected features
                top_features = min(12, len(gefa_indices))  # Show up to 12 features
                selected_indices = gefa_indices[:top_features]
                selected_adj = adj_matrix[np.ix_(selected_indices, selected_indices)]
                
                # Create network positions in a circle
                n_nodes = len(selected_indices)
                angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
                radius = 1.0
                x_pos = radius * np.cos(angles)
                y_pos = radius * np.sin(angles)
                
                # Draw edges first (so they appear behind nodes)
                edge_count = 0
                for i in range(n_nodes):
                    for j in range(i+1, n_nodes):
                        if selected_adj[i, j] > 0:
                            plt.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]], 
                                    'gray', alpha=0.5, linewidth=1, zorder=1)
                            edge_count += 1
                
                # Draw nodes with importance-based sizing
                if gefa_scores is not None:
                    node_sizes = []
                    for idx in selected_indices:
                        size = 100 + (gefa_scores[idx] * 200)  # Scale based on importance
                        node_sizes.append(size)
                else:
                    node_sizes = [150] * n_nodes
                
                scatter = plt.scatter(x_pos, y_pos, c=range(n_nodes), s=node_sizes, 
                                    cmap='viridis', alpha=0.8, zorder=5, edgecolors='black')
                
                # Add feature labels
                for i, idx in enumerate(selected_indices):
                    feature_name = feature_names[idx]
                    # Truncate long feature names
                    if len(feature_name) > 10:
                        display_name = feature_name[:8] + "..."
                    else:
                        display_name = feature_name
                    
                    # Position labels outside the circle
                    label_radius = radius + 0.2
                    label_x = label_radius * np.cos(angles[i])
                    label_y = label_radius * np.sin(angles[i])
                    
                    plt.annotate(display_name, (label_x, label_y), 
                                ha='center', va='center', fontsize=7, 
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
                
                plt.title(f'GEFA Feature Graph\n({n_nodes} top features, {edge_count} edges)')
                plt.axis('equal')
                plt.axis('off')
                
                # Add colorbar for node importance
                if gefa_scores is not None:
                    cbar = plt.colorbar(scatter, ax=plt.gca(), shrink=0.8)
                    cbar.set_label('Feature Rank', rotation=270, labelpad=15)
                
                print(f"   ✅ Feature graph visualization created: {n_nodes} nodes, {edge_count} edges")
                
            except Exception as e:
                print(f"   ❌ Error creating feature graph visualization: {e}")
                plt.text(0.5, 0.5, f'Feature Graph\nVisualization Error:\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=8)
                plt.axis('off')
        else:
            print("   ⚠️ GEFA data not available for graph visualization")
            plt.text(0.5, 0.5, 'GEFA Feature Graph\nNot Available\n(GEFA did not complete)', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            plt.axis('off')

        # Plot 3: Method comparison (CV scores)
        plt.subplot(3, 4, 3)
        if all_methods_results:
            methods = [result[0] for result in all_methods_results if result[1]]
            rf_scores = [result[2].get('Random Forest', {}).get('mean', 0) for result in all_methods_results if result[1]]
            
            if methods and rf_scores:
                colors = ['red' if method == 'GEFA' else 'skyblue' for method in methods]
                bars = plt.bar(range(len(methods)), rf_scores, alpha=0.8, color=colors)
                plt.xlabel('Method')
                plt.ylabel('CV Accuracy')
                plt.title('Cross-Validation Performance\n(Random Forest)')
                plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars, rf_scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 4: Class distribution
        plt.subplot(3, 4, 4)
        class_counts = np.bincount(y)
        plt.bar(range(len(class_counts)), class_counts, alpha=0.8, color='lightgreen')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')

        # Plot 5: GEFA vs Linear Methods Feature Importance
        plt.subplot(3, 4, 5)
        if (gefa_scores is not None and gefa_indices is not None and 
            lasso_scores is not None and lasso_indices is not None):
            
            x_pos = np.arange(min(10, k))
            width = 0.35
            
            # GEFA importance (normalized)
            gefa_imp = gefa_scores[gefa_indices[:10]]
            gefa_imp_norm = gefa_imp / (np.max(gefa_imp) + 1e-8)
            
            # Best linear method importance (normalized)
            lasso_imp = lasso_scores[lasso_indices[:10]]
            lasso_imp_norm = lasso_imp / (np.max(lasso_imp) + 1e-8)
            
            plt.bar(x_pos - width/2, gefa_imp_norm, width, label='GEFA', alpha=0.8, color='red')
            plt.bar(x_pos + width/2, lasso_imp_norm, width, label='Lasso', alpha=0.8, color='blue')
            
            plt.xlabel('Feature Rank')
            plt.ylabel('Normalized Importance')
            plt.title('GEFA vs Lasso\nTop 10 Features')
            plt.legend()
            plt.xticks(x_pos, [f'F{i+1}' for i in range(len(x_pos))])
            plt.grid(True, alpha=0.3)

        # Plot 6: Performance comparison across classifiers
        plt.subplot(3, 4, 6)
        if all_methods_results:
            methods = [result[0] for result in all_methods_results if result[1]]
            classifiers = ['Random Forest', 'SVM']
            if XGBOOST_AVAILABLE:
                classifiers.append('XGBoost')
            
            gefa_scores_clf = []
            linear_scores_clf = []
            
            for clf in classifiers:
                gefa_score = next((result[2].get(clf, {}).get('mean', 0) 
                                 for result in all_methods_results if result[0] == 'GEFA'), 0)
                
                # Average of linear methods
                linear_scores = [result[2].get(clf, {}).get('mean', 0) 
                               for result in all_methods_results 
                               if result[0] in ['Lasso', 'Ridge', 'Elastic Net']]
                avg_linear = np.mean(linear_scores) if linear_scores else 0
                
                gefa_scores_clf.append(gefa_score)
                linear_scores_clf.append(avg_linear)
            
            x_pos = np.arange(len(classifiers))
            width = 0.35
            
            plt.bar(x_pos - width/2, gefa_scores_clf, width, label='GEFA', alpha=0.8, color='red')
            plt.bar(x_pos + width/2, linear_scores_clf, width, label='Avg Linear', alpha=0.8, color='blue')
            
            plt.xlabel('Classifier')
            plt.ylabel('CV Accuracy')
            plt.title('GEFA vs Linear Methods\nAcross Classifiers')
            plt.legend()
            plt.xticks(x_pos, classifiers)
            plt.grid(True, alpha=0.3)

        # Plot 7: Method ranking with GEFA highlighted
        plt.subplot(3, 4, 7)
        if all_methods_results and best_scores:
            methods = [result[0] for result in all_methods_results if result[1]]
            sorted_indices = np.argsort(best_scores)[::-1]
            sorted_methods = [methods[i] for i in sorted_indices]
            sorted_scores = [best_scores[i] for i in sorted_indices]
            
            colors = ['red' if method == 'GEFA' else 'lightblue' for method in sorted_methods]
            bars = plt.barh(range(len(sorted_methods)), sorted_scores, alpha=0.8, color=colors)
            plt.xlabel('Average CV Score')
            plt.title('Method Ranking\n(GEFA in Red)')
            plt.yticks(range(len(sorted_methods)), sorted_methods)
            plt.grid(True, alpha=0.3)
            
            # Add score labels
            for bar, score in zip(bars, sorted_scores):
                plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontsize=8)

        # Plot 8: Feature Graph Adjacency Matrix Heatmap
        plt.subplot(3, 4, 8)
        if adj_matrix is not None and gefa_indices is not None:
            try:
                print("   Creating adjacency matrix heatmap...")
                # Show adjacency matrix for top selected features
                top_n = min(15, len(gefa_indices))  # Show top 15 features max
                selected_indices = gefa_indices[:top_n]
                selected_adj = adj_matrix[np.ix_(selected_indices, selected_indices)]
                
                # Create heatmap
                im = plt.imshow(selected_adj, cmap='Blues', aspect='auto', interpolation='nearest')
                plt.colorbar(im, shrink=0.8)
                
                # Add feature names as labels (shortened)
                feature_labels = []
                for idx in selected_indices:
                    name = feature_names[idx]
                    if len(name) > 12:
                        label = name[:10] + ".."
                    else:
                        label = name
                    feature_labels.append(label)
                
                plt.xticks(range(top_n), feature_labels, rotation=45, ha='right', fontsize=7)
                plt.yticks(range(top_n), feature_labels, fontsize=7)
                plt.title(f'Feature Graph Adjacency Matrix\n(Top {top_n} GEFA Features)')
                
                # Add connection strength text annotations for small matrices
                if top_n <= 10:
                    for i in range(top_n):
                        for j in range(top_n):
                            if selected_adj[i, j] > 0:
                                plt.text(j, i, '●', ha="center", va="center", 
                                       color="red", fontsize=8, fontweight='bold')
                
                print(f"   ✅ Adjacency matrix heatmap created ({top_n}x{top_n})")
                
            except Exception as e:
                print(f"   ❌ Error creating adjacency matrix heatmap: {e}")
                plt.text(0.5, 0.5, f'Adjacency Matrix\nVisualization Error:\n{str(e)[:30]}...', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=8)
                plt.axis('off')
        else:
            print("   ⚠️ No GEFA data for adjacency matrix")
            plt.text(0.5, 0.5, 'Feature Graph\nAdjacency Matrix\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            plt.axis('off')

        # Plot 9: GEFA Graph Statistics and Analysis
        plt.subplot(3, 4, 9)
        if adj_matrix is not None:
            try:
                print("   Creating graph statistics visualization...")
                # Calculate comprehensive graph statistics
                node_degrees = np.sum(adj_matrix, axis=1)
                total_edges = np.sum(adj_matrix)
                n_features = len(feature_names)
                density = total_edges / (n_features * (n_features - 1)) if n_features > 1 else 0
                
                # Create degree distribution histogram
                plt.hist(node_degrees, bins=min(20, len(np.unique(node_degrees))), 
                        alpha=0.7, color='skyblue', edgecolor='black')
                plt.xlabel('Node Degree (# Connections)')
                plt.ylabel('Number of Features')
                plt.title(f'Feature Connectivity Distribution\n(Density: {density:.3f})')
                plt.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = f"""
Avg: {np.mean(node_degrees):.1f}
Max: {np.max(node_degrees):.0f}
Min: {np.min(node_degrees):.0f}
Std: {np.std(node_degrees):.1f}"""
                
                plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=8)
                
                print(f"   ✅ Graph statistics visualization created")
                
            except Exception as e:
                print(f"   ❌ Error creating graph statistics: {e}")
                # Fallback to text-based statistics
                if adj_matrix is not None:
                    node_degrees = np.sum(adj_matrix, axis=1)
                    stats_text = f"""GEFA Graph Statistics

Total Features: {len(feature_names)}
Selected Features: {k}
Graph Edges: {np.sum(adj_matrix):.0f}
Graph Density: {np.sum(adj_matrix) / (len(feature_names) * (len(feature_names) - 1)):.3f}

Avg Node Degree: {np.mean(node_degrees):.1f}
Max Node Degree: {np.max(node_degrees):.0f}
Min Node Degree: {np.min(node_degrees):.0f}

Most Connected:
{feature_names[np.argmax(node_degrees)][:20]}"""
                    
                    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                            fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
                    plt.axis('off')
        else:
            print("   ⚠️ No adjacency matrix for graph statistics")
            plt.text(0.5, 0.5, 'GEFA Graph\nStatistics\nNot Available\n(GEFA incomplete)', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            plt.axis('off')

        # Plot 10: GEFA Performance Summary
        plt.subplot(3, 4, 10)
        if gefa_performance is not None:
            performance_text = f"""🧠 GEFA PERFORMANCE

Overall Score: {gefa_performance:.4f}

GEFA Components:
✅ Graph Neural Network
✅ Attention Mechanism  
✅ Feature Graph Construction
✅ GNNExplainer Enhancement

Training:
• 50 epochs (full training)
• Correlation-based graph
• Batch size: 16
• Hidden dim: 32

Selected Features:
{gefa_features[:3] if gefa_features else ['N/A']}..."""
            
            plt.text(0.05, 0.95, performance_text, transform=plt.gca().transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))
            plt.axis('off')
        else:
            plt.text(0.5, 0.5, 'GEFA Results\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')

        # Plot 11: Linear Methods Summary
        plt.subplot(3, 4, 11)
        linear_summary = f"""📏 LINEAR METHODS

Methods Tested:
• Lasso (L1 regularization)
• Ridge (L2 regularization)  
• Elastic Net (L1+L2)

Cross-Validation: 5-fold
Automatic hyperparameter selection

Performance:"""
        
        if all_methods_results:
            for method_name, _, cv_results in all_methods_results:
                if method_name in ['Lasso', 'Ridge', 'Elastic Net']:
                    score = cv_results.get('Random Forest', {}).get('mean', 0)
                    linear_summary += f"\n• {method_name}: {score:.3f}"
        
        plt.text(0.05, 0.95, linear_summary, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.axis('off')

        # Plot 12: Overall Summary (FIXED f-string formatting)
        plt.subplot(3, 4, 12)
        
        # FIXED: Format the GEFA score properly
        if gefa_performance is not None:
            gefa_score_text = f"{gefa_performance:.4f}"
        else:
            gefa_score_text = "N/A"
        
        summary_text = f"""📊 OVERALL SUMMARY

Dataset: {len(beats):,} beats
Features: {num_features}
Classes: {len(class_names)}

🏆 Best Method: {best_method}
🏆 Best Score: {best_score:.4f}

🧠 GEFA Score: {gefa_score_text}

Methods Compared: {len(all_methods_results)}
• GEFA (GNN-based)
• Lasso, Ridge, Elastic Net
• Random Forest baseline

Technology:
• GNNExplainer: {'✅' if GNNEXPLAINER_AVAILABLE else '❌'}
• XGBoost: {'✅' if XGBOOST_AVAILABLE else '❌'}"""

        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('gefa_vs_linear_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

    # FIXED: Final summary with proper formatting
    print(f"\n🎉 GEFA vs Linear Methods Analysis Complete!")
    print(f"📊 Results saved to 'gefa_vs_linear_fixed.png'")
    print(f"📋 Summary: {num_features} features from {len(beats):,} ECG beats")
    print(f"🏆 Overall best method: {best_method} (Score: {best_score:.4f})")
    print(f"📈 Classes: {class_names}")
    print(f"🔬 GNNExplainer: {'Available & Used (ULTRA-FAST)' if GNNEXPLAINER_AVAILABLE else 'Not Available'}")
    print(f"🚀 XGBoost: {'Available & Used' if XGBOOST_AVAILABLE else 'RF fallback used'}")
    print(f"⚡ Total methods compared: {len(all_methods_results)}")
    
    # GEFA-specific results summary
    if gefa_performance is not None:
        print(f"\n🧠 GEFA PERFORMANCE ANALYSIS:")
        print(f"   GEFA Score: {gefa_performance:.4f}")
        
        if adj_matrix is not None:
            graph_density = np.sum(adj_matrix) / (len(feature_names) * (len(feature_names) - 1))
            print(f"   Feature Graph Edges: {np.sum(adj_matrix):.0f}")
            print(f"   Feature Graph Density: {graph_density:.4f}")
        
        if gefa_features:
            print(f"   Top GEFA Features: {gefa_features[:5]}")
    else:
        print(f"\n⚠️ GEFA did not complete - check error messages above")
        print(f"📏 Analysis completed with linear methods only")
    
    print(f"\n📊 FINAL RANKING:")
    if all_methods_results and best_scores:
        methods = [result[0] for result in all_methods_results if result[1]]
        sorted_indices = np.argsort(best_scores)[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            method = methods[idx]
            score = best_scores[idx]
            if method == "GEFA":
                medal = f"🧠 {rank}."
            else:
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            print(f"   {medal} {method}: {score:.4f}")
    
    # Feature graph analysis summary
    if adj_matrix is not None and gefa_features is not None:
        print(f"\n🕸️ GEFA FEATURE GRAPH INSIGHTS:")
        node_degrees = np.sum(adj_matrix, axis=1)
        most_connected_idx = np.argmax(node_degrees)
        total_edges = np.sum(adj_matrix)
        density = total_edges / (len(feature_names) * (len(feature_names) - 1)) if len(feature_names) > 1 else 0
        
        print(f"   📊 Graph Structure:")
        print(f"      Total features: {len(feature_names)}")
        print(f"      Total edges: {total_edges:.0f}")
        print(f"      Graph density: {density:.4f}")
        print(f"      Average connections per feature: {np.mean(node_degrees):.1f}")
        
        print(f"   🔗 Feature Connectivity:")
        print(f"      Most connected: {feature_names[most_connected_idx]} ({node_degrees[most_connected_idx]:.0f} connections)")
        
        # Show top 3 most connected features
        top_connected = np.argsort(node_degrees)[-3:][::-1]
        print(f"      Top connected features:")
        for i, idx in enumerate(top_connected, 1):
            print(f"         {i}. {feature_names[idx][:30]} ({node_degrees[idx]:.0f} connections)")
        
        print(f"   🎯 GEFA Selection:")
        print(f"      Selected top features: {gefa_features[:5]}")
        print(f"      Graph captures feature relationships for better selection!")
    else:
        print(f"\n⚠️ GEFA Feature Graph not available - GEFA did not complete successfully")

if __name__ == "__main__":
    main()
# %%