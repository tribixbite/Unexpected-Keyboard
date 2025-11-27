# quantize_transformer_for_qnn.py
import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import numpy as np
import json
import os

class SwipeCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_data_path, model_input_names, batch_size=1, max_seq_length=150):
        super().__init__()
        self.data = self.load_swipe_calibration_data(calibration_data_path)
        self.input_names = model_input_names
        self.current_index = 0
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def load_swipe_calibration_data(self, data_path):
        # Load SwipeMLData from exported JSON
        if not os.path.exists(data_path):
            print(f"Error: Calibration data not found at {data_path}")
            return []
        with open(data_path, 'r') as f:
            swipe_data = [json.loads(line) for line in f]
        
        calibration_samples = []
        for sample in swipe_data[:100]:  # Use 100 representative samples
            if 'path' not in sample or 'word' not in sample:
                continue
            # Extract trajectory features [max_seq_length, 6]
            trajectory_features = self.process_trajectory(sample['path'])
            # Extract nearest keys [max_seq_length]  
            nearest_keys = self.process_nearest_keys(sample['word'])
            # Create source mask [max_seq_length]
            src_mask = self.create_source_mask(len(sample['path']))
            
            calibration_samples.append({
                'trajectory_features': trajectory_features,
                'nearest_keys': nearest_keys,
                'src_mask': src_mask,
                'actual_length': np.array(len(sample['path']), dtype=np.int32)
            })
        
        return calibration_samples
    
    def process_trajectory(self, path):
        # Dummy processing - replace with actual feature extraction if needed
        features = np.zeros((self.max_seq_length, 6), dtype=np.float32)
        for i in range(min(len(path), self.max_seq_length)):
            features[i, 0] = path[i]['x']
            features[i, 1] = path[i]['y']
        return features

    def process_nearest_keys(self, word):
        # Dummy processing
        keys = np.zeros(self.max_seq_length, dtype=np.int64)
        return keys

    def create_source_mask(self, length):
        mask = np.ones(self.max_seq_length, dtype=bool)
        mask[:length] = False
        return mask

    def get_next(self):
        if self.current_index >= len(self.data):
            return None
        
        sample = self.data[self.current_index]
        self.current_index += 1
        
        input_dict = {
            'trajectory_features': np.expand_dims(sample['trajectory_features'], axis=0).astype(np.float32),
            'nearest_keys': np.expand_dims(sample['nearest_keys'], axis=0).astype(np.int64),
            'actual_length': sample['actual_length'].astype(np.int32)
        }

        # For decoder
        if 'memory' in self.input_names:
            input_dict['memory'] = np.random.rand(1, self.max_seq_length, 256).astype(np.float32)
            input_dict['target_tokens'] = np.zeros((1, 20), dtype=np.int64)


        # Filter to only the names the model expects
        final_inputs = {name: input_dict[name] for name in self.input_names if name in input_dict}

        return final_inputs

# Quantize encoder model
def quantize_encoder_for_qnn(source_model, dest_model):
    if not os.path.exists(source_model):
        print(f"Source model not found: {source_model}")
        return

    print("Quantizing encoder...")
    calibration_reader = SwipeCalibrationDataReader(
        'sample_data.ndjson', 
        ['trajectory_features', 'nearest_keys', 'actual_length']
    )
    
    quantize_static(
        model_input=source_model,
        model_output=dest_model,
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QUInt8,
        extra_options={
            'ActivationSymmetric': True,
            'WeightSymmetric': True,
            'AddQDQPairToWeight': True,
            'QDQOpTypePerChannel': False
        }
    )
    print("Encoder quantization complete.")

# Quantize decoder model
def quantize_decoder_for_qnn(source_model, dest_model):
    if not os.path.exists(source_model):
        print(f"Source model not found: {source_model}")
        return
        
    print("Quantizing decoder...")
    calibration_reader = SwipeCalibrationDataReader(
        'sample_data.ndjson',
        ['memory', 'target_tokens', 'actual_src_length'] 
    )
    
    quantize_static(
        model_input=source_model,
        model_output=dest_model,
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QUInt8,
        extra_options={
            'ActivationSymmetric': True,
            'WeightSymmetric': True, 
            'AddQDQPairToWeight': True,
            'QDQOpTypePerChannel': False
        }
    )
    print("Decoder quantization complete.")

if __name__ == '__main__':
    # These paths assume the script is run from the `ml_training` directory
    # The models are in the `assets/models` directory relative to the project root.
    # Adjust paths as necessary.
    project_root = os.path.join(os.path.dirname(__file__), '..')
    models_dir = os.path.join(project_root, 'assets', 'models')
    
    encoder_model = os.path.join(models_dir, 'swipe_encoder_android.onnx')
    quantized_encoder_model = os.path.join(models_dir, 'swipe_encoder_quantized_uint8.onnx')

    decoder_model = os.path.join(models_dir, 'swipe_decoder_android.onnx')
    quantized_decoder_model = os.path.join(models_dir, 'swipe_decoder_quantized_uint8.onnx')
    
    quantize_encoder_for_qnn(encoder_model, quantized_encoder_model)
    quantize_decoder_for_qnn(decoder_model, quantized_decoder_model)
    print("All models quantized.")
