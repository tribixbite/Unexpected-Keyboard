# 🚀 Neural Model Quantization Implementation Plan for Samsung S25U

## 📊 Current Performance Analysis

### ✅ Optimization Progress
- **Baseline**: 2.4-19 seconds per prediction (unacceptable)
- **Current Optimized**: 3.5-8.5 seconds (5x improvement achieved)
- **Next Target**: <1 second with proper NPU utilization  
- **Final Target**: <500ms with batched quantized models

### 🔍 Critical Insight from Expert Review
**QNN HTP backend REQUIRES quantized models** - current FP32 models cannot utilize Snapdragon NPU effectively.

---

## 🎯 Priority 1: Static Quantization Pipeline

### **Critical Requirements for Samsung S25U QNN:**
- **Data Types**: QNN supports uint8 and uint16 (NOT int8)
- **Calibration**: Representative swipe gesture dataset required
- **Format**: S8S8 with QDQ (QuantizeLinear/DequantizeLinear) operations
- **Architecture**: Transformer-specific attention quantization

### **Implementation Strategy:**

#### Phase 1: Calibration Data Collection ✅ (Already Have)
```python
# Use existing SwipeMLData from calibration activity
# Load from: /data/data/juloo.keyboard2.debug/databases/swipe_ml_data.db
# Format: Trajectory features [x,y,vx,vy,ax,ay] + nearest keys
```

#### Phase 2: Static Quantization Script
```python
# quantize_transformer_for_qnn.py
import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import numpy as np
import json

class SwipeCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_data_path, model_input_names):
        super().__init__()
        self.data = self.load_swipe_calibration_data(calibration_data_path)
        self.input_names = model_input_names
        self.current_index = 0
    
    def load_swipe_calibration_data(self, data_path):
        # Load SwipeMLData from exported JSON
        with open(data_path, 'r') as f:
            swipe_data = [json.loads(line) for line in f]
        
        calibration_samples = []
        for sample in swipe_data[:100]:  # Use 100 representative samples
            # Extract trajectory features [150, 6]
            trajectory_features = self.process_trajectory(sample['trace_points'])
            # Extract nearest keys [150]  
            nearest_keys = self.process_nearest_keys(sample['key_sequence'])
            # Create source mask [150]
            src_mask = self.create_source_mask(len(sample['trace_points']))
            
            calibration_samples.append({
                'trajectory_features': trajectory_features,
                'nearest_keys': nearest_keys,
                'src_mask': src_mask
            })
        
        return calibration_samples
    
    def get_next(self):
        if self.current_index >= len(self.data):
            return None
        
        sample = self.data[self.current_index]
        self.current_index += 1
        
        return {
            'trajectory_features': sample['trajectory_features'].astype(np.float32),
            'nearest_keys': sample['nearest_keys'].astype(np.int64),
            'src_mask': sample['src_mask'].astype(bool)
        }

# Quantize encoder model
def quantize_encoder_for_qnn():
    calibration_reader = SwipeCalibrationDataReader(
        'exported_swipe_data.jsonl', 
        ['trajectory_features', 'nearest_keys', 'src_mask']
    )
    
    quantize_static(
        model_input="models/swipe_encoder.onnx",
        model_output="models/swipe_encoder_quantized_uint8.onnx",
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QUInt8,      # QNN prefers uint8
        activation_type=QuantType.QUInt8,  # QNN requirement
        extra_options={
            'ActivationSymmetric': True,    # Recommended for transformers
            'WeightSymmetric': True,        # Better performance
            'AddQDQPairToWeight': True,     # Required for QNN
            'QDQOpTypePerChannel': False    # Per-tensor quantization
        }
    )

# Quantize decoder model
def quantize_decoder_for_qnn():
    # Similar implementation for decoder with appropriate input names
    calibration_reader = SwipeCalibrationDataReader(
        'exported_swipe_data.jsonl',
        ['memory', 'target_tokens', 'src_mask', 'target_mask'] 
    )
    
    quantize_static(
        model_input="models/swipe_decoder.onnx",
        model_output="models/swipe_decoder_quantized_uint8.onnx",
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
```

---

## 🎯 Priority 2: Dynamic Batch Export + Quantization

### **Combined Optimization Strategy:**

```python
# export_optimized_models.py - Combined dynamic + quantization
import torch
import torch.onnx
from transformers import AutoModel, AutoTokenizer

def export_optimized_transformer():
    # Load trained model
    model = load_trained_swipe_model()
    
    # Export with dynamic batch dimensions
    dummy_inputs = create_dummy_inputs()
    
    torch.onnx.export(
        model,
        dummy_inputs,
        "models/swipe_decoder_dynamic.onnx",
        input_names=['memory', 'target_tokens', 'src_mask', 'target_mask'],
        output_names=['logits'],
        dynamic_axes={
            'memory': {0: 'batch_size'},           # [batch_size, 150, 256]
            'target_tokens': {0: 'batch_size'},    # [batch_size, seq_length]
            'src_mask': {0: 'batch_size'},         # [batch_size, 150]
            'target_mask': {0: 'batch_size'},      # [batch_size, seq_length]
            'logits': {0: 'batch_size'}            # [batch_size, seq_length, vocab_size]
        },
        opset_version=14,  # QNN compatibility
        do_constant_folding=True,
        verbose=False
    )
    
    # Then quantize the dynamic model
    quantize_dynamic_model_for_qnn("models/swipe_decoder_dynamic.onnx")

def quantize_dynamic_model_for_qnn(model_path):
    # Apply static quantization to the dynamic model
    # This creates the final production model: dynamic batch + quantized
    pass
```

---

## 🎯 Priority 3: Android Integration Updates

### **Model Loading Updates:**
```java
// Update OnnxSwipePredictor to load quantized models
private boolean loadQuantizedModels() {
    try {
        // Load quantized encoder
        byte[] encoderData = loadModelFromAssets("models/swipe_encoder_quantized_uint8.onnx");
        _encoderSession = _ortEnvironment.createSession(encoderData, createOptimizedSessionOptions("Encoder"));
        
        // Load quantized decoder with dynamic batch support
        byte[] decoderData = loadModelFromAssets("models/swipe_decoder_dynamic_quantized_uint8.onnx");
        _decoderSession = _ortEnvironment.createSession(decoderData, createOptimizedSessionOptions("Decoder"));
        
        logDebug("🎯 Quantized models loaded - NPU acceleration ready");
        return true;
    }
    catch (Exception e) {
        Log.e(TAG, "Failed to load quantized models", e);
        return false;
    }
}
```

### **Batch Processing Implementation:**
```java
// With dynamic batch decoder, implement true batch processing
private List<BeamSearchCandidate> runBatchedBeamSearch(OrtSession.Result encoderResults, 
                                                       OnnxTensor srcMaskTensor, 
                                                       SwipeTrajectoryProcessor.TrajectoryFeatures features) {
    OnnxTensor memory = (OnnxTensor) encoderResults.get(0);
    
    // Initialize all beams
    List<BeamSearchState> beams = initializeBeams();
    
    for (int step = 0; step < _maxLength; step++) {
        if (beams.isEmpty()) break;
        
        // CRITICAL: Batch all beams into single tensors
        OnnxTensor batchedTokens = createBatchedTokensTensor(beams);      // [beam_width, seq_length]
        OnnxTensor batchedMasks = createBatchedMasksTensor(beams);        // [beam_width, seq_length]
        OnnxTensor batchedMemory = replicateMemoryForBatch(memory, beams.size()); // [beam_width, 150, 256]
        
        // SINGLE DECODER INFERENCE for all beams (8x speedup)
        try (OrtSession.Result batchedResult = _decoderSession.run(Map.of(
            "memory", batchedMemory,
            "target_tokens", batchedTokens,
            "target_mask", batchedMasks,
            "src_mask", replicateSrcMaskForBatch(srcMaskTensor, beams.size())
        ))) {
            // Process batched results and update all beams
            beams = processBatchedResults(beams, batchedResult);
        }
        
        // Apply early termination and beam pruning
        if (shouldTerminateEarly(beams, step)) break;
        beams = pruneBeams(beams);
    }
    
    return convertBeamsToCanidates(beams);
}
```

---

## 📈 Expected Performance Impact

### **With Static Quantization (uint8):**
- **QNN NPU Utilization**: Order-of-magnitude speedup on Snapdragon HTP
- **Expected**: 1-2 second predictions (vs current 3.5-8.5s)

### **With Dynamic Batch + Quantization:**
- **Batch Processing**: 8x speedup from single decoder call vs 8 sequential calls
- **Combined NPU + Batching**: Target <200ms per prediction  
- **Production Ready**: Real-time neural swipe typing achieved

---

## 🛠️ Implementation Timeline

### **Week 1: Static Quantization**
1. Export calibration data from SwipeMLDataStore
2. Implement CalibrationDataReader for swipe gestures
3. Apply static quantization with uint8 for QNN compatibility
4. Test quantized model performance on Samsung S25U

### **Week 2: Dynamic Batch Export**  
1. Modify model export script for dynamic batch dimensions
2. Re-export encoder and decoder with [-1, ...] shapes
3. Quantize dynamic models with same calibration pipeline
4. Validate dynamic model loading and basic inference

### **Week 3: Batch Processing Implementation**
1. Implement batched tensor creation for beam search
2. Replace sequential decoder calls with single batched call
3. Handle batched result processing and beam updates
4. Test complete batched + quantized pipeline

### **Week 4: Performance Validation**
1. Comprehensive performance testing on Samsung S25U
2. Validate <500ms target achievement
3. Comparison testing: FP32 vs Quantized vs Batched
4. Production readiness validation

---

This comprehensive plan addresses the two critical bottlenecks identified by expert consultation: **model quantization for NPU utilization** and **batch processing for architectural efficiency**.