# Complete Web Demo Neural Flow Documentation

## 1. ENCODER PHASE

### Input Preparation:
```javascript
// Constants
const MAX_SEQUENCE_LENGTH = 150;
const trajectoryData = new Float32Array(MAX_SEQUENCE_LENGTH * 6);
const nearestKeysData = new BigInt64Array(MAX_SEQUENCE_LENGTH);
const srcMaskData = new Uint8Array(MAX_SEQUENCE_LENGTH);

// Fill trajectory features
for (let i = 0; i < MAX_SEQUENCE_LENGTH; i++) {
    const point = features.path[i];
    const baseIdx = i * 6;
    
    // Position (normalized 0-1)
    trajectoryData[baseIdx + 0] = point.x;
    trajectoryData[baseIdx + 1] = point.y;
    
    // Velocity (difference from previous point)
    if (i > 0) {
        const prevPoint = features.path[i - 1];
        trajectoryData[baseIdx + 2] = point.x - prevPoint.x; // vx
        trajectoryData[baseIdx + 3] = point.y - prevPoint.y; // vy
    }
    
    // Acceleration (difference of velocities)
    if (i > 1) {
        const prevVx = trajectoryData[(i-1) * 6 + 2];
        const prevVy = trajectoryData[(i-1) * 6 + 3];
        trajectoryData[baseIdx + 4] = trajectoryData[baseIdx + 2] - prevVx; // ax
        trajectoryData[baseIdx + 5] = trajectoryData[baseIdx + 3] - prevVy; // ay
    }
    
    // Nearest key mapping
    nearestKeysData[i] = BigInt(keyMap[point.key] || 0);
    
    // Source mask: 1 for padded, 0 for real
    srcMaskData[i] = i >= actualLength ? 1 : 0;
}
```

### Tensor Creation:
```javascript
const trajectoryTensor = new ort.Tensor('float32', trajectoryData, [1, 150, 6]);
const nearestKeysTensor = new ort.Tensor('int64', nearestKeysData, [1, 150]);
const srcMaskTensor = new ort.Tensor('bool', srcMaskData, [1, 150]);
```

### Encoder Inference:
```javascript
const encoderOutput = await encoderSession.run({ 
    trajectory_features: trajectoryTensor,
    nearest_keys: nearestKeysTensor,
    src_mask: srcMaskTensor
});
// Result: memory tensor with shape matching encoder output
```

## 2. DECODER PHASE (BEAM SEARCH)

### Beam Initialization:
```javascript
let beams = [{
    tokens: [BigInt(SOS_IDX)],  // Start with SOS token (2)
    score: 0,
    finished: false
}];
```

### For Each Beam Search Step:
```javascript
for (let step = 0; step < maxLength; step++) {
    const candidates = [];
    
    for (const beam of beams) {
        if (beam.finished) {
            candidates.push(beam);
            continue;
        }
        
        // Pad tokens to fixed length (20)
        const DECODER_SEQ_LENGTH = 20;
        const paddedTokens = new BigInt64Array(DECODER_SEQ_LENGTH);
        
        for (let i = 0; i < beam.tokens.length && i < DECODER_SEQ_LENGTH; i++) {
            paddedTokens[i] = beam.tokens[i];
        }
        for (let i = beam.tokens.length; i < DECODER_SEQ_LENGTH; i++) {
            paddedTokens[i] = BigInt(PAD_IDX);
        }
        
        // Create target mask: 1 for padded positions
        const tgtMask = new Uint8Array(DECODER_SEQ_LENGTH);
        for (let i = beam.tokens.length; i < DECODER_SEQ_LENGTH; i++) {
            tgtMask[i] = 1;
        }
        
        // Source mask: all zeros (no encoder padding)
        const srcMaskArray = new Uint8Array(memory.dims[1]);
        srcMaskArray.fill(0);
        
        // Run decoder
        const decoderOutput = await decoderSession.run({
            memory: memory,
            target_tokens: new ort.Tensor('int64', paddedTokens, [1, DECODER_SEQ_LENGTH]),
            target_mask: new ort.Tensor('bool', tgtMask, [1, DECODER_SEQ_LENGTH]),
            src_mask: new ort.Tensor('bool', srcMaskArray, [1, memory.dims[1]])
        });
        
        // Get logits - CRITICAL: shape is [1, DECODER_SEQ_LENGTH, 30]
        const logits = decoderOutput.logits;
        const logitsData = logits.data;  // Flattened array access!
        const vocabSize = 30;
        
        // Extract logits for next token position
        const tokenPosition = Math.min(beam.tokens.length - 1, DECODER_SEQ_LENGTH - 1);
        const startIdx = tokenPosition * vocabSize;
        const endIdx = startIdx + vocabSize;
        const relevantLogits = logitsData.slice(startIdx, endIdx);
        
        // Apply softmax and get top tokens
        const probs = softmax(Array.from(relevantLogits));
        const topK = getTopKIndices(probs, beamWidth);
        
        for (const idx of topK) {
            const newBeam = {
                tokens: [...beam.tokens, BigInt(idx)],
                score: beam.score + Math.log(probs[idx]),
                finished: idx === EOS_IDX
            };
            candidates.push(newBeam);
        }
    }
    
    // Select top beams and continue
    candidates.sort((a, b) => b.score - a.score);
    beams = candidates.slice(0, beamWidth);
    
    // Early termination
    if (beams.every(b => b.finished) || 
        (step >= 10 && beams.filter(b => b.finished).length >= 3)) {
        break;
    }
}
```

## 3. TOKEN TO WORD CONVERSION

```javascript
const predictions = beams.map(beam => {
    const chars = [];
    for (const token of beam.tokens) {
        const idx = Number(token);
        if (idx === SOS_IDX || idx === EOS_IDX || idx === PAD_IDX) continue;
        if (idxToChar[idx] && !idxToChar[idx].startsWith('<')) {
            chars.push(idxToChar[idx]);
        }
    }
    return chars.join('');
}).filter(word => word.length > 0);
```

## KEY INSIGHTS:

1. **Logits Access**: `logits.data` - flattened array, NOT 3D array!
2. **Tensor Shapes**: [1, 20, 30] but accessed as flat array with indexing
3. **Source Mask**: All zeros for decoder (no encoder padding assumed)
4. **Token Padding**: Fixed 20-length sequences with proper masking
5. **Beam Search**: Standard implementation with score tracking
6. **Word Generation**: Filter out special tokens, join characters

The critical error in my implementation: I'm trying to cast to float[][][] but web demo accesses as flattened data array!