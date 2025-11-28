# 🚀 ONNX Neural System Latency Optimization Plan

## 📊 Current Performance Analysis

### ✅ System Status
- **Accuracy**: 100% - All test words predicted correctly at rank 1
- **ONNX Models**: Both encoder (5.3MB) and decoder (7.2MB) loading successfully  
- **Pipeline**: Complete feature extraction → encoder → beam search → decoder working
- **⚠️ CRITICAL ISSUE**: Prediction latency 2.4-19 seconds (unacceptable)

### 🔍 Calibration Log Analysis
**From real device testing:**
- Model loading: ~260ms total (acceptable)
- Encoder inference: ~120ms (acceptable)
- **MAIN BOTTLENECK**: Decoder beam search steps taking 100-1000ms EACH
- Total prediction times:
  - 'her': 2.4s (5 beam steps)
  - 'old': 6.9s (4 beam steps) 
  - 'two': 11.0s (8 beam steps)
  - 'for': 12.3s (4 beam steps)
  - 'not': 9.8s (4 beam steps)
  - 'how': 19.4s (5 beam steps)

**Key Finding**: Reducing beam_size and max_tokens didn't significantly improve speed, indicating tensor operations and memory allocation are the bottleneck, not model complexity.

---

## 🎯 Priority 1: Critical Optimizations (2-5x speedup expected)

### 1. Memory Pre-allocation and Reuse ⚡
**Current Issue**: Creating new tensors for every beam search step
**Solution**: Pre-allocate tensor buffers and reuse them

```java
// Pre-allocate reusable tensors during initialization
private OnnxTensor _reusableTargetTokensTensor;
private OnnxTensor _reusableTargetMaskTensor; 
private FloatBuffer _reusableLogitsBuffer;

// Reuse instead of OnnxTensor.createTensor() in beam search
private void updateReusableTensors(BeamSearchState beam) {
    // Update tensor data in-place instead of creating new tensors
    _reusableTargetTokensTensor.getBuffer().clear();
    // ... update buffer data
}
```

### 2. ONNX Session Persistence 🧠
**Current Issue**: Models may be reloading between predictions
**Solution**: Keep sessions in memory permanently

```java
// Add session lifecycle management
private static OnnxSwipePredictor _singletonInstance;
private boolean _keepSessionsLoaded = true;

public static OnnxSwipePredictor getInstance(Context context) {
    if (_singletonInstance == null) {
        _singletonInstance = new OnnxSwipePredictor(context);
        _singletonInstance.initialize();
    }
    return _singletonInstance;
}

// Never close sessions unless explicitly requested
public void cleanup() {
    if (!_keepSessionsLoaded) {
        // Close sessions
    }
}
```

### 3. Batch Tensor Operations 📦
**Current Issue**: Processing beams sequentially in loops
**Solution**: Batch multiple beams into single tensor operations

```java
// Instead of: for each beam -> create tensor -> inference
// Do: create batched tensor for all beams -> single inference -> split results
private List<BeamSearchState> runBatchedBeamStep(List<BeamSearchState> beams, 
                                                  OnnxTensor memory, 
                                                  OnnxTensor srcMask) {
    int batchSize = beams.size();
    // Create batched tensors [batch_size, seq_len] instead of [1, seq_len]
    long[][] batchedTokens = new long[batchSize][decoderSeqLength];
    boolean[][] batchedMasks = new boolean[batchSize][decoderSeqLength];
    
    // ... populate batch data
    
    // Single decoder inference for all beams
    OnnxTensor batchedTargetTokens = OnnxTensor.createTensor(_ortEnvironment, batchedTokens);
    OnnxTensor batchedTargetMask = OnnxTensor.createTensor(_ortEnvironment, batchedMasks);
    
    // ... run inference once, process results
}
```

---

## 🎯 Priority 2: Web App Optimizations (2-3x speedup expected)

### 4. Vocabulary Filtering and Caching 📚
**Missing from Android**: The web app has sophisticated vocabulary optimization

```java
// Port from swipe-vocabulary.js
public class OptimizedVocabulary {
    private Map<String, Float> wordFreq;
    private Set<String> commonWords;      // Fast path for top words
    private Set<String> top5000;          // Pre-filtered vocabulary
    private Map<Integer, List<String>> wordsByLength;  // Length-based lookup
    
    // Fast filtering: check common words first, then top5000, then full vocabulary
    public List<PredictionCandidate> filterPredictions(List<BeamSearchCandidate> raw) {
        List<PredictionCandidate> filtered = new ArrayList<>();
        
        for (BeamSearchCandidate candidate : raw) {
            String word = candidate.word.toLowerCase();
            
            // Fast path for common words (biggest speedup)
            if (commonWords.contains(word)) {
                float freq = wordFreq.get(word);
                float score = calculateCombinedScore(candidate.confidence, freq, 1.2f);
                filtered.add(new PredictionCandidate(word, score, "common"));
                continue;
            }
            
            // Check top 5000 words
            if (top5000.contains(word)) {
                // ... similar logic
            }
            
            // Full vocabulary with frequency threshold
            // ...
        }
        
        return filtered;
    }
    
    // Combined NN confidence + word frequency scoring
    private float calculateCombinedScore(float confidence, float frequency, float boost) {
        float freqScore = (float)(Math.log10(frequency + 1e-10) / -10.0);
        return (0.6f * confidence + 0.4f * freqScore) * boost;
    }
}
```

### 5. Early Termination Strategies ⏹️
**Add smart stopping conditions**:

```java
// Stop beam search early if confidence is sufficient
private boolean shouldTerminateEarly(List<BeamSearchState> beams, int step) {
    if (step >= 2) { // Minimum steps for short words
        BeamSearchState best = beams.get(0);
        if (best.score > EARLY_TERMINATION_THRESHOLD && 
            best.tokens.size() >= 3 && 
            isValidWordEnding(best)) {
            return true;
        }
    }
    return false;
}

// Prune low-probability beams aggressively
private List<BeamSearchState> pruneBeams(List<BeamSearchState> beams, float threshold) {
    if (beams.size() <= 2) return beams; // Keep minimum beams
    
    float bestScore = beams.get(0).score;
    return beams.stream()
        .filter(beam -> (bestScore - beam.score) < threshold)
        .collect(Collectors.toList());
}
```

### 6. Reduced Model Precision 🎛️
**INT8 Quantization** (if not already applied):

```java
// During model loading, verify quantization is applied
private void loadOptimizedModels() {
    SessionOptions sessionOpts = new SessionOptions();
    
    // Enable all performance optimizations
    sessionOpts.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);
    sessionOpts.setIntraOpThreads(2); // Use 2 threads for tensor ops
    sessionOpts.setInterOpThreads(1);  // Single thread for sequential ops
    
    // Load with optimized options
    _encoderSession = _ortEnvironment.createSession(encoderModelData, sessionOpts);
    _decoderSession = _ortEnvironment.createSession(decoderModelData, sessionOpts);
}
```

---

## 🎯 Priority 3: System-Level Optimizations (1.5-2x speedup expected)

### 7. Async Processing with Proper Threading 🧵

```java
// Dedicated thread pool for ONNX operations
private ExecutorService _onnxExecutor = Executors.newSingleThreadExecutor(
    new ThreadFactory() {
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r, "ONNX-Inference");
            t.setPriority(Thread.NORM_PRIORITY + 1); // Slightly higher priority
            return t;
        }
    });

// Async prediction with cancellation support
public Future<PredictionResult> predictAsync(SwipeInput input) {
    return _onnxExecutor.submit(() -> predict(input));
}
```

### 8. Memory Pool Management 🏊

```java
// Pre-allocated buffer pools to avoid GC pressure
private Queue<FloatBuffer> _trajectoryBufferPool = new ConcurrentLinkedQueue<>();
private Queue<LongBuffer> _tokenBufferPool = new ConcurrentLinkedQueue<>();
private Queue<boolean[][]> _maskBufferPool = new ConcurrentLinkedQueue<>();

// Reuse buffers instead of allocating new ones
private FloatBuffer getTrajectoryBuffer() {
    FloatBuffer buffer = _trajectoryBufferPool.poll();
    if (buffer == null) {
        buffer = FloatBuffer.allocate(MAX_SEQUENCE_LENGTH * TRAJECTORY_FEATURES);
    }
    buffer.clear();
    return buffer;
}

private void returnTrajectoryBuffer(FloatBuffer buffer) {
    _trajectoryBufferPool.offer(buffer);
}
```

---

## 🎯 Priority 4: Advanced Optimizations (1.2-1.5x speedup expected)

### 9. Custom Tokenizer Optimization 🔤

```java
// Cache tokenized nearest keys to avoid repeated computation
private Map<Character, Integer> _charToTokenCache = new HashMap<>();
private int[] _reusableNearestKeysArray = new int[MAX_SEQUENCE_LENGTH];

// Optimized tokenization
public int[] tokenizeNearestKeys(List<Character> nearestKeys) {
    Arrays.fill(_reusableNearestKeysArray, PAD_IDX);
    
    for (int i = 0; i < Math.min(nearestKeys.size(), MAX_SEQUENCE_LENGTH); i++) {
        Character ch = nearestKeys.get(i);
        Integer token = _charToTokenCache.get(ch);
        if (token == null) {
            token = _tokenizer.charToIndex(ch);
            _charToTokenCache.put(ch, token);
        }
        _reusableNearestKeysArray[i] = token;
    }
    
    return _reusableNearestKeysArray;
}
```

### 10. Test Word Randomization 🎲
**For better calibration testing**:

```java
// Use full 150k vocabulary instead of small hardcoded list
private List<String> loadFullVocabulary() {
    try {
        InputStream vocabStream = _context.getAssets().open("dictionaries/en_full_150k.txt");
        // Load and return full vocabulary list
        return Arrays.asList(IOUtils.toString(vocabStream, "UTF-8").split("\n"));
    } catch (IOException e) {
        Log.e(TAG, "Failed to load full vocabulary", e);
        return getDefaultTestWords(); // fallback
    }
}

// Random word selection for calibration
private String getRandomTestWord() {
    if (_fullVocabulary == null) {
        _fullVocabulary = loadFullVocabulary();
    }
    return _fullVocabulary.get(_random.nextInt(_fullVocabulary.size()));
}
```

---

## 📈 Expected Performance Improvements

### Optimized Latency Targets:
- **Current**: 2.4-19 seconds per prediction
- **After Priority 1**: 0.5-3 seconds (5-6x improvement)
- **After Priority 2**: 0.2-1 seconds (additional 2-3x improvement)  
- **After Priority 3**: 0.1-0.5 seconds (additional 2x improvement)
- **Target**: **<200ms for real-time typing** (90%+ improvement)

### Implementation Order:
1. **Week 1**: Memory pre-allocation, session persistence, vocabulary caching
2. **Week 2**: Batch operations, early termination, threading optimization  
3. **Week 3**: Memory pools, tokenizer optimization, model quantization
4. **Week 4**: Testing, fine-tuning, deployment

---

## ⚡ Quick Wins (Immediate Implementation)

These can be implemented immediately for fast results:

### 1. Session Persistence (Expected: 2x speedup)
Keep ONNX sessions loaded in memory to prevent model reload overhead.

### 2. Vocabulary Caching (Expected: 1.5x speedup)  
Pre-load full vocabulary with frequency data for instant lookup.

### 3. Tensor Reuse (Expected: 3x speedup)
Stop creating new tensors for every beam search step - reuse pre-allocated buffers.

### 4. Early Termination (Expected: 2x speedup)
Stop beam search when high-confidence short words are found.

**Combined Expected Improvement: 5-10x faster predictions**
**Target: Reduce 10+ second predictions to 1-2 seconds immediately**

---

This comprehensive optimization plan addresses the root causes of latency identified in the calibration log and ports proven techniques from the web demo to achieve real-time neural swipe typing performance.