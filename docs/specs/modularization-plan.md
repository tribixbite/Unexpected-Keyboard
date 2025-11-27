# Modularization Plan: ONNX Cleanup + Java to Kotlin Refactoring

**Status**: PLANNING
**Priority**: MEDIUM (Post-bug-fix cleanup)
**Estimated Time**: 4-6 hours
**Risk Level**: MEDIUM

---

## Executive Summary

**Goals**:
1. **Reduce APK size by ~20 MB** - Remove duplicate/old ONNX model files
2. **Improve maintainability** - Split large Java files into focused Kotlin modules
3. **Follow Kotlin best practices** - Use existing patterns from the codebase
4. **Maintain stability** - All 672 tests must pass, no functionality loss

**Approach**: Two-phase hybrid strategy
- **Phase 1** (Quick): ONNX file cleanup - 15 minutes
- **Phase 2** (Deep): Modularize OnnxSwipePredictor.java - 3-5 hours

---

## Phase 1: ONNX File Cleanup (READY TO EXECUTE)

### Current State
```
assets/models/
├── swipe_encoder_android.onnx        (5.1M) - v2 float32, DUPLICATE
├── swipe_decoder_android.onnx        (4.6M) - v2 float32, DUPLICATE
├── bs/
│   ├── swipe_encoder_android.onnx    (5.0M) - v2 INT8, OLD
│   └── swipe_decoder_android.onnx    (5.1M) - v2 INT8, OLD
└── bs2/
    ├── swipe_encoder_android.onnx    (5.1M) - v2 INT8 calibrated, KEEP
    └── swipe_decoder_android.onnx    (4.8M) - v2 INT8 calibrated, KEEP

Total: 30.6 MB (19.8 MB are duplicates/old versions)
```

### Target State
```
assets/models/
├── swipe_encoder_android.onnx        (5.1M) - Moved from bs2/
└── swipe_decoder_android.onnx        (4.8M) - Moved from bs2/

Total: 9.9 MB (saved 20.7 MB)
```

### Execution Steps

**Step 1: Backup current models** (safety)
```bash
cp -r assets/models assets/models.backup
```

**Step 2: Move bs2 models to parent**
```bash
mv assets/models/bs2/swipe_encoder_android.onnx assets/models/swipe_encoder_android_new.onnx
mv assets/models/bs2/swipe_decoder_android.onnx assets/models/swipe_decoder_android_new.onnx
```

**Step 3: Remove old models**
```bash
rm assets/models/swipe_encoder_android.onnx
rm assets/models/swipe_decoder_android.onnx
rm -rf assets/models/bs/
rm -rf assets/models/bs2/
```

**Step 4: Rename new models**
```bash
mv assets/models/swipe_encoder_android_new.onnx assets/models/swipe_encoder_android.onnx
mv assets/models/swipe_decoder_android_new.onnx assets/models/swipe_decoder_android.onnx
```

**Step 5: Update code references**

File: `srcs/juloo.keyboard2/OnnxSwipePredictor.java`

Current (lines 243-254):
```java
boolean useQuantized = (_config != null && _config.neural_use_quantized);

if (useQuantized)
{
    // INT8 quantized models with broadcast support (calibrated, v2)
    encoderPath = "models/bs2/swipe_encoder_android.onnx";
    decoderPath = "models/bs2/swipe_decoder_android.onnx";
    _maxSequenceLength = 250;
    _modelAccuracy = "73.4%";
    _modelSource = "builtin-quantized-v2";
    Log.i(TAG, "Loading v2 INT8 quantized models (calibrated, broadcast-enabled, XNNPACK-optimized)");
}
```

Change to:
```java
boolean useQuantized = (_config != null && _config.neural_use_quantized);

if (useQuantized)
{
    // INT8 quantized models (default - now in models/ root)
    encoderPath = "models/swipe_encoder_android.onnx";
    decoderPath = "models/swipe_decoder_android.onnx";
    _maxSequenceLength = 250;
    _modelAccuracy = "73.4%";
    _modelSource = "builtin-quantized-v2";
    Log.i(TAG, "Loading v2 INT8 quantized models (calibrated, broadcast-enabled, XNNPACK-optimized)");
}
```

**Note**: The float32 fallback path (lines 257-265) already points to "models/swipe_encoder_android.onnx", which will now be the INT8 version. Need to decide:
- **Option A**: Remove float32 support entirely (quantized only)
- **Option B**: Keep quantized models, remove toggle (always use INT8)
- **Option C**: Add float32 models back if needed

**Recommendation**: Option B - Remove the toggle, always use INT8 quantized models (smaller, faster, 73.4% accuracy is sufficient)

**Step 6: Build and test**
```bash
./build-on-termux.sh
# Install and test swipe typing
# Verify logcat shows correct model paths
```

**Step 7: Cleanup backup**
```bash
rm -rf assets/models.backup
```

### Success Criteria
- ✅ APK size reduced by ~20 MB
- ✅ Swipe typing still works
- ✅ Logcat shows: "Loading v2 INT8 quantized models"
- ✅ Model paths are "models/swipe_encoder_android.onnx"
- ✅ All tests pass

---

## Phase 2: OnnxSwipePredictor Modularization

### Current State Analysis

**OnnxSwipePredictor.java** (2,723 lines) - Responsibilities:
1. **Model Loading** (lines 205-560) - ONNX session initialization, path management
2. **Inference Execution** (lines 580-950) - Encoder/decoder tensor operations
3. **Beam Search** (lines 950-1180) - Decoding logic, vocabulary filtering
4. **Trajectory Processing** (lines 1180-1280) - Coordinate normalization, resampling
5. **Configuration Management** (lines 1280-1400) - setConfig, parameter updates
6. **Memory Management** (lines 1400-2100) - Buffer pools, tensor reuse, cleanup
7. **Utility Methods** (lines 2100-2723) - Helper functions, debugging, statistics

### Target Module Structure

Following existing Kotlin patterns (`PredictionViewSetup.kt`, `ConfigPropagator.kt`, etc.):

```
srcs/juloo.keyboard2/onnx/
├── OnnxModelLoader.kt              (~300 lines)
│   - Model path resolution
│   - ONNX session creation
│   - Model configuration loading
│   - Thread-safe initialization (synchronized)
│
├── OnnxInferenceEngine.kt          (~400 lines)
│   - Encoder execution
│   - Decoder execution
│   - Tensor input/output management
│   - Session cleanup
│
├── OnnxBeamSearchDecoder.kt        (~350 lines)
│   - Beam search algorithm
│   - Vocabulary filtering
│   - Top-K prediction
│   - Confidence scoring
│
├── OnnxMemoryManager.kt            (~400 lines)
│   - Buffer pooling
│   - Tensor reuse
│   - Memory cleanup
│   - Performance optimization
│
├── OnnxConfigManager.kt            (~200 lines)
│   - Configuration updates
│   - Model version switching
│   - Parameter management
│   - Thread-safe setConfig
│
└── OnnxSwipePredictor.kt           (~300 lines)
    - Facade/Coordinator
    - Public API
    - Delegates to above modules
    - Maintains backward compatibility
```

### Modularization Strategy

**Pattern**: Delegate Pattern (following existing `PredictionCoordinator` style)

**Step 1: Create module interfaces**
```kotlin
// Define contracts for each module
interface ModelLoader {
    fun initialize(): Boolean
    fun getModelPath(version: String): Pair<String, String>
    fun cleanup()
}

interface InferenceEngine {
    fun encode(input: FloatArray): FloatArray
    fun decode(encoderOutput: FloatArray, decoderInput: IntArray): FloatArray
}

// etc.
```

**Step 2: Extract modules one at a time**
1. Create `OnnxModelLoader.kt` - Extract initialization logic
2. Create `OnnxInferenceEngine.kt` - Extract tensor operations
3. Create `OnnxBeamSearchDecoder.kt` - Extract decoding logic
4. Create `OnnxMemoryManager.kt` - Extract pooling logic
5. Create `OnnxConfigManager.kt` - Extract config logic
6. Refactor `OnnxSwipePredictor` to delegate

**Step 3: Maintain thread safety**
- Keep `synchronized` on public methods in `OnnxSwipePredictor`
- Internal modules can be unsynchronized (protected by outer layer)
- Maintain `volatile` for `_isInitialized` flag

**Step 4: Preserve singleton pattern**
- Existing code uses singleton for memory optimization
- Keep singleton in `OnnxSwipePredictor`
- Modules are instance-scoped, owned by predictor

### Testing Strategy

**For each module extraction**:
1. Extract module code to new Kotlin file
2. Update `OnnxSwipePredictor` to use module
3. Build (must succeed)
4. Run tests (all must pass)
5. Test swipe typing manually
6. Commit with detailed message

**Test Coverage**:
- Unit tests for each module (new files in `test/`)
- Integration test for full prediction pipeline
- Smoke test for swipe typing functionality

### Kotlin Conversion Guidelines

**Follow existing patterns**:
```kotlin
// 1. Data classes for DTOs
data class ModelConfig(
    val encoderPath: String,
    val decoderPath: String,
    val maxSequenceLength: Int,
    val accuracy: String
)

// 2. Companion objects for factory methods
companion object {
    @JvmStatic
    fun create(context: Context, config: Config): OnnxModelLoader {
        return OnnxModelLoader(context, config)
    }
}

// 3. Extension functions for helpers
private fun FloatArray.normalize(): FloatArray {
    // ...
}

// 4. Null safety with ?.let
session?.let { sess ->
    sess.close()
}

// 5. when expressions instead of switch
when (modelVersion) {
    "v2" -> loadV2Models()
    "custom" -> loadCustomModels()
    else -> throw IllegalArgumentException()
}
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Breaking model loading | LOW | HIGH | Test after each module extraction |
| Losing thread safety | LOW | HIGH | Keep synchronized in facade |
| Performance regression | MEDIUM | MEDIUM | Benchmark before/after |
| Too many small files | LOW | LOW | Follow SRP, limit to 6 modules |
| Breaking backward compat | LOW | HIGH | Keep public API identical |

---

## Timeline Estimate

**Phase 1: ONNX Cleanup**
- Planning: 10 min (done)
- Execution: 10 min
- Testing: 5 min
- **Total: 25 minutes**

**Phase 2: OnnxSwipePredictor Modularization**
- Module 1 (ModelLoader): 45 min
- Module 2 (InferenceEngine): 60 min
- Module 3 (BeamSearchDecoder): 45 min
- Module 4 (MemoryManager): 60 min
- Module 5 (ConfigManager): 30 min
- Module 6 (Refactor facade): 45 min
- Testing each: 15 min × 6 = 90 min
- **Total: 5 hours 15 minutes**

**GRAND TOTAL**: ~5.5 hours

---

## Decision Points

### AWAITING USER CONFIRMATION:

**Question 1**: ONNX Model Strategy
- [ ] **Option A**: Keep only INT8 quantized, remove float32 entirely
- [ ] **Option B**: Keep both, but make INT8 default (current)
- [ ] **Option C**: Add float32 back later if needed

**Recommendation**: Option A (simpler, smaller APK, sufficient accuracy)

**Question 2**: Modularization Scope
- [ ] **Option A**: Only OnnxSwipePredictor (recommended for now)
- [ ] **Option B**: OnnxSwipePredictor + SettingsActivity
- [ ] **Option C**: All 8 large Java files (too aggressive)

**Recommendation**: Option A (focused, manageable, high-value)

**Question 3**: Module Granularity
- [ ] **Option A**: 6 modules as specified above
- [ ] **Option B**: Fewer modules (3-4, larger files)
- [ ] **Option C**: More modules (8-10, very small files)

**Recommendation**: Option A (balanced, follows SRP)

---

## Ready to Execute?

When user says **"go"**, execute in this order:

1. **Immediate**: Phase 1 (ONNX cleanup) - 25 minutes
2. **Wait for user confirmation**: Show results, get approval
3. **Then**: Phase 2 Module 1 (ModelLoader extraction)
4. **Then**: Modules 2-6 sequentially with testing after each

**Note**: User can stop at any point. Each module is independently committable.

---

## Documentation Updates Needed

After completion:
- [ ] Update `memory/pm.md` with refactoring status
- [ ] Create `docs/specs/onnx-architecture.md` (module diagram)
- [ ] Update `CLAUDE.md` with new module locations
- [ ] Update `README.md` with file structure

---

**Status**: ⏸️ AWAITING USER "GO" COMMAND
