# OnnxSwipePredictor Refactoring Specification

**Status**: IN PROGRESS
**Current**: Analysis Phase
**Target**: Modular Kotlin-based Architecture

---

## Current State Analysis

### File: `srcs/juloo.keyboard2/OnnxSwipePredictor.java`
- **Lines**: 2,708
- **Public Methods**: 23
- **Private Methods/Fields**: 87
- **Dependencies**: Heavy ONNX Runtime usage, direct tensor manipulation

### Existing Kotlin Modules (Already Created)
Located in `srcs/juloo.keyboard2/onnx/`:
1. ✅ `ModelLoader.kt` - Model file loading, session creation
2. ✅ `EncoderWrapper.kt` - Encoder tensor operations
3. ✅ `DecoderWrapper.kt` - Decoder tensor operations
4. ✅ `BeamSearchEngine.kt` - Beam search decoding logic
5. ✅ `TensorFactory.kt` - Tensor creation utilities
6. ✅ `MemoryPool.kt` - Buffer pooling for GC reduction
7. ✅ `BroadcastSupport.kt` - Broadcast model configuration

**Current Problem**: These modules exist but are NOT being used by OnnxSwipePredictor.java

---

## Refactoring Strategy

### Approach: Incremental Integration (Conservative, Low-Risk)

**Rationale**:
- Existing modules are already well-structured
- OnnxSwipePredictor is critical (swipe typing core)
- Must maintain thread safety (recently added synchronized)
- 672 tests must pass after each change
- Build must succeed after each integration

### Phase 2A: Integrate Existing Modules (Current Phase)

**Step 1: Analyze Module Compatibility**
- Map OnnxSwipePredictor methods to existing modules
- Identify gaps where modules don't cover functionality
- Verify thread safety contracts

**Step 2: Create Facade Pattern**
- Keep OnnxSwipePredictor as coordinator (facade)
- Delegate to existing Kotlin modules
- Maintain public API unchanged
- Preserve singleton pattern

**Step 3: Incremental Integration**
Each module integrated separately with full testing:
1. Integrate ModelLoader
2. Integrate EncoderWrapper + DecoderWrapper
3. Integrate BeamSearchEngine
4. Integrate TensorFactory + MemoryPool
5. Final cleanup and documentation

---

## Module Mapping Analysis

### 1. Model Loading (Lines 206-560)

**Current OnnxSwipePredictor responsibilities**:
- `initialize()` - Load ONNX models from assets
- Path resolution (v2, custom, etc.)
- Session creation with NNAPI/XNNPACK
- Model configuration reading

**Target Module**: `ModelLoader.kt`

**Integration Plan**:
```java
// OLD (in OnnxSwipePredictor.java):
private boolean initialize() {
    byte[] encoderModelData = loadModelFromAssets(encoderPath);
    _encoderSession = _ortEnvironment.createSession(encoderModelData, sessionOptions);
    // ... 300+ lines of init logic
}

// NEW (delegate to ModelLoader):
private ModelLoader _modelLoader;

private boolean initialize() {
    if (_modelLoader == null) {
        _modelLoader = new ModelLoader(_context, _ortEnvironment);
    }
    ModelLoader.ModelSessions sessions = _modelLoader.loadModels(
        encoderPath, decoderPath, _currentModelVersion
    );
    _encoderSession = sessions.getEncoder();
    _decoderSession = sessions.getDecoder();
    // ... remaining initialization
}
```

**Benefits**:
- Reduces initialize() from 355 lines to ~50 lines
- Model loading logic centralized in one place
- Easier to test model loading independently

---

### 2. Inference Execution (Lines 580-950)

**Current OnnxSwipePredictor responsibilities**:
- Encoder tensor creation and execution
- Decoder tensor creation and execution
- Input/output tensor management
- Dimension handling for broadcast models

**Target Modules**: `EncoderWrapper.kt` + `DecoderWrapper.kt`

**Integration Plan**:
```java
// OLD (in OnnxSwipePredictor.java):
private float[][] runEncoder(float[][][] inputTensor) {
    OnnxTensor inputOnnx = OnnxTensor.createTensor(_ortEnvironment, inputTensor);
    OrtSession.Result result = _encoderSession.run(Collections.singletonMap("input", inputOnnx));
    // ... 150+ lines of tensor manipulation
}

// NEW (delegate to EncoderWrapper):
private EncoderWrapper _encoderWrapper;

private float[][] runEncoder(float[][][] inputTensor) {
    if (_encoderWrapper == null) {
        _encoderWrapper = new EncoderWrapper(_ortEnvironment, _encoderSession);
    }
    return _encoderWrapper.encode(inputTensor);
}
```

**Benefits**:
- Encoder/decoder logic isolated
- Easier to optimize tensor operations
- Better error handling

---

### 3. Beam Search Decoding (Lines 950-1180)

**Current OnnxSwipePredictor responsibilities**:
- Beam search algorithm
- Vocabulary filtering with VocabularyTrie
- Top-K prediction selection
- Confidence scoring

**Target Module**: `BeamSearchEngine.kt`

**Integration Plan**:
```java
// OLD (in OnnxSwipePredictor.java):
private List<String> beamSearch(float[][] encoderOutput, ...) {
    // ... 230+ lines of complex beam search logic
}

// NEW (delegate to BeamSearchEngine):
private BeamSearchEngine _beamSearchEngine;

private List<String> beamSearch(float[][] encoderOutput, ...) {
    if (_beamSearchEngine == null) {
        _beamSearchEngine = new BeamSearchEngine(_vocabulary, _tokenizer);
    }
    return _beamSearchEngine.decode(encoderOutput, _beamWidth, _maxLength);
}
```

**Benefits**:
- Complex beam search logic isolated
- Easier to optimize (GPU offloading potential)
- Better testability

---

### 4. Memory Management (Lines 1400-2100)

**Current OnnxSwipePredictor responsibilities**:
- Buffer pooling for encoder/decoder tensors
- Tensor reuse to avoid allocations
- Memory cleanup on session teardown

**Target Module**: `MemoryPool.kt`

**Integration Plan**:
```java
// OLD (scattered throughout OnnxSwipePredictor.java):
private float[][][] _encoderInputBuffer;  // Pre-allocated
private float[][] _encoderOutputBuffer;   // Pre-allocated
// ... manual buffer management

// NEW (use MemoryPool):
private MemoryPool _memoryPool;

public void initialize() {
    _memoryPool = new MemoryPool(_maxSequenceLength, HIDDEN_DIM);
    // Module handles all buffer allocation/reuse
}
```

**Benefits**:
- Centralized memory management
- Easier to track memory usage
- Consistent pooling strategy

---

### 5. Configuration Management (Lines 1280-1400)

**Current OnnxSwipePredictor responsibilities**:
- `setConfig()` - Handle config changes
- Model version switching
- Parameter updates (beam width, max length, etc.)
- Thread-safe config updates

**New Module Needed**: `OnnxConfigManager.kt` (TO BE CREATED)

**Integration Plan**:
```kotlin
// NEW MODULE: OnnxConfigManager.kt
class OnnxConfigManager(
    private val modelLoader: ModelLoader,
    private val context: Context
) {
    fun handleConfigChange(
        oldConfig: Config,
        newConfig: Config,
        onReload: (ModelLoader.ModelSessions) -> Unit
    ): Boolean {
        val versionChanged = newConfig.neural_model_version != oldConfig.neural_model_version
        val pathsChanged = checkCustomPaths(newConfig, oldConfig)

        if (versionChanged || pathsChanged) {
            val sessions = modelLoader.loadModels(...)
            onReload(sessions)
            return true
        }
        return false
    }
}
```

---

## Implementation Plan

### Week 1: Integration Phase

**Day 1-2: ModelLoader Integration**
- [ ] Add ModelLoader import to OnnxSwipePredictor
- [ ] Replace initialize() logic with ModelLoader delegation
- [ ] Test: Build, run tests, verify swipe typing works
- [ ] Commit: "refactor(onnx): integrate ModelLoader module"

**Day 3: Encoder/Decoder Integration**
- [ ] Add EncoderWrapper + DecoderWrapper imports
- [ ] Replace runEncoder() and runDecoder() with wrappers
- [ ] Test: Build, run tests, verify predictions
- [ ] Commit: "refactor(onnx): integrate Encoder/Decoder wrappers"

**Day 4: BeamSearch Integration**
- [ ] Add BeamSearchEngine import
- [ ] Replace beamSearch() logic with engine delegation
- [ ] Test: Build, run tests, verify beam search accuracy
- [ ] Commit: "refactor(onnx): integrate BeamSearchEngine"

**Day 5: MemoryPool Integration**
- [ ] Add MemoryPool import
- [ ] Replace manual buffer management with pool
- [ ] Test: Build, run tests, check for memory leaks
- [ ] Commit: "refactor(onnx): integrate MemoryPool"

**Day 6: Create OnnxConfigManager**
- [ ] Create new OnnxConfigManager.kt module
- [ ] Extract setConfig() logic to module
- [ ] Test: Build, test config changes
- [ ] Commit: "refactor(onnx): extract ConfigManager module"

**Day 7: Final Cleanup + Kotlin Conversion**
- [ ] Convert OnnxSwipePredictor.java → OnnxSwipePredictor.kt
- [ ] Remove all extracted code (now in modules)
- [ ] Keep only facade methods (delegating to modules)
- [ ] Final testing: Full swipe typing test suite
- [ ] Commit: "refactor(onnx): complete modularization"

---

## Success Criteria

✅ **Functional**:
- All 672 tests pass
- Swipe typing works identically to before
- No regression in accuracy or performance

✅ **Code Quality**:
- OnnxSwipePredictor reduced from 2,708 → ~400 lines
- Each module has single responsibility
- Clear separation of concerns

✅ **Maintainability**:
- New features can be added to individual modules
- Bugs easier to isolate and fix
- Code easier to understand and review

✅ **Performance**:
- No performance regression
- Memory usage unchanged or improved
- GC pressure reduced (better pooling)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Breaking swipe typing | Test after each module integration |
| Losing thread safety | Keep synchronized in facade |
| Performance regression | Benchmark before/after each module |
| Complex merge conflicts | Small, focused commits |
| Breaking existing tests | Run full test suite after each change |

---

## Current Status

- [x] Phase 1: ONNX file cleanup (17MB APK reduction)
- [ ] Phase 2A: Module integration (IN PROGRESS)
  - [x] Step 1: Analysis (COMPLETED)
  - [x] Step 2: ModelLoader integration (COMPLETED - commit dd99324c)
  - [ ] Step 3: Encoder/Decoder integration (IN PROGRESS)
  - [ ] Step 4: BeamSearch integration
  - [ ] Step 5: MemoryPool integration
  - [ ] Step 6: ConfigManager creation
  - [ ] Step 7: Final Kotlin conversion
- [ ] Phase 2B: Bottleneck fixes (UI rendering optimization)

## Progress Log

### 2025-11-22: ModelLoader Integration Complete
- **Commit**: dd99324c "refactor(onnx): integrate ModelLoader module"
- **Changes**:
  * Added import for juloo.keyboard2.onnx.ModelLoader
  * Added `_modelLoader` field to OnnxSwipePredictor
  * Replaced 150+ lines of manual model loading with ModelLoader delegation
  * Simplified initialize() method
  * Build successful (v1.32.636)
- **Benefits**:
  * Cleaner separation of concerns
  * Model loading logic now testable independently
  * Hardware acceleration fallback chain encapsulated
  * Model caching and optimization handled by module

### 2025-11-22: EncoderWrapper + DecoderWrapper Integration Complete
- **Commit**: ab434168 "refactor(onnx): integrate EncoderWrapper + DecoderWrapper + TensorFactory"
- **Changes**:
  * Added imports for EncoderWrapper, DecoderWrapper, TensorFactory
  * Added fields for modular inference components
  * Initialized components after model loading
  * Replaced encoder inference with EncoderWrapper.encode()
  * Simplified predict() method - removed 40+ lines of tensor creation
  * Added runBeamSearch() overload accepting OnnxTensor
  * Build successful (v1.32.637)
- **Benefits**:
  * Encoder execution logic encapsulated and testable
  * TensorFactory handles all tensor creation
  * Cleaner error handling and timing logs
  * Simplified memory management (EncoderWrapper owns input tensors)

### 2025-11-22: Partial MemoryPool Integration
- **Status**: Import added, field created, not fully integrated
- **Reason**: Full integration requires replacing buffer access throughout 400+ lines of beam search code
- **Deferral**: Will be completed in final Kotlin conversion phase
- **Next**: OnnxConfigManager module creation

---

**Next Action**: Begin ModelLoader integration
