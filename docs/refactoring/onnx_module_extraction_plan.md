# OnnxSwipePredictor Refactoring Plan

## Current State
- **Total Lines**: 2484 lines
- **Inner Classes**: 3 (BeamSearchState, IndexValue, BeamSearchCandidate)
- **Private Methods**: 25
- **Public Methods**: 22
- **Language**: Java (will migrate to Kotlin)

## Identified Logical Modules

### 1. MemoryPool.kt (Priority 1)
**Lines**: ~100-120, ~840-900
**Responsibility**: Managing pre-allocated tensor buffers for performance
**Fields**:
- `_pooledTokensByteBuffer`: Reusable ByteBuffer for tokens
- `_pooledMemoryArray`: Reusable memory replication array
- `_preallocBatchedTokens`: Pre-allocated beam search buffers
- Buffer capacity tracking

**Why First**: Self-contained, no complex dependencies, pure data management

### 2. TensorFactory.kt (Priority 2)
**Lines**: ~1440-1650
**Responsibility**: Creating and managing ONNX tensors
**Methods**:
- `createTrajectoryTensor()`: Convert trajectory features to ONNX tensor
- `createNearestKeysTensor()`: Convert key sequence to tensor
- `createActualLengthTensor()`: Create length tensor
- `createTargetTokensTensor()`: Create decoder input tensor
- Tensor shape validation

**Why Second**: Used by encoder, decoder, and beam search modules

### 3. BroadcastSupport.kt (Priority 3)
**Lines**: ~1520-1570, ~1767-1801
**Responsibility**: Handling broadcast-enabled model configuration
**Methods**:
- `readModelConfig()`: Parse model_config.json
- `shouldBroadcastMemory()`: Determine if broadcast is needed
- `createBroadcastMemoryTensor()`: Handle memory broadcast logic
- `createBroadcastSrcLengthTensor()`: Handle src_length broadcast

**Why Third**: Standalone utility, encapsulates broadcast logic cleanly

### 4. ModelConfig.kt (Priority 4)
**Lines**: ~200-450
**Responsibility**: Model configuration and metadata
**Data Class**:
```kotlin
data class ModelConfig(
    val accuracy: Float,
    val dModel: Int,
    val maxSeqLen: Int,
    val maxWordLen: Int,
    val broadcastEnabled: Boolean
)
```
**Methods**:
- `loadFromAssets()`: Read model_config.json
- `validate()`: Verify configuration consistency

**Why Fourth**: Simple data class, needed by ModelLoader

### 5. EncoderWrapper.kt (Priority 5)
**Lines**: ~680-800
**Responsibility**: Encoder inference operations
**Methods**:
- `encode(trajectory, keys, length)`: Run encoder inference
- `prepareEncoderInputs()`: Create encoder input tensors
- `processEncoderOutput()`: Extract memory from results
- Performance timing

**Why Fifth**: Encapsulates encoder, needed before decoder

### 6. DecoderWrapper.kt (Priority 6)
**Lines**: ~1650-1900 (decoder sections)
**Responsibility**: Decoder inference operations
**Methods**:
- `decode(memory, tokens, srcLength)`: Run decoder inference
- `prepareDecoderInputs()`: Create decoder input tensors
- `processDecoderOutput()`: Extract logits from results
- Handle batched vs sequential modes
- Integrate with BroadcastSupport

**Why Sixth**: Encapsulates decoder, depends on BroadcastSupport

### 7. BeamSearchEngine.kt (Priority 7)
**Lines**: ~1650-2100
**Responsibility**: Core beam search algorithm
**Classes**:
- `BeamSearchState`: Track beam state during search
- `BeamCandidate`: Final beam candidate with score
- `BeamSearchConfig`: Search parameters (width, max_length, etc.)

**Methods**:
- `search(memory, srcLength)`: Main beam search loop
- `expandBeams()`: Generate candidates for each beam
- `selectTopBeams()`: Prune to top-k beams
- `isFinished()`: Check termination conditions
- `convertToWords()`: Convert token sequences to strings

**Why Seventh**: Complex algorithm, depends on decoder and tensor factory

### 8. ModelLoader.kt (Priority 8)
**Lines**: ~200-450
**Responsibility**: Loading and initializing ONNX models
**Methods**:
- `initialize()`: Main initialization entry point
- `loadModelFromAssets()`: Load model bytes from assets/URIs
- `createNnapiSessionOptions()`: Configure NNAPI hardware acceleration
- `tryEnableHardwareAcceleration()`: Fallback chain (NNAPI→QNN→XNNPACK→CPU)
- `verifyExecutionProvider()`: Verify hardware acceleration working

**Why Eighth**: Coordinates everything, depends on all other modules

### 9. OnnxSwipePredictor (Main Coordinator)
**Remaining**: ~300-400 lines
**Responsibility**: Public API and module coordination
**Methods**:
- `predict()`: Main prediction entry point
- `predictAsync()`: Async prediction wrapper
- Singleton management
- Configuration updates
- Debug logging coordination

## Refactoring Sequence

### Phase 1: Data & Utilities (3 modules)
1. **MemoryPool.kt** - Self-contained buffer management
2. **TensorFactory.kt** - Tensor creation utilities
3. **BroadcastSupport.kt** - Broadcast configuration logic

**Milestone**: Core utilities extracted, OnnxSwipePredictor still works

### Phase 2: Model Components (3 modules)
4. **ModelConfig.kt** - Configuration data class
5. **EncoderWrapper.kt** - Encoder operations
6. **DecoderWrapper.kt** - Decoder operations

**Milestone**: Encoder/decoder encapsulated, beam search still in main class

### Phase 3: Algorithm & Coordination (2 modules + refactor)
7. **BeamSearchEngine.kt** - Beam search algorithm
8. **ModelLoader.kt** - Model initialization
9. **OnnxSwipePredictor refactor** - Update to use all new modules

**Milestone**: Complete refactoring, all modules extracted

## Testing Strategy

**After Each Module**:
1. ✅ Build successfully (no compilation errors)
2. ✅ Run unit tests if available
3. ✅ Manual swipe test (verify predictions unchanged)
4. ✅ Git commit with descriptive message

**Final Verification**:
1. Full test suite passes
2. Performance benchmarks unchanged
3. Memory usage unchanged
4. Predictions identical to pre-refactor

## Benefits

### Maintainability
- Each module < 400 lines (vs 2484 line monolith)
- Clear separation of concerns
- Easy to locate specific functionality

### Testability
- Can unit test each module in isolation
- Mock dependencies easily
- Faster test execution

### Clarity
- Self-documenting module names
- Reduced cognitive load
- Easier onboarding for new developers

### Kotlin Advantages
- Null safety prevents NPEs
- Data classes reduce boilerplate
- Extension functions for utilities
- Coroutines for async operations

### Future Extensibility
- Easy to add new model backends
- Can swap beam search algorithms
- Plug different tensor factories

## Risk Mitigation

1. **Incremental approach**: Extract one module at a time
2. **Test after each step**: Catch regressions early
3. **Git commits**: Easy to revert if needed
4. **Keep original**: Don't delete until fully migrated
5. **Performance monitoring**: Ensure no slowdowns

## Success Criteria

- [ ] All 8 modules extracted
- [ ] OnnxSwipePredictor < 400 lines
- [ ] All tests passing
- [ ] No performance regression
- [ ] No prediction quality regression
- [ ] Code coverage maintained or improved
