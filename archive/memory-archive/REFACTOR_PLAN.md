# Refactor Plan - PredictionController Extraction

**Status**: Planned (not yet implemented)
**Complexity**: High (3-5 days)
**Risk**: Medium (requires extensive testing)

---

## Problem Statement

`Keyboard2.java` is currently 1200+ lines with multiple responsibilities:
1. InputMethodService lifecycle management
2. Prediction logic (regular typing + swipe typing)
3. Auto-correction
4. Context tracking
5. UI coordination
6. Event handling

This violates Single Responsibility Principle and makes the code:
- Hard to test (requires Android instrumentation)
- Hard to maintain (changes affect multiple concerns)
- Hard to understand (prediction logic mixed with IME callbacks)

---

## Proposed Solution: PredictionController Pattern

Extract all prediction logic into a dedicated `PredictionController` class that can be unit tested independently.

### Architecture

```
Keyboard2 (InputMethodService)
    ↓ implements
ImePredictionApi (interface)
    ↑ uses
PredictionController
    ↓ owns
WordPredictor, NeuralEngine, BigramModel, etc.
```

### Key Components

**1. ImePredictionApi Interface** (NEW)
```java
public interface ImePredictionApi {
    InputConnection getInputConnection();
    void showSuggestions(List<String> words, List<Integer> scores);
    void clearSuggestions();
    EditorInfo getEditorInfo();
}
```

**Benefits**:
- Decouples PredictionController from InputMethodService
- Enables unit testing with mock implementation
- Clear contract for IME interactions

**2. PredictionController Class** (NEW)
```java
public class PredictionController {
    // Moved from Keyboard2.java:
    private WordPredictor wordPredictor;
    private NeuralSwipeTypingEngine neuralEngine;
    private AsyncPredictionHandler asyncPredictionHandler;
    private UserAdaptationManager adaptationManager;
    private StringBuilder currentWord;
    private List<String> contextWords;
    private String lastAutoInsertedWord;
    private PredictionSource lastCommitSource;

    // Public API called by Keyboard2:
    public void onCharacterTyped(String text);
    public void onSwipeGesture(List<PointF> path, List<Long> timestamps);
    public void onSuggestionSelected(String word);
    public void onBackspace();
    public void onStartInput();
    public void onDestroy();

    // Private implementation:
    private void handlePredictionResults(List<String> predictions, List<Integer> scores);
    private void updateContext(String word);
    private void performAutoCorrection(String completedWord);
}
```

**Benefits**:
- All prediction state in one place
- Can be unit tested without Android framework
- Clear API surface (6 public methods)

**3. Simplified Keyboard2.java** (MODIFIED)
```java
public class Keyboard2 extends InputMethodService implements ImePredictionApi {
    private PredictionController predictionController;
    private Keyboard2View keyboardView;
    private SuggestionBar suggestionBar;

    @Override
    public void onCreate() {
        super.onCreate();
        predictionController = new PredictionController(this, config, this);
    }

    // IME lifecycle methods (no change)
    // Implement ImePredictionApi interface
    // Delegate events to predictionController
}
```

**Benefits**:
- Focused on Android integration
- ~400-500 lines instead of 1200+
- Easier to understand IME callbacks

---

## Migration Strategy

### Phase 1: Create Infrastructure (Day 1)
1. Create `ImePredictionApi.java` interface
2. Create empty `PredictionController.java` class
3. Make `Keyboard2` implement `ImePredictionApi`
4. Build and verify no regressions

### Phase 2: Move State (Day 2)
1. Move fields to `PredictionController`:
   - `_wordPredictor`, `_neuralEngine`, `_asyncPredictionHandler`
   - `_currentWord`, `_contextWords`
   - `_lastAutoInsertedWord`, `_lastCommitSource`
2. Update constructor to initialize components
3. Add getters/setters as needed
4. Build and test

### Phase 3: Move Logic (Days 3-4)
1. Extract `handleRegularTyping()` → `onCharacterTyped()`
2. Extract `handleSwipeTyping()` → `onSwipeGesture()`
3. Extract `onSuggestionSelected()` → keep same name
4. Extract `handleBackspace()` → `onBackspace()`
5. Extract `updateContext()`, `handlePredictionResults()`
6. Extract auto-correction logic
7. Update all call sites in `Keyboard2`
8. Build and test after each extraction

### Phase 4: Testing & Validation (Day 5)
1. Test regular typing predictions
2. Test swipe typing
3. Test auto-correction (normal + Termux mode)
4. Test context tracking
5. Test suggestion selection
6. Test backspace behavior
7. Test in multiple apps (Termux, Messages, Gmail)

---

## Risks & Mitigations

### Risk 1: Breaking Auto-Correction
**Mitigation**: Test extensively in Termux and normal apps. Verify:
- "thid" + space → "this " (normal app)
- "thid" + space → "this" (Termux app)
- Capitalization preservation

### Risk 2: Thread Safety Issues
**Mitigation**:
- Keep async handling in PredictionController
- Use same thread coordination as current code
- Test rapid typing scenarios

### Risk 3: InputConnection Lifecycle
**Mitigation**:
- Always get InputConnection via `imeApi.getInputConnection()`
- Never cache InputConnection in PredictionController
- Check for null before every use

### Risk 4: Regression in Existing Features
**Mitigation**:
- Extract one method at a time
- Build and test after each change
- Keep git history clean with small commits
- Use feature flag to toggle new vs old code path

---

## Testing Checklist

After completing refactor, verify:

- [ ] Regular typing predictions work
- [ ] Swipe typing works
- [ ] Auto-correction works in normal apps (with space)
- [ ] Auto-correction works in Termux (no space)
- [ ] Capitalization preservation (teh→the, Teh→The, TEH→THE)
- [ ] Context tracking (bigram predictions)
- [ ] Suggestion bar updates correctly
- [ ] Suggestion selection replaces word correctly
- [ ] Backspace updates predictions
- [ ] No memory leaks (test with Android Profiler)
- [ ] No crashes on orientation change
- [ ] No crashes on keyboard switch

---

## Files to Create

1. `srcs/juloo.keyboard2/ImePredictionApi.java` (new, ~30 lines)
2. `srcs/juloo.keyboard2/PredictionController.java` (new, ~600 lines)

## Files to Modify

1. `srcs/juloo.keyboard2/Keyboard2.java` (reduce from 1200 to ~500 lines)
2. `srcs/juloo.keyboard2/Keyboard2View.java` (update swipe callback)

## Estimated Line Count After Refactor

- `Keyboard2.java`: ~500 lines (IME lifecycle + UI)
- `PredictionController.java`: ~600 lines (prediction logic)
- `ImePredictionApi.java`: ~30 lines (interface)

**Total**: Same ~1200 lines, but better organized and testable.

---

## Future Enhancements (Post-Refactor)

Once `PredictionController` is extracted:

1. **Unit Tests**: Write JUnit tests for prediction logic without Android
2. **Multiple Languages**: Add language-specific prediction controllers
3. **ML Training**: Collect user correction data in PredictionController
4. **A/B Testing**: Easily swap prediction algorithms
5. **Performance Profiling**: Measure prediction latency in isolation

---

## Decision

**Recommendation**: Defer this refactor until after testing auto-correction feature in production.

**Reasoning**:
- Auto-correction just implemented (v1.32.114-121)
- Code is working and stable
- Need user feedback on auto-correction behavior
- Large refactor risks introducing regressions
- Better to validate current code first

**Timeline**: Revisit in 1-2 weeks after auto-correction has been tested.

---

## References

- Gemini 2.5 Pro analysis: See memory/REFACTOR_PLAN.md for full recommendation
- Current code: `Keyboard2.java:1-1800`
- Related: `WordPredictor.java`, `NeuralSwipeTypingEngine.java`
