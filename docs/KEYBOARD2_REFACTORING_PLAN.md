# Keyboard2.java Refactoring Plan

**Date**: 2025-11-13
**Current Size**: 2,397 lines
**Target**: Split into focused, single-responsibility classes

---

## 🎯 Goals

1. **Improve Maintainability**: Reduce cognitive load by splitting 2,397 lines into focused classes
2. **Single Responsibility**: Each class handles one concern
3. **Testability**: Smaller classes easier to unit test
4. **Reduce Merge Conflicts**: Multiple developers can work on different components
5. **Preserve Functionality**: Zero behavioral changes, only structure

---

## 📊 Current Structure Analysis

### Field Categories (25 total)

**Views & UI (7 fields)**:
- `_keyboardView`, `_emojiPane`, `_clipboard_pane`, `_contentPaneContainer`
- `_clipboardSearchBox`, `_clipboardHistoryView`, `_inputViewContainer`

**Prediction Engines (5 fields)**:
- `_wordPredictor`, `_neuralEngine`, `_asyncPredictionHandler`
- `_suggestionBar`, `_dictionaryManager`

**Context & State (6 fields)**:
- `_currentWord`, `_contextWords`, `_wasLastInputSwipe`
- `_lastAutoInsertedWord`, `_lastCommitSource`, `_currentSwipeData`

**Contraction System (2 fields)**:
- `_nonPairedContractions`, `_knownContractions`

**Configuration (3 fields)**:
- `_config`, `_handler`, `_foldStateTracker`

**Other (2 fields)**:
- `_mlDataStore`, `_adaptationManager`

### Method Categories (38 methods)

**Lifecycle (4 methods)**:
- `onCreate()`, `onDestroy()`, `onStartInputView()`, `onFinishInputView()`

**Input Handling (6 methods)**:
- `handleRegularTyping()`, `handleBackspace()`, `handleDeleteLastWord()`
- `handleSwipeTyping()`, `handle_text_typed()`, `handle_backspace()`

**Prediction Management (8 methods)**:
- `handlePredictionResults()`, `updatePredictionsForCurrentWord()`
- `updateSwipePredictions()`, `completeSwipePredictions()`, `clearSwipePredictions()`
- `updateCGRPredictions()`, `checkCGRPredictions()`, `onSuggestionSelected()`

**Context Management (2 methods)**:
- `updateContext()`, `loadContractionMappings()`

**Configuration (5 methods)**:
- `refresh_config()`, `refresh_action_label()`, `refresh_special_layout()`
- `onSharedPreferenceChanged()`, `refreshSubtypeImm()`

**Clipboard Management (1 method)**:
- `showDateFilterDialog()`

**View Management (4 methods)**:
- `setInputView()`, `inflate_view()`, `updateFullscreenMode()`, `updateSoftInputWindowLayoutParams()`

**Layout Management (3 methods)**:
- `setNeuralKeyboardLayout()`, `refreshAccentsOption()`, `checkAndPromptDefaultIME()`

**Utilities (5 methods)**:
- `sendDebugLog()`, `getUserKeyboardHeightPercent()`, `actionLabel_of_imeAction()`
- `getEnabledSubtypes()`, `onUpdateSelection()`

---

## 🏗️ Proposed Architecture

### Phase 1: Extract Managers (Low Risk)

#### 1.1 ClipboardManager
**Responsibility**: Handle all clipboard-related functionality

**Extracted Members**:
- Fields: `_clipboard_pane`, `_clipboardSearchMode`, `_clipboardSearchBox`, `_clipboardHistoryView`
- Methods: `showDateFilterDialog()`, clipboard pane setup/teardown

**Interface**:
```java
public class ClipboardManager {
  private ViewGroup _clipboardPane;
  private boolean _searchMode;

  public ClipboardManager(Context context, Config config);
  public ViewGroup getClipboardPane();
  public void showDateFilterDialog(View anchorView);
  public void setSearchMode(boolean enabled);
  public void clear();
}
```

**Benefits**:
- Isolated clipboard functionality
- Easy to test independently
- Clear ownership of clipboard features

**Risks**: ⚠️ Low
- Minimal dependencies on other components
- Well-defined boundary

---

#### 1.2 ContractionManager
**Responsibility**: Load and manage contraction mappings

**Extracted Members**:
- Fields: `_nonPairedContractions`, `_knownContractions`
- Methods: `loadContractionMappings()`

**Interface**:
```java
public class ContractionManager {
  private Map<String, String> _nonPairedContractions;
  private Set<String> _knownContractions;

  public ContractionManager(Context context);
  public void loadMappings();
  public boolean isKnownContraction(String word);
  public String getNonPairedMapping(String word);
}
```

**Benefits**:
- Self-contained contraction logic
- Easy to extend with new contraction types
- Clear separation of concerns

**Risks**: ⚠️ Low
- No dependencies on prediction engines
- Pure data management

---

#### 1.3 PredictionContextTracker
**Responsibility**: Track typing context for predictions

**Extracted Members**:
- Fields: `_currentWord`, `_contextWords`, `_wasLastInputSwipe`
- Fields: `_lastAutoInsertedWord`, `_lastCommitSource`
- Methods: `updateContext()`

**Interface**:
```java
public class PredictionContextTracker {
  private StringBuilder _currentWord;
  private List<String> _contextWords;
  private boolean _wasLastInputSwipe;

  public PredictionContextTracker();
  public void appendToCurrentWord(String text);
  public void commitWord(String word, PredictionSource source);
  public void clearCurrentWord();
  public String getCurrentWord();
  public List<String> getContextWords();
  public boolean wasLastInputSwipe();
}
```

**Benefits**:
- Centralized context management
- Easier to add n-gram support
- Clear state tracking

**Risks**: ⚠️ Low
- Used by prediction methods but doesn't depend on them
- Well-defined interface

---

### Phase 2: Extract Coordinators (Medium Risk)

#### 2.1 PredictionCoordinator
**Responsibility**: Coordinate all prediction engines and results

**Extracted Members**:
- Fields: `_wordPredictor`, `_neuralEngine`, `_asyncPredictionHandler`
- Fields: `_suggestionBar`, `_dictionaryManager`, `_mlDataStore`, `_adaptationManager`
- Methods: `handlePredictionResults()`, `updatePredictionsForCurrentWord()`
- Methods: `updateSwipePredictions()`, `completeSwipePredictions()`, `clearSwipePredictions()`
- Methods: `updateCGRPredictions()`, `checkCGRPredictions()`

**Interface**:
```java
public class PredictionCoordinator {
  private WordPredictor _wordPredictor;
  private NeuralSwipeTypingEngine _neuralEngine;
  private AsyncPredictionHandler _asyncHandler;
  private SuggestionBar _suggestionBar;

  public PredictionCoordinator(Context context, Config config);
  public void setConfig(Config config);
  public void requestTypingPredictions(String currentWord, List<String> context);
  public void requestSwipePredictions(List<KeyboardData.Key> swipedKeys);
  public void handlePredictionResults(List<String> predictions, List<Integer> scores);
  public void updateSuggestionBar(List<String> predictions);
  public void clearPredictions();
  public void shutdown();
}
```

**Benefits**:
- All prediction logic in one place
- Easier to add new prediction engines
- Clear coordination point

**Risks**: ⚠️⚠️ Medium
- Multiple dependencies (engines, suggestion bar)
- Needs careful callback handling
- Performance-sensitive code

---

#### 2.2 InputCoordinator
**Responsibility**: Coordinate input handling across typing and swiping

**Extracted Members**:
- Methods: `handleRegularTyping()`, `handleBackspace()`, `handleDeleteLastWord()`
- Methods: `handleSwipeTyping()`, `onSuggestionSelected()`

**Interface**:
```java
public class InputCoordinator {
  private PredictionCoordinator _predictionCoordinator;
  private PredictionContextTracker _contextTracker;
  private InputConnection _inputConnection;

  public InputCoordinator(InputConnection ic, PredictionCoordinator pc, PredictionContextTracker ct);
  public void handleTyping(String text);
  public void handleBackspace();
  public void handleDeleteLastWord();
  public void handleSwipeGesture(List<KeyboardData.Key> keys);
  public void handleSuggestionSelection(String word);
}
```

**Benefits**:
- Clear input handling flow
- Easier to add new input types
- Testable input logic

**Risks**: ⚠️⚠️⚠️ High
- Touches core keyboard functionality
- Needs careful coordination with predictions
- Performance-critical path

---

### Phase 3: Extract Configuration (Low-Medium Risk)

#### 3.1 ConfigurationManager
**Responsibility**: Manage config refresh and preference changes

**Extracted Members**:
- Fields: `_config`, `_foldStateTracker`
- Methods: `refresh_config()`, `onSharedPreferenceChanged()`
- Methods: `refreshSubtypeImm()`, `refresh_action_label()`

**Interface**:
```java
public class ConfigurationManager implements SharedPreferences.OnSharedPreferenceChangeListener {
  private Config _config;
  private FoldStateTracker _foldStateTracker;

  public ConfigurationManager(Context context, Resources res);
  public void refresh(Resources res);
  public Config getConfig();
  public void onSharedPreferenceChanged(SharedPreferences prefs, String key);
  public void registerConfigChangeListener(ConfigChangeListener listener);
}

public interface ConfigChangeListener {
  void onConfigChanged(Config newConfig);
  void onThemeChanged(int newTheme);
}
```

**Benefits**:
- Centralized config management
- Easy to add config listeners
- Clear config change propagation

**Risks**: ⚠️⚠️ Low-Medium
- Used by many components
- Needs careful listener management

---

### Phase 4: Extract View Management (High Risk)

#### 4.1 ViewManager
**Responsibility**: Manage keyboard views and panes

**Extracted Members**:
- Fields: `_keyboardView`, `_emojiPane`, `_inputViewContainer`, `_contentPaneContainer`
- Methods: `setInputView()`, `inflate_view()`, `updateFullscreenMode()`
- Methods: `updateSoftInputWindowLayoutParams()`, `setNeuralKeyboardLayout()`

**Interface**:
```java
public class ViewManager {
  private Keyboard2View _keyboardView;
  private ViewGroup _emojiPane;
  private LinearLayout _inputViewContainer;

  public ViewManager(Context context);
  public Keyboard2View getKeyboardView();
  public ViewGroup getEmojiPane();
  public void setInputView(View v);
  public void updateFullscreenMode(int height);
  public void recreateViews(Config config);
}
```

**Benefits**:
- Isolated view lifecycle
- Easier to test view logic
- Clear view ownership

**Risks**: ⚠️⚠️⚠️ High
- Core Android IME integration
- Complex view lifecycle
- Needs careful Activity/Service context handling

---

## 📋 Refactoring Phases (Recommended Order)

### Phase 1: Extraction (Safest First) ✅ RECOMMENDED START
1. ✅ **ContractionManager** - Completely isolated, no dependencies
2. ✅ **ClipboardManager** - Minimal dependencies, well-defined
3. ✅ **PredictionContextTracker** - State management only, no logic

**Estimated Effort**: 4-6 hours
**Risk**: ⚠️ Low
**Benefits**: Immediate reduction of ~300 lines from Keyboard2.java

---

### Phase 2: Coordination (Medium Complexity)
1. ⚠️ **ConfigurationManager** - Used by many components, needs listeners
2. ⚠️ **PredictionCoordinator** - Complex but well-bounded

**Estimated Effort**: 8-12 hours
**Risk**: ⚠️⚠️ Medium
**Benefits**: Reduction of ~600 lines, clearer prediction flow

---

### Phase 3: Core Refactoring (Most Complex)
1. ⚠️⚠️⚠️ **InputCoordinator** - Performance-critical
2. ⚠️⚠️⚠️ **ViewManager** - Android IME integration

**Estimated Effort**: 12-16 hours
**Risk**: ⚠️⚠️⚠️ High
**Benefits**: Final reduction of ~800 lines, complete separation

---

## 🎯 Target Architecture

```
Keyboard2.java (final: ~500-700 lines)
├── Lifecycle methods (onCreate, onDestroy, onStart, onFinish)
├── Android IME callbacks (onUpdateSelection, onEvaluateFullscreenMode)
├── Receiver inner class (KeyEventHandler.IReceiver implementation)
└── Component coordination
    ├── ClipboardManager
    ├── ContractionManager
    ├── PredictionContextTracker
    ├── ConfigurationManager
    ├── PredictionCoordinator
    ├── InputCoordinator
    └── ViewManager
```

---

## 📝 Implementation Guidelines

### Testing Strategy
1. **Before Refactoring**:
   - Document all current behaviors
   - Create integration test suite
   - Build and verify on device

2. **During Refactoring**:
   - Extract one class at a time
   - Build after each extraction
   - Test on device after each extraction
   - Commit after each successful extraction

3. **After Refactoring**:
   - Run full integration tests
   - Performance testing (no regressions)
   - User acceptance testing

### Code Guidelines
- Preserve all existing functionality (zero behavioral changes)
- Use dependency injection for all components
- Maintain backward compatibility with existing callbacks
- Document all extracted interfaces
- Add unit tests for extracted classes

### Rollback Strategy
- Each extraction is a separate commit
- Easy to revert individual extractions
- Keep feature branch until fully validated
- Merge to main only after device testing

---

## 🚨 Risks and Mitigation

### Risk 1: Performance Regression
**Likelihood**: Medium
**Impact**: High
**Mitigation**:
- Profile before and after each extraction
- Inline critical path methods if needed
- Benchmark swipe typing latency

### Risk 2: Android IME Lifecycle Issues
**Likelihood**: Medium
**Impact**: High
**Mitigation**:
- Test on multiple Android versions (API 21-34)
- Careful context handling (Service vs Activity)
- Test rotation, app switching, keyboard show/hide

### Risk 3: Memory Leaks
**Likelihood**: Low
**Impact**: High
**Mitigation**:
- Use WeakReferences for callbacks
- Proper cleanup in onDestroy()
- LeakCanary integration for testing

### Risk 4: Broken Predictions
**Likelihood**: Low
**Impact**: Critical
**Mitigation**:
- Preserve all existing prediction flow
- Test edge cases (long words, fast typing, rapid swipes)
- A/B test with original code

---

## ✅ Success Criteria

1. **Line Count**: Keyboard2.java reduced from 2,397 to <700 lines
2. **Functionality**: Zero behavioral changes (bit-for-bit identical output)
3. **Performance**: No measurable performance regression (<5% variance)
4. **Testability**: All extracted classes have unit tests
5. **Maintainability**: Each class has single, clear responsibility
6. **Documentation**: All interfaces documented with Javadoc

---

## 🗓️ Estimated Timeline

**Phase 1 (Low Risk)**: 1 week
- ContractionManager: 1 day
- ClipboardManager: 2 days
- PredictionContextTracker: 2 days

**Phase 2 (Medium Risk)**: 2 weeks
- ConfigurationManager: 4 days
- PredictionCoordinator: 6 days

**Phase 3 (High Risk)**: 3 weeks
- InputCoordinator: 8 days
- ViewManager: 7 days

**Total**: 6 weeks (30 working days)

**Buffer**: +2 weeks for testing and unexpected issues

---

## 📖 Related Documentation

- **[memory/pm.md](../../memory/pm.md)** - Project management and current status
- **[CLAUDE.md](../CLAUDE.md)** - Build commands and development workflow
- **[docs/specs/](../docs/specs/)** - Technical specifications

---

## 🤝 Next Steps

1. **Review this plan** with team/maintainers
2. **Start with Phase 1** (ContractionManager) - lowest risk
3. **Build and test** after each extraction
4. **Document learnings** as we go
5. **Adjust plan** based on what we learn

---

**Status**: 📝 PLANNING
**Owner**: Claude + User
**Priority**: HIGH (technical debt)
**Blocked By**: None (ready to start)

---

**Last Updated**: 2025-11-13
