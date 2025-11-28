# Remaining Java File Migration Plan

**Status**: Kotlin migration 98.6% complete (145/148 files)
**Blocking Issue**: R8/D8 8.6.17 bug prevents APK builds (see [R8-BUG-WORKAROUND.md](../R8-BUG-WORKAROUND.md))
**Migration Paused**: Waiting for R8 bug fix

---

## 📊 Remaining Files (3 files, 4,070 lines)

### Priority Overview

1. **SwipeCalibrationActivity.java** - 1,321 lines (LOW RISK, can migrate now)
2. **SettingsActivity.java** - 2,051 lines (LOW RISK, can migrate now)
3. **Keyboard2.java** - 698 lines (HIGH RISK, migrate LAST)

---

## 1. SwipeCalibrationActivity.java Migration

**File**: `srcs/juloo.keyboard2/SwipeCalibrationActivity.java`
**Size**: 1,321 lines
**Complexity**: MEDIUM
**Risk**: LOW (standalone activity, no core dependencies)
**Estimated Time**: 2-3 hours
**Can Migrate Now**: ✅ YES (doesn't affect core keyboard functionality)

### Overview
Standalone calibration activity for neural swipe typing. Collects training data for ML model.

### Structure Analysis
- **Base Class**: `Activity` (standard Android activity)
- **Private Fields**: 27 fields
  - UI components: 11 (TextView, Button, ProgressBar, etc.)
  - State: 16 (word lists, timestamps, neural engine, etc.)
- **Methods**: ~40 methods
  - Lifecycle: `onCreate()`, `onStart()`, `onStop()`
  - UI handlers: button clicks, touch events
  - Data collection: swipe tracking, ML data storage
  - Word management: vocabulary loading, session preparation

### Key Components

#### UI Components (11 fields)
```java
private TextView _instructionText;
private TextView _currentWordText;
private TextView _progressText;
private TextView _benchmarkText;
private ProgressBar _progressBar;
private NeuralKeyboardView _keyboardView;
private Button _nextButton;
private Button _skipButton;
private Button _exportButton;
private Button _benchmarkButton;
private TextView _resultsTextBox;
private Button _copyResultsButton;
```

**Kotlin Migration**:
- All nullable: `private var _instructionText: TextView? = null`
- Initialize in `onCreate()` using `findViewById()`
- Use `?.` safe calls for all UI access

#### Neural/ML Components
```java
private NeuralSwipeTypingEngine _neuralEngine;
private Config _config;
private SwipeMLDataStore _mlDataStore;
```

**Kotlin Migration**:
- `lateinit var _neuralEngine: NeuralSwipeTypingEngine`
- `lateinit var _config: Config`
- `lateinit var _mlDataStore: SwipeMLDataStore`

#### Collections
```java
private List<String> fullVocabulary = null;
private List<String> _sessionWords;
private List<PointF> _currentSwipePoints = new ArrayList<>();
private List<Long> _currentSwipeTimestamps = new ArrayList<>();
private List<Long> _predictionTimes = new ArrayList<>();
private Map<String, String> nonPairedContractions = new HashMap<>();
```

**Kotlin Migration**:
- `private var fullVocabulary: List<String>? = null`
- `private lateinit var _sessionWords: List<String>`
- `private val _currentSwipePoints = mutableListOf<PointF>()`
- `private val _currentSwipeTimestamps = mutableListOf<Long>()`
- `private val _predictionTimes = mutableListOf<Long>()`
- `private val nonPairedContractions = mutableMapOf<String, String>()`

### Migration Steps

1. **Create SwipeCalibrationActivity.kt**
   - Convert class declaration: `class SwipeCalibrationActivity : Activity()`
   - Convert all fields with proper nullability

2. **Convert UI Initialization**
   - `onCreate()` → nullable findViewById with safe calls
   - Button listeners → lambda syntax
   - Touch event handling → override with proper types

3. **Convert Data Loading Methods**
   - `loadFullVocabulary()` → use `use {}` for resource streams
   - `loadContractionMappings()` → Kotlin collection builders
   - `prepareRandomSessionWords()` → `shuffled()`, `take()`

4. **Convert Touch Handling**
   - `NeuralKeyboardView.onTouchEvent()` → override with `MotionEvent?`
   - Path recording → Kotlin collection methods

5. **Convert Export/Benchmark Methods**
   - File I/O → `use {}` for auto-close
   - Clipboard → safe call operators
   - Toast → Kotlin extension

### Testing Requirements
- ✅ Compile check only (R8 bug blocks builds)
- ⏸️ Runtime testing deferred until R8 fix
- 📋 Manual test plan:
  1. Launch calibration activity from settings
  2. Verify word display and progress
  3. Complete swipe for word
  4. Verify prediction accuracy display
  5. Export training data
  6. Verify JSON format

### Dependencies
**Used By**: SettingsActivity (launch intent)
**Uses**:
- NeuralSwipeTypingEngine (Kotlin ✅)
- Config (Kotlin ✅)
- SwipeMLDataStore (Kotlin ✅)
- Standard Android UI classes

**Migration Blockers**: NONE - can migrate now

---

## 2. SettingsActivity.java Migration

**File**: `srcs/juloo.keyboard2/SettingsActivity.java`
**Size**: 2,051 lines
**Complexity**: HIGH (many methods, but mostly independent)
**Risk**: LOW (UI only, no runtime keyboard logic)
**Estimated Time**: 4-5 hours
**Can Migrate Now**: ✅ YES (doesn't affect core keyboard functionality)

### Overview
Main settings activity using Android PreferenceActivity. Handles all app preferences, backup/restore, model management, and feature configuration.

### Structure Analysis
- **Base Class**: `PreferenceActivity` (deprecated but still used)
- **Implements**: `SharedPreferences.OnSharedPreferenceChangeListener`
- **Private Fields**: 9 constants, 1 instance field
- **Methods**: ~40 methods
  - Lifecycle: `onCreate()`, `onResume()`, `onPause()`, `onActivityResult()`
  - Preference handlers: 20+ methods
  - File operations: backup, restore, export, import (10+ methods)
  - UI updates: summaries, stats, version info (8+ methods)

### Key Components

#### Constants (Request Codes)
```java
private static final int REQUEST_CODE_BACKUP = 1001;
private static final int REQUEST_CODE_RESTORE = 1002;
private static final int REQUEST_CODE_NEURAL_ENCODER = 1003;
private static final int REQUEST_CODE_NEURAL_DECODER = 1004;
private static final int REQUEST_CODE_EXPORT_CUSTOM_DICT = 1006;
private static final int REQUEST_CODE_IMPORT_CUSTOM_DICT = 1007;
private static final int REQUEST_CODE_EXPORT_CLIPBOARD = 1008;
private static final int REQUEST_CODE_IMPORT_CLIPBOARD = 1009;
```

**Kotlin Migration**:
```kotlin
companion object {
    private const val REQUEST_CODE_BACKUP = 1001
    private const val REQUEST_CODE_RESTORE = 1002
    // ... etc
}
```

#### Instance Fields
```java
private BackupRestoreManager backupRestoreManager;
```

**Kotlin Migration**:
```kotlin
private lateinit var backupRestoreManager: BackupRestoreManager
```

### Migration Steps

1. **Convert Class Declaration**
   - `class SettingsActivity : PreferenceActivity(), SharedPreferences.OnSharedPreferenceChangeListener`
   - Move constants to companion object

2. **Convert onCreate()**
   - Keep try-catch for direct-boot mode
   - Convert `addPreferencesFromResource()` → `@Suppress("DEPRECATION")`
   - Convert preference handlers to Kotlin lambdas

3. **Convert Preference Handlers**
   - `setupBackupRestoreHandlers()` → lambda listeners
   - `setupCGRResetButtons()` → lambda listeners
   - Update preference summaries → safe calls

4. **Convert File Operations**
   - `performBackup(Uri)` → use `contentResolver.openOutputStream()?.use {}`
   - `performRestore(Uri)` → use `contentResolver.openInputStream()?.use {}`
   - Export/import methods → Kotlin I/O extensions

5. **Convert UI Updates**
   - `updateNeuralModelInfo()` → safe calls, string templates
   - `updateClipboardStats()` → Kotlin string formatting
   - `updateCGRParameterSummaries()` → safe calls

6. **Convert Activity Results**
   - `onActivityResult()` → when expression for request codes
   - Handle file picker results with safe calls

### Testing Requirements
- ✅ Compile check only (R8 bug blocks builds)
- ⏸️ Runtime testing deferred until R8 fix
- 📋 Manual test plan:
  1. Launch settings from keyboard
  2. Verify all preference screens load
  3. Test backup/restore operations
  4. Test model import/export
  5. Test dictionary import/export
  6. Verify clipboard history export/import
  7. Test CGR preset buttons
  8. Verify all preference summaries update

### Dependencies
**Used By**: Keyboard2 (launch intent), KeyboardReceiver
**Uses**:
- Config (Kotlin ✅)
- BackupRestoreManager (Kotlin ✅)
- FoldStateTracker (Kotlin ✅)
- SwipeMLDataStore (Kotlin ✅)
- SwipeMLTrainer (Kotlin ✅)
- ClipboardHistoryService (Kotlin ✅)
- Standard Android Preference classes

**Migration Blockers**: NONE - can migrate now

### Known Issues
- PreferenceActivity is deprecated (replaced by PreferenceFragmentCompat)
- Consider full rewrite to AndroidX Preferences in future
- Current migration: keep PreferenceActivity, just convert to Kotlin

---

## 3. Keyboard2.java Migration

**File**: `srcs/juloo.keyboard2/Keyboard2.java`
**Size**: 698 lines
**Complexity**: VERY HIGH
**Risk**: VERY HIGH (core InputMethodService)
**Estimated Time**: 5-6 hours
**Can Migrate Now**: ⚠️ RISKY (migrate LAST after testing other migrations)

### Overview
Main InputMethodService implementation. Orchestrates all keyboard functionality, manages lifecycle, coordinates between components.

### Structure Analysis
- **Base Class**: `InputMethodService` (Android IME framework)
- **Implements**:
  - `SharedPreferences.OnSharedPreferenceChangeListener`
  - `SuggestionBar.OnSuggestionSelectedListener`
  - `ConfigChangeListener`
- **Private Fields**: 65 total
  - Manager instances: 15+ (LayoutManager, ClipboardManager, etc.)
  - UI components: 5+ (Keyboard2View, SuggestionBar, etc.)
  - State: 10+ (config, action ID, handlers, etc.)
- **Methods**: ~50 methods
  - Lifecycle: `onCreate()`, `onDestroy()`, `onStartInput()`, etc.
  - View management: `onCreateInputView()`, `setInputView()`
  - Input handling: `handleRegularTyping()`, `handleBackspace()`, etc.
  - Configuration: `onConfigurationChanged()`, `refreshConfig()`

### Key Components

#### Manager Instances (Already Kotlin ✅)
All manager classes have already been migrated to Kotlin:
```java
private LayoutManager _layoutManager;
private ClipboardManager _clipboardManager;
private ConfigurationManager _configManager;
private PredictionCoordinator _predictionCoordinator;
private PredictionContextTracker _contextTracker;
private ContractionManager _contractionManager;
private InputCoordinator _inputCoordinator;
private SuggestionHandler _suggestionHandler;
private NeuralLayoutHelper _neuralLayoutHelper;
private SubtypeManager _subtypeManager;
private KeyboardReceiver _receiver;
private KeyEventReceiverBridge _receiverBridge;
private MLDataCollector _mlDataCollector;
```

**Kotlin Migration**:
- All nullable: `private var _layoutManager: LayoutManager? = null`
- Initialize in lifecycle methods (onCreate, onCreateInputView)
- Use `?.` safe calls everywhere

#### View Components
```java
private Keyboard2View _keyboardView;
private KeyEventHandler _keyeventhandler;
private ViewGroup _emojiPane = null;
private FrameLayout _contentPaneContainer = null;
private SuggestionBar _suggestionBar;
private LinearLayout _inputViewContainer;
```

**Kotlin Migration**:
- `private var _keyboardView: Keyboard2View? = null`
- `private var _keyeventhandler: KeyEventHandler? = null`
- Already initialized as null - keep nullable

#### State Fields
```java
private Config _config;
private Handler _handler;
public int actionId;
```

**Kotlin Migration**:
- `private var _config: Config? = null`
- `private lateinit var _handler: Handler`
- `var actionId: Int = 0` (public var, keep accessible)

### Migration Steps

1. **Convert Class Declaration**
   ```kotlin
   class Keyboard2 : InputMethodService(),
       SharedPreferences.OnSharedPreferenceChangeListener,
       SuggestionBar.OnSuggestionSelectedListener,
       ConfigChangeListener {
   ```

2. **Convert Field Declarations**
   - All managers → nullable: `private var _manager: Manager? = null`
   - All views → nullable or lateinit based on initialization
   - Constants → companion object

3. **Convert Lifecycle Methods**
   - `onCreate()` → initialize handlers, register receivers
   - `onDestroy()` → cleanup with safe calls
   - `onCreateInputView()` → inflate and initialize views
   - `onStartInput()` → handle editor info, update state
   - `onFinishInput()` → cleanup state

4. **Convert Input Handling**
   - `handleRegularTyping(String)` → delegate to InputCoordinator
   - `handleBackspace()` → delegate to InputCoordinator
   - `handleDeleteLastWord()` → delegate to InputCoordinator
   - Use safe calls for all delegations: `_inputCoordinator?.handleTyping(text)`

5. **Convert Configuration**
   - `onConfigurationChanged()` → update UI state
   - `refreshConfig()` → reload all managers with new config
   - `onSharedPreferenceChanged()` → handle pref updates

6. **Convert View Management**
   - `setInputView()` → override with View? parameter
   - `getCurrentInputView()` → return View?
   - Safe call chains for view access

7. **Convert Event Handlers**
   - `onSuggestionSelected()` → delegate to SuggestionHandler
   - `onConfigChanged()` → update cached config reference

### Critical Considerations

**⚠️ HIGH RISK AREAS**:
1. **InputMethodService Lifecycle**
   - Complex Android framework interactions
   - View creation and destruction timing
   - Configuration changes during input

2. **Null Safety**
   - Many managers initialized in different lifecycle stages
   - Must use safe calls consistently
   - Avoid lateinit where initialization is conditional

3. **Public API**
   - `actionId` must remain public for KeyEventHandler access
   - `getCurrentInputConnection()` called from multiple managers
   - `getConnectionToken()` for IBinder access

4. **Threading**
   - Handler operations must maintain timing
   - View updates on UI thread only
   - Safe access from background threads

### Testing Requirements
- ✅ Compile check only (R8 bug blocks builds)
- ⏸️ Runtime testing deferred until R8 fix
- 📋 **CRITICAL** Manual test plan (MUST pass all):
  1. Launch keyboard in any app
  2. Type characters (tap each key)
  3. Swipe typing gesture
  4. Backspace single character
  5. Backspace to delete word
  6. Show/hide emoji pane
  7. Show/hide clipboard pane
  8. Switch layouts (text → numeric → emoji → text)
  9. Shift/Caps lock
  10. Compose key sequences
  11. Voice typing switch
  12. IME action button (return/send/go)
  13. Auto-capitalization (start of sentence)
  14. Prediction suggestions
  15. Suggestion selection
  16. Configuration change (rotate device)
  17. App switch (keyboard hide/show)
  18. Multiple input fields in same app

### Dependencies
**Used By**: Android system (InputMethodService contract)
**Uses**: ALL managers and components (entire codebase)

**Migration Blockers**:
- ⚠️ Should be migrated LAST
- ⚠️ Requires R8 bug fix for testing
- ⚠️ Needs SwipeCalibrationActivity and SettingsActivity migrated first

---

## 🎯 Migration Strategy

### Phase 1: Low-Risk Activities (Can Do Now)
1. ✅ Migrate **SwipeCalibrationActivity.java** → Kotlin
   - Standalone activity, no core dependencies
   - Easy to verify compilation
   - 2-3 hours

2. ✅ Migrate **SettingsActivity.java** → Kotlin
   - UI only, no runtime keyboard logic
   - Independent preference handlers
   - 4-5 hours

**Commit after each migration** with comprehensive commit message.

### Phase 2: Core Service (Wait for R8 Fix)
3. ⏸️ **WAIT** for R8/D8 bug fix
   - Monitor Android Gradle Plugin updates
   - Test with newer AGP versions as released
   - Verify last working build (v1.32.860) still works

4. ✅ Once builds work: Migrate **Keyboard2.java** → Kotlin
   - **CRITICAL**: Most important file
   - Requires full runtime testing
   - 5-6 hours + extensive testing

### Phase 3: Post-Migration
5. ✅ Full regression testing
   - Test all 18 scenarios from Keyboard2 test plan
   - Test calibration activity
   - Test all settings screens
   - Performance benchmarks

6. ✅ Update documentation
   - Mark migration 100% complete
   - Update CHANGELOG.md
   - Create migration retrospective

---

## 🔧 Migration Commands

### For Each File:
```bash
# 1. Create Kotlin file
cp srcs/juloo.keyboard2/FileName.java srcs/juloo.keyboard2/FileName.kt

# 2. Convert to Kotlin syntax
# (manual editing)

# 3. Delete Java file
git rm srcs/juloo.keyboard2/FileName.java

# 4. Verify compilation
./gradlew compileDebugKotlin

# 5. Commit
git add -A
git commit -m "refactor(migration): Migrate FileName.java to Kotlin

- Converted to Kotlin with proper null safety
- All fields properly typed (nullable/lateinit)
- Methods converted to Kotlin syntax
- (Size: X → Y lines, -Z%)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Verification (When R8 Fixed):
```bash
# Full build
./build-on-termux.sh

# Install
adb install -r build/outputs/apk/debug/juloo.keyboard2.debug.apk

# Test
# (manual testing of all scenarios)
```

---

## 📊 Migration Progress Tracking

| File | Lines | Status | Compile | Runtime | Notes |
|------|-------|--------|---------|---------|-------|
| SwipeCalibrationActivity.java | 1,321 | ⏸️ Ready | ⏸️ Blocked (R8) | ⏸️ Blocked (R8) | Can migrate now |
| SettingsActivity.java | 2,051 | ⏸️ Ready | ⏸️ Blocked (R8) | ⏸️ Blocked (R8) | Can migrate now |
| Keyboard2.java | 698 | ⏸️ Ready | ⏸️ Blocked (R8) | ⏸️ Blocked (R8) | Migrate LAST |

**Total Remaining**: 4,070 lines (2.7% of codebase)

---

## 🚧 Current Blockers

1. **R8/D8 Bug** (NullPointerException in R8 8.6.17)
   - Blocks all compilation verification
   - Blocks all runtime testing
   - See [R8-BUG-WORKAROUND.md](../R8-BUG-WORKAROUND.md)

2. **AAPT2 ARM64 Wrapper**
   - Gradle cache corruption from Gradle 8.9 experiment
   - Solution: Use `./build-on-termux.sh` script only
   - Never run `./gradlew` directly on Termux ARM64

---

## ✅ Recommendation

**WAIT for R8 bug fix before migrating these files.**

**Reasoning**:
1. Cannot verify compilation (R8 crashes during DEX)
2. Cannot test runtime behavior (no APK builds)
3. Risk of introducing bugs that can't be detected
4. Current Kotlin migration is already 98.6% complete
5. These 3 files are NOT blocking any other work

**Alternative**: If R8 fix takes too long (>1 month), consider:
- Migrating SwipeCalibrationActivity and SettingsActivity anyway (lower risk)
- Deferring Keyboard2.java until R8 is fixed (highest risk)
- Using v1.32.860 build for production until migration completes

---

**Last Updated**: 2025-11-26
**Status**: Ready for migration pending R8 bug fix
