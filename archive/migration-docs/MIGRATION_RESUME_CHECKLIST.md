# Kotlin Migration Resume Checklist

**When R8/D8 bug is fixed - Use this checklist to resume migration**

---

## Prerequisites ✓

Before starting, verify:

- [ ] R8 bug is fixed in new AGP version
- [ ] AGP upgrade is compatible with current Gradle version
- [ ] AAPT2 ARM64 wrapper still works with new AGP
- [ ] Test build with existing Kotlin files succeeds
- [ ] All existing tests pass

**Test Command**:
```bash
./build-on-termux.sh
# Should complete successfully without R8/D8 NullPointerException
```

---

## Phase 1: SwipeCalibrationActivity Migration (2-3 hours)

**File**: `srcs/juloo.keyboard2/SwipeCalibrationActivity.java` (1,321 lines)

**Reference**: See [docs/REMAINING_JAVA_MIGRATION.md](docs/REMAINING_JAVA_MIGRATION.md) lines 26-197

### Steps:

1. **Pre-Migration Verification**
   - [ ] Read complete migration plan in REMAINING_JAVA_MIGRATION.md
   - [ ] Review SwipeCalibrationActivity structure (27 fields, ~40 methods)
   - [ ] Verify all dependencies are Kotlin (NeuralSwipeTypingEngine, Config, SwipeMLDataStore)
   - [ ] Run existing tests: `./gradlew test --tests "*SwipeCalibration*"`

2. **Create Kotlin File**
   - [ ] Copy to: `srcs/juloo.keyboard2/SwipeCalibrationActivity.kt`
   - [ ] Update imports to Kotlin syntax
   - [ ] Convert class declaration to Kotlin

3. **Field Migration**
   - [ ] Convert 11 UI components to nullable vars: `private var _instructionText: TextView? = null`
   - [ ] Convert 16 state variables (use lateinit for non-null)
   - [ ] Convert collections: `ArrayList` → `mutableListOf`, `HashMap` → `mutableMapOf`

4. **Method Migration**
   - [ ] Convert lifecycle methods (onCreate, onResume, onPause)
   - [ ] Convert UI event handlers (onClick, onTouch)
   - [ ] Convert data collection methods
   - [ ] Update null safety with safe calls (`?.`) and Elvis operators (`?:`)

5. **Testing**
   - [ ] Build: `./build-on-termux.sh`
   - [ ] Fix any compilation errors
   - [ ] Run unit tests
   - [ ] Test on device (calibration flow works)
   - [ ] Verify data collection saves correctly

6. **Commit**
   - [ ] Delete Java file: `git rm srcs/juloo.keyboard2/SwipeCalibrationActivity.java`
   - [ ] Commit with message:
   ```
   feat(migration): migrate SwipeCalibrationActivity to Kotlin

   - Converted 1,321 lines to Kotlin with null safety
   - All UI components properly nullable
   - Data collection methods preserved
   - All tests passing

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   ```

---

## Phase 2: SettingsActivity Migration (4-5 hours)

**File**: `srcs/juloo.keyboard2/SettingsActivity.java` (2,051 lines)

**Reference**: See [docs/REMAINING_JAVA_MIGRATION.md](docs/REMAINING_JAVA_MIGRATION.md) lines 199-340

### Steps:

1. **Pre-Migration Verification**
   - [ ] Read complete migration plan
   - [ ] Review SettingsActivity structure (20+ preference fragments)
   - [ ] Verify Config.java is Kotlin (it is - Config.kt exists)
   - [ ] Check for Android framework dependencies
   - [ ] Run existing tests: `./gradlew test --tests "*Settings*"`

2. **Create Kotlin File**
   - [ ] Copy to: `srcs/juloo.keyboard2/SettingsActivity.kt`
   - [ ] Update imports (android.preference.* → AndroidX if possible)
   - [ ] Convert class declaration

3. **Fragment Migration** (20+ fragments)
   - [ ] Convert inner PreferenceFragment classes
   - [ ] Update preference loading: `addPreferencesFromResource` → `preferenceManager.setSharedPreferencesName`
   - [ ] Convert preference change listeners
   - [ ] Handle nullable types for preference values

4. **State Management**
   - [ ] Convert SavedInstanceState handling
   - [ ] Update lifecycle callbacks
   - [ ] Convert result codes and intents

5. **Testing**
   - [ ] Build: `./build-on-termux.sh`
   - [ ] Fix compilation errors
   - [ ] Run unit tests
   - [ ] Test on device:
     - [ ] All preference screens load
     - [ ] Settings save correctly
     - [ ] No crashes on navigation
     - [ ] Config updates propagate to keyboard

6. **Commit**
   - [ ] Delete Java file
   - [ ] Commit with conventional message

---

## Phase 3: Keyboard2 Migration (5-6 hours) - LAST, HIGHEST RISK

**File**: `srcs/juloo.keyboard2/Keyboard2.java` (698 lines)

**Reference**: See [docs/REMAINING_JAVA_MIGRATION.md](docs/REMAINING_JAVA_MIGRATION.md) lines 342-493

**⚠️ CRITICAL**: This is the InputMethodService - migrate LAST after all others succeed

### Steps:

1. **Pre-Migration Verification**
   - [ ] ✅ SwipeCalibrationActivity migrated and tested
   - [ ] ✅ SettingsActivity migrated and tested
   - [ ] ✅ All tests passing
   - [ ] ✅ App works correctly with first 2 migrations
   - [ ] Read complete migration plan THOROUGHLY
   - [ ] Review Keyboard2 structure (InputMethodService lifecycle)
   - [ ] Verify all dependencies are Kotlin (Keyboard2View, Config, etc.)

2. **Create Kotlin File**
   - [ ] **BACKUP FIRST**: `git commit -m "chore: backup before Keyboard2 migration"`
   - [ ] Copy to: `srcs/juloo.keyboard2/Keyboard2.kt`
   - [ ] Convert InputMethodService inheritance
   - [ ] Update lifecycle methods

3. **State Management** (CRITICAL)
   - [ ] Convert _config field (lateinit or nullable)
   - [ ] Convert _keyboardView (nullable)
   - [ ] Convert _theme (nullable)
   - [ ] Handle InputConnection safely (always nullable)

4. **Service Lifecycle** (HIGH RISK)
   - [ ] Convert onCreate() - initialization order matters
   - [ ] Convert onCreateInputView() - must return non-null View
   - [ ] Convert onStartInput() - handle null EditorInfo safely
   - [ ] Convert onFinishInput() - cleanup properly
   - [ ] Convert onDestroy() - prevent leaks

5. **Input Handling** (CRITICAL)
   - [ ] Convert onKeyDown/onKeyUp - maintain event flow
   - [ ] Convert sendKeyChar - encoding issues possible
   - [ ] Convert commitText - InputConnection null safety
   - [ ] Convert deleteSurroundingText - boundary checks

6. **Testing** (EXTENSIVE)
   - [ ] Build: `./build-on-termux.sh`
   - [ ] Fix ALL compilation errors
   - [ ] Run ALL unit tests: `./gradlew test`
   - [ ] Run integration tests: `./gradlew test --tests "*Integration*"`
   - [ ] Device testing (comprehensive):
     - [ ] Keyboard activates on input field tap
     - [ ] All keys type correctly
     - [ ] Swipe gestures work (corner swipes)
     - [ ] Shift/modifier keys work
     - [ ] Special keys work (delete, enter, etc.)
     - [ ] Emoji keyboard works
     - [ ] Voice input switching works
     - [ ] Clipboard history works
     - [ ] Swipe typing predictions work
     - [ ] Settings changes apply correctly
     - [ ] No crashes on rotation
     - [ ] No memory leaks (test for 10+ minutes)

7. **Performance Verification**
   - [ ] Monitor logcat for errors
   - [ ] Check memory usage (no leaks)
   - [ ] Verify typing latency (should be instant)
   - [ ] Test in multiple apps (Discord, browser, etc.)

8. **Commit** (Only after ALL tests pass)
   - [ ] Delete Java file
   - [ ] Commit with message:
   ```
   feat(migration): migrate Keyboard2 InputMethodService to Kotlin

   CRITICAL: Core InputMethodService migrated to Kotlin
   - Converted 698 lines with complete null safety
   - All lifecycle methods properly handled
   - Input handling preserved with safe calls
   - Extensive testing completed (18 scenarios)
   - Zero regressions confirmed

   Migration now 100% complete - entire codebase is Kotlin!

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   ```

---

## Phase 4: Test Files Migration (6-8 hours)

**Files**: 8 Java test files (1,043 lines total)

**Reference**: See [docs/JAVA_TEST_MIGRATION.md](docs/JAVA_TEST_MIGRATION.md)

### High Priority Tests (3-4 hours):

1. **KeyValueTest.java** (~200 lines)
   - [ ] Migrate to KeyValueTest.kt
   - [ ] Update to use Kotlin property syntax
   - [ ] Test companion object methods
   - [ ] Run: `./gradlew test --tests "KeyValueTest"`

2. **SwipeGestureRecognizerTest.java** (~180 lines)
   - [ ] Migrate to SwipeGestureRecognizerTest.kt
   - [ ] Update mock gesture paths to Kotlin collections
   - [ ] Run: `./gradlew test --tests "SwipeGestureRecognizerTest"`

3. **NeuralPredictionTest.java** (~150 lines)
   - [ ] Migrate to NeuralPredictionTest.kt
   - [ ] Update ONNX runtime async handling
   - [ ] Run: `./gradlew test --tests "NeuralPredictionTest"`

### Medium Priority Tests (2-3 hours):

4. **ComposeKeyTest.java** (~150 lines)
5. **KeyValueParserTest.java** (~120 lines)
6. **ModmapTest.java** (~100 lines)
7. **ContractionManagerTest.java** (~100 lines)

### Low Priority Tests (30 minutes):

8. **onnx/SimpleBeamSearchTest.java** (~43 lines)

**Common Pattern**:
```kotlin
// Before (Java)
@Test
public void testSomething() {
    assertEquals("expected", actual);
}

// After (Kotlin)
@Test
fun testSomething() {
    assertEquals("expected", actual)
}
```

---

## Phase 5: Final Verification (4-6 hours)

### Full Test Suite

- [ ] Run all tests: `./gradlew test`
- [ ] Verify 100% test pass rate
- [ ] Check test coverage report
- [ ] Fix any failing tests

### Build Verification

- [ ] Clean build: `./gradlew clean && ./build-on-termux.sh`
- [ ] Verify APK size (should be similar or smaller)
- [ ] Check release build: `./build-on-termux.sh release`
- [ ] Verify R8/ProGuard works for release builds

### Performance Benchmarks

- [ ] Measure typing latency (should be <16ms)
- [ ] Measure swipe prediction time (should be <100ms)
- [ ] Check memory usage (should be <50MB)
- [ ] Monitor for memory leaks (10+ minute test)

### Device Testing (18 Scenarios)

**Basic Functionality**:
- [ ] 1. Keyboard activates on input tap
- [ ] 2. All alphanumeric keys work
- [ ] 3. Shift key capitalizes
- [ ] 4. Number/symbol layer works
- [ ] 5. Delete key works
- [ ] 6. Enter/Return key works

**Advanced Features**:
- [ ] 7. Corner swipe gestures work
- [ ] 8. Circle gesture works
- [ ] 9. Emoji keyboard works
- [ ] 10. Voice input switching works
- [ ] 11. Clipboard history works
- [ ] 12. Settings changes apply

**Swipe Typing**:
- [ ] 13. Swipe gestures recognized
- [ ] 14. Predictions appear correctly
- [ ] 15. Top prediction is accurate
- [ ] 16. Suggestion bar updates

**Edge Cases**:
- [ ] 17. Screen rotation no crashes
- [ ] 18. Low memory no crashes

### Regression Testing

- [ ] Compare with v1.32.860 (last working build)
- [ ] Verify no performance degradation
- [ ] Verify no new crashes
- [ ] Verify all features still work

---

## Phase 6: Cleanup & Documentation (2-3 hours)

### Code Cleanup

- [ ] Remove any temporary workarounds
- [ ] Fix any remaining Kotlin warnings
- [ ] Update KDoc comments
- [ ] Format code consistently

### Documentation Updates

- [ ] Update MIGRATION_STATUS.md to 100% complete
- [ ] Update memory/pm.md with completion details
- [ ] Create migration retrospective
- [ ] Update CHANGELOG.md with migration notes

### Final Commits

- [ ] Bump version to v1.33.0 (major migration milestone)
- [ ] Create tag: `git tag v1.33.0-kotlin-complete`
- [ ] Push to remote: `git push origin feature/swipe-typing --tags`

---

## Success Criteria

**Migration is complete when**:

- [x] 98.6% Kotlin (current - 145/148 files)
- [ ] **100% Kotlin** (148/148 files)
- [ ] **Zero Java files** in srcs/juloo.keyboard2
- [ ] **Zero Java test files** (all 38 tests in Kotlin)
- [ ] **All tests pass** (100% pass rate)
- [ ] **APK builds successfully** (debug and release)
- [ ] **All features work** (18 test scenarios pass)
- [ ] **No regressions** (compared to v1.32.860)
- [ ] **Performance targets met** (<16ms latency, <100ms predictions)
- [ ] **Documentation complete** (CHANGELOG, retrospective)

---

## Estimated Timeline

**Total: 16-22 hours** (when R8 is fixed)

- Phase 1: SwipeCalibrationActivity (2-3 hours)
- Phase 2: SettingsActivity (4-5 hours)
- Phase 3: Keyboard2 (5-6 hours) - CRITICAL, highest risk
- Phase 4: Test files (6-8 hours)
- Phase 5: Verification (4-6 hours)
- Phase 6: Cleanup (2-3 hours)

**Recommended schedule**:
- Day 1 (8 hours): Phases 1-2 (SwipeCalibration + Settings)
- Day 2 (8 hours): Phase 3 + start Phase 4 (Keyboard2 + tests)
- Day 3 (6 hours): Finish Phase 4 + Phase 5 (tests + verification)
- Day 4 (4 hours): Phase 6 (cleanup + docs)

---

## Emergency Rollback

**If critical issues arise during Keyboard2 migration**:

```bash
# Rollback to last working state
git reset --hard HEAD~1  # Undo last commit
./build-on-termux.sh     # Build previous working version

# Or use last known good build
git checkout 2544cf9d    # v1.32.860 (Pointers migration)
./build-on-termux.sh
```

**Critical**: Always commit working states before risky migrations!

---

## Current Blockers

- ⏸️ **R8/D8 8.6.17 bug** - See [R8-BUG-WORKAROUND.md](R8-BUG-WORKAROUND.md)
- ⏸️ **Bug report submitted** - See [R8-BUG-REPORT.md](R8-BUG-REPORT.md)
- ⏸️ **Waiting for AGP update** with fixed R8 version

**Monitor**: https://developer.android.com/studio/releases/gradle-plugin

---

**Last Updated**: 2025-11-26
**Status**: Ready to resume when R8 is fixed
**Progress**: 98.6% complete (145/148 files)
