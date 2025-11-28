# Kotlin Migration Plan - Remaining Files

## Current Status (2025-11-26) - UPDATED

**⚠️ NOTE**: This file is now superseded by comprehensive migration documentation.

**See instead**:
- **[../MIGRATION_STATUS.md](../MIGRATION_STATUS.md)** - Complete overview
- **[../MIGRATION_RESUME_CHECKLIST.md](../MIGRATION_RESUME_CHECKLIST.md)** - Execution plan
- **[../docs/REMAINING_JAVA_MIGRATION.md](../docs/REMAINING_JAVA_MIGRATION.md)** - Main file plans
- **[../docs/JAVA_TEST_MIGRATION.md](../docs/JAVA_TEST_MIGRATION.md)** - Test file plans

**Migration Progress**: 98.6% (145/148 files)
**Test Coverage**: 38 test files (30 Kotlin + 8 Java)
**Build Status**: ⚠️ Kotlin compilation ✅ | DEX compilation ❌ (R8 8.6.17 bug)
**Blocker**: R8/D8 bug prevents APK builds (see [../R8-BUG-WORKAROUND.md](../R8-BUG-WORKAROUND.md))

## Remaining Files (7 files, 7,724 lines)

### Priority 1: Data Classes (Recommended First) ⭐

#### 1. KeyValue.java (868 lines) - HIGH COMPLEXITY
**Type**: Immutable value class with bit-packing
**Complexity**: HIGH - Core data class used throughout codebase

**Key Features**:
- 5 inner enums: Event, Modifier, Editing, Placeholder, Kind
- Bit-packed encoding: FLAGS (8 bits) + KIND (4 bits) + VALUE (20 bits)
- 32 static factory methods
- Immutable design with `with*` methods
- Comparable implementation
- Inner classes: Slider, Macro

**Migration Strategy**:
```kotlin
// Convert to sealed class hierarchy or keep as value class
data class KeyValue private constructor(
    private val payload: Comparable<*>,
    private val code: Int
) : Comparable<KeyValue> {

    enum class Kind { Char, Keyevent, Event, ... }

    companion object {
        // 32 factory methods here
        fun makeChar(c: Char, flags: Int = 0): KeyValue = ...
        fun makeKeyevent(code: Int, symbol: String): KeyValue = ...
    }

    // Bit manipulation methods
    fun getKind(): Kind = Kind.values()[(code and KIND_BITS) ushr KIND_OFFSET]
}
```

**Test Requirements**:
- 50+ test methods minimum
- Test all 32 factory methods
- Test bit manipulation (getKind, getFlags, getValue)
- Test all accessor methods (getChar, getEvent, getModifier, etc.)
- Test immutability (with* methods create new instances)
- Test compareTo and equals
- Test edge cases (max values, flag combinations)

**Estimated Time**: 2-3 hours
**Risk**: HIGH (core class, breaking changes affect entire codebase)

---

#### 2. KeyboardData.java (703 lines) - MEDIUM COMPLEXITY
**Type**: Keyboard layout data model
**Complexity**: MEDIUM - Data structure with layout logic

**Key Features**:
- Layout representation (rows, keys, modifiers)
- Key lookup and positioning
- Layout transformation methods
- Numpad and pinentry variants

**Migration Strategy**:
```kotlin
data class KeyboardData(
    val rows: List<Row>,
    val name: String?,
    val script: String?
) {
    data class Row(val keys: List<Key>, val height: Float)
    data class Key(val key: KeyValue, val width: Float, val shift: Float)

    fun findKeyWithValue(kv: KeyValue): Key? = ...
}
```

**Test Requirements**:
- 40+ test methods
- Test layout construction
- Test key lookup
- Test coordinate calculations
- Test layout transformations

**Estimated Time**: 2 hours
**Risk**: MEDIUM (data model, well-defined behavior)

---

### Priority 2: View/Touch Classes

#### 3. Pointers.java (1,048 lines) - HIGH COMPLEXITY
**Type**: Multi-touch and gesture tracking
**Complexity**: HIGH - Complex state machine

**Key Features**:
- Multi-touch tracking
- Gesture detection (swipe, long-press)
- Modifier state management
- Touch event processing

**Migration Strategy**: Extract gesture detection logic, use Kotlin coroutines for async handling

**Estimated Time**: 3-4 hours
**Risk**: HIGH (touch handling is critical)

---

#### 4. Keyboard2View.java (1,035 lines) - HIGH COMPLEXITY
**Type**: Custom View rendering keyboard
**Complexity**: HIGH - Canvas drawing, touch handling

**Key Features**:
- Custom drawing (onDraw)
- Touch event delegation
- Layout management
- Animation

**Migration Strategy**: Kotlin with careful Canvas API usage

**Estimated Time**: 3-4 hours
**Risk**: HIGH (UI rendering critical)

---

### Priority 3: Activities

#### 5. SwipeCalibrationActivity.java (1,321 lines) - MEDIUM COMPLEXITY
**Type**: Calibration UI activity
**Complexity**: MEDIUM - Activity with ML integration

**Key Features**:
- Calibration gesture collection
- ML model testing
- Settings persistence

**Migration Strategy**: Standard Activity → Kotlin with viewBinding

**Estimated Time**: 2-3 hours
**Risk**: MEDIUM (isolated feature)

---

#### 6. SettingsActivity.java (2,051 lines) - HIGH COMPLEXITY
**Type**: Main settings activity
**Complexity**: HIGH - Large preference UI

**Key Features**:
- Extensive preference management
- Dynamic preference creation
- Theme handling
- Export/import

**Migration Strategy**: Break into fragments, use PreferenceFragmentCompat

**Estimated Time**: 4-5 hours
**Risk**: MEDIUM (UI, but well-tested)

---

### Priority 4: Main Orchestrator (LAST)

#### 7. Keyboard2.java (698 lines) - VERY HIGH COMPLEXITY
**Type**: InputMethodService orchestrator
**Complexity**: VERY HIGH - Central coordinator

**Key Features**:
- InputMethodService lifecycle
- Component initialization (15+ managers)
- View management
- Configuration coordination
- Receiver registration

**Why Last**:
- Depends on all other classes
- Most complex integration point
- Requires all helpers to be in Kotlin first
- High risk of breaking changes

**Migration Strategy**:
1. Ensure ALL other classes migrated first
2. Migrate in small chunks with intermediate commits
3. Test after each chunk
4. Keep Java version until 100% confident

**Estimated Time**: 5-6 hours
**Risk**: VERY HIGH (breaks everything if wrong)

---

## Recommended Migration Order

### Session 1: KeyValue.java ⭐
- **Goal**: Migrate KeyValue to Kotlin
- **Steps**:
  1. Read entire KeyValue.java
  2. Create KeyValue.kt with all factory methods
  3. Create KeyValueTest.kt (50+ tests)
  4. Build and test
  5. Commit with comprehensive tests
- **Success Criteria**: All tests pass, no regressions

### Session 2: KeyboardData.java
- **Goal**: Migrate KeyboardData to Kotlin
- **Steps**:
  1. Migrate KeyboardData.kt
  2. Create KeyboardDataTest.kt (40+ tests)
  3. Build and test
  4. Commit
- **Success Criteria**: Layout loading works, no crashes

### Session 3: Pointers.java
- **Goal**: Migrate touch/gesture handling
- **Steps**:
  1. Migrate Pointers.kt
  2. Create PointersTest.kt
  3. Test touch handling extensively
  4. Commit
- **Success Criteria**: Touch input works perfectly

### Session 4: Keyboard2View.java
- **Goal**: Migrate view rendering
- **Steps**:
  1. Migrate Keyboard2View.kt
  2. Test rendering
  3. Commit
- **Success Criteria**: Keyboard displays correctly

### Session 5: Activities
- **Goal**: Migrate SwipeCalibrationActivity and SettingsActivity
- **Steps**:
  1. Migrate both activities
  2. Test UI flows
  3. Commit
- **Success Criteria**: All settings work, calibration works

### Session 6: Keyboard2.java (FINAL)
- **Goal**: Migrate main orchestrator
- **Steps**:
  1. Verify ALL dependencies migrated
  2. Migrate Keyboard2.kt in small chunks
  3. Test extensively after each chunk
  4. Final commit
- **Success Criteria**: 100% Kotlin, all features working

---

## Testing Strategy

### For Each Migration:
1. ✅ Create comprehensive test file BEFORE committing
2. ✅ Follow MockitoJUnitRunner patterns
3. ✅ Use Arrange-Act-Assert structure
4. ✅ Test edge cases and error handling
5. ✅ Run `./pre-commit-tests.sh` before commit
6. ✅ Update pm.md with progress

### Integration Testing:
- Run full build after each migration
- Test keyboard functionality end-to-end
- Test on real device via ADB
- Monitor logcat for crashes

---

## Risk Mitigation

### High-Risk Files (KeyValue, Pointers, Keyboard2View, Keyboard2):
1. Create feature branch for each
2. Migrate in smallest possible chunks
3. Commit after each successful chunk
4. Keep Java backup until fully tested
5. Test on device after migration

### If Something Breaks:
1. Check git log for last working commit
2. Use `git bisect` to find breaking change
3. Revert specific commit if needed
4. Fix issue before proceeding

---

## Success Metrics

- ✅ 100% Kotlin (0 Java files remaining)
- ✅ All tests passing
- ✅ No functionality regressions
- ✅ Build time not increased
- ✅ APK size not increased
- ✅ All features working on device

---

## Blockers / Issues

None currently identified.

---

## Notes

- KeyValue.java is highest priority but also highest risk
- Consider creating backup branch before starting
- Test thoroughly on device, not just unit tests
- Migration should maintain 100% backward compatibility
- All public APIs must preserve exact signatures during migration
