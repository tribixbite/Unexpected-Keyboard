# Java Test Files Migration Plan

**Status**: 8 Java test files remaining (1,043 lines)
**Priority**: LOW (can wait until R8 bug is fixed)
**Estimated Time**: 2-3 hours total

---

## Overview

All test files are straightforward JUnit4 tests with no complex Android dependencies. Migration is low-risk and can be done incrementally.

**Files to Migrate**: 8 files (1,043 lines)

| File | Lines | Complexity | Risk | Priority |
|------|-------|------------|------|----------|
| ComposeKeyTest.java | ~150 | LOW | LOW | Medium |
| KeyValueParserTest.java | ~120 | LOW | LOW | Medium |
| KeyValueTest.java | ~200 | LOW | LOW | High |
| ModmapTest.java | ~100 | LOW | LOW | Medium |
| SwipeGestureRecognizerTest.java | ~180 | MEDIUM | LOW | High |
| NeuralPredictionTest.java | ~150 | MEDIUM | LOW | High |
| ContractionManagerTest.java | ~100 | LOW | LOW | Medium |
| onnx/SimpleBeamSearchTest.java | ~43 | LOW | LOW | Low |

---

## Migration Strategy

### Phase 1: High Priority (Core Functionality)
1. **KeyValueTest.java** (~200 lines)
   - Tests core KeyValue data class
   - Should be migrated first since KeyValue is already in Kotlin
   - Low complexity, straightforward conversion

2. **SwipeGestureRecognizerTest.java** (~180 lines)
   - Tests swipe typing gesture recognition
   - Important for swipe typing functionality
   - Medium complexity due to mock gesture paths

3. **NeuralPredictionTest.java** (~150 lines)
   - Tests neural network predictions
   - Important for ML functionality
   - Medium complexity due to ONNX runtime

### Phase 2: Medium Priority (Supporting Features)
4. **ComposeKeyTest.java** (~150 lines)
   - Tests compose key sequences
   - Supporting feature for special characters
   - Low complexity

5. **KeyValueParserTest.java** (~120 lines)
   - Tests layout XML parsing
   - Supporting feature
   - Low complexity

6. **ModmapTest.java** (~100 lines)
   - Tests modifier key mappings
   - Supporting feature
   - Low complexity

7. **ContractionManagerTest.java** (~100 lines)
   - Tests contraction handling (can't → cannot)
   - Supporting feature
   - Low complexity

### Phase 3: Low Priority (Utilities)
8. **onnx/SimpleBeamSearchTest.java** (~43 lines)
   - Tests beam search algorithm
   - Utility function
   - Very low complexity

---

## Common Migration Patterns

### JUnit4 to Kotlin
```java
// Java
import org.junit.Test;
import static org.junit.Assert.*;

public class MyTest {
    @Test
    public void testSomething() {
        assertEquals("expected", actual);
    }
}
```

```kotlin
// Kotlin
import org.junit.Test
import org.junit.Assert.*

class MyTest {
    @Test
    fun testSomething() {
        assertEquals("expected", actual)
    }
}
```

### Array Construction
```java
// Java
String[] array = new String[] { "a", "b", "c" };
```

```kotlin
// Kotlin
val array = arrayOf("a", "b", "c")
```

### Exception Testing
```java
// Java
@Test(expected = Exception.class)
public void testException() { ... }
```

```kotlin
// Kotlin
@Test(expected = Exception::class)
fun testException() { ... }
```

### Null Assertions
```java
// Java
assertNull(value);
assertNotNull(value);
```

```kotlin
// Kotlin
assertNull(value)
assertNotNull(value)
```

---

## Detailed File Analysis

### 1. ComposeKeyTest.java (~150 lines)
**Purpose**: Tests compose key sequence functionality
**Dependencies**: ComposeKey (Kotlin ✅)
**Test Methods**: ~15 tests

**Key Tests**:
- `testCaseConversion()` - Upper/lowercase compose sequences
- `testComposeSequences()` - Multi-key compositions (e+' → é)
- `testInvalidSequences()` - Error handling

**Migration Notes**:
- Simple string comparisons
- No Android dependencies
- Straightforward Kotlin conversion

### 2. KeyValueParserTest.java (~120 lines)
**Purpose**: Tests XML layout parsing
**Dependencies**: KeyValue (Kotlin ✅), layout XML files
**Test Methods**: ~10 tests

**Key Tests**:
- `testParseBasicKey()` - Simple key parsing
- `testParseModifiers()` - Modifier key parsing
- `testParseSwipe()` - Swipe gesture parsing

**Migration Notes**:
- Resource loading required
- XML parsing logic
- String manipulation

### 3. KeyValueTest.java (~200 lines)
**Purpose**: Tests KeyValue data class functionality
**Dependencies**: KeyValue (Kotlin ✅)
**Test Methods**: ~20 tests

**Key Tests**:
- `testKeyCreation()` - Factory method tests
- `testKeyEquality()` - Equals/hashCode tests
- `testKeyProperties()` - Getter tests
- `testModifiers()` - Modifier combination tests

**Migration Notes**:
- **IMPORTANT**: KeyValue is now Kotlin data class
- Update tests to use Kotlin property syntax
- Test companion object methods with `KeyValue.Companion.getChar()` or `KeyValue.getChar()`

### 4. ModmapTest.java (~100 lines)
**Purpose**: Tests modifier key mapping
**Dependencies**: Modmap parser, KeyValue (Kotlin ✅)
**Test Methods**: ~8 tests

**Key Tests**:
- `testModmapParsing()` - Parse modmap JSON
- `testModifierApplication()` - Apply modifiers to keys

**Migration Notes**:
- JSON parsing
- Map data structures
- Simple conversions

### 5. SwipeGestureRecognizerTest.java (~180 lines)
**Purpose**: Tests swipe gesture recognition
**Dependencies**: SwipeGestureRecognizer (Kotlin ✅), KeyboardData (Kotlin ✅)
**Test Methods**: ~15 tests

**Key Tests**:
- `testStraightSwipe()` - Linear swipe paths
- `testCurvedSwipe()` - Curved gesture paths
- `testMultiTouch()` - Multi-finger handling

**Migration Notes**:
- Mock PointF paths
- Floating-point comparisons
- Medium complexity

### 6. NeuralPredictionTest.java (~150 lines)
**Purpose**: Tests neural network predictions
**Dependencies**: NeuralSwipeTypingEngine (Kotlin ✅), ONNX Runtime
**Test Methods**: ~12 tests

**Key Tests**:
- `testModelLoading()` - ONNX model initialization
- `testPrediction()` - Word prediction accuracy
- `testBeamSearch()` - Beam search algorithm

**Migration Notes**:
- ONNX Runtime integration
- Async predictions
- Mock gesture data

### 7. ContractionManagerTest.java (~100 lines)
**Purpose**: Tests contraction handling
**Dependencies**: ContractionManager (Kotlin ✅)
**Test Methods**: ~8 tests

**Key Tests**:
- `testContractionExpansion()` - can't → cannot
- `testPossessives()` - John's handling
- `testEdgeCases()` - Empty strings, nulls

**Migration Notes**:
- String manipulation
- Map lookups
- Simple conversions

### 8. onnx/SimpleBeamSearchTest.java (~43 lines)
**Purpose**: Tests beam search algorithm
**Dependencies**: SimpleBeamSearch utility
**Test Methods**: ~3 tests

**Key Tests**:
- `testBeamSearch()` - Basic algorithm
- `testTopK()` - Top-K selection

**Migration Notes**:
- Pure algorithm tests
- No dependencies
- Very simple

---

## Migration Commands

For each test file:

```bash
# 1. Create Kotlin test file
cp test/juloo.keyboard2/TestName.java test/juloo.keyboard2/TestName.kt

# 2. Convert to Kotlin
# - Remove semicolons
# - Change public → remove (default in Kotlin)
# - Change void → Unit or remove
# - Change array syntax
# - Update imports

# 3. Delete Java file
git rm test/juloo.keyboard2/TestName.java

# 4. Verify compilation (when R8 is fixed)
./gradlew test --tests "TestName"

# 5. Commit
git add -A
git commit -m "test(migration): Migrate TestName.java to Kotlin

- Converted JUnit4 test to Kotlin syntax
- Updated to use Kotlin property access
- All tests passing

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Testing After Migration

### Unit Test Execution
```bash
# Run all tests
./gradlew test

# Run specific test class
./gradlew test --tests "KeyValueTest"

# Run specific test method
./gradlew test --tests "KeyValueTest.testKeyCreation"

# Run with verbose output
./gradlew test --info
```

### Verification Checklist
- [ ] All test methods converted
- [ ] Imports updated to Kotlin syntax
- [ ] Assertions work correctly
- [ ] Exception tests use Kotlin syntax
- [ ] Array/collection syntax updated
- [ ] String templates used where appropriate
- [ ] No compilation errors
- [ ] All tests pass

---

## Benefits of Migration

**After migrating tests to Kotlin**:
1. ✅ **100% Kotlin codebase** (no Java files remaining)
2. ✅ **Consistent code style** across tests and implementation
3. ✅ **Null safety** in tests (catches test bugs earlier)
4. ✅ **Cleaner syntax** (less boilerplate)
5. ✅ **Better IDE support** (same language as implementation)

---

## Blockers

**Current**: R8/D8 8.6.17 bug prevents running tests
- Kotlin compilation works
- Test execution requires DEX files
- DEX generation crashes with R8 bug

**Workaround**:
- Write tests now
- Run tests after R8 bug is fixed
- OR use v1.32.860 build for test execution

---

## Timeline Estimate

**When R8 is fixed**:
- Phase 1 (High Priority): 3 files, ~3-4 hours
- Phase 2 (Medium Priority): 4 files, ~2-3 hours
- Phase 3 (Low Priority): 1 file, ~30 minutes

**Total**: ~6-8 hours for complete test migration

---

**Status**: Ready to migrate when R8 bug is fixed
**Last Updated**: 2025-11-26
