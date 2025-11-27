# Avoiding Integration Issues: Lessons Learned

## Problem: Tests Pass But App Crashes

### What Happened

During Phase 4 refactoring (v1.32.409), we extracted `SubtypeLayoutInitializer`:
- Created 104-line Kotlin utility
- Wrote 36 comprehensive unit tests
- All tests passed ✅
- Build succeeded ✅
- **App crashed on load** ❌

### Root Cause

**Unit tests with mocks don't catch runtime integration issues.**

```kotlin
// Unit test - uses mocks, doesn't catch real issues
@Mock private lateinit var mockKeyboard2: Keyboard2
@Mock private lateinit var mockConfig: Config
@Mock private lateinit var mockResources: Resources

val initializer = SubtypeLayoutInitializer(
    mockKeyboard2,   // Not a real service!
    mockConfig,      // Not real config!
    mockKeyboardView // Not a real view!
)
```

The mock objects behave perfectly in tests but real Android framework objects have:
- Initialization order requirements
- Lifecycle constraints
- Null safety issues
- Resource loading quirks

## Pattern Recognition

### Red Flags Indicating Need for Integration Tests

✅ **Always need integration tests when code:**

1. **Uses Android Framework APIs**
   - `Context`, `Resources`, `SharedPreferences`
   - `InputMethodService`, `View`, `ViewGroup`
   - `EditorInfo`, `InputConnection`
   - Any `android.*` imports

2. **Initializes Framework Components**
   - Creating `LayoutManager`, `SubtypeManager`
   - Loading `KeyboardData` from resources
   - Setting up Views or ViewGroups
   - Accessing system services

3. **Has Complex Initialization Order**
   - Depends on A being created before B
   - Requires null checks on initialization
   - Lazy initialization patterns
   - Lifecycle-dependent code

4. **Touches Real Android Resources**
   - `R.xml.*`, `R.layout.*`, `R.string.*`
   - File I/O or database access
   - System properties or settings

### Example: SubtypeLayoutInitializer

```kotlin
// RED FLAGS:
class SubtypeLayoutInitializer(
    private val keyboard2: Keyboard2,        // ⚠️ InputMethodService
    private val config: Config,              // ⚠️ Uses SharedPreferences
    private val keyboardView: Keyboard2View  // ⚠️ Custom View
) {
    fun refreshSubtypeAndLayout(...): InitializationResult {
        val subtypeManager = SubtypeManager(keyboard2)  // ⚠️ Creates framework object
        var defaultLayout = subtypeManager.refreshSubtype(config, resources)  // ⚠️ Framework call
        if (defaultLayout == null) {
            defaultLayout = KeyboardData.load(resources, R.xml.latn_qwerty_us)  // ⚠️ Resources!
        }
        val layoutManager = LayoutManager(keyboard2, config, defaultLayout)  // ⚠️ Complex init
        val layoutBridge = LayoutBridge.create(layoutManager, keyboardView)   // ⚠️ View dependency
    }
}
```

**5 red flags = definitely needs integration test!**

## Prevention Strategy

### Step 1: Identify Risk Level

Before creating a new utility, assess risk:

**Low Risk** (unit tests sufficient):
- Pure Kotlin/Java logic
- No Android framework usage
- Simple data transformations
- Stateless utilities

**High Risk** (needs integration tests):
- Android framework APIs ✅
- View/UI components ✅
- Resource loading ✅
- Initialization patterns ✅

### Step 2: Write Both Test Types

For high-risk components, write **both**:

#### Unit Tests (Fast, Isolated)
```kotlin
// test/juloo.keyboard2/SubtypeLayoutInitializerTest.kt
@RunWith(MockitoJUnitRunner::class)
class SubtypeLayoutInitializerTest {
    @Mock private lateinit var mockKeyboard2: Keyboard2
    // Tests with mocks - fast, but limited
}
```

#### Integration Tests (Slower, Real Framework)
```kotlin
// test/juloo.keyboard2/integration/SubtypeLayoutInitializerIntegrationTest.kt
@RunWith(AndroidJUnit4::class)
class SubtypeLayoutInitializerIntegrationTest {
    @Test
    fun testWithRealAndroidFramework() {
        val context = ApplicationProvider.getApplicationContext()
        val resources = context.resources

        // Use REAL Android components
        val layout = KeyboardData.load(resources, R.xml.latn_qwerty_us)
        assertNotNull("Should load real layout", layout)
    }
}
```

### Step 3: Use Test-and-Deploy Pipeline

**Never skip the deployment verification step!**

```bash
# ❌ BAD - Old workflow
./gradlew test       # Unit tests pass
./build-on-termux.sh # Build succeeds
# Install manually
# ⚠️ App crashes!

# ✅ GOOD - New workflow
./build-test-deploy.sh
# - Runs unit tests
# - Builds APK
# - Installs via ADB
# - Monitors logcat
# - Verifies no crashes
# ✅ Catches issues immediately!
```

## Updated Workflow

### For Every New Utility

1. **Create utility class** (Kotlin)
2. **Write unit tests** (20+ cases with mocks)
3. **Assess risk level** (check red flags above)
4. **If high risk:** Write integration test
5. **Run pre-commit tests**: `./pre-commit-tests.sh`
6. **Full deployment test**: `./build-test-deploy.sh`
7. **Monitor logcat** during first launch
8. **Manual verification** of functionality
9. **Commit only if all pass**

### Checklist Before Committing

- [ ] Unit tests written and passing
- [ ] Integration test added (if high risk)
- [ ] `./pre-commit-tests.sh` passes
- [ ] `./build-test-deploy.sh` succeeds
- [ ] No crashes in logcat
- [ ] Manual functionality verification
- [ ] Documentation updated

## Tools We Built

### 1. build-test-deploy.sh
Complete pipeline with crash detection:
```bash
./build-test-deploy.sh
# ═══════════════════════════════════════════
# STEP 1: Running Unit Tests
# ✅ All unit tests passed
# ═══════════════════════════════════════════
# STEP 2: Building APK
# ✅ APK built successfully
# ═══════════════════════════════════════════
# STEP 3: ADB Connection
# ✅ Connected to 192.168.1.247:5555
# ═══════════════════════════════════════════
# STEP 4: Installing APK
# ✅ APK installed successfully
# ═══════════════════════════════════════════
# STEP 5: Smoke Tests
# ✅ No crashes detected in initial 5s
# ═══════════════════════════════════════════
# STEP 6: Verification
# ✅ IME registered successfully
```

### 2. pre-commit-tests.sh
Fast verification before committing:
```bash
./pre-commit-tests.sh
# [1/4] Checking Kotlin/Java compilation...
# ✓ Compilation successful
# [2/4] Running unit tests...
# ✓ Unit tests passed (21 test classes)
# [3/4] Checking for unfinished work markers...
# ✓ No unfinished work markers
# [4/4] Checking version info...
# ✓ Version updated in build.gradle
```

### 3. smoke-test.sh
Post-install verification:
```bash
./smoke-test.sh
# Testing: Package installation...
# ✓ Package installed
# Testing: IME service registration...
# ✓ IME service registered
# Testing: IME enable without crash...
# ✓ IME enabled without crashes
```

## Common Mistakes and Fixes

### Mistake 1: Only Writing Unit Tests

**Problem**: Mocks hide real issues

**Fix**: Add integration tests for framework code

```kotlin
// ❌ Only this
@Test
fun testWithMocks() {
    `when`(mockResources.getBoolean(any())).thenReturn(true)
}

// ✅ Also add this
@Test
fun testWithRealResources() {
    val context = ApplicationProvider.getApplicationContext()
    val value = context.resources.getBoolean(R.bool.debug_logs)
    // Uses real resources, catches real issues!
}
```

### Mistake 2: Skipping Deployment Verification

**Problem**: Build succeeds but app crashes

**Fix**: Always run full pipeline

```bash
# ❌ Don't do this
./gradlew test && ./build-on-termux.sh
# Stops here - no crash detection!

# ✅ Do this instead
./build-test-deploy.sh
# Includes crash detection!
```

### Mistake 3: Not Monitoring Logcat

**Problem**: Miss early crash indicators

**Fix**: Monitor logcat on first run

```bash
# Terminal 1
adb logcat -c && adb logcat | grep -i "exception\|fatal\|keyboard"

# Terminal 2
./build-test-deploy.sh
```

### Mistake 4: Testing Only Happy Path

**Problem**: Edge cases cause crashes

**Fix**: Test null, empty, boundary cases

```kotlin
@Test fun testWithNullManager() { ... }
@Test fun testWithNullResources() { ... }
@Test fun testWithEmptyLayout() { ... }
@Test fun testMultipleInitializations() { ... }
```

## Success Metrics

Track these to measure testing effectiveness:

- **Test Coverage**: >80% line coverage, 100% for utilities
- **Test Count**: 20+ tests per utility class
- **Build Success Rate**: >95% with pre-commit tests
- **Crash Rate**: <1% after deployment verification
- **Time to Detection**: <5 minutes (vs hours with manual testing)

## Future Improvements

### Automated CI/CD
```yaml
# GitHub Actions
- name: Test and Build
  run: |
    ./gradlew test
    ./gradlew assembleDebug

- name: Run Integration Tests
  run: ./gradlew connectedAndroidTest
```

### Automated Crash Reporting
```kotlin
// Auto-report crashes from field testing
Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
    logCrashReport(throwable)
}
```

### Test Coverage Reports
```bash
./gradlew testDebugUnitTestCoverage
# Generates HTML coverage report
```

## Summary

**The Problem**: Unit tests alone don't catch integration issues

**The Solution**: Three-tier testing strategy
1. Unit Tests (fast, isolated)
2. Integration Tests (real framework)
3. Smoke Tests (deployed verification)

**The Workflow**: test → build → deploy → verify

**The Tools**: Automated scripts that catch issues early

**The Result**: Faster development, fewer crashes, more confidence

---

**Remember**: If it touches Android framework APIs, it needs integration tests!

## Themed Context Issues (v1.32.415)

### What Happened

During documentation condensing (v1.32.414), clipboard functionality started crashing:
- Build succeeded ✅
- App loaded successfully ✅
- **Opening clipboard crashed** ❌

### Root Cause

**Layout inflation without themed context causes attribute resolution failures.**

```java
// WRONG - causes "UnsupportedOperationException: Failed to resolve attribute"
_clipboardPane = (ViewGroup)layoutInflater.inflate(R.layout.clipboard_pane, null);
```

When layout XML uses theme attributes like `?attr/colorKey`, they can only be resolved if the inflation happens with a properly themed context:

```xml
<!-- clipboard_pane.xml -->
<TextView 
    android:background="?attr/colorKey"     <!-- Requires themed context! -->
    android:textColor="?attr/colorLabel"/>  <!-- Requires themed context! -->
```

### The Fix

**Always wrap context with theme before inflating views that use theme attributes:**

```java
// CORRECT - theme attributes resolve properly
Context themedContext = new ContextThemeWrapper(_context, _config.theme);
_clipboardPane = (ViewGroup)View.inflate(themedContext, R.layout.clipboard_pane, null);
```

### Red Flags for Themed Context Issues

✅ **Always use themed context when:**

1. **Layout Uses Theme Attributes**
   - `?attr/colorKeyboard`, `?attr/colorKey`
   - `?attr/colorLabel`, `?attr/colorSubLabel`
   - Any `?attr/*` reference in XML

2. **Inflating Views with LayoutInflater**
   - `layoutInflater.inflate(R.layout.*, null)`
   - Must wrap context: `new ContextThemeWrapper(context, theme)`

3. **Creating Dialogs or Panes**
   - Emoji pane, clipboard pane, settings dialogs
   - Any custom UI components using theme

4. **Error Messages Indicate Theme Issues**
   - "UnsupportedOperationException: Failed to resolve attribute"
   - "Error inflating class"
   - Theme-related crashes in LayoutInflater

### Prevention Strategy

**Pattern to Always Use:**

```java
// Step 1: Get theme from config
int theme = _config.theme;

// Step 2: Wrap context with theme
Context themedContext = new ContextThemeWrapper(context, theme);

// Step 3: Inflate with themed context
View view = View.inflate(themedContext, R.layout.your_layout, null);
```

**Reusable Helper (Keyboard2.java):**

```java
public View inflate_view(int layout) {
    return View.inflate(new ContextThemeWrapper(this, _config.theme), layout, null);
}
```

### Testing Strategy

**Unit Tests** (Document Requirements):
```kotlin
@Test
fun testThemedInflationPattern_documented() {
    // REQUIREMENT: Context must be wrapped with theme
    // Pattern: new ContextThemeWrapper(context, config.theme)
    
    // REQUIREMENT: Use View.inflate() with themed context
    // Pattern: View.inflate(themedContext, layout, null)
    
    assertTrue("Themed inflation pattern documented", true)
}
```

**Integration Tests** (Verify Actual Inflation):
- Test on real Android device/emulator
- Verify layouts inflate without crashes
- Check theme attributes resolve correctly
- Test with different themes (light/dark)

**Smoke Tests** (Runtime Verification):
- Open clipboard pane
- Switch themes
- Verify no inflation crashes
- Check visual appearance

### Example: ClipboardManager Fix (v1.32.415)

**Before (Crashed)**:
```java
public ViewGroup getClipboardPane(LayoutInflater layoutInflater) {
    if (_clipboardPane == null) {
        _clipboardPane = (ViewGroup)layoutInflater.inflate(R.layout.clipboard_pane, null);
        // ERROR: Theme attributes can't resolve!
    }
    return _clipboardPane;
}
```

**After (Fixed)**:
```java
public ViewGroup getClipboardPane(LayoutInflater layoutInflater) {
    if (_clipboardPane == null) {
        // Inflate with themed context (v1.32.415: fix theme attribute resolution)
        Context themedContext = new ContextThemeWrapper(_context, _config.theme);
        _clipboardPane = (ViewGroup)View.inflate(themedContext, R.layout.clipboard_pane, null);
    }
    return _clipboardPane;
}
```

### Lessons Learned

1. **Theme Attributes Are Not Optional** - Views using `?attr/*` MUST have themed context
2. **LayoutInflater Is Not Enough** - Raw LayoutInflater doesn't apply theme
3. **Integration Testing Catches This** - Unit tests can't detect theme resolution failures
4. **Consistent Pattern Prevents Issues** - Always use `inflate_view()` helper or ContextThemeWrapper

### Checklist for Theme-Safe Code

- [ ] Does layout XML use `?attr/*` attributes?
- [ ] Is View.inflate() or LayoutInflater used?
- [ ] Is context wrapped with ContextThemeWrapper?
- [ ] Is config.theme validated (> 0)?
- [ ] Are smoke tests verifying UI opens correctly?
- [ ] Is integration test covering actual inflation?

### Related Issues

- ReceiverInitializer null layoutManager (v1.32.413) - Initialization order
- SubtypeLayoutInitializer crash (v1.32.410) - Framework integration
- Clipboard theme crash (v1.32.415) - Themed context (this issue)

All three demonstrate: **Unit tests pass ≠ App works**

