# Testing Guide for Unexpected Keyboard

## Overview

This project uses a comprehensive three-tier testing strategy:

1. **Unit Tests** - Fast, isolated tests with mocks (JUnit + Mockito)
2. **Integration Tests** - Android framework tests (AndroidX Test)
3. **Smoke Tests** - Post-deployment verification (ADB shell scripts)

## Quick Start

### Run All Tests and Deploy
```bash
./build-test-deploy.sh
```

This script will:
1. Run all unit tests
2. Build the debug APK
3. Connect via ADB
4. Install the APK
5. Run smoke tests
6. Monitor for crashes

### Run Tests Only
```bash
# Unit tests only
./gradlew test

# Integration tests (requires connected device/emulator)
./gradlew connectedAndroidTest

# Specific test class
./gradlew test --tests "SubtypeLayoutInitializerTest"

# Smoke tests after manual install
./smoke-test.sh
```

## Test Structure

### Unit Tests (`test/juloo.keyboard2/`)

Located in `test/juloo.keyboard2/`, these tests use Mockito to isolate components:

```
test/juloo.keyboard2/
├── MLDataCollectorTest.kt
├── KeyboardReceiverTest.kt
├── SubtypeLayoutInitializerTest.kt
├── LayoutBridgeTest.kt
└── ... (20+ test suites)
```

**Coverage**: 607 test cases, 8,230 lines, 100% coverage of extracted utilities

**Run**: `./gradlew test`

### Integration Tests (`test/juloo.keyboard2/integration/`)

Tests that use real Android framework components:

```
test/juloo.keyboard2/integration/
└── KeyboardIntegrationTest.kt
```

**Run**: `./gradlew connectedAndroidTest` (requires device/emulator)

### Smoke Tests (`smoke-test.sh`)

Post-installation verification via ADB:
- Package installed
- IME service registered
- No crashes on enable
- Permissions configured
- App data directory exists

**Run**: `./smoke-test.sh` (requires ADB connection)

## Testing Workflow

### Before Every Commit

```bash
# 1. Run unit tests
./gradlew test

# 2. If tests pass, build and deploy
./build-test-deploy.sh

# 3. Manually verify keyboard functionality
```

### Adding New Features

1. **Write tests first** (TDD approach):
   ```bash
   # Create test file
   touch test/juloo.keyboard2/NewFeatureTest.kt

   # Write failing tests
   # Implement feature
   # Run tests until they pass
   ./gradlew test --tests "NewFeatureTest"
   ```

2. **Add integration test** if touching Android framework:
   ```kotlin
   // test/juloo.keyboard2/integration/NewFeatureIntegrationTest.kt
   @RunWith(AndroidJUnit4::class)
   class NewFeatureIntegrationTest { ... }
   ```

3. **Deploy and smoke test**:
   ```bash
   ./build-test-deploy.sh
   ```

### Debugging Test Failures

#### Unit Test Failures

```bash
# Run with stacktrace
./gradlew test --stacktrace

# View HTML report
open build/reports/tests/test/index.html

# Run specific test with debug output
./gradlew test --tests "SpecificTest.testMethod" --debug
```

#### Integration Test Failures

```bash
# View logcat during test
adb logcat -c && ./gradlew connectedAndroidTest & adb logcat

# Pull test reports from device
adb pull /sdcard/Download/test-results/
```

#### Deployment Failures

Check logs:
- `build-debug.log` - Build output
- `test-results.log` - Unit test results
- `logcat-monitor.log` - Runtime crash logs

## Common Issues and Solutions

### Issue: "Tests pass but app crashes on load"

**Root Cause**: Unit tests with mocks don't catch runtime integration issues

**Solution**:
1. Run integration tests: `./gradlew connectedAndroidTest`
2. Use `build-test-deploy.sh` which monitors logcat
3. Add integration test for the new component

### Issue: "ADB connection fails"

**Solution**:
```bash
# Manual ADB setup
adb tcpip 5555
adb connect 192.168.1.247:5555

# Or set environment variables
export ADB_HOST=192.168.1.100
export ADB_PORT_RANGE=30000-50000
./build-test-deploy.sh
```

### Issue: "Gradle test tasks not found"

**Solution**:
```bash
# Initialize test infrastructure
./gradlew tasks --all | grep test

# Clean and rebuild
./gradlew clean test
```

## Test Coverage Goals

- **Unit Tests**: 100% coverage of all extracted utilities
- **Integration Tests**: Critical paths and Android framework interactions
- **Smoke Tests**: Basic functionality verification

### Current Coverage (v1.32.410)

- Unit Tests: 21 suites, 607 tests, 8,230 lines
- Integration Tests: 1 suite, 4 tests
- Smoke Tests: 6 checks

## Best Practices

### Writing Good Tests

1. **AAA Pattern** (Arrange, Act, Assert):
   ```kotlin
   @Test
   fun testFeature() {
       // Arrange
       val input = "test"

       // Act
       val result = feature.process(input)

       // Assert
       assertEquals("expected", result)
   }
   ```

2. **Test Names**: Descriptive and specific
   ```kotlin
   ✅ testRefreshSubtypeAndLayout_firstCall_createsLayoutBridge()
   ❌ testFeature()
   ```

3. **Edge Cases**: Test null, empty, boundary values
   ```kotlin
   @Test
   fun testWithNullInput() { ... }

   @Test
   fun testWithEmptyList() { ... }

   @Test
   fun testWithMaxValue() { ... }
   ```

4. **Isolation**: Each test should be independent
   ```kotlin
   @Before
   fun setUp() {
       // Reset state before each test
   }
   ```

### Avoiding Integration Issues

**Before creating a new utility:**

1. Write comprehensive unit tests (20+ test cases)
2. Add integration test if using Android framework
3. Use `build-test-deploy.sh` to verify
4. Monitor logcat for crashes
5. Test manually on device

**Red flags that indicate need for integration tests:**
- Uses Android Context, Resources, or Services
- Creates Views or interacts with UI
- Accesses SharedPreferences or ContentProviders
- Initializes framework components

## Continuous Integration

Future enhancement: GitHub Actions workflow

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: ./gradlew test
      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: test-reports
          path: build/reports/tests/
```

## Test Maintenance

### When to Update Tests

- Feature changes → Update corresponding tests
- Bug fixes → Add regression test
- Refactoring → Ensure tests still pass
- New dependencies → Add integration tests

### Test Review Checklist

- [ ] All new code has unit tests
- [ ] Edge cases covered
- [ ] Integration tests for Android framework usage
- [ ] `build-test-deploy.sh` passes
- [ ] Manual verification on device
- [ ] Test documentation updated

## Resources

- [JUnit 4 Documentation](https://junit.org/junit4/)
- [Mockito Documentation](https://site.mockito.org/)
- [AndroidX Test Guide](https://developer.android.com/training/testing)
- [Gradle Testing](https://docs.gradle.org/current/userguide/java_testing.html)

## Support

Issues with testing? Check:
1. `test/juloo.keyboard2/README_TESTS.md` - Test suite documentation
2. Build logs in `build/reports/tests/`
3. Create issue on GitHub with test output
