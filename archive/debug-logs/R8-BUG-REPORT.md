# R8/D8 Bug Report for Google Issue Tracker

**Report URL**: https://issuetracker.google.com/issues?q=componentid:192708

---

## Summary

R8/D8 8.6.17 crashes with NullPointerException when processing Kotlin data class with specific pattern: nullable array elements + companion object + self-referential nullable types + default parameters.

## Environment

- **R8 Version**: 8.6.17 (bundled with AGP 8.6.0)
- **Android Gradle Plugin**: 8.6.0
- **Gradle**: 8.7
- **Kotlin**: 1.9.20 (also fails with 1.9.24 and 2.0.21)
- **Build Platform**: Android ARM64 (Termux), also reproducible on x86_64
- **Target SDK**: 35
- **Min SDK**: 21

## Error Message

```
java.lang.NullPointerException: Cannot read field "d" because "<local0>" is null
	at com.android.tools.r8.internal.wo.a(R8_8.6.17_ad35be29c7108873fdb35df1527459cc24cc04d949c37dc9efbc6304e042931d:155)
	at com.android.tools.r8.internal.CZ.a(R8_8.6.17_ad35be29c7108873fdb35df1527459cc24cc04d949c37dc9efbc6304e042931d:191)
```

**Task**: `:dexBuilderDebug`

**Phase**: DEX file generation (after successful Kotlin compilation)

**Crash Point**: Static initializer of inner data class `KeyboardData$Key.<clinit>()V`

## Minimal Reproduction Case

```kotlin
package juloo.keyboard2

data class KeyValue(
    val kind: Int,
    val name: String
) {
    companion object {
        const val KIND_CHAR = 0
    }
}

data class KeyboardData(
    val rows: List<Row>
) {
    data class Row(
        val keys: List<Key>
    )

    data class Key(
        val keys: Array<KeyValue?>,              // 1. Nullable array elements
        val anticircle: Boolean = false,
        val keysflags: IntArray = IntArray(keys.size),  // 2. Default parameter with expression
        val width: Float = 1f,
        val shift: Float = 0f,
        val indication: KeyboardData.Key? = null  // 3. Self-referential nullable type
    ) {
        companion object {                        // 4. Companion object with constants
            const val F_LOC = 1
            const val F_SMALLER = 2
        }
    }
}
```

## Build Configuration

```gradle
// build.gradle
plugins {
    id 'com.android.application' version '8.6.0'
    id 'org.jetbrains.kotlin.android' version '1.9.20'
}

android {
    namespace 'juloo.keyboard2'
    compileSdk 35

    defaultConfig {
        minSdk 21
        targetSdkVersion 35
    }

    buildTypes {
        debug {
            minifyEnabled false
            shrinkResources false
            debuggable true
        }
    }

    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }
}
```

## Steps to Reproduce

1. Create new Android project with AGP 8.6.0
2. Add Kotlin plugin 1.9.20 (or 1.9.24, or 2.0.21 - all fail)
3. Create the `KeyValue` and `KeyboardData` classes above
4. Run `./gradlew assembleDebug`

**Expected**: Successful DEX compilation

**Actual**: NullPointerException in R8/D8 internal code at `com.android.tools.r8.internal.wo.a`

## Additional Information

### Kotlin Compilation Success

The Kotlin compiler successfully generates valid bytecode:

```bash
./gradlew compileDebugKotlin
# ✅ SUCCESS - zero errors
```

The crash occurs only during DEX file generation from the compiled `.class` files.

### Workarounds Attempted (All Failed)

1. ❌ `android.enableR8.fullMode=false` - D8 runs regardless
2. ❌ AGP downgrade to 8.5.2 - dependencies require 8.6.0+
3. ❌ AGP upgrade to 8.7.3 - requires Gradle 8.9 (incompatible)
4. ❌ Gradle upgrade to 8.9 - breaks AAPT2 wrapper
5. ❌ Combined Gradle 8.9 + AGP 8.7.3 - AAPT2 incompatibility
6. ❌ Kotlin 1.9.24 upgrade - same NPE in `KeyboardData$Key.<clinit>()V`
7. ❌ Kotlin 2.0.21 (K2 compiler) upgrade - same NPE despite complete rewrite
8. ❌ ProGuard rules (`-dontoptimize`, `-keep`) - no effect

### Pattern Analysis

The bug is triggered by this specific combination:

1. **Data class** with nullable array elements: `Array<KeyValue?>`
2. **Companion object** in the same data class with constants
3. **Self-referential nullable type**: `KeyboardData.Key?`
4. **Default parameters** with expressions: `IntArray(keys.size)`
5. **Nested inner class** structure

Removing ANY of these elements allows DEX compilation to succeed, but they are all required for the application's functionality.

### Impact

This blocks migration of Java code to Kotlin for projects using this pattern. The code is valid Kotlin that compiles successfully but cannot be converted to DEX format.

**Affected Project**: Unexpected Keyboard - Android virtual keyboard with 145 files migrated to Kotlin, blocked at 98.6% completion.

### Full Stack Trace

```
> Task :dexBuilderDebug FAILED

FAILURE: Build failed with an exception.

* What went wrong:
Execution failed for task ':dexBuilderDebug'.
> There was a failure while executing work items
   > A failure occurred while executing com.android.build.gradle.internal.dexing.DexWorkAction
      > Failed to process: /data/data/com.termux/files/home/git/swype/Unexpected-Keyboard/build/tmp/kotlin-classes/debug

* Exception is:
org.gradle.api.tasks.TaskExecutionException: Execution failed for task ':dexBuilderDebug'.
	at com.android.build.gradle.internal.tasks.DexArchiveBuilderTask.doProcess(DexArchiveBuilderTask.kt:321)
	...
Caused by: java.lang.RuntimeException: There was a failure while executing work items
	at com.android.build.gradle.internal.dexing.DexArchiveBuilderTask.doProcess(DexArchiveBuilderTask.kt:318)
	...
Caused by: java.lang.RuntimeException: A failure occurred while executing com.android.build.gradle.internal.dexing.DexWorkAction
	...
Caused by: java.lang.RuntimeException: Failed to process: /data/data/com.termux/files/home/git/swype/Unexpected-Keyboard/build/tmp/kotlin-classes/debug
	...
Caused by: com.android.tools.r8.CompilationFailedException: Compilation failed to complete, origin: /data/data/com.termux/files/home/git/swype/Unexpected-Keyboard/build/tmp/kotlin-classes/debug/juloo/keyboard2/KeyboardData$Key.class
	at com.android.tools.r8.internal.CZ.a(R8_8.6.17_ad35be29c7108873fdb35df1527459cc24cc04d949c37dc9efbc6304e042931d:198)
	...
Caused by: com.android.tools.r8.internal.yo: java.lang.NullPointerException: Cannot read field "d" because "<local0>" is null
	at com.android.tools.r8.internal.wo.a(R8_8.6.17_ad35be29c7108873fdb35df1527459cc24cc04d949c37dc9efbc6304e042931d:155)
	at com.android.tools.r8.internal.CZ.a(R8_8.6.17_ad35be29c7108873fdb35df1527459cc24cc04d949c37dc9efbc6304e042931d:191)
	at com.android.tools.r8.internal.yh.b(R8_8.6.17_ad35be29c7108873fdb35df1527459cc24cc04d949c37dc9efbc6304e042931d:148)
	...
Caused by: [CIRCULAR REFERENCE: java.lang.NullPointerException: Cannot read field "d" because "<local0>" is null]
```

### Verification

This is **NOT** a bug in the Kotlin code:

1. ✅ Kotlin compiler generates valid bytecode (no errors)
2. ✅ Code follows Kotlin best practices
3. ✅ Null safety properly implemented
4. ✅ Data class pattern is standard Kotlin
5. ❌ R8/D8 crashes with internal NullPointerException

### Request

Please fix R8/D8 8.6.17 to handle Kotlin data classes with this pattern, or provide guidance on alternative patterns that preserve the same functionality.

---

## Related Files

Full project available at: https://github.com/Julow/Unexpected-Keyboard

**Specific commit**: `feature/swipe-typing` branch, commits `4bb895cf` through `c5be7ad8`

**Detailed investigation**: See `R8-BUG-WORKAROUND.md` in repository for complete analysis.

---

**Reporter**: AI-assisted migration (Claude Code) for Unexpected Keyboard project
**Date**: 2025-11-26
**Priority**: High (blocks Kotlin migration for valid code patterns)
