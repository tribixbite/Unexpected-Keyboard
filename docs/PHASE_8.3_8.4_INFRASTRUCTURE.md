# Phase 8.3 & 8.4: Multi-Language Infrastructure

**Date**: 2025-11-27
**Status**: Planning
**Priority**: HIGH
**Estimated Duration**: 1-2 weeks
**Prerequisites**: Phase 7 complete (v1.32.907), Phase 8.1 analyzed, Phase 8.2 planned

---

## 📋 Overview

Phase 8.3 and 8.4 build the Android infrastructure needed for multi-language support. This work can be done NOW, before the language models are trained, and will be ready when the models from Phase 8.2 are available.

**Existing Foundation**:
- ✅ `LanguageDetector.kt` exists with en/es/fr/de support
- ✅ Character frequency analysis (60% weight)
- ✅ Common word detection (40% weight)
- ✅ Confidence thresholds (0.6 minimum)
- ✅ Supports 4 languages: English, Spanish, French, German

**What Needs to Be Built**:
1. Multi-language model loading infrastructure
2. Language switching logic
3. Multi-language dictionary management
4. Integration with prediction pipeline
5. Settings UI for language selection

---

## 🎯 Phase 8.3: Language Auto-Detection Enhancement

### Current State (LanguageDetector.kt)

**Already Implemented**:
```kotlin
class LanguageDetector {
    // ✅ Character frequency patterns for en/es/fr/de
    private val languageCharFreqs: Map<String, Map<Char, Float>>

    // ✅ Common words for en/es/fr/de
    private val languageCommonWords: Map<String, Array<String>>

    // ✅ Detection from text
    fun detectLanguage(text: String?): String?

    // ✅ Detection from word list
    fun detectLanguageFromWords(words: List<String>?): String?
}
```

**Scoring Algorithm** (Already Implemented):
- Character frequency: 60% weight
- Common words: 40% weight
- Minimum confidence: 0.6 (60%)
- Minimum text length: 10 characters

### Enhancements Needed

**1. Add Portuguese Support** (Language missing)
```kotlin
private fun initializePortuguesePatterns() {
    val ptChars = mapOf(
        'a' to 14.6f,
        'e' to 12.6f,
        'o' to 10.7f,
        's' to 7.8f,
        'r' to 6.5f,
        'i' to 6.2f,
        't' to 4.7f,
        'n' to 5.0f,
        'm' to 4.7f,
        'd' to 5.0f
    )
    languageCharFreqs["pt"] = ptChars

    val ptWords = arrayOf(
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
        "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais"
    )
    languageCommonWords["pt"] = ptWords
}
```

**2. Integrate with Prediction Pipeline**

**File**: `WordPredictor.kt`
```kotlin
class WordPredictor(
    private val context: Context,
    private val config: Config
) {
    private val languageDetector = LanguageDetector()
    private var currentLanguage = "en" // Default
    private val recentWords = mutableListOf<String>()

    // Add words to detection context
    fun onWordTyped(word: String) {
        recentWords.add(word)
        if (recentWords.size > 10) {
            recentWords.removeAt(0)
        }

        // Detect language every 5 words
        if (recentWords.size >= 5) {
            val detected = languageDetector.detectLanguageFromWords(recentWords)
            if (detected != null && detected != currentLanguage) {
                Log.i(TAG, "Language switch detected: $currentLanguage → $detected")
                switchLanguage(detected)
            }
        }
    }

    private fun switchLanguage(newLanguage: String) {
        if (!config.enableMultiLanguage) {
            return // Multi-language disabled in settings
        }

        // TODO: Load new language model (Phase 8.2)
        // TODO: Load new language dictionary (Phase 8.4)
        currentLanguage = newLanguage
    }
}
```

**3. Add Language Switching API**

**File**: `MultiLanguageManager.kt` (NEW)
```kotlin
package juloo.keyboard2

import android.content.Context
import android.util.Log
import ai.onnxruntime.OrtSession

/**
 * Manages multiple language models and automatic language switching
 */
class MultiLanguageManager(
    private val context: Context,
    private val defaultLanguage: String = "en"
) {
    companion object {
        private const val TAG = "MultiLanguageManager"
        private const val SWITCH_LATENCY_TARGET_MS = 100
    }

    // Active language
    private var activeLanguage: String = defaultLanguage

    // Cached models (lazy loading)
    private val modelCache = mutableMapOf<String, LanguageModel>()

    // Language detector
    private val detector = LanguageDetector()

    data class LanguageModel(
        val language: String,
        val encoder: OrtSession,
        val decoder: OrtSession,
        val vocabulary: OptimizedVocabulary
    )

    /**
     * Get current active language
     */
    fun getCurrentLanguage(): String = activeLanguage

    /**
     * Get supported languages
     */
    fun getSupportedLanguages(): Array<String> {
        return detector.getSupportedLanguages()
    }

    /**
     * Load language model (lazy)
     */
    fun loadLanguageModel(language: String): LanguageModel? {
        // Check cache first
        modelCache[language]?.let { return it }

        try {
            // Load encoder
            val encoderPath = "models/swipe_encoder_${language}.onnx"
            val encoder = createSessionFromAsset(context, encoderPath)

            // Load decoder
            val decoderPath = "models/swipe_decoder_${language}.onnx"
            val decoder = createSessionFromAsset(context, decoderPath)

            // Load dictionary
            val dictPath = "${language}_enhanced.bin"
            val vocabulary = OptimizedVocabulary(context, dictPath)

            val model = LanguageModel(language, encoder, decoder, vocabulary)
            modelCache[language] = model

            Log.i(TAG, "Loaded language model: $language")
            return model

        } catch (e: Exception) {
            Log.e(TAG, "Failed to load language model: $language", e)
            return null
        }
    }

    /**
     * Switch to a different language
     * @return true if switch succeeded, false if language unavailable
     */
    fun switchLanguage(newLanguage: String): Boolean {
        if (newLanguage == activeLanguage) {
            return true // Already active
        }

        val startTime = System.currentTimeMillis()

        // Load new language model
        val model = loadLanguageModel(newLanguage)
        if (model == null) {
            Log.e(TAG, "Cannot switch to $newLanguage - model not available")
            return false
        }

        // Atomic switch
        activeLanguage = newLanguage

        val switchTime = System.currentTimeMillis() - startTime
        Log.i(TAG, "Switched to $newLanguage (${switchTime}ms)")

        if (switchTime > SWITCH_LATENCY_TARGET_MS) {
            Log.w(TAG, "Language switch exceeded target latency: ${switchTime}ms > ${SWITCH_LATENCY_TARGET_MS}ms")
        }

        return true
    }

    /**
     * Detect language from recent context and switch if needed
     */
    fun detectAndSwitch(recentWords: List<String>, confidenceThreshold: Float = 0.7f): String? {
        val detected = detector.detectLanguageFromWords(recentWords)
        if (detected != null && detected != activeLanguage) {
            // TODO: Add confidence score to detector
            if (switchLanguage(detected)) {
                return detected
            }
        }
        return null
    }

    /**
     * Preload language model for faster switching
     */
    fun preloadLanguage(language: String) {
        Thread {
            loadLanguageModel(language)
        }.start()
    }

    /**
     * Unload unused language models to free memory
     */
    fun unloadUnusedModels(keepActive: Boolean = true) {
        val toRemove = mutableListOf<String>()
        for ((lang, _) in modelCache) {
            if (!keepActive || lang != activeLanguage) {
                toRemove.add(lang)
            }
        }

        for (lang in toRemove) {
            modelCache.remove(lang)?.let {
                it.encoder.close()
                it.decoder.close()
                Log.i(TAG, "Unloaded language model: $lang")
            }
        }
    }

    /**
     * Create ONNX session from asset
     */
    private fun createSessionFromAsset(context: Context, assetPath: String): OrtSession {
        // Reuse existing implementation from ModelVersionManager
        return ModelVersionManager.createOnnxSessionFromAsset(context, assetPath)
    }

    /**
     * Cleanup all resources
     */
    fun cleanup() {
        for ((lang, model) in modelCache) {
            model.encoder.close()
            model.decoder.close()
            Log.i(TAG, "Cleaned up language model: $lang")
        }
        modelCache.clear()
    }
}
```

---

## 🗂️ Phase 8.4: Multi-Language Dictionary Infrastructure

### Current State

**Existing Dictionary System**:
- `OptimizedVocabulary.kt` loads `en_enhanced.bin`
- Binary format for fast lookup
- Frequency-based ranking
- Size: ~2MB per dictionary

**What Exists**:
```kotlin
class OptimizedVocabulary(
    private val context: Context,
    private val dictFileName: String = "en_enhanced.bin"
) {
    // ✅ Binary dictionary loading
    // ✅ Fast lookup (O(1) hash map)
    // ✅ Frequency ranking
}
```

### Multi-Language Dictionary Manager

**File**: `MultiLanguageDictionaryManager.kt` (NEW)
```kotlin
package juloo.keyboard2

import android.content.Context
import android.util.Log
import java.util.concurrent.ConcurrentHashMap

/**
 * Manages multiple language-specific dictionaries with lazy loading
 */
class MultiLanguageDictionaryManager(
    private val context: Context
) {
    companion object {
        private const val TAG = "MultiLanguageDictionaryManager"
    }

    // Cached dictionaries (language code → OptimizedVocabulary)
    private val dictionaries = ConcurrentHashMap<String, OptimizedVocabulary>()

    /**
     * Load dictionary for a specific language
     * @param language Language code (en, es, fr, pt, de)
     * @return OptimizedVocabulary or null if not found
     */
    fun loadDictionary(language: String): OptimizedVocabulary? {
        // Check cache first
        dictionaries[language]?.let { return it }

        try {
            val filename = "${language}_enhanced.bin"
            val vocab = OptimizedVocabulary(context, filename)
            dictionaries[language] = vocab

            Log.i(TAG, "Loaded dictionary: $language ($filename)")
            return vocab

        } catch (e: Exception) {
            Log.e(TAG, "Failed to load dictionary: $language", e)
            return null
        }
    }

    /**
     * Get dictionary for active language (fallback to English)
     */
    fun getDictionary(language: String): OptimizedVocabulary {
        return dictionaries[language]
            ?: loadDictionary(language)
            ?: loadDictionary("en") // Fallback to English
            ?: throw IllegalStateException("No dictionaries available")
    }

    /**
     * Preload dictionary asynchronously
     */
    fun preloadDictionary(language: String) {
        Thread {
            loadDictionary(language)
        }.start()
    }

    /**
     * Unload unused dictionaries to free memory
     */
    fun unloadDictionary(language: String) {
        dictionaries.remove(language)
        Log.i(TAG, "Unloaded dictionary: $language")
    }

    /**
     * Get list of loaded dictionaries
     */
    fun getLoadedLanguages(): Set<String> {
        return dictionaries.keys.toSet()
    }

    /**
     * Get memory usage estimate
     */
    fun getMemoryUsageMB(): Float {
        // Estimate: ~2MB per dictionary
        return dictionaries.size * 2.0f
    }

    /**
     * Clear all cached dictionaries
     */
    fun clearAll() {
        dictionaries.clear()
        Log.i(TAG, "Cleared all dictionaries")
    }
}
```

---

## ⚙️ Settings UI Integration

### Add Multi-Language Settings

**File**: `res/xml/settings.xml`
```xml
<!-- Multi-Language Settings -->
<PreferenceCategory
    android:title="Multi-Language Support"
    android:key="pref_category_multilang">

    <CheckBoxPreference
        android:key="pref_enable_multilang"
        android:title="Enable Multi-Language"
        android:summary="Automatically switch between languages based on typing"
        android:defaultValue="false" />

    <ListPreference
        android:key="pref_primary_language"
        android:title="Primary Language"
        android:summary="Default language for predictions"
        android:entries="@array/language_names"
        android:entryValues="@array/language_codes"
        android:defaultValue="en"
        android:dependency="pref_enable_multilang" />

    <CheckBoxPreference
        android:key="pref_auto_detect_language"
        android:title="Auto-Detect Language"
        android:summary="Automatically detect and switch languages while typing"
        android:defaultValue="true"
        android:dependency="pref_enable_multilang" />

    <SeekBarPreference
        android:key="pref_language_detection_sensitivity"
        android:title="Detection Sensitivity"
        android:summary="How quickly to switch languages"
        android:defaultValue="60"
        android:max="100"
        android:dependency="pref_auto_detect_language" />

</PreferenceCategory>
```

**File**: `res/values/arrays.xml`
```xml
<!-- Language names for UI -->
<string-array name="language_names">
    <item>English</item>
    <item>Spanish</item>
    <item>French</item>
    <item>Portuguese</item>
    <item>German</item>
</string-array>

<!-- Language codes -->
<string-array name="language_codes">
    <item>en</item>
    <item>es</item>
    <item>fr</item>
    <item>pt</item>
    <item>de</item>
</string-array>
```

### Update Config.kt

```kotlin
// Add to Config class
val enableMultiLanguage: Boolean
    get() = prefs.getBoolean("pref_enable_multilang", false)

val primaryLanguage: String
    get() = prefs.getString("pref_primary_language", "en") ?: "en"

val autoDetectLanguage: Boolean
    get() = prefs.getBoolean("pref_auto_detect_language", true)

val languageDetectionSensitivity: Float
    get() = prefs.getInt("pref_language_detection_sensitivity", 60) / 100.0f
```

---

## 🔄 Integration with Prediction Pipeline

### Update WordPredictor

**File**: `WordPredictor.kt` (modifications)
```kotlin
class WordPredictor(
    private val context: Context,
    private val config: Config
) {
    // Add multi-language support
    private val multiLangManager = MultiLanguageManager(context, config.primaryLanguage)
    private val multiDictManager = MultiLanguageDictionaryManager(context)

    init {
        // Preload primary language
        if (config.enableMultiLanguage) {
            multiLangManager.preloadLanguage(config.primaryLanguage)
            multiDictManager.preloadDictionary(config.primaryLanguage)
        }
    }

    fun predict(swipeData: SwipeData, context: List<String>): List<Prediction> {
        // Detect language from context if enabled
        if (config.enableMultiLanguage && config.autoDetectLanguage && context.size >= 5) {
            multiLangManager.detectAndSwitch(context, config.languageDetectionSensitivity)
        }

        // Get current language model and dictionary
        val currentLang = multiLangManager.getCurrentLanguage()
        val dictionary = multiDictManager.getDictionary(currentLang)

        // Use language-specific model for prediction
        // TODO: Integrate with NeuralSwipeTypingEngine when models available

        return emptyList() // Placeholder
    }

    fun cleanup() {
        multiLangManager.cleanup()
        multiDictManager.clearAll()
    }
}
```

---

## 📊 Testing Plan

### Unit Tests

**File**: `test/juloo.keyboard2/MultiLanguageManagerTest.kt` (NEW)
```kotlin
class MultiLanguageManagerTest {
    @Test
    fun testLanguageSwitching() {
        // Test switching between languages
        // Verify latency < 100ms
        // Verify correct model loaded
    }

    @Test
    fun testLanguageDetection() {
        // Test detection accuracy
        // Test Spanish text → es
        // Test French text → fr
        // Test mixed text handling
    }

    @Test
    fun testDictionaryLoading() {
        // Test lazy loading
        // Test caching
        // Test memory management
    }
}
```

### Integration Tests

1. **Language Detection Accuracy**:
   - Type Spanish words → detect "es"
   - Type French words → detect "fr"
   - Type mixed English/Spanish → handle gracefully

2. **Switching Performance**:
   - Measure latency (target: <100ms)
   - Verify no crashes during switch
   - Test rapid switching

3. **Memory Management**:
   - Load all 5 languages
   - Verify memory usage <50MB total
   - Test unloading unused languages

---

## 📦 File Structure (After Phase 8.3 & 8.4)

```
srcs/juloo.keyboard2/
├── LanguageDetector.kt              # ✅ EXISTS - add Portuguese
├── MultiLanguageManager.kt          # 🆕 NEW - model loading & switching
├── MultiLanguageDictionaryManager.kt # 🆕 NEW - dictionary management
├── WordPredictor.kt                  # ✏️ MODIFY - integrate multi-lang
├── Config.kt                         # ✏️ MODIFY - add settings
└── ModelVersionManager.kt            # ✏️ MODIFY - multi-lang support

res/xml/
└── settings.xml                      # ✏️ MODIFY - add multi-lang settings

res/values/
├── arrays.xml                        # ✏️ MODIFY - add language arrays
└── strings.xml                       # ✏️ MODIFY - add multi-lang strings

test/juloo.keyboard2/
└── MultiLanguageManagerTest.kt       # 🆕 NEW - comprehensive tests
```

---

## 🎯 Implementation Timeline

### Week 1: Core Infrastructure (Days 1-5)
**Day 1-2**: Multi-Language Manager
- Create `MultiLanguageManager.kt`
- Implement model loading/switching logic
- Add caching and lazy loading
- Unit tests

**Day 3-4**: Dictionary Manager
- Create `MultiLanguageDictionaryManager.kt`
- Implement per-language dictionary loading
- Add memory management
- Unit tests

**Day 5**: Language Detector Enhancement
- Add Portuguese patterns to `LanguageDetector.kt`
- Improve confidence scoring
- Add detection tests

### Week 2: Integration & Polish (Days 6-10)
**Day 6-7**: Prediction Pipeline Integration
- Update `WordPredictor.kt`
- Integrate language detection
- Add automatic switching logic
- Integration tests

**Day 8**: Settings UI
- Add multi-language settings screen
- Update `Config.kt`
- Add language selection UI

**Day 9**: Testing & Optimization
- Performance testing (switching latency)
- Memory profiling
- Fix issues

**Day 10**: Documentation & Release
- Update documentation
- Code review
- Merge to main

---

## ✅ Success Criteria

**Phase 8.3 & 8.4 Complete When**:
- ✅ `MultiLanguageManager` implemented and tested
- ✅ `MultiLanguageDictionaryManager` implemented and tested
- ✅ Portuguese added to `LanguageDetector`
- ✅ `WordPredictor` integrated with multi-language system
- ✅ Settings UI for language selection added
- ✅ Language switching latency <100ms
- ✅ Memory usage <50MB for all 5 languages
- ✅ All unit tests passing (>80% coverage)
- ✅ Integration tests passing
- ✅ Documentation updated

**Ready for Phase 8.2**:
When language models (es, fr, pt, de) are trained, they can be dropped into `assets/models/` and the infrastructure will automatically support them!

---

## 🔜 Phase 9 Preview

**Next Priorities**:
- Dictionary compression (30MB → 15MB via binary compression)
- On-demand language pack downloads
- Model quantization (if needed after multi-language testing)
- Additional languages (Russian, Arabic, Chinese, Japanese)

---

**Phase 8.3 & 8.4 Planning Complete!**
**Ready to begin implementation - infrastructure can be built NOW**
