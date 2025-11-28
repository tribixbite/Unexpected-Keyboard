# Phase 8: Multi-Language Support & Model Optimization

**Status**: Planning
**Priority**: High
**Target**: v1.33.x series
**Estimated Duration**: 6-8 weeks
**Prerequisites**: Phase 7 complete (v1.32.907)

---

## 📋 Overview

Phase 8 expands Unexpected Keyboard's capabilities to support multiple languages and optimizes model size through quantization. This phase builds on the intelligent prediction foundation from Phase 7 to deliver a truly multilingual typing experience with reduced APK size.

---

## 🎯 Goals

### Primary Objectives
1. **Multi-Language Models**: Train and deploy models for Spanish, French, German, Portuguese
2. **Language Auto-Detection**: Automatically switch models based on typing context
3. **Model Quantization**: Reduce APK size by 20-30% (47MB → 33-37MB)
4. **Unified Architecture**: Single codebase supporting all languages

### Success Metrics
- Languages supported: 1 (English) → 5 (English + 4 new languages)
- APK size: 47MB → 35MB (FP16 quantization)
- Prediction accuracy per language: >75% Top-1
- Language detection accuracy: >90%
- Model switching latency: <100ms

---

## 🚀 Phase 8 Sub-Phases

### 8.1: Model Quantization (4-6 weeks)

**Priority**: HIGH - Reduces APK size before adding more models

**Description**: Quantize existing English encoder/decoder models from FP32 to FP16 for 50% size reduction with minimal accuracy loss.

#### Implementation Components

**1. Quantization Pipeline** (Python):
```python
# ml_training/quantize_model.py
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(model_path, output_path, quantization_mode='fp16'):
    """
    Quantize ONNX model to reduce size

    Args:
        model_path: Path to FP32 ONNX model
        output_path: Path for quantized model
        quantization_mode: 'fp16', 'int8', or 'dynamic'
    """
    model = onnx.load(model_path)

    if quantization_mode == 'fp16':
        # Float16 quantization (best accuracy/size tradeoff)
        quantized_model = convert_float_to_float16(model)
    elif quantization_mode == 'int8':
        # INT8 quantization (maximum size reduction)
        quantized_model = quantize_dynamic(
            model,
            weight_type=QuantType.QUInt8
        )

    onnx.save(quantized_model, output_path)
```

**2. Benchmark Script** (Python):
```python
# ml_training/benchmark_quantization.py
def benchmark_models():
    """
    Compare FP32 vs FP16 vs INT8 models

    Metrics:
    - Model file size (MB)
    - Inference latency (ms)
    - Top-1/Top-5 accuracy
    - Memory usage (RAM)
    """
    models = {
        'fp32': 'models/encoder_fp32.onnx',
        'fp16': 'models/encoder_fp16.onnx',
        'int8': 'models/encoder_int8.onnx'
    }

    results = {}
    for name, path in models.items():
        results[name] = {
            'size_mb': get_file_size(path),
            'latency_ms': benchmark_inference(path),
            'accuracy': evaluate_accuracy(path, test_set),
            'memory_mb': measure_memory(path)
        }

    return results
```

**3. Android Integration** (Kotlin):
```kotlin
// srcs/juloo.keyboard2/QuantizedModelLoader.kt
class QuantizedModelLoader(private val context: Context) {
    enum class PrecisionMode {
        FP32,    // Full precision (47MB per model)
        FP16,    // Half precision (23MB per model)
        INT8     // Integer quantization (12MB per model)
    }

    fun loadModel(
        modelType: ModelType,
        precision: PrecisionMode
    ): OrtSession {
        val modelPath = when (precision) {
            PrecisionMode.FP32 -> "models/${modelType}_fp32.onnx"
            PrecisionMode.FP16 -> "models/${modelType}_fp16.onnx"
            PrecisionMode.INT8 -> "models/${modelType}_int8.onnx"
        }

        return createSessionFromAsset(context, modelPath)
    }
}
```

**4. Settings UI**:
- Model Precision dropdown (Auto/High Quality/Balanced/Small Size)
- Auto mode: FP16 on devices with 4GB+ RAM, INT8 on low-end devices
- Manual override option

#### Quantization Strategy

**Recommended**: FP16 for all models
- **Size**: 50% reduction (47MB → 23MB)
- **Accuracy**: <0.5% loss (negligible)
- **Inference**: 1.5-2x faster on modern devices
- **Compatibility**: Supported on all ARM64 devices

**Alternative**: Dynamic INT8
- **Size**: 75% reduction (47MB → 12MB)
- **Accuracy**: 2-5% loss (acceptable with calibration)
- **Inference**: 3-4x faster on devices with INT8 acceleration
- **Complexity**: Requires calibration dataset

#### Testing Plan

1. **Accuracy Benchmarks**:
   - Measure Top-1/Top-5 accuracy on test set
   - Compare FP32 vs FP16 vs INT8
   - Target: <1% accuracy loss for FP16

2. **Performance Benchmarks**:
   - Inference latency across device types
   - Memory usage profiling
   - APK size verification

3. **A/B Testing**:
   - Deploy FP16 to 50% of users
   - Monitor crash rates, prediction quality
   - Gather user feedback

**Estimated Time**: 1-2 weeks (quantization + testing)

---

### 8.2: Multi-Language Model Training (2-3 weeks)

**Priority**: MEDIUM - After quantization complete

**Description**: Train encoder/decoder models for Spanish, French, German, Portuguese using existing training pipeline.

#### Languages to Support

**Phase 8.2.1 - Romance Languages** (similar to English):
1. **Spanish (es)** - 500M speakers
   - Training data: OpenSubtitles, Common Crawl
   - Dictionary: 100K most common words
   - Expected accuracy: 75-80% Top-1

2. **French (fr)** - 280M speakers
   - Training data: OpenSubtitles, Wikipedia
   - Dictionary: 90K words (handles accents)
   - Expected accuracy: 72-77% Top-1

3. **Portuguese (pt)** - 250M speakers
   - Training data: OpenSubtitles, Brazilian Portuguese corpus
   - Dictionary: 85K words
   - Expected accuracy: 72-77% Top-1

**Phase 8.2.2 - Germanic Language**:
4. **German (de)** - 135M speakers
   - Training data: OpenSubtitles, German Wikipedia
   - Dictionary: 120K words (compound words)
   - Expected accuracy: 70-75% Top-1

#### Training Process

**Per Language**:
1. Collect swipe trajectories dataset (100K samples)
2. Train character-level encoder/decoder
3. Quantize to FP16 immediately
4. Package with language-specific dictionary
5. Test on validation set
6. Deploy to APK

**Optimization**:
- Share encoder weights across similar languages (Romance group)
- Use transfer learning from English model
- Quantize during training (QAT) for better accuracy

**APK Size Estimate**:
- English (FP16): 23MB
- Spanish (FP16): 22MB
- French (FP16): 21MB
- Portuguese (FP16): 20MB
- German (FP16): 23MB
- **Total**: ~110MB for 5 languages
- **vs Current**: 47MB (English only)
- **Increase**: +63MB

**Mitigation**:
- Download additional languages on-demand (Phase 9)
- Base APK: English only (23MB with FP16)
- Language packs: Optional downloads

**Estimated Time**: 2-3 weeks (4 languages × 3-4 days each)

---

### 8.3: Language Auto-Detection (1-2 weeks)

**Priority**: MEDIUM - Essential for multi-language UX

**Description**: Enhance existing LanguageDetector to automatically switch models based on typing context.

#### Detection Strategy

**Current State**: Basic character-range detection exists in LanguageDetector.kt

**Enhancements Needed**:

**1. N-gram Frequency Analysis**:
```kotlin
class LanguageDetector {
    // Character bigram frequencies per language
    private val bigramFrequencies = mapOf(
        "en" to mapOf("th" to 3.56, "he" to 3.07, "in" to 2.43...),
        "es" to mapOf("de" to 4.23, "la" to 3.42, "el" to 2.98...),
        "fr" to mapOf("le" to 3.78, "de" to 3.45, "la" to 2.87...),
        "de" to mapOf("en" to 3.92, "er" to 3.67, "ch" to 2.54...),
        "pt" to mapOf("de" to 3.98, "os" to 3.21, "as" to 2.76...)
    )

    fun detectLanguage(text: String): LanguageCode {
        // Analyze character bigrams
        val bigramScores = calculateBigramScores(text)

        // Check dictionary words
        val dictScores = checkDictionaryWords(text)

        // Combine scores
        return getBestMatch(bigramScores, dictScores)
    }
}
```

**2. Dictionary Word Matching**:
- Check if recent words exist in each language dictionary
- Weight by word frequency (common words = stronger signal)
- Threshold: >3 words from same language → switch models

**3. Keyboard Layout Hints**:
- QWERTY → English/French likely
- QWERTZ → German likely
- AZERTY → French likely
- Use as prior probability

**4. User Preference**:
- Primary language setting (default model)
- Fallback when detection uncertain
- Manual language picker in keyboard

#### Model Switching Logic

```kotlin
class MultiLanguageManager {
    private var activeModel: LanguageCode = LanguageCode.ENGLISH
    private val recentWords = mutableListOf<String>()

    fun handleWordTyped(word: String) {
        recentWords.add(word)
        if (recentWords.size > 5) {
            val detected = languageDetector.detectFromContext(recentWords)

            if (detected != activeModel && detected.confidence > 0.8) {
                switchLanguage(detected.language)
            }
        }
    }

    private fun switchLanguage(newLang: LanguageCode) {
        // Async model loading
        val newModel = modelManager.loadModel(newLang)

        // Atomic swap
        activeModel = newLang
        swipeEngine.setModel(newModel)

        Log.i(TAG, "Switched to $newLang (detected from context)")
    }
}
```

#### Testing

**Test Cases**:
1. Type English sentence → stays in English
2. Type "Hola cómo estás" → switches to Spanish
3. Type "Bonjour comment allez-vous" → switches to French
4. Mix English/Spanish in same message → smart switching
5. Low confidence → stays in current language (no thrashing)

**Metrics**:
- Detection accuracy: >90% after 5 words
- False positive rate: <5%
- Switching latency: <100ms

**Estimated Time**: 1-2 weeks

---

### 8.4: Multi-Language Dictionary Infrastructure (1 week)

**Priority**: MEDIUM - Required for language models to work

**Description**: Extend dictionary system to support multiple language-specific dictionaries with efficient loading.

#### Dictionary Architecture

**Current**: Single `en_enhanced.bin` dictionary

**Proposed**: Per-language dictionaries with lazy loading

```kotlin
class MultiLanguageDictionaryManager(private val context: Context) {
    private val dictionaries = ConcurrentHashMap<LanguageCode, OptimizedVocabulary>()

    fun loadDictionary(language: LanguageCode): OptimizedVocabulary {
        return dictionaries.getOrPut(language) {
            val filename = "${language.code}_enhanced.bin"
            OptimizedVocabulary(context, filename)
        }
    }

    fun preloadLanguage(language: LanguageCode) {
        // Async preload for smooth switching
        backgroundExecutor.execute {
            loadDictionary(language)
        }
    }
}
```

#### Dictionary Files

**Structure**:
```
assets/dictionaries/
├── en_enhanced.bin         # English (100K words)
├── es_enhanced.bin         # Spanish (90K words)
├── fr_enhanced.bin         # French (85K words)
├── pt_enhanced.bin         # Portuguese (80K words)
└── de_enhanced.bin         # German (95K words)
```

**Generation**:
- Use existing `scripts/generate_binary_dict.py`
- Source: OpenSubtitles frequency lists per language
- Format: Same binary format as English

**Size Estimate**:
- Per dictionary: ~1-2MB compressed
- Total: ~8MB for 5 languages

**Estimated Time**: 1 week (dictionary generation + integration)

---

## 📦 APK Size Management

### Size Breakdown (Post-Phase 8)

**Base APK** (English only):
- Code: 5MB
- English Model (FP16): 23MB
- English Dictionary: 2MB
- Resources: 3MB
- **Total**: 33MB ✅ (down from 47MB)

**Full APK** (All 5 languages):
- Code: 5MB
- Models (5 × ~22MB FP16): 110MB
- Dictionaries (5 × 2MB): 10MB
- Resources: 3MB
- **Total**: 128MB ❌ (too large)

### Strategy: Language Pack Downloads

**Phase 8 Implementation**: Include all in APK (acceptable for power users)

**Phase 9 Plan**: On-demand downloads
- Base APK: English only (33MB)
- Language packs: Download in Settings
- Cache on device, auto-update

**For Now**: Accept 128MB APK
- Target audience: users who want multi-language
- Most users only use 1-2 languages anyway
- Future optimization: detect primary language during onboarding

---

## 🧪 Testing Strategy

### Per Language

1. **Accuracy Testing**:
   - Test set: 1,000 swipes per language
   - Metrics: Top-1, Top-3, Top-5 accuracy
   - Target: >75% Top-1

2. **Language Detection Testing**:
   - Mixed language text samples
   - False positive rate
   - Switching latency

3. **Integration Testing**:
   - All 5 languages in one app
   - Memory usage with multiple models loaded
   - Model switching stability

### Cross-Language

1. **Bilingual Users**:
   - Type English, then Spanish, then English
   - Verify smooth switching
   - No model thrashing

2. **Low-End Devices**:
   - Test on 2GB RAM devices
   - Verify model eviction works
   - Acceptable performance degradation

---

## 🎯 Phase 8 Timeline

**Week 1-2**: Model Quantization
- Quantize English encoder/decoder to FP16
- Benchmark accuracy/speed
- Update Android loader
- Test on devices

**Week 3-4**: Spanish & French Models
- Train Spanish model
- Train French model
- Quantize both to FP16
- Integration testing

**Week 5-6**: Portuguese & German Models
- Train Portuguese model
- Train German model
- Quantize both to FP16
- Integration testing

**Week 7**: Language Auto-Detection
- Enhance LanguageDetector
- Implement model switching
- Integration with MultiLanguageManager
- Testing

**Week 8**: Dictionary Infrastructure & Polish
- Generate language dictionaries
- Multi-language dictionary manager
- Full integration testing
- Performance optimization
- Bug fixes

**Total**: 8 weeks

---

## 🚧 Risks & Mitigation

**Risk 1**: APK size too large (128MB)
- **Mitigation**: Quantize aggressively (FP16 all models)
- **Fallback**: Ship English-only, download others

**Risk 2**: Inference too slow on low-end devices
- **Mitigation**: INT8 quantization for low-end
- **Fallback**: Disable multi-language on <2GB RAM devices

**Risk 3**: Language detection inaccurate
- **Mitigation**: Require >80% confidence before switching
- **Fallback**: Manual language picker always available

**Risk 4**: Training data quality varies per language
- **Mitigation**: Use OpenSubtitles (consistent across languages)
- **Fallback**: Start with Romance languages (most data)

---

## 📊 Success Criteria

**Must Have**:
- ✅ English model quantized to FP16 (23MB)
- ✅ 4 new language models trained and working
- ✅ Language auto-detection >90% accuracy
- ✅ APK size <140MB (all languages)

**Nice to Have**:
- Model sharing between Romance languages
- On-demand language downloads
- INT8 quantization option

**Phase 8 Complete When**:
- All 5 languages trained and quantized
- Language detection working reliably
- Full integration testing passed
- APK released on GitHub

---

## 🔜 Phase 9 Preview

**Focus**: Advanced Features & Optimization
- On-demand language pack downloads
- Quantization-Aware Training (QAT) for better INT8
- Cross-language learning (transfer learning)
- Additional languages (Russian, Arabic, Chinese, Japanese)
- Cloud sync for personalization data (opt-in)

---

**Phase 8 Planning Complete!**
**Ready to begin implementation after Phase 7 user feedback collected.**
