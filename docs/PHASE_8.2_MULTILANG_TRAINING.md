# Phase 8.2: Multi-Language Model Training

**Date**: 2025-11-27
**Status**: Planning
**Priority**: HIGH
**Estimated Duration**: 2-3 weeks (4 languages × 3-4 days each)
**Prerequisites**: Phase 7 complete (v1.32.907), Phase 8.1 analyzed (quantization deferred)

---

## 📋 Overview

Phase 8.2 trains encoder/decoder models for 4 additional languages using the existing training pipeline. This delivers true multi-language swipe typing support with 75%+ Top-1 accuracy per language.

---

## 🎯 Languages to Support

### Priority Order (Romance Languages First)

**1. Spanish (es)** - 500M speakers
- **Training Data**: OpenSubtitles Spanish corpus + Common Crawl
- **Dictionary Size**: 100K most common Spanish words
- **Expected Accuracy**: 75-80% Top-1 (similar structure to English)
- **Special Considerations**: Accented characters (á, é, í, ó, ú, ñ, ü)
- **Model Size**: ~10MB (FP32 encoder + decoder)

**2. French (fr)** - 280M speakers
- **Training Data**: OpenSubtitles French corpus + Wikipedia FR
- **Dictionary Size**: 90K words (handles accents: à, â, é, è, ê, ë, etc.)
- **Expected Accuracy**: 72-77% Top-1
- **Special Considerations**: Cedilla (ç), ligatures, silent letters
- **Model Size**: ~10MB

**3. Portuguese (pt)** - 250M speakers
- **Training Data**: OpenSubtitles PT + Brazilian Portuguese corpus
- **Dictionary Size**: 85K words
- **Expected Accuracy**: 72-77% Top-1 (Brazilian + European variants)
- **Special Considerations**: Tildes (ã, õ), accents (á, â, à, etc.)
- **Model Size**: ~10MB

**4. German (de)** - 135M speakers
- **Training Data**: OpenSubtitles DE + German Wikipedia
- **Dictionary Size**: 120K words (compound words!)
- **Expected Accuracy**: 70-75% Top-1 (harder due to compounds)
- **Special Considerations**: Umlauts (ä, ö, ü, ß), compound words
- **Model Size**: ~10MB

**Total Models**: 4 languages × 10MB = 40MB additional
**Total APK**: 47MB (current) + 40MB (models) + 30MB (dicts) = **117MB**

---

## 🏗️ Training Pipeline Architecture

### Existing Infrastructure (Reuse)

**Current Training Pipeline** (`ml_training/`):
```
ml_training/
├── train_swipe_model.py          # Main training script ✅
├── preprocess_data.py            # Data preprocessing ✅
├── generate_synthetic_swipes.py   # Synthetic data generation ✅
├── export_to_onnx.py             # ONNX export ✅
├── quantize_models.py            # INT8 quantization ✅
├── quantize_fp16.py              # FP16 quantization (Phase 8.1) ✅
└── requirements.txt               # Dependencies ✅
```

**Reusable Components**:
- ✅ Encoder/Decoder architecture (same for all languages)
- ✅ Character-level tokenization (works for any alphabet)
- ✅ Training loop and optimization
- ✅ ONNX export pipeline
- ✅ Evaluation metrics (Top-1, Top-3, Top-5)

**What Needs Adaptation**:
- Language-specific vocabulary files
- Language-specific training datasets
- Character set adjustments (accented characters)
- Dictionary generation per language

---

## 📦 Per-Language Training Workflow

### Step-by-Step Process (Spanish Example)

**1. Data Collection** (1-2 days)
```bash
# Download OpenSubtitles Spanish corpus
wget http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/es.txt.gz

# Download Spanish Common Crawl (optional, larger)
# Or use Wikipedia Spanish dump
wget https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles.xml.bz2

# Extract and clean
gunzip es.txt.gz
python ml_training/clean_corpus.py es.txt > es_cleaned.txt

# Generate Spanish vocabulary (100K words)
python scripts/generate_vocabulary.py \
    --input es_cleaned.txt \
    --output dictionaries/es_vocabulary.txt \
    --size 100000 \
    --language es
```

**2. Generate Training Data** (1 day)
```bash
# Generate synthetic swipe trajectories from Spanish text
python ml_training/generate_synthetic_swipes.py \
    --vocabulary dictionaries/es_vocabulary.txt \
    --output ml_training/spanish_swipes.ndjson \
    --num_samples 100000 \
    --layout qwerty \
    --language es

# Output: 100K synthetic swipe samples with:
# - Swipe trajectories (x, y coordinates)
# - Target words (ground truth)
# - Nearest key sequences
```

**3. Train Encoder/Decoder** (2-3 days on GPU)
```bash
# Train Spanish encoder
python ml_training/train_swipe_model.py \
    --model encoder \
    --language es \
    --data ml_training/spanish_swipes.ndjson \
    --vocabulary dictionaries/es_vocabulary.txt \
    --epochs 20 \
    --batch_size 64 \
    --output models/swipe_encoder_es.pth

# Train Spanish decoder
python ml_training/train_swipe_model.py \
    --model decoder \
    --language es \
    --data ml_training/spanish_swipes.ndjson \
    --vocabulary dictionaries/es_vocabulary.txt \
    --epochs 15 \
    --batch_size 64 \
    --output models/swipe_decoder_es.pth
```

**Training Hardware Requirements**:
- GPU: NVIDIA RTX 3060 or better (recommended)
- CPU: Possible but 10-20x slower
- RAM: 16GB minimum
- Storage: 50GB per language (datasets + checkpoints)

**Training Time Estimates**:
- GPU (RTX 3060): 8-12 hours total per language
- CPU (fallback): 4-7 days per language

**4. Export to ONNX** (30 minutes)
```bash
# Export encoder to ONNX (Android-compatible)
python ml_training/export_to_onnx.py \
    --model encoder \
    --checkpoint models/swipe_encoder_es.pth \
    --output assets/models/swipe_encoder_es.onnx \
    --opset 14

# Export decoder to ONNX
python ml_training/export_to_onnx.py \
    --model decoder \
    --checkpoint models/swipe_decoder_es.pth \
    --output assets/models/swipe_decoder_es.onnx \
    --opset 14
```

**5. Generate Binary Dictionary** (10 minutes)
```bash
# Convert Spanish vocabulary to binary format
python scripts/generate_binary_dict.py \
    --input dictionaries/es_vocabulary.txt \
    --output assets/dictionaries/es_enhanced.bin \
    --language es
```

**6. Validation Testing** (1 hour)
```bash
# Evaluate on test set
python ml_training/evaluate_model.py \
    --encoder assets/models/swipe_encoder_es.onnx \
    --decoder assets/models/swipe_decoder_es.onnx \
    --test_data ml_training/spanish_test.ndjson \
    --metrics top1,top3,top5

# Expected output:
# Top-1 Accuracy: 75.3%
# Top-3 Accuracy: 89.1%
# Top-5 Accuracy: 93.7%
# Avg Inference Time: 23ms
```

**7. Android Integration** (2 hours)
```kotlin
// Update ModelVersionManager.kt to support Spanish
val SUPPORTED_LANGUAGES = listOf("en", "es", "fr", "pt", "de")

fun loadLanguageModel(language: String): Pair<OrtSession, OrtSession> {
    val encoderPath = "models/swipe_encoder_${language}.onnx"
    val decoderPath = "models/swipe_decoder_${language}.onnx"

    val encoder = createSessionFromAsset(context, encoderPath)
    val decoder = createSessionFromAsset(context, decoderPath)

    return Pair(encoder, decoder)
}
```

---

## 🔄 Parallel Training Strategy

### Optimize for Throughput

**Option A: Sequential Training** (Conservative)
- Train Spanish → French → Portuguese → German
- Total time: 4 languages × 3-4 days = 12-16 days
- Advantage: Monitor quality, adjust hyperparameters between languages
- Disadvantage: Slower

**Option B: Parallel Training** (Aggressive)
- Train all 4 languages simultaneously on different GPU partitions
- Total time: 3-4 days (same as 1 language)
- Advantage: Much faster completion
- Disadvantage: Requires multiple GPUs or cloud resources

**Option C: Hybrid** (Recommended)
- Train Spanish first (validate pipeline works)
- Train French + Portuguese in parallel (similar Romance structure)
- Train German separately (different structure)
- Total time: 7-10 days

---

## 📊 Quality Assurance

### Testing Requirements

**Per-Language Validation**:
1. **Accuracy Benchmarks**:
   - Test set: 1,000 swipes per language
   - Metrics: Top-1, Top-3, Top-5 accuracy
   - Target: >75% Top-1, >90% Top-3

2. **Character Coverage**:
   - Spanish: á, é, í, ó, ú, ñ, ü
   - French: à, â, é, è, ê, ë, ï, ô, û, ù, ç
   - Portuguese: á, â, ã, à, é, ê, í, ó, ô, õ, ú, ç
   - German: ä, ö, ü, ß

3. **Common Phrases**:
   - Spanish: "hola", "cómo estás", "gracias", "por favor"
   - French: "bonjour", "comment allez-vous", "merci", "s'il vous plaît"
   - Portuguese: "olá", "como está", "obrigado", "por favor"
   - German: "hallo", "wie geht es", "danke", "bitte"

4. **Performance**:
   - Inference latency: <50ms per swipe
   - Memory usage: <100MB per loaded model
   - Model loading time: <500ms

---

## 🗂️ File Structure (After Phase 8.2)

```
Unexpected-Keyboard/
├── assets/
│   ├── models/
│   │   ├── swipe_encoder_android.onnx      # English (5.1MB) ✅
│   │   ├── swipe_decoder_android.onnx      # English (4.8MB) ✅
│   │   ├── swipe_encoder_es.onnx           # Spanish (5MB) 🆕
│   │   ├── swipe_decoder_es.onnx           # Spanish (5MB) 🆕
│   │   ├── swipe_encoder_fr.onnx           # French (5MB) 🆕
│   │   ├── swipe_decoder_fr.onnx           # French (5MB) 🆕
│   │   ├── swipe_encoder_pt.onnx           # Portuguese (5MB) 🆕
│   │   ├── swipe_decoder_pt.onnx           # Portuguese (5MB) 🆕
│   │   ├── swipe_encoder_de.onnx           # German (5MB) 🆕
│   │   └── swipe_decoder_de.onnx           # German (5MB) 🆕
│   └── dictionaries/
│       ├── en_enhanced.bin                 # English (2MB) ✅
│       ├── es_enhanced.bin                 # Spanish (2MB) 🆕
│       ├── fr_enhanced.bin                 # French (2MB) 🆕
│       ├── pt_enhanced.bin                 # Portuguese (2MB) 🆕
│       └── de_enhanced.bin                 # German (2MB) 🆕
├── ml_training/
│   ├── datasets/
│   │   ├── english_swipes.ndjson           # Existing ✅
│   │   ├── spanish_swipes.ndjson           # 🆕
│   │   ├── french_swipes.ndjson            # 🆕
│   │   ├── portuguese_swipes.ndjson        # 🆕
│   │   └── german_swipes.ndjson            # 🆕
│   └── models/
│       └── (PyTorch checkpoints for each language)
```

**APK Size Impact**:
- Models: +40MB (4 languages × 10MB)
- Dictionaries: +8MB (4 languages × 2MB)
- **Total Increase**: +48MB
- **New APK Size**: 47MB + 48MB = **95MB**

---

## 🚀 Implementation Timeline

### Week 1: Spanish & French
**Days 1-2**: Data collection and preprocessing
- Download OpenSubtitles corpora
- Generate vocabularies (Spanish 100K, French 90K)
- Create synthetic swipe datasets

**Days 3-5**: Model training
- Train Spanish encoder/decoder
- Train French encoder/decoder
- Export to ONNX

**Days 6-7**: Validation and integration
- Test accuracy on validation sets
- Integrate into Android app
- Basic testing

### Week 2: Portuguese & German
**Days 8-9**: Data collection
- Portuguese corpus + vocabulary (85K)
- German corpus + vocabulary (120K)

**Days 10-12**: Model training
- Train Portuguese encoder/decoder
- Train German encoder/decoder
- Export to ONNX

**Days 13-14**: Integration and testing
- Add all 4 languages to app
- Multi-language testing
- Performance profiling

### Week 3: Polish and Release
**Days 15-17**: Quality assurance
- Comprehensive accuracy testing
- Cross-language switching tests
- Memory profiling

**Days 18-21**: Documentation and release
- Update Phase 8 documentation
- Create release notes
- Build multi-language APK
- Deploy to GitHub

---

## 🛠️ Development Scripts (New)

### Create Multi-Language Training Script

**File**: `ml_training/train_all_languages.sh`
```bash
#!/bin/bash
# Train all Phase 8.2 languages sequentially

LANGUAGES=("es" "fr" "pt" "de")
LANG_NAMES=("Spanish" "French" "Portuguese" "German")
VOCAB_SIZES=(100000 90000 85000 120000)

for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    NAME="${LANG_NAMES[$i]}"
    VOCAB_SIZE="${VOCAB_SIZES[$i]}"

    echo "========================================"
    echo "Training $NAME ($LANG)"
    echo "========================================"

    # Step 1: Generate vocabulary
    python scripts/generate_vocabulary.py \
        --input datasets/${LANG}_corpus.txt \
        --output dictionaries/${LANG}_vocabulary.txt \
        --size $VOCAB_SIZE \
        --language $LANG

    # Step 2: Generate synthetic swipes
    python ml_training/generate_synthetic_swipes.py \
        --vocabulary dictionaries/${LANG}_vocabulary.txt \
        --output ml_training/${LANG}_swipes.ndjson \
        --num_samples 100000 \
        --language $LANG

    # Step 3: Train encoder
    python ml_training/train_swipe_model.py \
        --model encoder \
        --language $LANG \
        --data ml_training/${LANG}_swipes.ndjson \
        --epochs 20 \
        --output models/swipe_encoder_${LANG}.pth

    # Step 4: Train decoder
    python ml_training/train_swipe_model.py \
        --model decoder \
        --language $LANG \
        --data ml_training/${LANG}_swipes.ndjson \
        --epochs 15 \
        --output models/swipe_decoder_${LANG}.pth

    # Step 5: Export to ONNX
    python ml_training/export_to_onnx.py \
        --model encoder \
        --checkpoint models/swipe_encoder_${LANG}.pth \
        --output assets/models/swipe_encoder_${LANG}.onnx

    python ml_training/export_to_onnx.py \
        --model decoder \
        --checkpoint models/swipe_decoder_${LANG}.pth \
        --output assets/models/swipe_decoder_${LANG}.onnx

    # Step 6: Generate binary dictionary
    python scripts/generate_binary_dict.py \
        --input dictionaries/${LANG}_vocabulary.txt \
        --output assets/dictionaries/${LANG}_enhanced.bin

    echo "✅ $NAME complete!"
    echo ""
done

echo "🎉 All languages trained successfully!"
```

---

## 📝 Success Criteria

**Phase 8.2 Complete When**:
- ✅ Spanish model trained and tested (>75% Top-1)
- ✅ French model trained and tested (>72% Top-1)
- ✅ Portuguese model trained and tested (>72% Top-1)
- ✅ German model trained and tested (>70% Top-1)
- ✅ All models exported to ONNX
- ✅ All dictionaries generated (binary format)
- ✅ Android integration complete
- ✅ Multi-language APK builds successfully
- ✅ APK size <100MB
- ✅ Basic language switching works
- ✅ Documentation updated

**Deliverables**:
1. 4 encoder ONNX models (es, fr, pt, de)
2. 4 decoder ONNX models (es, fr, pt, de)
3. 4 binary dictionaries (es, fr, pt, de)
4. Training scripts and documentation
5. Multi-language APK release

---

## 🔜 Phase 8.3 Preview

**Next Phase**: Language Auto-Detection
- Implement N-gram frequency analysis
- Add dictionary word matching
- Keyboard layout hints
- Confidence-based model switching
- <100ms switching latency
- >90% detection accuracy

**Phase 8.4**: Multi-Language Dictionary Infrastructure
- Lazy loading of language dictionaries
- Efficient multi-language lookup
- Memory management for multiple dicts

---

**Phase 8.2 Planning Complete!**
**Ready to begin training Spanish model as proof-of-concept**
