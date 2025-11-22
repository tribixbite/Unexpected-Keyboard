# Unexpected Keyboard - Neural Swipe Typing Edition 🚀

> **⚡ Production-ready swipe typing powered by ONNX neural networks**

**Fork by [@tribixbite](https://github.com/tribixbite)** | [Original Repo](https://github.com/Julow/Unexpected-Keyboard)

## 🎯 What's New in This Fork

This fork implements a **production-ready neural network-based swipe typing system** with state-of-the-art prediction accuracy:

### ✨ Core Features

#### 1. **Neural Network Swipe Prediction**
- **ONNX Runtime** with transformer encoder-decoder architecture
- **Beam search decoding** for high-quality predictions
- **Real-time trajectory processing** optimized for mobile
- Trained on real swipe gesture data

#### 2. **50k Enhanced Vocabulary**
- **50,000 English words** with real frequency data from Google Books Ngram corpus
- **Hierarchical organization**: common → top5000 → full dictionary
- **Custom word support** via Dictionary Manager
- **Frequency-based ranking** for better predictions

#### 3. **Dual Autocorrect System**
- **Beam Search Autocorrect**: Fuzzy matching during prediction (custom words + dictionary)
- **Final Output Autocorrect**: Safety net after beam search, before text insertion
- **Levenshtein Distance** algorithm for accurate typo correction
- **User-configurable**: Enable/disable either or both systems

#### 4. **Correction Presets**
- **One-click adjustment** of fuzzy matching sensitivity
- **Strict**: Minimize false corrections (high accuracy)
- **Balanced**: Default middle ground
- **Lenient**: Maximize corrections (flexible matching)
- Automatically controls 4 fuzzy matching parameters

#### 5. **Dictionary Manager**
- **Material Design 3 UI** with tabbed interface
- **View all 50k words** with frequency data
- **Add custom words** with editable frequency
- **Disable/enable** individual words
- **Import/export** custom dictionaries

#### 6. **Advanced Debug Tools**
- **Swipe Debug Screen**: Real-time pipeline visualization
  - Trajectory visualization
  - Neural network internals
  - Beam search outputs
  - Vocabulary filtering details
  - Score breakdowns
- **Detailed logging**: Toggle pipeline logging and raw outputs
- **Performance monitoring**: Track prediction latency and accuracy

### 🔧 Technical Architecture

```
User Swipe Gesture
    ↓
Trajectory Normalization (x,y coordinates + velocity/acceleration)
    ↓
ONNX Neural Network (Transformer Encoder-Decoder)
    ↓
Character Probabilities per Timestep
    ↓
Beam Search Decoding (configurable width, default: 2)
    ↓
Vocabulary Filtering (50k dictionary with frequency boost)
    ↓
Optional: Beam Search Autocorrect (fuzzy matching)
    ↓
Hybrid Scoring (NN confidence × dictionary frequency × match quality)
    ↓
Deduplication (LinkedHashMap, keep highest score)
    ↓
Optional: Final Output Autocorrect (second-chance fuzzy match)
    ↓
Text Insertion
```

**Key Algorithm Features:**
- **Transformer architecture**: State-of-the-art sequence modeling
- **Hybrid scoring**: `base_score = (confidence_weight × NN_confidence) + (frequency_weight × dict_frequency)`
- **Match quality impact**: `final_score = base_score × (match_quality³) × tier_boost`
- **Tier boosts**: Common (1.3×), Top5k (1.0×), Rare (0.75×)
- **Fuzzy matching**: Edit distance or positional matching (user-selectable)

## 📱 Installation

### Pre-built APK
```bash
# Latest build (~47MB with ONNX Runtime)
build/outputs/apk/debug/juloo.keyboard2.debug.apk
```

### Build from Source

#### Linux/Mac
```bash
# Clone this fork
git clone https://github.com/tribixbite/Unexpected-Keyboard.git
cd Unexpected-Keyboard

# Build debug APK
./gradlew assembleDebug

# Install on device
adb install -r build/outputs/apk/debug/juloo.keyboard2.debug.apk
```

#### Termux (Android ARM64)
```bash
# One-time setup (installs SDK, JDK, build tools)
./setup-arm64-buildtools.sh

# Build debug APK
./build-on-termux.sh

# Build release APK
./build-on-termux.sh release
```

## 🚀 Quick Start

### 1. Enable the Keyboard
   - Settings → System → Languages & input → Virtual keyboard → Manage keyboards
   - Enable "Unexpected Keyboard"
   - Select "Unexpected Keyboard" when typing

### 2. Configure Swipe Typing
   - Open any app with text input
   - Tap keyboard settings icon (or swipe from spacebar)
   - Navigate to **Swipe Typing** section
   - Review and adjust settings:
     - ✅ Enable autocorrect options (recommended defaults)
     - ✅ Choose correction preset (Balanced recommended)
     - ✅ Adjust scoring weights if needed

### 3. Optional: Customize Dictionary
   - Settings → Dictionary Manager
   - **Add custom words** with frequency values
   - **Disable unwanted words** from suggestions
   - **Import/export** personal dictionaries

### 4. Optional: Debug View
   - Settings → Swipe Typing → Launch Debug Screen
   - Watch real-time prediction pipeline
   - Monitor neural network outputs
   - Analyze scoring and ranking

## ⚙️ Configuration Options

All settings accessible via **Settings → Swipe Typing**:

### Autocorrect Controls
- **Enable Beam Search Corrections** - Fuzzy match during prediction (default: ON)
- **Enable Final Output Corrections** - Second-chance autocorrect (default: ON)
- **Correction Style** - Preset: strict/balanced/lenient (default: balanced)
- **Matching Algorithm** - Edit Distance (recommended) or Positional (legacy)

### Fuzzy Matching Fine-Tuning
- **Typo Forgiveness** - Max length difference: 0-5 (default: 2)
- **Starting Letter Accuracy** - Prefix match length: 0-4 (default: 2)
- **Correction Search Depth** - Beam candidates: 1-10 (default: 3)
- **Character Match Threshold** - Match ratio: 0.5-0.9 (default: 0.67)
- **Minimum Frequency** - Custom word floor: 100-5000 (default: 1000)

### Scoring Weights
- **Prediction Source** - 0-100%: Dictionary ↔ AI Model (default: 50/50)
- **Common Words Boost** - 0.5-2.0× multiplier (default: 1.3×)
- **Frequent Words Boost** - 0.5-2.0× multiplier (default: 1.2×)
- **Rare Words Penalty** - 0.0-1.5× reduction (default: 0.75×)

### Debug Options
- **Detailed Pipeline Logging** - Full pipeline trace in logcat
- **Show Raw NN Output** - Display unfiltered neural network predictions
- **Show Raw Beam Predictions** - Display predictions with "raw:" prefix

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Dictionary Size** | 50,000 words |
| **Prediction Latency** | <100ms average |
| **APK Size** | ~47MB (includes ONNX Runtime) |
| **Memory Usage** | Optimized for mobile |
| **Android Compatibility** | API 21+ (Android 5.0+) |
| **Neural Network Model** | Transformer encoder-decoder |
| **Beam Width** | 2 (configurable) |

## 🔍 Advanced Usage

### Dictionary Manager

**Add Custom Words:**
```
1. Settings → Dictionary Manager → Custom tab
2. Tap + (Add) button
3. Enter word and frequency (100-10000)
4. Word immediately available in predictions
```

**Import/Export:**
```
1. Custom tab → Menu → Export Dictionary
2. Saves to: /sdcard/Download/custom_dictionary_YYYYMMDD.txt
3. Format: word,frequency (one per line)
4. Import via: Menu → Import Dictionary
```

**Disable Unwanted Words:**
```
1. Dictionary tab → Find word
2. Long-press → Disable
3. Word excluded from predictions
4. Re-enable anytime from Disabled tab
```

### Debug Screen

**Launch:** Settings → Swipe Typing → Swipe Debug

**Features:**
- **Live trajectory**: Visualize normalized path
- **NN outputs**: See character probabilities
- **Beam search**: Track candidate generation
- **Vocabulary filter**: Watch dictionary matching
- **Score breakdown**: Understand ranking logic
- **Performance**: Monitor prediction latency

### Logging

**Enable Detailed Logging:**
```
Settings → Swipe Typing → Debug Options → Detailed Pipeline Logging
```

**View Logs:**
```bash
# Real-time swipe prediction logs
adb logcat | grep "SwipePredictor\|WordPredictor\|OnnxSwipe"

# Neural network details
adb logcat | grep "ONNX"

# Autocorrect operations
adb logcat | grep "AUTOCORRECT\|FUZZY"
```

## 🛠️ Development

### Project Structure
```
srcs/juloo.keyboard2/
├── OnnxSwipePredictor.java       # Neural network prediction engine
├── OptimizedVocabulary.java      # 50k dictionary + fuzzy matching
├── WordPredictor.java             # Prediction coordination
├── Keyboard2.java                 # IME service + final autocorrect
├── SuggestionBar.java             # Prediction display
├── DictionaryManagerActivity.java # Dictionary management UI
├── SwipeDebugActivity.java        # Debug visualization
├── SettingsActivity.java          # Settings UI + presets
└── Config.java                    # Configuration management

assets/
├── models/
│   └── swipe_model.onnx          # ONNX neural network model
└── dictionaries/
    └── en_enhanced.txt            # 50k word vocabulary

res/xml/
└── settings.xml                   # Settings UI definitions
```

### Key Implementation Details

**Neural Network:**
- File: `OnnxSwipePredictor.java`
- Input: Normalized trajectory (x,y + velocity/acceleration features)
- Output: Character probabilities per timestep
- Decoding: Beam search with vocabulary filtering

**Vocabulary System:**
- File: `OptimizedVocabulary.java`
- Lines 717-753: Levenshtein distance implementation
- Lines 755-815: Dual-mode match quality (edit distance vs positional)
- Lines 133-159: Configuration loading

**Autocorrect:**
- Beam autocorrect: `OptimizedVocabulary.java` lines 307, 412
- Final autocorrect: `Keyboard2.java` lines 928-941

**Correction Presets:**
- File: `SettingsActivity.java`
- Lines 895-900: Preset change listener
- Lines 929-965: Preset application logic

### Build System

**Gradle Tasks:**
```bash
# Standard build
./gradlew assembleDebug
./gradlew assembleRelease

# Run tests
./gradlew test

# Generate resources
./gradlew genLayoutsList
./gradlew checkKeyboardLayouts
./gradlew compileComposeSequences
```

**Termux Build:**
```bash
# Uses qemu-x86_64 for AAPT2 emulation
# Wrapper in tools/aapt2-arm64/
./build-on-termux.sh
```

## 📝 Changelog

### v1.32.231 - Correction Preset System
- ✅ Implemented correction preset functionality
- ✅ Added reset button for swipe corrections
- ✅ One-click adjustment of 4 fuzzy matching parameters

### v1.32.229 - Final Autocorrect
- ✅ Fixed raw: prefix bug (regex mismatch)
- ✅ Implemented final autocorrect functionality
- ✅ Safety net for raw predictions

### v1.32.227 - Levenshtein Distance
- ✅ Implemented edit distance algorithm
- ✅ Better handling of insertions/deletions
- ✅ User-selectable matching algorithm

### v1.32.226 - Deduplication
- ✅ Fixed duplicate words in suggestion bar
- ✅ Added UI toggles for beam/final autocorrect
- ✅ Added UI toggle for raw predictions

### v1.32.180 - 50k Dictionary Upgrade
- ✅ Upgraded to 50k word vocabulary
- ✅ Real frequency data from Google Books Ngram
- ✅ Editable frequency in Dictionary Manager

### v1.32.0 - Neural Network Foundation
- ✅ ONNX Runtime integration
- ✅ Transformer encoder-decoder model
- ✅ Beam search decoding
- ✅ Vocabulary filtering system

## 🤝 Contributing

Contributions welcome! Priority areas:

### High Priority
- [ ] Multi-language support (currently English-only)
- [ ] Personalized predictions based on usage
- [ ] Context-aware predictions (previous word)
- [ ] Model quantization for smaller APK size

### Medium Priority
- [ ] Emoji swipe support
- [ ] Offline learning capability
- [ ] Advanced gesture customization
- [ ] Cloud sync for custom dictionaries

### Low Priority
- [ ] Additional neural network architectures
- [ ] On-device model training
- [ ] Speech-to-text integration

**Development Workflow:**
1. Check `memory/pm.md` for current project status
2. Follow conventional commit format (see `CLAUDE.md`)
3. Build and test: `./build-on-termux.sh` or `./gradlew assembleDebug`
4. Update `STATUS.md` with changes
5. Submit PR with detailed description

## 📄 License

This fork maintains the original **GNU General Public License v3.0**.

## 🙏 Credits

- **Original Unexpected Keyboard**: [@Julow](https://github.com/Julow)
- **ONNX Runtime**: Microsoft
- **Dictionary Source**: Google Books Ngram corpus
- **UI Framework**: Material Design 3
- **Inspiration**: FlorisBoard, OpenBoard

## 📧 Contact

- **Fork Author**: [@tribixbite](https://github.com/tribixbite)
- **Issues**: [GitHub Issues](https://github.com/tribixbite/Unexpected-Keyboard/issues)
- **Documentation**: See `STATUS.md` for detailed implementation status

---

**Current Status**: ✅ **Production Ready** - Highly optimized and performance-tuned (v1.32.644)

### 🚀 Recent Performance Improvements (v1.32.635-644)

- **2-3x faster swipe processing** (141-226ms saved per swipe)
- **100x faster Termux deletion** (<10ms vs 900ms)
- **Zero UI allocations** from object pooling
- **71% code reduction** in core module (2,397 → 692 lines)
- **-26% APK size** (65MB → 48MB)
- **Thread-safe initialization** with race condition fixes
- **Enhanced logging** with proper Android practices

For detailed technical documentation, see:
- **STATE_SUMMARY_v1.32.643.md** - Complete performance metrics and architecture
- **UTILITY_SCRIPTS.md** - Development and monitoring tools
- **SWIPE_LAG_DEBUG.md** - Termux performance investigation
- **CLAUDE.md** - Development guidelines and build instructions
- **memory/pm.md** - Project management and roadmap
