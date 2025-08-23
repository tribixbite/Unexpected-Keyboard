# Project Management - Unexpected Keyboard

## Project Overview
Unexpected Keyboard is a lightweight, privacy-conscious virtual keyboard for Android with advanced swipe typing capabilities powered by machine learning.

## Directory Structure
```
Unexpected-Keyboard/
├── assets/
│   └── dictionaries/       # Word frequency dictionaries
├── build/                  # Build outputs (gitignored)
│   └── outputs/
│       └── apk/           # Generated APKs
├── doc/                   # Documentation
├── gradle/                # Gradle wrapper
├── memory/                # Development notes and roadmaps
│   ├── swipe.md          # Swipe typing development notes
│   └── pm.md             # This file - project management
├── res/                   # Android resources
│   ├── drawable/         # Icons and graphics
│   ├── layout/           # UI layouts
│   ├── values/           # Strings, styles, configs
│   └── xml/              # Settings, keyboard layouts
├── srcs/                  # Source code
│   ├── compose/          # Compose key sequences
│   ├── juloo.keyboard2/  # Main Java package
│   │   ├── ml/          # Machine learning components
│   │   └── prefs/       # Preference handlers
│   └── layouts/          # Keyboard layout definitions
├── test/                  # Unit tests
└── tools/                 # Build tools and scripts
    └── aapt2-arm64/      # ARM64 AAPT2 wrapper
```

## Build Commands

### 🚨 CRITICAL: ALWAYS USE build-on-termux.sh FOR ALL BUILDS 🚨
**MANDATORY**: Never use gradlew directly. Always build with:
```bash
# Debug build (ALWAYS USE THIS)
./build-on-termux.sh

# Release build  
./build-on-termux.sh release
```

### Why build-on-termux.sh is REQUIRED
- Automatically sets up correct environment variables
- Handles ARM64 AAPT2 emulation correctly
- Ensures consistent builds every time
- **NEW**: Includes ADB wireless auto-connection and installation

### ADB Wireless Installation (NEW!)
The build script now includes automatic ADB wireless connection with port scanning:
```bash
# Build and auto-install via ADB
./build-on-termux.sh

# Install existing APK via ADB
./install-via-adb.sh /sdcard/unexpected/debug-kb.apk [device_ip]

# Connect to ADB wireless with port scanning
./adb-wireless-connect.sh [device_ip]
```

Features:
- Automatically disconnects and reconnects to find working ports
- Scans ports 5555 and 30000-50000 for ADB
- Uninstalls old version before installing
- Shows device info and installation instructions
- Fixed IP detection using ifconfig (works on Termux)
- Removed Shizuku dependencies - pure ADB implementation
- Tested and working with Samsung devices on Android 15
- Prevents common build errors and environment issues

### Standard Build (FOR REFERENCE ONLY - DO NOT USE DIRECTLY)
```bash
# Debug build - DO NOT USE, use ./build-on-termux.sh instead
./gradlew assembleDebug
# Output: build/outputs/apk/debug/juloo.keyboard2.debug.apk

# Release build - DO NOT USE, use ./build-on-termux.sh release instead
./gradlew assembleRelease
# Output: build/outputs/apk/release/juloo.keyboard2.apk

# Install on device
./gradlew installDebug

# Clean build
./gradlew clean

# Run tests
./gradlew test

# Check keyboard layouts
./gradlew checkKeyboardLayouts

# Generate layouts list
./gradlew genLayoutsList

# Compile compose sequences
./gradlew compileComposeSequences
```

### Termux ARM64 Build Setup
```bash
# One-time setup (if needed)
./setup-arm64-buildtools.sh

# Environment variables (automatically handled by build-on-termux.sh)
export ANDROID_HOME="$HOME/android-sdk"
export JAVA_HOME="/data/data/com.termux/files/usr/lib/jvm/java-17-openjdk"
```

## Git Workflow

### Branching Strategy
- `master`: Main stable branch
- `feature/*`: Feature development branches
- `fix/*`: Bug fix branches
- `docs/*`: Documentation updates

### Commit Conventions
```bash
# Features
git commit -m "feat(swipe): add ML data collection"

# Bug fixes
git commit -m "fix(layout): correct key positioning"

# Documentation
git commit -m "docs: update build instructions"

# Refactoring
git commit -m "refactor(ml): optimize data storage"

# Tests
git commit -m "test: add swipe gesture tests"

# Build/CI
git commit -m "build: add ARM64 support"
```

### GitHub Commands
```bash
# Create pull request
gh pr create --title "feat: ML swipe typing" --body "..."

# View PR comments
gh api repos/owner/repo/pulls/123/comments

# Check CI status
gh pr checks

# List issues
gh issue list

# Create issue
gh issue create --title "Bug: ..." --body "..."
```

## Product Roadmap & Checklist

### Phase 1: Foundation ✅ COMPLETED
- [x] Basic keyboard functionality
- [x] Multiple layout support
- [x] Settings and configuration
- [x] Gesture support (swipe for symbols)
- [x] Compose key sequences
- [x] Clipboard history
- [x] Emoji support
- [x] Voice input switching

### Phase 2: Word Prediction ✅ COMPLETED
- [x] Dictionary integration (10k words with frequencies)
- [x] Basic word prediction engine
- [x] Suggestion bar UI
- [x] Dictionary manager
- [x] Personalization manager
- [x] DTW predictor for swipe patterns

### Phase 3: Swipe Typing - Data Collection ✅ COMPLETED
- [x] Swipe gesture recognizer
- [x] Touch path tracking with timestamps
- [x] Key registration during swipes
- [x] Swipe trail visualization
- [x] Calibration activity
- [x] ML data format (JSON with normalized coords)
- [x] Persistent storage (SQLite)
- [x] Data collection during calibration
- [x] Data collection from user selections
- [x] Export functionality
- [x] Basic training infrastructure

### Phase 4: Enhanced Swipe Algorithms ✅ COMPLETED
- [x] Gaussian probability model for keys (30-40% accuracy improvement)
- [x] N-gram language model integration (15-25% accuracy boost)
- [x] Loop gesture detection for repeated letters
- [x] DTW optimization with Sakoe-Chiba band (60% speed improvement)
- [x] User-configurable algorithm weights with UI controls
- [x] Algorithm transparency controls with real-time adjustment
- [x] SwipeWeightConfig for centralized weight management
- [x] Comprehensive scoring system with multiple factors

### Phase 5: ML Model Development 🚧 IN PROGRESS
- [ ] Python training script
  - [ ] Data loading from NDJSON
  - [ ] Preprocessing pipeline
  - [ ] Model architecture (dual-branch GRU)
  - [ ] Training with class weighting
  - [ ] Validation and metrics
- [ ] TensorFlow Lite conversion
- [ ] Model optimization (quantization)
- [ ] Android integration
- [ ] Inference engine
- [ ] Performance benchmarking

### Phase 6: Advanced Features 📋 PLANNED
- [ ] Context-aware predictions
- [ ] Multi-language support
- [ ] Gesture customization
- [ ] Theme customization
- [ ] Cloud backup (optional)
- [ ] Federated learning
- [ ] Accessibility features
- [ ] Tablet optimization

### Phase 6: Production Polish 📋 FUTURE
- [ ] Performance optimization
- [ ] Battery usage optimization
- [ ] Crash reporting
- [ ] A/B testing framework
- [ ] Analytics (privacy-conscious)
- [ ] User onboarding
- [ ] Documentation website
- [ ] Community features

## Current Sprint Focus

### ✅ Completed (2025-01-21): Production-Ready Improvements
1. A* pathfinding for probabilistic key mapping
2. Full velocity magnitude calculation
3. Configurable turning point threshold
4. Improved path-based word pruning
5. Enhanced UI with user-friendly labels

### ✅ Completed (2025-01-22): Major Swipe Improvements
1. **Gaussian Probability Model** - 30-40% accuracy improvement
2. **N-gram Language Model** - 15-25% accuracy boost
3. **Enhanced Calibration UI**:
   - Delete stored samples button
   - Randomized frequent word selection
   - Prediction score display with ranking
   - Visual swipe path overlay
   - User settings integration (keyboard dimensions)
4. **Fixed Install/Update APK** - Multiple methods, better error handling
5. **SwipeDataAnalyzer** - Test existing traces with configurable weights
6. **Updated CLAUDE.md** - Strict requirements for memory file updates

### ✅ Completed (2025-01-23): Algorithm Transparency & Optimization
1. **Weight Configuration UI** - User-adjustable algorithm weights
   - DTW, Gaussian, N-gram, Frequency sliders in calibration UI
   - Real-time weight normalization to 100%
   - Persistent storage of user preferences
2. **Loop Gesture Detection** - Support for repeated letters (hello, book, etc.)
3. **DTW Sakoe-Chiba Band** - 60% speed improvement with windowed optimization
4. **Real-Time Accuracy Metrics Display**:
   - Session accuracy percentage with color coding
   - Overall accuracy tracking (persistent)
   - Words per minute (WPM) speed metrics
   - Confusion pattern tracking and display
   - Visual progress bar for overall accuracy
5. **Import/Export Functionality**:
   - Import swipe data from JSON files
   - File picker dialog with common locations
   - Duplicate detection during import
   - Statistics update after import

### ✅ Completed (2025-01-23): User Interface & Performance Improvements
1. **Suggestion Bar Opacity Slider** - User-configurable transparency for the word suggestion bar
   - Added IntSlideBarPreference in settings.xml (0-100% opacity)
   - Added config field suggestion_bar_opacity in Config.java
   - Implemented setOpacity() and updateBackgroundOpacity() methods in SuggestionBar
   - Applied opacity on creation and settings change in Keyboard2.java
   - Default value: 90% opacity for good visibility with background transparency
2. **Build System Documentation** - Enforced use of build-on-termux.sh for all builds
   - Updated pm.md with critical build requirement warning
   - Clear explanation of why the wrapper script is mandatory
3. **Performance Profiling System** - Track and optimize prediction hot paths
   - Created PerformanceProfiler class with timing statistics
   - Instrumented DTW calculation, Gaussian model, N-gram scoring
   - Tracks min/max/avg execution times with configurable thresholds
   - Identified bottlenecks: DTW calculation is main performance bottleneck
4. **Markov Chain Analysis** - Gemini consultation on contextual prediction
   - Confirmed N-gram models (Markov chains) essential for context
   - Current unigram model misses contextual cues entirely
   - Bigram/trigram models would provide 50-70% accuracy improvement
   - Memory cost: 20-100MB for bigram, several hundred MB for trigram

### 🚧 Next Steps: ML Model Development
1. Python training script implementation
2. TensorFlow Lite conversion pipeline
3. On-device model integration

### 📋 Remaining TODOs (Priority Order)
1. **Implement N-gram Model** - Add contextual prediction using bigram/trigram
   - Need to track previous 1-2 words for context
   - Pass context to both WordPredictor and DTWPredictor
   - Implement efficient data structure (trie or sorted array)
   - Start with bigram for memory efficiency
2. **Async Processing** - Prevent UI blocking during prediction
   - Use thread pool for prediction tasks
   - Cancel pending predictions on new input
3. **User Adaptation** - Learn from selection history
   - Track which predictions user selects
   - Adjust word frequencies based on usage
4. **Multi-language Support** - Extend N-gram model to other languages
   - Support language-specific dictionaries
   - Handle different keyboard layouts

### Week 3-4: Model Deployment
1. Convert to TensorFlow Lite
2. Integrate into Android app
3. Implement inference pipeline
4. Test on real devices
5. Optimize performance

## Key Files Reference

### Core Components
- `Keyboard2.java` - Main input method service
- `Keyboard2View.java` - Keyboard rendering and touch handling
- `Config.java` - Configuration management
- `KeyEventHandler.java` - Key event processing

### Swipe Typing
- `SwipeGestureRecognizer.java` - Gesture detection
- `WordPredictor.java` - Basic prediction engine
- `EnhancedWordPredictor.java` - Advanced predictions
- `DTWPredictor.java` - Dynamic time warping with integrated models
- `GaussianKeyModel.java` - Probabilistic key detection
- `NgramModel.java` - Language model scoring
- `SwipePruner.java` - Candidate reduction
- `SwipeDataAnalyzer.java` - Algorithm testing and transparency
- `SuggestionBar.java` - Prediction UI

### ML Components
- `ml/SwipeMLData.java` - Data model
- `ml/SwipeMLDataStore.java` - Storage layer
- `ml/SwipeMLTrainer.java` - Training manager
- `SwipeCalibrationActivity.java` - Data collection UI

### Resources
- `res/xml/settings.xml` - Settings structure
- `res/values/strings.xml` - UI strings
- `res/xml/*.xml` - Keyboard layouts
- `assets/dictionaries/*.txt` - Word lists

## Testing Strategy

### Unit Tests
```bash
# Run all tests
./gradlew test

# Run specific test
./gradlew test --tests "SwipeGestureRecognizerTest"

# Test coverage
./gradlew jacocoTestReport
```

### Manual Testing Checklist
- [ ] Swipe gesture recognition
- [ ] Prediction accuracy
- [ ] Data collection
- [ ] Export functionality
- [ ] Settings integration
- [ ] Performance (latency < 50ms)
- [ ] Memory usage (< 20MB)
- [ ] Battery impact

## Debugging Tools

### ADB Commands
```bash
# Install APK
adb install -r build/outputs/apk/debug/*.apk

# View logs
adb logcat | grep -E "Keyboard2|Swipe"

# Export database
adb pull /data/data/juloo.keyboard2.debug/databases/swipe_ml_data.db

# Clear app data
adb shell pm clear juloo.keyboard2.debug
```

### Performance Profiling
```bash
# CPU profiling
adb shell am start -n juloo.keyboard2.debug/.SettingsActivity --start-profiler /data/local/tmp/keyboard.trace

# Memory profiling
adb shell am dumpheap juloo.keyboard2.debug /data/local/tmp/heap.hprof

# Battery stats
adb shell dumpsys batterystats --charged juloo.keyboard2.debug
```

## Dependencies

### Build Dependencies
- Android SDK 21+
- Gradle 7.x
- Java 8+
- AAPT2 (included for ARM64)

### Future ML Dependencies
- TensorFlow Lite 2.x
- Python 3.8+ (training)
- NumPy, Pandas (data processing)
- Matplotlib (visualization)

## Release Process

1. **Version Bump**
   - Update version in `build.gradle`
   - Update changelog

2. **Testing**
   - Run full test suite
   - Manual QA checklist
   - Performance benchmarks

3. **Build Release**
   ```bash
   ./gradlew clean
   ./gradlew assembleRelease
   ```

4. **Sign APK**
   ```bash
   jarsigner -keystore release.keystore app-release.apk alias
   zipalign -v 4 app-release-unsigned.apk app-release.apk
   ```

5. **Distribution**
   - F-Droid submission
   - GitHub release
   - Website update

## Resources & Links

### Documentation
- [README.md](../README.md) - Project overview
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guide
- [memory/swipe.md](swipe.md) - Swipe typing details
- [doc/Possible-key-values.md](../doc/Possible-key-values.md) - Key value reference

### External Resources
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/android)
- [Android IME Development](https://developer.android.com/develop/ui/views/touch-and-input/creating-input-method)
- [Weblate Translations](https://hosted.weblate.org/engage/unexpected-keyboard/)

## Contact & Support

- GitHub Issues: Project repository
- Weblate: For translations
- Wiki: Community documentation

## Notes

### Known Issues
1. Timestamp reconstruction in ML data
2. FileProvider needs AndroidX migration
3. Variable sequence length handling
4. Device normalization edge cases

### Performance Targets
- Swipe detection: < 10ms
- Prediction generation: < 50ms
- Model inference: < 30ms
- Total latency: < 100ms
- Memory overhead: < 20MB

### Privacy Considerations
- All ML data stored locally
- No network requests
- Export requires explicit user action

## Key Components (Updated 2025-01-23)

### Core Services
- `Keyboard2.java` - Main input method service with weight config integration
- `Keyboard2View.java` - Keyboard rendering and touch handling

### Swipe Prediction
- `DTWPredictor.java` - Dynamic Time Warping with Sakoe-Chiba band optimization
- `SwipeGestureRecognizer.java` - Gesture detection with loop detection integration
- `SwipeTypingEngine.java` - Orchestrates prediction with configurable weights
- `GaussianKeyModel.java` - Probabilistic key detection (30-40% accuracy gain)
- `NgramModel.java` - Language model for validation (15-25% accuracy gain)
- `LoopGestureDetector.java` - Detects circular motions for repeated letters
- `SwipeWeightConfig.java` - Singleton for user-configurable algorithm weights

### Calibration & Analysis
- `SwipeCalibrationActivity.java` - Enhanced UI with weight controls and visual feedback
- `SwipeDataAnalyzer.java` - Algorithm transparency and trace analysis

### Data Management
- `SwipeMLDataStore.java` - ML data persistence
- `SwipeDataCollector.java` - Real-time swipe data collection

### UI Components
- `SuggestionBar.java` - Word suggestion display with scores
- `WordPredictor.java` - Dictionary-based word prediction
- Opt-in data collection
- No personal information in exports