# Unexpected Keyboard - Advanced Swipe Typing Fork 🚀

> **⚡ Professional swipe typing implementation with two-pass prioritized prediction system**

**Fork by [@tribixbite](https://github.com/tribixbite)** | [Original Repo](https://github.com/Julow/Unexpected-Keyboard)

## 🎯 What's New in This Fork

This fork implements a **production-ready swipe typing system** with advanced prediction algorithms:

### ✨ Core Features

#### 1. **Two-Pass Prioritized Prediction System**
- **Priority bucket**: Words matching first AND last characters of swipe
- **Secondary bucket**: Other candidates based on path matching
- **Guaranteed inclusion** of all first+last matches (no frequency bias)
- **Dynamic limits**: 10 predictions for swipes, 5 for regular typing

#### 2. **Smart Reset Behavior**
- Resets predictor on ANY non-letter input (not just specific punctuation)
- Clears state on spaces, punctuation, numbers, symbols
- Handles multi-character paste operations

#### 3. **Enhanced Dictionary System**
- **10,000+ word dictionary** from FlorisBoard
- Frequency-weighted predictions
- No fallback to basic dictionary

#### 4. **Visual Calibration System**
- Interactive QWERTY keyboard for swipe training
- Records actual touch points and timing
- Saves patterns to SharedPreferences
- Progress tracking (10 words × 2 repetitions)

#### 5. **Comprehensive Logging**
- Real-time swipe logs: `/data/data/com.termux/files/home/swipe_log.txt`
- Calibration logs: `/data/data/com.termux/files/home/calibration_log.txt`
- Debug output with timestamps and stack traces

### 🔧 Technical Implementation

```java
// Two-pass prediction algorithm
if (firstChar == seqFirst && lastChar == seqLast) {
    // Priority: Pure quality score, no frequency multiplication
    int score = 10000 + (innerMatches * 100);
    priorityMatches.add(new WordCandidate(word, score));
}
```

**Key Algorithm Features:**
- First+last character matching for long swipes
- Example: "tghgfdsasddxcfvhbnmkjhytfds" → "thanks"
- Inner character counting for ranking
- Swipe detection threshold: >6 characters

## 📱 Installation

### Pre-built APK
```bash
# Latest debug build (3.9MB)
build/outputs/apk/debug/juloo.keyboard2.debug.apk
```

### Build from Source
```bash
# Clone this fork
git clone https://github.com/tribixbite/Unexpected-Keyboard.git
cd Unexpected-Keyboard

# Build on Linux/Mac
./gradlew assembleDebug

# Build on Termux (Android)
./build-on-termux.sh

# Install
adb install -r build/outputs/apk/debug/juloo.keyboard2.debug.apk
```

## 🚀 Quick Start

1. **Enable the Keyboard**
   - Settings → Language & Input → Unexpected Keyboard → Enable

2. **Turn on Swipe Typing**
   - Open keyboard settings (swipe from spacebar)
   - Enable "Swipe Typing"
   - Enable "Word Predictions"

3. **Calibrate (Optional)**
   - Settings → Swipe Calibration
   - Follow 10 test words
   - Improves accuracy for your swipe style

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Dictionary Size | ~10,000 words |
| Max Predictions (Swipe) | 10 |
| Max Predictions (Typing) | 5 |
| Swipe Detection | >6 characters |
| APK Size | 3.9MB |
| Memory Usage | <50MB |

## 🔍 Debugging

### View Real-time Logs
```bash
# Swipe predictions
tail -f /data/data/com.termux/files/home/swipe_log.txt

# Calibration data
tail -f /data/data/com.termux/files/home/calibration_log.txt
```

### Calibration Data Location
- **SharedPreferences**: `/data/data/juloo.keyboard2.debug/shared_prefs/swipe_calibration.xml`
- **Format**: `word:duration,x1,y1,x2,y2,...`

## 🛠️ Development

### Project Structure
```
srcs/juloo.keyboard2/
├── WordPredictor.java         # Two-pass prediction system
├── Keyboard2.java              # Main service with reset logic
├── SwipeCalibrationActivity.java # Visual calibration UI
├── SuggestionBar.java          # Prediction display
└── SwipeGestureRecognizer.java # Path tracking

assets/
├── dictionaries/
│   └── en_enhanced.txt        # 10k word dictionary
└── libjni_latinimegoogle.so   # DTW library (future)
```

### Key Changes from Original

1. **Separate prediction flags** (`swipe_typing_enabled` vs `word_prediction_enabled`)
2. **Two-pass prediction** prevents frequency bias
3. **Dynamic prediction limits** based on input type
4. **Visual calibration** with keyboard display
5. **File-based logging** for debugging

## 📝 Changelog

### v2.0.0 - Swipe Typing Release
- ✅ Two-pass prioritized prediction system
- ✅ First+last character matching guarantee
- ✅ Reset on any non-letter input
- ✅ Visual calibration with QWERTY keyboard
- ✅ Comprehensive logging system
- ✅ 10k enhanced dictionary
- ✅ Dynamic prediction limits

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Integrate `libjni_latinimegoogle.so` for DTW
- [ ] Use calibration data in predictions
- [ ] Add more languages
- [ ] Implement word learning
- [ ] Add gesture customization

## 📄 License

This fork maintains the original GNU General Public License v3.0.

## 🙏 Credits

- Original Unexpected Keyboard by [@Julow](https://github.com/Julow)
- Enhanced dictionary from [FlorisBoard](https://github.com/florisboard/florisboard)
- Swipe implementation inspired by [OpenBoard](https://github.com/dslul/openboard)

## 📧 Contact

- **Fork Author**: [@tribixbite](https://github.com/tribixbite)
- **Issues**: [GitHub Issues](https://github.com/tribixbite/Unexpected-Keyboard/issues)

---

**Note**: This is an experimental fork focused on swipe typing. For the stable original version without swipe, see the [original repository](https://github.com/Julow/Unexpected-Keyboard).