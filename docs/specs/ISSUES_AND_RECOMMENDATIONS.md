# Issues and Recommendations

**Date**: 2025-11-28
**Version**: v1.32.948

## 1. Summary of Investigation

After thorough code analysis, the following issues were investigated:

### 1.1 DELETE_LAST_WORD Function
**Status**: ✅ Code path is correct

**Call Chain**:
1. Layout XML: `nw="delete_last_word"` on backspace key
2. KeyValue.kt: `"delete_last_word" -> editingKey("word", Editing.DELETE_LAST_WORD)`
3. KeyEventHandler.kt:99: `KeyValue.Kind.Editing -> handleEditingKey(key.getEditing())`
4. KeyEventHandler.kt:244: `KeyValue.Editing.DELETE_LAST_WORD -> recv.handle_delete_last_word()`
5. KeyboardReceiver.kt:228-229: `override fun handle_delete_last_word() { keyboard2.handleDeleteLastWord() }`
6. Keyboard2.kt:620-622: `fun handleDeleteLastWord() { _suggestionBridge.handleDeleteLastWord() }`
7. SuggestionBridge.kt:81-86: Delegates to SuggestionHandler with IC and EditorInfo
8. SuggestionHandler.kt:593-717: Implements actual deletion logic

**Implementation Logic**:
- Termux: Uses Ctrl+W (delete word backward terminal sequence)
- Normal apps: Uses `deleteSurroundingText()` with word boundary detection
- Fallback: Ctrl+Backspace key event

**Recommendation**: Runtime testing needed - code appears correct.

### 1.2 Settings/Clipboard Short Swipe Triggers
**Status**: ✅ Code path is correct

**Key Positions** (from bottom_row.xml):
- `switch_clipboard`: key3 (SW/southwest) on ctrl key
- `config` (settings): key4 (SE/southeast) on fn key

**Short Gesture Logic** (Pointers.kt:209-303):
- Enabled via `_config.short_gestures_enabled`
- For non-char keys: `shouldBlockGesture = isCharKey && ptr.modifiers.size() > 0` (false for ctrl/fn)
- Direction calculation: Uses `atan2(dy, dx)` and `DIRECTION_TO_INDEX` mapping
- Fallback: `getNearestKeyAtDirection()` searches nearby if exact direction undefined

**Recommendation**: Runtime testing needed. If not working, check logcat for "SHORT_SWIPE" debug messages.

### 1.3 Auto-Capitalization
**Status**: ✅ Code appears correctly migrated

**Implementation** (Autocapitalisation.kt):
- Enabled via `Config.globalConfig().autocapitalisation`
- Trigger characters: Only space (' ') triggers caps check
- Uses `InputConnection.getCursorCapsMode(capsMode)` to determine if shift should enable

**Potential Issue**: The `is_trigger_character()` function only checks for space. Punctuation (.!?) does not trigger auto-cap.

**Recommendation**: Add punctuation to trigger characters:
```kotlin
private fun is_trigger_character(c: Char): Boolean {
    return when (c) {
        ' ', '.', '!', '?' -> true
        else -> false
    }
}
```

### 1.4 Shift+Swipe for ALL CAPS
**Status**: ✅ Implementation exists but may have issues

**Implementation** (InputCoordinator.kt:54-55, 408-415):
- Tracks: `wasShiftActiveAtSwipeStart: Boolean`
- Set at: Line 732 in `handleSwipeTyping()`
- Applied at: Line 410-415 converts word to uppercase if `wasShiftActiveAtSwipeStart && isSwipeAutoInsert`

**Flow**:
1. Keyboard2View.kt:309: `val wasShiftActive = _mods.has(KeyValue.Modifier.SHIFT)`
2. Keyboard2View.kt:312: Passes to `handleSwipeTyping()`
3. Keyboard2.kt:643: Passes to InputCoordinator
4. InputCoordinator.kt:732: Stores in `wasShiftActiveAtSwipeStart`

**Recommendation**: Runtime testing needed. Shift state may clear before swipe ends.

### 1.5 Punctuation Insertion After Swiped Words
**Status**: ⚠️ Not fully implemented

**Current Behavior**:
- Swiped words are inserted with trailing space (InputCoordinator.kt:426)
- Period must be manually typed
- No automatic period insertion before space

**Recommendation**: Implement double-space-to-period feature:
- When user types space twice quickly, replace with ". "
- Track last character committed and timestamp

## 2. Hardcoded Constants That Should Be Settings

### 2.1 High Priority (User-Visible Behavior)

| Constant | File | Current Value | Suggested Setting Name |
|----------|------|---------------|----------------------|
| `maxTapDurationMs` | GestureClassifier.kt:11 | 150ms | tap_duration_threshold |
| `MIN_DWELL_TIME_MS` | ImprovedSwipeGestureRecognizer.kt:36 | 10L | swipe_min_dwell_time |
| `MIN_SWIPE_DISTANCE` | ImprovedSwipeGestureRecognizer.kt:35 | 50.0f | swipe_min_distance |
| `MIN_KEY_DISTANCE` | ImprovedSwipeGestureRecognizer.kt:37 | 40.0f | swipe_min_key_distance |
| `NOISE_THRESHOLD` | ImprovedSwipeGestureRecognizer.kt:41 | 2.0f | swipe_noise_threshold |
| `HIGH_VELOCITY_THRESHOLD` | ImprovedSwipeGestureRecognizer.kt:45 | 1000.0f | swipe_high_velocity_threshold |
| `SLIDING_SPEED_SMOOTHING` | Pointers.kt:1022 | 0.7f | slider_speed_smoothing |
| `SLIDING_SPEED_MAX` | Pointers.kt:1023 | 4f | slider_speed_max |

### 2.2 Medium Priority (Algorithm Tuning)

| Constant | File | Current Value | Suggested Setting Name |
|----------|------|---------------|----------------------|
| `SAMPLING_POINTS` | EnhancedWordPredictor.kt:486 | 50 | swipe_sampling_points |
| `SMOOTHING_WINDOW` | EnhancedWordPredictor.kt:493 | 3 | swipe_smoothing_window |
| `SMOOTHING_FACTOR` | EnhancedWordPredictor.kt:494 | 0.5f | swipe_smoothing_factor |
| `SHAPE_WEIGHT` | EnhancedWordPredictor.kt:487 | 0.4f | swipe_shape_weight |
| `LOCATION_WEIGHT` | EnhancedWordPredictor.kt:488 | 0.3f | swipe_location_weight |
| `FREQUENCY_WEIGHT` | EnhancedWordPredictor.kt:489 | 0.3f | swipe_frequency_weight |
| `LENGTH_PENALTY` | EnhancedWordPredictor.kt:490 | 0.1f | swipe_length_penalty |
| `LAMBDA` | BigramModel.kt:21 | 0.95f | bigram_interpolation_lambda |
| `UNIGRAM_WEIGHT` | NgramModel.kt:314 | 0.1f | ngram_unigram_weight |
| `BIGRAM_WEIGHT` | NgramModel.kt:315 | 0.3f | ngram_bigram_weight |
| `TRIGRAM_WEIGHT` | NgramModel.kt:316 | 0.6f | ngram_trigram_weight |

### 2.3 Low Priority (Advanced/Debug)

| Constant | File | Current Value | Suggested Setting Name |
|----------|------|---------------|----------------------|
| `CLOSURE_THRESHOLD` | LoopGestureDetector.kt:282 | 30.0f | loop_closure_threshold |
| `MIN_LOOP_RADIUS` | LoopGestureDetector.kt:273 | 15.0f | loop_min_radius |
| `MIN_LOOP_POINTS` | LoopGestureDetector.kt:279 | 8 | loop_min_points |
| `ROTATION_THRESHOLD` | Gesture.kt:127 | 2 | gesture_rotation_threshold |
| `MIN_CONFIDENCE_THRESHOLD` | LanguageDetector.kt:16 | 0.6f | language_detection_confidence |
| `MIN_TEXT_LENGTH` | LanguageDetector.kt:17 | 10 | language_detection_min_length |

## 3. Settings Already Exposed in Config.kt

The following settings ARE properly exposed:
- `swipe_typing_enabled`
- `short_gestures_enabled`, `short_gesture_min_distance`
- `neural_prediction_enabled`, `neural_beam_width`, `neural_confidence_threshold`
- `swipe_confidence_weight`, `swipe_frequency_weight`
- `swipe_common_words_boost`, `swipe_top5000_boost`, `swipe_rare_words_penalty`
- `autocorrect_enabled`, `autocorrect_min_word_length`, `autocorrect_char_match_threshold`
- `prediction_context_boost`, `prediction_frequency_scale`
- `longPressTimeout`, `longPressInterval`

## 4. Action Items

### ✅ Fixes Applied

1. **Auto-capitalization punctuation triggers** (Autocapitalisation.kt:149-154)
   - ✅ Added '.', '!', '?', '\n' to `is_trigger_character()`

2. **Double-space-to-period** (KeyEventHandler.kt:207-236)
   - ✅ Tracks last character + timestamp
   - ✅ Replaces double-space with ". " within configurable threshold
   - ✅ Properly resets tracking to prevent triple-space issues

3. **Advanced Gesture Tuning Settings** (v1.32.953)
   - ✅ Created "Advanced Gesture Tuning" preference screen in Settings
   - ✅ Exposed 7 high-priority hardcoded constants as user settings:
     - `tap_duration_threshold` (150ms) - Max duration for tap gesture
     - `double_space_threshold` (500ms) - Period replacement timing
     - `swipe_min_dwell_time` (10ms) - Key registration during swipe
     - `swipe_noise_threshold` (2.0px) - Movement noise filter
     - `swipe_high_velocity_threshold` (1000 px/sec) - Fast swipe detection
     - `slider_speed_smoothing` (0.7) - Slider movement smoothing
     - `slider_speed_max` (4.0x) - Maximum slider acceleration
   - ✅ Updated GestureClassifier.kt to use Config.tap_duration_threshold
   - ✅ Updated KeyEventHandler.kt to use Config.double_space_threshold
   - ✅ Updated ImprovedSwipeGestureRecognizer.kt to use Config values
   - ✅ Updated Pointers.kt to use Config slider speed values

### Settings Remaining (Lower Priority)

1. Algorithm tuning weights (SAMPLING_POINTS, SHAPE_WEIGHT, etc.)
2. Loop gesture thresholds (CLOSURE_THRESHOLD, MIN_LOOP_RADIUS)
3. Language detection settings (already partially exposed)

### Testing Required

1. Build and deploy to device
2. Test DELETE_LAST_WORD with logcat monitoring
3. Test short swipe on ctrl→clipboard and fn→settings
4. Test shift+swipe produces ALL CAPS
5. Test auto-capitalization after sentences
6. Test double-space-to-period feature

## 5. Files Modified/Created

### v1.32.952 (Session 11)
- Created: `docs/specs/ISSUES_AND_RECOMMENDATIONS.md` (this file)
- Modified: `Autocapitalisation.kt` - Added punctuation triggers
- Modified: `KeyEventHandler.kt` - Added double-space-to-period feature

### v1.32.953 (Session 12)
- Modified: `Config.kt` - Added 7 new configurable settings
- Modified: `GestureClassifier.kt` - Use Config.tap_duration_threshold
- Modified: `KeyEventHandler.kt` - Use Config.double_space_threshold
- Modified: `ImprovedSwipeGestureRecognizer.kt` - Use Config for thresholds
- Modified: `Pointers.kt` - Use Config for slider speed values
- Modified: `res/xml/settings.xml` - Added "Advanced Gesture Tuning" screen
- Modified: `.github/workflows/build.yml` - Update to main branch
- Modified: `.github/workflows/deploy-web-demo.yml` - Update to main branch
- Modified: `gen_layouts.py` - Fix XML encoding case for CI

### v1.32.954 (Session 13)
- Archived ~100 obsolete MD files to `archive/` directories
  - `archive/debug-logs/` - Bug analyses and debug sessions
  - `archive/migration-docs/` - Kotlin migration documentation
  - `archive/planning-docs/` - Phase planning documents
  - `archive/session-notes/` - Development session notes
  - `archive/memory-archive/` - Superseded memory files
- Cleaned root to essential files: CHANGELOG.md, CLAUDE.md, CONTRIBUTING.md, README.md
- Kept active docs in: docs/, docs/specs/, memory/
