# Swipe Typing Feature Implementation Roadmap

## Overview
This document outlines the complete implementation plan for adding swipe typing (gesture typing) functionality to Unexpected Keyboard. The feature allows users to type words by swiping across letters on the keyboard.

## Key Requirements
- ✅ Must not interfere with existing swipe-to-corner gestures for modifiers
- ✅ Must not interfere with long press functionality
- ✅ Must not interfere with key repeat functionality  
- ✅ Visual swipe trail must be shown while swiping
- ✅ Feature must be toggleable in settings
- ✅ Must work alongside existing keyboard features

## Implementation Phases

### Phase 1: Core Infrastructure ✅
**Goal**: Establish foundation for swipe typing without breaking existing features

#### 1.1 Swipe Detection System
- [x] Create `SwipeGestureRecognizer.java` 
  - Track continuous touch paths across keys
  - Differentiate between swipe typing and regular swipes
  - Minimum 2 alphabetic keys required for swipe typing
  - Track timestamps and distances

#### 1.2 Pointer System Integration
- [ ] Update `Pointers.java`
  - Add swipe typing state tracking
  - Ensure no interference with:
    - Long press detection (FLAG_P_LATCHABLE)
    - Key repeat functionality  
    - Sliding keys (space bar slider)
  - Add `FLAG_P_SWIPE_TYPING` flag
  - Skip long press timer when swipe typing detected

#### 1.3 Gesture State Management
- [ ] Update `Gesture.java`
  - Add `SwipeTyping` and `Ended_swipe_typing` states
  - Add `SwipeType` gesture name
  - Ensure gesture state machine handles new states

### Phase 2: Word Prediction System

#### 2.1 Dictionary Management
- [ ] Create `DictionaryManager.java`
  - Load language-specific dictionaries from assets
  - Support user custom words
  - Cache dictionaries in memory
  - Handle multiple languages

#### 2.2 Word Prediction Engine  
- [ ] Create `WordPredictor.java`
  - Implement pattern matching algorithm
  - Use edit distance with keyboard adjacency
  - Rank predictions by frequency
  - Support fuzzy matching for inaccurate swipes

#### 2.3 Dictionary Files
- [ ] Create `assets/dictionaries/` directory
- [ ] Add language dictionaries:
  - `en.txt` - English (with frequencies)
  - `es.txt` - Spanish
  - `fr.txt` - French
  - `de.txt` - German
  - Format: `word[TAB]frequency`

### Phase 3: Visual Feedback & UI

#### 3.1 Swipe Trail Rendering
- [ ] Update `Keyboard2View.java`
  - Add `_swipeTrailPaint` for trail visualization
  - Draw trail in `onDraw()` method
  - Clear trail on gesture end
  - Make trail color/width configurable

#### 3.2 Suggestion Bar UI
- [ ] Create `SuggestionBar.java`
  - Display top 5 word predictions
  - Support tap-to-insert
  - Highlight primary suggestion
  - Handle suggestion selection callbacks

#### 3.3 Layout Integration
- [ ] Create `res/layout/keyboard_with_suggestions.xml`
  - LinearLayout with suggestion bar on top
  - Keyboard view below
  - Configurable suggestion bar height

### Phase 4: Settings & Configuration

#### 4.1 Configuration Options
- [ ] Update `Config.java`
  ```java
  public boolean swipe_typing_enabled;      // Master toggle
  public boolean show_suggestion_bar;       // Show/hide suggestions
  public int suggestion_bar_height;         // Height in dp (30-60)
  public boolean swipe_trail_visible;       // Show/hide trail
  public int swipe_trail_color;            // Trail color
  public float swipe_trail_width;          // Trail width in dp
  public boolean auto_space_after_word;    // Auto-add space
  public boolean vibrate_on_word_commit;   // Vibration feedback
  ```

#### 4.2 Settings UI
- [ ] Update `res/xml/settings.xml`
  - Add "Swipe Typing" preference category
  - Master enable/disable toggle
  - Suggestion bar settings
  - Visual feedback settings
  - Behavior settings

#### 4.3 String Resources
- [ ] Update `res/values/strings.xml`
  ```xml
  <string name="pref_swipe_typing_title">Enable swipe typing</string>
  <string name="pref_swipe_typing_summary">Type words by swiping across letters</string>
  <string name="pref_show_suggestion_bar_title">Show suggestions</string>
  <string name="pref_suggestion_bar_height_title">Suggestion bar height</string>
  <string name="pref_swipe_trail_visible_title">Show swipe trail</string>
  <string name="pref_auto_space_title">Auto-space after words</string>
  ```

### Phase 5: Main Integration

#### 5.1 Keyboard2 Integration
- [ ] Update `Keyboard2.java`
  - Initialize `DictionaryManager` in `onCreate()`
  - Create suggestion bar when swipe typing enabled
  - Handle `handleSwipeTyping()` method
  - Implement `commitSuggestion()` method
  - Update view creation logic

#### 5.2 Touch Event Handling
- [ ] Update `Keyboard2View.onTouch()`
  - Start swipe tracking on ACTION_DOWN
  - Update path on ACTION_MOVE
  - Trigger prediction on ACTION_UP
  - Ensure no interference with:
    - Multi-touch (check pointer count)
    - Special keys (non-alphabetic)
    - Modifier keys

#### 5.3 Input Method Integration
- [ ] Handle word commitment
  - Use `InputConnection.commitText()`
  - Add automatic spacing
  - Handle composing text
  - Clear suggestions after commit

### Phase 6: Testing & Quality

#### 6.1 Unit Tests
- [ ] Create `test/SwipeGestureRecognizerTest.java`
  - Test gesture detection
  - Test path tracking
  - Test key sequence building

- [ ] Create `test/WordPredictorTest.java`
  - Test prediction accuracy
  - Test edit distance calculation
  - Test frequency ranking

#### 6.2 Integration Tests
- [ ] Test with existing features:
  - Long press still works
  - Key repeat still works
  - Swipe-to-corner modifiers work
  - Space bar sliding works
  - Circle gestures work

#### 6.3 Edge Cases
- [ ] Handle rapid swipes
- [ ] Handle very slow swipes
- [ ] Handle backtracking paths
- [ ] Handle non-alphabetic key interruptions
- [ ] Handle screen rotation
- [ ] Handle keyboard switching

### Phase 7: Performance & Polish

#### 7.1 Performance Optimization
- [ ] Implement dictionary caching
- [ ] Use background thread for predictions
- [ ] Optimize path rendering
- [ ] Minimize memory allocations

#### 7.2 User Experience
- [ ] Add haptic feedback on word selection
- [ ] Smooth trail animation
- [ ] Suggestion animation
- [ ] Loading indicators for dictionaries

#### 7.3 Accessibility
- [ ] Screen reader support for suggestions
- [ ] High contrast mode support
- [ ] Large text support

## Implementation Order

1. **Core Safety** - Ensure no interference with existing features
2. **Basic Swipe Detection** - Get path tracking working
3. **Settings Toggle** - Allow enabling/disabling
4. **Visual Trail** - Show swipe path
5. **Word Prediction** - Basic dictionary matching
6. **Suggestion UI** - Display predictions
7. **Integration** - Connect all components
8. **Testing** - Verify everything works
9. **Polish** - Optimize and refine

## Testing Checklist

### Functional Tests
- [ ] Swipe typing can be enabled/disabled
- [ ] Trail appears when swiping
- [ ] Suggestions appear for valid swipes
- [ ] Tapping suggestion inserts word
- [ ] Space automatically added after word
- [ ] Non-letter keys ignored during swipe

### Compatibility Tests  
- [ ] Long press menu still appears
- [ ] Key repeat still works
- [ ] Swipe-to-corner modifiers work
- [ ] Space bar slider works
- [ ] Number/symbol keys work normally
- [ ] Emoji panel works
- [ ] Clipboard panel works

### Edge Case Tests
- [ ] Single key press works normally
- [ ] Very short swipes work as taps
- [ ] Multi-touch ignored for swipe typing
- [ ] Screen rotation preserves state
- [ ] Language switching updates dictionary

## Success Criteria

1. **No Regressions** - All existing features continue to work
2. **Accurate Predictions** - 80%+ accuracy for common words
3. **Responsive UI** - <100ms prediction time
4. **Smooth Animation** - 60fps trail rendering
5. **Low Memory** - <10MB for dictionaries
6. **User Control** - All features configurable

## Files to Create/Modify

### New Files
- `srcs/juloo.keyboard2/SwipeGestureRecognizer.java` ✅
- `srcs/juloo.keyboard2/WordPredictor.java`
- `srcs/juloo.keyboard2/DictionaryManager.java`
- `srcs/juloo.keyboard2/SuggestionBar.java`
- `res/layout/keyboard_with_suggestions.xml`
- `assets/dictionaries/en.txt`
- `test/juloo.keyboard2/SwipeGestureRecognizerTest.java`
- `test/juloo.keyboard2/WordPredictorTest.java`

### Modified Files
- `srcs/juloo.keyboard2/Pointers.java`
- `srcs/juloo.keyboard2/Gesture.java`
- `srcs/juloo.keyboard2/Keyboard2View.java`
- `srcs/juloo.keyboard2/Keyboard2.java`
- `srcs/juloo.keyboard2/Config.java`
- `res/xml/settings.xml`
- `res/values/strings.xml`

## Notes

- Swipe typing should only activate for alphabetic keys
- Minimum swipe distance required to avoid accidental activation
- Trail should disappear immediately when finger lifted
- Predictions should be language-aware
- User dictionary should persist across sessions
- Feature should be disabled by default initially