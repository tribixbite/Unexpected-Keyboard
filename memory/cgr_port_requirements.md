# CGR Library Port Requirements

## Requirements Summary
- **DO NOT TOUCH** non-swipe prediction system - it's working perfectly (prefix-based predictions in WordPredictor.java)
- **REWRITE** gesture prediction system using the Continuous Gesture Recognizer (CGR) library
- **CAREFULLY** and **COMPLETELY** port the Lua files from memory/ directory
- **NO PLACEHOLDERS** or modifications to the mechanisms
- **CLEAN PERFECT PORT** of all functions
- **INTEGRATE** en.txt dictionary for word suggestions
- **REAL-TIME** predictions that update while user is swiping (after more than 2 chars swiped)
- **AVOID INTERFERENCE** with punctuation single key swipes
- **PERSIST** best candidate predictions until selected after swipe event
- **CLEAR** predictions on space or non-letter key

## Current System Analysis

### What NOT to Touch (Working Perfectly)
1. **WordPredictor.java** - Non-swipe prefix-based prediction system
2. **SuggestionBar.java** - Suggestion display system
3. **Regular typing predictions** - All prefix-based word completion

### What to Replace (Swipe Gesture Recognition)
1. **SwipeGestureRecognizer.java** - Current swipe gesture tracking
2. **Gesture prediction logic** - Replace with CGR-based system

### CGR Library Components (From Lua Files)
1. **CGR.lua** - Main continuous gesture recognition library
2. **Main.lua** - Integration example showing usage patterns

## Technical Requirements

### CGR Library Port
- Port all functions from CGR.lua without modification
- Maintain exact same algorithms and mathematical operations
- Preserve all constants and parameters
- Keep the same data structures and processing flow
- Implement vector math operations (vec2 equivalent)

### Word Dictionary Integration
- Use existing en.txt dictionary (word + frequency format)
- Load dictionary words as gesture templates
- Map QWERTY keyboard layout to coordinate system
- Generate gesture templates for common words

### Real-time Prediction Features
- Start predictions after 2+ characters have been swiped
- Update predictions continuously during swipe gesture
- Avoid single-key swipes (punctuation protection)
- Show top 3-5 word candidates in SuggestionBar
- Clear predictions on space/punctuation/non-letter keys
- Persist predictions until user selects one

### Integration Points
- Hook into existing Keyboard2View touch handling
- Integrate with SuggestionBar for display
- Preserve existing non-swipe prediction workflow
- Maintain current keyboard layout and key detection

## Implementation Strategy

### Phase 1: CGR Library Port
1. Create ContinuousGestureRecognizer.java
2. Port all CGR functions from Lua to Java
3. Implement Point, Template, Pattern classes
4. Add gesture recognition methods

### Phase 2: Word Template Generation
1. Load en.txt dictionary
2. Generate keyboard coordinate templates for words
3. Create gesture templates using CGR format
4. Build word-to-template mapping

### Phase 3: Real-time Integration
1. Hook into SwipeGestureRecognizer or replace it
2. Feed touch points to CGR recognition
3. Update SuggestionBar during swipe
4. Handle prediction clearing and persistence

### Phase 4: Testing and Optimization
1. Test with various swipe patterns
2. Verify non-swipe predictions still work
3. Optimize performance for real-time use
4. Fine-tune recognition parameters

## Files to Create/Modify
- **New**: ContinuousGestureRecognizer.java (CGR port)
- **New**: SwipeWordTemplateGenerator.java (template creation)
- **New**: RealTimeSwipePredictor.java (integration)
- **Modify**: SwipeGestureRecognizer.java or replace entirely
- **Preserve**: WordPredictor.java, SuggestionBar.java

## Success Criteria
- Non-swipe predictions continue working unchanged
- Swipe gestures generate real-time word predictions
- Predictions update smoothly during swipe
- Single-key swipes don't trigger predictions
- Predictions persist until selection or clearing
- System builds and runs without errors