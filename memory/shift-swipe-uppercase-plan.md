# Shift+Swipe Uppercase Feature Plan

## Requirement
When shift is pressed/latched and user performs swipe typing (long swipe across keys), the resulting word should be ALL UPPERCASE instead of just capitalizing the first letter.

## Current Behavior
- Shift+swipe only capitalizes first letter (e.g., "Hello")
- This is inherited from standard autocapitalization

## Desired Behavior
- Shift+swipe should produce ALL CAPS (e.g., "HELLO")

## Implementation Strategy

### 1. Detect Shift State During Swipe
**Location**: `Keyboard2View.kt:301 onSwipeEnd()`

Need to capture modifier state when swipe begins and ends:
- Check if shift is latched in Pointers
- Pass this information through the swipe handling chain

### 2. Apply Uppercase Transformation
**Location**: `InputCoordinator.kt:392 onSuggestionSelected()`

Before building `textToInsert`, check if swipe was performed with shift active:
```kotlin
// Pseudo-code:
if (wasSwipeWithShift) {
    processedWord = processedWord.uppercase()
}
```

### 3. Implementation Steps

1. Add shift state tracking to swipe data:
   - Keyboard2View captures shift state at swipe start
   - Pass through handleSwipeTyping chain
   - Store in contextTracker or currentSwipeData

2. Modify onSuggestionSelected:
   - Check if shift was active during swipe
   - Apply uppercase() transformation before commit
   - Clear shift state after insertion

3. Alternative: Check shift state at commit time:
   - Use keyeventhandler to check current shift state
   - Simpler but might miss if shift is released quickly

### 4. Code Locations

- **Shift detection**: Pointers.kt (has modifier tracking)
- **Swipe initiation**: Keyboard2View.kt:301-314
- **Swipe handling**: Keyboard2.kt:634-643 → InputCoordinator.kt:670-778
- **Text commit**: InputCoordinator.kt:211-420 (onSuggestionSelected)

### 5. Edge Cases

- What if shift is released mid-swipe? → Use shift state at swipe START
- What about fn+swipe or ctrl+swipe? → Only apply to SHIFT modifier
- Capslock vs shift? → Treat capslock same as shift for this feature

## Status

- [ ] Implement shift state capture
- [ ] Pass through swipe chain
- [ ] Apply uppercase transformation
- [ ] Test edge cases
- [ ] Update changelog

