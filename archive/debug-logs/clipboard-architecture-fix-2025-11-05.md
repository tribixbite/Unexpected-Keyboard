# Clipboard Architecture Fix - Gboard Pattern Implementation (2025-11-05)

## Problem Identified by Gemini Consultation

**Original Implementation (WRONG):**
```java
case SWITCH_CLIPBOARD:
  setInputView(_clipboard_pane);  // ❌ Replaces entire keyboard!
```

This made it impossible for users to type in the search box without an external keyboard.

---

## Solution: Gboard/SwiftKey Pattern

**Gemini's Recommendation:**
> "The dominant and most effective pattern, used by major keyboards, is to **repurpose the candidates view area** or a similar integrated 'toolbar' space above the main keyboard keys. The keyboard view itself remains visible and active."

**Visual Layout:**
```
┌────────────────────────────────────┐
│  SuggestionBar (word predictions)  │
├────────────────────────────────────┤
│  📋 Clipboard Pane                 │ ← Shows/hides as needed
│  [Search: ___________________]     │
│  • Clipboard item 1                │
│  • Clipboard item 2                │
│  • Clipboard item 3                │
├────────────────────────────────────┤
│  Q  W  E  R  T  Y  U  I  O  P      │ ← ALWAYS VISIBLE
│   A  S  D  F  G  H  J  K  L        │
│    Z  X  C  V  B  N  M             │
└────────────────────────────────────┘
```

---

## Architecture Implementation

### Before (Problematic):
```
SWITCH_CLIPBOARD → setInputView(_clipboard_pane)
  Result: Keyboard disappears, user can't type
```

### After (Gboard Pattern):
```
_inputViewContainer (LinearLayout, vertical)
  ├─ SuggestionBar (HorizontalScrollView)
  ├─ _contentPaneContainer (FrameLayout) [NEW]
  │   ├─ Visibility toggles between VISIBLE/GONE
  │   └─ Contains _clipboard_pane or _emojiPane
  └─ KeyboardView [ALWAYS VISIBLE]
```

### Code Changes

**1. Added Container Field (Line 55):**
```java
private FrameLayout _contentPaneContainer = null; // Container for emoji/clipboard panes
```

**2. Created Container During Setup (Lines 530-538):**
```java
// Add content pane container (for clipboard/emoji) between suggestion bar and keyboard
_contentPaneContainer = new FrameLayout(this);
_contentPaneContainer.setLayoutParams(new LinearLayout.LayoutParams(
  LinearLayout.LayoutParams.MATCH_PARENT,
  (int)TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 200,
    getResources().getDisplayMetrics())));
_contentPaneContainer.setVisibility(View.GONE); // Hidden by default
_inputViewContainer.addView(_contentPaneContainer);
```

**3. Show Clipboard Pane (Lines 770-782):**
```java
case SWITCH_CLIPBOARD:
  // ... initialize clipboard pane if needed ...

  // Show clipboard pane in content container (keyboard stays visible below)
  if (_contentPaneContainer != null)
  {
    _contentPaneContainer.removeAllViews();
    _contentPaneContainer.addView(_clipboard_pane);
    _contentPaneContainer.setVisibility(View.VISIBLE);  // ✓ Toggle visibility
  }
  else
  {
    // Fallback for when predictions disabled (no container)
    setInputView(_clipboard_pane);
  }
  break;
```

**4. Hide Clipboard Pane (Lines 794-804):**
```java
case SWITCH_BACK_CLIPBOARD:
  // Exit clipboard search mode
  _clipboardSearchMode = false;
  if (_clipboardSearchBox != null)
  {
    _clipboardSearchBox.setText("");
    _clipboardSearchBox.setHint("Tap to search, type on keyboard...");
  }

  // Hide content pane (keyboard remains visible)
  if (_contentPaneContainer != null)
  {
    _contentPaneContainer.setVisibility(View.GONE);  // ✓ Just hide it
  }
  else
  {
    // Fallback for when predictions disabled
    setInputView(_keyboardView);
  }
  break;
```

---

## User Experience Flow

### Opening Clipboard:
1. User swipes to clipboard icon
2. `SWITCH_CLIPBOARD` event fires
3. Content pane container becomes VISIBLE
4. Clipboard pane shows ABOVE keyboard
5. Keyboard remains visible and active below
6. User can immediately type in search box

### Searching:
1. User taps search box
2. Search mode activates (`_clipboardSearchMode = true`)
3. Hint changes: "Type on keyboard below..."
4. User types on visible keyboard
5. Input routes to search box via `appendToClipboardSearch()`
6. Results filter in real-time

### Closing:
1. User swipes back to keyboard
2. `SWITCH_BACK_CLIPBOARD` event fires
3. Content pane container becomes GONE
4. Clipboard pane hides
5. Keyboard remains visible (was never hidden)

---

## Benefits

✅ **No External Keyboard Needed** - Users can type using on-screen keyboard
✅ **Modern UX** - Matches Gboard/SwiftKey behavior users expect
✅ **No Jarring Transitions** - Keyboard stays visible, smooth experience
✅ **Proper IME Architecture** - Follows Android InputMethodService best practices
✅ **Graceful Fallback** - Works even when predictions disabled

---

## Gemini's Architectural Guidance (Key Points)

> **Q: Can we dynamically expand the input view?**
> "Technically, you could try... However, this is strongly discouraged. The IME window height is managed by the system and making drastic changes can cause screen flicker, layout jank, and a jarring UX."

> **Q: Do we need a separate Activity/Dialog?**
> "Major UX drawback: It breaks the user's context. Launching an activity takes the user out of the app they were typing in. This is likely to feel jarring."

> **Q: What's the standard Android pattern?**
> "The dominant pattern is to **repurpose the candidates view area** above the main keyboard keys. The keyboard remains visible and active."

> **Q: How do Gboard/SwiftKey handle this?**
> "They have a 'toolbar' area above the keys. When you tap clipboard, that toolbar expands. The main keyboard below doesn't change. Its key presses are intercepted and programmatically sent to the internal search EditText."

---

## Implementation Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| **Gboard Pattern (Chosen)** | ✅ Best UX<br>✅ Keyboard always visible<br>✅ Natural, integrated feel | ⚠️ More complex state management<br>⚠️ Manual input routing |
| Separate Activity/Dialog | ✅ Simple implementation<br>✅ Standard Android components | ❌ Poor UX<br>❌ Breaks typing context<br>❌ Feels like separate app |
| Expanding View | - | ❌ Fights framework<br>❌ Screen flicker/jank<br>❌ Brittle across devices |

---

## Testing Requirements

**On-Device Testing:**
1. ✅ Open clipboard pane → keyboard should remain visible below
2. ✅ Tap search box → should enter search mode with hint "Type on keyboard below..."
3. ✅ Type on keyboard → text should appear in search box
4. ✅ Search results should filter in real-time
5. ✅ Backspace should work in search box
6. ✅ Swipe back → pane should hide, keyboard should remain visible
7. ✅ Test with predictions enabled (container exists)
8. ✅ Test with predictions disabled (fallback path)
9. ✅ Test emoji pane (should work similarly)

**Build Status:**
- ✅ BUILD SUCCESSFUL
- Version: v1.32.295 (345)
- APK: /storage/emulated/0/unexpected/unexpected-keyboard-v1.32.295-345.apk
- Commit: e4fabfc8

---

## Code References

- **Container Creation**: Keyboard2.java:530-538
- **Clipboard Show**: Keyboard2.java:770-782
- **Clipboard Hide**: Keyboard2.java:786-805 (SWITCH_BACK_CLIPBOARD)
- **Emoji Show**: Keyboard2.java:728-740
- **Search Mode**: Keyboard2.java:750-761 (click handler)
- **Input Routing**: KeyEventHandler.java:219-234 (send_text routing)
- **Settings UI**: settings.xml:133 (IntSlideBarPreference for height config)
- **Layout**: clipboard_pane.xml:2 (root LinearLayout), lines 14-16 (history ScrollView)
- **Ctrl Key with ABC**: bottom_row.xml:3 (key7 southwest swipe)

---

## Related Commits

1. `2a11f728` - Initial search box keyboard routing (wrong architecture)
2. `55033e18` - Fixed architecture using Gboard pattern
3. `9d8d6193` - Fixed Settings crash (IntSlideBarPreference)
4. `31d21964` - Updated documentation
5. `5258a7bd` - Made History label visible with proper spacing
6. `022f9d46` - ABC button visible, search box theme fixed
7. `07bfa681` - Updated documentation
8. `e4fabfc8` - Removed redundant bar, southwest swipe, independent scrolling

---

## UX Improvements Implemented

### Settings Crash Fix (Commit 9d8d6193)
**Issue**: Settings Activity crashed with `ClassNotFoundException: SliderPreference`
**Fix**: Changed to `IntSlideBarPreference` which exists in codebase
- Removed invalid `unit` attribute
- Changed summary format to `%s%%` for dynamic value display
- Settings now accessible without crash

### Height Configuration (Commit 9d8d6193)
**Feature**: Configurable clipboard pane height via Settings
- Default: 30% of screen height
- Range: 10-50%
- Setting path: Settings → Clipboard → Clipboard pane height
- Dynamic slider shows percentage in real-time

### Search Bar Styling (Commit 9d8d6193)
**Fix**: Search bar matched History label styling
- Text size: 14sp (matches History label)
- Padding reduced: 4dp vertical
- Vertical centering within shared row

### Keyboard Visibility (Commit 55033e18)
**Fix**: ABC button always visible (Gboard pattern)
- Keyboard stays visible below content pane
- No need to restart keyboard to close pane
- Natural, integrated feel

### History Label Visibility (Commit 5258a7bd)
**Issue**: "History" label was invisible/cut off next to search box
**Fix**: Proper spacing and margin overrides
- Override marginTop to 0dp (was 14dp from clipboardHeading style)
- Added 8dp right margin between History and search box
- Increased parent padding top to 8dp
- Added baselineAligned for proper text alignment
- History label now clearly visible

### ABC Button & Search Theme (Commit 022f9d46)
**Issues**:
1. ABC/back button not visible (bottom row clipped)
2. Search box white background didn't respect dark theme
3. Search text visibility concerns

**Fixes**:
1. **Bottom Row Visibility**:
   - Changed ScrollView from fixed 300dp to flexible height (0dp weight=1)
   - Allows bottom row with switch_back_clipboard to render
   - Root LinearLayout uses fill_parent height

2. **Search Box Theme**:
   - Changed background from system drawable to ?attr/colorKey
   - Matches keyboard theme colors
   - Added textColorHint=?attr/colorSubLabel for hint text

3. **Search Text Visibility**:
   - Using ?attr/colorLabel for text color
   - Should be clearly visible on key background

### UX Refinements (Commit e4fabfc8)
**Issues**:
1. Redundant horizontal ABC button bar when keyboard is always visible
2. Need intuitive way to close clipboard from main keyboard
3. Lost separate scrolling for pinned vs history sections

**Fixes**:
1. **Removed Redundant Bottom Row**:
   - Deleted `<Keyboard2View layout="@xml/clipboard_bottom_row"/>` from clipboard_pane.xml
   - Bar was unnecessary since keyboard stays visible in Gboard pattern
   - Cleaner, less cluttered interface

2. **Added Southwest Swipe to Ctrl Key**:
   - Added `key7="switch_back_clipboard"` to Ctrl key in bottom_row.xml
   - Southwest swipe on Ctrl now shows "ABC" and closes clipboard
   - More intuitive: open clipboard (southeast), close clipboard (southwest)
   - Consistent with keyboard's gesture-based design

3. **Restored Independent Scrolling**:
   - Pinned section: Direct child with maxHeight="140dp" and scrollbars
   - History section: Wrapped in separate ScrollView with layout_weight="1"
   - Each section scrolls independently as originally designed
   - Better UX when sections have many items

---

## Future Enhancements

- ✅ ~~Consider making content pane height configurable~~ (DONE)
- Add smooth animations for show/hide transitions
- Implement drag-to-resize for content pane
- Add visual indicator when in search mode
- Consider ESC key to exit search mode explicitly

---

## Gemini Consultation Summary

**Model Used:** gemini-2.5-pro
**Consultation ID:** c0a9abf6-f3fc-4dfd-a79d-1b3731a8ac90

**Key Insight:**
> "Do not replace the keyboard view. Instead, augment it with a content pane above it. The most robust and user-friendly solution is to emulate the Gboard/SwiftKey pattern."

**Critical Recommendation:**
> "Think of your IME as a main keyboard view that can be 'decorated' with different content panes above it, rather than replacing the keyboard entirely."

This consultation was instrumental in identifying the architectural flaw and providing the correct solution pattern.
