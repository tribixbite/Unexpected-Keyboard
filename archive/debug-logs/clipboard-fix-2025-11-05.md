# Clipboard System Fix - November 5, 2025

## Issues Resolved

### 🔴 Critical Issue #1: Silent Clipboard Listener Failure
**Problem**: On Android 10+ (API 29+), clipboard monitoring silently failed when app wasn't default IME. The constructor attempted to register a clipboard listener once, caught SecurityException, logged a warning, but never retried. Comment claimed "we'll retry when app gains focus" but there was NO retry logic.

**Impact**: Clipboard history stopped recording new clips. Users thought feature was broken.

**Fix Applied**:
- Added `attemptToRegisterListener()` method with retry logic
- Added `isDefaultIme()` check for Android 10+ permission requirements
- Call retry from `Keyboard2.setInputView()` when keyboard gains focus
- Added `getClipboardStatus()` for user feedback

**Files Modified**:
- `srcs/juloo.keyboard2/ClipboardHistoryService.java` - Added retry logic and permission checks
- `srcs/juloo.keyboard2/Keyboard2.java` - Call retry on input view set

---

### 🔴 Critical Issue #2: Initialization Race Condition
**Problem**: `_paste_callback` was static and set via `on_startup()`. Clipboard UI could be opened before `on_startup()` was called, leading to null callback and paste failures (or crashes in older versions without null check).

**Impact**: Pasting from clipboard history failed silently or crashed.

**Fix Applied**:
- Changed `_paste_callback` from static to instance variable
- Modified `on_startup()` to set instance variable directly
- Added logging when paste fails due to null callback
- Ensured callback initialization happens during service creation

**Files Modified**:
- `srcs/juloo.keyboard2/ClipboardHistoryService.java` - Instance variable pattern

---

### 🟠 High Priority: Misleading Documentation
**Problem**: Comment stated "history is not persisted and can be forgotten as soon as the app stops" - completely false after SQLite refactor.

**Fix Applied**: Updated documentation to accurately reflect SQLite-based persistence.

---

### 🟡 Medium Priority: Dead Code Cleanup
**Problem**: 20-line commented-out block for removed history entry deletion feature.

**Fix Applied**: Deleted dead code from `ClipboardHistoryView.java`

---

## Technical Implementation Details

### New Methods Added

#### `ClipboardHistoryService.attemptToRegisterListener()`
```java
/**
 * Attempt to register clipboard listener. Safe to call multiple times.
 * On Android 10+, requires app to be default IME for clipboard access.
 * Call this from keyboard lifecycle methods (e.g., onStartInputView) to retry.
 */
```
- Checks if listener already registered (idempotent)
- Verifies app is default IME on Android 10+
- Attempts registration with proper exception handling
- Adds current clip if registration succeeds
- Logs clear messages for debugging

#### `ClipboardHistoryService.isDefaultIme()`
```java
/**
 * Check if this keyboard is set as the default input method.
 * Required for clipboard access on Android 10+.
 */
```
- Queries system settings for default IME
- Returns true if this app is default
- Used to prevent permission failures

#### `ClipboardHistoryService.getClipboardStatus()`
```java
/**
 * Get clipboard feature status for user feedback.
 * Returns status message indicating if clipboard monitoring is active.
 */
```
- Shows if history is disabled in settings
- Indicates when default IME is required
- Shows active entry count when monitoring
- Useful for settings UI or debug views

---

## Testing Recommendations

### Manual Testing Steps
1. **Test Default IME Requirement** (Android 10+):
   - Install app but don't set as default keyboard
   - Open clipboard pane - should log warning about permissions
   - Set as default keyboard and reopen - should activate monitoring

2. **Test Persistence**:
   - Copy several text items with keyboard active
   - Force stop the app
   - Reopen clipboard pane - items should persist

3. **Test Paste Functionality**:
   - Add items to clipboard history
   - Click paste button on each item
   - Verify text is inserted into editor

4. **Test Listener Retry**:
   - Start app, minimize (keyboard loses focus)
   - Copy text in another app
   - Return to keyboard - listener should reactivate and capture clip

### Log Monitoring
Check logcat for these messages:
- `ClipboardHistory: Clipboard listener registered successfully`
- `ClipboardHistory: Clipboard access requires this keyboard to be set as default input method`
- `ClipboardHistory: Cannot paste - callback not initialized` (should never appear after fix)

---

## Code Quality Improvements

### Before Fix
- ❌ Silent failures on Android 10+
- ❌ No retry logic despite comments claiming retry
- ❌ Race condition in callback initialization
- ❌ Misleading documentation
- ❌ Dead code cluttering view class

### After Fix
- ✅ Explicit permission checks with logging
- ✅ Retry logic when keyboard gains focus
- ✅ Safe callback initialization
- ✅ Accurate documentation
- ✅ Clean, maintainable code
- ✅ User-facing status method

---

## Database Layer (Already Functional)

The persistence layer was **already well-implemented**:
- ✅ SQLite database with proper schema
- ✅ Indices for performance (content_hash, timestamp, expiry)
- ✅ Duplicate detection via content hashing
- ✅ TTL support with pinning override
- ✅ Configurable size limits
- ✅ Thread-safe singleton pattern

**The persistence wasn't broken - it just wasn't being initialized properly due to permission restrictions.**

---

## Build Information

**Build**: v1.32.282-332
**APK**: `build/outputs/apk/debug/juloo.keyboard2.debug.apk`
**Size**: 58MB
**Status**: BUILD SUCCESSFUL

---

## Commit Information

**Commit**: 0256305a
**Branch**: feature/swipe-typing
**Message**: fix(clipboard): resolve crashes and persistence failures

**Files Changed**:
- `srcs/juloo.keyboard2/ClipboardHistoryService.java` (+96/-15)
- `srcs/juloo.keyboard2/Keyboard2.java` (+6/-0)
- `srcs/juloo.keyboard2/ClipboardHistoryView.java` (-20)
- `build.gradle` (version bump)

---

## Next Steps / Recommendations

### Immediate
1. **Test on Android 10+ device** to verify permission handling
2. **Test persistence** across app restarts
3. **Monitor logs** for any remaining permission issues

### Future Enhancements
1. **Settings UI Integration**: Show clipboard status from `getClipboardStatus()`
2. **User Guidance**: Add dialog when clipboard access denied directing user to enable as default IME
3. **Prepared Statements**: Optimize database queries (low priority)
4. **Dependency Injection**: Refactor away from static singletons (architectural improvement)

### Documentation Updates
- Update user documentation about default IME requirement on Android 10+
- Add troubleshooting section for clipboard permission issues
- Document clipboard status API for settings integration

---

## Root Cause Analysis Summary

**Why clipboard history wasn't persisting**:
1. Listener registration failed on Android 10+ due to permission restrictions
2. SecurityException was caught but never retried
3. No user feedback about permission requirements
4. Database worked fine - just never received new clips

**Why paste was crashing**:
1. Callback initialization race condition
2. Static variable could be null when UI opened early
3. Some code paths may have lacked null checks (now all safe)

**Both issues are now resolved with proper lifecycle management and permission handling.**
