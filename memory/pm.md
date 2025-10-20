# Project Management - Unexpected Keyboard

**Quick Links**:
- 📋 **[CHANGELOG.md](CHANGELOG.md)** - Complete version history with technical details
- 🧠 **[swipe.md](swipe.md)** - ML/Neural network implementation details
- 📚 **[../CLAUDE.md](../CLAUDE.md)** - Build commands and development workflow

---

## 🔥 Current Status (2025-10-20)

**Latest Version**: v1.32.157 (206)
**Build Status**: ✅ BUILD SUCCESSFUL
**Branch**: feature/swipe-typing

### Recent Work (v1.32.157)

**Dictionary Manager UI - COMPLETE**
- Modern Material Design dark mode UI with 4 tabs
- Active Words: View/disable words from main BigramModel dictionary
- Disabled Words: Re-enable previously disabled words
- User Dictionary: Manage Android system user dictionary
- Custom Words: Add/edit/delete app-specific custom words
- Real-time search with 300ms debouncing
- Filter dropdown: all/active/disabled/user
- Auto-switch tabs when search has no results
- RecyclerView + DiffUtil for efficient list updates
- ViewPager2 + TabLayout + Fragments architecture
- Kotlin-based with coroutines for async operations
- SharedPreferences for disabled/custom word storage
- BigramModel.getAllWords() exposes dictionary
- APK size: 43MB → 47MB (Material Design + Kotlin libs)
- Access via Settings → "📚 Dictionary Manager"

**Previous (v1.32.156)**: Removed migration code, no backwards compat needed

**Previous (v1.32.152)**: Fixed import to store ListPreferences as strings - COMPLETE
- Root cause: ListPreference ALWAYS stores values as strings, even numeric ones
- Crashed importing: circle_sensitivity="2", clipboard_history_limit="0" as integers
- ClassCastException: `Integer cannot be cast to String` in ListPreference.onSetInitialValue
- Solution: Removed ALL entries from isIntegerStoredAsString - ListPreferences handle conversion internally
- Backup/restore now FULLY FUNCTIONAL - all 171 preferences import correctly

**Previous (v1.32.151)**: Gemini-validated fixes (show_numpad, JsonArray guards, export logging)

**Previous (v1.32.143)**: Float vs Int type detection fix (8 float preferences whitelisted)

**Previous (v1.32.141)**: **Full Backup/Restore with Layouts & Extra Keys** - Gemini-validated JSON handling
- Properly exports and restores layouts, extra_keys, and custom_extra_keys
- Parses JSON-string preferences during export to avoid double-encoding
- Converts JsonElement back to JSON string during import
- All user settings now fully restorable (previously layouts/extra_keys were skipped)
- Only internal state preferences excluded (version, current_layout indices)

**Previous (v1.32.138)**: **Improved Backup/Restore Robustness** - Gemini-validated enhancements
- Handle integer-as-string preferences (circle_sensitivity, show_numpad, etc.)
- Relaxed theme validation for forward compatibility
- Prevents ClassCastException from ListPreference values

**Previous (v1.32.137)**: **Fixed Backup/Restore Crash** - Blacklist complex preferences
- Fixed crash loop when importing settings
- Skip preferences with custom serialization (layouts, extra_keys, etc.)
- These preferences have dedicated save/load methods in their classes
- Settings activity now works properly after restore

**Previous (v1.32.136)**: **Backup/Restore Configuration System** - Complete settings management
- Replaced non-functional ML data settings with proper backup/restore
- Export all preferences to `kb-config-YYYYMMDD_HHMMSS.json` with metadata
- Version-tolerant import (accepts any recognized keys, skips unknown)
- Uses Storage Access Framework (Android 15 compatible, no permissions)
- Validates ranges for integers/floats on import
- Warns about screen size mismatches from different devices
- Prompts for app restart after restore
- Added Gson dependency for robust JSON serialization

**Previous (v1.32.133)**: **17 Two-Letter Word Shortcuts** - Added "be", reorganized layout
- Added: be (b→NW)
- Reorganized: me (m→NW from NE), as (a→E from S), quote (m→NE)
- Complete list (17): to, it, as, so, do, up, me, we, in, of, on, hi, no, go, by, is, be
- All include auto-space for faster typing

**Previous (v1.32.132)**: Added "is" (i→SW), moved * to i→NW

**Previous (v1.32.131)**: Auto-spacing for all 2-letter words
- All 15 words insert with trailing space ("to " instead of "to")
- Reorganized: `of`(o→NW), `we`(w→SE), `-`(g→NW), `go`(g→NE)

**Previous (v1.32.130)**: Added go, by; reorganized me position

**Previous (v1.32.129)**: Fixed do/so directions, added 6 words (we, in, of, on, hi, no)

**Previous (v1.32.128)**: SE Hit Zone Expansion
- Expanded SE position from 22.5° to 45° hit zone (makes `}` and `]` easier)
- Changed DIRECTION_TO_INDEX: dirs 4-6 → SE (was 5-6)

**Previous (v1.32.122-127)**: Swipe Symbols Documentation & Debug Logging
- Created comprehensive spec: `docs/specs/SWIPE_SYMBOLS.md`
- Added detailed direction logging: `adb logcat | grep SHORT_SWIPE`

**Previous (v1.32.114-121)**: Auto-Correction Feature & WordPredictor Refactor
- Fuzzy matching auto-correction with capitalization preservation
- Removed legacy swipe fallback system (~200 lines)
- Unified scoring with early fusion

**Files**: `Pointers.java`, `docs/specs/SWIPE_SYMBOLS.md`

See [CHANGELOG.md](CHANGELOG.md) for detailed technical documentation.

---

## 📌 Known Issues

### High Priority
None currently

### Medium Priority
- **Code Organization**: `Keyboard2.java` is 1200+ lines (needs splitting)
- **Documentation**: Some legacy docs need updating

### Low Priority
- **Swipe Symbol UX**: NE position still has narrow hit zone (22.5°) - SE fixed to 45° in v1.32.128
- **SwipeDebugActivity**: EditText focus issue (Android IME architectural limitation)
- Consider adding undo mechanism for auto-correction
- Consider adding more common word shortcuts (is, we, go, on, in, etc.)

---

## 🎯 Next Steps

### Immediate Tasks
1. Test auto-correction in both Termux and normal apps
2. Refactor `Keyboard2.java` into smaller focused files

### Future Enhancements
- Consider ML-based auto-correction (learning from user corrections)
- Improve context model with n-gram support (currently bigram only)
- Add spell-check dictionary for rare/technical words

---

## 🛠️ Quick Reference

### Build Commands
```bash
# Build debug APK
./build-on-termux.sh

# Build release APK
./build-on-termux.sh release

# Install on device
./gradlew installDebug
```

### Git Workflow
```bash
# Status
git status

# Commit
git add -A
git commit -m "type(scope): description"

# View log
git log --oneline -20
```

### Testing
```bash
# Run all tests
./gradlew test

# Check layouts
./gradlew checkKeyboardLayouts
```

---

## 📊 Project Stats

**Lines of Code** (core prediction system):
- `Keyboard2.java`: ~1200 lines (needs refactor)
- `WordPredictor.java`: ~516 lines
- `NeuralSwipeTypingEngine.java`: ~800 lines
- `BigramModel.java`: ~440 lines

**Total**: ~3000 lines of prediction/autocorrect logic

---

## 📝 Development Notes

### Architecture Principles
1. **Neural-first**: ONNX handles all swipe typing, no fallbacks
2. **Early fusion**: Apply context before selecting candidates
3. **App-aware**: Detect Termux app for smart spacing
4. **User control**: All weights configurable via settings

### Code Conventions
- Use conventional commits: `type(scope): description`
- Build and test after every change
- Update CHANGELOG.md for user-facing changes
- Document complex algorithms with inline comments

---

For complete version history and detailed technical documentation, see [CHANGELOG.md](CHANGELOG.md).
