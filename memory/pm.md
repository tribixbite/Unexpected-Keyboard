# Project Management - Unexpected Keyboard

**Quick Links**:
- 📋 **[CHANGELOG.md](CHANGELOG.md)** - Complete version history with technical details
- 🧠 **[swipe.md](swipe.md)** - ML/Neural network implementation details
- 📚 **[../CLAUDE.md](../CLAUDE.md)** - Build commands and development workflow

---

## 🔥 Current Status (2025-10-22)

**Latest Version**: v1.32.198 (247)
**Build Status**: ✅ BUILD SUCCESSFUL - Raw/Closest Predictions Restored
**Branch**: feature/swipe-typing

### Recent Work (v1.32.198)

**Raw/Closest Predictions Restored**
- **Issue**: v1.32.194 removed raw predictions from UI (made them log-only)
- **Impact**: Horizontal scroll bar had nothing extra to show, users couldn't see NN's actual predictions
- **Fix**: Re-added top 3 raw beam search predictions to UI
  - Shows what neural network actually predicted vs vocabulary filtering
  - Clean format: just the words, no bracketed markers in UI
  - Only added if not already in filtered results
  - Scored based on NN confidence (0-1000 range)
- **Example**:
  - Filtered: "hello" (vocab-validated, frequency boosted)
  - Raw/Closest: "helo", "hallo" (NN predicted, may be filtered by vocab)
- **Impact**: Users can now see all predictions, horizontal scroll works properly
- **Files**: OnnxSwipePredictor.java

**Previous (v1.32.197)**: Dictionary Manager System Freeze Fix

### Previous Work (v1.32.197)

**Dictionary Manager System Freeze Fix - AsyncListDiffer + Coroutine Cancellation**
- **Root Cause Analysis**: Complete system freeze when typing in Dictionary Manager search
  - DiffUtil.calculateDiff() ran synchronously on main thread with 50k words
  - O(n²) complexity: 50k × 50k = 2.5 billion comparisons per fragment
  - All 4 fragments updated simultaneously on every keystroke
  - Main thread blocked for 100ms+ per fragment (400ms+ total UI freeze)
  - On slower devices (Termux ARM64) caused complete system lockup
- **Performance Fix**: Replaced manual DiffUtil with AsyncListDiffer
  - **Before**: Manual DiffUtil.calculateDiff() blocked main thread
  - **After**: AsyncListDiffer automatically runs diff on background thread
  - Added coroutine cancellation to prevent concurrent search operations
  - Proper CancellationException handling for cancelled searches
  - **Impact**: Search now smooth and responsive, no system freeze
- **Files**: WordListAdapter.kt (AsyncListDiffer implementation), WordListFragment.kt (coroutine cancellation)

**Previous (v1.32.196)**: Horizontal Scrollable Suggestion Bar

**Horizontal Scrollable Suggestion Bar**
- **Before**: SuggestionBar used LinearLayout with 5 fixed TextViews (predictions cut off)
- **After**: Wrapped in HorizontalScrollView with dynamically created TextViews
- Shows ALL predictions from neural network, not just first 5
- Smooth horizontal scrolling for long prediction lists
- **Files**: keyboard_with_suggestions.xml, SuggestionBar.java, Keyboard2.java

**Previous (v1.32.194)**: Debug Output Fix

**Debug Output Fix - Bracketed Text Only in Logs**
- **Issue**: Predictions showing "indermination [closest:0.84]" in actual UI
- **Fix**: Changed to log debug output only, not add to predictions list
- Top 5 beam search candidates logged with [kept]/[filtered] markers
- Debug output goes to Log.d() and logDebug(), not shown to users
- **Files**: OnnxSwipePredictor.java

**Previous (v1.32.192)**: Swipe Prediction Pipeline Analysis

**Swipe Prediction Pipeline Analysis + Raw/Closest Display**
- **Pipeline Documentation**: Created comprehensive `docs/specs/SWIPE_PREDICTION_PIPELINE.md`
  - Complete end-to-end analysis: Input → Encoder → Beam Search → Vocab Filter → Display
  - Identified 3 issues with prediction transparency
  - Performance breakdown: 30-75ms total (target <100ms ✅)
  - Memory usage: ~15 MB total (acceptable ✅)
  - Test cases for common words, typos, and uncommon terms
  - Recommendations for future improvements
- **Raw/Closest Predictions Display**: Fixed debug mode to always show beam search outputs
  - **Before**: Raw NN outputs only shown when ALL predictions filtered out
  - **After**: Always shows top 3 raw beam search outputs alongside filtered predictions
  - **Markers**: `[raw:X.XX]` for words kept by vocab, `[closest:X.XX]` for words filtered out
  - **Impact**: Users can now see what neural network predicted vs vocabulary filtering
  - **Example**:
    ```
    Filtered predictions: hello (975)
    Raw/Closest: helo [closest:0.92], hello [raw:0.85]
    ```
  - Helps debug "why didn't my swipe predict X?" questions
  - Shows when vocabulary corrects NN typo predictions
  - Reveals when NN predicts uncommon words correctly but vocab filters them
- **Files**: OnnxSwipePredictor.java, docs/specs/SWIPE_PREDICTION_PIPELINE.md

**Previous (v1.32.191)**: Dictionary Manager Bug Fixes

**Dictionary Manager Bug Fixes - Search Performance + UI Fixes**
- **Search Performance**: Fixed search lag by using prefix indexing
  - **Before**: filter() iterated ALL 50k words in memory on main thread (caused lag)
  - **After**: Uses dataSource.searchWords() with O(1) prefix indexing
  - Changed WordListFragment.filter() to call DictionaryDataSource.searchWords()
  - **Impact**: Search is now instant, no lag when typing in search box
- **RecyclerView Position Bug**: Fixed wrong word labels after filtering
  - **Before**: Using stale position parameter caused wrong word labels
  - **After**: Uses holder.bindingAdapterPosition for stable current position
  - Added bounds checking for WordEditableAdapter
  - **Impact**: Word labels now display correctly after search/filter operations
- **Prediction Reload**: Fixed add/delete/edit not updating predictions
  - **Before**: Deleting/adding custom words didn't remove/add them from predictions
  - **After**: All dictionary changes call refreshAllTabs() to reload predictions
  - Added refreshAllTabs() calls to deleteWord(), showAddDialog(), showEditDialog()
  - **Impact**: Custom word changes reflected in typing and swipe predictions instantly
- **Files**: WordListFragment.kt, WordListAdapter.kt

**Previous (v1.32.187)**: Prefix Indexing Implementation - 100x Performance Improvement

**Prefix Indexing Implementation - 100x Performance Improvement**
- **WordPredictor.java**: Implemented prefix indexing for typing predictions
  - Added _prefixIndex HashMap with O(1) lookup
  - buildPrefixIndex() creates 1-3 char prefix mappings during dictionary load
  - getPrefixCandidates() reduces iterations from 50k → 100-500 per keystroke
  - Memory cost: +2 MB (acceptable for 100x speedup)
  - **Impact**: Typing predictions now scale efficiently with 50k vocabulary, no input lag
- **DictionaryDataSource.kt**: Implemented prefix indexing for Dictionary Manager search
  - Added prefixIndex to MainDictionarySource class
  - buildPrefixIndex() creates prefix → words mapping
  - searchWords() uses O(1) lookup instead of O(n) linear search
  - **Impact**: Dictionary Manager search instant for 50k words
- **Kotlin Fix**: Merged two companion objects (TAG + PREFIX_INDEX_MAX_LENGTH)
- **Documentation**: Updated BEAM_SEARCH_VOCABULARY.md v2.0 → v2.1
  - Documented prefix indexing implementation
  - Moved O(n) iteration from Known Issues to Performance Optimizations (✅ FIXED)
  - Updated Future Enhancements with implementation details
  - Added v2.1 changelog with technical analysis

**Previous (v1.32.184)**: 50k Vocabulary Scaling Fixes + Comprehensive Specs

**CRITICAL: 50k Vocabulary Scaling Fixes + Comprehensive Documentation**
- **User Dict CRITICAL Fix**: freq 250 → 9000, tier 1 → tier 2 (was ranked at position 48,736 out of 50k!)
- **Rare Words**: Penalty 0.9x → 0.75x (strengthened for 50k vocab)
- **Common Boost**: 1.2x → 1.3x (increased for 50k vocab)
- **Tier 1 Threshold**: 5000 → 3000 (tightened: 6% of vocab instead of 10%)
- **Performance WARNING**: WordPredictor iterates ALL 50k words on every keystroke (5x slower than 10k)
  - TODO added for prefix indexing implementation (would provide 100x speedup)
- **Documentation**: Created comprehensive `docs/specs/BEAM_SEARCH_VOCABULARY.md`
  - All constants with rationale
  - Memory/performance analysis (7MB, 265-530ms load)
  - Scaling considerations and future enhancements
- **Documentation**: Updated `docs/specs/DICTIONARY_MANAGER.md` with 50k vocabulary details
- **Impact**: User dictionary words now rank correctly, better filtering, comprehensive specs for future scaling

**Previous (v1.32.183)**: Fixed Beam Search Scoring Bug + Hybrid Frequency Model
- **Bug Fixed**: Scoring formula was inverted - rare words scored higher than common words!
- **Root Cause**: `log10(frequency) / -10.0` inverted the 0-1 normalized frequency
- **Fix**: Use frequency directly (already normalized 0-1 by loading code)
- **Hybrid Frequencies**: Custom/user words now use actual frequency values in beam search
  - Custom words: Normalize 1-10000 → 0-1, assign tier 2 if >=8000, else tier 1
  - User dict: Normalize 250 → ~0.025, assign tier 1
  - Previous: All hardcoded to 0.01 with tier 1 (ignored user's frequency choices)
- **Impact**: Common words now rank correctly, custom word frequencies affect swipe predictions
- **Credit**: Gemini-2.5-pro identified the scoring bug during consultation

**Previous (v1.32.182)**: Dictionary Manager UI - Display Raw Frequencies
- **UI**: Dictionary Manager now shows raw frequency values from JSON (128-255)
- **Fixed**: Was showing scaled values (2516 for 'inflicting'), now shows raw (159)
- **Internal**: WordPredictor/OptimizedVocabulary still use scaled values for scoring
- **Consistency**: Main dictionary shows 128-255, custom words use 1-10000 (user-editable range)

**Previous (v1.32.181)**: 50k Enhanced Dictionary - 5x Dictionary Size with Real Frequencies
- **Size**: Upgraded from 10k to 49,981 words
- **Format**: JSON format with actual frequency data (128-255 range)
- **Scaling**: WordPredictor scales to 100-10000, OptimizedVocabulary normalizes to 0-1
- **Tier System**: OptimizedVocabulary assigns tiers by sorted frequency (top 100 = tier 2, top 5000 = tier 1)
- **Fallback**: All three loaders (WordPredictor, OptimizedVocabulary, DictionaryDataSource) support both JSON and text formats
- **Impact**: Better prediction accuracy with real word frequency data, expanded vocabulary coverage

**Previous (v1.32.180)**: Editable Frequency - Full Control Over Word Priority
- **Add Dialog**: Two fields (word + frequency), default 100, range 1-10000
- **Edit Dialog**: Edit both word and frequency, preserves values
- **Validation**: Numeric keyboard, automatic range clamping via coerceIn()
- **UI**: Clean LinearLayout with proper padding and hints
- **Impact**: Frequency affects prediction ranking in both typing and swipe

**Previous (v1.32.178)**: Live Dictionary Reload - Immediate Updates Without Restart
- **Auto-Reload**: Custom/user/disabled words update immediately when changed
- **Typing**: Lazy reload on next prediction (static signal flag, zero overhead)
- **Swipe**: Immediate reload via singleton (one-time cost)
- **Trigger**: Dictionary Manager calls reload after add/delete/toggle
- **Performance**: Only reloads small dynamic sets, not 10k main dictionary
- **UX**: Custom words appear instantly in predictions without keyboard restart

**Previous (v1.32.176)**: Dictionary Integration - Custom/User Words + Disabled Filtering
- **Typing Predictions**: Custom words and user dictionary now included
- **Swipe/Beam Search**: Custom words and user dictionary now included with high priority
- **Disabled Words**: Filtered from BOTH typing and swipe predictions
- **Performance**: Single load during init, cached in memory (O(1) lookups, no I/O overhead)
- **Complete**: All dictionary sources (Main/Custom/User) unified in predictions
- **Complete**: Disabled words excluded from all prediction paths

**Previous (v1.32.174)**: Dictionary Manager - Custom Tab + Crash Fixes
- **Fixed**: Custom tab now shows "+ Add New Word" button (was showing "no words found")
- **Fixed**: getFilteredCount() override in WordEditableAdapter includes add button
- **Fixed**: lateinit crash when toggling words across tabs
- **Functional**: All 4 tabs working - Active (10k words), Disabled, User, Custom
- **Functional**: Add/Edit/Delete custom words via dialogs
- **Stable**: No crashes during word toggling or tab switching

**Previous (v1.32.170)**: Dictionary Manager - Full 10k Dictionary Loading
- **Fixed**: MainDictionarySource now loads full 10,000 words from assets/dictionaries/en_enhanced.txt
- **Fixed**: Parsing changed from tab-separated to word-per-line format
- **Data**: All 10k words displayed with default frequency 100
- **Verified**: Logcat confirms "Loaded 10000 words from main dictionary"
- Complete dictionary viewing: All 10k+ words accessible in Active tab

**Previous (v1.32.167)**: Dictionary Manager - Polished Material3 UI + Functional Integration
- **UI**: Material3.DayNight.NoActionBar theme with clean dark colors
- **UI**: Toolbar widget (no overlap), MaterialSwitch, MaterialButton components
- **UI**: Proper spacing, typography, theme attributes matching CustomCamera quality
- **Functional**: WordPredictor filters disabled words from predictions
- **Functional**: Disabled words persisted in SharedPreferences
- **Functional**: Toggle switches affect actual predictions in keyboard
- **Integration**: setContext() called for all WordPredictor instances
- Complete dictionary control: Active/Disabled/User/Custom word management

**Previous (v1.32.163)**: Dictionary Manager - Crash Fixes
- Fixed Theme.AppCompat crash: Created DictionaryManagerTheme
- Fixed lateinit adapter crash: Added initialization checks
- Activity launches successfully and is fully functional

**Previous (v1.32.160)**: Dictionary Manager - Gemini Code Review Fixes
- Fixed filter dropdown to properly filter by WordSource (not switch tabs)
- Filter now filters within current tab: ALL/MAIN/USER/CUSTOM
- Optimized UserDictionary search to use database-level LIKE filtering (much faster)
- Changed isNotEmpty() to isNotBlank() for word validation (prevents whitespace-only words)

**Previous (v1.32.157)**: Dictionary Manager UI - Initial Implementation
- Modern Material Design dark mode UI with 4 tabs
- Active/Disabled/User/Custom word management
- Real-time search with 300ms debouncing
- Auto-switch tabs when search has no results
- RecyclerView + DiffUtil + ViewPager2 + Fragments
- Kotlin + coroutines
- APK size: 43MB → 47MB (Material Design + Kotlin)
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
