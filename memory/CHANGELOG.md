# Changelog - Unexpected Keyboard

Complete version history with detailed technical documentation.

---

## v1.32.179-184 - Enhanced Dictionary & Frequency Control (2025-10-21)

### 50k Dictionary Upgrade with Real Frequency Data

**v1.32.184** (233) - ✅ BUILD SUCCESSFUL - 50k Vocabulary Scaling Fixes
- fix(beam-search): CRITICAL - user dictionary words ranked at position 48,736
  - Changed user dict frequency: 250 → 9000 (normalized 0.025 → 0.90)
  - Changed user dict tier: 1 → 2 (common boost instead of top5000)
  - User explicitly added these words - should rank in top 10%, not bottom 3%!
- fix(beam-search): strengthen rare words penalty for 50k vocabulary
  - Changed RARE_WORDS_PENALTY: 0.9 → 0.75 (10% → 25% penalty)
  - With 46k tier 0 words, 10% penalty too weak to filter obscure words
- feat(beam-search): increase common words boost for 50k vocabulary
  - Changed COMMON_WORDS_BOOST: 1.2 → 1.3 (20% → 30% boost)
  - Helps common words compete better in larger vocabulary
- feat(beam-search): tighten tier 1 threshold for 50k vocabulary
  - Changed tier 1 threshold: 5000 → 3000 (10% → 6% of vocabulary)
  - Top 10% too broad for 50k words, 6% better represents "common but not top 100"
- perf(typing): add TODO for prefix indexing in WordPredictor
  - WARNING: Iterates ALL 50,131 words on every keystroke (5x slower than 10k)
  - Prefix indexing would reduce iterations from 50k → ~200 (100x speedup)
  - Critical for future scaling, acceptable for now
- docs(specs): create comprehensive BEAM_SEARCH_VOCABULARY.md specification
  - All constants with rationale and scaling analysis
  - Memory usage: 7 MB (acceptable for modern devices)
  - Loading time: 265-530ms (one-time startup cost)
  - Performance targets and known issues
  - Future enhancements (prefix indexing, binary format, adaptive tiers)
- docs(specs): update DICTIONARY_MANAGER.md for 50k vocabulary
  - Updated dictionary size: 10k+ → 50k+
  - Updated MainDictionarySource with JSON format details
  - Updated DictionaryWord frequency ranges by source
  - Added changelog for 50k upgrade
- **Files**: OptimizedVocabulary.java (42-46, 252-258, 573-588), WordPredictor.java (456-459)

**v1.32.183** (232) - ✅ BUILD SUCCESSFUL - CRITICAL: Fixed Beam Search Scoring Bug
- fix(beam-search): correct inverted frequency scoring formula
- **Critical Bug**: Scoring formula `log10(frequency) / -10.0` was inverted
  - Rare words (freq=0.0) scored 1.0, common words (freq=1.0) scored 0.0
  - This contradicted the goal of using frequency to rank predictions
- **Fix**: Use frequency directly (already normalized 0-1 by loading code)
- **New formula**: `(CONFIDENCE_WEIGHT * confidence + FREQUENCY_WEIGHT * frequency) * boost`
- feat(beam-search): implement hybrid frequency model for custom/user words
  - Custom words: Normalize freq (1-10000) → 0-1, dynamic tier (>=8000 = tier 2, else tier 1)
  - User dict: Normalize freq (250) → ~0.025, tier 1
  - Previous: All custom/user words hardcoded to freq=0.01, tier=1 (ignored user settings)
- **Impact**: Common words now rank correctly, custom frequencies affect swipe predictions
- **Credit**: Gemini-2.5-pro identified scoring bug during consultation
- **Files**: OptimizedVocabulary.java lines 186-194 (scoring), 512-602 (hybrid frequencies)

**v1.32.182** (231) - ✅ BUILD SUCCESSFUL - Display Raw Frequencies in UI
- fix(dictionary): show raw frequency values in Dictionary Manager
- Dictionary Manager now displays raw JSON frequencies (128-255 range)
- Fixed: was showing scaled values (e.g., 2516 for 'inflicting' instead of raw 159)
- Internal scoring still uses scaled values (WordPredictor: 100-10000, OptimizedVocabulary: 0-1)
- Main dictionary shows 128-255, custom words use 1-10000 (user-editable range)

**v1.32.181** (230) - 50k Dictionary with Real Frequencies
- feat(dictionary): upgrade from 10k to 50k words with actual frequency data
- Format: JSON format `{"word": freq, ...}` with 49,981 words
- Frequency range: 128-255 raw values from source data
- **WordPredictor.java**: Loads JSON, scales frequencies 128-255 → 100-10000 for scoring
- **OptimizedVocabulary.java**: Two-pass loading (collect, sort by freq, assign tiers)
  - Tier assignment based on sorted position: top 100 = tier 2, top 5000 = tier 1, rest = tier 0
  - Normalizes frequencies to 0-1 range for beam search
- **DictionaryDataSource.kt**: Displays frequencies in Dictionary Manager UI (100-10000 range)
- All three loaders support JSON format with fallback to text format
- APK size: 47MB → 48MB (+789K dictionary file)
- Impact: Better prediction accuracy with real word frequency data, 5x vocabulary coverage

**v1.32.180** (229) - Editable Word Frequency in Custom Tab
- feat(dictionary): add editable frequency fields to custom word dialogs
- Add dialog: Two fields (word + frequency), default 100, range 1-10000
- Edit dialog: Both word and frequency editable, preserves existing values
- Validation: Numeric keyboard for frequency input, automatic range clamping via coerceIn()
- UI: Clean LinearLayout with proper padding and hints
- Impact: Frequency directly affects prediction ranking in both typing and swipe

**v1.32.179** - (skipped, version auto-increment)

---

## v1.32.157-178 - Dictionary Manager (2025-10-21)

### Complete Dictionary Management System

**v1.32.178** (227) - ✅ BUILD SUCCESSFUL - Live Dictionary Reload
- feat(predictions): add auto-reload for dictionary changes
- Custom/user/disabled words update immediately when changed
- Typing: Lazy reload on next prediction (static signal flag, zero overhead)
- Swipe: Immediate reload via singleton (one-time cost)
- Performance: Only reloads small dynamic sets, not 10k main dictionary
- UX: Custom words appear instantly in predictions without keyboard restart

**v1.32.176** (225) - Full Dictionary Integration
- feat(predictions): integrate custom/user dict + filter disabled words
- Typing predictions now include custom words and user dictionary
- Swipe beam search now includes custom words and user dictionary (high priority)
- Disabled words filtered from BOTH typing and swipe predictions
- Performance: Single load during init, cached in memory (O(1) lookups, no I/O)

**v1.32.174** (223) - Custom Tab + Crash Fixes
- fix(dictionary): show Custom tab + Add button, fix lateinit crash
- Custom tab now shows "+ Add New Word" button (was showing "no words found")
- Fixed getFilteredCount() override in WordEditableAdapter includes add button
- Fixed lateinit crash when toggling words across tabs
- All 4 tabs fully functional

**v1.32.170** (219) - Full 10k Dictionary Loading
- fix(dictionary): load full 10k word dictionary in Manager
- Fixed MainDictionarySource parsing to handle word-per-line format
- Dictionary Manager now displays all 10,000 words from assets

**v1.32.167** (216) - Polished Material3 UI + Functional Integration
- Material3.DayNight.NoActionBar theme with clean dark colors
- Toolbar widget (no overlap), MaterialSwitch, MaterialButton components
- WordPredictor filters disabled words from predictions
- Disabled words persisted in SharedPreferences
- Toggle switches affect actual predictions in keyboard
- setContext() called for all WordPredictor instances

**v1.32.163** (212) - Dictionary Manager Crash Fixes
- Fixed Theme.AppCompat crash: Created DictionaryManagerTheme
- Fixed lateinit adapter crash: Added initialization checks
- Activity launches successfully and is fully functional

**v1.32.160** (209) - Gemini Code Review Fixes
- Fixed filter dropdown to properly filter by WordSource (not switch tabs)
- Filter now filters within current tab: ALL/MAIN/USER/CUSTOM
- Optimized UserDictionary search to use database-level LIKE filtering
- Changed isNotEmpty() to isNotBlank() for word validation

**v1.32.157** (206) - Initial Implementation
- Modern Material Design dark mode UI with 4 tabs
- Active/Disabled/User/Custom word management
- Real-time search with 300ms debouncing
- Auto-switch tabs when search has no results
- RecyclerView + DiffUtil + ViewPager2 + Fragments
- Kotlin + coroutines
- APK size: 43MB → 47MB (Material Design + Kotlin)
- Access via Settings → "📚 Dictionary Manager"

**Architecture**:
- `DictionaryDataSource.kt` - Interface for all dictionary sources
- `MainDictionarySource` - Loads 10k words from assets
- `DisabledDictionarySource` - Manages disabled word list via SharedPreferences
- `UserDictionarySource` - Android UserDictionary ContentProvider integration
- `CustomDictionarySource` - App-specific custom words via SharedPreferences JSON
- `DictionaryWord.kt` - Data class with word/frequency/source/enabled
- `WordListAdapter.kt` - RecyclerView adapters (toggle/editable variants)
- `WordListFragment.kt` - Fragment for each tab with coroutines
- `DictionaryManagerActivity.kt` - Main activity with ViewPager2

**Integration**:
- `WordPredictor.java` - Loads custom/user words during init, filters disabled
- `OptimizedVocabulary.java` - Loads custom/user words into beam search, filters disabled
- `OnnxSwipePredictor.java` - Exposes reloadVocabulary() for live updates

**Commits**:
- 552c4c5d - docs(pm): update to v1.32.178 with live dictionary reload
- f998fd9f - feat(predictions): add auto-reload for dictionary changes
- 0d9db2e6 - docs(pm): update to v1.32.176 with full dictionary integration
- a5727918 - feat(predictions): integrate custom/user dict + filter disabled words
- 7cda6dd3 - docs(pm): update to v1.32.174 with Custom tab fixes
- cdef9137 - fix(dictionary): show Custom tab + Add button, fix lateinit crash
- 516a5030 - docs(pm): update to v1.32.170 with full dictionary loading
- f09af62e - fix(dictionary): load full 10k word dictionary in Manager
- [... see git log for complete history]

---

## v1.32.114-121 - Auto-Correction Feature (2025-10-19)

### Auto-Correction Implementation & Fixes

**v1.32.121** (170) - ✅ BUILD SUCCESSFUL
- fix(autocorrect): detect Termux app automatically via EditorInfo.packageName
- Auto-correction now only removes trailing space in actual Termux app
- In other apps: adds space for normal typing flow
- Ignores global termux_mode setting to avoid breaking normal apps

**v1.32.120** (169)
- fix(autocorrect): respect Termux mode spacing behavior

**v1.32.119** (168)
- fix(autocorrect): delete both word and space before replacing
- Fixed deletion count: word.length() + 1 (includes trailing space)

**v1.32.116** (165)
- fix(autocorrect): insert corrected word with trailing space
- Fixed "thid" → "tthis" bug

**v1.32.114** (163)
- feat(autocorrect): implement auto-correction with fuzzy matching
- Algorithm: same length + first 2 letters + positional char match ≥ 0.67
- Preserves capitalization (teh→the, Teh→The, TEH→THE)
- 4 configurable settings via UI

**Commits**:
- 0242b08d - fix(autocorrect): detect Termux app automatically
- 082a0706 - fix(autocorrect): respect Termux mode spacing
- 053039ec - fix(autocorrect): delete both word and space
- 3cd702a3 - fix(autocorrect): insert with trailing space
- 0028e3cd - feat(autocorrect): implement auto-correction

---

## v1.32.99-113 - WordPredictor Refactor (2025-10-19)

### Swipe Fallback Removal & Unified Scoring

**v1.32.102** (162)
- feat(settings): add UI controls for prediction weights with persistence
- Removed 8 deprecated "Advanced Word Prediction" settings
- Added 2 functional weights: context_boost, frequency_scale

**v1.32.101** (161)
- feat(predict): implement unified scoring with early fusion context
- Context now applied to ALL candidates before selecting top N
- Formula: `score = prefix × adaptation × (1 + (contextMult - 1) × boost) × log(freq / scale)`

**v1.32.100** (160)
- refactor(config): completely remove deprecated settings (no backwards compat)

**v1.32.99** (159-160)
- refactor(predict): remove WordPredictor swipe fallback system
- Deleted 8 methods (~200 lines): calculateSwipeScore, calculateEditDistance, etc.
- Simplified two-list to single-list architecture
- refactor(config): remove deprecated endpoint weight fields

**Commits**:
- 33a043d8 - refactor(settings): remove 8 deprecated settings
- 31c6fefd - feat(settings): add UI controls for weights
- c81cc537 - feat(predict): unified scoring with early fusion
- 98ca1ca6 - refactor(config): remove deprecated settings
- e496ab75 - refactor(config): remove endpoint weight fields
- 69b85256 - refactor(predict): single-list architecture
- 4452162e - refactor(predict): remove swipe fallback

---

## v1.32.98-102 - Four Critical Prediction Bugs Fixed (2025-10-19)

**v1.32.102** (151)
- fix(swipe): fix Termux mode swipe replacement leaves space
- Always delete word + trailing space for swipe auto-insertions
- Keyboard2.java:996 - Changed to `deleteCount = word.length() + 1`

**v1.32.101** (150)
- fix(typing): type-then-swipe duplicates manual word
- Only commit space, not whole word (text already in editor)
- Keyboard2.java:864 - Changed to `ic.commitText(" ", 1)`

**v1.32.100** (149)
- fix(typing): typing completion deletes all text since last prediction
- Delete typed prefix when selecting prediction
- Keyboard2.java:1025-1034 - Added deletion of _currentWord.length()

**v1.32.99** (148)
- fix(swipe): swipe replacement leaves first character
- Strip debug annotations before storing in _lastAutoInsertedWord
- Keyboard2.java:884 - Added `.replaceAll(" \\[raw:[0-9.]+\\]$", "")`

**Core Understanding**: Manual typing commits characters immediately via `KeyEventHandler.send_text()`, while `_currentWord` is just a tracking buffer. Text is already in the editor!

**Commit**: a0dcfee7 - fix(swipe): fix multiple prediction selection bugs

---

## v1.32.96-97 - Three Major Improvements (2025-10-17)

**v1.32.97** (146)
- feat(ui): add default IME selection prompt
- docs(prediction): comprehensive Advanced Prediction Settings documentation
- Documented that 8 legacy settings are NO LONGER USED by neural network

**v1.32.96** (145)
- fix(swipe): double character bug when swiping after typing
- Only delete when replacing auto-inserted predictions, not typed text
- Fixed: Type "i" → swipe "think" → result "i think" (was " think")

---

## v1.32.94-95 - Clipboard Security & WordPredictor Docs (2025-10-17)

**v1.32.95** (144)
- docs(predict): add comprehensive WordPredictor documentation
- Documented dual prediction system (Neural + WordPredictor)

**v1.32.94** (143)
- fix(security): prevent clipboard history from accepting empty/whitespace
- Mitigates clipboard junk accumulation attack vector

---

## Earlier Versions

See git log for complete history:
```bash
git log --oneline --decorate
```

For detailed technical documentation of any version, see commit messages and file-specific documentation.
