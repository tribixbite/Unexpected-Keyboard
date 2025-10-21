# Changelog - Unexpected Keyboard

Complete version history with detailed technical documentation.

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
