# Dictionary Manager Specification

**Version**: 1.0
**Status**: Implemented
**Platform**: Android API 21+
**Implementation**: Kotlin with Material Design

---

## Table of Contents

1. [Overview](#overview)
2. [User Requirements](#user-requirements)
3. [Architecture](#architecture)
4. [UI Components](#ui-components)
5. [Data Sources](#data-sources)
6. [User Workflows](#user-workflows)
7. [Technical Implementation](#technical-implementation)
8. [Performance Requirements](#performance-requirements)
9. [Error Handling](#error-handling)
10. [Testing](#testing)

---

## Overview

The Dictionary Manager provides a comprehensive UI for managing dictionary words used in prediction and auto-correction features. Users can view, enable/disable, add, edit, and delete words across four different dictionary sources.

### Goals

- Provide fine-grained control over dictionary words
- Support multiple dictionary sources (main, user, custom)
- Enable users to disable problematic words without deleting them
- Allow custom word additions with frequency control
- Integrate with Android's system UserDictionary
- Maintain high performance with large dictionaries (50k+ words)

### Non-Goals

- Editing main BigramModel dictionary words
- Modifying word frequencies for main dictionary
- Bulk import/export of custom dictionaries
- Spell checking or word validation

---

## User Requirements

### Functional Requirements

1. **FR-1**: User must be able to view all words from the main dictionary
2. **FR-2**: User must be able to disable/enable main dictionary words
3. **FR-3**: User must be able to view all disabled words
4. **FR-4**: User must be able to re-enable disabled words
5. **FR-5**: User must be able to view Android UserDictionary words
6. **FR-6**: User must be able to manage Android UserDictionary (add/delete)
7. **FR-7**: User must be able to add custom words with frequency
8. **FR-8**: User must be able to edit custom words
9. **FR-9**: User must be able to delete custom words
10. **FR-10**: User must be able to search all words in real-time
11. **FR-11**: User must be able to filter words by source (MAIN/USER/CUSTOM)
12. **FR-12**: UI must auto-switch tabs when current tab has no search results

### Non-Functional Requirements

1. **NFR-1**: Search must debounce at 300ms to prevent excessive filtering
2. **NFR-2**: UserDictionary search must use database-level filtering for performance
3. **NFR-3**: UI must use RecyclerView with DiffUtil for efficient updates
4. **NFR-4**: Dark mode UI must be touch-friendly and mobile-optimized
5. **NFR-5**: Must work on Android API 21+ (Android 5.0 Lollipop)
6. **NFR-6**: Must not require external permissions (except READ_USER_DICTIONARY)

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              DictionaryManagerActivity                       │
│  - Search Input (debounced 300ms)                           │
│  - Filter Spinner (ALL/MAIN/USER/CUSTOM)                    │
│  - Reset Button                                              │
│  - TabLayout (4 tabs)                                        │
│  - ViewPager2                                                │
└──────────────┬──────────────────────────────────────────────┘
               │
               ├── WordListFragment (Active)
               │   └── WordToggleAdapter
               │       └── MainDictionarySource + DisabledDictionarySource
               │
               ├── WordListFragment (Disabled)
               │   └── WordToggleAdapter
               │       └── DisabledDictionarySource
               │
               ├── WordListFragment (User Dict)
               │   └── WordToggleAdapter
               │       └── UserDictionarySource
               │
               └── WordListFragment (Custom)
                   └── WordEditableAdapter
                       └── CustomDictionarySource
```

### Data Flow

```
User Input (Search/Filter)
    ↓
DictionaryManagerActivity (debounce + state)
    ↓
performSearch(query, sourceFilter)
    ↓
WordListFragment.filter(query, sourceFilter)
    ↓
BaseWordAdapter.filter(query, sourceFilter)
    ↓
DiffUtil.calculateDiff()
    ↓
RecyclerView updates
```

---

## UI Components

### Main Activity Layout

**File**: `res/layout/activity_dictionary_manager.xml`

```xml
┌──────────────────────────────────────────────────────┐
│  [Search Input] [Filter ▼] [Reset]                  │
├──────────────────────────────────────────────────────┤
│  [ Active ] [ Disabled ] [ User Dict ] [ Custom ]   │
├──────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────┐ │
│  │ word1                           [Toggle]       │ │
│  │ word2                           [Toggle]       │ │
│  │ word3                           [Toggle]       │ │
│  │ ...                                            │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Components**:
- `EditText search_input`: Real-time search with TextWatcher
- `Spinner filter_spinner`: WordSource filter (ALL/MAIN/USER/CUSTOM)
- `Button reset_button`: Clears search and filter
- `TabLayout tab_layout`: 4 tabs navigation
- `ViewPager2 view_pager`: Fragment container

### Fragment Layout

**File**: `res/layout/fragment_word_list.xml`

**Components**:
- `RecyclerView recycler_view`: Main word list
- `TextView empty_text`: "No words found" message
- `ProgressBar loading_progress`: Loading indicator

### List Item Layouts

#### Toggle Item (Active/Disabled/User)

**File**: `res/layout/item_word_toggle.xml`

```
┌─────────────────────────────────────────────┐
│ word_text (bold, 16sp)        [○ Toggle]   │
│ frequency_text (gray, 12sp)                 │
└─────────────────────────────────────────────┘
```

#### Editable Item (Custom)

**File**: `res/layout/item_word_editable.xml`

```
┌─────────────────────────────────────────────────┐
│ word_text (bold, 16sp)     [Edit] [Del]        │
│ frequency_text (gray, 12sp)                     │
└─────────────────────────────────────────────────┘
```

**Special**: First item is "+ Add New Word" with hidden buttons

---

## Data Sources

### 1. MainDictionarySource

**Purpose**: Read-only access to main dictionary (50k words with real frequencies)

**Data Source**: `assets/dictionaries/en_enhanced.json`

**Format**: JSON object
```json
{"the": 255, "of": 254, "to": 254, "and": 254, ...}
```

**Statistics**:
- **Word Count**: 49,981
- **File Size**: 789 KB
- **Frequency Range**: 128-255 (raw values displayed in UI)

**Operations**:
- `getAllWords()`: Returns List<DictionaryWord> with WordSource.MAIN
- `searchWords(query)`: Filters getAllWords() by query (in-memory)
- `toggleWord(word, enabled)`: Delegates to DisabledDictionarySource

**Notes**:
- Words are marked disabled based on DisabledDictionarySource
- Frequencies displayed as-is (128-255) in Dictionary Manager UI
- Internal prediction engines scale frequencies for scoring:
  - WordPredictor: 128-255 → 100-10000
  - OptimizedVocabulary: 128-255 → 0.0-1.0

### 2. DisabledDictionarySource

**Purpose**: Manage list of disabled words

**Data Source**: `SharedPreferences` key: `"disabled_words"` (StringSet)

**Operations**:
- `getAllWords()`: Returns disabled words as List<DictionaryWord>
- `searchWords(query)`: Filters disabled words by query
- `toggleWord(word, enabled)`: Adds/removes from StringSet
- `setWordEnabled(word, enabled)`: Helper method

**Storage Format**:
```json
{
  "disabled_words": ["word1", "word2", "word3"]
}
```

### 3. UserDictionarySource

**Purpose**: Access Android system UserDictionary

**Data Source**: `UserDictionary.Words` ContentProvider

**Operations**:
- `getAllWords()`: Query ContentProvider with null selection
- `searchWords(query)`: Query with `WORD LIKE ?` for performance
- `addWord(word, frequency)`: Use `UserDictionary.Words.addWord()`
- `deleteWord(word)`: Delete via ContentProvider
- `updateWord(old, new, freq)`: Delete + Add

**Performance**:
- Search uses database-level filtering: `WHERE WORD LIKE '%query%'`
- Avoids loading entire dictionary into memory

**Permissions**:
- Requires `READ_USER_DICTIONARY` in AndroidManifest.xml

### 4. CustomDictionarySource

**Purpose**: App-specific custom words

**Data Source**: `SharedPreferences` key: `"custom_words"` (JSON)

**Operations**:
- `getAllWords()`: Parse JSON to Map<String, Int>
- `searchWords(query)`: Filter custom words by query
- `addWord(word, frequency)`: Add to map, save JSON
- `deleteWord(word)`: Remove from map, save JSON
- `updateWord(old, new, freq)`: Remove old, add new, save JSON

**Storage Format**:
```json
{
  "custom_words": "{\"word1\":100,\"word2\":200,\"word3\":50}"
}
```

---

## User Workflows

### Workflow 1: Search Words

1. User types in search box
2. TextWatcher triggers after 300ms debounce
3. DictionaryManagerActivity calls performSearch(query)
4. All 4 fragments filter their lists
5. If current tab has 0 results, auto-switch to first tab with results
6. RecyclerView updates via DiffUtil

### Workflow 2: Filter by Source

1. User selects filter from Spinner (MAIN/USER/CUSTOM)
2. onItemSelected() calls applyFilter(filterType)
3. performSearch() maps FilterType → WordSource
4. All fragments filter by both query AND source
5. RecyclerView updates

### Workflow 3: Disable Main Dictionary Word

1. User navigates to Active tab
2. User toggles switch for word
3. WordToggleAdapter calls onToggle(word, false)
4. WordListFragment calls dataSource.toggleWord(word, false)
5. DisabledDictionarySource adds word to StringSet
6. Fragment calls loadWords() to refresh
7. Activity calls refreshAllTabs() to update other tabs

### Workflow 4: Add Custom Word

1. User navigates to Custom tab
2. User taps "+ Add New Word" item
3. Dialog shows with EditText
4. User enters word and taps "Add"
5. Validation: word.isNotBlank()
6. CustomDictionarySource.addWord(word, 100)
7. Fragment calls loadWords() to refresh

### Workflow 5: Edit Custom Word

1. User navigates to Custom tab
2. User taps "Edit" button on word
3. Dialog shows with EditText pre-filled
4. User modifies word and taps "Save"
5. Validation: newWord.isNotBlank() && newWord != oldWord
6. CustomDictionarySource.updateWord(old, new, freq)
7. Fragment calls loadWords() to refresh

### Workflow 6: Delete Custom Word

1. User navigates to Custom tab
2. User taps "Del" button on word
3. Confirmation dialog shows
4. User taps "Delete"
5. CustomDictionarySource.deleteWord(word)
6. Fragment calls loadWords() to refresh

---

## Technical Implementation

### Data Models

#### DictionaryWord

```kotlin
data class DictionaryWord(
    val word: String,           // The word text
    val frequency: Int = 0,     // Frequency range varies by source
    val source: WordSource,     // MAIN/USER/CUSTOM
    val enabled: Boolean = true // Disabled state
) : Comparable<DictionaryWord>
```

**Frequency Ranges by Source**:
- **MAIN**: 128-255 (raw values from JSON dictionary)
- **USER**: Typically 250 (Android UserDictionary default)
- **CUSTOM**: 1-10000 (user-editable, default 100)

**Sorting**: By frequency descending, then alphabetically

#### WordSource

```kotlin
enum class WordSource {
    MAIN,   // BigramModel dictionary
    USER,   // Android UserDictionary
    CUSTOM  // App-specific custom words
}
```

#### FilterType

```kotlin
enum class FilterType {
    ALL,    // Show all sources
    MAIN,   // Show only MAIN source
    USER,   // Show only USER source
    CUSTOM  // Show only CUSTOM source
}
```

### Adapters

#### BaseWordAdapter

```kotlin
abstract class BaseWordAdapter : RecyclerView.Adapter<RecyclerView.ViewHolder>() {
    protected var allWords: List<DictionaryWord> = emptyList()
    protected var filteredWords: List<DictionaryWord> = emptyList()

    fun setWords(words: List<DictionaryWord>)
    fun filter(query: String, sourceFilter: WordSource? = null)
    fun getFilteredCount(): Int
}
```

**Filtering Logic**:
```kotlin
filteredWords = allWords.filter { word ->
    val matchesQuery = if (query.isBlank()) true
        else word.word.contains(query, ignoreCase = true)

    val matchesSource = if (sourceFilter == null) true
        else word.source == sourceFilter

    matchesQuery && matchesSource
}
```

**DiffUtil**: Used for efficient RecyclerView updates

#### WordToggleAdapter

Extends BaseWordAdapter

**ViewHolder**: ToggleViewHolder
- `TextView word_text`
- `TextView frequency_text`
- `SwitchCompat enable_toggle`

**Callback**: `onToggle: (DictionaryWord, Boolean) -> Unit`

#### WordEditableAdapter

Extends BaseWordAdapter

**ViewHolders**:
1. AddViewHolder (position 0): "+ Add New Word"
2. EditableViewHolder (position 1+): Word with Edit/Del buttons

**Callbacks**:
- `onEdit: (DictionaryWord) -> Unit`
- `onDelete: (DictionaryWord) -> Unit`
- `onAdd: () -> Unit`

### Fragment Lifecycle

```kotlin
onCreate() → initializeDataSource()
onViewCreated() → setupAdapter() → loadWords()
loadWords() → lifecycleScope.launch → dataSource.getAllWords()
filter() → adapter.filter() → updateEmptyState()
```

### Activity State Management

```kotlin
private var currentSearchQuery = ""
private var currentFilter: FilterType = FilterType.ALL
```

**Issue**: State lost on configuration changes (rotation)
**Future**: Use ViewModel for state persistence

---

## Performance Requirements

### Search Performance

- **Debounce**: 300ms to prevent excessive filtering
- **UserDictionary**: Database-level filtering (`LIKE` clause)
- **MainDictionary**: In-memory filtering (BigramModel has ~2000 words)
- **Custom**: In-memory filtering (typically < 100 words)

### Memory Usage

- **RecyclerView**: Only visible items in memory
- **DiffUtil**: Calculates minimal updates
- **Coroutines**: Background loading on Dispatchers.IO

### UI Responsiveness

- **Target**: < 100ms for filter updates
- **Loading indicators**: Show during initial data load
- **Empty states**: Show when filtered results = 0

---

## Error Handling

### UserDictionary Permission Denied

**Scenario**: App not set as default IME

**Current**: Shows empty list

**Future**: Show message "Set as default keyboard to access UserDictionary"

### Add Word Failure

**Scenario**: UserDictionary.Words.addWord() throws exception

**Handling**: Show AlertDialog with error message

### Delete Word Failure

**Scenario**: ContentResolver.delete() throws exception

**Handling**: Show AlertDialog with error message

### Edit Word Failure

**Scenario**: updateWord() throws exception

**Handling**: Show AlertDialog with error message

### Empty Search Results

**Scenario**: Search/filter returns 0 results

**Handling**:
1. Show "No words found" TextView
2. If query not empty, auto-switch to first tab with results

---

## Testing

### Unit Tests

Not implemented - Kotlin code not yet covered by unit tests

### Manual Test Cases

#### TC-1: View Active Words

**Steps**:
1. Open Settings → Dictionary Manager
2. Verify Active tab selected by default
3. Verify words from BigramModel displayed
4. Verify frequency shown for each word

**Expected**: List of words with frequencies, sorted by frequency

#### TC-2: Search Words

**Steps**:
1. Open Dictionary Manager
2. Type "the" in search box
3. Wait 300ms
4. Verify results filtered

**Expected**: Only words containing "the" (case-insensitive)

#### TC-3: Filter by Source

**Steps**:
1. Open Dictionary Manager
2. Select "CUSTOM" from filter dropdown
3. Verify only custom words shown across all tabs

**Expected**: Only words with WordSource.CUSTOM visible

#### TC-4: Disable Word

**Steps**:
1. Navigate to Active tab
2. Toggle off switch for "the"
3. Navigate to Disabled tab
4. Verify "the" appears

**Expected**: Word moved from Active to Disabled

#### TC-5: Add Custom Word

**Steps**:
1. Navigate to Custom tab
2. Tap "+ Add New Word"
3. Enter "test123"
4. Tap "Add"

**Expected**: "test123" appears in list with frequency 100

#### TC-6: Auto-Switch Tabs

**Steps**:
1. Open Dictionary Manager (Active tab)
2. Select "CUSTOM" filter
3. Verify tab switches to Custom

**Expected**: Automatically switches to first tab with results

#### TC-7: Reset Search/Filter

**Steps**:
1. Enter search query "abc"
2. Select "USER" filter
3. Tap "Reset" button

**Expected**: Search cleared, filter = ALL, all words shown

---

## Known Issues

### 1. State Loss on Configuration Change

**Severity**: Medium

**Description**: Search query and filter selection lost on screen rotation

**Workaround**: User must re-enter search/filter

**Fix**: Implement ViewModel for state persistence

### 2. Full List Reload on Changes

**Severity**: Low

**Description**: Toggle/add/delete triggers full data reload instead of incremental update

**Impact**: UI flickers briefly on data changes

**Fix**: Update adapter in-memory instead of reloading from data source

### 3. No Permission Error Message

**Severity**: Low

**Description**: UserDictionary tab shows empty if app not default IME

**Workaround**: User must set app as default keyboard

**Fix**: Show explanatory message instead of empty list

---

## Future Enhancements

### 1. ViewModel Architecture

- Centralized DataSource instances
- State persistence across config changes
- LiveData/StateFlow for reactive updates

### 2. Bulk Operations

- Select multiple words for bulk disable/delete
- Import/export custom dictionary CSV

### 3. Word Statistics

- Show usage frequency for custom words
- Show last used timestamp
- Sort by most recently used

### 4. Advanced Filtering

- Filter by frequency range
- Filter by word length
- Combine multiple filters (AND/OR logic)

### 5. Undo/Redo

- Undo accidental deletions
- Redo operations
- Operation history

---

## File Structure

```
srcs/juloo.keyboard2/
├── DictionaryWord.kt              # Data model
├── DictionaryDataSource.kt        # Interface + 4 implementations
├── WordListAdapter.kt             # BaseWordAdapter + 2 subclasses
├── WordListFragment.kt            # Fragment for each tab
└── DictionaryManagerActivity.kt   # Main activity

res/layout/
├── activity_dictionary_manager.xml  # Main activity layout
├── fragment_word_list.xml           # Fragment layout
├── item_word_toggle.xml             # Toggle item layout
└── item_word_editable.xml           # Editable item layout

res/values/
└── styles.xml                       # ToggleSwitchStyle

res/xml/
└── settings.xml                     # Dictionary Manager preference

AndroidManifest.xml                  # Activity + permission
```

---

## Integration Points

### BigramModel

**Methods Used**:
- `BigramModel.getInstance(Context)`: Get singleton instance
- `BigramModel.getAllWords()`: Get all words from current language
- `BigramModel.getWordFrequency(String)`: Get frequency (0-1000)

**Files Modified**:
- `srcs/juloo.keyboard2/BigramModel.java`: Added getAllWords() and getWordFrequency() methods

### SharedPreferences

**Keys**:
- `disabled_words`: StringSet of disabled word strings
- `custom_words`: JSON string of {word: frequency} map

**Access**:
- `DirectBootAwarePreferences.get_shared_preferences(Context)`

### UserDictionary

**ContentProvider**:
- URI: `UserDictionary.Words.CONTENT_URI`
- Columns: `WORD`, `FREQUENCY`

**API**:
- `UserDictionary.Words.addWord(context, word, frequency, null, null)`
- `contentResolver.delete(URI, selection, args)`

---

## Dependencies

### Kotlin
- `kotlin-stdlib:1.9.20`
- `kotlinx-coroutines-android:1.7.3`

### Material Design
- `material:1.11.0`
- `recyclerview:1.3.2`
- `viewpager2:1.0.0`
- `appcompat:1.6.1`
- `constraintlayout:2.1.4`

### Build
- Kotlin plugin: `org.jetbrains.kotlin.android:1.9.20`
- ViewBinding enabled

---

## Changelog

### v1.32.181-184 (2025-10-21)
- **MAJOR**: Upgraded main dictionary from 10k to 50k words with real frequencies
- Dictionary source changed from BigramModel to JSON (assets/dictionaries/en_enhanced.json)
- Frequency range changed: probability*1000 (0-1000) → raw values (128-255)
- Added editable frequency for custom words (default 100, range 1-10000)
- Display raw frequency values in UI (128-255 for main, 1-10000 for custom)
- Internal scoring engines scale frequencies appropriately:
  - WordPredictor: 128-255 → 100-10000
  - OptimizedVocabulary: 128-255 → 0.0-1.0
- Updated spec to reflect 50k vocabulary scaling

### v1.32.157 (2025-10-20)
- Initial implementation
- 4 tabs: Active, Disabled, User Dict, Custom
- Search with debouncing
- Filter dropdown
- Auto-tab-switching

### v1.32.160 (2025-10-20)
- Fixed filter dropdown to filter by source (not switch tabs)
- Optimized UserDictionary search with database LIKE filtering
- Changed isNotEmpty() to isNotBlank() for validation
- Gemini code review improvements

---

## References

- [Android UserDictionary API](https://developer.android.com/reference/android/provider/UserDictionary.Words)
- [Material Design Components](https://material.io/components)
- [RecyclerView DiffUtil](https://developer.android.com/reference/androidx/recyclerview/widget/DiffUtil)
- [ViewPager2](https://developer.android.com/reference/androidx/viewpager2/widget/ViewPager2)
