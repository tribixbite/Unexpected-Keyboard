# Clipboard Manager Specification

**Version**: 1.0
**Status**: Implemented
**Platform**: Android API 21+
**Implementation**: Java with SQLite

---

## Table of Contents

1. [Overview](#overview)
2. [User Requirements](#user-requirements)
3. [Architecture](#architecture)
4. [UI Components](#ui-components)
5. [Data Storage](#data-storage)
6. [User Workflows](#user-workflows)
7. [Technical Implementation](#technical-implementation)
8. [Import/Export](#importexport)
9. [Performance Requirements](#performance-requirements)
10. [Error Handling](#error-handling)
11. [Known Issues & Future Enhancements](#known-issues--future-enhancements)

---

## Overview

The Clipboard Manager provides persistent clipboard history with pinning, search, and import/export capabilities. Users can view clipboard history, pin important items, expand multi-line entries, and backup/restore all clipboard data.

### Goals

- Provide persistent clipboard history across app restarts
- Allow pinning of frequently used clipboard entries
- Support search and filtering of clipboard history
- Enable import/export for backup and device migration
- Maintain high performance with large clipboard histories
- Provide expand/collapse for multi-line entries

### Non-Goals

- Cloud synchronization of clipboard data
- OCR or image clipboard support
- Cross-device clipboard sharing
- Real-time clipboard syncing with other apps

---

## User Requirements

### Functional Requirements

1. **FR-1**: User must be able to view clipboard history (active entries)
2. **FR-2**: User must be able to pin clipboard entries for permanent storage
3. **FR-3**: User must be able to view all pinned entries
4. **FR-4**: User must be able to paste any entry from history or pinned
5. **FR-5**: User must be able to delete pinned entries entirely (v1.32.309)
6. **FR-6**: User must be able to search clipboard history (implemented)
7. **FR-7**: User must be able to expand/collapse multi-line entries (v1.32.308)
8. **FR-8**: User must be able to export all clipboard data to JSON (v1.32.306)
9. **FR-9**: User must be able to import clipboard data from JSON with duplicate prevention (v1.32.306)
10. **FR-10**: Clipboard entries must expire after configurable period (default 7 days)
11. **FR-11**: Pinned entries must never expire
12. **FR-12**: User must be able to configure maximum size per clipboard item to prevent system overload (v1.32.327)
13. **FR-13**: User must be able to view clipboard storage statistics (entry count and size) (v1.32.329)
14. **FR-14**: User must be able to choose between count-based and size-based clipboard history limits (v1.32.330)

### Non-Functional Requirements

1. **NFR-1**: Clipboard history must survive app restarts (SQLite persistent storage)
2. **NFR-2**: Search must use database-level filtering for performance
3. **NFR-3**: UI must be responsive with 1000+ clipboard entries
4. **NFR-4**: Duplicate consecutive entries must be prevented
5. **NFR-5**: Must work without external storage permissions (uses SAF for import/export)
6. **NFR-6**: Multi-line entries must collapse by default to save screen space

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              ClipboardPane (Main UI)                         │
│  - Search Input                                              │
│  - Clipboard History Section                                 │
│  - Pinned Section                                             │
└──────────────┬──────────────────────────────────────────────┘
               │
               ├── ClipboardHistoryView (Active entries)
               │   └── ClipboardEntriesAdapter
               │       └── ClipboardHistoryService
               │           └── ClipboardDatabase (SQLite)
               │
               └── ClipboardPinView (Pinned entries)
                   └── ClipboardPinEntriesAdapter
                       └── ClipboardHistoryService
                           └── ClipboardDatabase (SQLite)
```

### Data Flow

```
User Copies Text
    ↓
System ClipboardManager
    ↓
ClipboardHistoryService.onClipboardChange()
    ↓
Check if duplicate (consecutive entries)
    ↓
ClipboardDatabase.addClipboardEntry(content, expiryTimestamp)
    ↓
SQLite insert with content hash for duplicate detection
    ↓
Notify listeners
    ↓
ClipboardHistoryView.on_clipboard_history_change()
    ↓
update_data() refreshes UI
```

### Settings Integration

```
Settings → Backup & Restore
    ↓
[Export Clipboard History] [Import Clipboard History]
    ↓
Storage Access Framework file picker
    ↓
SettingsActivity.performExportClipboard()
    OR
SettingsActivity.performImportClipboard()
    ↓
ClipboardDatabase.exportToJSON() / importFromJSON()
    ↓
JSON file with active and pinned entries
```

---

## UI Components

### Clipboard Pane Layout

**File**: `res/layout/clipboard_pane.xml`

**Layout Structure** (v1.32.313+):
```xml
┌────────────────────────────────────────────────────┐
│  PINNED (scrollable, dynamic minHeight)             │
│  ┌──────────────────────────────────────────────┐  │
│  │ Pinned entry 1...      [↓] [📋] [🗑]        │  │
│  │ Pinned entry 2...      [↓] [📋] [🗑]        │  │
│  └──────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────┤
│  [↑Pinned ↓Unpinned] │ [Tap to search...]         │
├────────────────────────────────────────────────────┤
│  HISTORY (scrollable, remaining space)              │
│  ┌──────────────────────────────────────────────┐  │
│  │ Entry text...          [↓] [📋] [📌]        │  │
│  │ Multi-line expanded... [↑] [📋] [📌]        │  │
│  │ ...                                          │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

**Key Changes** (v1.32.313-324):
- **Removed**: "Pinned" heading row (allows pinned section to expand upward)
- **Changed**: "History" heading to "↑Pinned ↓Unpinned" using Unicode arrows (U+2191, U+2193)
- **Split**: Heading/search bar now 50/50 horizontal layout (layout_weight="0.5" each)
- **Dynamic Height**: Pinned section uses minHeight when 2+ items exist (currently 100dp, being tuned)

**Components**:
- `ScrollView`: Pinned section wrapper with maxHeight="600dp", dynamic minHeight
- `MaxHeightListView clipboard_pin_view`: Pinned clipboard entries (extends with minHeight logic)
- `TextView clipboard_history_heading`: "↑Pinned ↓Unpinned" label (left 50%)
- `TextView clipboard_search`: Search input (right 50%, hint="Tap to search...")
- `MaxHeightListView clipboard_history_view`: Active clipboard entries

### Clipboard History Entry Layout

**File**: `res/layout/clipboard_history_entry.xml`

**Structure** (v1.32.308 - horizontal layout):
```xml
<LinearLayout android:orientation="horizontal">
  <TextView android:id="@+id/clipboard_entry_text"
            android:maxLines="1"
            android:ellipsize="end"/>
  <LinearLayout android:layout_gravity="top">
    <View android:id="@+id/clipboard_entry_expand"/>  <!-- Shown for multi-line -->
    <View android:id="@+id/clipboard_entry_paste"/>
    <View android:id="@+id/clipboard_entry_addpin"/>
  </LinearLayout>
</LinearLayout>
```

**Buttons**:
- `clipboard_entry_expand`: Expand/collapse toggle (only visible for multi-line entries)
- `clipboard_entry_paste`: Insert clipboard content into editor
- `clipboard_entry_addpin`: Pin this entry (moves to pinned section)

### Clipboard Pin Entry Layout

**File**: `res/layout/clipboard_pin_entry.xml`

**Structure** (v1.32.308 - horizontal layout):
```xml
<LinearLayout android:orientation="horizontal">
  <TextView android:id="@+id/clipboard_pin_text"
            android:maxLines="1"
            android:ellipsize="end"/>
  <LinearLayout android:layout_gravity="top">
    <View android:id="@+id/clipboard_pin_expand"/>   <!-- Shown for multi-line -->
    <View android:id="@+id/clipboard_pin_paste"/>
    <View android:id="@+id/clipboard_pin_remove"/>
  </LinearLayout>
</LinearLayout>
```

**Buttons**:
- `clipboard_pin_expand`: Expand/collapse toggle (only visible for multi-line entries)
- `clipboard_pin_paste`: Insert clipboard content into editor
- `clipboard_pin_remove`: Delete entry entirely from database (v1.32.309)

### Expand/Collapse Icon

**File**: `res/drawable/ic_expand_more.xml`

Material Design chevron-down icon (24dp):
- Default state (rotation=0°): Points down, indicates "expand"
- Expanded state (rotation=180°): Points up, indicates "collapse"

---

## Data Storage

### ClipboardDatabase (SQLite)

**File**: `srcs/juloo.keyboard2/ClipboardDatabase.java`

**Purpose**: Persistent storage for clipboard history using SQLite

**Database Schema**:
```sql
CREATE TABLE clipboard_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL,
  timestamp INTEGER NOT NULL,
  expiry_timestamp INTEGER NOT NULL,
  is_pinned INTEGER DEFAULT 0,
  content_hash TEXT NOT NULL
);

CREATE INDEX idx_content_hash ON clipboard_entries (content_hash);
CREATE INDEX idx_timestamp ON clipboard_entries (timestamp DESC);
CREATE INDEX idx_expiry ON clipboard_entries (expiry_timestamp);
```

**Columns**:
- `id`: Auto-increment primary key
- `content`: The clipboard text content
- `timestamp`: Unix timestamp when entry was created
- `expiry_timestamp`: Unix timestamp when entry expires (0 for pinned)
- `is_pinned`: 0 = active entry, 1 = pinned entry
- `content_hash`: String hash of content for duplicate detection

**Key Methods**:

#### addClipboardEntry(String content, long expiryTimestamp)
- **Purpose**: Add new clipboard entry with duplicate detection
- **Location**: ClipboardDatabase.java:87-136
- **Duplicate Check**: Uses content_hash for fast lookup
- **Returns**: boolean (true if added, false if duplicate)

#### getActiveClipboardEntries()
- **Purpose**: Get all non-expired, non-pinned entries
- **Location**: ClipboardDatabase.java:138-177
- **Query**: `WHERE is_pinned = 0 AND expiry_timestamp > currentTime ORDER BY timestamp DESC`
- **Returns**: List<String>

#### getPinnedEntries()
- **Purpose**: Get all pinned entries
- **Location**: ClipboardDatabase.java:179-216
- **Query**: `WHERE is_pinned = 1 ORDER BY timestamp DESC`
- **Returns**: List<String>

#### removeClipboardEntry(String content)
- **Purpose**: Delete entry entirely from database
- **Location**: ClipboardDatabase.java:218-240
- **Used By**: ClipboardPinView.remove_entry() (v1.32.309)
- **Returns**: boolean (true if deleted)

#### setPinnedStatus(String content, boolean isPinned)
- **Purpose**: Toggle pinned status of an entry
- **Location**: ClipboardDatabase.java:295-323
- **Behavior**: Sets is_pinned flag and expiry_timestamp (0 if pinned)
- **Returns**: boolean (true if updated)

#### cleanupExpiredEntries()
- **Purpose**: Remove all expired entries
- **Location**: ClipboardDatabase.java:266-293
- **Query**: `DELETE WHERE is_pinned = 0 AND expiry_timestamp > 0 AND expiry_timestamp < currentTime`
- **Returns**: int (count of deleted entries)

#### exportToJSON()
- **Purpose**: Export all entries to structured JSON
- **Location**: ClipboardDatabase.java:416-495
- **Returns**: JSONObject with active_entries, pinned_entries, metadata

#### importFromJSON(JSONObject importData)
- **Purpose**: Import entries from JSON with duplicate prevention
- **Location**: ClipboardDatabase.java:497-604
- **Returns**: int[] {activeAdded, pinnedAdded, duplicatesSkipped}

---

## User Workflows

### Workflow 1: View Clipboard History

1. User opens keyboard clipboard pane (swipe up gesture or button)
2. ClipboardHistoryView loads from ClipboardDatabase
3. Calls cleanupExpiredEntries() first
4. Calls getActiveClipboardEntries() to load active entries
5. ClipboardPinView calls getPinnedEntries() to load pinned entries
6. Both views display in separate sections

### Workflow 1a: Search Clipboard History

1. User taps the search box (clipboard_search TextView)
2. Keyboard2.java sets _clipboardSearchMode = true
3. Search box hint changes to "Type on keyboard below..."
4. User types on physical keyboard below clipboard pane
5. Keyboard2.java intercepts keypresses in search mode
6. For each keypress, calls _clipboardHistoryView.setSearchFilter(searchText)
7. ClipboardHistoryView.applyFilter() filters entries by search text (case-insensitive contains)
8. Both history and pinned lists update to show only matching entries
9. User can clear search by tapping back or switching panes

**Implementation**:
- Keyboard2.java:56 - _clipboardSearchMode flag
- Keyboard2.java:764-778 - Search box click handler
- Keyboard2.java:919 - isInClipboardSearchMode() method
- ClipboardHistoryView.java:43-69 - setSearchFilter() and applyFilter()
- Uses database-level filtering for performance

### Workflow 2: Pin Clipboard Entry

1. User taps "pin" button on entry in history section
2. ClipboardHistoryView.pin_entry(pos) called
3. ClipboardHistoryService.set_pinned_status(clip, true)
4. ClipboardDatabase updates is_pinned = 1, expiry_timestamp = 0
5. ClipboardPinView.refresh_pinned_items() updates pinned section
6. Entry appears in pinned section, removed from history

### Workflow 3: Delete Pinned Entry (v1.32.309)

1. User taps "delete" button on pinned entry
2. Confirmation dialog shows: "Remove this clipboard entry?"
3. User taps "Remove"
4. ClipboardPinView.remove_entry(pos) called
5. ClipboardHistoryService.remove_history_entry(clip) called
6. ClipboardDatabase.removeClipboardEntry(clip) deletes row
7. If entry was current clipboard, system clipboard is cleared
8. Entry is completely removed (not moved to history)

### Workflow 4: Expand Multi-Line Entry (v1.32.308)

1. User views entry with newline characters (\n)
2. Expand button automatically appears before paste button
3. Entry defaults to collapsed (maxLines=1, ellipsize=END)
4. Expand button rotation = 0° (chevron pointing down)
5. User taps expand button
6. TextView changes to maxLines=Integer.MAX_VALUE, ellipsize=null
7. Expand button rotates 180° (chevron pointing up)
8. State saved in _expandedStates HashMap keyed by position
9. User taps again to collapse back to 1 line

### Workflow 5: Export Clipboard (v1.32.306)

1. User navigates to Settings → Backup & Restore
2. User taps "Export Clipboard History" preference
3. System shows Storage Access Framework file picker
4. User selects save location
5. SettingsActivity.performExportClipboard():
   - Gets ClipboardDatabase instance
   - Calls exportToJSON()
   - Writes to file via ContentResolver
6. File saved with name: `clipboard-history-YYYYMMDD_HHMMSS.json`
7. Toast shows: "Successfully exported:\n• N active entry/ies\n• M pinned entry/ies"

**File Structure**:
- res/xml/settings.xml:140 - Export button preference
- srcs/juloo.keyboard2/SettingsActivity.java:35 - REQUEST_CODE_EXPORT_CLIPBOARD
- srcs/juloo.keyboard2/SettingsActivity.java:1317-1396 - performExportClipboard()

**JSON Format**:
```json
{
  "active_entries": [
    {
      "content": "text content",
      "timestamp": 1731369060000,
      "expiry_timestamp": 1731455460000
    }
  ],
  "pinned_entries": [
    {
      "content": "important text",
      "timestamp": 1731369000000,
      "expiry_timestamp": 0
    }
  ],
  "export_version": 1,
  "export_date": "2025-11-11 20:31:00",
  "total_active": 5,
  "total_pinned": 2
}
```

### Workflow 6: Import Clipboard (v1.32.306)

1. User navigates to Settings → Backup & Restore
2. User taps "Import Clipboard History" preference
3. System shows Storage Access Framework file picker
4. User selects JSON file
5. SettingsActivity.performImportClipboard():
   - Reads JSON file via ContentResolver
   - Parses JSON and validates structure
   - Calls ClipboardDatabase.importFromJSON()
   - Smart merge:
     - Checks content_hash for duplicates
     - Skips entries that already exist
     - Preserves original timestamps and expiry dates
     - Maintains pinned status from import
6. Toast shows: "Successfully imported:\n• N active entry/ies added\n• M pinned entry/ies added\n• K duplicate(s) skipped"

**File Structure**:
- res/xml/settings.xml:141 - Import button preference
- srcs/juloo.keyboard2/SettingsActivity.java:36 - REQUEST_CODE_IMPORT_CLIPBOARD
- srcs/juloo.keyboard2/SettingsActivity.java:1398-1488 - performImportClipboard()

**Duplicate Prevention**:
- Uses content_hash (String.valueOf(content.hashCode()))
- Query: `SELECT id FROM clipboard_entries WHERE content_hash = ? LIMIT 1`
- If exists, increments duplicatesSkipped counter
- If new, inserts with original timestamp and expiry

---

## Technical Implementation

### ClipboardHistoryService

**File**: `srcs/juloo.keyboard2/ClipboardHistoryService.java`

**Purpose**: Service layer between UI and database, handles clipboard monitoring

**Key Components**:

#### Singleton Pattern
```java
private static ClipboardHistoryService _service = null;
public static ClipboardHistoryService get_service(Context ctx)
```

#### Clipboard Monitoring
```java
private final ClipboardManager.OnPrimaryClipChangedListener _listener =
    new OnPrimaryClipChangedListener() {
      public void onPrimaryClipChanged() {
        add_clip_to_history();
      }
    };
```

#### Duplicate Prevention
- Checks if new clipboard matches last entry in history
- Only adds if different from most recent entry
- Prevents consecutive duplicates

#### Methods

**add_clip_to_history()**
- **Location**: ClipboardHistoryService.java:241-262
- Reads current clipboard content
- Checks for consecutive duplicates
- Calls ClipboardDatabase.addClipboardEntry()
- Notifies UI listeners

**clear_expired_and_get_history()**
- **Location**: ClipboardHistoryService.java:141-150
- Calls ClipboardDatabase.cleanupExpiredEntries()
- Calls ClipboardDatabase.getActiveClipboardEntries()
- Returns List<String>

**get_pinned_entries()**
- **Location**: ClipboardHistoryService.java:152-155
- Delegates to ClipboardDatabase.getPinnedEntries()
- Returns List<String>

**set_pinned_status(String clip, boolean isPinned)**
- **Location**: ClipboardHistoryService.java:157-162
- Delegates to ClipboardDatabase.setPinnedStatus()
- Notifies UI listeners

**remove_history_entry(String clip)**
- **Location**: ClipboardHistoryService.java:212-239
- Checks if entry is current clipboard
- Clears system clipboard if removing current entry
- Calls ClipboardDatabase.removeClipboardEntry()
- Notifies UI listeners

**paste(String text)** (static)
- **Location**: ClipboardHistoryService.java:264-272
- Sets clipboard content
- Sends ACTION_PASTE broadcast
- Used by both ClipboardHistoryView and ClipboardPinView

#### Maximum Item Size Enforcement (v1.32.327)

**Purpose**: Prevent system overload from accidentally copying massive text (entire files, large documents)

**Implementation** (ClipboardHistoryService.java:251-277):
```java
// Check maximum item size limit
int maxSizeKb = Config.globalConfig().clipboard_max_item_size_kb;
if (maxSizeKb > 0)
{
  try
  {
    int sizeBytes = clip.getBytes(java.nio.charset.StandardCharsets.UTF_8).length;
    int maxSizeBytes = maxSizeKb * 1024;

    if (sizeBytes > maxSizeBytes)
    {
      // Item exceeds size limit - reject and notify user
      android.util.Log.w("ClipboardHistory", "Clipboard item too large: " + sizeBytes + " bytes");

      // Show toast notification to user
      String message = String.format("Clipboard item too large (%d KB). Limit is %d KB.",
          sizeBytes / 1024, maxSizeKb);
      android.widget.Toast.makeText(_context, message, android.widget.Toast.LENGTH_LONG).show();
      return; // Don't add to clipboard history
    }
  }
  catch (Exception e)
  {
    android.util.Log.e("ClipboardHistory", "Error checking clipboard item size: " + e.getMessage());
    // Continue with add if size check fails
  }
}
```

**User Setting**: Settings → Clipboard → "Maximum size per item"
**Options** (res/values/arrays.xml:125-140):
- 100 KB (restrictive)
- 250 KB
- 500 KB (recommended default)
- 1 MB
- 5 MB
- Unlimited (maxSizeKb = 0, bypasses check)

**Behavior**:
- Calculates UTF-8 byte size of clipboard content
- Compares against user-configured limit
- If exceeded:
  - Rejects item (doesn't add to clipboard history)
  - Shows Toast notification with actual and limit sizes
  - Logs warning to logcat
- If check fails (exception):
  - Continues with add (fail-open for reliability)
  - Logs error to logcat

#### Clipboard Storage Statistics (v1.32.329)

**Purpose**: Provide users with visibility into clipboard usage (entry count and storage size)

**Implementation**:

**ClipboardDatabase.StorageStats** (ClipboardDatabase.java:365-383):
```java
public static class StorageStats
{
    public int totalEntries;      // Total entries in database
    public int activeEntries;     // Non-expired entries
    public int pinnedEntries;     // Pinned entries only
    public long totalSizeBytes;   // Total UTF-8 byte size
    public long activeSizeBytes;  // Active entries size
    public long pinnedSizeBytes;  // Pinned entries size
}
```

**ClipboardDatabase.getStorageStats()** (ClipboardDatabase.java:389-454):
- Queries all entries with content, is_pinned, and expiry_timestamp
- Calculates UTF-8 byte size for each entry
- Separates stats into total, active (non-expired), and pinned
- Returns StorageStats object with complete breakdown

**ClipboardHistoryService.getStorageStats()** (ClipboardHistoryService.java:323-341):
- Calls ClipboardDatabase.getStorageStats()
- Formats output using formatBytes() helper (B/KB/MB)
- Returns multi-line string:
  - Line 1: "X active entries (Y KB/MB)"
  - Line 2 (if pinned exist): "Z pinned (W KB/MB)"

**ClipboardHistoryService.formatBytes()** (ClipboardHistoryService.java:344-358):
- Converts bytes to human-readable format
- < 1024: "X B"
- < 1 MB: "X.X KB" (1 decimal)
- >= 1 MB: "X.XX MB" (2 decimals)

**SettingsActivity Integration**:

**Preference Definition** (settings.xml:136):
```xml
<Preference android:key="clipboard_storage_stats"
            android:title="Clipboard usage"
            android:summary="Loading statistics..."
            android:dependency="clipboard_history_enabled"
            android:selectable="false"/>
```

**Dynamic Update** (SettingsActivity.java:1856-1879):
- onCreate(): Calls updateClipboardStats() on initial display
- onResume(): Calls updateClipboardStats() when returning to settings (line 1794)
- updateClipboardStats(): Fetches stats from ClipboardHistoryService and updates preference summary
- Handles service unavailable and error cases gracefully

**User Experience**:
- Appears as non-clickable informational preference
- Disabled when clipboard_history_enabled is false
- Updates immediately when settings screen opens
- Shows real-time clipboard storage usage
- Helps users understand clipboard memory footprint

#### Clipboard Limit Type Toggle (v1.32.330)

**Purpose**: Allow users to choose between count-based and size-based clipboard history limits

**Implementation**:

**Limit Type Preference** (settings.xml:132):
```xml
<ListPreference android:key="clipboard_limit_type"
                android:title="History limit type"
                android:summary="Choose between count-based or size-based limit"
                android:dependency="clipboard_history_enabled"
                android:defaultValue="count"
                android:entries="@array/pref_clipboard_limit_type_entries"
                android:entryValues="@array/pref_clipboard_limit_type_values"/>
```

**Options** (arrays.xml:141-148):
- "count" (default): Item count (default) - limit by number of entries
- "size": Total storage size - limit by total MB

**Size Limit Preference** (settings.xml:134):
```xml
<ListPreference android:key="clipboard_size_limit_mb"
                android:title="Maximum total size"
                android:summary="Total storage size limit for clipboard history"
                android:dependency="clipboard_history_enabled"
                android:defaultValue="10"
                android:entries="@array/pref_clipboard_size_limit_mb_entries"
                android:entryValues="@array/pref_clipboard_size_limit_mb_values"/>
```

**Size Limit Options** (arrays.xml:149-166):
- 1 MB
- 5 MB
- 10 MB (recommended default)
- 25 MB
- 50 MB
- 100 MB
- Unlimited (0 = no limit)

**ClipboardDatabase.applySizeLimitBytes()** (ClipboardDatabase.java:512-596):
```java
public int applySizeLimitBytes(int maxSizeMB)
{
    if (maxSizeMB <= 0)
        return 0; // No limit

    long maxSizeBytes = maxSizeMB * 1024L * 1024L; // Convert MB to bytes

    // Get all non-pinned active entries ordered by timestamp (oldest first)
    // Calculate cumulative size and identify entries to delete
    // Delete oldest entries until total size is under limit
}
```

**Algorithm**:
1. Query all non-pinned active entries ordered by timestamp ASC (oldest first)
2. Iterate through entries, calculating cumulative UTF-8 byte size
3. Once cumulative size exceeds maxSizeBytes, mark remaining entries for deletion
4. Build and execute DELETE query with IN clause
5. Log entries deleted and total size

**ClipboardHistoryService Integration** (ClipboardHistoryService.java:287-310):
```java
// Apply size limits if configured (based on limit type)
String limitType = Config.globalConfig().clipboard_limit_type;
if ("size".equals(limitType))
{
    // Apply size-based limit (total MB)
    int maxSizeMB = Config.globalConfig().clipboard_size_limit_mb;
    if (maxSizeMB > 0)
    {
        _database.applySizeLimitBytes(maxSizeMB);
    }
}
else
{
    // Apply count-based limit (default)
    int maxHistorySize = Config.globalConfig().clipboard_history_limit;
    if (maxHistorySize > 0)
    {
        _database.applySizeLimit(maxHistorySize);
    }
}
```

**Config Fields** (Config.java:76-77):
```java
public String clipboard_limit_type; // "count" or "size" - type of history limit
public int clipboard_size_limit_mb; // Maximum total size in MB when using size-based limit, 0 = unlimited
```

**Default Values**:
- clipboard_limit_type: "count" (backward compatible)
- clipboard_size_limit_mb: 10 (10MB recommended)

**User Experience**:
- Both limit type and size limit preferences visible in Settings → Clipboard
- Count limit used when limit type is "count" (default)
- Size limit used when limit type is "size"
- Limit applied after each new clipboard entry is added
- Oldest entries deleted first to stay within limit
- Pinned entries never deleted by limit enforcement

### ClipboardHistoryView

**File**: `srcs/juloo.keyboard2/ClipboardHistoryView.java`

**Purpose**: ListView for active clipboard entries with expand/collapse

**Key Features**:

#### Expand/Collapse State Management (v1.32.308)
```java
// Track expanded state: position -> isExpanded
java.util.Map<Integer, Boolean> _expandedStates = new java.util.HashMap<>();
```

**Location**: ClipboardHistoryView.java:24

#### Multi-Line Detection
```java
final boolean isMultiLine = text.contains("\n");
final boolean isExpanded = _expandedStates.containsKey(pos) && _expandedStates.get(pos);
```

**Location**: ClipboardHistoryView.java:137-138

#### Dynamic UI Updates
- Show/hide expand button based on isMultiLine
- Set maxLines to 1 (collapsed) or Integer.MAX_VALUE (expanded)
- Rotate expand button 0° or 180°
- Toggle state on click and call notifyDataSetChanged()

**Location**: ClipboardHistoryView.java:140-176

#### Search Filter (TODO)
```java
public void setSearchFilter(String filter)
```

**Location**: ClipboardHistoryView.java:43-47
**Status**: Method exists but not yet hooked up to search input

### ClipboardPinView

**File**: `srcs/juloo.keyboard2/ClipboardPinView.java`

**Purpose**: ListView for pinned clipboard entries with expand/collapse and dynamic height

**Key Features**:

Same expand/collapse implementation as ClipboardHistoryView:
- _expandedStates HashMap for state tracking
- Multi-line detection with text.contains("\n")
- Dynamic button visibility and rotation
- Location: ClipboardPinView.java:24, 100-140

**Critical Change (v1.32.309)**:

#### remove_entry() - Delete Entirely
```java
// OLD (v1.32.308):
_service.set_pinned_status(clip, false);  // Just unpinned

// NEW (v1.32.309):
_service.remove_history_entry(clip);      // Delete entirely
```

**Location**: ClipboardPinView.java:58
**Reason**: User expectation - delete should remove completely, not move to history

**Dynamic Height (v1.32.319-325)**:

#### updateParentMinHeight() - User-Configurable Pinned Section Size
```java
private void updateParentMinHeight()
{
  if (_entries.size() >= 2)
  {
    // Read user preference for pinned section size (default 100dp = 2-3 rows)
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(getContext());
    int minHeightDp = Integer.parseInt(prefs.getString("clipboard_pinned_rows", "100"));
    int minHeightPx = (int)(minHeightDp * getResources().getDisplayMetrics().density);
    setMinimumHeight(minHeightPx);
  }
  else
  {
    // Clear minHeight when less than 2 items
    setMinimumHeight(0);
  }
}
```

**Location**: ClipboardPinView.java:52-67
**Called From**: refresh_pinned_items() after updating _entries
**Purpose**: Ensure pinned section displays user-preferred number of entries without requiring scroll
**User Setting**: Settings → Clipboard → "Pinned section size" (v1.32.325)
**Options**:
- 70dp: 1-2 rows (compact)
- 100dp: 2-3 rows (default)
- 140dp: 3-4 rows (comfortable)
- 200dp: 4-5 rows (expanded)

### MaxHeightListView

**File**: `srcs/juloo.keyboard2/MaxHeightListView.java`

**Purpose**: Custom ListView that enforces both maxHeight and minHeight constraints

**Key Features**:

#### maxHeight Enforcement (Original)
```java
if (_maxHeight > 0)
{
  heightMeasureSpec = View.MeasureSpec.makeMeasureSpec(_maxHeight, View.MeasureSpec.AT_MOST);
}
```

**Location**: MaxHeightListView.java:59-69

#### minHeight Enforcement (v1.32.322)
```java
// Respect minHeight if set
int minHeight = getSuggestedMinimumHeight();
if (minHeight > 0)
{
  int measuredHeight = getMeasuredHeight();
  if (measuredHeight < minHeight)
  {
    // Enforce minHeight
    setMeasuredDimension(getMeasuredWidth(), minHeight);
  }
}
```

**Location**: MaxHeightListView.java:73-83
**Reason**: Default onMeasure() only handled maxHeight, ignored minHeight set by setMinimumHeight()
**Impact**: Allows ClipboardPinView to expand to minimum size even when content wraps smaller

---

## Import/Export

### Export Implementation

**Method**: `ClipboardDatabase.exportToJSON()`
**Location**: ClipboardDatabase.java:416-495

**Process**:
1. Query all entries: `SELECT content, timestamp, expiry_timestamp, is_pinned`
2. Separate into active and pinned arrays based on is_pinned flag
3. Create JSON structure with metadata
4. Return JSONObject

**JSON Structure**:
```json
{
  "active_entries": [
    {"content": "...", "timestamp": 123, "expiry_timestamp": 456}
  ],
  "pinned_entries": [
    {"content": "...", "timestamp": 789, "expiry_timestamp": 0}
  ],
  "export_version": 1,
  "export_date": "2025-11-11 20:31:00",
  "total_active": 5,
  "total_pinned": 2
}
```

### Import Implementation

**Method**: `ClipboardDatabase.importFromJSON(JSONObject importData)`
**Location**: ClipboardDatabase.java:497-604

**Process**:
1. Parse active_entries and pinned_entries arrays
2. For each entry:
   - Calculate content_hash
   - Check if exists: `SELECT id FROM clipboard_entries WHERE content_hash = ?`
   - If duplicate, skip and increment counter
   - If new, insert with original timestamp and expiry
3. Return counts: [activeAdded, pinnedAdded, duplicatesSkipped]

**Duplicate Detection**:
- Uses content_hash (String.valueOf(content.hashCode()))
- Fast lookup via indexed column
- Preserves original entries (no overwrites)

**Data Integrity**:
- Validates JSON structure (required fields)
- Preserves timestamps from export
- Maintains pinned status
- Handles empty arrays gracefully

---

## Performance Requirements

### Database Performance

- **Indexes**: content_hash, timestamp, expiry_timestamp
- **Duplicate Check**: O(1) hash lookup via index
- **Active Entries Query**: O(log n) with timestamp index
- **Cleanup**: O(n) for expired entries (run on view open)

### Memory Usage

- **ListView**: Only visible items in memory (view recycling)
- **Expand State**: HashMap O(1) lookup, minimal memory (only expanded items stored)
- **Database Cursor**: Closed immediately after reading to list

### UI Responsiveness

- **Target**: < 100ms for list updates
- **Expand/Collapse**: Instant with notifyDataSetChanged()
- **Search**: Database-level filtering (when implemented)

---

## Error Handling

### Database Errors

**Scenario**: SQLite write failure

**Handling**:
- Log error with android.util.Log.e()
- Return false from add/remove methods
- UI shows no change (silent failure)

**Future**: Show toast with error message

### Import Errors

**Scenario**: Invalid JSON format or missing fields

**Handling**:
- Try-catch around JSON parsing
- Show error toast with specific message
- Return early, no partial imports

**Error Messages**:
- "Invalid clipboard data file"
- "Failed to read clipboard file"
- "No clipboard entries to import"

### Export Errors

**Scenario**: File write failure (storage full, permission denied)

**Handling**:
- Try-catch around file write
- Show error toast
- File not created

**Error Messages**:
- "Failed to export clipboard history"
- "No clipboard entries to export"

### Delete Errors

**Scenario**: Database delete failure

**Handling**:
- Show error dialog (already implemented)
- Entry remains in list
- User can retry

---

## Known Issues & Future Enhancements

### Known Issues

#### 1. No Import Validation Feedback

**Severity**: Low

**Description**: Import doesn't show which entries were duplicates

**Current**: Shows counts only ("• 5 duplicates skipped")

**Enhancement**: Show list of skipped entries for user review

#### 2. No Clipboard History Count Limit

**Severity**: Low

**Description**: Clipboard history entry count can grow indefinitely (only expired entries cleaned up)

**Risk**: Database size growth over time (mitigated by max item size limit in v1.32.327)

**Note**: Max item size limit (v1.32.327) prevents individual large items, but total entry count still unlimited

**Future**: Add max entry count setting (e.g., keep last 1000 entries)

#### 3. Expand State Lost on Refresh

**Severity**: Low

**Description**: When clipboard updates, _expandedStates HashMap cleared

**Workaround**: User must re-expand entries

**Fix**: Use stable IDs (content hash) instead of position
- **Complexity**: Medium - requires adapter changes

#### 4. Pinned Section minHeight Tuning (RESOLVED v1.32.325)

**Severity**: Low

**Status**: Resolved - Made User-Configurable

**Description**: Dynamic minHeight for pinned section needed tuning for different user preferences

**History**:
- 400dp → 5.5 entries visible (too large)
- 200dp → 5.5 entries visible (too large)
- 140dp → 4 entries visible (still too large)
- 100dp → 2-3 entries visible (good for most users)

**Solution (v1.32.325)**: Added user-configurable setting in Settings → Clipboard → "Pinned section size"
- 70dp: 1-2 rows (compact)
- 100dp: 2-3 rows (default)
- 140dp: 3-4 rows (comfortable)
- 200dp: 4-5 rows (expanded)

**Implementation**:
- `res/xml/settings.xml:134` - ListPreference with dependency on clipboard_history_enabled
- `res/values/arrays.xml:113-124` - Options and values arrays
- `ClipboardPinView.java:57-58` - Reads SharedPreferences value dynamically

**Note**: Entry heights still vary with content (single-line vs expanded), but users can now choose their preferred size

### Future Enhancements

#### 1. Rich Text Support

- Store formatted text (HTML)
- Preview formatting in UI
- Export with formatting preserved

#### 2. Image Clipboard Support

- Store image URIs or thumbnails
- Preview in list
- Export images with metadata

#### 3. Clipboard Categories/Tags

- User-defined categories
- Filter by category
- Auto-categorize (URLs, emails, code)

#### 4. Cloud Backup

- Optional cloud sync
- End-to-end encryption
- Cross-device clipboard sharing

#### 5. Smart Suggestions

- Frequently pasted entries
- Context-aware suggestions
- Quick paste shortcuts

#### 6. Clipboard History Limit Setting

- Max entry count (e.g., 100, 500, 1000)
- Max age in days
- Configurable in Settings

#### 7. Bulk Operations

- Select multiple entries
- Bulk delete
- Bulk pin/unpin
- Bulk export selected

---

## File Structure

```
srcs/juloo.keyboard2/
├── ClipboardDatabase.java           # SQLite storage (604 lines)
├── ClipboardHistoryService.java     # Service layer (342 lines)
├── ClipboardHistoryView.java        # Active entries list (194 lines)
├── ClipboardPinView.java            # Pinned entries list + dynamic height (178 lines)
├── MaxHeightListView.java           # Custom ListView with min/max height (86 lines)
└── SettingsActivity.java            # Import/export handlers (partial)

res/layout/
├── clipboard_pane.xml               # Main clipboard pane (v1.32.313+ reorganized)
├── clipboard_history_entry.xml      # Active entry item (horizontal)
└── clipboard_pin_entry.xml          # Pinned entry item (horizontal)

res/values/
├── strings.xml                      # Contains "↑Pinned ↓Unpinned" heading (line 153)
└── styles.xml                       # clipboardEntry style with 7dp margins

res/drawable/
└── ic_expand_more.xml               # Chevron icon for expand/collapse

res/xml/
└── settings.xml                     # Export/import preference buttons
```

---

## Integration Points

### System ClipboardManager

**API**:
- `ClipboardManager.getPrimaryClip()`: Read clipboard
- `ClipboardManager.setPrimaryClip()`: Write clipboard
- `ClipboardManager.clearPrimaryClip()`: Clear clipboard (API 28+)
- `ClipboardManager.addPrimaryClipChangedListener()`: Monitor changes

**Permissions**: None required (clipboard access automatic for IME)

### Storage Access Framework

**Used For**: Import/export file picker

**API**:
- `Intent.ACTION_CREATE_DOCUMENT`: Export file picker
- `Intent.ACTION_OPEN_DOCUMENT`: Import file picker
- `ContentResolver.openOutputStream()`: Write export file
- `ContentResolver.openInputStream()`: Read import file

**MIME Type**: `application/json`

**Permissions**: None required (user grants per-file access via SAF)

### SharedPreferences

**Not Used**: Clipboard data stored in SQLite, not SharedPreferences

**Reason**: SQLite provides better query performance and data integrity

---

## Dependencies

### Core Android

- `android.database.sqlite.*` - SQLite database
- `android.content.ClipboardManager` - System clipboard
- `android.content.ContentResolver` - File I/O via SAF
- `org.json.*` - JSON parsing

### UI Components

- `android.widget.ListView` (via MaxHeightListView)
- `android.widget.BaseAdapter`
- `android.widget.TextView`
- `android.view.View`

---

## Changelog

### v1.32.330 (2025-11-12)
- **FEATURE**: Added clipboard limit type toggle (count vs size)
  - Toggle between count-based limit (number of entries) and size-based limit (total MB)
  - Settings → Clipboard → "History limit type"
  - Count-based: Uses existing "History entry limit" setting (100-1000 entries or unlimited)
  - Size-based: New "Maximum total size" setting (1MB, 5MB, 10MB, 25MB, 50MB, 100MB, or unlimited)
  - Default: count-based for backward compatibility
  - Dynamically applies appropriate limit when adding new clipboard entries
- **Files Modified**:
  - res/values/arrays.xml: Added limit type toggle and size limit arrays
  - res/xml/settings.xml: Added clipboard_limit_type and clipboard_size_limit_mb preferences
  - srcs/juloo.keyboard2/Config.java: Added clipboard_limit_type and clipboard_size_limit_mb fields
  - srcs/juloo.keyboard2/ClipboardDatabase.java: Added applySizeLimitBytes() method
  - srcs/juloo.keyboard2/ClipboardHistoryService.java: Updated add_clip() to use appropriate limit

### v1.32.329 (2025-11-12)
- **FEATURE**: Added clipboard usage statistics display
  - Real-time display in Settings → Clipboard → "Clipboard usage"
  - Shows active entry count and size (e.g., "5 active entries (12.3 KB)")
  - Shows pinned entry count and size when pinned items exist (e.g., "2 pinned (3.4 KB)")
  - Refreshes automatically when settings screen is opened
  - Updates dynamically when clipboard changes
- **Files Modified**:
  - srcs/juloo.keyboard2/ClipboardDatabase.java: Added StorageStats class and getStorageStats()
  - srcs/juloo.keyboard2/ClipboardHistoryService.java: Updated getStorageStats() and added formatBytes()
  - res/xml/settings.xml: Added clipboard_storage_stats preference
  - srcs/juloo.keyboard2/SettingsActivity.java: Added updateClipboardStats() and onResume() integration

### v1.32.327 (2025-11-12)
- **FEATURE**: Added maximum item size limit for clipboard entries
  - Prevents system overload from accidentally copying massive text (entire files, large documents)
  - User-configurable in Settings → Clipboard → "Maximum size per item"
  - Options: 100 KB, 250 KB, 500 KB (default), 1 MB, 5 MB, Unlimited
  - Rejects items exceeding limit with Toast notification showing actual and limit sizes
  - Calculates UTF-8 byte size for accurate measurement
  - Fail-open design: continues with add if size check fails
- **Files Modified**:
  - res/values/arrays.xml: Added size limit options and values arrays
  - res/xml/settings.xml: Added "Maximum size per item" ListPreference
  - srcs/juloo.keyboard2/Config.java: Added clipboard_max_item_size_kb field
  - srcs/juloo.keyboard2/ClipboardHistoryService.java: Added size check in add_clip()

### v1.32.325 (2025-11-12)
- **FEATURE**: Made pinned section size user-configurable
  - Settings → Clipboard → "Pinned section size"
  - Options: 70dp (1-2 rows), 100dp (2-3 rows, default), 140dp (3-4 rows), 200dp (4-5 rows)
  - Allows users to customize how many pinned entries display without scrolling
- **Files Modified**:
  - res/values/arrays.xml: Added pinned section size options
  - res/xml/settings.xml: Added "Pinned section size" ListPreference
  - srcs/juloo.keyboard2/ClipboardPinView.java: Added updateParentMinHeight() with SharedPreferences integration

### v1.32.309 (2025-11-11)
- **BUGFIX**: Fixed pinned clipboard deletion to delete entirely
  - Changed ClipboardPinView.remove_entry() from set_pinned_status(false) to remove_history_entry()
  - Delete button now completely removes entry from database
  - Entry no longer moves back to history, completely deleted
  - System clipboard cleared if deleting current entry

### v1.32.308 (2025-11-11)
- **UI**: Corrected clipboard entry layout to horizontal
  - Text and buttons now share same horizontal line
  - Buttons aligned to top using android:layout_gravity="top"
  - Reverted from vertical layout (v1.32.307) based on user feedback

### v1.32.307 (2025-11-11)
- **FEATURE**: Added expand/collapse for multi-line clipboard entries
  - Automatically detects multi-line entries (contains \n)
  - Defaults to collapsed (maxLines=1)
  - Expand button appears before paste button for multi-line entries
  - Button rotates 180° when expanded (visual feedback)
  - State tracked in _expandedStates HashMap per position
  - Implemented in both ClipboardHistoryView and ClipboardPinView
- **UI**: Moved action buttons to top-right corner
  - Changed from center-vertical alignment to top-aligned
  - Better UX for multi-line entries

### v1.32.306 (2025-11-11)
- **FEATURE**: Added export clipboard history to JSON
  - Button in Settings → Backup & Restore
  - Exports both active and pinned entries
  - Structured JSON with metadata (version, date, counts)
  - Filename: `clipboard-history-YYYYMMDD_HHMMSS.json`
  - Toast shows export summary
- **FEATURE**: Added import clipboard history from JSON
  - Button in Settings → Backup & Restore
  - Smart merge with duplicate prevention (uses content hash)
  - Preserves timestamps and pinned status from import
  - Toast shows import summary with added/skipped counts
- **Files Modified**:
  - res/xml/settings.xml: Added export/import buttons
  - srcs/juloo.keyboard2/ClipboardDatabase.java: Added exportToJSON() and importFromJSON()
  - srcs/juloo.keyboard2/SettingsActivity.java: Added export/import handlers
- **Integration**: Uses Storage Access Framework for file picker

### v1.32.0 (Initial Implementation)
- SQLite-based persistent clipboard history
- Pinned clipboard entries
- Automatic expiry (24h default for active entries)
- Duplicate prevention (consecutive entries)
- ListView UI with separate sections for history and pinned
- Paste functionality
- Pin/unpin functionality

---

## Sub-Optimal Areas & Remaining Work

### High Priority

1. **No Clipboard Entry Count Limit** (partially addressed v1.32.327)
   - **Issue**: Database can grow indefinitely (entry count)
   - **Impact**: Potential performance degradation over time
   - **Mitigation**: Max item size limit (v1.32.327) prevents individual large items
   - **Remaining**: Need max entry count setting
   - **Fix**: Add max entry count setting (e.g., 100, 500, 1000), implement in applySizeLimit()
   - **Effort**: Medium (~50 lines)
   - **Files**: ClipboardDatabase.java (applySizeLimit method exists, needs count variant), Config.java, res/xml/settings.xml

### Medium Priority

2. **Expand State Lost on Refresh**
   - **Issue**: _expandedStates uses position, lost when list updates
   - **Impact**: User must re-expand entries after clipboard changes
   - **Fix**: Use content hash as stable ID instead of position
   - **Effort**: Medium (~40 lines)
   - **Files**: ClipboardHistoryView.java, ClipboardPinView.java

3. **No Import Validation Details**
   - **Issue**: Import shows counts but not which entries were skipped
   - **Impact**: User doesn't know what was skipped
   - **Fix**: Build list of skipped entries, show in dialog
   - **Effort**: Low (~30 lines)
   - **Files**: SettingsActivity.java

4. **Silent Database Errors**
   - **Issue**: Database failures logged but not shown to user
   - **Impact**: User unaware of data loss
   - **Fix**: Show toast on database errors
   - **Effort**: Low (~10 lines per error point)
   - **Files**: ClipboardDatabase.java, ClipboardHistoryService.java

### Low Priority

5. **No Bulk Operations**
   - **Issue**: Cannot select multiple entries for deletion/pinning
   - **Impact**: Tedious for cleanup
   - **Fix**: Add selection mode with checkboxes
   - **Effort**: High (~200 lines)
   - **Files**: All clipboard view files

6. **No Rich Text Support**
   - **Issue**: Only plain text stored
   - **Impact**: Formatting lost
   - **Fix**: Store ClipData.Item with HTML
   - **Effort**: High (~150 lines)
   - **Files**: ClipboardDatabase.java, view files

---

## References

- [Android ClipboardManager API](https://developer.android.com/reference/android/content/ClipboardManager)
- [SQLite on Android](https://developer.android.com/training/data-storage/sqlite)
- [Storage Access Framework](https://developer.android.com/guide/topics/providers/document-provider)
- [Material Design Expand/Collapse](https://material.io/components/lists#behavior)
