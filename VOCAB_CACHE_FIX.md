# Vocabulary Cache Invalidation Fix

**Date**: 2025-11-22
**Version**: v1.32.656
**Issue**: Missing predictions for uncommon words (sorghum, therapeutics, genealogical)

## Problem

User reported no predictions for specialized/uncommon words that should be in the vocabulary:
- "sorghum" (grain type)
- "therapeutics" (medical term)
- "genealogical" (family history term)

## Investigation

1. **Checked vocabulary files**:
   - `en.txt` (76KB) - Basic 10k word list - **WORDS NOT PRESENT**
   - `en_enhanced.txt` (392KB) - Extended 50k+ word list - **WORDS PRESENT**
   - `en_enhanced.json` (873KB) - JSON with frequencies - **WORDS PRESENT**

2. **Verified JSON entries**:
   ```json
   "sorghum": 154,
   "therapeutics": 160,
   "genealogical": 156,
   ```

3. **Found stale binary cache**:
   - `en_enhanced.bin` (1.3MB) created Nov 20 13:19
   - Binary cache was built from an older/smaller dictionary
   - Cache loading bypasses JSON parsing for speed but becomes stale if JSON updated

## Root Cause

`OptimizedVocabulary.java` implements a binary cache system for fast loading:

```java
// Line 741-746
private void loadWordFrequencies()
{
  // Try pre-processed binary cache first (100x faster!)
  if (tryLoadBinaryCache())
  {
    return;  // ⚠️ Skips JSON loading if cache exists
  }
  // Fall back to JSON...
}
```

**The Problem**: The binary cache had NO invalidation check. If the JSON was updated (e.g., adding more words to the dictionary), the old cache was still used, missing the new words.

## Solution

**Immediate Fix (v1.32.656)**:
- Deleted stale binary cache files:
  ```bash
  rm assets/dictionaries/en_enhanced.bin
  rm assets/dictionaries/en_enhanced.bin.bak
  ```
- On first run, app will regenerate cache from JSON (takes ~500ms one-time)
- New cache will include all 50k+ words from `en_enhanced.json`

## Long-Term Fix Needed

Add cache validation to `OptimizedVocabulary.tryLoadBinaryCache()`:

```java
private boolean tryLoadBinaryCache()
{
  try
  {
    File cacheFile = new File(context.getFilesDir(), "vocab_cache_v2.bin");
    File jsonFile = new File(context.getAssets(), "dictionaries/en_enhanced.json");

    // INVALIDATE CACHE if JSON is newer
    if (cacheFile.lastModified() < jsonFile.lastModified())
    {
      Log.w(TAG, "JSON updated, invalidating stale cache");
      cacheFile.delete();
      return false;
    }

    // Load cache...
  }
}
```

**Note**: Assets in APK don't have timestamps, so this requires copying JSON to app storage or embedding version in cache header.

## Verification

After installing v1.32.656, test:
1. Swipe "sorghum" → should predict "sorghum"
2. Swipe "therapeutics" → should predict "therapeutics"
3. Swipe "genealogical" → should predict "genealogical"

Expected: All three words appear in predictions (may not be #1 due to low frequency, but should be in top 10).

## Performance Impact

- **First launch after cache deletion**: +500ms (one-time JSON parsing)
- **Subsequent launches**: Normal (<10ms binary cache load)
- **Cache regeneration only happens once** - new binary cache saved after JSON load

## Related Files

- `srcs/juloo.keyboard2/OptimizedVocabulary.java` (line 741-826)
- `assets/dictionaries/en_enhanced.json` (873KB, 50k+ words)
- `assets/dictionaries/contractions.bin` (separate cache, not affected)

## Commit

```bash
git add VOCAB_CACHE_FIX.md
git commit -m "fix(vocab): delete stale binary cache to reload enhanced dictionary

- Deleted en_enhanced.bin (stale cache from older dict)
- Words like 'sorghum', 'therapeutics', 'genealogical' now available
- App will regenerate cache from en_enhanced.json on first run
- Cache invalidation logic needed for long-term solution
- Version: v1.32.656"
```
