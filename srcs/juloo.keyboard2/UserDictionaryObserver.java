package juloo.keyboard2;

import android.content.Context;
import android.content.SharedPreferences;
import android.database.ContentObserver;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.provider.UserDictionary;
import android.util.Log;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

/**
 * Observes changes to UserDictionary and custom words, providing incremental updates.
 *
 * OPTIMIZATION: Eliminates need for periodic reload checks
 * - ContentObserver detects UserDictionary changes immediately
 * - SharedPreferences listener detects custom word changes
 * - Caches data to avoid repeated content provider queries and JSON parsing
 * - Provides incremental word sets (added/removed) for efficient updates
 *
 * Usage:
 * <pre>
 * UserDictionaryObserver observer = new UserDictionaryObserver(context);
 * observer.setChangeListener(new UserDictionaryObserver.ChangeListener() {
 *   @Override
 *   public void onUserDictionaryChanged(Set<String> addedWords, Set<String> removedWords) {
 *     // Update predictor incrementally
 *   }
 *
 *   @Override
 *   public void onCustomWordsChanged(Map<String, Integer> addedWords,
 *                                    Set<String> removedWords) {
 *     // Update predictor incrementally
 *   }
 * });
 * observer.start();
 *
 * // When done:
 * observer.stop();
 * </pre>
 */
public class UserDictionaryObserver extends ContentObserver
{
  private static final String TAG = "UserDictionaryObserver";

  private final Context _context;
  private final Handler _handler;

  // Cached user dictionary words (for detecting changes)
  private final Map<String, Integer> _cachedUserWords = new HashMap<>();

  // Cached custom words from SharedPreferences
  private final Map<String, Integer> _cachedCustomWords = new HashMap<>();

  // SharedPreferences listener
  private final SharedPreferences.OnSharedPreferenceChangeListener _prefsListener;

  // Change notification listener
  private ChangeListener _changeListener;

  /**
   * Listener interface for dictionary change events.
   */
  public interface ChangeListener
  {
    /**
     * Called when UserDictionary words are added or removed.
     *
     * @param addedWords Words added to UserDictionary (word -> frequency)
     * @param removedWords Words removed from UserDictionary
     */
    void onUserDictionaryChanged(Map<String, Integer> addedWords, Set<String> removedWords);

    /**
     * Called when custom words are added, removed, or modified.
     *
     * @param addedOrModified Words added or with frequency changed (word -> frequency)
     * @param removed Words removed from custom dictionary
     */
    void onCustomWordsChanged(Map<String, Integer> addedOrModified, Set<String> removed);
  }

  public UserDictionaryObserver(Context context)
  {
    super(new Handler(Looper.getMainLooper()));
    _context = context;
    _handler = new Handler(Looper.getMainLooper());

    // Create SharedPreferences listener
    _prefsListener = (prefs, key) -> {
      if ("custom_words".equals(key))
      {
        Log.d(TAG, "Custom words changed in SharedPreferences");
        checkCustomWordsChanges();
      }
    };
  }

  /**
   * Set the change listener for dictionary updates.
   */
  public void setChangeListener(ChangeListener listener)
  {
    _changeListener = listener;
  }

  /**
   * Start observing dictionary changes.
   * Registers ContentObserver and SharedPreferences listener.
   */
  public void start()
  {
    // Register ContentObserver for UserDictionary
    _context.getContentResolver().registerContentObserver(
      UserDictionary.Words.CONTENT_URI,
      true,
      this
    );

    // Register SharedPreferences listener
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(_context);
    prefs.registerOnSharedPreferenceChangeListener(_prefsListener);

    // Initial load of cached data
    loadUserDictionaryCache();
    loadCustomWordsCache();

    Log.d(TAG, "Started observing dictionary changes");
  }

  /**
   * Stop observing dictionary changes.
   * Unregisters all observers and listeners.
   */
  public void stop()
  {
    _context.getContentResolver().unregisterContentObserver(this);

    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(_context);
    prefs.unregisterOnSharedPreferenceChangeListener(_prefsListener);

    Log.d(TAG, "Stopped observing dictionary changes");
  }

  /**
   * Called when UserDictionary content changes.
   */
  @Override
  public void onChange(boolean selfChange, Uri uri)
  {
    Log.d(TAG, "UserDictionary changed: " + uri);
    checkUserDictionaryChanges();
  }

  /**
   * Load UserDictionary into cache.
   */
  private void loadUserDictionaryCache()
  {
    _cachedUserWords.clear();

    try
    {
      android.database.Cursor cursor = _context.getContentResolver().query(
        UserDictionary.Words.CONTENT_URI,
        new String[]{
          UserDictionary.Words.WORD,
          UserDictionary.Words.FREQUENCY
        },
        null,
        null,
        null
      );

      if (cursor != null)
      {
        int wordIndex = cursor.getColumnIndex(UserDictionary.Words.WORD);
        int freqIndex = cursor.getColumnIndex(UserDictionary.Words.FREQUENCY);

        while (cursor.moveToNext())
        {
          String word = cursor.getString(wordIndex).toLowerCase();
          int frequency = (freqIndex >= 0) ? cursor.getInt(freqIndex) : 1000;
          _cachedUserWords.put(word, frequency);
        }

        cursor.close();
        Log.d(TAG, "Loaded " + _cachedUserWords.size() + " user dictionary words into cache");
      }
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to load user dictionary cache", e);
    }
  }

  /**
   * Load custom words from SharedPreferences into cache.
   */
  private void loadCustomWordsCache()
  {
    _cachedCustomWords.clear();

    try
    {
      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(_context);
      String customWordsJson = prefs.getString("custom_words", "{}");

      if (!customWordsJson.equals("{}"))
      {
        org.json.JSONObject jsonObj = new org.json.JSONObject(customWordsJson);
        java.util.Iterator<String> keys = jsonObj.keys();

        while (keys.hasNext())
        {
          String word = keys.next().toLowerCase();
          int frequency = jsonObj.optInt(word, 1000);
          _cachedCustomWords.put(word, frequency);
        }

        Log.d(TAG, "Loaded " + _cachedCustomWords.size() + " custom words into cache");
      }
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to load custom words cache", e);
    }
  }

  /**
   * Check for UserDictionary changes and notify listener.
   */
  private void checkUserDictionaryChanges()
  {
    try
    {
      Map<String, Integer> currentWords = new HashMap<>();

      android.database.Cursor cursor = _context.getContentResolver().query(
        UserDictionary.Words.CONTENT_URI,
        new String[]{
          UserDictionary.Words.WORD,
          UserDictionary.Words.FREQUENCY
        },
        null,
        null,
        null
      );

      if (cursor != null)
      {
        int wordIndex = cursor.getColumnIndex(UserDictionary.Words.WORD);
        int freqIndex = cursor.getColumnIndex(UserDictionary.Words.FREQUENCY);

        while (cursor.moveToNext())
        {
          String word = cursor.getString(wordIndex).toLowerCase();
          int frequency = (freqIndex >= 0) ? cursor.getInt(freqIndex) : 1000;
          currentWords.put(word, frequency);
        }

        cursor.close();
      }

      // Compute differences
      Map<String, Integer> addedWords = new HashMap<>();
      Set<String> removedWords = new HashSet<>();

      // Find added words
      for (Map.Entry<String, Integer> entry : currentWords.entrySet())
      {
        if (!_cachedUserWords.containsKey(entry.getKey()))
        {
          addedWords.put(entry.getKey(), entry.getValue());
        }
      }

      // Find removed words
      for (String word : _cachedUserWords.keySet())
      {
        if (!currentWords.containsKey(word))
        {
          removedWords.add(word);
        }
      }

      // Update cache
      _cachedUserWords.clear();
      _cachedUserWords.putAll(currentWords);

      // Notify listener if there are changes
      if (!addedWords.isEmpty() || !removedWords.isEmpty())
      {
        Log.i(TAG, String.format("UserDictionary changed: +%d words, -%d words",
          addedWords.size(), removedWords.size()));

        if (_changeListener != null)
        {
          _changeListener.onUserDictionaryChanged(addedWords, removedWords);
        }
      }
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to check user dictionary changes", e);
    }
  }

  /**
   * Check for custom words changes and notify listener.
   */
  private void checkCustomWordsChanges()
  {
    try
    {
      Map<String, Integer> currentWords = new HashMap<>();

      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(_context);
      String customWordsJson = prefs.getString("custom_words", "{}");

      if (!customWordsJson.equals("{}"))
      {
        org.json.JSONObject jsonObj = new org.json.JSONObject(customWordsJson);
        java.util.Iterator<String> keys = jsonObj.keys();

        while (keys.hasNext())
        {
          String word = keys.next().toLowerCase();
          int frequency = jsonObj.optInt(word, 1000);
          currentWords.put(word, frequency);
        }
      }

      // Compute differences
      Map<String, Integer> addedOrModified = new HashMap<>();
      Set<String> removed = new HashSet<>();

      // Find added or modified words
      for (Map.Entry<String, Integer> entry : currentWords.entrySet())
      {
        Integer cachedFreq = _cachedCustomWords.get(entry.getKey());
        if (cachedFreq == null || !cachedFreq.equals(entry.getValue()))
        {
          addedOrModified.put(entry.getKey(), entry.getValue());
        }
      }

      // Find removed words
      for (String word : _cachedCustomWords.keySet())
      {
        if (!currentWords.containsKey(word))
        {
          removed.add(word);
        }
      }

      // Update cache
      _cachedCustomWords.clear();
      _cachedCustomWords.putAll(currentWords);

      // Notify listener if there are changes
      if (!addedOrModified.isEmpty() || !removed.isEmpty())
      {
        Log.i(TAG, String.format("Custom words changed: +%d words, -%d words",
          addedOrModified.size(), removed.size()));

        if (_changeListener != null)
        {
          _changeListener.onCustomWordsChanged(addedOrModified, removed);
        }
      }
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to check custom words changes", e);
    }
  }

  /**
   * Get current cached user dictionary words.
   */
  public Map<String, Integer> getCachedUserWords()
  {
    return new HashMap<>(_cachedUserWords);
  }

  /**
   * Get current cached custom words.
   */
  public Map<String, Integer> getCachedCustomWords()
  {
    return new HashMap<>(_cachedCustomWords);
  }
}
