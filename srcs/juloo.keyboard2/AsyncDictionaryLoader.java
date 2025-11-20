package juloo.keyboard2;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Asynchronous dictionary loader with background thread execution.
 *
 * Ensures dictionary loading never blocks the main thread, providing
 * responsive UI during startup and language switches.
 *
 * OPTIMIZATION: Prevents UI freezes during dictionary loading
 * - Main thread remains responsive
 * - User receives feedback about loading state
 * - Predictions become available asynchronously
 *
 * Usage:
 * <pre>
 * AsyncDictionaryLoader loader = new AsyncDictionaryLoader();
 * loader.loadDictionaryAsync(context, language, new AsyncDictionaryLoader.LoadCallback() {
 *   @Override
 *   public void onLoadStarted(String language) {
 *     // Show loading indicator
 *   }
 *
 *   @Override
 *   public void onLoadComplete(Map<String, Integer> dictionary,
 *                              Map<String, Set<String>> prefixIndex) {
 *     // Hide loading indicator, enable predictions
 *   }
 *
 *   @Override
 *   public void onLoadFailed(String language, Exception error) {
 *     // Show error message
 *   }
 * });
 * </pre>
 */
public class AsyncDictionaryLoader
{
  private static final String TAG = "AsyncDictionaryLoader";

  // Single-threaded executor for sequential dictionary loading
  // (only one dictionary should load at a time)
  private static final ExecutorService EXECUTOR = Executors.newSingleThreadExecutor(r -> {
    Thread t = new Thread(r, "DictionaryLoader");
    t.setPriority(Thread.NORM_PRIORITY - 1); // Slightly lower priority
    return t;
  });

  // Handler for callbacks on main thread
  private final Handler _mainHandler = new Handler(Looper.getMainLooper());

  // Current loading task (for cancellation)
  private Future<?> _currentTask;

  /**
   * Callback interface for asynchronous dictionary loading.
   */
  public interface LoadCallback
  {
    /**
     * Called on main thread when loading starts.
     */
    void onLoadStarted(String language);

    /**
     * Called on main thread when loading completes successfully.
     *
     * @param dictionary Map of words to frequencies
     * @param prefixIndex Map of prefixes to matching words
     */
    void onLoadComplete(Map<String, Integer> dictionary,
                        Map<String, Set<String>> prefixIndex);

    /**
     * Called on main thread if loading fails.
     */
    void onLoadFailed(String language, Exception error);
  }

  /**
   * Load dictionary asynchronously on background thread.
   *
   * This method returns immediately. The callback will be invoked on the
   * main thread when loading completes or fails.
   *
   * @param context Android context for asset access
   * @param language Language code (e.g., "en")
   * @param callback Callback for load events (called on main thread)
   */
  public void loadDictionaryAsync(final Context context,
                                   final String language,
                                   final LoadCallback callback)
  {
    // Cancel any previous loading task
    if (_currentTask != null && !_currentTask.isDone())
    {
      _currentTask.cancel(true);
      Log.d(TAG, "Cancelled previous dictionary load");
    }

    // Notify on main thread that loading started
    _mainHandler.post(() -> callback.onLoadStarted(language));

    // Submit loading task to background thread
    _currentTask = EXECUTOR.submit(() -> {
      try
      {
        long startTime = System.currentTimeMillis();

        // Try binary format first (fast path)
        String binaryFilename = "dictionaries/" + language + "_enhanced.bin";
        final Map<String, Integer> dictionary = new java.util.HashMap<>();
        final Map<String, Set<String>> prefixIndex = new java.util.HashMap<>();

        boolean loadedBinary = BinaryDictionaryLoader.loadDictionaryWithPrefixIndex(
          context, binaryFilename, dictionary, prefixIndex);

        if (!loadedBinary)
        {
          // Fall back to JSON format (slow path)
          Log.d(TAG, "Binary dictionary not available, falling back to JSON");

          String jsonFilename = "dictionaries/" + language + "_enhanced.json";
          try
          {
            java.io.BufferedReader reader = new java.io.BufferedReader(
              new java.io.InputStreamReader(context.getAssets().open(jsonFilename)));
            StringBuilder jsonBuilder = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null)
            {
              jsonBuilder.append(line);
            }
            reader.close();

            // Parse JSON object
            org.json.JSONObject jsonDict = new org.json.JSONObject(jsonBuilder.toString());
            java.util.Iterator<String> keys = jsonDict.keys();
            while (keys.hasNext())
            {
              String word = keys.next().toLowerCase();
              int frequency = jsonDict.getInt(word);
              // Scale frequency to 100-10000 range
              int scaledFreq = 100 + (int)((frequency - 128) / 127.0 * 9900);
              dictionary.put(word, scaledFreq);
            }

            // Build prefix index
            for (String word : dictionary.keySet())
            {
              int maxLen = Math.min(3, word.length());
              for (int len = 1; len <= maxLen; len++)
              {
                String prefix = word.substring(0, len);
                prefixIndex.computeIfAbsent(prefix, k -> new java.util.HashSet<>()).add(word);
              }
            }

            Log.d(TAG, "Loaded JSON dictionary: " + jsonFilename);
          }
          catch (Exception e)
          {
            throw new RuntimeException("Failed to load dictionary: " + language, e);
          }
        }

        long loadTime = System.currentTimeMillis() - startTime;
        Log.i(TAG, String.format("Dictionary loaded in %dms on background thread: %d words, %d prefixes",
          loadTime, dictionary.size(), prefixIndex.size()));

        // Notify success on main thread
        _mainHandler.post(() -> callback.onLoadComplete(dictionary, prefixIndex));
      }
      catch (final Exception e)
      {
        Log.e(TAG, "Dictionary loading failed: " + language, e);
        // Notify failure on main thread
        _mainHandler.post(() -> callback.onLoadFailed(language, e));
      }
    });
  }

  /**
   * Cancel any ongoing dictionary load.
   */
  public void cancel()
  {
    if (_currentTask != null && !_currentTask.isDone())
    {
      _currentTask.cancel(true);
      Log.d(TAG, "Dictionary load cancelled");
    }
  }

  /**
   * Shutdown the executor service.
   * Should be called when the loader is no longer needed.
   */
  public static void shutdown()
  {
    EXECUTOR.shutdown();
    Log.d(TAG, "Dictionary loader executor shutdown");
  }
}
