package juloo.keyboard2;

import android.content.Context;
import android.content.res.Resources;
import android.util.DisplayMetrics;
import juloo.keyboard2.ml.SwipeMLData;
import juloo.keyboard2.ml.SwipeMLDataStore;

/**
 * Collects and stores ML training data from swipe gestures.
 *
 * This class centralizes logic for:
 * - Collecting ML data when user selects swipe predictions
 * - Copying trace points from temporary swipe data
 * - Copying registered keys from swipe data
 * - Storing ML data in the data store
 * - Handling normalization/denormalization of coordinates
 *
 * Responsibilities:
 * - Check if ML data collection should occur (was last input swipe?)
 * - Create SwipeMLData objects with correct dimensions
 * - Copy trace points and registered keys from current swipe
 * - Store collected data in ML data store
 *
 * NOT included (remains in Keyboard2):
 * - Retrieving current swipe data from InputCoordinator
 * - Accessing PredictionCoordinator and ML data store
 * - Context tracking (wasLastInputSwipe)
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.370).
 */
public class MLDataCollector
{
  private final Context _context;

  /**
   * Creates a new MLDataCollector.
   *
   * @param context Android context for accessing resources
   */
  public MLDataCollector(Context context)
  {
    _context = context;
  }

  /**
   * Collects and stores ML data from a swipe gesture when user selects a suggestion.
   *
   * @param word Selected word from suggestion
   * @param currentSwipeData Current swipe data containing trace points and registered keys
   * @param keyboardHeight Height of keyboard view for ML data
   * @param mlDataStore ML data store to save the data
   * @return true if data was collected and stored, false otherwise
   */
  public boolean collectAndStoreSwipeData(String word, SwipeMLData currentSwipeData,
                                         int keyboardHeight, SwipeMLDataStore mlDataStore)
  {
    if (currentSwipeData == null || mlDataStore == null)
    {
      return false;
    }

    try
    {
      // Strip "raw:" prefix before storing ML data
      String cleanWord = word.replaceAll("^raw:", "");

      // Create a new ML data object with the selected word
      DisplayMetrics metrics = _context.getResources().getDisplayMetrics();
      SwipeMLData mlData = new SwipeMLData(cleanWord, "user_selection",
                                          metrics.widthPixels, metrics.heightPixels,
                                          keyboardHeight);

      // Copy trace points from the temporary data
      for (SwipeMLData.TracePoint point : currentSwipeData.getTracePoints())
      {
        // Add points with their original normalized values and timestamps
        // Since they're already normalized, we need to denormalize then renormalize
        // to ensure proper storage
        float rawX = point.x * metrics.widthPixels;
        float rawY = point.y * metrics.heightPixels;
        // Reconstruct approximate timestamp (this is a limitation of the current design)
        long timestamp = System.currentTimeMillis() - 1000 + point.tDeltaMs;
        mlData.addRawPoint(rawX, rawY, timestamp);
      }

      // Copy registered keys
      for (String key : currentSwipeData.getRegisteredKeys())
      {
        mlData.addRegisteredKey(key);
      }

      // Store the ML data
      mlDataStore.storeSwipeData(mlData);
      return true;
    }
    catch (Exception e)
    {
      android.util.Log.e("MLDataCollector", "Error collecting ML data", e);
      return false;
    }
  }
}
