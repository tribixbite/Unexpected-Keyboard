package juloo.keyboard2;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.PointF;
import android.util.Log;
import android.view.WindowManager;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Helper class for neural engine layout configuration and CGR prediction display.
 *
 * This class centralizes logic for:
 * - Calculating dynamic keyboard dimensions based on user preferences
 * - Extracting key positions from keyboard layout for neural engine
 * - Setting up neural engine with real key positions
 * - Managing CGR (Continuous Gesture Recognition) prediction display
 * - Updating suggestion bar with swipe predictions (legacy methods)
 *
 * Responsibilities:
 * - Dynamic keyboard height calculation (orientation/foldable-aware)
 * - Key position extraction via reflection on Keyboard2View
 * - Neural engine configuration with accurate key positions
 * - CGR prediction integration with suggestion bar
 * - Legacy swipe prediction display methods
 *
 * NOT included (remains in Keyboard2):
 * - InputMethodService lifecycle methods
 * - View creation and management
 * - Configuration management
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.362).
 */
public class NeuralLayoutHelper
{
  private static final String TAG = "NeuralLayoutHelper";

  private final Context _context;
  private Config _config;
  private final PredictionCoordinator _predictionCoordinator;
  private Keyboard2View _keyboardView; // Non-final - updated when view changes
  private SuggestionBar _suggestionBar; // Non-final - updated when suggestion bar changes

  // Debug mode
  private boolean _debugMode = false;
  private DebugLogger _debugLogger;

  /**
   * Interface for sending debug logs.
   * Implemented by Keyboard2 to bridge to its sendDebugLog method.
   */
  public interface DebugLogger
  {
    void sendDebugLog(String message);
  }

  /**
   * Creates a new NeuralLayoutHelper.
   *
   * @param context Android context for resource access
   * @param config Configuration instance
   * @param predictionCoordinator Prediction coordinator for neural engine access
   */
  public NeuralLayoutHelper(Context context,
                           Config config,
                           PredictionCoordinator predictionCoordinator)
  {
    _context = context;
    _config = config;
    _predictionCoordinator = predictionCoordinator;
  }

  /**
   * Updates configuration.
   *
   * @param newConfig Updated configuration
   */
  public void setConfig(Config newConfig)
  {
    _config = newConfig;
  }

  /**
   * Sets the keyboard view reference.
   *
   * @param keyboardView Keyboard view for dimension and layout access
   */
  public void setKeyboardView(Keyboard2View keyboardView)
  {
    _keyboardView = keyboardView;
  }

  /**
   * Sets the suggestion bar reference.
   *
   * @param suggestionBar Suggestion bar for displaying predictions
   */
  public void setSuggestionBar(SuggestionBar suggestionBar)
  {
    _suggestionBar = suggestionBar;
  }

  /**
   * Sets debug mode and logger.
   *
   * @param enabled Whether debug mode is enabled
   * @param logger Debug logger implementation
   */
  public void setDebugMode(boolean enabled, DebugLogger logger)
  {
    _debugMode = enabled;
    _debugLogger = logger;
  }

  /**
   * Sends a debug log message if debug mode is enabled.
   */
  private void sendDebugLog(String message)
  {
    if (_debugMode && _debugLogger != null)
    {
      _debugLogger.sendDebugLog(message);
    }
  }

  /**
   * Calculate dynamic keyboard height based on user settings (like calibration page).
   * Supports orientation, foldable devices, and user height preferences.
   *
   * @return Calculated keyboard height in pixels
   */
  public float calculateDynamicKeyboardHeight()
  {
    try
    {
      // Get screen dimensions
      android.util.DisplayMetrics metrics = new android.util.DisplayMetrics();
      WindowManager wm = (WindowManager) _context.getSystemService(Context.WINDOW_SERVICE);
      wm.getDefaultDisplay().getMetrics(metrics);

      // Check foldable state
      FoldStateTracker foldTracker = new FoldStateTracker(_context);
      boolean foldableUnfolded = foldTracker.isUnfolded();

      // Check orientation
      boolean isLandscape = _context.getResources().getConfiguration().orientation ==
                            android.content.res.Configuration.ORIENTATION_LANDSCAPE;

      // Get user height preference (same logic as calibration)
      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(_context);
      int keyboardHeightPref;

      if (isLandscape)
      {
        String key = foldableUnfolded ? "keyboard_height_landscape_unfolded" : "keyboard_height_landscape";
        keyboardHeightPref = prefs.getInt(key, 50);
      }
      else
      {
        String key = foldableUnfolded ? "keyboard_height_unfolded" : "keyboard_height";
        keyboardHeightPref = prefs.getInt(key, 35);
      }

      // Calculate dynamic height
      float keyboardHeightPercent = keyboardHeightPref / 100.0f;
      float calculatedHeight = metrics.heightPixels * keyboardHeightPercent;

      return calculatedHeight;
    }
    catch (Exception e)
    {
      // Fallback to view height if available
      if (_keyboardView != null)
      {
        return _keyboardView.getHeight();
      }
      return 0;
    }
  }

  /**
   * Get user keyboard height percentage for logging.
   *
   * @return User's keyboard height preference as percentage
   */
  public int getUserKeyboardHeightPercent()
  {
    try
    {
      FoldStateTracker foldTracker = new FoldStateTracker(_context);
      boolean foldableUnfolded = foldTracker.isUnfolded();
      boolean isLandscape = _context.getResources().getConfiguration().orientation ==
                            android.content.res.Configuration.ORIENTATION_LANDSCAPE;

      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(_context);

      if (isLandscape)
      {
        String key = foldableUnfolded ? "keyboard_height_landscape_unfolded" : "keyboard_height_landscape";
        return prefs.getInt(key, 50);
      }
      else
      {
        String key = foldableUnfolded ? "keyboard_height_unfolded" : "keyboard_height";
        return prefs.getInt(key, 35);
      }
    }
    catch (Exception e)
    {
      return 35; // Default
    }
  }

  /**
   * Update swipe predictions by checking keyboard view for CGR results.
   */
  public void updateCGRPredictions()
  {
    if (_suggestionBar != null && _keyboardView != null)
    {
      List<String> cgrPredictions = _keyboardView.getCGRPredictions();
      if (!cgrPredictions.isEmpty())
      {
        _suggestionBar.setSuggestions(cgrPredictions);
      }
    }
  }

  /**
   * Check and update CGR predictions (call this periodically or on swipe events).
   */
  public void checkCGRPredictions()
  {
    if (_keyboardView != null && _suggestionBar != null)
    {
      // Enable always visible mode to prevent UI flickering
      _suggestionBar.setAlwaysVisible(true);

      List<String> cgrPredictions = _keyboardView.getCGRPredictions();
      boolean areFinal = _keyboardView.areCGRPredictionsFinal();

      if (!cgrPredictions.isEmpty())
      {
        _suggestionBar.setSuggestions(cgrPredictions);
      }
      else
      {
        // Show empty suggestions but keep bar visible
        _suggestionBar.setSuggestions(new ArrayList<>());
      }
    }
  }

  /**
   * Update swipe predictions in real-time during gesture (legacy method).
   *
   * @param predictions List of prediction strings
   */
  public void updateSwipePredictions(List<String> predictions)
  {
    if (_suggestionBar != null && predictions != null && !predictions.isEmpty())
    {
      _suggestionBar.setSuggestions(predictions);
    }
  }

  /**
   * Complete swipe predictions after gesture ends (legacy method).
   *
   * @param finalPredictions Final list of prediction strings
   */
  public void completeSwipePredictions(List<String> finalPredictions)
  {
    if (_suggestionBar != null && finalPredictions != null && !finalPredictions.isEmpty())
    {
      _suggestionBar.setSuggestions(finalPredictions);
    }
  }

  /**
   * Clear swipe predictions (legacy method).
   */
  public void clearSwipePredictions()
  {
    if (_suggestionBar != null)
    {
      // Don't actually clear - just show empty suggestions to keep bar visible
      _suggestionBar.setSuggestions(new ArrayList<>());
    }
  }

  /**
   * Extract key positions from keyboard layout and set them on neural engine.
   * CRITICAL for neural swipe typing - without this, key detection fails completely!
   */
  public void setNeuralKeyboardLayout()
  {
    if (_predictionCoordinator.getNeuralEngine() == null || _keyboardView == null)
    {
      Log.w(TAG, "Cannot set neural layout - engine or view is null");
      return;
    }

    Map<Character, PointF> keyPositions = extractKeyPositionsFromLayout();

    if (keyPositions != null && !keyPositions.isEmpty())
    {
      _predictionCoordinator.getNeuralEngine().setRealKeyPositions(keyPositions);
      Log.d(TAG, "Set " + keyPositions.size() + " key positions on neural engine");

      // Calculate QWERTY area bounds from key positions
      // Use q (top row) and m (bottom row) to determine the vertical extent
      if (keyPositions.containsKey('q') && keyPositions.containsKey('m'))
      {
        PointF qPos = keyPositions.get('q');  // Top row center
        PointF mPos = keyPositions.get('m');  // Bottom row center

        // Estimate row height from the distance between rows
        // q is in row 0, a is in row 1, z/m are in row 2
        PointF aPos = keyPositions.getOrDefault('a', qPos);
        float rowHeight = (mPos.y - qPos.y) / 2.0f;  // Approximate row height

        // QWERTY bounds: from top of first row to bottom of last row
        // qwertyTop = q.y - rowHeight/2 (top edge of row 0)
        // qwertyHeight = 3 * rowHeight (all 3 rows)
        float qwertyTop = qPos.y - rowHeight / 2.0f;
        float qwertyHeight = 3.0f * rowHeight;

        // Ensure qwertyTop is non-negative
        if (qwertyTop < 0) {
          qwertyHeight += qwertyTop;  // Reduce height by the negative amount
          qwertyTop = 0;
        }

        // Set bounds on neural engine
        _predictionCoordinator.getNeuralEngine().setQwertyAreaBounds(qwertyTop, qwertyHeight);
        Log.d(TAG, String.format("Set QWERTY bounds: top=%.0f, height=%.0f (q.y=%.0f, m.y=%.0f)",
            qwertyTop, qwertyHeight, qPos.y, mPos.y));

        // Touch Y-offset for fat finger compensation
        // DISABLED (v1.32.467): The 37% offset was too aggressive and may have caused issues
        // with top row key detection. Setting to 0 to isolate QWERTY bounds fix.
        // TODO: Re-enable with smaller offset (10-15%) after verifying bounds work correctly
        float touchYOffset = 0.0f;  // Was: rowHeight * 0.37f
        _predictionCoordinator.getNeuralEngine().setTouchYOffset(touchYOffset);
        Log.d(TAG, String.format("Touch Y-offset: %.0f pixels (DISABLED for debugging, row height=%.0f)",
            touchYOffset, rowHeight));

        // Debug output only when debug mode is active
        if (_debugMode)
        {
          sendDebugLog(String.format(">>> Neural engine: %d key positions set\n", keyPositions.size()));
          sendDebugLog(String.format(">>> QWERTY bounds: top=%.0f, height=%.0f\n", qwertyTop, qwertyHeight));
          sendDebugLog(String.format(">>> Touch Y-offset: %.0f px (fat finger compensation)\n", touchYOffset));
          PointF zPos = keyPositions.getOrDefault('z', mPos);
          sendDebugLog(String.format(">>> Samples: q=(%.0f,%.0f) a=(%.0f,%.0f) z=(%.0f,%.0f)\n",
              qPos.x, qPos.y, aPos.x, aPos.y, zPos.x, zPos.y));
        }
      }
      else
      {
        Log.w(TAG, "Cannot calculate QWERTY bounds - missing q or m key positions");
        // Debug output only when debug mode is active
        if (_debugMode)
        {
          sendDebugLog(String.format(">>> Neural engine: %d key positions set\n", keyPositions.size()));
        }
      }
    }
    else
    {
      Log.e(TAG, "Failed to extract key positions from layout");
    }
  }

  /**
   * Extract character key positions from the keyboard layout using reflection.
   * Returns a map of character -> center point (in pixels), or null on error.
   *
   * @return Map of character to center point, or null on error
   */
  private Map<Character, PointF> extractKeyPositionsFromLayout()
  {
    if (_keyboardView == null)
    {
      Log.w(TAG, "Cannot extract key positions - keyboardView is null");
      return null;
    }

    try
    {
      // Use reflection to access keyboard data from view
      java.lang.reflect.Field keyboardField = _keyboardView.getClass().getDeclaredField("_keyboard");
      keyboardField.setAccessible(true);
      KeyboardData keyboard = (KeyboardData) keyboardField.get(_keyboardView);

      if (keyboard == null)
      {
        Log.w(TAG, "Keyboard data is null after reflection");
        return null;
      }

      // Get view dimensions
      float keyboardWidth = _keyboardView.getWidth();
      float keyboardHeight = _keyboardView.getHeight();

      if (keyboardWidth == 0 || keyboardHeight == 0)
      {
        Log.w(TAG, "Keyboard dimensions are zero");
        return null;
      }

      // Calculate scale factors (layout units -> pixels)
      float scaleX = keyboardWidth / keyboard.keysWidth;
      float scaleY = keyboardHeight / keyboard.keysHeight;

      // Extract center positions of all character keys
      Map<Character, PointF> keyPositions = new HashMap<>();
      float currentY = 0;

      for (KeyboardData.Row row : keyboard.rows)
      {
        currentY += row.shift * scaleY;
        float centerY = currentY + (row.height * scaleY / 2.0f);
        float currentX = 0;

        for (KeyboardData.Key key : row.keys)
        {
          currentX += key.shift * scaleX;

          // Only process character keys
          if (key.keys != null && key.keys.length > 0 && key.keys[0] != null)
          {
            KeyValue kv = key.keys[0];
            if (kv.getKind() == KeyValue.Kind.Char)
            {
              char c = kv.getChar();
              float centerX = currentX + (key.width * scaleX / 2.0f);
              keyPositions.put(c, new PointF(centerX, centerY));
            }
          }

          currentX += key.width * scaleX;
        }

        currentY += row.height * scaleY;
      }

      return keyPositions;
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to extract key positions", e);
      return null;
    }
  }
}
