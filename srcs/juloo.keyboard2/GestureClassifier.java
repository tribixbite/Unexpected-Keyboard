package juloo.keyboard2;

import android.content.Context;
import android.util.TypedValue;

/**
 * Unified gesture classifier that determines if a touch gesture is a TAP or SWIPE
 * Eliminates race conditions by providing single source of truth for gesture classification
 */
public class GestureClassifier
{
  public enum GestureType
  {
    TAP,
    SWIPE
  }

  /**
   * Data structure containing all gesture information needed for classification
   */
  public static class GestureData
  {
    public final boolean hasLeftStartingKey;
    public final float totalDistance;
    public final long timeElapsed;
    public final float keyWidth;

    public GestureData(boolean hasLeftStartingKey, float totalDistance, long timeElapsed, float keyWidth)
    {
      this.hasLeftStartingKey = hasLeftStartingKey;
      this.totalDistance = totalDistance;
      this.timeElapsed = timeElapsed;
      this.keyWidth = keyWidth;
    }
  }

  private final Context _context;
  private final long MAX_TAP_DURATION_MS = 150; // Maximum duration for a tap

  public GestureClassifier(Context context)
  {
    _context = context;
  }

  /**
   * Classify a gesture as TAP or SWIPE based on multiple criteria
   *
   * A gesture is a SWIPE if:
   * - User left the starting key AND
   * - (Distance exceeds minimum threshold OR time exceeds tap duration)
   *
   * Otherwise it's a TAP
   */
  public GestureType classify(GestureData gesture)
  {
    // Calculate dynamic threshold based on key size
    // Use half the key width as minimum swipe distance
    // Note: gesture.keyWidth is already in pixels (from key.width * _keyWidth)
    float minSwipeDistance = gesture.keyWidth / 2.0f;

    // Clear criteria: SWIPE if left starting key AND (distance OR time threshold met)
    if (gesture.hasLeftStartingKey &&
        (gesture.totalDistance >= minSwipeDistance ||
         gesture.timeElapsed > MAX_TAP_DURATION_MS))
    {
      return GestureType.SWIPE;
    }

    return GestureType.TAP;
  }

  /**
   * Convert dp to pixels using display density
   */
  private float dpToPx(float dp)
  {
    return TypedValue.applyDimension(
      TypedValue.COMPLEX_UNIT_DIP,
      dp,
      _context.getResources().getDisplayMetrics()
    );
  }
}
