package juloo.keyboard2;

import android.graphics.PointF;
import android.util.Log;
import java.util.ArrayList;
import java.util.List;

/**
 * Processes swipe trajectories for neural network input
 * CRITICAL FIX: Matches working cleverkeys implementation exactly
 * - Pads both coordinates AND nearestKeys to 150
 * - Uses integer token indices (not characters)
 * - Repeats last key for padding (not PAD tokens)
 * - Filters duplicate starting points
 */
public class SwipeTrajectoryProcessor
{
  private static final String TAG = "SwipeTrajectoryProcessor";

  // Keyboard layout for nearest key detection
  private java.util.Map<Character, PointF> _keyPositions;
  public float _keyboardWidth = 1.0f;
  public float _keyboardHeight = 1.0f;

  // Resampling configuration
  private SwipeResampler.ResamplingMode _resamplingMode = SwipeResampler.ResamplingMode.TRUNCATE;

  public SwipeTrajectoryProcessor()
  {
    // Log.d(TAG, "SwipeTrajectoryProcessor initialized");
  }

  /**
   * Set keyboard dimensions and key positions
   */
  public void setKeyboardLayout(java.util.Map<Character, PointF> keyPositions,
    float width, float height)
  {
    _keyPositions = keyPositions;
    _keyboardWidth = width;
    _keyboardHeight = height;

    // Log.d(TAG, String.format("Keyboard layout set: %.0fx%.0f with %d keys",
      // width, height, keyPositions != null ? keyPositions.size() : 0));
  }

  /**
   * Set resampling mode for trajectory processing
   */
  public void setResamplingMode(SwipeResampler.ResamplingMode mode)
  {
    _resamplingMode = mode;
    Log.d(TAG, "Resampling mode set to: " + mode);
  }

  /**
   * Extract trajectory features - MATCHES WORKING CLEVERKEYS
   * Takes SwipeInput for compatibility but processes like cleverkeys
   */
  public TrajectoryFeatures extractFeatures(SwipeInput input, int maxSequenceLength)
  {
    List<PointF> coordinates = input.coordinates;
    List<Long> timestamps = input.timestamps;

    if (coordinates == null || coordinates.isEmpty())
    {
      return new TrajectoryFeatures();
    }

    // Log.d(TAG, String.format("🔬 Extracting features from %d raw points", coordinates.size()));

    // 1. Filter duplicate starting points (FIX #34 from cleverkeys)
    List<PointF> filteredCoords = filterDuplicateStartingPoints(coordinates);
    if (filteredCoords.size() < coordinates.size()) {
      // Log.d(TAG, String.format("🔧 Filtered %d duplicate starting points (%d → %d)",
        // coordinates.size() - filteredCoords.size(), coordinates.size(), filteredCoords.size()));
    }

    // 2. Normalize coordinates FIRST (0-1 range) - matches cleverkeys
    List<PointF> normalizedCoords = normalizeCoordinates(filteredCoords);

    // 3. Detect nearest keys from filtered, un-normalized coordinates
    // CRITICAL: Returns integer token indices, not characters!
    List<Integer> nearestKeys = detectNearestKeys(filteredCoords);

    // 4. Apply resampling if sequence exceeds maxSequenceLength
    List<PointF> processedCoords = normalizedCoords;
    List<Integer> processedKeys = nearestKeys;

    if (normalizedCoords.size() > maxSequenceLength && _resamplingMode != SwipeResampler.ResamplingMode.TRUNCATE)
    {
      // Convert to 2D array for resampling (only x,y for coordinate resampling)
      float[][] coordArray = new float[normalizedCoords.size()][2];
      for (int i = 0; i < normalizedCoords.size(); i++)
      {
        coordArray[i][0] = normalizedCoords.get(i).x;
        coordArray[i][1] = normalizedCoords.get(i).y;
      }

      // Resample coordinates
      float[][] resampledArray = SwipeResampler.resample(coordArray, maxSequenceLength, _resamplingMode);

      // Convert back to PointF list
      processedCoords = new ArrayList<>();
      for (float[] point : resampledArray)
      {
        processedCoords.add(new PointF(point[0], point[1]));
      }

      // For nearest keys, use DISCARD mode (makes most sense for discrete values)
      int[][] keyArray = new int[nearestKeys.size()][1];
      for (int i = 0; i < nearestKeys.size(); i++)
      {
        keyArray[i][0] = nearestKeys.get(i);
      }

      // Convert to float for resampling, then back
      float[][] keyFloatArray = new float[keyArray.length][1];
      for (int i = 0; i < keyArray.length; i++)
      {
        keyFloatArray[i][0] = (float)keyArray[i][0];
      }

      float[][] resampledKeys = SwipeResampler.resample(keyFloatArray, maxSequenceLength, SwipeResampler.ResamplingMode.DISCARD);

      processedKeys = new ArrayList<>();
      for (float[] key : resampledKeys)
      {
        processedKeys.add((int)key[0]);
      }

      // Only log if actually resampling occurred (performance: avoid string formatting when not needed)
      if (android.util.Log.isLoggable(TAG, android.util.Log.DEBUG))
      {
        Log.d(TAG, String.format("🔄 Resampled trajectory: %d → %d points (mode: %s)",
          normalizedCoords.size(), maxSequenceLength, _resamplingMode));
      }
    }

    // 5. Calculate velocities and accelerations on ACTUAL trajectory (before padding)
    // CRITICAL: Training calculates features first, then pads feature array with zeros
    // If we pad coordinates first, we get velocity spikes at padding boundary!
    int actualLength = processedCoords.size();
    List<TrajectoryPoint> points = new ArrayList<>();

    for (int i = 0; i < actualLength; i++)
    {
      TrajectoryPoint point = new TrajectoryPoint();
      point.x = processedCoords.get(i).x;
      point.y = processedCoords.get(i).y;

      if (i == 0) {
        point.vx = 0.0f;
        point.vy = 0.0f;
        point.ax = 0.0f;
        point.ay = 0.0f;
      } else {
        TrajectoryPoint prev = points.get(i - 1);
        point.vx = point.x - prev.x;
        point.vy = point.y - prev.y;

        if (i == 1) {
          point.ax = 0.0f;
          point.ay = 0.0f;
        } else {
          point.ax = point.vx - prev.vx;
          point.ay = point.vy - prev.vy;
        }
      }

      points.add(point);
    }

    // 6. Truncate or pad features to maxSequenceLength
    // Training: traj_features = np.pad(traj_features, ((0, pad_len), (0, 0)), mode="constant")
    if (points.size() > maxSequenceLength) {
      // Truncate
      points = new ArrayList<>(points.subList(0, maxSequenceLength));
    } else {
      // Pad with zeros [0, 0, 0, 0, 0, 0]
      while (points.size() < maxSequenceLength) {
        TrajectoryPoint zeroPadding = new TrajectoryPoint();
        zeroPadding.x = 0.0f;
        zeroPadding.y = 0.0f;
        zeroPadding.vx = 0.0f;
        zeroPadding.vy = 0.0f;
        zeroPadding.ax = 0.0f;
        zeroPadding.ay = 0.0f;
        points.add(zeroPadding);
      }
    }

    // 7. Truncate or pad nearest_keys with PAD token (0)
    // Training: nearest_keys = nearest_keys + [self.tokenizer.pad_idx] * pad_len
    List<Integer> finalNearestKeys;
    if (processedKeys.size() >= maxSequenceLength) {
      finalNearestKeys = processedKeys.subList(0, maxSequenceLength);
    } else {
      finalNearestKeys = new ArrayList<>(processedKeys);
      while (finalNearestKeys.size() < maxSequenceLength) {
        finalNearestKeys.add(0);  // PAD token
      }
    }

    // Verification logging (first 3 points)
    if (!points.isEmpty()) {
      // Log.d(TAG, "🔬 Feature calculation (first 3 points):");
      for (int i = 0; i < Math.min(3, points.size()); i++) {
        TrajectoryPoint p = points.get(i);
        int key = finalNearestKeys.get(i);
        // Log.d(TAG, String.format("   Point[%d]: x=%.4f, y=%.4f, vx=%.4f, vy=%.4f, ax=%.4f, ay=%.4f, nearest_key=%d",
          // i, p.x, p.y, p.vx, p.vy, p.ax, p.ay, key));
      }
    }

    TrajectoryFeatures features = new TrajectoryFeatures();
    features.normalizedPoints = points;
    features.nearestKeys = finalNearestKeys;  // Now integer token indices!
    features.actualLength = Math.min(filteredCoords.size(), maxSequenceLength);

    // Log.d(TAG, String.format("✅ Extracted features: %d points, %d keys (both padded to %d)",
      // points.size(), finalNearestKeys.size(), maxSequenceLength));

    return features;
  }

  /**
   * Filter duplicate starting points to prevent zero velocity (FIX #34 from cleverkeys)
   * Android may report same coordinates multiple times before finger movement detected
   */
  private List<PointF> filterDuplicateStartingPoints(List<PointF> coordinates)
  {
    if (coordinates.isEmpty()) return coordinates;

    float threshold = 1f; // 1 pixel tolerance
    List<PointF> filtered = new ArrayList<>();
    filtered.add(coordinates.get(0));

    // Skip consecutive duplicates at the start
    int i = 1;
    while (i < coordinates.size()) {
      PointF prev = filtered.get(filtered.size() - 1);
      PointF curr = coordinates.get(i);

      float dx = Math.abs(curr.x - prev.x);
      float dy = Math.abs(curr.y - prev.y);

      // If this point is different, keep it and all remaining points
      if (dx > threshold || dy > threshold) {
        filtered.addAll(coordinates.subList(i, coordinates.size()));
        break;
      }
      i++;
    }

    return filtered;
  }

  /**
   * Normalize coordinates to [0, 1] range
   */
  private List<PointF> normalizeCoordinates(List<PointF> coordinates)
  {
    List<PointF> normalized = new ArrayList<>();
    for (PointF point : coordinates) {
      float x = (point.x / _keyboardWidth);
      float y = (point.y / _keyboardHeight);
      // Clamp to [0,1]
      x = Math.max(0f, Math.min(1f, x));
      y = Math.max(0f, Math.min(1f, y));
      normalized.add(new PointF(x, y));
    }
    return normalized;
  }

  /**
   * Detect nearest key for each coordinate using real keyboard layout
   * CRITICAL: Returns integer token indices (4-29 for a-z), NOT characters!
   */
  private List<Integer> detectNearestKeys(List<PointF> coordinates)
  {
    List<Integer> nearestKeys = new ArrayList<>();
    StringBuilder debugKeySeq = new StringBuilder();
    char lastDebugChar = '\0';

    for (PointF point : coordinates)
    {
      // ALWAYS use Python KeyboardGrid for nearest key detection
      // The model was trained on this specific grid layout, NOT real keyboard positions
      // Using real positions causes key mismatches (e.g., 'x' detected as 'd')
      int tokenIndex = detectKeyFromQwertyGrid(point);
      nearestKeys.add(tokenIndex);
      // Convert back to char for debug display
      char debugChar = tokenIndex >= 4 && tokenIndex <= 29 ? (char)('a' + (tokenIndex - 4)) : '?';
      if (debugChar != lastDebugChar) {
        debugKeySeq.append(debugChar);
        lastDebugChar = debugChar;
      }
    }

    // Log the deduplicated key sequence detected from trajectory
    Log.d(TAG, String.format("Neural key detection: \"%s\" (deduplicated from %d trajectory points)",
        debugKeySeq.toString(), coordinates.size()));

    return nearestKeys;
  }

  /**
   * Find nearest key using real key positions
   */
  private char findNearestKey(PointF point)
  {
    char nearestKey = 'a';
    float minDistance = Float.MAX_VALUE;

    for (java.util.Map.Entry<Character, PointF> entry : _keyPositions.entrySet())
    {
      PointF keyPos = entry.getValue();
      float dx = point.x - keyPos.x;
      float dy = point.y - keyPos.y;
      float distance = dx * dx + dy * dy;

      if (distance < minDistance)
      {
        minDistance = distance;
        nearestKey = entry.getKey();
      }
    }

    return nearestKey;
  }

  /**
   * Detect nearest key using grid-based approach with QWERTY layout
   * MUST MATCH Python KeyboardGrid exactly:
   * - 3 rows (height = 1/3 each)
   * - key_w = 0.1
   * - row offsets: top=0.0, mid=0.05, bot=0.15
   */
  private int detectKeyFromQwertyGrid(PointF point)
  {
    // QWERTY layout rows
    String row0 = "qwertyuiop";  // 10 keys, x starts at 0.0
    String row1 = "asdfghjkl";   // 9 keys, x starts at 0.05
    String row2 = "zxcvbnm";     // 7 keys, x starts at 0.15

    // Normalize to [0,1] - matches Python KeyboardGrid
    float nx = point.x / _keyboardWidth;
    float ny = point.y / _keyboardHeight;

    // Clamp to [0,1]
    nx = Math.max(0f, Math.min(1f, nx));
    ny = Math.max(0f, Math.min(1f, ny));

    // Grid dimensions matching Python
    float keyWidth = 0.1f;   // 1/10
    float rowHeight = 1.0f / 3.0f;  // 3 rows only!

    // Row offsets (absolute in normalized space)
    float row0_x0 = 0.0f;
    float row1_x0 = 0.05f;
    float row2_x0 = 0.15f;

    // Find nearest key by checking distance to each key center
    char nearestKey = 'a';
    float minDist = Float.MAX_VALUE;

    // Check row 0 (qwertyuiop)
    for (int i = 0; i < row0.length(); i++) {
      float cx = row0_x0 + i * keyWidth + keyWidth / 2.0f;
      float cy = 0.0f * rowHeight + rowHeight / 2.0f;
      float dist = (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy);
      if (dist < minDist) {
        minDist = dist;
        nearestKey = row0.charAt(i);
      }
    }

    // Check row 1 (asdfghjkl)
    for (int i = 0; i < row1.length(); i++) {
      float cx = row1_x0 + i * keyWidth + keyWidth / 2.0f;
      float cy = 1.0f * rowHeight + rowHeight / 2.0f;
      float dist = (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy);
      if (dist < minDist) {
        minDist = dist;
        nearestKey = row1.charAt(i);
      }
    }

    // Check row 2 (zxcvbnm)
    for (int i = 0; i < row2.length(); i++) {
      float cx = row2_x0 + i * keyWidth + keyWidth / 2.0f;
      float cy = 2.0f * rowHeight + rowHeight / 2.0f;
      float dist = (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy);
      if (dist < minDist) {
        minDist = dist;
        nearestKey = row2.charAt(i);
      }
    }

    return charToTokenIndex(nearestKey);
  }

  /**
   * Convert character to token index (a-z → 4-29, others → 0)
   */
  private int charToTokenIndex(char c)
  {
    if (c >= 'a' && c <= 'z') {
      return (c - 'a') + 4; // a=4, b=5, ..., z=29
    }
    return 0; // PAD_IDX for unknown characters
  }

  /**
   * Trajectory point with position, velocity, and acceleration
   */
  public static class TrajectoryPoint
  {
    public float x = 0.0f;
    public float y = 0.0f;
    public float vx = 0.0f;
    public float vy = 0.0f;
    public float ax = 0.0f;
    public float ay = 0.0f;

    public TrajectoryPoint() {}

    public TrajectoryPoint(TrajectoryPoint other)
    {
      this.x = other.x;
      this.y = other.y;
      this.vx = other.vx;
      this.vy = other.vy;
      this.ax = other.ax;
      this.ay = other.ay;
    }
  }

  /**
   * Complete feature set for neural network input
   * CRITICAL FIX: nearestKeys is now List<Integer> (token indices)
   */
  public static class TrajectoryFeatures
  {
    public List<TrajectoryPoint> normalizedPoints = new ArrayList<>();
    public List<Integer> nearestKeys = new ArrayList<>();  // Changed from List<Character>!
    public int actualLength = 0;
  }
}
