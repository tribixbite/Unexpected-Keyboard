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

  // QWERTY area bounds for proper normalization (v1.32.463)
  // The model expects normalized coords over QWERTY keys only, not full view
  private float _qwertyAreaTop = 0.0f;      // Y offset where QWERTY starts (below suggestion bar, etc.)
  private float _qwertyAreaHeight = 0.0f;   // Height of QWERTY key area only

  // Touch Y-offset compensation (v1.32.466)
  // Users typically touch ~74 pixels above key center due to fat finger effect
  // This offset is added to raw Y coordinates before normalization
  private float _touchYOffset = 0.0f;

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
   * Set QWERTY area bounds for proper coordinate normalization.
   * The neural model expects coordinates normalized over the QWERTY key area only,
   * not the full keyboard view (which may include suggestion bar, number row, etc.)
   *
   * @param qwertyTop Y offset in pixels where QWERTY keys start
   * @param qwertyHeight Height in pixels of the QWERTY key area
   */
  public void setQwertyAreaBounds(float qwertyTop, float qwertyHeight)
  {
    _qwertyAreaTop = qwertyTop;
    _qwertyAreaHeight = qwertyHeight;
    Log.d(TAG, String.format("📐 QWERTY area bounds set: top=%.0f, height=%.0f (full kb height=%.0f)",
        qwertyTop, qwertyHeight, _keyboardHeight));
  }

  /**
   * Set touch Y-offset compensation for fat finger effect.
   * Users typically touch ~74 pixels above key centers due to finger geometry.
   * This offset is added to raw Y coordinates before normalization.
   *
   * @param offset Pixels to add to Y coordinate (positive = shift down toward key center)
   */
  public void setTouchYOffset(float offset)
  {
    _touchYOffset = offset;
    Log.d(TAG, String.format("📐 Touch Y-offset set: %.0f pixels", offset));
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

    // CRITICAL: Use raw coordinates directly - model was trained on raw data
    // DO NOT filter duplicates - it corrupts actual_length and changes what model sees
    // (v1.32.470 fix)

    // 1. Normalize coordinates (0-1 range)
    List<PointF> normalizedCoords = normalizeCoordinates(coordinates);

    // 2. Apply resampling if sequence exceeds maxSequenceLength
    List<PointF> processedCoords = normalizedCoords;
    List<Long> processedTimestamps = timestamps;

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

      // Resample timestamps as well to maintain correspondence
      processedTimestamps = new ArrayList<>();
      int origSize = timestamps.size();
      int newSize = processedCoords.size();
      for (int i = 0; i < newSize; i++) {
        int origIdx = (int) ((long) i * (origSize - 1) / (newSize - 1));
        processedTimestamps.add(timestamps.get(origIdx));
      }


      // Only log if actually resampling occurred (performance: avoid string formatting when not needed)
      if (android.util.Log.isLoggable(TAG, android.util.Log.DEBUG))
      {
        Log.d(TAG, String.format("🔄 Resampled trajectory: %d → %d points (mode: %s)",
          normalizedCoords.size(), maxSequenceLength, _resamplingMode));
      }
    }

    // 3. Detect nearest keys from FINAL processed coordinates (already normalized!)
    // CRITICAL: Must happen AFTER resampling to maintain point-key correspondence
    List<Integer> processedKeys = detectNearestKeys(processedCoords);

    // 4. Calculate velocities and accelerations using TrajectoryFeatureCalculator (v1.32.472)
    // CRITICAL: Must match Python training code exactly!
    // - Velocity = position_change / time_change
    // - Acceleration = velocity_change / time_change
    // - All clipped to [-10, 10]
    int actualLength = processedCoords.size();
    List<TrajectoryPoint> points = new ArrayList<>();

    // Use Kotlin TrajectoryFeatureCalculator for correct feature calculation
    // CRITICAL: Use processedCoords and processedTimestamps
    List<TrajectoryFeatureCalculator.FeaturePoint> featurePoints =
        TrajectoryFeatureCalculator.INSTANCE.calculateFeatures(processedCoords, processedTimestamps);

    // Convert to TrajectoryPoint list
    for (TrajectoryFeatureCalculator.FeaturePoint fp : featurePoints)
    {
      TrajectoryPoint point = new TrajectoryPoint();
      point.x = fp.getX();
      point.y = fp.getY();
      point.vx = fp.getVx();
      point.vy = fp.getVy();
      point.ax = fp.getAx();
      point.ay = fp.getAy();
      points.add(point);
    }

    // 5. Truncate or pad features to maxSequenceLength
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

    // 6. Truncate or pad nearest_keys with PAD token (0)
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
    features.actualLength = actualLength;  // From processedCoords.size()

    // Log.d(TAG, String.format("✅ Extracted features: %d points, %d keys (both padded to %d)",
      // points.size(), finalNearestKeys.size(), maxSequenceLength));

    return features;
  }

  /**
   * Normalize coordinates to [0, 1] range
   * Uses QWERTY area bounds if set, otherwise falls back to full keyboard dimensions
   */
  private List<PointF> normalizeCoordinates(List<PointF> coordinates)
  {
    // CRITICAL: Check if keyboard dimensions are set correctly
    // If still at default 1.0f, coordinates won't normalize properly
    if (_keyboardWidth <= 1.0f || _keyboardHeight <= 1.0f) {
      Log.w(TAG, String.format("⚠️ Keyboard dimensions not set! Using defaults: %.1f x %.1f",
          _keyboardWidth, _keyboardHeight));
      // Try to infer from coordinates
      float maxX = 0, maxY = 0;
      for (PointF p : coordinates) {
        maxX = Math.max(maxX, p.x);
        maxY = Math.max(maxY, p.y);
      }
      if (maxX > 1.0f) _keyboardWidth = maxX * 1.1f;  // Add 10% margin
      if (maxY > 1.0f) _keyboardHeight = maxY * 1.1f;
      Log.d(TAG, String.format("📐 Inferred keyboard size: %.0f x %.0f", _keyboardWidth, _keyboardHeight));
    }

    // Determine normalization parameters
    // If QWERTY area bounds are set, use them for Y normalization
    // This ensures Y coordinates span [0,1] for just the QWERTY key area
    float yTop = _qwertyAreaTop;
    float yHeight = _qwertyAreaHeight > 0 ? _qwertyAreaHeight : _keyboardHeight;
    boolean usingQwertyBounds = _qwertyAreaHeight > 0;

    List<PointF> normalized = new ArrayList<>();
    for (PointF point : coordinates) {
      float x = (point.x / _keyboardWidth);

      // Apply touch Y-offset compensation (fat finger effect)
      // Users typically touch ~74 pixels above key centers
      float adjustedY = point.y + _touchYOffset;

      // For Y: normalize over QWERTY area if bounds are set
      float y;
      if (usingQwertyBounds) {
        // Map QWERTY area [yTop, yTop+yHeight] to [0, 1]
        y = (adjustedY - yTop) / yHeight;
      } else {
        // Fall back to full keyboard height
        y = (adjustedY / _keyboardHeight);
      }

      // Clamp to [0,1]
      x = Math.max(0f, Math.min(1f, x));
      y = Math.max(0f, Math.min(1f, y));
      normalized.add(new PointF(x, y));
    }

    // Log normalization info for first and last points
    if (!coordinates.isEmpty() && !normalized.isEmpty()) {
      PointF rawFirst = coordinates.get(0);
      PointF normFirst = normalized.get(0);
      PointF rawLast = coordinates.get(coordinates.size() - 1);
      PointF normLast = normalized.get(normalized.size() - 1);

      if (usingQwertyBounds) {
        Log.d(TAG, String.format("📐 QWERTY NORMALIZATION: top=%.0f, height=%.0f (kb=%.0fx%.0f)",
            yTop, yHeight, _keyboardWidth, _keyboardHeight));
        Log.d(TAG, String.format("📐 RAW first=(%.0f,%.0f) last=(%.0f,%.0f)",
            rawFirst.x, rawFirst.y, rawLast.x, rawLast.y));
        Log.d(TAG, String.format("📐 NORMALIZED first=(%.3f,%.3f) last=(%.3f,%.3f)",
            normFirst.x, normFirst.y, normLast.x, normLast.y));
        // Show expected Y for z row (should be ~0.833)
        Log.d(TAG, String.format("📐 For z at pixel y=496: normalized y = %.3f", (496 - yTop) / yHeight));
      } else {
        Log.d(TAG, String.format("📐 Normalization: kb=%.0fx%.0f, raw=(%.0f,%.0f) → norm=(%.3f,%.3f)",
            _keyboardWidth, _keyboardHeight, rawFirst.x, rawFirst.y, normFirst.x, normFirst.y));
      }
    }

    return normalized;
  }

  /**
   * Detect nearest key for each coordinate using KeyboardGrid
   * CRITICAL: Returns integer token indices (4-29 for a-z), NOT characters!
   *
   * Input coordinates MUST be normalized to [0,1] range.
   */
  private List<Integer> detectNearestKeys(List<PointF> normalizedCoordinates)
  {
    List<Integer> nearestKeys = new ArrayList<>();
    StringBuilder debugKeySeq = new StringBuilder();
    char lastDebugChar = '\0';

    // Log first few coordinates for debugging
    if (!normalizedCoordinates.isEmpty()) {
      PointF first = normalizedCoordinates.get(0);
      PointF last = normalizedCoordinates.get(normalizedCoordinates.size() - 1);
      Log.d(TAG, String.format("🔍 Detecting keys from %d normalized points: first=(%.3f,%.3f) last=(%.3f,%.3f)",
          normalizedCoordinates.size(), first.x, first.y, last.x, last.y));
    }

    for (PointF point : normalizedCoordinates)
    {
      // Use Kotlin KeyboardGrid for nearest key detection
      int tokenIndex = KeyboardGrid.INSTANCE.getNearestKeyToken(point.x, point.y);
      nearestKeys.add(tokenIndex);

      // Convert back to char for debug display
      char debugChar = tokenIndex >= 4 && tokenIndex <= 29 ? (char)('a' + (tokenIndex - 4)) : '?';
      if (debugChar != lastDebugChar) {
        debugKeySeq.append(debugChar);
        lastDebugChar = debugChar;
      }
    }

    // Log the deduplicated key sequence detected from trajectory
    Log.d(TAG, String.format("🎯 DETECTED KEY SEQUENCE: \"%s\" (from %d points)",
        debugKeySeq.toString(), normalizedCoordinates.size()));

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
   *
   * NOTE: Input point must already be normalized to [0,1] range!
   */
  private int detectKeyFromQwertyGrid(PointF normalizedPoint)
  {
    // QWERTY layout rows
    String row0 = "qwertyuiop";  // 10 keys, x starts at 0.0
    String row1 = "asdfghjkl";   // 9 keys, x starts at 0.05
    String row2 = "zxcvbnm";     // 7 keys, x starts at 0.15

    // Input is already normalized - just clamp for safety
    float nx = Math.max(0f, Math.min(1f, normalizedPoint.x));
    float ny = Math.max(0f, Math.min(1f, normalizedPoint.y));

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
