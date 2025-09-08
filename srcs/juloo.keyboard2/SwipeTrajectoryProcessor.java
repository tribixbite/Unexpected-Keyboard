package juloo.keyboard2;

import android.graphics.PointF;
import android.util.Log;
import java.util.ArrayList;
import java.util.List;

/**
 * Processes swipe trajectories for neural network input
 * Extracts features: position, velocity, acceleration, nearest keys
 * Matches the feature extraction pipeline from the web demo
 */
public class SwipeTrajectoryProcessor
{
  private static final String TAG = "SwipeTrajectoryProcessor";
  
  // Keyboard layout for nearest key detection
  private java.util.Map<Character, PointF> _keyPositions;
  private float _keyboardWidth = 1.0f;
  private float _keyboardHeight = 1.0f;
  
  public SwipeTrajectoryProcessor()
  {
    Log.d(TAG, "SwipeTrajectoryProcessor initialized");
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
    
    Log.d(TAG, String.format("Keyboard layout set: %dx%.0f with %d keys", 
      (int)width, height, keyPositions != null ? keyPositions.size() : 0));
  }
  
  /**
   * Extract trajectory features from swipe input
   */
  public TrajectoryFeatures extractFeatures(SwipeInput input, int maxSequenceLength)
  {
    List<PointF> rawPath = input.coordinates;
    if (rawPath == null || rawPath.isEmpty())
    {
      return new TrajectoryFeatures();
    }
    
    Log.d(TAG, String.format("Extracting features from %d raw points", rawPath.size()));
    
    // Normalize coordinates to [0,1] range
    List<TrajectoryPoint> normalizedPoints = normalizeTrajectory(rawPath, maxSequenceLength);
    
    // Calculate velocities and accelerations
    calculateDerivatives(normalizedPoints);
    
    // Find nearest keys for each point
    List<Character> nearestKeys = findNearestKeys(rawPath);
    
    TrajectoryFeatures features = new TrajectoryFeatures();
    features.normalizedPoints = normalizedPoints;
    features.nearestKeys = nearestKeys;
    features.actualLength = Math.min(rawPath.size(), maxSequenceLength);
    
    Log.d(TAG, String.format("Extracted features: %d points, %d keys", 
      normalizedPoints.size(), nearestKeys.size()));
    
    return features;
  }
  
  private List<TrajectoryPoint> normalizeTrajectory(List<PointF> rawPath, int maxLength)
  {
    List<TrajectoryPoint> normalized = new ArrayList<>();
    
    // Sample or interpolate to target length
    List<PointF> resampled = resamplePath(rawPath, Math.min(rawPath.size(), maxLength));
    
    // Normalize coordinates to [0,1]
    for (PointF point : resampled)
    {
      TrajectoryPoint normPoint = new TrajectoryPoint();
      normPoint.x = point.x / _keyboardWidth;
      normPoint.y = point.y / _keyboardHeight;
      
      // Clamp to [0,1] range
      normPoint.x = Math.max(0.0f, Math.min(1.0f, normPoint.x));
      normPoint.y = Math.max(0.0f, Math.min(1.0f, normPoint.y));
      
      normalized.add(normPoint);
    }
    
    // Pad to maxLength if necessary
    while (normalized.size() < maxLength)
    {
      // Pad with last point (or zeros if empty)
      TrajectoryPoint lastPoint = normalized.isEmpty() ? 
        new TrajectoryPoint() : normalized.get(normalized.size() - 1);
      normalized.add(new TrajectoryPoint(lastPoint));
    }
    
    return normalized;
  }
  
  private void calculateDerivatives(List<TrajectoryPoint> points)
  {
    // Calculate velocities (first derivative)
    for (int i = 1; i < points.size(); i++)
    {
      TrajectoryPoint curr = points.get(i);
      TrajectoryPoint prev = points.get(i - 1);
      
      curr.vx = curr.x - prev.x;
      curr.vy = curr.y - prev.y;
    }
    
    // Calculate accelerations (second derivative)
    for (int i = 2; i < points.size(); i++)
    {
      TrajectoryPoint curr = points.get(i);
      TrajectoryPoint prev = points.get(i - 1);
      
      curr.ax = curr.vx - prev.vx;
      curr.ay = curr.vy - prev.vy;
    }
  }
  
  private List<PointF> resamplePath(List<PointF> originalPath, int targetLength)
  {
    if (originalPath.size() <= targetLength)
    {
      return new ArrayList<>(originalPath);
    }
    
    // Simple downsampling by taking evenly spaced points
    List<PointF> resampled = new ArrayList<>();
    float step = (float)(originalPath.size() - 1) / (targetLength - 1);
    
    for (int i = 0; i < targetLength; i++)
    {
      int idx = Math.round(i * step);
      idx = Math.min(idx, originalPath.size() - 1);
      resampled.add(originalPath.get(idx));
    }
    
    return resampled;
  }
  
  private List<Character> findNearestKeys(List<PointF> path)
  {
    List<Character> nearestKeys = new ArrayList<>();
    
    if (_keyPositions == null || _keyPositions.isEmpty())
    {
      // Fallback: return 'a' for all points
      for (int i = 0; i < path.size(); i++)
      {
        nearestKeys.add('a');
      }
      return nearestKeys;
    }
    
    // Find nearest key for each point
    for (PointF point : path)
    {
      char nearestKey = findNearestKey(point);
      nearestKeys.add(nearestKey);
    }
    
    return nearestKeys;
  }
  
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
   */
  public static class TrajectoryFeatures
  {
    public List<TrajectoryPoint> normalizedPoints = new ArrayList<>();
    public List<Character> nearestKeys = new ArrayList<>();
    public int actualLength = 0;
  }
}