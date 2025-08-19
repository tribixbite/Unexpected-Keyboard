package juloo.keyboard2;

import android.graphics.PointF;
import java.util.ArrayList;
import java.util.List;

/**
 * Recognizes swipe gestures across keyboard keys and tracks the path
 * for word prediction.
 */
public class SwipeGestureRecognizer
{
  private final List<PointF> _swipePath;
  private final List<KeyboardData.Key> _touchedKeys;
  private final List<Long> _timestamps;
  private boolean _isSwipeTyping;
  private long _startTime;
  private float _totalDistance;
  private KeyboardData.Key _lastKey;
  
  // Minimum distance to consider it a swipe typing gesture
  private static final float MIN_SWIPE_DISTANCE = 50.0f;
  // Maximum time between touch points to continue swipe
  private static final long MAX_POINT_INTERVAL_MS = 500;
  
  public SwipeGestureRecognizer()
  {
    _swipePath = new ArrayList<>();
    _touchedKeys = new ArrayList<>();
    _timestamps = new ArrayList<>();
    _isSwipeTyping = false;
  }
  
  /**
   * Start tracking a new swipe gesture
   */
  public void startSwipe(float x, float y, KeyboardData.Key key)
  {
    reset();
    _swipePath.add(new PointF(x, y));
    if (key != null && key.keys[0] != null && isAlphabeticKey(key.keys[0]))
    {
      _touchedKeys.add(key);
      _lastKey = key;
    }
    _startTime = System.currentTimeMillis();
    _timestamps.add(_startTime);
    _totalDistance = 0;
  }
  
  /**
   * Add a point to the current swipe path
   */
  public void addPoint(float x, float y, KeyboardData.Key key)
  {
    if (_swipePath.isEmpty())
      return;
      
    long now = System.currentTimeMillis();
    long timeSinceStart = now - _startTime;
    
    // Check if this should be considered swipe typing
    if (!_isSwipeTyping && timeSinceStart > 100 && _totalDistance > MIN_SWIPE_DISTANCE)
    {
      _isSwipeTyping = shouldConsiderSwipeTyping();
    }
    
    PointF lastPoint = _swipePath.get(_swipePath.size() - 1);
    float dx = x - lastPoint.x;
    float dy = y - lastPoint.y;
    float distance = (float)Math.sqrt(dx * dx + dy * dy);
    _totalDistance += distance;
    
    _swipePath.add(new PointF(x, y));
    _timestamps.add(now);
    
    // Add key if it's different from the last one and is alphabetic
    if (key != null && key != _lastKey && key.keys[0] != null && isAlphabeticKey(key.keys[0]))
    {
      _touchedKeys.add(key);
      _lastKey = key;
    }
  }
  
  /**
   * End the swipe gesture and return the touched keys if it was swipe typing
   */
  public List<KeyboardData.Key> endSwipe()
  {
    if (_isSwipeTyping && _touchedKeys.size() >= 2)
    {
      return new ArrayList<>(_touchedKeys);
    }
    return null;
  }
  
  /**
   * Check if the current gesture should be considered swipe typing
   */
  private boolean shouldConsiderSwipeTyping()
  {
    // Need at least 2 alphabetic keys
    if (_touchedKeys.size() < 2)
      return false;
      
    // Check if all touched keys are alphabetic
    for (KeyboardData.Key key : _touchedKeys)
    {
      if (key.keys[0] == null || !isAlphabeticKey(key.keys[0]))
        return false;
    }
    
    return true;
  }
  
  /**
   * Check if a KeyValue represents an alphabetic character
   */
  private boolean isAlphabeticKey(KeyValue kv)
  {
    if (kv.getKind() != KeyValue.Kind.Char)
      return false;
    char c = kv.getChar();
    return Character.isLetter(c);
  }
  
  /**
   * Get the current swipe path for rendering
   */
  public List<PointF> getSwipePath()
  {
    return new ArrayList<>(_swipePath);
  }
  
  /**
   * Check if currently in swipe typing mode
   */
  public boolean isSwipeTyping()
  {
    return _isSwipeTyping;
  }
  
  /**
   * Reset the recognizer for a new gesture
   */
  public void reset()
  {
    _swipePath.clear();
    _touchedKeys.clear();
    _timestamps.clear();
    _isSwipeTyping = false;
    _lastKey = null;
    _totalDistance = 0;
  }
  
  /**
   * Get the sequence of characters from touched keys
   */
  public String getKeySequence()
  {
    if (_touchedKeys.isEmpty())
      return "";
      
    StringBuilder sb = new StringBuilder();
    for (KeyboardData.Key key : _touchedKeys)
    {
      if (key.keys[0] != null && key.keys[0].getKind() == KeyValue.Kind.Char)
      {
        char c = key.keys[0].getChar();
        if (Character.isLetter(c))
          sb.append(c);
      }
    }
    return sb.toString();
  }
}
