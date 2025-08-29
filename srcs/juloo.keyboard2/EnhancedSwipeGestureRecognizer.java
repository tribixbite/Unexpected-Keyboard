package juloo.keyboard2;

import android.content.Context;
import android.graphics.PointF;
import java.util.ArrayList;
import java.util.List;

/**
 * Enhanced Swipe Gesture Recognizer with CGR-based word prediction
 * Combines the existing key-based tracking with continuous gesture recognition
 * for real-time word predictions while preserving all existing functionality
 */
public class EnhancedSwipeGestureRecognizer
{
  // Original SwipeGestureRecognizer functionality
  private final List<PointF> _swipePath;
  private final List<KeyboardData.Key> _touchedKeys;
  private final List<Long> _timestamps;
  private boolean _isSwipeTyping;
  private long _startTime;
  private float _totalDistance;
  private KeyboardData.Key _lastKey;
  private LoopGestureDetector _loopDetector;
  
  // New CGR-based prediction system
  private final RealTimeSwipePredictor _swipePredictor;
  private boolean _cgrInitialized;
  
  // Original constants (preserved exactly)
  private static final float MIN_SWIPE_DISTANCE = 50.0f;
  private static final long MAX_POINT_INTERVAL_MS = 500;
  private static final float VELOCITY_THRESHOLD = 0.15f;
  private static final float MIN_POINT_DISTANCE = 25.0f;
  private static final long MIN_DWELL_TIME_MS = 30;
  
  // Callback interface for predictions
  public interface OnSwipePredictionListener
  {
    void onSwipePredictionUpdate(List<String> predictions);
    void onSwipePredictionComplete(List<String> finalPredictions);
    void onSwipePredictionCleared();
  }
  
  private OnSwipePredictionListener _predictionListener;
  
  public EnhancedSwipeGestureRecognizer()
  {
    // Initialize original functionality
    _swipePath = new ArrayList<>();
    _touchedKeys = new ArrayList<>();
    _timestamps = new ArrayList<>();
    _isSwipeTyping = false;
    _loopDetector = new LoopGestureDetector(100.0f, 80.0f);
    
    // Initialize new CGR system
    _swipePredictor = new RealTimeSwipePredictor();
    _cgrInitialized = false;
  }
  
  /**
   * Initialize the CGR prediction system
   */
  public void initializePredictionSystem(Context context)
  {
    if (!_cgrInitialized)
    {
      _swipePredictor.initialize(context);
      
      // Set up prediction callbacks
      _swipePredictor.setOnSwipePredictionListener(new RealTimeSwipePredictor.OnSwipePredictionListener()
      {
        @Override
        public void onSwipePredictionUpdate(List<String> predictions)
        {
          if (_predictionListener != null)
          {
            _predictionListener.onSwipePredictionUpdate(predictions);
          }
        }
        
        @Override
        public void onSwipePredictionComplete(List<String> finalPredictions)
        {
          if (_predictionListener != null)
          {
            _predictionListener.onSwipePredictionComplete(finalPredictions);
          }
        }
        
        @Override
        public void onSwipePredictionCleared()
        {
          if (_predictionListener != null)
          {
            _predictionListener.onSwipePredictionCleared();
          }
        }
      });
      
      _cgrInitialized = true;
    }
  }
  
  /**
   * Set prediction listener
   */
  public void setOnSwipePredictionListener(OnSwipePredictionListener listener)
  {
    _predictionListener = listener;
  }
  
  /**
   * Set keyboard dimensions for loop detection (preserved original functionality)
   */
  public void setKeyboardDimensions(float keyWidth, float keyHeight)
  {
    _loopDetector = new LoopGestureDetector(keyWidth, keyHeight);
  }
  
  /**
   * Start tracking a new swipe gesture (enhanced with CGR)
   */
  public void startSwipe(float x, float y, KeyboardData.Key key)
  {
    // Original functionality
    reset();
    _swipePath.add(new PointF(x, y));
    
    if (key != null && key.keys != null && key.keys.length > 0 && key.keys[0] != null && isAlphabeticKey(key.keys[0]))
    {
      _touchedKeys.add(key);
      _lastKey = key;
    }
    
    _startTime = System.currentTimeMillis();
    _timestamps.add(_startTime);
    _totalDistance = 0;
    
    // New CGR functionality - start gesture prediction
    if (_cgrInitialized)
    {
      _swipePredictor.onTouchBegan(x, y);
    }
  }
  
  /**
   * Add a point to the current swipe path (enhanced with CGR)
   */
  public void addPoint(float x, float y, KeyboardData.Key key)
  {
    if (_swipePath.isEmpty())
      return;
    
    // Original functionality (preserved exactly)
    long now = System.currentTimeMillis();
    long timeSinceStart = now - _startTime;
    
    if (!_isSwipeTyping && timeSinceStart > 100 && _totalDistance > MIN_SWIPE_DISTANCE)
    {
      _isSwipeTyping = shouldConsiderSwipeTyping();
    }
    
    PointF lastPoint = _swipePath.get(_swipePath.size() - 1);
    float dx = x - lastPoint.x;
    float dy = y - lastPoint.y;
    float distance = (float)Math.sqrt(dx * dx + dy * dy);
    
    if (distance < MIN_POINT_DISTANCE && _swipePath.size() > 1)
    {
      return;
    }
    
    _totalDistance += distance;
    _swipePath.add(new PointF(x, y));
    _timestamps.add(now);
    
    long timeDelta = _timestamps.size() > 0 ? 
                     now - _timestamps.get(_timestamps.size() - 1) : 0;
    float velocity = timeDelta > 0 ? distance / timeDelta : 0;
    
    if (key != null && key != _lastKey && key.keys != null && key.keys.length > 0 && key.keys[0] != null && isAlphabeticKey(key.keys[0]))
    {
      if (velocity > VELOCITY_THRESHOLD && timeDelta < MIN_DWELL_TIME_MS)
      {
        return;
      }
      
      boolean isDuplicate = false;
      int size = _touchedKeys.size();
      if (size >= 3)
      {
        for (int i = Math.max(0, size - 3); i < size; i++)
        {
          if (_touchedKeys.get(i) == key)
          {
            isDuplicate = true;
            break;
          }
        }
      }
      
      if (!isDuplicate && (distance > 35.0f || _touchedKeys.isEmpty()))
      {
        _touchedKeys.add(key);
        _lastKey = key;
      }
    }
    
    // New CGR functionality - update gesture prediction
    if (_cgrInitialized)
    {
      _swipePredictor.onTouchMoved(x, y);
    }
  }
  
  /**
   * End the swipe gesture and return the touched keys (enhanced with CGR)
   */
  public List<KeyboardData.Key> endSwipe()
  {
    // Log detailed swipe data for analysis (original functionality)
    logSwipeData();
    
    // New CGR functionality - complete gesture prediction
    if (_cgrInitialized && !_swipePath.isEmpty())
    {
      PointF lastPoint = _swipePath.get(_swipePath.size() - 1);
      _swipePredictor.onTouchEnded(lastPoint.x, lastPoint.y);
    }
    
    // Original return logic (preserved)
    if (_isSwipeTyping && _touchedKeys.size() >= 2)
    {
      return new ArrayList<>(_touchedKeys);
    }
    return null;
  }
  
  /**
   * Handle key press events (for clearing predictions)
   */
  public void onKeyPressed(KeyValue key)
  {
    if (_cgrInitialized)
    {
      _swipePredictor.onKeyPressed(key);
    }
  }
  
  /**
   * Select a prediction (called when user taps on suggestion)
   */
  public void selectPrediction(String word)
  {
    if (_cgrInitialized)
    {
      _swipePredictor.selectPrediction(word);
    }
  }
  
  /**
   * Get current predictions
   */
  public List<String> getCurrentPredictions()
  {
    if (_cgrInitialized)
    {
      return _swipePredictor.getCurrentPredictions();
    }
    return new ArrayList<>();
  }
  
  /**
   * Check if predictions are persisting
   */
  public boolean arePredictionsPersisting()
  {
    return _cgrInitialized && _swipePredictor.arePredictionsPersisting();
  }
  
  /**
   * Check if the current gesture should be considered swipe typing (original)
   */
  private boolean shouldConsiderSwipeTyping()
  {
    if (_touchedKeys.size() < 2)
      return false;
      
    for (KeyboardData.Key key : _touchedKeys)
    {
      if (key.keys == null || key.keys.length == 0 || key.keys[0] == null || !isAlphabeticKey(key.keys[0]))
        return false;
    }
    
    return true;
  }
  
  /**
   * Check if a KeyValue represents an alphabetic character (original)
   */
  private boolean isAlphabeticKey(KeyValue kv)
  {
    if (kv.getKind() != KeyValue.Kind.Char)
      return false;
    char c = kv.getChar();
    return Character.isLetter(c);
  }
  
  /**
   * Get the current swipe path for rendering (original)
   */
  public List<PointF> getSwipePath()
  {
    return new ArrayList<>(_swipePath);
  }
  
  /**
   * Check if currently in swipe typing mode (original)
   */
  public boolean isSwipeTyping()
  {
    return _isSwipeTyping;
  }
  
  /**
   * Reset the recognizer for a new gesture (enhanced)
   */
  public void reset()
  {
    // Original functionality
    _swipePath.clear();
    _touchedKeys.clear();
    _timestamps.clear();
    _isSwipeTyping = false;
    _lastKey = null;
    _totalDistance = 0;
    
    // New CGR functionality - only reset if not persisting predictions
    if (_cgrInitialized && !_swipePredictor.arePredictionsPersisting())
    {
      _swipePredictor.reset();
    }
  }
  
  /**
   * Get the sequence of characters from touched keys (original)
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
  
  /**
   * Get the enhanced key sequence with loop detection (original)
   */
  public String getEnhancedKeySequence()
  {
    String baseSequence = getKeySequence();
    if (baseSequence.isEmpty() || _swipePath.size() < 10)
      return baseSequence;
    
    List<LoopGestureDetector.Loop> loops = _loopDetector.detectLoops(_swipePath, _touchedKeys);
    
    if (loops.isEmpty())
      return baseSequence;
    
    String enhanced = _loopDetector.applyLoops(baseSequence, loops, _swipePath);
    return enhanced;
  }
  
  /**
   * Get timestamps for ML data collection (original)
   */
  public List<Long> getTimestamps()
  {
    return new ArrayList<>(_timestamps);
  }
  
  /**
   * Log comprehensive swipe data for analysis and debugging (original)
   */
  private void logSwipeData()
  {
    if (_swipePath.isEmpty())
      return;
    
    // Original logging code preserved
    StringBuilder pathStr = new StringBuilder("Path: ");
    for (int i = 0; i < Math.min(_swipePath.size(), 20); i++)
    {
      PointF p = _swipePath.get(i);
      pathStr.append(String.format("(%.0f,%.0f) ", p.x, p.y));
    }
    if (_swipePath.size() > 20)
      pathStr.append("... (" + (_swipePath.size() - 20) + " more points)");
    
    StringBuilder keysStr = new StringBuilder("Touched keys: ");
    for (KeyboardData.Key key : _touchedKeys)
    {
      if (key.keys[0] != null && key.keys[0].getKind() == KeyValue.Kind.Char)
      {
        keysStr.append(key.keys[0].getChar()).append(" ");
      }
    }
    
    if (_swipePath.size() >= 2)
    {
      float avgVelocity = _totalDistance / (System.currentTimeMillis() - _startTime);
      PointF start = _swipePath.get(0);
      PointF end = _swipePath.get(_swipePath.size() - 1);
      float directDistance = (float)Math.sqrt(
        Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2)
      );
      float straightnessRatio = directDistance / _totalDistance;
    }
  }
  
  /**
   * Check if CGR system is initialized
   */
  public boolean isPredictionSystemInitialized()
  {
    return _cgrInitialized && _swipePredictor.isInitialized();
  }
  
  /**
   * Clear all predictions (force clear including persistent ones)
   */
  public void clearAllPredictions()
  {
    if (_cgrInitialized)
    {
      _swipePredictor.clearPredictions();
    }
  }
}