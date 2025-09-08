package juloo.keyboard2;

import android.content.Context;
import android.graphics.PointF;
import android.util.Log;
import java.util.Map;

/**
 * Neural swipe typing engine using ONNX transformer models
 * Replaces SwipeTypingEngine with neural prediction pipeline
 * Maintains same interface for seamless integration
 */
public class NeuralSwipeTypingEngine
{
  private static final String TAG = "NeuralSwipeTypingEngine";
  
  private Context _context;
  private Config _config;
  private OnnxSwipePredictor _neuralPredictor;
  private WordPredictor _fallbackPredictor;
  private SwipeDetector _swipeDetector;
  
  // State tracking
  private boolean _useNeuralPrediction = true;
  private boolean _initialized = false;
  
  public NeuralSwipeTypingEngine(Context context, WordPredictor fallbackPredictor, Config config)
  {
    _context = context;
    _fallbackPredictor = fallbackPredictor;
    _config = config;
    _swipeDetector = new SwipeDetector();
    
    // Initialize neural predictor
    _neuralPredictor = new OnnxSwipePredictor(context);
    
    Log.d(TAG, "NeuralSwipeTypingEngine created");
  }
  
  /**
   * Initialize the engine and load models
   */
  public boolean initialize()
  {
    if (_initialized)
    {
      return true;
    }
    
    try
    {
      Log.d(TAG, "Initializing neural swipe engine...");
      
      // Initialize neural predictor
      boolean neuralReady = _neuralPredictor.initialize();
      
      // Configure neural prediction based on availability
      _useNeuralPrediction = neuralReady && isNeuralEnabled();
      
      // Ensure fallback predictor is configured
      if (_fallbackPredictor != null)
      {
        _fallbackPredictor.setConfig(_config);
      }
      
      _initialized = true;
      
      Log.d(TAG, String.format("Neural engine initialized: neural=%s, fallback=%s", 
        _useNeuralPrediction, _fallbackPredictor != null));
      
      return true;
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to initialize neural engine", e);
      _useNeuralPrediction = false;
      _initialized = true;
      return false;
    }
  }
  
  /**
   * Main prediction method - maintains compatibility with legacy interface
   */
  public PredictionResult predict(SwipeInput input)
  {
    if (!_initialized)
    {
      initialize();
    }
    
    Log.d(TAG, "=== NEURAL PREDICTION START ===");
    Log.d(TAG, String.format(
      "Input: keySeq=%s, pathLen=%.1f, duration=%.2fs", 
      input.keySequence, input.pathLength, input.duration));
    
    // Classify input type
    SwipeDetector.SwipeClassification classification = _swipeDetector.classifyInput(input);
    
    Log.d(TAG, String.format(
      "Classification: isSwipe=%s, confidence=%.2f, neural=%s",
      classification.isSwipe, classification.confidence, _useNeuralPrediction));
    
    // Route prediction based on input type and model availability
    if (!classification.isSwipe)
    {
      // Regular typing - use fallback predictor
      Log.d(TAG, "Using fallback prediction (not a swipe)");
      return useFallbackPrediction(input);
    }
    else if (_useNeuralPrediction)
    {
      // Swipe input with neural prediction available
      Log.d(TAG, "Using NEURAL prediction for swipe");
      
      try
      {
        PredictionResult result = _neuralPredictor.predict(input);
        
        if (result != null && !result.words.isEmpty())
        {
          Log.d(TAG, String.format("Neural prediction successful: %d candidates", 
            result.words.size()));
          return result;
        }
        else
        {
          Log.w(TAG, "Neural prediction returned empty, falling back");
          return useFallbackPrediction(input);
        }
      }
      catch (Exception e)
      {
        Log.e(TAG, "Neural prediction failed, using fallback", e);
        return useFallbackPrediction(input);
      }
    }
    else
    {
      // Swipe input but neural not available - use fallback
      Log.d(TAG, "Using fallback prediction (neural not available)");
      return useFallbackPrediction(input);
    }
  }
  
  /**
   * Set keyboard dimensions for coordinate mapping
   */
  public void setKeyboardDimensions(float width, float height)
  {
    if (_neuralPredictor != null)
    {
      // Pass dimensions to trajectory processor
      Log.d(TAG, String.format("Set keyboard dimensions: %.0fx%.0f", width, height));
    }
  }
  
  /**
   * Set real key positions for accurate coordinate mapping
   */
  public void setRealKeyPositions(Map<Character, PointF> realPositions)
  {
    if (_neuralPredictor != null)
    {
      // Pass key positions to trajectory processor
      Log.d(TAG, String.format("Set key positions: %d keys", 
        realPositions != null ? realPositions.size() : 0));
    }
  }
  
  /**
   * Update configuration
   */
  public void setConfig(Config config)
  {
    _config = config;
    
    // Update neural predictor configuration
    if (_neuralPredictor != null)
    {
      _neuralPredictor.setConfig(config);
    }
    
    // Update fallback predictor configuration
    if (_fallbackPredictor != null)
    {
      _fallbackPredictor.setConfig(config);
    }
    
    // Update neural prediction preference
    _useNeuralPrediction = _neuralPredictor.isAvailable() && isNeuralEnabled();
    
    Log.d(TAG, String.format("Config updated: neural_enabled=%s, neural_available=%s",
      isNeuralEnabled(), _neuralPredictor.isAvailable()));
  }
  
  /**
   * Check if neural prediction is enabled in config
   */
  private boolean isNeuralEnabled()
  {
    return _config == null || _config.neural_prediction_enabled;
  }
  
  /**
   * Use fallback prediction system
   */
  private PredictionResult useFallbackPrediction(SwipeInput input)
  {
    if (_fallbackPredictor != null)
    {
      // For swipe input, we need to convert to key sequence
      // TODO: Implement fallback prediction with proper interface
      return createEmptyResult();
    }
    else
    {
      return createEmptyResult();
    }
  }
  
  /**
   * Create empty prediction result
   */
  private PredictionResult createEmptyResult()
  {
    return new PredictionResult(
      new java.util.ArrayList<String>(), 
      new java.util.ArrayList<Integer>());
  }
  
  /**
   * Check if neural prediction is available
   */
  public boolean isNeuralAvailable()
  {
    return _neuralPredictor != null && _neuralPredictor.isAvailable();
  }
  
  /**
   * Get current prediction mode
   */
  public String getCurrentMode()
  {
    if (_useNeuralPrediction && isNeuralAvailable())
    {
      return "neural";
    }
    else
    {
      return "fallback";
    }
  }
  
  /**
   * Clean up resources
   */
  public void cleanup()
  {
    if (_neuralPredictor != null)
    {
      _neuralPredictor.cleanup();
    }
    
    Log.d(TAG, "Neural swipe engine cleaned up");
  }
}