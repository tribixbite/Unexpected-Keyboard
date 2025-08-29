package juloo.keyboard2;

import android.content.Context;
import android.graphics.PointF;
import java.util.ArrayList;
import java.util.List;

/**
 * Real-time swipe prediction system using CGR library
 * Integrates with existing keyboard to provide live word suggestions during swipe gestures
 */
public class RealTimeSwipePredictor implements ContinuousSwipeGestureRecognizer.OnGesturePredictionListener
{
  private final ContinuousSwipeGestureRecognizer gestureRecognizer;
  private final WordGestureTemplateGenerator templateGenerator;
  private final List<String> currentPredictions;
  private final List<String> persistentPredictions;
  private boolean isInitialized;
  private boolean predictionsPersisting;
  
  // Callback interface for keyboard integration
  public interface OnSwipePredictionListener
  {
    void onSwipePredictionUpdate(List<String> predictions);
    void onSwipePredictionComplete(List<String> finalPredictions);
    void onSwipePredictionCleared();
  }
  
  private OnSwipePredictionListener predictionListener;
  
  public RealTimeSwipePredictor()
  {
    gestureRecognizer = new ContinuousSwipeGestureRecognizer();
    templateGenerator = new WordGestureTemplateGenerator();
    currentPredictions = new ArrayList<>();
    persistentPredictions = new ArrayList<>();
    isInitialized = false;
    predictionsPersisting = false;
    
    // Set up gesture recognizer callbacks
    gestureRecognizer.setOnGesturePredictionListener(this);
    gestureRecognizer.setMinPointsForPrediction(6); // Start after 2+ chars swiped (reduced for better responsiveness)
  }
  
  /**
   * Initialize the predictor with dictionary data
   */
  public void initialize(Context context)
  {
    if (isInitialized) return;
    
    android.util.Log.d("RealTimeSwipePredictor", "Initializing swipe predictor...");
    
    // Load dictionary in background thread to avoid blocking UI
    new Thread(() -> {
      try
      {
        templateGenerator.loadDictionary(context);
        
        // Generate templates for most frequent words (practical vocabulary size)
        List<ContinuousGestureRecognizer.Template> templates = 
          templateGenerator.generateBalancedWordTemplates(3000); // Practical vocabulary for swipe typing
        
        // Set templates in gesture recognizer
        gestureRecognizer.setTemplateSet(templates);
        
        isInitialized = true;
        android.util.Log.d("RealTimeSwipePredictor", 
          "Swipe predictor initialized with " + templates.size() + " word templates");
      }
      catch (Exception e)
      {
        android.util.Log.e("RealTimeSwipePredictor", "Failed to initialize: " + e.getMessage());
      }
    }).start();
  }
  
  /**
   * Set the prediction callback listener
   */
  public void setOnSwipePredictionListener(OnSwipePredictionListener listener)
  {
    this.predictionListener = listener;
  }
  
  /**
   * Handle touch begin event
   */
  public void onTouchBegan(float x, float y)
  {
    if (!isInitialized) return;
    
    // Clear any persistent predictions when starting new gesture
    clearPersistentPredictions();
    
    gestureRecognizer.onTouchBegan(x, y);
  }
  
  /**
   * Handle touch move event
   */
  public void onTouchMoved(float x, float y)
  {
    if (!isInitialized) return;
    
    gestureRecognizer.onTouchMoved(x, y);
  }
  
  /**
   * Handle touch end event
   */
  public void onTouchEnded(float x, float y)
  {
    if (!isInitialized) return;
    
    gestureRecognizer.onTouchEnded(x, y);
  }
  
  /**
   * Check if gesture is currently active
   */
  public boolean isGestureActive()
  {
    return gestureRecognizer.isGestureActive();
  }
  
  /**
   * Get current gesture path for visualization
   */
  public List<PointF> getCurrentGesturePath()
  {
    return gestureRecognizer.getCurrentGesturePoints();
  }
  
  /**
   * Clear predictions (called on space or non-letter key)
   */
  public void clearPredictions()
  {
    currentPredictions.clear();
    clearPersistentPredictions();
    gestureRecognizer.clearResults();
  }
  
  /**
   * Clear persistent predictions
   */
  private void clearPersistentPredictions()
  {
    if (predictionsPersisting)
    {
      persistentPredictions.clear();
      predictionsPersisting = false;
      
      if (predictionListener != null)
      {
        predictionListener.onSwipePredictionCleared();
      }
    }
  }
  
  /**
   * Get current predictions (persistent if available, otherwise current)
   */
  public List<String> getCurrentPredictions()
  {
    if (predictionsPersisting && !persistentPredictions.isEmpty())
    {
      return new ArrayList<>(persistentPredictions);
    }
    return new ArrayList<>(currentPredictions);
  }
  
  /**
   * Check if predictions are currently persisting
   */
  public boolean arePredictionsPersisting()
  {
    return predictionsPersisting;
  }
  
  /**
   * Reset predictor state
   */
  public void reset()
  {
    currentPredictions.clear();
    clearPersistentPredictions();
    gestureRecognizer.reset();
  }
  
  // ContinuousSwipeGestureRecognizer.OnGesturePredictionListener implementation
  
  @Override
  public void onGesturePrediction(List<ContinuousGestureRecognizer.Result> predictions)
  {
    // Convert CGR results to word predictions
    currentPredictions.clear();
    
    int maxPredictions = Math.min(5, predictions.size()); // Show top 5 predictions
    for (int i = 0; i < maxPredictions; i++)
    {
      ContinuousGestureRecognizer.Result result = predictions.get(i);
      if (result.prob > 0.1) // Only show predictions with reasonable confidence
      {
        currentPredictions.add(result.template.id);
      }
    }
    
    // Only update UI if we're not in persistent mode and we have predictions
    if (!predictionsPersisting && !currentPredictions.isEmpty())
    {
      if (predictionListener != null)
      {
        predictionListener.onSwipePredictionUpdate(new ArrayList<>(currentPredictions));
      }
    }
  }
  
  @Override
  public void onGestureComplete(List<ContinuousGestureRecognizer.Result> finalPredictions)
  {
    // Convert final results to persistent predictions
    persistentPredictions.clear();
    
    int maxPredictions = Math.min(5, finalPredictions.size());
    for (int i = 0; i < maxPredictions; i++)
    {
      ContinuousGestureRecognizer.Result result = finalPredictions.get(i);
      if (result.prob > 0.05) // Slightly lower threshold for final predictions
      {
        persistentPredictions.add(result.template.id);
      }
    }
    
    // Enable persistence if we have good predictions
    if (!persistentPredictions.isEmpty())
    {
      predictionsPersisting = true;
      
      if (predictionListener != null)
      {
        predictionListener.onSwipePredictionComplete(new ArrayList<>(persistentPredictions));
      }
      
      android.util.Log.d("RealTimeSwipePredictor", 
        "Persisting " + persistentPredictions.size() + " predictions: " + persistentPredictions);
    }
    else
    {
      // No good predictions, clear everything
      if (predictionListener != null)
      {
        predictionListener.onSwipePredictionCleared();
      }
    }
  }
  
  @Override
  public void onGestureCleared()
  {
    currentPredictions.clear();
    // Don't clear persistent predictions here - they should persist until space/punctuation
  }
  
  /**
   * Check if initialized
   */
  public boolean isInitialized()
  {
    return isInitialized;
  }
  
  /**
   * Get dictionary size (for debugging)
   */
  public int getDictionarySize()
  {
    return templateGenerator.getDictionarySize();
  }
  
  /**
   * Handle key press (to clear predictions on space/punctuation)
   */
  public void onKeyPressed(KeyValue key)
  {
    if (key == null) return;
    
    // Clear predictions on space or non-letter keys
    if (key.getKind() == KeyValue.Kind.Char)
    {
      char c = key.getChar();
      if (c == ' ' || !Character.isLetter(c))
      {
        clearPredictions();
      }
    }
    else if (key.getKind() != KeyValue.Kind.Char)
    {
      // Any non-character key (like backspace, enter, etc.)
      clearPredictions();
    }
  }
  
  /**
   * Select a prediction (called when user taps on suggestion)
   */
  public void selectPrediction(String word)
  {
    android.util.Log.d("RealTimeSwipePredictor", "Selected prediction: " + word);
    clearPredictions();
  }
}