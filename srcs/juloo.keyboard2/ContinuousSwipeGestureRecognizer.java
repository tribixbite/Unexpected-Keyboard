package juloo.keyboard2;

import android.graphics.PointF;
import java.util.ArrayList;
import java.util.List;

/**
 * Continuous Swipe Gesture Recognizer Integration
 * 
 * Based on Main.lua usage patterns, this class integrates the CGR library
 * with Android touch handling for swipe typing recognition.
 */
public class ContinuousSwipeGestureRecognizer
{
  private final ContinuousGestureRecognizer cgr;
  private final List<ContinuousGestureRecognizer.Point> gesturePointsList;
  private final List<ContinuousGestureRecognizer.Result> results;
  private boolean newTouch;
  private boolean gestureActive;
  private int minPointsForPrediction = 8; // Start predictions after 8 points (avoid punctuation interference)
  
  // Callback interface for real-time predictions
  public interface OnGesturePredictionListener
  {
    void onGesturePrediction(List<ContinuousGestureRecognizer.Result> predictions);
    void onGestureComplete(List<ContinuousGestureRecognizer.Result> finalPredictions);
    void onGestureCleared();
  }
  
  private OnGesturePredictionListener predictionListener;
  
  public ContinuousSwipeGestureRecognizer()
  {
    cgr = new ContinuousGestureRecognizer();
    gesturePointsList = new ArrayList<>();
    results = new ArrayList<>(); // Pre-allocated results list like in Lua
    gestureActive = false;
    newTouch = false;
    
    // Initialize with directional templates for testing
    // This will be replaced with word templates later
    cgr.setTemplateSet(ContinuousGestureRecognizer.createDirectionalTemplates());
  }
  
  /**
   * Set the prediction listener for real-time callbacks
   */
  public void setOnGesturePredictionListener(OnGesturePredictionListener listener)
  {
    this.predictionListener = listener;
  }
  
  /**
   * Set template set for recognition
   */
  public void setTemplateSet(List<ContinuousGestureRecognizer.Template> templates)
  {
    cgr.setTemplateSet(templates);
  }
  
  /**
   * Handle touch begin event (equivalent to CurrentTouch.state == BEGAN)
   */
  public void onTouchBegan(float x, float y)
  {
    clearPoints(gesturePointsList);
    gesturePointsList.add(new ContinuousGestureRecognizer.Point(x, y));
    newTouch = true;
    gestureActive = true;
    
    // Clear any existing predictions
    if (predictionListener != null)
    {
      predictionListener.onGestureCleared();
    }
  }
  
  /**
   * Handle touch move event (equivalent to CurrentTouch.state == MOVING)
   */
  public void onTouchMoved(float x, float y)
  {
    if (!gestureActive) return;
    
    gesturePointsList.add(new ContinuousGestureRecognizer.Point(x, y));
    
    // Provide real-time predictions if we have enough points
    if (gesturePointsList.size() >= minPointsForPrediction)
    {
      try
      {
        List<ContinuousGestureRecognizer.Result> currentResults = cgr.recognize(gesturePointsList);
        
        // Only send prediction updates if we have results and a listener
        if (currentResults != null && !currentResults.isEmpty() && predictionListener != null)
        {
          predictionListener.onGesturePrediction(currentResults);
        }
      }
      catch (Exception e)
      {
        // Handle any recognition errors gracefully
        android.util.Log.w("ContinuousSwipeGestureRecognizer", "Recognition error during move: " + e.getMessage());
      }
    }
  }
  
  /**
   * Handle touch end event (equivalent to CurrentTouch.state == ENDED)
   */
  public void onTouchEnded(float x, float y)
  {
    if (!gestureActive) return;
    
    gesturePointsList.add(new ContinuousGestureRecognizer.Point(x, y));
    
    if (newTouch)
    {
      newTouch = false;
      
      // Perform final recognition like in the Lua version
      try
      {
        if (gesturePointsList.size() >= 2) // Need at least 2 points for recognition
        {
          List<ContinuousGestureRecognizer.Result> finalResults = cgr.recognize(gesturePointsList);
          
          if (finalResults != null && !finalResults.isEmpty())
          {
            // Store results for persistence
            results.clear();
            results.addAll(finalResults);
            
            // Notify listener of final results
            if (predictionListener != null)
            {
              predictionListener.onGestureComplete(finalResults);
            }
            
            // Debug logging (like CGR_printResults in Lua)
            printResults(finalResults);
          }
        }
      }
      catch (Exception e)
      {
        android.util.Log.e("ContinuousSwipeGestureRecognizer", "Recognition error on end: " + e.getMessage());
      }
    }
    
    gestureActive = false;
  }
  
  /**
   * Clear gesture points (equivalent to clearPoints in Lua)
   */
  private void clearPoints(List<ContinuousGestureRecognizer.Point> points)
  {
    points.clear();
  }
  
  /**
   * Check if gesture is currently active
   */
  public boolean isGestureActive()
  {
    return gestureActive;
  }
  
  /**
   * Get current gesture points for visualization
   */
  public List<PointF> getCurrentGesturePoints()
  {
    List<PointF> androidPoints = new ArrayList<>();
    for (ContinuousGestureRecognizer.Point pt : gesturePointsList)
    {
      androidPoints.add(new PointF((float)pt.x, (float)pt.y));
    }
    return androidPoints;
  }
  
  /**
   * Get the last recognition results (for persistence)
   */
  public List<ContinuousGestureRecognizer.Result> getLastResults()
  {
    return new ArrayList<>(results);
  }
  
  /**
   * Get the best prediction from last results
   */
  public ContinuousGestureRecognizer.Result getBestPrediction()
  {
    if (results.isEmpty()) return null;
    return results.get(0); // Results are sorted by probability
  }
  
  /**
   * Clear stored results (called on space/punctuation)
   */
  public void clearResults()
  {
    results.clear();
    if (predictionListener != null)
    {
      predictionListener.onGestureCleared();
    }
  }
  
  /**
   * Set minimum points required before starting predictions
   */
  public void setMinPointsForPrediction(int minPoints)
  {
    this.minPointsForPrediction = Math.max(2, minPoints);
  }
  
  /**
   * Print results for debugging (equivalent to CGR_printResults in Lua)
   */
  private void printResults(List<ContinuousGestureRecognizer.Result> resultList)
  {
    for (ContinuousGestureRecognizer.Result result : resultList)
    {
      android.util.Log.d("ContinuousSwipeGestureRecognizer", 
        "Result: " + result.template.id + " : " + result.prob);
    }
  }
  
  /**
   * Check results quality (equivalent to CGR_checkResults in Lua)
   * Returns true if the best result is confident enough
   */
  public boolean isResultConfident()
  {
    if (results.size() < 2) return false;
    
    ContinuousGestureRecognizer.Result r1 = results.get(0);
    ContinuousGestureRecognizer.Result r2 = results.get(1);
    
    double similarity = (r2.prob / r1.prob) * r2.prob;
    
    if (r1.prob > 0.7)
    {
      if (similarity < 95)
      {
        android.util.Log.d("ContinuousSwipeGestureRecognizer", 
          "CHECK: Using: " + r1.template.id + " : " + r1.prob);
        return true;
      }
      else
      {
        android.util.Log.d("ContinuousSwipeGestureRecognizer", 
          "CHECK: First two probabilities too close to call");
        return false;
      }
    }
    else
    {
      android.util.Log.d("ContinuousSwipeGestureRecognizer", 
        "CHECK: Probability not high enough (<0.7), discarding user input");
      return false;
    }
  }
  
  /**
   * Reset the recognizer state
   */
  public void reset()
  {
    clearPoints(gesturePointsList);
    results.clear();
    gestureActive = false;
    newTouch = false;
    
    if (predictionListener != null)
    {
      predictionListener.onGestureCleared();
    }
  }
}