package juloo.keyboard2;

import android.content.Context;
import android.content.SharedPreferences;

/**
 * Configuration for swipe algorithm weights
 * Provides centralized access to user-configured algorithm weights
 */
public class SwipeWeightConfig
{
  private static SwipeWeightConfig _instance;
  
  private float _dtwWeight;
  private float _gaussianWeight;
  private float _ngramWeight;
  private float _frequencyWeight;
  private Context _context;
  
  // Default weights
  private static final float DEFAULT_DTW_WEIGHT = 0.4f;
  private static final float DEFAULT_GAUSSIAN_WEIGHT = 0.3f;
  private static final float DEFAULT_NGRAM_WEIGHT = 0.2f;
  private static final float DEFAULT_FREQUENCY_WEIGHT = 0.1f;
  
  private SwipeWeightConfig(Context context)
  {
    _context = context.getApplicationContext();
    loadWeights();
  }
  
  /**
   * Get singleton instance
   */
  public static synchronized SwipeWeightConfig getInstance(Context context)
  {
    if (_instance == null)
    {
      _instance = new SwipeWeightConfig(context);
    }
    return _instance;
  }
  
  /**
   * Load weights from SharedPreferences
   */
  public void loadWeights()
  {
    SharedPreferences prefs = _context.getSharedPreferences("swipe_weights", Context.MODE_PRIVATE);
    _dtwWeight = prefs.getFloat("dtw_weight", DEFAULT_DTW_WEIGHT);
    _gaussianWeight = prefs.getFloat("gaussian_weight", DEFAULT_GAUSSIAN_WEIGHT);
    _ngramWeight = prefs.getFloat("ngram_weight", DEFAULT_NGRAM_WEIGHT);
    _frequencyWeight = prefs.getFloat("frequency_weight", DEFAULT_FREQUENCY_WEIGHT);
    
    // Normalize weights
    normalizeWeights();
  }
  
  /**
   * Save weights to SharedPreferences
   */
  public void saveWeights(float dtw, float gaussian, float ngram, float frequency)
  {
    _dtwWeight = dtw;
    _gaussianWeight = gaussian;
    _ngramWeight = ngram;
    _frequencyWeight = frequency;
    
    normalizeWeights();
    
    SharedPreferences prefs = _context.getSharedPreferences("swipe_weights", Context.MODE_PRIVATE);
    SharedPreferences.Editor editor = prefs.edit();
    editor.putFloat("dtw_weight", _dtwWeight);
    editor.putFloat("gaussian_weight", _gaussianWeight);
    editor.putFloat("ngram_weight", _ngramWeight);
    editor.putFloat("frequency_weight", _frequencyWeight);
    editor.apply();
  }
  
  /**
   * Normalize weights to sum to 1.0
   */
  private void normalizeWeights()
  {
    float total = _dtwWeight + _gaussianWeight + _ngramWeight + _frequencyWeight;
    if (total > 0)
    {
      _dtwWeight /= total;
      _gaussianWeight /= total;
      _ngramWeight /= total;
      _frequencyWeight /= total;
    }
    else
    {
      // Reset to defaults if all zeros
      _dtwWeight = DEFAULT_DTW_WEIGHT;
      _gaussianWeight = DEFAULT_GAUSSIAN_WEIGHT;
      _ngramWeight = DEFAULT_NGRAM_WEIGHT;
      _frequencyWeight = DEFAULT_FREQUENCY_WEIGHT;
    }
  }
  
  // Getters
  public float getDtwWeight() { return _dtwWeight; }
  public float getGaussianWeight() { return _gaussianWeight; }
  public float getNgramWeight() { return _ngramWeight; }
  public float getFrequencyWeight() { return _frequencyWeight; }
  
  /**
   * Apply weights to compute final score
   */
  public float computeWeightedScore(float dtwScore, float gaussianScore, float ngramScore, float frequencyScore)
  {
    return dtwScore * _dtwWeight +
           gaussianScore * _gaussianWeight +
           ngramScore * _ngramWeight +
           frequencyScore * _frequencyWeight;
  }
  
  /**
   * Get a string representation of current weights
   */
  public String toString()
  {
    return String.format("Weights: DTW=%.0f%%, Gaussian=%.0f%%, N-gram=%.0f%%, Frequency=%.0f%%",
                        _dtwWeight * 100, _gaussianWeight * 100, 
                        _ngramWeight * 100, _frequencyWeight * 100);
  }
}