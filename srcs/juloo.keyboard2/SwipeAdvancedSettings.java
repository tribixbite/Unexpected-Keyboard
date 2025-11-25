package juloo.keyboard2;

import android.content.Context;
import android.content.SharedPreferences;

/**
 * Advanced swipe typing settings that can be tuned by users
 * These parameters affect swipe recognition accuracy and behavior
 */
public class SwipeAdvancedSettings
{
  private static SwipeAdvancedSettings _instance;
  private final SharedPreferences _prefs;
  
  // Gaussian model parameters (affects key hit detection)
  private float _gaussianSigmaXFactor = 0.4f;  // Default: 40% of key width
  private float _gaussianSigmaYFactor = 0.35f; // Default: 35% of key height
  private float _gaussianMinProbability = 0.01f;
  
  // Path pruning parameters
  private float _minPathLengthRatio = 0.3f; // Minimum path length relative to word length
  private float _maxPathLengthRatio = 3.0f; // Maximum path length relative to word length
  
  // N-gram model parameters
  private float _ngramSmoothingFactor = 0.1f; // Smoothing for unseen bigrams
  private int _contextWindowSize = 2; // Number of previous words to consider
  
  // Beam Search Parameters (New)
  private float _neuralBeamAlpha = 1.2f;
  private float _neuralBeamPruneConfidence = 0.8f;
  private float _neuralBeamScoreGap = 5.0f;
  
  private SwipeAdvancedSettings(Context context)
  {
    _prefs = context.getSharedPreferences("swipe_advanced", Context.MODE_PRIVATE);
    loadSettings();
  }
  
  public static synchronized SwipeAdvancedSettings getInstance(Context context)
  {
    if (_instance == null)
    {
      _instance = new SwipeAdvancedSettings(context);
    }
    return _instance;
  }
  
  private void loadSettings()
  {
    _gaussianSigmaXFactor = Config.safeGetFloat(_prefs, "gaussian_sigma_x", 0.4f);
    _gaussianSigmaYFactor = Config.safeGetFloat(_prefs, "gaussian_sigma_y", 0.35f);
    _gaussianMinProbability = Config.safeGetFloat(_prefs, "gaussian_min_prob", 0.01f);

    _minPathLengthRatio = Config.safeGetFloat(_prefs, "min_path_length_ratio", 0.3f);
    _maxPathLengthRatio = Config.safeGetFloat(_prefs, "max_path_length_ratio", 3.0f);

    _ngramSmoothingFactor = Config.safeGetFloat(_prefs, "ngram_smoothing", 0.1f);
    _contextWindowSize = _prefs.getInt("context_window", 2);
    
    _neuralBeamAlpha = Config.safeGetFloat(_prefs, "neural_beam_alpha", 1.2f);
    _neuralBeamPruneConfidence = Config.safeGetFloat(_prefs, "neural_beam_prune_confidence", 0.8f);
    _neuralBeamScoreGap = Config.safeGetFloat(_prefs, "neural_beam_score_gap", 5.0f);
  }
  
  public void saveSettings()
  {
    SharedPreferences.Editor editor = _prefs.edit();
    
    editor.putFloat("gaussian_sigma_x", _gaussianSigmaXFactor);
    editor.putFloat("gaussian_sigma_y", _gaussianSigmaYFactor);
    editor.putFloat("gaussian_min_prob", _gaussianMinProbability);
    
    editor.putFloat("min_path_length_ratio", _minPathLengthRatio);
    editor.putFloat("max_path_length_ratio", _maxPathLengthRatio);
    
    editor.putFloat("ngram_smoothing", _ngramSmoothingFactor);
    editor.putInt("context_window", _contextWindowSize);
    
    editor.putFloat("neural_beam_alpha", _neuralBeamAlpha);
    editor.putFloat("neural_beam_prune_confidence", _neuralBeamPruneConfidence);
    editor.putFloat("neural_beam_score_gap", _neuralBeamScoreGap);
    
    editor.apply();
  }
  
  // Getters
  public float getGaussianSigmaXFactor() { return _gaussianSigmaXFactor; }
  public float getGaussianSigmaYFactor() { return _gaussianSigmaYFactor; }
  public float getGaussianMinProbability() { return _gaussianMinProbability; }
  
  public float getMinPathLengthRatio() { return _minPathLengthRatio; }
  public float getMaxPathLengthRatio() { return _maxPathLengthRatio; }
  
  public float getNgramSmoothingFactor() { return _ngramSmoothingFactor; }
  public int getContextWindowSize() { return _contextWindowSize; }
  
  public float getNeuralBeamAlpha() { return _neuralBeamAlpha; }
  public float getNeuralBeamPruneConfidence() { return _neuralBeamPruneConfidence; }
  public float getNeuralBeamScoreGap() { return _neuralBeamScoreGap; }
  
  // Setters with validation
  public void setGaussianSigmaXFactor(float value) 
  { 
    _gaussianSigmaXFactor = Math.max(0.1f, Math.min(1.0f, value));
    saveSettings();
  }
  
  public void setGaussianSigmaYFactor(float value) 
  { 
    _gaussianSigmaYFactor = Math.max(0.1f, Math.min(1.0f, value));
    saveSettings();
  }
  
  public void setGaussianMinProbability(float value) 
  { 
    _gaussianMinProbability = Math.max(0.001f, Math.min(0.1f, value));
    saveSettings();
  }
  
  public void setMinPathLengthRatio(float value) 
  { 
    _minPathLengthRatio = Math.max(0.1f, Math.min(1.0f, value));
    saveSettings();
  }
  
  public void setMaxPathLengthRatio(float value) 
  { 
    _maxPathLengthRatio = Math.max(1.5f, Math.min(5.0f, value));
    saveSettings();
  }
  
  public void setNgramSmoothingFactor(float value) 
  { 
    _ngramSmoothingFactor = Math.max(0.01f, Math.min(1.0f, value));
    saveSettings();
  }
  
  public void setContextWindowSize(int value) 
  { 
    _contextWindowSize = Math.max(1, Math.min(5, value));
    saveSettings();
  }
  
  public void setNeuralBeamAlpha(float value)
  {
    _neuralBeamAlpha = Math.max(0.0f, Math.min(5.0f, value));
    saveSettings();
  }
  
  public void setNeuralBeamPruneConfidence(float value)
  {
    _neuralBeamPruneConfidence = Math.max(0.0f, Math.min(1.0f, value));
    saveSettings();
  }
  
  public void setNeuralBeamScoreGap(float value)
  {
    _neuralBeamScoreGap = Math.max(0.0f, Math.min(20.0f, value));
    saveSettings();
  }
  
  /**
   * Reset all settings to defaults
   */
  public void resetToDefaults()
  {
    _gaussianSigmaXFactor = 0.4f;
    _gaussianSigmaYFactor = 0.35f;
    _gaussianMinProbability = 0.01f;
    
    _minPathLengthRatio = 0.3f;
    _maxPathLengthRatio = 3.0f;
    
    _ngramSmoothingFactor = 0.1f;
    _contextWindowSize = 2;
    
    _neuralBeamAlpha = 1.2f;
    _neuralBeamPruneConfidence = 0.8f;
    _neuralBeamScoreGap = 5.0f;
    
    saveSettings();
  }
}