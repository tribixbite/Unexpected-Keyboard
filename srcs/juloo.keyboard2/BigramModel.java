package juloo.keyboard2;

import android.content.Context;
import android.util.Log;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Word-level bigram model for contextual predictions.
 * Provides P(word | previous_word) probabilities.
 */
public class BigramModel
{
  private static final String TAG = "BigramModel";
  
  // Bigram probabilities: "prev_word|current_word" -> probability
  private final Map<String, Float> _bigramProbs;
  
  // Unigram fallback: word -> probability
  private final Map<String, Float> _unigramProbs;
  
  // Smoothing parameters
  private static final float LAMBDA = 0.95f; // Interpolation weight for bigram
  private static final float MIN_PROB = 0.0001f; // Minimum probability for unseen words
  
  public BigramModel()
  {
    _bigramProbs = new HashMap<>();
    _unigramProbs = new HashMap<>();
    initializeCommonBigrams();
  }
  
  /**
   * Initialize with most common English word bigrams
   */
  private void initializeCommonBigrams()
  {
    // Most frequent word bigrams in English
    // Format: "previous|current" -> probability
    
    // After "the"
    _bigramProbs.put("the|end", 0.01f);
    _bigramProbs.put("the|first", 0.015f);
    _bigramProbs.put("the|last", 0.012f);
    _bigramProbs.put("the|best", 0.010f);
    _bigramProbs.put("the|world", 0.008f);
    _bigramProbs.put("the|time", 0.007f);
    _bigramProbs.put("the|day", 0.006f);
    _bigramProbs.put("the|way", 0.005f);
    
    // After "a"
    _bigramProbs.put("a|lot", 0.02f);
    _bigramProbs.put("a|little", 0.015f);
    _bigramProbs.put("a|few", 0.012f);
    _bigramProbs.put("a|good", 0.010f);
    _bigramProbs.put("a|great", 0.008f);
    _bigramProbs.put("a|new", 0.007f);
    _bigramProbs.put("a|long", 0.006f);
    
    // After "to"
    _bigramProbs.put("to|be", 0.03f);
    _bigramProbs.put("to|have", 0.02f);
    _bigramProbs.put("to|do", 0.015f);
    _bigramProbs.put("to|go", 0.012f);
    _bigramProbs.put("to|get", 0.010f);
    _bigramProbs.put("to|make", 0.008f);
    _bigramProbs.put("to|see", 0.007f);
    
    // After "of"
    _bigramProbs.put("of|the", 0.05f);
    _bigramProbs.put("of|course", 0.02f);
    _bigramProbs.put("of|all", 0.015f);
    _bigramProbs.put("of|this", 0.012f);
    _bigramProbs.put("of|his", 0.010f);
    _bigramProbs.put("of|her", 0.008f);
    
    // After "in"
    _bigramProbs.put("in|the", 0.04f);
    _bigramProbs.put("in|a", 0.02f);
    _bigramProbs.put("in|this", 0.015f);
    _bigramProbs.put("in|order", 0.012f);
    _bigramProbs.put("in|fact", 0.010f);
    _bigramProbs.put("in|case", 0.008f);
    
    // After "I"
    _bigramProbs.put("i|am", 0.03f);
    _bigramProbs.put("i|have", 0.025f);
    _bigramProbs.put("i|will", 0.02f);
    _bigramProbs.put("i|was", 0.018f);
    _bigramProbs.put("i|can", 0.015f);
    _bigramProbs.put("i|would", 0.012f);
    _bigramProbs.put("i|think", 0.010f);
    _bigramProbs.put("i|know", 0.008f);
    _bigramProbs.put("i|want", 0.007f);
    
    // After "you"
    _bigramProbs.put("you|are", 0.025f);
    _bigramProbs.put("you|can", 0.02f);
    _bigramProbs.put("you|have", 0.018f);
    _bigramProbs.put("you|will", 0.015f);
    _bigramProbs.put("you|want", 0.012f);
    _bigramProbs.put("you|know", 0.010f);
    _bigramProbs.put("you|need", 0.008f);
    
    // After "it"
    _bigramProbs.put("it|is", 0.04f);
    _bigramProbs.put("it|was", 0.025f);
    _bigramProbs.put("it|will", 0.015f);
    _bigramProbs.put("it|would", 0.012f);
    _bigramProbs.put("it|has", 0.010f);
    _bigramProbs.put("it|can", 0.008f);
    
    // After "that"
    _bigramProbs.put("that|is", 0.025f);
    _bigramProbs.put("that|was", 0.02f);
    _bigramProbs.put("that|the", 0.015f);
    _bigramProbs.put("that|it", 0.012f);
    _bigramProbs.put("that|you", 0.010f);
    _bigramProbs.put("that|he", 0.008f);
    
    // After "with"
    _bigramProbs.put("with|the", 0.03f);
    _bigramProbs.put("with|a", 0.02f);
    _bigramProbs.put("with|his", 0.015f);
    _bigramProbs.put("with|her", 0.012f);
    _bigramProbs.put("with|my", 0.010f);
    _bigramProbs.put("with|your", 0.008f);
    
    // Common unigram probabilities (fallback)
    _unigramProbs.put("the", 0.07f);
    _unigramProbs.put("be", 0.04f);
    _unigramProbs.put("to", 0.035f);
    _unigramProbs.put("of", 0.03f);
    _unigramProbs.put("and", 0.028f);
    _unigramProbs.put("a", 0.025f);
    _unigramProbs.put("in", 0.022f);
    _unigramProbs.put("that", 0.02f);
    _unigramProbs.put("have", 0.018f);
    _unigramProbs.put("i", 0.017f);
    _unigramProbs.put("it", 0.015f);
    _unigramProbs.put("for", 0.014f);
    _unigramProbs.put("not", 0.013f);
    _unigramProbs.put("on", 0.012f);
    _unigramProbs.put("with", 0.011f);
    _unigramProbs.put("he", 0.010f);
    _unigramProbs.put("as", 0.009f);
    _unigramProbs.put("you", 0.009f);
    _unigramProbs.put("do", 0.008f);
    _unigramProbs.put("at", 0.008f);
  }
  
  /**
   * Load bigram data from a file (future enhancement)
   */
  public void loadFromFile(Context context, String filename)
  {
    // Future: Load comprehensive bigram data from assets
    // Format: prev_word current_word probability
    try
    {
      BufferedReader reader = new BufferedReader(
        new InputStreamReader(context.getAssets().open(filename)));
      String line;
      while ((line = reader.readLine()) != null)
      {
        String[] parts = line.split("\\s+");
        if (parts.length >= 3)
        {
          String bigram = parts[0].toLowerCase() + "|" + parts[1].toLowerCase();
          float prob = Float.parseFloat(parts[2]);
          _bigramProbs.put(bigram, prob);
        }
      }
      reader.close();
      Log.d(TAG, "Loaded " + _bigramProbs.size() + " bigrams from " + filename);
    }
    catch (IOException e)
    {
      Log.e(TAG, "Failed to load bigram file: " + filename, e);
    }
  }
  
  /**
   * Get the probability of a word given the previous word(s)
   * Uses linear interpolation between bigram and unigram probabilities
   */
  public float getContextualProbability(String word, List<String> context)
  {
    if (word == null || word.isEmpty())
      return MIN_PROB;
    
    word = word.toLowerCase();
    
    // If no context, return unigram probability
    if (context == null || context.isEmpty())
    {
      return _unigramProbs.getOrDefault(word, MIN_PROB);
    }
    
    // Get the previous word
    String prevWord = context.get(context.size() - 1).toLowerCase();
    String bigramKey = prevWord + "|" + word;
    
    // Look up bigram probability
    float bigramProb = _bigramProbs.getOrDefault(bigramKey, 0.0f);
    
    // Look up unigram probability (fallback)
    float unigramProb = _unigramProbs.getOrDefault(word, MIN_PROB);
    
    // Linear interpolation: λ * P(word|prev) + (1-λ) * P(word)
    float interpolatedProb = LAMBDA * bigramProb + (1 - LAMBDA) * unigramProb;
    
    // Ensure minimum probability
    return Math.max(interpolatedProb, MIN_PROB);
  }
  
  /**
   * Score a word based on context (returns log probability for numerical stability)
   */
  public float scoreWord(String word, List<String> context)
  {
    float prob = getContextualProbability(word, context);
    // Return log probability to avoid underflow
    return (float)Math.log(prob);
  }
  
  /**
   * Get a multiplier for prediction scoring (1.0 = neutral, >1.0 = boost, <1.0 = penalty)
   */
  public float getContextMultiplier(String word, List<String> context)
  {
    if (context == null || context.isEmpty())
      return 1.0f;
    
    float contextProb = getContextualProbability(word, context);
    float baseProb = _unigramProbs.getOrDefault(word.toLowerCase(), MIN_PROB);
    
    // Return ratio of contextual to base probability
    // This gives a boost when context makes the word more likely
    float multiplier = contextProb / baseProb;
    
    // Cap the multiplier to avoid extreme values
    return Math.min(Math.max(multiplier, 0.1f), 10.0f);
  }
  
  /**
   * Add a bigram observation (for user adaptation)
   */
  public void addBigram(String prevWord, String word, float weight)
  {
    String bigramKey = prevWord.toLowerCase() + "|" + word.toLowerCase();
    float currentProb = _bigramProbs.getOrDefault(bigramKey, 0.0f);
    // Simple exponential smoothing for adaptation
    float newProb = 0.9f * currentProb + 0.1f * weight;
    _bigramProbs.put(bigramKey, newProb);
  }
  
  /**
   * Get statistics about the model
   */
  public String getStatistics()
  {
    return String.format("BigramModel: %d bigrams, %d unigrams",
                        _bigramProbs.size(), _unigramProbs.size());
  }
}