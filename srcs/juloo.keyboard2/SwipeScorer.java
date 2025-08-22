package juloo.keyboard2;

import java.util.List;

/**
 * Unified scoring system applying all 8 confidence weights
 */
public class SwipeScorer
{
  private Config _config;
  
  public SwipeScorer()
  {
    _config = null;
  }
  
  public void setConfig(Config config)
  {
    _config = config;
  }
  
  /**
   * Calculate final score using all 8 weight factors
   */
  public float calculateFinalScore(SwipeTypingEngine.ScoredCandidate candidate,
                                  SwipeInput input,
                                  SwipeDetector.SwipeClassification classification,
                                  Config config)
  {
    if (config == null)
      config = _config;
      
    float score = candidate.score;
    
    // 1. Shape matching weight (DTW distance correlation)
    if (candidate.source.equals("DTW") || candidate.source.equals("Both"))
    {
      // DTW candidates already have shape matching built into their score
      // Apply the shape weight as a multiplier
      score *= config.swipe_confidence_shape_weight;
    }
    else
    {
      // For sequence-only candidates, reduce score based on lack of shape matching
      score *= (0.5f + 0.5f * config.swipe_confidence_shape_weight);
    }
    
    // 2. Location accuracy weight (how well the swipe hit the keys)
    float locationAccuracy = calculateLocationAccuracy(candidate.word, input);
    score *= applyWeight(locationAccuracy, config.swipe_confidence_location_weight);
    
    // 3. Word frequency weight (already partially applied in base score)
    // Apply additional frequency weight adjustment
    float frequencyFactor = getWordFrequencyFactor(candidate.word);
    score *= applyWeight(frequencyFactor, config.swipe_confidence_frequency_weight);
    
    // 4. Velocity weight (consistency of swipe speed)
    float velocityScore = calculateVelocityScore(input);
    score *= applyWeight(velocityScore, config.swipe_confidence_velocity_weight);
    
    // 5. First letter weight
    if (matchesFirstLetter(candidate.word, input))
    {
      score *= config.swipe_first_letter_weight;
    }
    
    // 6. Last letter weight
    if (matchesLastLetter(candidate.word, input))
    {
      score *= config.swipe_last_letter_weight;
    }
    
    // 7. Endpoint bonus (both first and last match)
    if (matchesFirstLetter(candidate.word, input) && matchesLastLetter(candidate.word, input))
    {
      score *= config.swipe_endpoint_bonus_weight;
    }
    
    // 8. Apply swipe quality multiplier
    score *= getQualityMultiplier(classification);
    
    // 9. Apply strict endpoint filtering if enabled
    if (config.swipe_require_endpoints)
    {
      if (!matchesFirstLetter(candidate.word, input) || !matchesLastLetter(candidate.word, input))
      {
        score *= 0.1f; // Heavily penalize non-matching endpoints in strict mode
      }
    }
    
    return score;
  }
  
  /**
   * Calculate how accurately the swipe path matches the word's key locations
   */
  private float calculateLocationAccuracy(String word, SwipeInput input)
  {
    if (input.keySequence.isEmpty() || word.isEmpty())
      return 0.5f;
    
    // Simple heuristic: check how many characters from the word appear in the key sequence
    int matches = 0;
    String lowerWord = word.toLowerCase();
    String lowerSeq = input.keySequence.toLowerCase();
    
    for (char c : lowerWord.toCharArray())
    {
      if (lowerSeq.indexOf(c) >= 0)
        matches++;
    }
    
    float accuracy = (float)matches / lowerWord.length();
    
    // Bonus for ordered matches
    if (isSubsequence(lowerWord, lowerSeq))
    {
      accuracy = Math.min(1.0f, accuracy * 1.2f);
    }
    
    return accuracy;
  }
  
  /**
   * Check if word is a subsequence of the key sequence
   */
  private boolean isSubsequence(String word, String sequence)
  {
    int seqIndex = 0;
    for (char c : word.toCharArray())
    {
      int found = sequence.indexOf(c, seqIndex);
      if (found == -1)
        return false;
      seqIndex = found + 1;
    }
    return true;
  }
  
  /**
   * Get word frequency factor (placeholder - should use actual frequency data)
   */
  private float getWordFrequencyFactor(String word)
  {
    // Common words get higher scores
    // This is a simplified implementation - should use actual frequency data
    String[] veryCommon = {"the", "and", "for", "are", "but", "not", "you", "can", "with", "have"};
    String[] common = {"this", "that", "from", "they", "what", "when", "make", "like", "time", "just"};
    
    for (String w : veryCommon)
    {
      if (word.equalsIgnoreCase(w))
        return 1.0f;
    }
    
    for (String w : common)
    {
      if (word.equalsIgnoreCase(w))
        return 0.8f;
    }
    
    // Default frequency factor
    return 0.5f;
  }
  
  /**
   * Calculate velocity consistency score
   */
  private float calculateVelocityScore(SwipeInput input)
  {
    if (input.velocityProfile.isEmpty())
      return 0.5f;
    
    // Calculate coefficient of variation (CV) for velocity
    float sum = 0;
    float sumSquared = 0;
    
    for (float v : input.velocityProfile)
    {
      sum += v;
      sumSquared += v * v;
    }
    
    float mean = sum / input.velocityProfile.size();
    if (mean == 0)
      return 0.5f;
      
    float variance = (sumSquared / input.velocityProfile.size()) - (mean * mean);
    float stdDev = (float)Math.sqrt(variance);
    float cv = stdDev / mean;
    
    // Lower CV means more consistent velocity (better)
    // CV typically ranges from 0 to 2+ for swipes
    if (cv < 0.3f)
      return 1.0f; // Very consistent
    else if (cv < 0.6f)
      return 0.8f; // Good consistency
    else if (cv < 1.0f)
      return 0.6f; // Moderate consistency
    else if (cv < 1.5f)
      return 0.4f; // Poor consistency
    else
      return 0.2f; // Very inconsistent
  }
  
  /**
   * Check if word's first letter matches swipe start
   */
  private boolean matchesFirstLetter(String word, SwipeInput input)
  {
    if (word.isEmpty() || input.keySequence.isEmpty())
      return false;
    
    return Character.toLowerCase(word.charAt(0)) == 
           Character.toLowerCase(input.keySequence.charAt(0));
  }
  
  /**
   * Check if word's last letter matches swipe end
   */
  private boolean matchesLastLetter(String word, SwipeInput input)
  {
    if (word.isEmpty() || input.keySequence.isEmpty())
      return false;
    
    return Character.toLowerCase(word.charAt(word.length() - 1)) == 
           Character.toLowerCase(input.keySequence.charAt(input.keySequence.length() - 1));
  }
  
  /**
   * Apply weight with proper scaling
   */
  private float applyWeight(float value, float weight)
  {
    // Weight acts as a multiplier on the value's influence
    // weight = 1.0 means normal influence
    // weight > 1.0 means increased influence
    // weight < 1.0 means decreased influence
    
    // Use exponential scaling for more dramatic effect
    return (float)Math.pow(value, 2.0f - weight);
  }
  
  /**
   * Get quality-based score multiplier
   */
  private float getQualityMultiplier(SwipeDetector.SwipeClassification classification)
  {
    switch (classification.quality)
    {
      case HIGH:
        return 1.2f;
      case MEDIUM:
        return 1.0f;
      case LOW:
        return 0.8f;
      case NOT_SWIPE:
        return 0.5f;
      default:
        return 1.0f;
    }
  }
}