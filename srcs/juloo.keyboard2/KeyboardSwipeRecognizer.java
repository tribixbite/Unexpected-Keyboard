package juloo.keyboard2;

import android.content.Context;
import android.graphics.PointF;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Keyboard-Specific Swipe Recognition Algorithm
 * 
 * Replaces CGR with algorithm designed specifically for constrained keyboard gestures
 * Based on Bayesian framework: P(word | swipe) ∝ P(swipe | word) × P(word)
 * 
 * Key differences from CGR:
 * - Key proximity matching instead of shape recognition
 * - Letter sequence validation with order enforcement
 * - Start point emphasis over end point (users begin precisely, end sloppily)
 * - Keyboard-specific cost function for constrained paths
 */
public class KeyboardSwipeRecognizer
{
  // Keyboard-specific cost function weights (user-configurable)
  private double proximityWeight = 1.0;      // α - Distance to key centers
  private double missingKeyPenalty = 10.0;   // β - Missing required letters
  private double extraKeyPenalty = 2.0;      // γ - Passing over wrong letters  
  private double orderPenalty = 5.0;         // δ - Out-of-order letter sequence
  private double startPointWeight = 3.0;     // ε - Start point accuracy emphasis
  
  // Infrastructure components (reusing existing code)
  private WordGestureTemplateGenerator templateGenerator;
  private BigramModel bigramModel;
  private NgramModel ngramModel;
  private UserAdaptationManager userAdaptation;
  private SwipeWeightConfig weightConfig;
  private GaussianKeyModel gaussianModel;
  
  // Key zone parameters (INCREASED for better detection)
  private double keyZoneRadius = 120.0;      // Larger zone for easier key detection
  private double pathSampleDistance = 10.0;  // More frequent sampling for better detection
  
  public KeyboardSwipeRecognizer(Context context)
  {
    templateGenerator = new WordGestureTemplateGenerator();
    templateGenerator.loadDictionary(context);
    
    try 
    {
      bigramModel = new BigramModel();
      ngramModel = new NgramModel();
      userAdaptation = UserAdaptationManager.getInstance(context);
      weightConfig = SwipeWeightConfig.getInstance(context);
      gaussianModel = new GaussianKeyModel();
    }
    catch (Exception e)
    {
      android.util.Log.w("KeyboardSwipeRecognizer", "Some components failed to initialize: " + e.getMessage());
    }
    
    android.util.Log.d("KeyboardSwipeRecognizer", "Initialized keyboard-specific recognition algorithm");
  }
  
  /**
   * Recognition result with detailed scoring breakdown
   */
  public static class RecognitionResult
  {
    public final String word;
    public final double totalScore;
    public final double proximityScore;
    public final double sequenceScore;
    public final double startPointScore;
    public final double languageModelScore;
    public final List<Character> detectedLetters;
    
    public RecognitionResult(String word, double totalScore, double proximityScore, 
                           double sequenceScore, double startPointScore, double languageModelScore,
                           List<Character> detectedLetters)
    {
      this.word = word;
      this.totalScore = totalScore;
      this.proximityScore = proximityScore;
      this.sequenceScore = sequenceScore;
      this.startPointScore = startPointScore;
      this.languageModelScore = languageModelScore;
      this.detectedLetters = detectedLetters;
    }
  }
  
  /**
   * Main recognition method - replaces CGR completely
   */
  public List<RecognitionResult> recognizeSwipe(List<PointF> swipePath, List<String> context)
  {
    if (swipePath.size() < 2)
    {
      return new ArrayList<>();
    }
    
    android.util.Log.d("KeyboardSwipeRecognizer", "Recognizing swipe with " + swipePath.size() + " points");
    
    // Step 1: Detect letter sequence from swipe path
    List<Character> detectedLetters = detectLetterSequence(swipePath);
    android.util.Log.d("KeyboardSwipeRecognizer", "Detected letter sequence: " + detectedLetters);
    
    // FALLBACK: If no letters detected, use simple heuristic
    if (detectedLetters.isEmpty())
    {
      android.util.Log.w("KeyboardSwipeRecognizer", "No letters detected - using fallback approach");
      // Add most common letters as fallback
      detectedLetters.add('a');
      detectedLetters.add('e');
      detectedLetters.add('t');
    }
    
    // Step 2: Generate candidate words containing these letters
    List<String> candidates = generateCandidateWords(detectedLetters);
    android.util.Log.d("KeyboardSwipeRecognizer", "Generated " + candidates.size() + " candidate words");
    
    // Step 3: Calculate scores for each candidate
    List<RecognitionResult> results = new ArrayList<>();
    
    for (String word : candidates)
    {
      RecognitionResult result = calculateWordScore(word, swipePath, detectedLetters, context);
      if (result.totalScore > 0.001) // Minimum threshold
      {
        results.add(result);
      }
    }
    
    // Step 4: Sort by total score (Bayesian combination)
    Collections.sort(results, new Comparator<RecognitionResult>()
    {
      @Override
      public int compare(RecognitionResult a, RecognitionResult b)
      {
        return Double.compare(b.totalScore, a.totalScore); // Descending order
      }
    });
    
    android.util.Log.d("KeyboardSwipeRecognizer", "Recognition complete: " + results.size() + " results");
    return results;
  }
  
  /**
   * Detect letter sequence from swipe path using key proximity (FIXED)
   */
  private List<Character> detectLetterSequence(List<PointF> swipePath)
  {
    List<Character> sequence = new ArrayList<>();
    Character lastLetter = null;
    
    // FIXED: Sample more frequently and check all points if needed
    int sampleInterval = Math.max(1, (int)(pathSampleDistance / 5)); // Sample every 4 points instead of 20
    
    for (int i = 0; i < swipePath.size(); i += sampleInterval)
    {
      PointF point = swipePath.get(i);
      Character nearestKey = getNearestKey(point);
      
      // DEBUG: Log detection attempts
      android.util.Log.d("KeyboardSwipeRecognizer", String.format("Point (%.0f,%.0f) → key '%s'", 
                        point.x, point.y, nearestKey != null ? nearestKey : "null"));
      
      // Add to sequence if it's a new letter (avoid duplicates)
      if (nearestKey != null && !nearestKey.equals(lastLetter))
      {
        sequence.add(nearestKey);
        lastLetter = nearestKey;
        android.util.Log.d("KeyboardSwipeRecognizer", "Added letter '" + nearestKey + "' to sequence");
      }
    }
    
    android.util.Log.d("KeyboardSwipeRecognizer", "Final detected sequence: " + sequence);
    return sequence;
  }
  
  /**
   * Find nearest key to a point (within key zone radius)
   */
  private Character getNearestKey(PointF point)
  {
    double minDistance = Double.MAX_VALUE;
    Character nearestKey = null;
    
    // Check all keyboard letters (would use actual key positions)
    String allLetters = "qwertyuiopasdfghjklzxcvbnm";
    for (char c : allLetters.toCharArray())
    {
      PointF keyCenter = getKeyCenter(c);
      if (keyCenter != null)
      {
        double distance = Math.sqrt(Math.pow(point.x - keyCenter.x, 2) + Math.pow(point.y - keyCenter.y, 2));
        
        // Only consider if within key zone radius
        if (distance <= keyZoneRadius && distance < minDistance)
        {
          minDistance = distance;
          nearestKey = c;
        }
      }
    }
    
    return nearestKey;
  }
  
  /**
   * Get key center coordinates (INTEGRATED with templateGenerator)
   */
  private PointF getKeyCenter(char letter)
  {
    ContinuousGestureRecognizer.Point coord = templateGenerator.getCharacterCoordinate(letter);
    return coord != null ? new PointF((float)coord.x, (float)coord.y) : null;
  }
  
  /**
   * Set keyboard dimensions for dynamic coordinate calculation
   */
  public void setKeyboardDimensions(float width, float height)
  {
    templateGenerator.setKeyboardDimensions(width, height);
    android.util.Log.d("KeyboardSwipeRecognizer", "Keyboard dimensions set: " + width + "x" + height);
  }
  
  /**
   * Generate candidate words using existing template generation (REUSED CODE)
   */
  private List<String> generateCandidateWords(List<Character> detectedLetters)
  {
    List<String> candidates = new ArrayList<>();
    
    if (detectedLetters.isEmpty()) return candidates;
    
    // REUSE: Get all templates from existing generator
    List<ContinuousGestureRecognizer.Template> allTemplates = 
      templateGenerator.generateBalancedWordTemplates(3000);
    
    // Filter to words containing detected letters in approximate order
    for (ContinuousGestureRecognizer.Template template : allTemplates)
    {
      String word = template.id.toLowerCase();
      
      // Check if word contains most detected letters
      if (wordContainsLetters(word, detectedLetters))
      {
        candidates.add(word);
      }
    }
    
    android.util.Log.d("KeyboardSwipeRecognizer", "Filtered to " + candidates.size() + 
                      " candidates from detected letters: " + detectedLetters);
    
    return candidates;
  }
  
  /**
   * Check if word contains most of the detected letters (REUSED logic)
   */
  private boolean wordContainsLetters(String word, List<Character> detectedLetters)
  {
    if (detectedLetters.isEmpty()) return false;
    
    int matchCount = 0;
    int lastFoundIndex = -1;
    
    // Check if detected letters appear in word in approximate order
    for (Character letter : detectedLetters)
    {
      int foundIndex = word.indexOf(letter, lastFoundIndex + 1);
      if (foundIndex != -1)
      {
        matchCount++;
        lastFoundIndex = foundIndex;
      }
    }
    
    // Require at least 60% of detected letters to match
    double matchRatio = (double)matchCount / detectedLetters.size();
    return matchRatio >= 0.6;
  }
  
  /**
   * Calculate comprehensive word score using keyboard-specific Bayesian equation
   */
  private RecognitionResult calculateWordScore(String word, List<PointF> swipePath, 
                                             List<Character> detectedLetters, List<String> context)
  {
    // P(swipe | word) - Likelihood calculation
    double proximityScore = calculateProximityScore(word, swipePath);
    double sequenceScore = calculateSequenceScore(word, detectedLetters);
    double startPointScore = calculateStartPointScore(word, swipePath);
    
    // Combine likelihood components
    double likelihood = proximityScore * sequenceScore * startPointScore;
    
    // P(word) - Prior probability  
    double languageModelScore = calculateLanguageModelScore(word, context);
    
    // Bayesian combination: P(word | swipe) ∝ P(swipe | word) × P(word)
    double totalScore = likelihood * languageModelScore;
    
    return new RecognitionResult(word, totalScore, proximityScore, sequenceScore, 
                               startPointScore, languageModelScore, detectedLetters);
  }
  
  /**
   * Calculate how well swipe path matches word's key positions (REUSED distance logic)
   */
  private double calculateProximityScore(String word, List<PointF> swipePath)
  {
    // REUSE: Get word template for key positions
    ContinuousGestureRecognizer.Template wordTemplate = templateGenerator.generateWordTemplate(word);
    if (wordTemplate == null) return 0.0;
    
    double totalScore = 0.0;
    int validComparisons = 0;
    
    // Compare swipe path proximity to word's key sequence
    for (int i = 0; i < Math.min(swipePath.size(), wordTemplate.pts.size()); i++)
    {
      PointF swipePoint = swipePath.get(i);
      ContinuousGestureRecognizer.Point templatePoint = wordTemplate.pts.get(i % wordTemplate.pts.size());
      
      // REUSE: Distance calculation logic from CGR
      double distance = Math.sqrt(Math.pow(swipePoint.x - templatePoint.x, 2) + 
                                 Math.pow(swipePoint.y - templatePoint.y, 2));
      
      // Convert distance to proximity score (closer = higher score)
      double proximityScore = Math.exp(-distance / keyZoneRadius);
      
      // Apply start point emphasis (users begin precisely)
      double pathPosition = (double)i / swipePath.size();
      double positionWeight = startPointWeight * (1.0 - pathPosition) + 1.0; // Higher weight at start
      
      totalScore += proximityScore * positionWeight;
      validComparisons++;
    }
    
    return validComparisons > 0 ? totalScore / validComparisons : 0.0;
  }
  
  /**
   * Calculate how well detected letters match word's letter sequence (IMPLEMENTED)
   */
  private double calculateSequenceScore(String word, List<Character> detectedLetters)
  {
    if (detectedLetters.isEmpty()) return 0.0;
    
    double score = 1.0;
    
    // Check each required letter in word
    int detectedIndex = 0;
    for (char requiredLetter : word.toCharArray())
    {
      boolean found = false;
      
      // Look for this letter in remaining detected sequence
      for (int i = detectedIndex; i < detectedLetters.size(); i++)
      {
        if (detectedLetters.get(i) == requiredLetter)
        {
          found = true;
          detectedIndex = i + 1; // Move forward for next letter
          break;
        }
      }
      
      // PENALTY: Missing required letter (high penalty)
      if (!found)
      {
        score *= Math.exp(-missingKeyPenalty);
        android.util.Log.d("KeyboardSwipeRecognizer", "Missing letter penalty for '" + requiredLetter + "' in " + word);
      }
    }
    
    // PENALTY: Extra letters not in word (lower penalty)
    for (Character detectedLetter : detectedLetters)
    {
      if (word.indexOf(detectedLetter) == -1)
      {
        score *= Math.exp(-extraKeyPenalty);
        android.util.Log.d("KeyboardSwipeRecognizer", "Extra letter penalty for '" + detectedLetter + "' not in " + word);
      }
    }
    
    // ORDER PENALTY: Check sequence order
    String detectedSequence = "";
    for (Character c : detectedLetters) detectedSequence += c;
    
    if (!isSubsequence(word, detectedSequence))
    {
      score *= Math.exp(-orderPenalty);
      android.util.Log.d("KeyboardSwipeRecognizer", "Order penalty for " + word + " vs detected " + detectedSequence);
    }
    
    return score;
  }
  
  /**
   * Check if detected sequence is a subsequence of word (order matters)
   */
  private boolean isSubsequence(String word, String sequence)
  {
    int wordIndex = 0;
    for (char c : sequence.toCharArray())
    {
      while (wordIndex < word.length() && word.charAt(wordIndex) != c)
      {
        wordIndex++;
      }
      if (wordIndex >= word.length()) return false;
      wordIndex++;
    }
    return true;
  }
  
  /**
   * Calculate start point accuracy (emphasized over end point) - YOUR INSIGHT
   */
  private double calculateStartPointScore(String word, List<PointF> swipePath)
  {
    if (swipePath.isEmpty() || word.isEmpty()) return 0.0;
    
    // Get first letter key position for word
    char firstLetter = word.charAt(0);
    PointF firstKeyCenter = getKeyCenter(firstLetter);
    if (firstKeyCenter == null) return 0.0;
    
    // Calculate distance from swipe start to first key
    PointF swipeStart = swipePath.get(0);
    double startDistance = Math.sqrt(Math.pow(swipeStart.x - firstKeyCenter.x, 2) + 
                                   Math.pow(swipeStart.y - firstKeyCenter.y, 2));
    
    // Convert to score (closer = higher score, with start emphasis)
    double startScore = Math.exp(-startDistance / keyZoneRadius);
    
    // EMPHASIS: Start point much more important than end point
    return Math.pow(startScore, startPointWeight); // Amplify start point accuracy
  }
  
  /**
   * Calculate language model score using existing infrastructure (INTEGRATED)
   */
  private double calculateLanguageModelScore(String word, List<String> context)
  {
    double score = 1.0;
    
    try
    {
      // Base word frequency (unigram)
      // TODO: Integrate with actual word frequency from templateGenerator
      double baseFrequency = 1.0; // Placeholder
      
      // Contextual probability (bigram/n-gram)
      if (bigramModel != null && context != null && !context.isEmpty())
      {
        String previousWord = context.get(context.size() - 1);
        float contextMultiplier = bigramModel.getContextMultiplier(word, context);
        score *= contextMultiplier;
      }
      
      // User adaptation (personal usage patterns)
      if (userAdaptation != null)
      {
        float adaptationMultiplier = userAdaptation.getAdaptationMultiplier(word);
        score *= adaptationMultiplier;
      }
      
      android.util.Log.d("KeyboardSwipeRecognizer", "Language model score for '" + word + "': " + score);
    }
    catch (Exception e)
    {
      android.util.Log.w("KeyboardSwipeRecognizer", "Language model error for " + word + ": " + e.getMessage());
    }
    
    return score;
  }
  
  /**
   * Set configurable weights for cost function
   */
  public void setWeights(double proximity, double missingKey, double extraKey, double order, double startPoint)
  {
    proximityWeight = proximity;
    missingKeyPenalty = missingKey;
    extraKeyPenalty = extraKey;
    orderPenalty = order;
    startPointWeight = startPoint;
    
    android.util.Log.d("KeyboardSwipeRecognizer", String.format("Weights updated: prox=%.1f, miss=%.1f, extra=%.1f, order=%.1f, start=%.1f",
                      proximity, missingKey, extraKey, order, startPoint));
  }
}