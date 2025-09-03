package juloo.keyboard2;

import android.content.Context;
import android.content.SharedPreferences;
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
  // Keyboard-specific cost function weights (user-configurable, PUBLIC for transparency)
  public double proximityWeight = 1.0;      // α - Distance to key centers
  public double missingKeyPenalty = 10.0;   // β - Missing required letters
  public double extraKeyPenalty = 2.0;      // γ - Passing over wrong letters  
  public double orderPenalty = 5.0;         // δ - Out-of-order letter sequence
  public double startPointWeight = 3.0;     // ε - Start point accuracy emphasis
  
  // Infrastructure components (reusing existing code)
  private WordGestureTemplateGenerator templateGenerator;
  private BigramModel bigramModel;
  private NgramModel ngramModel;
  private UserAdaptationManager userAdaptation;
  private SwipeWeightConfig weightConfig;
  private GaussianKeyModel gaussianModel;
  
  // Key zone parameters (PUBLIC for transparency)
  public double keyZoneRadius = 180.0;      // Larger zone for easier key detection  
  public double pathSampleDistance = 5.0;   // More frequent sampling for better detection
  
  public KeyboardSwipeRecognizer(Context context)
  {
    try
    {
      templateGenerator = new WordGestureTemplateGenerator();
      templateGenerator.loadDictionary(context);
      android.util.Log.d("KeyboardSwipeRecognizer", "Template generator initialized");
    }
    catch (Exception e)
    {
      android.util.Log.e("KeyboardSwipeRecognizer", "Template generator failed: " + e.getMessage());
      templateGenerator = null;
    }
    
    // Initialize components with error handling
    try { bigramModel = new BigramModel(); } 
    catch (Exception e) { android.util.Log.w("KeyboardSwipeRecognizer", "BigramModel failed: " + e.getMessage()); }
    
    try { ngramModel = new NgramModel(); } 
    catch (Exception e) { android.util.Log.w("KeyboardSwipeRecognizer", "NgramModel failed: " + e.getMessage()); }
    
    try { userAdaptation = UserAdaptationManager.getInstance(context); } 
    catch (Exception e) { android.util.Log.w("KeyboardSwipeRecognizer", "UserAdaptation failed: " + e.getMessage()); }
    
    try { weightConfig = SwipeWeightConfig.getInstance(context); } 
    catch (Exception e) { android.util.Log.w("KeyboardSwipeRecognizer", "WeightConfig failed: " + e.getMessage()); }
    
    try { gaussianModel = new GaussianKeyModel(); } 
    catch (Exception e) { android.util.Log.w("KeyboardSwipeRecognizer", "GaussianModel failed: " + e.getMessage()); }
    
    // Load weights from settings
    loadWeightsFromSettings(context);
    
    android.util.Log.d("KeyboardSwipeRecognizer", "Keyboard-specific recognition algorithm initialized (with error handling)");
  }
  
  /**
   * Load algorithm weights from settings
   */
  private void loadWeightsFromSettings(Context context)
  {
    try
    {
      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(context);
      
      // Load new algorithm weights
      proximityWeight = prefs.getInt("proximity_weight", 100) / 100.0;
      missingKeyPenalty = prefs.getInt("missing_key_penalty", 1000) / 100.0;
      extraKeyPenalty = prefs.getInt("extra_key_penalty", 200) / 100.0;
      orderPenalty = prefs.getInt("order_penalty", 500) / 100.0;
      startPointWeight = prefs.getInt("start_point_weight", 300) / 100.0;
      keyZoneRadius = prefs.getInt("key_zone_radius", 120);
      pathSampleDistance = prefs.getInt("path_sample_distance", 10);
      
      android.util.Log.d("KeyboardSwipeRecognizer", String.format("Weights loaded: prox=%.2f, miss=%.2f, extra=%.2f, order=%.2f, start=%.2f, zone=%.0f, sample=%.0f",
                        proximityWeight, missingKeyPenalty, extraKeyPenalty, orderPenalty, startPointWeight, keyZoneRadius, pathSampleDistance));
    }
    catch (Exception e)
    {
      android.util.Log.w("KeyboardSwipeRecognizer", "Failed to load settings, using defaults: " + e.getMessage());
    }
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
   * Full recognition method with comprehensive error reporting
   */
  public List<RecognitionResult> recognizeSwipe(List<PointF> swipePath, List<String> context)
  {
    List<RecognitionResult> results = new ArrayList<>();
    StringBuilder errorReport = new StringBuilder();
    
    errorReport.append("🔍 ALGORITHM VALIDATION:\n");
    
    // Step 0: Comprehensive input validation with error reporting
    if (swipePath == null) {
      errorReport.append("❌ CRITICAL: swipePath is NULL\n");
      logError(errorReport.toString());
      return results;
    }
    
    if (swipePath.size() == 0) {
      errorReport.append("❌ CRITICAL: 0 swipe points detected\n");
      logError(errorReport.toString());
      return results;
    }
    
    if (swipePath.size() < 2) {
      errorReport.append("⚠️  WARNING: Only " + swipePath.size() + " swipe point (need 2+)\n");
    }
    
    if (templateGenerator == null) {
      errorReport.append("❌ CRITICAL: templateGenerator is NULL\n");
      logError(errorReport.toString());
      return results;
    }
    
    errorReport.append("✅ Input validation: " + swipePath.size() + " swipe points, generator OK\n\n");
    
    // Step 1: Letter detection with error reporting
    errorReport.append("🔤 LETTER DETECTION:\n");
    List<Character> detectedLetters = detectLetterSequenceWithReporting(swipePath, errorReport);
    
    // Step 2: Candidate generation with error reporting  
    errorReport.append("\n📝 CANDIDATE GENERATION:\n");
    List<String> candidates = generateCandidateWordsWithReporting(detectedLetters, errorReport);
    
    // Step 3: Scoring with error reporting
    errorReport.append("\n📊 SCORING CALCULATION:\n");
    results = scoreWordsWithReporting(candidates, swipePath, detectedLetters, context, errorReport);
    
    // Final report
    errorReport.append("\n🎯 FINAL RESULTS: " + results.size() + " predictions generated\n");
    logError(errorReport.toString());
    
    return results;
  }
  
  /**
   * Simple proximity score between swipe and word template
   */
  private double calculateSimpleProximityScore(String word, List<PointF> swipePath)
  {
    try {
      ContinuousGestureRecognizer.Template template = templateGenerator.generateWordTemplate(word);
      if (template == null || template.pts.isEmpty()) return 0.0;
      
      double totalDistance = 0.0;
      int comparisons = 0;
      
      // Compare swipe points to template points
      int templateSize = template.pts.size();
      for (int i = 0; i < swipePath.size(); i++) {
        PointF swipePoint = swipePath.get(i);
        int templateIndex = (i * templateSize) / swipePath.size(); // Map to template
        
        // NULL SAFE: Check template bounds before access
        if (templateIndex >= template.pts.size()) templateIndex = template.pts.size() - 1;
        
        ContinuousGestureRecognizer.Point templatePoint = template.pts.get(templateIndex);
        double distance = Math.sqrt(Math.pow(swipePoint.x - templatePoint.x, 2) + 
                                   Math.pow(swipePoint.y - templatePoint.y, 2));
        totalDistance += distance;
        comparisons++;
      }
      
      if (comparisons == 0) return 0.0;
      
      double avgDistance = totalDistance / comparisons;
      // Convert distance to score (closer = higher score)
      return Math.exp(-avgDistance / 200.0); // Reasonable distance threshold
      
    } catch (Exception e) {
      android.util.Log.e("KeyboardSwipeRecognizer", "Error scoring word '" + word + "': " + e.getMessage());
      return 0.0;
    }
  }
  
  /**
   * Detect letter sequence from swipe path using key proximity (FIXED)
   */
  private List<Character> detectLetterSequence(List<PointF> swipePath)
  {
    List<Character> sequence = new ArrayList<>();
    Character lastLetter = null;
    PointF lastValidPoint = null;
    
    // IMPROVED: Much more selective sampling - target 3-8 letters max per word
    int sampleInterval = Math.max(20, swipePath.size() / 8); // Sample ~8 points across entire path
    double minDistanceForNewLetter = keyZoneRadius * 0.8; // Must move significantly to register new letter
    
    for (int i = 0; i < swipePath.size(); i += sampleInterval)
    {
      PointF point = swipePath.get(i);
      Character nearestKey = getNearestKey(point);
      
      if (nearestKey != null && !nearestKey.equals(lastLetter))
      {
        // FILTER: Require significant movement to register new letter
        boolean significantMovement = true;
        if (lastValidPoint != null) {
          double distanceFromLast = Math.sqrt(
            Math.pow(point.x - lastValidPoint.x, 2) + 
            Math.pow(point.y - lastValidPoint.y, 2));
          significantMovement = distanceFromLast >= minDistanceForNewLetter;
        }
        
        if (significantMovement) {
          sequence.add(nearestKey);
          lastLetter = nearestKey;
          lastValidPoint = new PointF(point.x, point.y);
          android.util.Log.d("KeyboardSwipeRecognizer", String.format("Added letter '%c' at (%.0f,%.0f) after %.0fpx movement", 
                            nearestKey, point.x, point.y, lastValidPoint != null ? 
                            Math.sqrt(Math.pow(point.x - lastValidPoint.x, 2) + Math.pow(point.y - lastValidPoint.y, 2)) : 0));
        }
      }
    }
    
    // SAFETY: Limit to reasonable word length (3-12 characters)
    if (sequence.size() > 12) {
      android.util.Log.w("KeyboardSwipeRecognizer", "Truncating over-detected sequence from " + 
                        sequence.size() + " to 12 letters");
      sequence = sequence.subList(0, 12);
    }
    
    android.util.Log.d("KeyboardSwipeRecognizer", "Final detected sequence: " + sequence + 
                      " (" + sequence.size() + " letters)");
    return sequence;
  }
  
  /**
   * Find nearest key to a point (within key zone radius) - ROBUST
   */
  private Character getNearestKey(PointF point)
  {
    if (templateGenerator == null)
    {
      android.util.Log.w("KeyboardSwipeRecognizer", "Template generator null - cannot detect keys");
      return null;
    }
    
    double minDistance = Double.MAX_VALUE;
    Character nearestKey = null;
    
    // Check all keyboard letters
    String allLetters = "qwertyuiopasdfghjklzxcvbnm";
    for (char c : allLetters.toCharArray())
    {
      PointF keyCenter = getKeyCenter(c);
      if (keyCenter != null)
      {
        double distance = Math.sqrt(Math.pow(point.x - keyCenter.x, 2) + Math.pow(point.y - keyCenter.y, 2));
        
        // DEBUG: Log distance calculations for troubleshooting
        if (distance <= keyZoneRadius * 2) // Log nearby keys
        {
          android.util.Log.v("KeyboardSwipeRecognizer", String.format("Point (%.0f,%.0f) to key '%c' at (%.0f,%.0f) = %.0f px", 
                            point.x, point.y, c, keyCenter.x, keyCenter.y, distance));
        }
        
        // Only consider if within key zone radius
        if (distance <= keyZoneRadius && distance < minDistance)
        {
          minDistance = distance;
          nearestKey = c;
        }
      }
      else
      {
        android.util.Log.v("KeyboardSwipeRecognizer", "No coordinate for key '" + c + "'");
      }
    }
    
    if (nearestKey != null)
    {
      android.util.Log.d("KeyboardSwipeRecognizer", String.format("Point (%.0f,%.0f) → key '%c' (distance %.0f)", 
                        point.x, point.y, nearestKey, minDistance));
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
    if (templateGenerator == null) {
      android.util.Log.e("KeyboardSwipeRecognizer", "Template generator is NULL - cannot set dimensions");
      return;
    }
    
    templateGenerator.setKeyboardDimensions(width, height);
    android.util.Log.e("KeyboardSwipeRecognizer", "Keyboard dimensions set: " + width + "x" + height);
  }
  
  /**
   * Set real key positions for 100% accurate coordinate mapping (CRITICAL FIX)
   * Replaces simplified grid calculations with actual keyboard layout positions
   */
  public void setRealKeyPositions(java.util.Map<Character, android.graphics.PointF> realPositions)
  {
    if (templateGenerator != null && realPositions != null) {
      templateGenerator.setRealKeyPositions(realPositions);
      android.util.Log.e("KeyboardSwipeRecognizer", "✅ SET REAL KEY POSITIONS: " + realPositions.size() + " keys");
      
      // Log key positions for verification
      if (realPositions.containsKey('a')) {
        android.graphics.PointF aPos = realPositions.get('a');
        android.util.Log.e("KeyboardSwipeRecognizer", "📍 Real 'a' position: (" + aPos.x + "," + aPos.y + ")");
      }
      if (realPositions.containsKey('m')) {
        android.graphics.PointF mPos = realPositions.get('m');
        android.util.Log.e("KeyboardSwipeRecognizer", "📍 Real 'm' position: (" + mPos.x + "," + mPos.y + ")");
      }
    }
  }
  
  /**
   * Generate candidate words using existing template generation (REUSED CODE)
   */
  private List<String> generateCandidateWords(List<Character> detectedLetters)
  {
    List<String> candidates = new ArrayList<>();
    
    if (detectedLetters.isEmpty()) return candidates;
    
    // FIXED: Use direct dictionary access instead of generating 3000 templates
    if (templateGenerator == null) return candidates;
    
    List<String> dictionary = templateGenerator.getDictionary();
    android.util.Log.d("KeyboardSwipeRecognizer", "Using dictionary with " + dictionary.size() + " words");
    
    // Filter dictionary words containing detected letters (much faster)
    for (String word : dictionary)
    {
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
   * Check if word contains detected letters (STRICTER filtering per Gemini)
   */
  private boolean wordContainsLetters(String word, List<Character> detectedLetters)
  {
    if (detectedLetters.isEmpty() || word.isEmpty()) return false;
    
    // FIXED: Proper null check to prevent crashes
    if (detectedLetters.isEmpty()) {
      android.util.Log.d("KeyboardSwipeRecognizer", "No detected letters - word '" + word + "' accepted by default");
      return true; // Accept all words when no letters detected
    }
    
    // Simple letter matching without strict requirements
    int matchCount = 0;
    for (char wordChar : word.toCharArray()) {
      for (Character detectedChar : detectedLetters) {
        if (wordChar == detectedChar) {
          matchCount++;
          break;
        }
      }
    }
    
    // Accept word if it contains any detected letters  
    boolean hasMatch = matchCount > 0;
    android.util.Log.d("KeyboardSwipeRecognizer", String.format("Word '%s': %d/%d letters match, accepted=%s", 
                      word, matchCount, word.length(), hasMatch));
    return hasMatch;
    
    /*
    // REMOVED: Overly strict filtering causing crashes
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
    
    // RELAXED: Require 50% match for reasonable candidate generation  
    double matchRatio = (double)matchCount / detectedLetters.size();
    return matchRatio >= 0.5;
    */
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
    
    // CRITICAL DEBUG: Log all score components to identify 0.000000 issue
    android.util.Log.e("KeyboardSwipeRecognizer", String.format("SCORE DEBUG '%s': prox=%.6f seq=%.6f start=%.6f lang=%.6f → total=%.6f",
                      word, proximityScore, sequenceScore, startPointScore, languageModelScore, totalScore));
    
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
    
    // Calculate distance from swipe start to first key (NULL SAFE)
    if (swipePath.isEmpty()) return 0.0;
    
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
      // Base word frequency (unigram) - INTEGRATED
      double baseFrequency = templateGenerator != null ? 
        templateGenerator.getWordFrequency(word) / 1000.0 : 1.0; // Normalize frequency
      
      // Contextual probability (bigram/n-gram) - NULL SAFE
      if (bigramModel != null && context != null && !context.isEmpty() && context.size() > 0)
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
  
  /**
   * Get transparent letter detection for debugging
   */
  public List<Character> getDetectedLetters(List<PointF> swipePath)
  {
    return detectLetterSequence(swipePath);
  }
  
  /**
   * Get transparent candidate generation for debugging
   */
  public List<String> getCandidates(List<Character> detectedLetters)
  {
    return generateCandidateWords(detectedLetters);
  }
  
  /**
   * Get transparent scoring breakdown for debugging
   */
  public RecognitionResult getDetailedScore(String word, List<PointF> swipePath, List<Character> detectedLetters)
  {
    return calculateWordScore(word, swipePath, detectedLetters, new ArrayList<>());
  }
  
  /**
   * Live parameter setters for real-time tuning
   */
  public void setKeyZoneRadius(double radius)
  {
    keyZoneRadius = radius;
    android.util.Log.d("KeyboardSwipeRecognizer", "Key Zone Radius updated to: " + radius);
  }
  
  public void setMissingKeyPenalty(double penalty)
  {
    missingKeyPenalty = penalty;
    android.util.Log.d("KeyboardSwipeRecognizer", "Missing Key Penalty updated to: " + penalty);
  }
  
  public void setStartPointWeight(double weight)
  {
    startPointWeight = weight;
    android.util.Log.d("KeyboardSwipeRecognizer", "Start Point Weight updated to: " + weight);
  }
  
  /**
   * Main algorithm component weight update methods
   */
  public void updateDTWWeight(float weight)
  {
    if (weightConfig != null) {
      // Get current weights and update DTW component
      float gaussian = weightConfig.getGaussianWeight();
      float ngram = weightConfig.getNgramWeight();
      float frequency = weightConfig.getFrequencyWeight();
      
      // Normalize to maintain sum = 1.0
      float total = weight + gaussian + ngram + frequency;
      if (total > 0) {
        weightConfig.saveWeights(weight/total, gaussian/total, ngram/total, frequency/total);
      }
      android.util.Log.d("KeyboardSwipeRecognizer", "DTW Weight updated to: " + weight);
    }
  }
  
  public void updateGaussianWeight(float weight)
  {
    if (weightConfig != null) {
      float dtw = weightConfig.getDtwWeight();
      float ngram = weightConfig.getNgramWeight();
      float frequency = weightConfig.getFrequencyWeight();
      
      float total = dtw + weight + ngram + frequency;
      if (total > 0) {
        weightConfig.saveWeights(dtw/total, weight/total, ngram/total, frequency/total);
      }
      android.util.Log.d("KeyboardSwipeRecognizer", "Gaussian Weight updated to: " + weight);
    }
  }
  
  public void updateNgramWeight(float weight)
  {
    if (weightConfig != null) {
      float dtw = weightConfig.getDtwWeight();
      float gaussian = weightConfig.getGaussianWeight();
      float frequency = weightConfig.getFrequencyWeight();
      
      float total = dtw + gaussian + weight + frequency;
      if (total > 0) {
        weightConfig.saveWeights(dtw/total, gaussian/total, weight/total, frequency/total);
      }
      android.util.Log.d("KeyboardSwipeRecognizer", "N-gram Weight updated to: " + weight);
    }
  }
  
  public void updateFrequencyWeight(float weight)
  {
    if (weightConfig != null) {
      float dtw = weightConfig.getDtwWeight();
      float gaussian = weightConfig.getGaussianWeight();
      float ngram = weightConfig.getNgramWeight();
      
      float total = dtw + gaussian + ngram + weight;
      if (total > 0) {
        weightConfig.saveWeights(dtw/total, gaussian/total, ngram/total, weight/total);
      }
      android.util.Log.d("KeyboardSwipeRecognizer", "Frequency Weight updated to: " + weight);
    }
  }
  
  /**
   * Comprehensive parameter setters for playground integration
   */
  public void setProximityWeight(double weight) { proximityWeight = weight; }
  public void setExtraKeyPenalty(double penalty) { extraKeyPenalty = penalty; }
  public void setOrderPenalty(double penalty) { orderPenalty = penalty; }
  public void setPathSampleDistance(double distance) { pathSampleDistance = distance; }
  
  /**
   * Apply all parameters from playground map
   */
  public void applyParameterMap(Map<String, Integer> params)
  {
    for (Map.Entry<String, Integer> entry : params.entrySet())
    {
      String paramName = entry.getKey();
      int value = entry.getValue();
      
      switch (paramName)
      {
        case "Key Zone Radius": setKeyZoneRadius(value); break;
        case "Missing Key Penalty": setMissingKeyPenalty(value / 100.0); break;
        case "Extra Key Penalty": setExtraKeyPenalty(value / 100.0); break;
        case "Order Penalty": setOrderPenalty(value / 100.0); break;
        case "Start Point Weight": setStartPointWeight(value / 100.0); break;
        case "Proximity Weight": setProximityWeight(value / 100.0); break;
        case "Path Sampling Rate": setPathSampleDistance(value); break;
        // Main algorithm component weights
        case "DTW Weight": updateDTWWeight(value / 100.0f); break;
        case "Gaussian Weight": updateGaussianWeight(value / 100.0f); break;
        case "N-gram Weight": updateNgramWeight(value / 100.0f); break;
        case "Frequency Weight": updateFrequencyWeight(value / 100.0f); break;
        default: android.util.Log.w("KeyboardSwipeRecognizer", "Unknown parameter: " + paramName);
      }
    }
    
    android.util.Log.d("KeyboardSwipeRecognizer", "Applied " + params.size() + " playground parameters");
  }
  
  /**
   * Log error to both LogCat and store for output text box display
   */
  private void logError(String message)
  {
    android.util.Log.e("KeyboardSwipeRecognizer", message);
    // Store error message for display in calibration text box
    lastErrorReport = message;
  }
  
  // Store last error report for display
  public String lastErrorReport = "";
  
  /**
   * Letter detection with comprehensive error reporting
   */
  private List<Character> detectLetterSequenceWithReporting(List<PointF> swipePath, StringBuilder errorReport)
  {
    List<Character> detectedLetters = new ArrayList<>();
    
    if (swipePath.isEmpty()) {
      errorReport.append("❌ No swipe points to analyze\n");
      return detectedLetters;
    }
    
    // Test keyboard coordinates
    PointF testPoint = swipePath.get(0);
    Character testKey = getNearestKey(testPoint);
    
    if (testKey == null) {
      errorReport.append("❌ CRITICAL: Key detection failed - no coordinates\n");
      errorReport.append("   First point (" + String.format("%.0f,%.0f", testPoint.x, testPoint.y) + ") → no key\n");
      
      // Test if template generator has any coordinates
      if (templateGenerator != null) {
        ContinuousGestureRecognizer.Point aCoord = templateGenerator.getCharacterCoordinate('a');
        if (aCoord == null) {
          errorReport.append("❌ Template generator has no coordinates for 'a'\n");
          errorReport.append("   Keyboard dimensions not set properly\n");
        } else {
          errorReport.append("✅ Template generator has coordinates: a=(" + 
                           String.format("%.0f,%.0f", aCoord.x, aCoord.y) + ")\n");
        }
      }
      return detectedLetters;
    }
    
    errorReport.append("✅ Key detection working: first point → '" + testKey + "'\n");
    
    // Perform full letter detection
    Character lastLetter = null;
    for (int i = 0; i < swipePath.size(); i += (int)pathSampleDistance) {
      PointF point = swipePath.get(i);
      Character nearestKey = getNearestKey(point);
      
      if (nearestKey != null && !nearestKey.equals(lastLetter)) {
        detectedLetters.add(nearestKey);
        lastLetter = nearestKey;
      }
    }
    
    errorReport.append("✅ Detected " + detectedLetters.size() + " letters: " + detectedLetters + "\n");
    return detectedLetters;
  }
  
  /**
   * Candidate generation with error reporting
   */
  private List<String> generateCandidateWordsWithReporting(List<Character> detectedLetters, StringBuilder errorReport)
  {
    List<String> candidates = new ArrayList<>();
    
    if (templateGenerator == null) {
      errorReport.append("❌ CRITICAL: Template generator is NULL\n");
      return candidates;
    }
    
    List<String> dictionary = templateGenerator.getDictionary();
    if (dictionary == null) {
      errorReport.append("❌ CRITICAL: Dictionary is NULL\n");
      return candidates;
    }
    
    if (dictionary.isEmpty()) {
      errorReport.append("❌ CRITICAL: Dictionary is empty\n");
      return candidates;
    }
    
    errorReport.append("✅ Dictionary loaded: " + dictionary.size() + " words\n");
    
    if (detectedLetters.isEmpty()) {
      errorReport.append("⚠️  No letters detected - accepting more candidate words\n");
      candidates.addAll(dictionary.subList(0, Math.min(200, dictionary.size()))); // Increased from 50 to 200
    } else {
      // Filter dictionary by detected letters
      for (String word : dictionary) {
        if (wordContainsLetters(word, detectedLetters)) {
          candidates.add(word);
        }
      }
    }
    
    errorReport.append("✅ Generated " + candidates.size() + " candidates\n");
    if (!candidates.isEmpty()) {
      errorReport.append("   Sample: " + candidates.subList(0, Math.min(5, candidates.size())) + "\n");
    }
    
    return candidates;
  }
  
  /**
   * Word scoring with error reporting
   */
  private List<RecognitionResult> scoreWordsWithReporting(List<String> candidates, List<PointF> swipePath, 
                                                         List<Character> detectedLetters, List<String> context, 
                                                         StringBuilder errorReport)
  {
    List<RecognitionResult> results = new ArrayList<>();
    
    if (candidates.isEmpty()) {
      errorReport.append("❌ CRITICAL: No candidates to score\n");
      return results;
    }
    
    errorReport.append("📊 Scoring " + candidates.size() + " candidates:\n");
    
    int scoredCount = 0;
    for (String word : candidates) {
      try {
        RecognitionResult result = calculateWordScore(word, swipePath, detectedLetters, context);
        results.add(result);
        scoredCount++;
        
        if (scoredCount <= 3) { // Show first 3 scores in detail
          errorReport.append(String.format("   %s: %.6f\n", word, result.totalScore));
        }
      } catch (Exception e) {
        errorReport.append("❌ Error scoring '" + word + "': " + e.getMessage() + "\n");
      }
    }
    
    if (scoredCount < candidates.size()) {
      errorReport.append("   ... " + (candidates.size() - scoredCount) + " more scored\n");
    }
    
    // Sort results
    Collections.sort(results, (a, b) -> Double.compare(b.totalScore, a.totalScore));
    
    errorReport.append("✅ Scoring complete: " + results.size() + " results\n");
    if (!results.isEmpty()) {
      RecognitionResult top = results.get(0);
      errorReport.append("   Top result: " + top.word + " (score: " + String.format("%.6f", top.totalScore) + ")\n");
    }
    
    return results;
  }
}