package juloo.keyboard2;

import android.graphics.PointF;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Orchestrates swipe typing prediction using multiple strategies
 */
public class SwipeTypingEngine
{
  private final KeyboardSwipeRecognizer _cgrRecognizer;
  private final WordPredictor _sequencePredictor;
  private final SwipeDetector _swipeDetector;
  private final SwipeScorer _scorer;
  private Config _config;
  private android.content.Context _context;
  
  public SwipeTypingEngine(android.content.Context context, WordPredictor sequencePredictor, Config config)
  {
    _context = context;
    _cgrRecognizer = new KeyboardSwipeRecognizer(context);
    _sequencePredictor = sequencePredictor;
    _swipeDetector = new SwipeDetector();
    _scorer = new SwipeScorer();
    _config = config;
    
    // Ensure predictors have config
    if (_sequencePredictor != null)
    {
      _sequencePredictor.setConfig(config);
    }
  }
  
  /**
   * Set keyboard dimensions for CGR recognizer
   */
  public void setKeyboardDimensions(float width, float height)
  {
    if (_cgrRecognizer != null)
    {
      _cgrRecognizer.setKeyboardDimensions(width, height);
    }
  }
  
  /**
   * Update configuration
   */
  public void setConfig(Config config)
  {
    _config = config;
    if (_sequencePredictor != null)
    {
      _sequencePredictor.setConfig(config);
    }
    _scorer.setConfig(config);
  }
  
  /**
   * Main prediction method
   */
  public WordPredictor.PredictionResult predict(SwipeInput input)
  {
    android.util.Log.d("SwipeTypingEngine", "=== PREDICTION START ===");
    android.util.Log.d("SwipeTypingEngine", String.format(
      "Input: keySeq=%s, pathLen=%.1f, duration=%.2fs, dirChanges=%d",
      input.keySequence, input.pathLength, input.duration, input.directionChanges));
    
    // Classify the input
    SwipeDetector.SwipeClassification classification = _swipeDetector.classifyInput(input);
    
    android.util.Log.d("SwipeTypingEngine", String.format(
      "Classification: isSwipe=%s, confidence=%.2f, quality=%s",
      classification.isSwipe, classification.confidence, classification.quality));
    
    // Choose prediction strategy based on classification
    if (!classification.isSwipe)
    {
      // Regular typing - use sequence predictor only
      android.util.Log.d("SwipeTypingEngine", "Using sequence prediction (not a swipe)");
      return _sequencePredictor.predictWordsWithScores(input.keySequence);
    }
    
    // For swipes, use hybrid prediction based on quality
    if (_swipeDetector.shouldUseDTW(classification))
    {
      android.util.Log.d("SwipeTypingEngine", "Using hybrid DTW + sequence prediction");
      return hybridPredict(input, classification);
    }
    else
    {
      android.util.Log.d("SwipeTypingEngine", "Using enhanced sequence prediction (low quality swipe)");
      return enhancedSequencePredict(input, classification);
    }
  }
  
  /**
   * Hybrid prediction combining DTW and sequence matching
   */
  private WordPredictor.PredictionResult hybridPredict(SwipeInput input, 
                                                       SwipeDetector.SwipeClassification classification)
  {
    List<ScoredCandidate> allCandidates = new ArrayList<>();
    
    // Get CGR predictions using working KeyboardSwipeRecognizer  
    if (_cgrRecognizer != null && input.coordinates.size() > 2)
    {
      try
      {
        // CRITICAL: Set keyboard dimensions for coordinate mapping (was missing!)
        android.util.DisplayMetrics metrics = new android.util.DisplayMetrics();
        android.view.WindowManager wm = (android.view.WindowManager) 
          _context.getSystemService(android.content.Context.WINDOW_SERVICE);
        wm.getDefaultDisplay().getMetrics(metrics);
        float keyboardWidth = metrics.widthPixels;
        float keyboardHeight = metrics.heightPixels * 0.35f; // Typical keyboard height
        
        _cgrRecognizer.setKeyboardDimensions(keyboardWidth, keyboardHeight);
        android.util.Log.d("SwipeTypingEngine", "Set CGR keyboard dimensions: " + keyboardWidth + "x" + keyboardHeight);
        
        List<KeyboardSwipeRecognizer.RecognitionResult> cgrResults = 
          _cgrRecognizer.recognizeSwipe(input.coordinates, new ArrayList<>());
        
        android.util.Log.d("SwipeTypingEngine", "CGR predictions: " + cgrResults.size() + " results");
        android.util.Log.d("SwipeTypingEngine", "Input coordinates: " + input.coordinates.size() + " points");
        
        // Log first few coordinate points for debugging
        if (input.coordinates.size() > 0) {
          android.graphics.PointF first = input.coordinates.get(0);
          android.graphics.PointF last = input.coordinates.get(input.coordinates.size() - 1);
          android.util.Log.d("SwipeTypingEngine", "Swipe path: (" + first.x + "," + first.y + ") → (" + last.x + "," + last.y + ")");
        }
        
        // Convert to scored candidates
        for (KeyboardSwipeRecognizer.RecognitionResult result : cgrResults)
        {
          allCandidates.add(new ScoredCandidate(result.word, (float)result.totalScore, "CGR"));
          android.util.Log.d("SwipeTypingEngine", "CGR result: " + result.word + " = " + result.totalScore);
        }
        
        // Log error report from CGR recognizer
        if (cgrResults.isEmpty()) {
          android.util.Log.w("SwipeTypingEngine", "No CGR results! Error report: " + _cgrRecognizer.lastErrorReport);
        }
      }
      catch (Exception e)
      {
        android.util.Log.e("SwipeTypingEngine", "CGR prediction failed: " + e.getMessage());
      }
    }
    
    // Get sequence predictions
    WordPredictor.PredictionResult sequenceResult = _sequencePredictor.predictWordsWithScores(input.keySequence);
    android.util.Log.d("SwipeTypingEngine", "Sequence predictions: " + sequenceResult.words);
    
    // Add sequence predictions to candidates
    for (int i = 0; i < sequenceResult.words.size(); i++)
    {
      String word = sequenceResult.words.get(i);
      int baseScore = sequenceResult.scores.get(i);
      
      // Check if word already exists from DTW
      ScoredCandidate existing = findCandidate(allCandidates, word);
      if (existing != null)
      {
        // Combine scores - word appears in both predictions
        existing.score += baseScore * 0.5f; // Boost for appearing in both
        existing.source = "Both";
      }
      else
      {
        allCandidates.add(new ScoredCandidate(word, baseScore, "Sequence"));
      }
    }
    
    // Apply unified scoring with all weights
    for (ScoredCandidate candidate : allCandidates)
    {
      candidate.finalScore = _scorer.calculateFinalScore(candidate, input, classification, _config);
    }
    
    // Sort by final score
    Collections.sort(allCandidates, new Comparator<ScoredCandidate>() {
      @Override
      public int compare(ScoredCandidate a, ScoredCandidate b) {
        return Float.compare(b.finalScore, a.finalScore);
      }
    });
    
    // Convert to result format with swipe-specific filtering
    List<String> words = new ArrayList<>();
    List<Integer> scores = new ArrayList<>();
    
    int maxResults = _config.swipe_typing_enabled ? 10 : 5;
    int resultCount = 0;
    for (ScoredCandidate candidate : allCandidates)
    {
      // DESIGN SPEC: Swipe predictions must be ≥3 characters minimum
      if (candidate.word.length() < 3)
      {
        android.util.Log.d("SwipeTypingEngine", String.format(
          "Filtered short word: '%s' (length=%d)", candidate.word, candidate.word.length()));
        continue;  // Skip 1-2 character words for swipe predictions
      }
      
      words.add(candidate.word);
      scores.add((int)candidate.finalScore);
      
      android.util.Log.d("SwipeTypingEngine", String.format(
        "Result %d: %s (score=%.0f, source=%s)", 
        resultCount + 1, candidate.word, candidate.finalScore, candidate.source));
      
      resultCount++;
      if (resultCount >= maxResults) break;
    }
    
    android.util.Log.d("SwipeTypingEngine", "=== PREDICTION END ===");
    return new WordPredictor.PredictionResult(words, scores);
  }
  
  /**
   * Enhanced sequence prediction for low-quality swipes
   */
  private WordPredictor.PredictionResult enhancedSequencePredict(SwipeInput input,
                                                                 SwipeDetector.SwipeClassification classification)
  {
    // Get base predictions
    WordPredictor.PredictionResult result = _sequencePredictor.predictWordsWithScores(input.keySequence);
    
    // Apply swipe-specific filtering and scoring adjustments
    List<String> filteredWords = new ArrayList<>();
    List<Integer> adjustedScores = new ArrayList<>();
    
    for (int i = 0; i < result.words.size(); i++)
    {
      String word = result.words.get(i);
      
      // DESIGN SPEC: Swipe predictions must be ≥3 characters minimum
      if (word.length() < 3)
      {
        android.util.Log.d("SwipeTypingEngine", String.format(
          "Filtered short word in enhanced sequence: '%s' (length=%d)", word, word.length()));
        continue;  // Skip 1-2 character words for swipe predictions
      }
      
      int baseScore = result.scores.get(i);
      // Apply confidence-based adjustment
      float adjustment = 1.0f + (classification.confidence * 0.5f);
      
      filteredWords.add(word);
      adjustedScores.add((int)(baseScore * adjustment));
    }
    
    return new WordPredictor.PredictionResult(filteredWords, adjustedScores);
  }
  
  /**
   * Find candidate in list by word
   */
  private ScoredCandidate findCandidate(List<ScoredCandidate> candidates, String word)
  {
    for (ScoredCandidate c : candidates)
    {
      if (c.word.equals(word))
        return c;
    }
    return null;
  }
  
  /**
   * Internal class for scored candidates
   */
  public static class ScoredCandidate
  {
    public String word;
    public float score;
    public float finalScore;
    public String source;
    
    public ScoredCandidate(String word, float score, String source)
    {
      this.word = word;
      this.score = score;
      this.finalScore = score;
      this.source = source;
    }
  }
}