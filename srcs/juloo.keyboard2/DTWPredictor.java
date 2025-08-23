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
 * Dynamic Time Warping based word predictor for swipe typing
 * This provides more accurate swipe-to-word matching than simple sequence matching
 */
public class DTWPredictor
{
  private final Map<String, List<PointF>> _wordPaths;
  private final Map<String, Integer> _wordFrequencies;
  private final WordPredictor _fallbackPredictor;
  private SwipePruner _pruner;
  private GaussianKeyModel _gaussianModel;
  private NgramModel _ngramModel;
  private SwipeWeightConfig _weightConfig;
  private float _keyboardWidth = 1.0f;
  private float _keyboardHeight = 1.0f;
  
  // Number of points to sample/resample gestures to for consistent comparison.
  // FlorisBoard uses 200 points - this is CRITICAL for accuracy.
  // Using too few points (e.g. 10) loses 95% of gesture information!
  private static final int SAMPLING_POINTS = 200;
  
  // QWERTY layout key positions (normalized 0-1)
  private static final Map<Character, PointF> KEY_POSITIONS = new HashMap<>();
  
  static {
    // Top row
    KEY_POSITIONS.put('q', new PointF(0.05f, 0.25f));
    KEY_POSITIONS.put('w', new PointF(0.15f, 0.25f));
    KEY_POSITIONS.put('e', new PointF(0.25f, 0.25f));
    KEY_POSITIONS.put('r', new PointF(0.35f, 0.25f));
    KEY_POSITIONS.put('t', new PointF(0.45f, 0.25f));
    KEY_POSITIONS.put('y', new PointF(0.55f, 0.25f));
    KEY_POSITIONS.put('u', new PointF(0.65f, 0.25f));
    KEY_POSITIONS.put('i', new PointF(0.75f, 0.25f));
    KEY_POSITIONS.put('o', new PointF(0.85f, 0.25f));
    KEY_POSITIONS.put('p', new PointF(0.95f, 0.25f));
    
    // Middle row
    KEY_POSITIONS.put('a', new PointF(0.10f, 0.50f));
    KEY_POSITIONS.put('s', new PointF(0.20f, 0.50f));
    KEY_POSITIONS.put('d', new PointF(0.30f, 0.50f));
    KEY_POSITIONS.put('f', new PointF(0.40f, 0.50f));
    KEY_POSITIONS.put('g', new PointF(0.50f, 0.50f));
    KEY_POSITIONS.put('h', new PointF(0.60f, 0.50f));
    KEY_POSITIONS.put('j', new PointF(0.70f, 0.50f));
    KEY_POSITIONS.put('k', new PointF(0.80f, 0.50f));
    KEY_POSITIONS.put('l', new PointF(0.90f, 0.50f));
    
    // Bottom row
    KEY_POSITIONS.put('z', new PointF(0.15f, 0.75f));
    KEY_POSITIONS.put('x', new PointF(0.25f, 0.75f));
    KEY_POSITIONS.put('c', new PointF(0.35f, 0.75f));
    KEY_POSITIONS.put('v', new PointF(0.45f, 0.75f));
    KEY_POSITIONS.put('b', new PointF(0.55f, 0.75f));
    KEY_POSITIONS.put('n', new PointF(0.65f, 0.75f));
    KEY_POSITIONS.put('m', new PointF(0.75f, 0.75f));
  }
  
  public DTWPredictor(WordPredictor fallback)
  {
    _wordPaths = new HashMap<>();
    _wordFrequencies = new HashMap<>();
    _fallbackPredictor = fallback;
    _pruner = null;
    _gaussianModel = new GaussianKeyModel();
    _ngramModel = new NgramModel();
    _weightConfig = null; // Will be set when context available
  }
  
  /**
   * Set weight configuration
   */
  public void setWeightConfig(SwipeWeightConfig config)
  {
    _weightConfig = config;
  }
  
  /**
   * Load dictionary and precompute word paths
   */
  public void loadDictionary(Map<String, Integer> dictionary)
  {
    for (Map.Entry<String, Integer> entry : dictionary.entrySet())
    {
      String word = entry.getKey().toLowerCase();
      int frequency = entry.getValue();
      
      List<PointF> path = wordToPath(word);
      if (path != null && path.size() > 0)
      {
        _wordPaths.put(word, path);
        _wordFrequencies.put(word, frequency);
      }
    }
    
    // Initialize pruner with dictionary
    _pruner = new SwipePruner(_wordFrequencies);
    android.util.Log.d("DTWPredictor", "Initialized pruner with " + _wordFrequencies.size() + " words");
  }
  
  /**
   * Convert a word to a path of key positions
   */
  private List<PointF> wordToPath(String word)
  {
    List<PointF> path = new ArrayList<>();
    for (char c : word.toCharArray())
    {
      PointF pos = KEY_POSITIONS.get(c);
      if (pos != null)
      {
        path.add(new PointF(pos.x, pos.y));
      }
    }
    return path;
  }
  
  /**
   * Convert key sequence to normalized path
   */
  private List<PointF> keySequenceToPath(String keySequence)
  {
    List<PointF> path = new ArrayList<>();
    for (char c : keySequence.toLowerCase().toCharArray())
    {
      PointF pos = KEY_POSITIONS.get(c);
      if (pos != null)
      {
        path.add(new PointF(pos.x, pos.y));
      }
    }
    return path;
  }
  
  /**
   * Set keyboard dimensions for coordinate normalization
   */
  public void setKeyboardDimensions(float width, float height)
  {
    _keyboardWidth = width;
    _keyboardHeight = height;
  }
  
  /**
   * Predict words using actual swipe coordinates
   */
  public DTWResult predictWithCoordinates(List<PointF> rawCoordinates, List<KeyboardData.Key> touchedKeys)
  {
    if (rawCoordinates == null || rawCoordinates.size() < 2)
    {
      return new DTWResult(new ArrayList<>(), new ArrayList<>(), 0.5f);
    }
    
    PerformanceProfiler.start("DTW.predictWithCoordinates");
    android.util.Log.d("DTWPredictor", "Predicting with " + rawCoordinates.size() + " coordinates");
    
    // Prune candidates by extremities first (like FlorisBoard)
    List<String> prunedCandidates = null;
    if (_pruner != null)
    {
      prunedCandidates = _pruner.pruneByExtremities(rawCoordinates, touchedKeys);
      android.util.Log.d("DTWPredictor", "Pruned to " + prunedCandidates.size() + " candidates by extremities");
      
      // Further prune by length
      float avgKeyWidth = _keyboardWidth / 10; // Approximate for QWERTY
      prunedCandidates = _pruner.pruneByLength(rawCoordinates, prunedCandidates, avgKeyWidth, 8.0f);
      android.util.Log.d("DTWPredictor", "Pruned to " + prunedCandidates.size() + " candidates by length");
    }
    
    // Normalize coordinates to 0-1 range
    List<PointF> normalizedPath = normalizeCoordinates(rawCoordinates);
    
    // Resample path to consistent number of points (CRITICAL for accuracy)
    // Using 200 points like FlorisBoard - NOT 10 which loses 95% of information!
    normalizedPath = resamplePath(normalizedPath, SAMPLING_POINTS);
    
    List<WordScore> candidates = new ArrayList<>();
    
    // Calculate DTW distance for pruned candidates only
    for (Map.Entry<String, List<PointF>> entry : _wordPaths.entrySet())
    {
      String word = entry.getKey();
      
      // Skip if not in pruned candidates
      if (prunedCandidates != null && !prunedCandidates.contains(word))
        continue;
      
      List<PointF> wordPath = entry.getValue();
      
      // Skip words that are too different in length
      if (Math.abs(word.length() - touchedKeys.size()) > touchedKeys.size() / 2 + 2)
        continue;
      
      // Calculate DTW distance
      PerformanceProfiler.start("DTW.calculateDTW");
      float distance = calculateDTW(normalizedPath, wordPath);
      PerformanceProfiler.end("DTW.calculateDTW");
      
      // Calculate Gaussian probability score
      PerformanceProfiler.start("Gaussian.getWordConfidence");
      float gaussianScore = _gaussianModel.getWordConfidence(word, normalizedPath);
      PerformanceProfiler.end("Gaussian.getWordConfidence");
      
      // Calculate N-gram language model score
      PerformanceProfiler.start("Ngram.scoreWord");
      float ngramScore = _ngramModel.scoreWord(word);
      PerformanceProfiler.end("Ngram.scoreWord");
      
      // Convert distance to score (lower distance = higher score)
      int frequency = _wordFrequencies.getOrDefault(word, 1000);
      
      // Combined scoring with configurable weights
      float dtwScore = 1.0f / (1.0f + distance);
      float freqScore = Math.min(1.0f, frequency / 10000.0f);
      float ngramNorm = ngramScore / 100.0f;
      
      float score;
      if (_weightConfig != null)
      {
        score = _weightConfig.computeWeightedScore(dtwScore, gaussianScore, ngramNorm, freqScore) * 1000;
      }
      else
      {
        // Fallback to default weights
        score = (dtwScore * 0.4f + 
                gaussianScore * 0.3f + 
                ngramNorm * 0.2f + 
                freqScore * 0.1f) * 1000;
      }
      
      candidates.add(new WordScore(word, score, distance));
    }
    
    // Sort by score
    Collections.sort(candidates, new Comparator<WordScore>() {
      @Override
      public int compare(WordScore a, WordScore b) {
        return Float.compare(b.score, a.score);
      }
    });
    
    // Extract top predictions
    List<String> words = new ArrayList<>();
    List<Float> scores = new ArrayList<>();
    
    for (int i = 0; i < Math.min(10, candidates.size()); i++)
    {
      WordScore candidate = candidates.get(i);
      words.add(candidate.word);
      scores.add(candidate.score);
    }
    
    // Calculate confidence based on top score distribution
    float confidence = calculateConfidence(candidates);
    
    android.util.Log.d("DTWPredictor", "DTW predictions: " + words);
    
    PerformanceProfiler.end("DTW.predictWithCoordinates");
    PerformanceProfiler.report(); // Print performance report
    return new DTWResult(words, scores, confidence);
  }
  
  /**
   * Normalize coordinates to 0-1 range
   */
  private List<PointF> normalizeCoordinates(List<PointF> coordinates)
  {
    List<PointF> normalized = new ArrayList<>();
    
    for (PointF point : coordinates)
    {
      float x = point.x / _keyboardWidth;
      float y = point.y / _keyboardHeight;
      // Clamp to 0-1 range
      x = Math.max(0, Math.min(1, x));
      y = Math.max(0, Math.min(1, y));
      normalized.add(new PointF(x, y));
    }
    
    return normalized;
  }
  
  /**
   * Calculate confidence based on score distribution
   */
  private float calculateConfidence(List<WordScore> candidates)
  {
    if (candidates.isEmpty())
      return 0;
    
    if (candidates.size() == 1)
      return 0.5f;
    
    // Confidence is higher when top score is significantly better than others
    float topScore = candidates.get(0).score;
    float secondScore = candidates.get(1).score;
    
    if (topScore <= 0)
      return 0;
    
    float ratio = secondScore / topScore;
    // ratio close to 1 = low confidence, ratio close to 0 = high confidence
    return 1.0f - ratio;
  }
  
  /**
   * Predict words using DTW algorithm (legacy method for compatibility)
   */
  public List<String> predictWords(String keySequence)
  {
    if (keySequence == null || keySequence.length() < 2)
      return new ArrayList<>();
    
    android.util.Log.d("DTWPredictor", "Predicting for sequence: " + keySequence);
    
    List<PointF> inputPath = keySequenceToPath(keySequence);
    if (inputPath.size() < 2)
    {
      // Fall back to regular predictor for short sequences
      return _fallbackPredictor.predictWords(keySequence);
    }
    
    // Resample path to consistent number of points for DTW comparison
    inputPath = resamplePath(inputPath, Math.min(50, inputPath.size()));
    
    List<WordScore> candidates = new ArrayList<>();
    
    // Calculate DTW distance for each word
    for (Map.Entry<String, List<PointF>> entry : _wordPaths.entrySet())
    {
      String word = entry.getKey();
      List<PointF> wordPath = entry.getValue();
      
      // Skip words that are too different in length
      if (Math.abs(word.length() - keySequence.length()) > keySequence.length() / 2)
        continue;
      
      // Calculate DTW distance
      float distance = calculateDTW(inputPath, wordPath);
      
      // Convert distance to score (lower distance = higher score)
      int frequency = _wordFrequencies.get(word);
      float score = (1.0f / (1.0f + distance)) * frequency;
      
      candidates.add(new WordScore(word, score, distance));
    }
    
    // Sort by score
    Collections.sort(candidates, new Comparator<WordScore>() {
      @Override
      public int compare(WordScore a, WordScore b) {
        return Float.compare(b.score, a.score);
      }
    });
    
    // Return top 5 predictions
    List<String> predictions = new ArrayList<>();
    for (int i = 0; i < Math.min(5, candidates.size()); i++)
    {
      predictions.add(candidates.get(i).word);
    }
    
    android.util.Log.d("DTWPredictor", "Predictions: " + predictions);
    return predictions;
  }
  
  /**
   * Resample a path to a specific number of points using linear interpolation.
   * This is CRITICAL for DTW accuracy - maintains gesture shape information.
   * Based on FlorisBoard's approach with 200 sample points.
   */
  private List<PointF> resamplePath(List<PointF> path, int targetPoints)
  {
    if (path.size() < 2)
      return path;
      
    if (path.size() == targetPoints)
      return path;
    
    // Calculate total path length
    float totalLength = 0;
    for (int i = 1; i < path.size(); i++)
    {
      PointF p1 = path.get(i - 1);
      PointF p2 = path.get(i);
      totalLength += distance(p1, p2);
    }
    
    if (totalLength == 0)
      return path;
    
    // Resample at equal intervals along the path
    List<PointF> resampled = new ArrayList<>();
    float intervalLength = totalLength / (targetPoints - 1);
    float accumulatedLength = 0;
    float currentSegmentStart = 0;
    int currentSegment = 0;
    
    resampled.add(new PointF(path.get(0).x, path.get(0).y));
    
    for (int i = 1; i < targetPoints - 1; i++)
    {
      float targetLength = i * intervalLength;
      
      // Find the segment containing this point
      while (currentSegment < path.size() - 1 && accumulatedLength + 
             distance(path.get(currentSegment), path.get(currentSegment + 1)) < targetLength)
      {
        accumulatedLength += distance(path.get(currentSegment), path.get(currentSegment + 1));
        currentSegment++;
      }
      
      if (currentSegment >= path.size() - 1)
        break;
      
      // Interpolate within the segment
      PointF p1 = path.get(currentSegment);
      PointF p2 = path.get(currentSegment + 1);
      float segmentLength = distance(p1, p2);
      float remainingLength = targetLength - accumulatedLength;
      
      if (segmentLength > 0)
      {
        float t = remainingLength / segmentLength;
        float x = p1.x + t * (p2.x - p1.x);
        float y = p1.y + t * (p2.y - p1.y);
        resampled.add(new PointF(x, y));
      }
    }
    
    // Add last point
    resampled.add(new PointF(path.get(path.size() - 1).x, path.get(path.size() - 1).y));
    
    // Ensure we have exactly targetPoints
    while (resampled.size() < targetPoints)
    {
      resampled.add(new PointF(path.get(path.size() - 1).x, path.get(path.size() - 1).y));
    }
    
    return resampled.subList(0, Math.min(targetPoints, resampled.size()));
  }
  
  /**
   * Calculate Dynamic Time Warping distance between two paths
   * Uses Sakoe-Chiba band optimization to reduce computation
   */
  private float calculateDTW(List<PointF> path1, List<PointF> path2)
  {
    int n = path1.size();
    int m = path2.size();
    
    // Use optimized version for large paths
    if (n > 50 || m > 50)
    {
      return calculateDTWWithBand(path1, path2);
    }
    
    // Original implementation for small paths
    return calculateDTWFull(path1, path2);
  }
  
  /**
   * Full DTW calculation without optimization
   * Used for small paths where optimization overhead isn't worth it
   */
  private float calculateDTWFull(List<PointF> path1, List<PointF> path2)
  {
    int n = path1.size();
    int m = path2.size();
    
    // Create DTW matrix
    float[][] dtw = new float[n + 1][m + 1];
    
    // Initialize
    for (int i = 1; i <= n; i++)
      dtw[i][0] = Float.MAX_VALUE;
    for (int j = 1; j <= m; j++)
      dtw[0][j] = Float.MAX_VALUE;
    dtw[0][0] = 0;
    
    // Fill DTW matrix
    for (int i = 1; i <= n; i++)
    {
      for (int j = 1; j <= m; j++)
      {
        float cost = distance(path1.get(i - 1), path2.get(j - 1));
        dtw[i][j] = cost + Math.min(
          dtw[i - 1][j],      // insertion
          Math.min(
            dtw[i][j - 1],    // deletion
            dtw[i - 1][j - 1] // match
          )
        );
      }
    }
    
    // Normalize by path length
    return dtw[n][m] / Math.max(n, m);
  }
  
  /**
   * Optimized DTW with Sakoe-Chiba band
   * Only calculates values within a band around the diagonal
   * Reduces complexity from O(n*m) to O(n*w) where w is band width
   */
  private float calculateDTWWithBand(List<PointF> path1, List<PointF> path2)
  {
    int n = path1.size();
    int m = path2.size();
    
    // Calculate band width (typically 10-20% of sequence length)
    int bandWidth = Math.max(Math.abs(n - m), Math.min(n, m) / 5);
    bandWidth = Math.max(bandWidth, 5); // Minimum band width
    
    // Create compact DTW matrix (only stores the band)
    // Using a sliding window approach to save memory
    float[][] dtw = new float[n + 1][2 * bandWidth + 1];
    
    // Initialize with MAX_VALUE
    for (int i = 0; i <= n; i++)
    {
      for (int j = 0; j < 2 * bandWidth + 1; j++)
      {
        dtw[i][j] = Float.MAX_VALUE;
      }
    }
    
    // Set starting point
    dtw[0][bandWidth] = 0;
    
    // Fill DTW matrix within the band
    for (int i = 1; i <= n; i++)
    {
      // Calculate j range within the band
      int jStart = Math.max(1, i - bandWidth);
      int jEnd = Math.min(m, i + bandWidth);
      
      for (int j = jStart; j <= jEnd; j++)
      {
        // Map j to band index
        int bandIndex = j - i + bandWidth;
        if (bandIndex < 0 || bandIndex >= 2 * bandWidth + 1)
          continue;
        
        float cost = distance(path1.get(i - 1), path2.get(j - 1));
        
        // Calculate minimum from three neighbors (if within band)
        float minCost = Float.MAX_VALUE;
        
        // From (i-1, j) - insertion
        if (i > 0 && j - (i - 1) + bandWidth >= 0 && j - (i - 1) + bandWidth < 2 * bandWidth + 1)
        {
          minCost = Math.min(minCost, dtw[i - 1][j - (i - 1) + bandWidth]);
        }
        
        // From (i, j-1) - deletion
        if (j > 1 && bandIndex - 1 >= 0)
        {
          minCost = Math.min(minCost, dtw[i][bandIndex - 1]);
        }
        
        // From (i-1, j-1) - match
        if (i > 0 && j > 0 && (j - 1) - (i - 1) + bandWidth >= 0 && 
            (j - 1) - (i - 1) + bandWidth < 2 * bandWidth + 1)
        {
          minCost = Math.min(minCost, dtw[i - 1][(j - 1) - (i - 1) + bandWidth]);
        }
        
        dtw[i][bandIndex] = cost + minCost;
      }
    }
    
    // Get final result
    int finalBandIndex = m - n + bandWidth;
    if (finalBandIndex < 0 || finalBandIndex >= 2 * bandWidth + 1)
    {
      // If final position is outside band, fall back to full DTW
      return calculateDTWFull(path1, path2);
    }
    
    // Normalize by path length
    return dtw[n][finalBandIndex] / Math.max(n, m);
  }
  
  /**
   * Calculate Euclidean distance between two points
   */
  private float distance(PointF p1, PointF p2)
  {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return (float)Math.sqrt(dx * dx + dy * dy);
  }
  
  /**
   * Helper class for word scoring
   */
  private static class WordScore
  {
    final String word;
    final float score;
    final float dtwDistance;
    
    WordScore(String word, float score, float dtwDistance)
    {
      this.word = word;
      this.score = score;
      this.dtwDistance = dtwDistance;
    }
  }
  
  /**
   * Result class for DTW predictions
   */
  public static class DTWResult
  {
    public final List<String> words;
    public final List<Float> scores;
    public final float confidence;
    
    public DTWResult(List<String> words, List<Float> scores, float confidence)
    {
      this.words = words;
      this.scores = scores;
      this.confidence = confidence;
    }
  }
}