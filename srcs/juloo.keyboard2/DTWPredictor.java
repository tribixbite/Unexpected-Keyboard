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
   * Predict words using DTW algorithm
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
    
    // Simplify the input path to reduce noise
    inputPath = simplifyPath(inputPath, 5);
    
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
      
      candidates.add(new WordScore(word, score));
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
   * Simplify a path by sampling points
   */
  private List<PointF> simplifyPath(List<PointF> path, int targetPoints)
  {
    if (path.size() <= targetPoints)
      return path;
    
    List<PointF> simplified = new ArrayList<>();
    float step = (float)(path.size() - 1) / (targetPoints - 1);
    
    for (int i = 0; i < targetPoints; i++)
    {
      int index = Math.round(i * step);
      simplified.add(path.get(Math.min(index, path.size() - 1)));
    }
    
    return simplified;
  }
  
  /**
   * Calculate Dynamic Time Warping distance between two paths
   */
  private float calculateDTW(List<PointF> path1, List<PointF> path2)
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
    
    WordScore(String word, float score)
    {
      this.word = word;
      this.score = score;
    }
  }
}