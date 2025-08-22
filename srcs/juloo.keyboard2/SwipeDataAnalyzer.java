package juloo.keyboard2;

import android.content.Context;
import android.util.Log;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import android.graphics.PointF;

/**
 * Analyzes existing swipe data against current prediction algorithms.
 * Provides transparency into algorithm performance and allows testing
 * of different configurations.
 */
public class SwipeDataAnalyzer
{
  private static final String TAG = "SwipeDataAnalyzer";
  
  private DTWPredictor _dtwPredictor;
  private GaussianKeyModel _gaussianModel;
  private NgramModel _ngramModel;
  private Map<String, Integer> _dictionary;
  
  // Algorithm weight configuration (user adjustable)
  private float _dtwWeight = 0.4f;
  private float _gaussianWeight = 0.3f;
  private float _ngramWeight = 0.2f;
  private float _frequencyWeight = 0.1f;
  
  // Analysis results
  public static class AnalysisResult
  {
    public String targetWord;
    public int rank;
    public float score;
    public float dtwDistance;
    public float gaussianScore;
    public float ngramScore;
    public List<String> top5Predictions;
    public long processingTimeMs;
    
    public boolean isCorrect() { return rank == 1; }
    public boolean isTop3() { return rank >= 1 && rank <= 3; }
    public boolean isTop5() { return rank >= 1 && rank <= 5; }
  }
  
  public SwipeDataAnalyzer(Context context)
  {
    _dtwPredictor = new DTWPredictor(null);
    _gaussianModel = new GaussianKeyModel();
    _ngramModel = new NgramModel();
    _dictionary = new HashMap<>();
    
    // Load dictionary
    loadDictionary(context);
  }
  
  /**
   * Set algorithm weights for scoring
   */
  public void setWeights(float dtw, float gaussian, float ngram, float frequency)
  {
    _dtwWeight = dtw;
    _gaussianWeight = gaussian;
    _ngramWeight = ngram;
    _frequencyWeight = frequency;
    
    // Normalize to sum to 1.0
    float sum = _dtwWeight + _gaussianWeight + _ngramWeight + _frequencyWeight;
    if (sum > 0)
    {
      _dtwWeight /= sum;
      _gaussianWeight /= sum;
      _ngramWeight /= sum;
      _frequencyWeight /= sum;
    }
  }
  
  /**
   * Load dictionary from assets
   */
  private void loadDictionary(Context context)
  {
    try
    {
      BufferedReader reader = new BufferedReader(
        new java.io.InputStreamReader(context.getAssets().open("dictionaries/en_US_enhanced.txt")));
      
      String line;
      while ((line = reader.readLine()) != null)
      {
        String[] parts = line.trim().split("\t");
        if (parts.length >= 1)
        {
          String word = parts[0].toLowerCase();
          int frequency = parts.length > 1 ? Integer.parseInt(parts[1]) : 1000;
          _dictionary.put(word, frequency);
        }
      }
      reader.close();
      
      _dtwPredictor.loadDictionary(_dictionary);
      Log.d(TAG, "Loaded " + _dictionary.size() + " words");
    }
    catch (IOException e)
    {
      Log.e(TAG, "Failed to load dictionary: " + e.getMessage());
    }
  }
  
  /**
   * Analyze a single swipe trace from JSON data
   */
  public AnalysisResult analyzeTrace(JSONObject swipeData) throws JSONException
  {
    long startTime = System.currentTimeMillis();
    AnalysisResult result = new AnalysisResult();
    
    // Extract target word
    result.targetWord = swipeData.getString("target_word").toLowerCase();
    
    // Extract trace points
    JSONArray traceArray = swipeData.getJSONArray("trace");
    List<PointF> trace = new ArrayList<>();
    
    for (int i = 0; i < traceArray.length(); i++)
    {
      JSONObject point = traceArray.getJSONObject(i);
      float x = (float)point.getDouble("x");
      float y = (float)point.getDouble("y");
      trace.add(new PointF(x, y));
    }
    
    // Extract registered keys
    JSONArray keysArray = swipeData.getJSONArray("registered_keys");
    List<KeyboardData.Key> touchedKeys = new ArrayList<>();
    
    for (int i = 0; i < keysArray.length(); i++)
    {
      String keyChar = keysArray.getString(i).toLowerCase();
      // Create simple key object
      KeyValue kv = KeyValue.makeStringKey(keyChar);
      KeyboardData.Key key = new KeyboardData.Key(
        new KeyValue[]{kv, null, null, null, null},
        null, 0, 1.0f, 0.0f, null
      );
      touchedKeys.add(key);
    }
    
    // Run prediction with custom weights
    PredictionResult prediction = predictWithCustomWeights(trace, touchedKeys);
    
    // Find rank of target word
    result.rank = -1;
    for (int i = 0; i < prediction.words.size(); i++)
    {
      if (prediction.words.get(i).equals(result.targetWord))
      {
        result.rank = i + 1;
        result.score = prediction.scores.get(i);
        break;
      }
    }
    
    // Store component scores
    result.dtwDistance = prediction.dtwDistance;
    result.gaussianScore = prediction.gaussianScore;
    result.ngramScore = prediction.ngramScore;
    
    // Store top 5 predictions
    result.top5Predictions = new ArrayList<>();
    for (int i = 0; i < Math.min(5, prediction.words.size()); i++)
    {
      result.top5Predictions.add(prediction.words.get(i));
    }
    
    result.processingTimeMs = System.currentTimeMillis() - startTime;
    
    return result;
  }
  
  /**
   * Custom prediction with adjustable weights
   */
  private PredictionResult predictWithCustomWeights(List<PointF> trace, List<KeyboardData.Key> touchedKeys)
  {
    PredictionResult result = new PredictionResult();
    
    // Normalize trace to [0,1]
    List<PointF> normalizedTrace = normalizeTrace(trace);
    
    // Resample to consistent points
    normalizedTrace = resamplePath(normalizedTrace, 200);
    
    // Calculate scores for each word
    List<WordScore> candidates = new ArrayList<>();
    
    for (Map.Entry<String, Integer> entry : _dictionary.entrySet())
    {
      String word = entry.getKey();
      int frequency = entry.getValue();
      
      // Skip words too different in length
      if (Math.abs(word.length() - touchedKeys.size()) > touchedKeys.size() / 2 + 2)
        continue;
      
      // Calculate component scores
      float dtwDistance = calculateDTW(normalizedTrace, wordToPath(word));
      float gaussianScore = _gaussianModel.getWordConfidence(word, normalizedTrace);
      float ngramScore = _ngramModel.scoreWord(word) / 100.0f; // Normalize
      float freqScore = Math.min(1.0f, frequency / 10000.0f);
      
      // Combined score with custom weights
      float dtwScore = 1.0f / (1.0f + dtwDistance);
      float combinedScore = (dtwScore * _dtwWeight + 
                            gaussianScore * _gaussianWeight + 
                            ngramScore * _ngramWeight + 
                            freqScore * _frequencyWeight) * 1000;
      
      WordScore ws = new WordScore(word, combinedScore);
      ws.dtwDistance = dtwDistance;
      ws.gaussianScore = gaussianScore;
      ws.ngramScore = ngramScore;
      candidates.add(ws);
    }
    
    // Sort by score
    candidates.sort((a, b) -> Float.compare(b.score, a.score));
    
    // Extract results
    result.words = new ArrayList<>();
    result.scores = new ArrayList<>();
    
    for (int i = 0; i < Math.min(10, candidates.size()); i++)
    {
      WordScore ws = candidates.get(i);
      result.words.add(ws.word);
      result.scores.add(ws.score);
      
      // Store component scores for the top result
      if (i == 0)
      {
        result.dtwDistance = ws.dtwDistance;
        result.gaussianScore = ws.gaussianScore;
        result.ngramScore = ws.ngramScore;
      }
    }
    
    return result;
  }
  
  /**
   * Analyze a JSON file containing multiple swipe traces
   */
  public Map<String, Object> analyzeFile(File jsonFile) throws IOException, JSONException
  {
    Map<String, Object> results = new HashMap<>();
    List<AnalysisResult> traces = new ArrayList<>();
    
    // Read JSON file
    BufferedReader reader = new BufferedReader(new FileReader(jsonFile));
    StringBuilder json = new StringBuilder();
    String line;
    while ((line = reader.readLine()) != null)
    {
      json.append(line);
    }
    reader.close();
    
    // Parse JSON
    JSONObject root = new JSONObject(json.toString());
    JSONArray dataArray = root.getJSONArray("data");
    
    // Analyze each trace
    int correct = 0;
    int top3 = 0;
    int top5 = 0;
    
    for (int i = 0; i < dataArray.length(); i++)
    {
      JSONObject swipeData = dataArray.getJSONObject(i);
      AnalysisResult result = analyzeTrace(swipeData);
      traces.add(result);
      
      if (result.isCorrect()) correct++;
      if (result.isTop3()) top3++;
      if (result.isTop5()) top5++;
    }
    
    // Calculate statistics
    results.put("traces", traces);
    results.put("total_traces", traces.size());
    results.put("accuracy_top1", (float)correct / traces.size());
    results.put("accuracy_top3", (float)top3 / traces.size());
    results.put("accuracy_top5", (float)top5 / traces.size());
    results.put("weights", String.format("DTW:%.2f Gaussian:%.2f Ngram:%.2f Freq:%.2f",
                                         _dtwWeight, _gaussianWeight, _ngramWeight, _frequencyWeight));
    
    return results;
  }
  
  // Helper classes and methods
  
  private static class PredictionResult
  {
    List<String> words = new ArrayList<>();
    List<Float> scores = new ArrayList<>();
    float dtwDistance;
    float gaussianScore;
    float ngramScore;
  }
  
  private static class WordScore
  {
    String word;
    float score;
    float dtwDistance;
    float gaussianScore;
    float ngramScore;
    
    WordScore(String w, float s)
    {
      word = w;
      score = s;
    }
  }
  
  private List<PointF> normalizeTrace(List<PointF> trace)
  {
    // Find bounds
    float minX = Float.MAX_VALUE, maxX = Float.MIN_VALUE;
    float minY = Float.MAX_VALUE, maxY = Float.MIN_VALUE;
    
    for (PointF p : trace)
    {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    
    // Normalize to [0,1]
    List<PointF> normalized = new ArrayList<>();
    float width = maxX - minX;
    float height = maxY - minY;
    
    if (width > 0 && height > 0)
    {
      for (PointF p : trace)
      {
        float x = (p.x - minX) / width;
        float y = (p.y - minY) / height;
        normalized.add(new PointF(x, y));
      }
    }
    else
    {
      normalized.addAll(trace);
    }
    
    return normalized;
  }
  
  private List<PointF> wordToPath(String word)
  {
    // Convert word to key positions
    // This is simplified - should use actual key layout
    List<PointF> path = new ArrayList<>();
    Map<Character, PointF> keyPositions = getQwertyLayout();
    
    for (char c : word.toCharArray())
    {
      PointF pos = keyPositions.get(c);
      if (pos != null)
      {
        path.add(new PointF(pos.x, pos.y));
      }
    }
    
    return path;
  }
  
  private Map<Character, PointF> getQwertyLayout()
  {
    Map<Character, PointF> layout = new HashMap<>();
    
    // Top row
    layout.put('q', new PointF(0.05f, 0.25f));
    layout.put('w', new PointF(0.15f, 0.25f));
    layout.put('e', new PointF(0.25f, 0.25f));
    layout.put('r', new PointF(0.35f, 0.25f));
    layout.put('t', new PointF(0.45f, 0.25f));
    layout.put('y', new PointF(0.55f, 0.25f));
    layout.put('u', new PointF(0.65f, 0.25f));
    layout.put('i', new PointF(0.75f, 0.25f));
    layout.put('o', new PointF(0.85f, 0.25f));
    layout.put('p', new PointF(0.95f, 0.25f));
    
    // Middle row
    layout.put('a', new PointF(0.10f, 0.50f));
    layout.put('s', new PointF(0.20f, 0.50f));
    layout.put('d', new PointF(0.30f, 0.50f));
    layout.put('f', new PointF(0.40f, 0.50f));
    layout.put('g', new PointF(0.50f, 0.50f));
    layout.put('h', new PointF(0.60f, 0.50f));
    layout.put('j', new PointF(0.70f, 0.50f));
    layout.put('k', new PointF(0.80f, 0.50f));
    layout.put('l', new PointF(0.90f, 0.50f));
    
    // Bottom row
    layout.put('z', new PointF(0.15f, 0.75f));
    layout.put('x', new PointF(0.25f, 0.75f));
    layout.put('c', new PointF(0.35f, 0.75f));
    layout.put('v', new PointF(0.45f, 0.75f));
    layout.put('b', new PointF(0.55f, 0.75f));
    layout.put('n', new PointF(0.65f, 0.75f));
    layout.put('m', new PointF(0.75f, 0.75f));
    
    return layout;
  }
  
  private List<PointF> resamplePath(List<PointF> path, int targetPoints)
  {
    // Simplified resampling - should use proper linear interpolation
    if (path.size() >= targetPoints)
      return path;
    
    List<PointF> resampled = new ArrayList<>();
    float step = (float)(path.size() - 1) / (targetPoints - 1);
    
    for (int i = 0; i < targetPoints; i++)
    {
      int index = Math.min((int)(i * step), path.size() - 1);
      resampled.add(path.get(index));
    }
    
    return resampled;
  }
  
  private float calculateDTW(List<PointF> path1, List<PointF> path2)
  {
    // Simplified DTW - should use full implementation
    if (path1.isEmpty() || path2.isEmpty())
      return Float.MAX_VALUE;
    
    int n = path1.size();
    int m = path2.size();
    float[][] dtw = new float[n + 1][m + 1];
    
    for (int i = 1; i <= n; i++)
      dtw[i][0] = Float.MAX_VALUE;
    for (int j = 1; j <= m; j++)
      dtw[0][j] = Float.MAX_VALUE;
    dtw[0][0] = 0;
    
    for (int i = 1; i <= n; i++)
    {
      for (int j = 1; j <= m; j++)
      {
        PointF p1 = path1.get(i - 1);
        PointF p2 = path2.get(j - 1);
        float cost = (float)Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
        
        dtw[i][j] = cost + Math.min(dtw[i-1][j], Math.min(dtw[i][j-1], dtw[i-1][j-1]));
      }
    }
    
    return dtw[n][m] / Math.max(n, m);
  }
}