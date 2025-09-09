package juloo.keyboard2;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * High-performance vocabulary system for neural swipe predictions
 * Matches web demo's multi-level caching approach for optimal performance
 */
public class NeuralVocabulary 
{
  private static final String TAG = "NeuralVocabulary";
  
  // Multi-level caching for O(1) lookups like web demo
  private Map<String, Float> _wordFreq;
  private Set<String> _commonWords;
  private Map<Integer, Set<String>> _wordsByLength;
  private Set<String> _top5000;
  private boolean _isLoaded = false;
  
  // Performance caches
  private Map<String, Boolean> _validWordCache;
  private Map<Integer, Float> _minFreqByLength;
  
  public NeuralVocabulary()
  {
    _wordFreq = new HashMap<>();
    _commonWords = new HashSet<>();
    _wordsByLength = new HashMap<>();
    _top5000 = new HashSet<>();
    _validWordCache = new HashMap<>();
    _minFreqByLength = new HashMap<>();
  }
  
  /**
   * Load vocabulary with multi-level caching like web demo
   */
  public boolean loadVocabulary()
  {
    Log.d(TAG, "Loading vocabulary with multi-level caching...");
    
    // Load fallback vocabulary for now (matching web demo pattern)
    loadFallbackVocabulary();
    
    // Build performance indexes
    buildPerformanceIndexes();
    
    _isLoaded = true;
    Log.d(TAG, String.format("Vocabulary loaded: %d words, %d by length, %d common", 
      _wordFreq.size(), _wordsByLength.size(), _commonWords.size()));
    
    return true;
  }
  
  /**
   * Ultra-fast word validation with caching
   */
  public boolean isValidWord(String word)
  {
    if (!_isLoaded) return false;
    
    // Check cache first (O(1))
    Boolean cached = _validWordCache.get(word);
    if (cached != null) return cached;
    
    // Fast path - check common words set (O(1))
    if (_commonWords.contains(word))
    {
      _validWordCache.put(word, true);
      return true;
    }
    
    // Check by length set (O(1))
    Set<String> wordsOfLength = _wordsByLength.get(word.length());
    boolean valid = wordsOfLength != null && wordsOfLength.contains(word);
    
    // Cache result
    _validWordCache.put(word, valid);
    return valid;
  }
  
  /**
   * Get word frequency (cached)
   */
  public float getWordFrequency(String word)
  {
    Float freq = _wordFreq.get(word);
    return freq != null ? freq : 0.0f;
  }
  
  /**
   * Filter predictions like web demo
   */
  public List<String> filterPredictions(List<String> predictions)
  {
    if (!_isLoaded) return predictions;
    
    List<String> filtered = new ArrayList<>();
    
    for (String word : predictions)
    {
      if (isValidWord(word))
      {
        filtered.add(word);
      }
    }
    
    return filtered;
  }
  
  private void loadFallbackVocabulary()
  {
    // Common English words with frequencies (matching web demo fallback)
    String[] fallbackWords = {
      "the", "of", "and", "to", "a", "in", "for", "is", "on", "that",
      "by", "this", "with", "i", "you", "it", "not", "or", "be", "are",
      "from", "at", "as", "your", "all", "would", "will", "there", "their",
      "what", "so", "if", "about", "which", "when", "one", "can", "had",
      "hello", "world", "test", "how", "now", "but", "our", "out", "old",
      "boy", "get", "has", "him", "his", "new", "see", "two", "who", "did"
    };
    
    for (int i = 0; i < fallbackWords.length; i++)
    {
      String word = fallbackWords[i];
      float freq = (float)(1e-4 * Math.pow(0.9, i)); // Decreasing frequency
      _wordFreq.put(word, freq);
    }
    
    // Common words are first 20
    for (int i = 0; i < Math.min(20, fallbackWords.length); i++)
    {
      _commonWords.add(fallbackWords[i]);
    }
    
    // Top 5000 is all we have
    for (String word : fallbackWords)
    {
      _top5000.add(word);
    }
  }
  
  private void buildPerformanceIndexes()
  {
    // Build words by length index for O(1) length-based filtering
    for (String word : _wordFreq.keySet())
    {
      int length = word.length();
      Set<String> wordsOfLength = _wordsByLength.get(length);
      if (wordsOfLength == null)
      {
        wordsOfLength = new HashSet<>();
        _wordsByLength.put(length, wordsOfLength);
      }
      wordsOfLength.add(word);
      
      // Track minimum frequency by length
      Float currentMinFreq = _minFreqByLength.get(length);
      float wordFreq = _wordFreq.get(word);
      if (currentMinFreq == null || wordFreq < currentMinFreq)
      {
        _minFreqByLength.put(length, wordFreq);
      }
    }
  }
  
  public boolean isLoaded()
  {
    return _isLoaded;
  }
  
  public int getVocabularySize()
  {
    return _wordFreq.size();
  }
}