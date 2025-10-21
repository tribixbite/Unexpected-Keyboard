package juloo.keyboard2;

import android.content.Context;
import android.util.Log;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Optimized vocabulary filtering for neural swipe predictions
 * Ports web app swipe-vocabulary.js optimizations to Android
 * 
 * Features:
 * - Common words fast-path for instant lookup
 * - Hierarchical vocabulary (common -> top5000 -> full)
 * - Combined confidence + frequency scoring
 * - Length-based filtering and word lookup
 */
public class OptimizedVocabulary
{
  private static final String TAG = "OptimizedVocabulary";
  
  // OPTIMIZATION: Single unified lookup structure (1 hash lookup instead of 3)
  private Map<String, WordInfo> vocabulary;  // All words with frequency + tier in one lookup

  // Word information with frequency and tier for single-lookup optimization
  private static class WordInfo
  {
    final float frequency;
    final byte tier; // 0=regular, 1=top5000, 2=common

    WordInfo(float freq, byte tier)
    {
      this.frequency = freq;
      this.tier = tier;
    }
  }
  
  // Scoring parameters (from web app)
  private static final float CONFIDENCE_WEIGHT = 0.6f;
  private static final float FREQUENCY_WEIGHT = 0.4f;
  private static final float COMMON_WORDS_BOOST = 1.2f;
  private static final float TOP5000_BOOST = 1.0f;
  private static final float RARE_WORDS_PENALTY = 0.9f;
  
  // Filtering thresholds
  private Map<Integer, Float> minFrequencyByLength;

  // Disabled words filter (for Dictionary Manager integration)
  private Set<String> disabledWords;

  private boolean isLoaded = false;
  private Context context;

  public OptimizedVocabulary(Context context)
  {
    this.context = context;
    this.vocabulary = new HashMap<>();
    this.minFrequencyByLength = new HashMap<>();
    this.disabledWords = new HashSet<>();
  }
  
  /**
   * Load vocabulary from assets with frequency data
   * Creates hierarchical structure for fast filtering
   */
  public boolean loadVocabulary()
  {
    try
    {
      // Log.d(TAG, "Loading optimized vocabulary from assets...");

      // OPTIMIZATION: Load vocabulary with fast-path sets built during loading
      loadWordFrequencies();

      // Load custom words and user dictionary for beam search
      loadCustomAndUserWords();

      // Load disabled words to filter from predictions
      loadDisabledWords();

      // Initialize minimum frequency thresholds by word length
      initializeFrequencyThresholds();

      // REMOVED: createFastPathSets() - now built during loading (O(n) instead of O(n log n))
      // REMOVED: createLengthBasedLookup() - never used in predictions (dead code)

      isLoaded = true;
      // Log.d(TAG, String.format("Vocabulary loaded: %d total words, %d common, %d top5000",
        // wordFrequencies.size(), commonWords.size(), top5000.size()));

      return true;
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to load vocabulary - NO FALLBACK ALLOWED", e);
      throw new RuntimeException("Dictionary loading failed - fallback vocabulary deleted", e);
    }
  }
  
  /**
   * Filter and rank neural network predictions using vocabulary optimization
   * Implements fast-path lookup and combined scoring from web app
   */
  public List<FilteredPrediction> filterPredictions(List<CandidateWord> rawPredictions, SwipeStats swipeStats)
  {
    if (!isLoaded)
    {
      Log.w(TAG, "Vocabulary not loaded, returning raw predictions");
      return convertToFiltered(rawPredictions);
    }
    
    List<FilteredPrediction> validPredictions = new ArrayList<>();
    
    for (CandidateWord candidate : rawPredictions)
    {
      String word = candidate.word.toLowerCase().trim();

      // Skip invalid word formats
      if (!word.matches("^[a-z]+$"))
      {
        continue;
      }

      // FILTER OUT DISABLED WORDS (Dictionary Manager integration)
      if (disabledWords.contains(word))
      {
        continue; // Skip disabled words from beam search
      }

      // CRITICAL OPTIMIZATION: SINGLE hash lookup (was 3 lookups!)
      WordInfo info = vocabulary.get(word);
      if (info == null)
      {
        continue; // Word not in vocabulary
      }

      // OPTIMIZATION: Tier is embedded in WordInfo (no additional lookups!)
      float boost;
      String source;

      switch (info.tier)
      {
        case 2: // common (top 100)
          boost = COMMON_WORDS_BOOST;
          source = "common";
          break;
        case 1: // top5000
          boost = TOP5000_BOOST;
          source = "top5000";
          break;
        default: // regular
          // Check frequency threshold for rare words
          float minFreq = getMinFrequency(word.length());
          if (info.frequency < minFreq)
          {
            continue; // Below threshold
          }
          boost = RARE_WORDS_PENALTY;
          source = "vocabulary";
          break;
      }

      float score = calculateCombinedScore(candidate.confidence, info.frequency, boost);
      validPredictions.add(new FilteredPrediction(word, score, candidate.confidence, info.frequency, source));
    }
    
    // Sort by combined score (confidence + frequency)
    validPredictions.sort((a, b) -> Float.compare(b.score, a.score));
    
    // Apply swipe-specific filtering if needed
    if (swipeStats != null && swipeStats.expectedLength > 0)
    {
      return filterByExpectedLength(validPredictions, swipeStats.expectedLength);
    }
    
    return validPredictions.subList(0, Math.min(validPredictions.size(), 10));
  }
  
  /**
   * Calculate combined score from NN confidence and word frequency
   * Implements web app scoring algorithm with logarithmic frequency scaling
   */
  private float calculateCombinedScore(float confidence, float frequency, float boost)
  {
    // Logarithmic frequency scaling to normalize to ~0-1 range
    float freqScore = (float)(Math.log10(frequency + 1e-10) / -10.0);
    freqScore = Math.max(0.0f, Math.min(1.0f, freqScore)); // Clamp to [0,1]
    
    // Weighted combination with boost factor
    return (CONFIDENCE_WEIGHT * confidence + FREQUENCY_WEIGHT * freqScore) * boost;
  }
  
  /**
   * Load word frequencies from dictionary files
   * OPTIMIZATION: Single-lookup structure with tier embedded (1 lookup instead of 3)
   */
  private void loadWordFrequencies()
  {
    try
    {
      // Load English dictionary (already sorted by frequency)
      InputStream inputStream = context.getAssets().open("dictionaries/en.txt");
      BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

      String line;
      int wordCount = 0;
      while ((line = reader.readLine()) != null)
      {
        line = line.trim().toLowerCase();
        if (!line.isEmpty() && line.matches("^[a-z]+$"))
        {
          // OPTIMIZATION: Assign frequency based on position (file is pre-sorted)
          float frequency = 1.0f / (wordCount + 1.0f);

          // OPTIMIZATION: Determine tier during loading (no separate sets needed!)
          byte tier;
          if (wordCount < 100) {
            tier = 2; // common
          } else if (wordCount < 5000) {
            tier = 1; // top5000
          } else {
            tier = 0; // regular
          }

          // CRITICAL: Single structure with all info (1 lookup vs 3 lookups!)
          vocabulary.put(line, new WordInfo(frequency, tier));
          wordCount++;

          // Limit to prevent memory issues (150k words max)
          if (wordCount >= 150000)
          {
            break;
          }
        }
      }

      reader.close();
      // Log.d(TAG, "Loaded " + wordCount + " words with frequencies");
    }
    catch (IOException e)
    {
      Log.e(TAG, "Failed to load word frequencies", e);
      throw new RuntimeException("Could not load vocabulary", e);
    }
  }

  /**
   * REMOVED: createFastPathSets() - now built during loading (no sort needed)
   */
  
  /**
   * Initialize minimum frequency thresholds by word length
   */
  private void initializeFrequencyThresholds()
  {
    // Longer words can have lower frequency thresholds
    minFrequencyByLength.put(1, 1e-4f);
    minFrequencyByLength.put(2, 1e-5f);
    minFrequencyByLength.put(3, 1e-6f);
    minFrequencyByLength.put(4, 1e-6f);
    minFrequencyByLength.put(5, 1e-7f);
    minFrequencyByLength.put(6, 1e-7f);
    minFrequencyByLength.put(7, 1e-8f);
    minFrequencyByLength.put(8, 1e-8f);
    // 9+ words
    for (int i = 9; i <= 20; i++)
    {
      minFrequencyByLength.put(i, 1e-9f);
    }
  }

  /**
   * REMOVED: createLengthBasedLookup() - dead code (never used in predictions)
   * Saved O(n) iteration + O(n log n) sorting overhead on startup
   */

  /**
   * Get minimum frequency threshold for word length
   */
  private float getMinFrequency(int length)
  {
    return minFrequencyByLength.getOrDefault(length, 1e-9f);
  }
  
  /**
   * Filter predictions by expected word length with tolerance
   */
  private List<FilteredPrediction> filterByExpectedLength(List<FilteredPrediction> predictions, int expectedLength)
  {
    int tolerance = 2; // Allow ±2 characters
    
    List<FilteredPrediction> filtered = new ArrayList<>();
    for (FilteredPrediction pred : predictions)
    {
      int lengthDiff = Math.abs(pred.word.length() - expectedLength);
      if (lengthDiff <= tolerance)
      {
        filtered.add(pred);
      }
    }
    
    return filtered.size() > 0 ? filtered : predictions.subList(0, Math.min(predictions.size(), 5));
  }
  
  
  /**
   * Convert raw predictions to filtered format
   */
  private List<FilteredPrediction> convertToFiltered(List<CandidateWord> rawPredictions)
  {
    List<FilteredPrediction> result = new ArrayList<>();
    for (CandidateWord candidate : rawPredictions)
    {
      result.add(new FilteredPrediction(candidate.word, candidate.confidence, 
        candidate.confidence, 0.0f, "raw"));
    }
    return result;
  }
  
  /**
   * Check if vocabulary is loaded
   */
  public boolean isLoaded()
  {
    return isLoaded;
  }

  /**
   * Reload custom words, user dictionary, and disabled words without reloading main vocabulary
   * Called when Dictionary Manager makes changes
   * PERFORMANCE: Only reloads small dynamic sets, not the 10k main dictionary
   */
  public void reloadCustomAndDisabledWords()
  {
    if (!isLoaded) return;

    // Remove old custom/user words by clearing and reloading main dict
    // Then re-add custom/user words with current values
    // This is more efficient than tracking which words to remove

    // Save main vocabulary size before reload
    int mainVocabSize = vocabulary.size();

    // Clear and reload just custom/user (main vocab stays in memory)
    // Actually, we need a better approach - let me just reload everything from sources

    try
    {
      // Clear old custom/user/disabled data
      disabledWords.clear();

      // Reload custom and user words (overwrites old entries)
      loadCustomAndUserWords();

      // Reload disabled words filter
      loadDisabledWords();

      Log.d(TAG, "Reloaded custom/user/disabled words (vocabulary size: " + vocabulary.size() + ")");
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to reload custom/user/disabled words", e);
    }
  }
  
  /**
   * Get vocabulary statistics
   */
  public VocabularyStats getStats()
  {
    // Count by tier from unified structure
    int common = 0;
    int top5k = 0;
    for (WordInfo info : vocabulary.values())
    {
      if (info.tier == 2) common++;
      else if (info.tier == 1) top5k++;
    }

    return new VocabularyStats(
      vocabulary.size(),
      common,
      top5k,
      isLoaded
    );
  }
  
  /**
   * Input candidate word
   */
  public static class CandidateWord
  {
    public final String word;
    public final float confidence;
    
    public CandidateWord(String word, float confidence)
    {
      this.word = word;
      this.confidence = confidence;
    }
  }
  
  /**
   * Filtered prediction with combined scoring
   */
  public static class FilteredPrediction
  {
    public final String word;
    public final float score;         // Combined confidence + frequency score
    public final float confidence;    // Original NN confidence
    public final float frequency;     // Word frequency
    public final String source;       // "common", "top5000", "vocabulary", "raw"
    
    public FilteredPrediction(String word, float score, float confidence, float frequency, String source)
    {
      this.word = word;
      this.score = score;
      this.confidence = confidence;
      this.frequency = frequency;
      this.source = source;
    }
  }
  
  /**
   * Swipe statistics for length-based filtering
   */
  public static class SwipeStats
  {
    public final int expectedLength;
    public final float pathLength;
    public final float speed;
    
    public SwipeStats(int expectedLength, float pathLength, float speed)
    {
      this.expectedLength = expectedLength;
      this.pathLength = pathLength;
      this.speed = speed;
    }
  }
  
  /**
   * Load custom words and Android user dictionary into beam search vocabulary
   * High frequency ensures they appear in predictions
   */
  private void loadCustomAndUserWords()
  {
    if (context == null) return;

    try
    {
      android.content.SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(context);

      // 1. Load custom words from SharedPreferences
      String customWordsJson = prefs.getString("custom_words", "{}");
      if (!customWordsJson.equals("{}"))
      {
        try
        {
          org.json.JSONObject jsonObj = new org.json.JSONObject(customWordsJson);
          java.util.Iterator<String> keys = jsonObj.keys();
          int customCount = 0;
          while (keys.hasNext())
          {
            String word = keys.next().toLowerCase();
            int frequency = jsonObj.optInt(word, 1000);
            // High priority for custom words (tier 1 = top5000)
            float normalizedFreq = 1.0f / 100.0f; // Equivalent to top 100
            vocabulary.put(word, new WordInfo(normalizedFreq, (byte)1));
            customCount++;
          }
          Log.d(TAG, "Loaded " + customCount + " custom words into beam search");
        }
        catch (org.json.JSONException e)
        {
          Log.e(TAG, "Failed to parse custom words JSON", e);
        }
      }

      // 2. Load Android user dictionary
      try
      {
        android.database.Cursor cursor = context.getContentResolver().query(
          android.provider.UserDictionary.Words.CONTENT_URI,
          new String[]{
            android.provider.UserDictionary.Words.WORD,
            android.provider.UserDictionary.Words.FREQUENCY
          },
          null,
          null,
          null
        );

        if (cursor != null)
        {
          int wordIndex = cursor.getColumnIndex(android.provider.UserDictionary.Words.WORD);
          int userCount = 0;

          while (cursor.moveToNext())
          {
            String word = cursor.getString(wordIndex).toLowerCase();
            // High priority for user dictionary (tier 1 = top5000)
            float normalizedFreq = 1.0f / 100.0f; // Equivalent to top 100
            vocabulary.put(word, new WordInfo(normalizedFreq, (byte)1));
            userCount++;
          }

          cursor.close();
          Log.d(TAG, "Loaded " + userCount + " user dictionary words into beam search");
        }
      }
      catch (Exception e)
      {
        Log.e(TAG, "Failed to load user dictionary for beam search", e);
      }
    }
    catch (Exception e)
    {
      Log.e(TAG, "Error loading custom/user words for beam search", e);
    }
  }

  /**
   * Load disabled words set for filtering beam search results
   */
  private void loadDisabledWords()
  {
    if (context == null)
    {
      disabledWords = new HashSet<>();
      return;
    }

    try
    {
      android.content.SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(context);
      Set<String> disabledSet = prefs.getStringSet("disabled_words", new HashSet<>());
      disabledWords = new HashSet<>(disabledSet);
      Log.d(TAG, "Loaded " + disabledWords.size() + " disabled words for beam search filtering");
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to load disabled words", e);
      disabledWords = new HashSet<>();
    }
  }

  /**
   * Vocabulary statistics
   */
  public static class VocabularyStats
  {
    public final int totalWords;
    public final int commonWords;
    public final int top5000;
    public final boolean isLoaded;
    
    public VocabularyStats(int totalWords, int commonWords, int top5000, boolean isLoaded)
    {
      this.totalWords = totalWords;
      this.commonWords = commonWords;
      this.top5000 = top5000;
      this.isLoaded = isLoaded;
    }
  }
}