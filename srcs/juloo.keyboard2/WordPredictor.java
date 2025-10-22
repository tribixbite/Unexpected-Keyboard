package juloo.keyboard2;

import android.content.Context;
import android.content.SharedPreferences;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Word prediction engine that matches swipe patterns to dictionary words
 */
public class WordPredictor
{
  private final Map<String, Integer> _dictionary;
  private BigramModel _bigramModel;
  private LanguageDetector _languageDetector;
  private String _currentLanguage;
  private List<String> _recentWords; // For language detection
  private static final int MAX_PREDICTIONS_TYPING = 5;
  private static final int MAX_PREDICTIONS_SWIPE = 10;
  private static final int MAX_EDIT_DISTANCE = 2;
  private static final int MAX_RECENT_WORDS = 20; // Keep last 20 words for language detection
  private Config _config;
  private UserAdaptationManager _adaptationManager;
  private Context _context; // For accessing SharedPreferences for disabled words
  private Set<String> _disabledWords; // Cache of disabled words

  // Static flag to signal all WordPredictor instances need to reload custom/user/disabled words
  private static volatile boolean _needsReload = false;
  private long _lastReloadTime = 0;
  
  public WordPredictor()
  {
    _dictionary = new HashMap<>();
    _bigramModel = new BigramModel();
    _languageDetector = new LanguageDetector();
    _currentLanguage = "en"; // Default to English
    _recentWords = new ArrayList<>();
    _config = null;
    _context = null;
    _disabledWords = new HashSet<>();
  }

  /**
   * Set context for accessing disabled words from SharedPreferences
   */
  public void setContext(Context context)
  {
    _context = context;
    loadDisabledWords();
  }

  /**
   * Load disabled words from SharedPreferences
   */
  private void loadDisabledWords()
  {
    if (_context == null) {
      _disabledWords = new HashSet<>();
      return;
    }

    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(_context);
    Set<String> disabledSet = prefs.getStringSet("disabled_words", new HashSet<>());
    // Create a new HashSet to avoid modifying the original
    _disabledWords = new HashSet<>(disabledSet);
    android.util.Log.d("WordPredictor", "Loaded " + _disabledWords.size() + " disabled words");
  }

  /**
   * Check if a word is disabled
   */
  private boolean isWordDisabled(String word)
  {
    return _disabledWords.contains(word.toLowerCase());
  }

  /**
   * Reload disabled words (called when Dictionary Manager updates the list)
   */
  public void reloadDisabledWords()
  {
    loadDisabledWords();
  }

  /**
   * Reload custom words and user dictionary (called when Dictionary Manager makes changes)
   * PERFORMANCE: Only reloads small dynamic sets, overwrites existing entries
   */
  public void reloadCustomAndUserWords()
  {
    if (_context != null)
    {
      loadCustomAndUserWords(_context);
      _lastReloadTime = System.currentTimeMillis();
      android.util.Log.d("WordPredictor", "Reloaded custom and user dictionary words");
    }
  }

  /**
   * Signal all WordPredictor instances to reload custom/user/disabled words on next prediction
   * Called by Dictionary Manager when user makes changes
   */
  public static void signalReloadNeeded()
  {
    _needsReload = true;
    android.util.Log.d("WordPredictor", "Reload signal set - all instances will reload on next prediction");
  }

  /**
   * Check if reload is needed and perform it
   * Called at start of prediction
   */
  private void checkAndReload()
  {
    if (_needsReload && _context != null)
    {
      reloadDisabledWords();
      reloadCustomAndUserWords();
      // Don't clear flag - let all instances reload
      android.util.Log.d("WordPredictor", "Auto-reloaded dictionaries due to signal");
    }
  }
  
  /**
   * Set the config for weight access
   */
  public void setConfig(Config config)
  {
    _config = config;
  }
  
  /**
   * Set the user adaptation manager for frequency adjustment
   */
  public void setUserAdaptationManager(UserAdaptationManager adaptationManager)
  {
    _adaptationManager = adaptationManager;
  }
  
  /**
   * Set the active language for N-gram predictions
   */
  public void setLanguage(String language)
  {
    _currentLanguage = language;
    if (_bigramModel != null)
    {
      _bigramModel.setLanguage(language);
      android.util.Log.d("WordPredictor", "N-gram language set to: " + language);
    }
  }
  
  /**
   * Get the current active language
   */
  public String getCurrentLanguage()
  {
    return _bigramModel != null ? _bigramModel.getCurrentLanguage() : "en";
  }
  
  /**
   * Check if a language is supported by the N-gram model
   */
  public boolean isLanguageSupported(String language)
  {
    return _bigramModel != null ? _bigramModel.isLanguageSupported(language) : false;
  }
  
  /**
   * Add a word to the recent words list for language detection
   */
  public void addWordToContext(String word)
  {
    if (word == null || word.trim().isEmpty())
      return;
    
    word = word.toLowerCase().trim();
    _recentWords.add(word);
    
    // Keep only the most recent words
    while (_recentWords.size() > MAX_RECENT_WORDS)
    {
      _recentWords.remove(0);
    }
    
    // Try to detect language change if we have enough words
    if (_recentWords.size() >= 5)
    {
      tryAutoLanguageDetection();
    }
  }
  
  /**
   * Try to automatically detect and switch language based on recent words
   */
  private void tryAutoLanguageDetection()
  {
    if (_languageDetector == null)
      return;
    
    String detectedLanguage = _languageDetector.detectLanguageFromWords(_recentWords);
    if (detectedLanguage != null && !detectedLanguage.equals(_currentLanguage))
    {
      // Only switch if the detected language is supported by our N-gram model
      if (_bigramModel.isLanguageSupported(detectedLanguage))
      {
        android.util.Log.d("WordPredictor", "Auto-detected language change from " + _currentLanguage + " to " + detectedLanguage);
        setLanguage(detectedLanguage);
      }
    }
  }
  
  /**
   * Manually detect language from a text sample
   */
  public String detectLanguage(String text)
  {
    return _languageDetector != null ? _languageDetector.detectLanguage(text) : null;
  }
  
  /**
   * Get the list of recent words used for language detection
   */
  public List<String> getRecentWords()
  {
    return new ArrayList<>(_recentWords);
  }
  
  /**
   * Clear the recent words context
   */
  public void clearContext()
  {
    _recentWords.clear();
  }
  
  /**
   * Load dictionary from assets
   */
  public void loadDictionary(Context context, String language)
  {
    _dictionary.clear();

    // Try JSON format first (50k words with frequencies)
    String jsonFilename = "dictionaries/" + language + "_enhanced.json";
    try
    {
      BufferedReader reader = new BufferedReader(
        new InputStreamReader(context.getAssets().open(jsonFilename)));
      StringBuilder jsonBuilder = new StringBuilder();
      String line;
      while ((line = reader.readLine()) != null)
      {
        jsonBuilder.append(line);
      }
      reader.close();

      // Parse JSON object
      org.json.JSONObject jsonDict = new org.json.JSONObject(jsonBuilder.toString());
      java.util.Iterator<String> keys = jsonDict.keys();
      while (keys.hasNext())
      {
        String word = keys.next().toLowerCase();
        int frequency = jsonDict.getInt(word);
        // Frequency is 128-255, scale to 100-10000 range for better scoring
        int scaledFreq = 100 + (int)((frequency - 128) / 127.0 * 9900);
        _dictionary.put(word, scaledFreq);
      }
      android.util.Log.d("WordPredictor", "Loaded JSON dictionary: " + jsonFilename + " with " + _dictionary.size() + " words");
    }
    catch (Exception e)
    {
      android.util.Log.w("WordPredictor", "JSON dictionary not found, trying text format: " + e.getMessage());

      // Fall back to text format (word-per-line)
      String textFilename = "dictionaries/" + language + "_enhanced.txt";
      try
      {
        BufferedReader reader = new BufferedReader(
          new InputStreamReader(context.getAssets().open(textFilename)));
        String line;
        while ((line = reader.readLine()) != null)
        {
          String word = line.trim().toLowerCase();
          if (!word.isEmpty())
          {
            _dictionary.put(word, 1000); // Default frequency
          }
        }
        reader.close();
        android.util.Log.d("WordPredictor", "Loaded text dictionary: " + textFilename + " with " + _dictionary.size() + " words");
      }
      catch (IOException e2)
      {
        android.util.Log.e("WordPredictor", "Failed to load dictionary: " + e2.getMessage());
      }
    }
    
    // Load custom words and user dictionary (additive to main dictionary)
    loadCustomAndUserWords(context);

    // Set the N-gram model language to match the dictionary
    setLanguage(language);
  }

  /**
   * Load custom words and Android user dictionary into predictions
   * Called during dictionary initialization for performance
   */
  private void loadCustomAndUserWords(Context context)
  {
    if (context == null) return;

    try
    {
      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(context);

      // 1. Load custom words from SharedPreferences
      String customWordsJson = prefs.getString("custom_words", "{}");
      if (!customWordsJson.equals("{}"))
      {
        try
        {
          // Parse JSON map: {"word": frequency, ...}
          org.json.JSONObject jsonObj = new org.json.JSONObject(customWordsJson);
          java.util.Iterator<String> keys = jsonObj.keys();
          int customCount = 0;
          while (keys.hasNext())
          {
            String word = keys.next().toLowerCase();
            int frequency = jsonObj.optInt(word, 1000);
            _dictionary.put(word, frequency);
            customCount++;
          }
          android.util.Log.d("WordPredictor", "Loaded " + customCount + " custom words");
        }
        catch (org.json.JSONException e)
        {
          android.util.Log.e("WordPredictor", "Failed to parse custom words JSON", e);
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
          int freqIndex = cursor.getColumnIndex(android.provider.UserDictionary.Words.FREQUENCY);
          int userCount = 0;

          while (cursor.moveToNext())
          {
            String word = cursor.getString(wordIndex).toLowerCase();
            int frequency = (freqIndex >= 0) ? cursor.getInt(freqIndex) : 1000;
            _dictionary.put(word, frequency);
            userCount++;
          }

          cursor.close();
          android.util.Log.d("WordPredictor", "Loaded " + userCount + " user dictionary words");
        }
      }
      catch (Exception e)
      {
        android.util.Log.e("WordPredictor", "Failed to load user dictionary", e);
      }
    }
    catch (Exception e)
    {
      android.util.Log.e("WordPredictor", "Error loading custom/user words", e);
    }
  }
  
  /**
   * Reset the predictor state - called after space/punctuation
   */
  public void reset()
  {
    // This method will be called from Keyboard2 to reset state
    // Dictionary remains loaded, just clears any internal state if needed
    android.util.Log.d("WordPredictor", "===== PREDICTOR RESET CALLED =====");
    android.util.Log.d("WordPredictor", "Stack trace: ", new Exception("Reset trace"));
  }

  /**
   * Predict words based on the sequence of touched keys
   * Returns list of predictions (for backward compatibility)
   */
  public List<String> predictWords(String keySequence)
  {
    PredictionResult result = predictWordsWithScores(keySequence);
    return result.words;
  }
  
  /**
   * Predict words with context (PUBLIC API - delegates to internal unified method)
   */
  public PredictionResult predictWordsWithContext(String keySequence, List<String> context)
  {
    return predictInternal(keySequence, context);
  }

  /**
   * Predict words and return with their scores (no context)
   */
  public PredictionResult predictWordsWithScores(String keySequence)
  {
    return predictInternal(keySequence, Collections.emptyList());
  }

  /**
   * UNIFIED prediction logic with early fusion of all signals
   * Context is applied to ALL candidates BEFORE selecting top N
   */
  private PredictionResult predictInternal(String keySequence, List<String> context)
  {
    if (keySequence == null || keySequence.isEmpty())
      return new PredictionResult(new ArrayList<>(), new ArrayList<>());

    // Check if dictionary changes require reload
    checkAndReload();

    PerformanceProfiler.start("Type.predictWordsWithScores");

    // UNIFIED SCORING with EARLY FUSION
    // Context is applied to ALL candidates BEFORE selecting top N
    List<WordCandidate> candidates = new ArrayList<>();
    String lowerSequence = keySequence.toLowerCase();

    android.util.Log.d("WordPredictor", "Predicting for: " + lowerSequence + " (len=" + lowerSequence.length() + ") with context: " + context);

    int maxPredictions = MAX_PREDICTIONS_TYPING;

    // Find all words that could match the typed prefix
    // TODO: CRITICAL PERFORMANCE - Iterates ALL 50,131 words on EVERY keystroke!
    // Should implement prefix indexing: Map<String, Set<String>> for 100x speedup
    // Example: "th" → {"the", "that", "there", ...} reduces 50k iterations to ~200
    for (Map.Entry<String, Integer> entry : _dictionary.entrySet())
    {
      String word = entry.getKey();
      int frequency = entry.getValue();

      // STRICT PREFIX MATCHING - Only suggest words that start with typed sequence
      if (!word.startsWith(lowerSequence))
        continue;  // Skip words that don't start with typed prefix

      // SKIP DISABLED WORDS - Filter out words disabled via Dictionary Manager
      if (isWordDisabled(word))
      {
        android.util.Log.d("WordPredictor", "Skipping disabled word: " + word);
        continue;
      }

      // UNIFIED SCORING: Combine ALL signals into one score BEFORE selection
      int score = calculateUnifiedScore(word, lowerSequence, frequency, context);

      if (score > 0)
      {
        candidates.add(new WordCandidate(word, score));
        android.util.Log.d("WordPredictor", "Candidate: " + word + " (score=" + score + ")");
      }
    }

    // Sort all candidates by score (descending)
    Collections.sort(candidates, new Comparator<WordCandidate>() {
      @Override
      public int compare(WordCandidate a, WordCandidate b) {
        return Integer.compare(b.score, a.score);
      }
    });

    // Extract top N predictions
    List<String> predictions = new ArrayList<>();
    List<Integer> scores = new ArrayList<>();

    for (WordCandidate candidate : candidates)
    {
      predictions.add(candidate.word);
      scores.add(candidate.score);
      if (predictions.size() >= maxPredictions) break;
    }

    android.util.Log.d("WordPredictor", "Final predictions (" + predictions.size() + "): " + predictions);
    android.util.Log.d("WordPredictor", "Scores: " + scores);

    PerformanceProfiler.end("Type.predictWordsWithScores");
    return new PredictionResult(predictions, scores);
  }
  
  /**
   * UNIFIED SCORING - Combines all prediction signals (early fusion)
   *
   * Combines: prefix quality + frequency + user adaptation + context probability
   * Context is evaluated for ALL candidates, not just top N (key improvement)
   *
   * @param word The word being scored
   * @param keySequence The typed prefix
   * @param frequency Dictionary frequency (higher = more common)
   * @param context Previous words for contextual prediction (can be empty)
   * @return Combined score
   */
  private int calculateUnifiedScore(String word, String keySequence, int frequency, List<String> context)
  {
    // 1. Base score from prefix match quality
    int prefixScore = calculatePrefixScore(word, keySequence);
    if (prefixScore == 0) return 0; // Should not happen if caller does prefix check

    // 2. User adaptation multiplier (learns user's vocabulary)
    float adaptationMultiplier = 1.0f;
    if (_adaptationManager != null)
    {
      adaptationMultiplier = _adaptationManager.getAdaptationMultiplier(word);
    }

    // 3. Context multiplier (bigram probability boost)
    float contextMultiplier = 1.0f;
    if (_bigramModel != null && context != null && !context.isEmpty())
    {
      contextMultiplier = _bigramModel.getContextMultiplier(word, context);
    }

    // 4. Frequency scaling (log to prevent common words from dominating)
    // Using log1p helps balance: "the" (freq ~10000) vs "think" (freq ~100)
    // Without log: "the" would always win. With log: context can override frequency
    // Scale factor is configurable (default: 1000.0)
    float frequencyScale = (_config != null) ? _config.prediction_frequency_scale : 1000.0f;
    float frequencyFactor = 1.0f + (float)Math.log1p(frequency / frequencyScale);

    // COMBINE ALL SIGNALS
    // Formula: prefixScore × adaptation × (1 + boosted_context) × freq_factor
    // Context boost is configurable (default: 2.0)
    // Higher boost = context has more influence on predictions
    float contextBoost = (_config != null) ? _config.prediction_context_boost : 2.0f;
    float finalScore = prefixScore
        * adaptationMultiplier
        * (1.0f + (contextMultiplier - 1.0f) * contextBoost)  // Configurable context boost
        * frequencyFactor;

    return (int)finalScore;
  }

  /**
   * Calculate base score for prefix-based matching (used by unified scoring)
   */
  private int calculatePrefixScore(String word, String keySequence)
  {
    // Direct match is highest score
    if (word.equals(keySequence))
      return 1000;

    // Word starts with sequence (this is guaranteed by caller, but score based on completion ratio)
    if (word.startsWith(keySequence))
    {
      // Score based on how much of the word is already typed
      float completionRatio = (float)keySequence.length() / word.length();

      // Higher score for more completion, but prefer shorter completions
      int baseScore = 800;

      // Bonus for more typed characters (longer prefix = more specific)
      int prefixBonus = keySequence.length() * 50;

      // Slight penalty for very long words to prefer common shorter words
      int lengthPenalty = Math.max(0, (word.length() - 6) * 10);

      return baseScore + prefixBonus - lengthPenalty;
    }

    return 0; // Should not reach here due to prefix check in caller
  }
  
  /**
   * Auto-correct a typed word after user presses space/punctuation.
   *
   * Finds dictionary words with:
   * - Same length
   * - Same first 2 letters
   * - High positional character match (default: 2/3 chars)
   *
   * Example: "teh" → "the", "Teh" → "The", "TEH" → "THE"
   *
   * @param typedWord The word user just finished typing
   * @return Corrected word, or original if no suitable correction found
   */
  public String autoCorrect(String typedWord)
  {
    if (_config == null || !_config.autocorrect_enabled || typedWord == null || typedWord.isEmpty())
    {
      return typedWord;
    }

    String lowerTypedWord = typedWord.toLowerCase();

    // 1. Do not correct words already in dictionary or user's vocabulary
    if (_dictionary.containsKey(lowerTypedWord) ||
        (_adaptationManager != null && _adaptationManager.getAdaptationMultiplier(lowerTypedWord) > 1.0f))
    {
      return typedWord;
    }

    // 2. Enforce minimum word length for correction
    if (lowerTypedWord.length() < _config.autocorrect_min_word_length)
    {
      return typedWord;
    }

    // 3. "Same first 2 letters" rule requires at least 2 characters
    if (lowerTypedWord.length() < 2)
    {
      return typedWord;
    }

    String prefix = lowerTypedWord.substring(0, 2);
    int wordLength = lowerTypedWord.length();
    WordCandidate bestCandidate = null;

    // 4. Iterate through dictionary to find candidates
    for (Map.Entry<String, Integer> entry : _dictionary.entrySet())
    {
      String dictWord = entry.getKey();

      // Heuristic 1: Must have same length
      if (dictWord.length() != wordLength)
      {
        continue;
      }

      // Heuristic 2: Must start with same first two letters
      if (!dictWord.startsWith(prefix))
      {
        continue;
      }

      // Heuristic 3: Calculate positional character match ratio
      int matchCount = 0;
      for (int i = 0; i < wordLength; i++)
      {
        if (lowerTypedWord.charAt(i) == dictWord.charAt(i))
        {
          matchCount++;
        }
      }

      float matchRatio = (float)matchCount / wordLength;
      if (matchRatio >= _config.autocorrect_char_match_threshold)
      {
        // Valid candidate - select if better than current best
        // "Better" = higher dictionary frequency
        int candidateFrequency = entry.getValue();
        if (bestCandidate == null || candidateFrequency > bestCandidate.score)
        {
          bestCandidate = new WordCandidate(dictWord, candidateFrequency);
        }
      }
    }

    // 5. Apply correction only if confident candidate found
    if (bestCandidate != null && bestCandidate.score >= _config.autocorrect_confidence_min_frequency)
    {
      // Preserve original capitalization (e.g., "Teh" → "The")
      String corrected = preserveCapitalization(typedWord, bestCandidate.word);
      android.util.Log.d("WordPredictor", "AUTO-CORRECT: '" + typedWord + "' → '" + corrected + "' (freq=" + bestCandidate.score + ")");
      return corrected;
    }

    return typedWord; // No suitable correction found
  }

  /**
   * Preserve capitalization of original word when applying correction.
   *
   * Examples:
   * - "teh" + "the" → "the"
   * - "Teh" + "the" → "The"
   * - "TEH" + "the" → "THE"
   */
  private String preserveCapitalization(String originalWord, String correctedWord)
  {
    if (originalWord.length() == 0 || correctedWord.length() == 0)
    {
      return correctedWord;
    }

    // Check if ALL uppercase
    boolean isAllUpper = true;
    for (int i = 0; i < originalWord.length(); i++)
    {
      if (Character.isLowerCase(originalWord.charAt(i)))
      {
        isAllUpper = false;
        break;
      }
    }

    if (isAllUpper)
    {
      return correctedWord.toUpperCase();
    }

    // Check if first letter uppercase (Title Case)
    if (Character.isUpperCase(originalWord.charAt(0)))
    {
      return Character.toUpperCase(correctedWord.charAt(0)) + correctedWord.substring(1);
    }

    return correctedWord;
  }

  /**
   * Get dictionary size
   */
  public int getDictionarySize()
  {
    return _dictionary.size();
  }
  
  /**
   * Helper class to store word candidates with scores
   */
  private static class WordCandidate
  {
    final String word;
    final int score;
    
    WordCandidate(String word, int score)
    {
      this.word = word;
      this.score = score;
    }
  }
  
  /**
   * Result class containing predictions and their scores
   */
  public static class PredictionResult
  {
    public final List<String> words;
    public final List<Integer> scores;
    
    public PredictionResult(List<String> words, List<Integer> scores)
    {
      this.words = words;
      this.scores = scores;
    }
  }
}