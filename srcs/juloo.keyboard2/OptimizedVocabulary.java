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
  
  // Scoring parameters (tuned for 50k vocabulary)
  private static final float CONFIDENCE_WEIGHT = 0.6f;
  private static final float FREQUENCY_WEIGHT = 0.4f;
  private static final float COMMON_WORDS_BOOST = 1.3f;  // Increased for 50k vocab
  private static final float TOP5000_BOOST = 1.0f;
  private static final float RARE_WORDS_PENALTY = 0.75f; // Strengthened for 50k vocab
  
  // Filtering thresholds
  private Map<Integer, Float> minFrequencyByLength;

  // Disabled words filter (for Dictionary Manager integration)
  private Set<String> disabledWords;

  // Contraction handling (for apostrophe display)
  // Maps base word -> list of contraction variants (e.g., "well" -> ["we'll"])
  private Map<String, List<String>> contractionPairings;
  // Maps apostrophe-free -> with apostrophe (e.g., "dont" -> "don't")
  private Map<String, String> nonPairedContractions;

  private boolean isLoaded = false;
  private Context context;

  public OptimizedVocabulary(Context context)
  {
    this.context = context;
    this.vocabulary = new HashMap<>();
    this.minFrequencyByLength = new HashMap<>();
    this.disabledWords = new HashSet<>();
    this.contractionPairings = new HashMap<>();
    this.nonPairedContractions = new HashMap<>();
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

      // Load contraction mappings for apostrophe display
      loadContractionMappings();

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

    // DEBUG: Show all raw beam search candidates BEFORE filtering
    // Enable debug mode if user has enabled "Detailed Pipeline Logging" in settings
    boolean debugMode = android.util.Log.isLoggable(TAG, android.util.Log.DEBUG);

    // v1.33+: Load ALL configurable parameters from preferences (OPTIMIZED: read once per swipe)
    float confidenceWeight = CONFIDENCE_WEIGHT;  // default
    float frequencyWeight = FREQUENCY_WEIGHT;    // default
    float commonBoost = COMMON_WORDS_BOOST;      // default
    float top5000Boost = TOP5000_BOOST;          // default
    float rarePenalty = RARE_WORDS_PENALTY;      // default

    // Autocorrect configuration (v1.33+: optimized to load once instead of inside loop)
    boolean swipeAutocorrectEnabled = true;  // default
    int maxLengthDiff = 2;
    int prefixLength = 2;
    int maxBeamCandidates = 3;
    int minWordLength = 3;
    float charMatchThreshold = 0.67f;
    boolean useEditDistance = true;  // v1.33.6: default to edit distance (more accurate)

    if (context != null)
    {
      try
      {
        android.content.SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(context);
        debugMode = prefs.getBoolean("swipe_debug_detailed_logging", false);

        // Read configurable scoring weights (v1.33+)
        confidenceWeight = prefs.getFloat("swipe_confidence_weight", CONFIDENCE_WEIGHT);
        frequencyWeight = prefs.getFloat("swipe_frequency_weight", FREQUENCY_WEIGHT);
        commonBoost = prefs.getFloat("swipe_common_words_boost", COMMON_WORDS_BOOST);
        top5000Boost = prefs.getFloat("swipe_top5000_boost", TOP5000_BOOST);
        rarePenalty = prefs.getFloat("swipe_rare_words_penalty", RARE_WORDS_PENALTY);

        // Read autocorrect configuration (v1.33.4: beam autocorrect only, final autocorrect handled separately)
        swipeAutocorrectEnabled = prefs.getBoolean("swipe_beam_autocorrect_enabled", true);
        maxLengthDiff = prefs.getInt("autocorrect_max_length_diff", 2);
        prefixLength = prefs.getInt("autocorrect_prefix_length", 2);
        maxBeamCandidates = prefs.getInt("autocorrect_max_beam_candidates", 3);
        minWordLength = prefs.getInt("autocorrect_min_word_length", 3);
        charMatchThreshold = prefs.getFloat("autocorrect_char_match_threshold", 0.67f);

        // v1.33.6: Fuzzy matching algorithm selection (edit_distance or positional)
        String fuzzyMatchMode = prefs.getString("swipe_fuzzy_match_mode", "edit_distance");
        useEditDistance = "edit_distance".equals(fuzzyMatchMode);
      }
      catch (Exception e)
      {
        // Ignore - use default values
      }
    }

    if (debugMode && !rawPredictions.isEmpty())
    {
      StringBuilder debug = new StringBuilder("\n🔍 VOCABULARY FILTERING DEBUG (top 10 beam search outputs):\n");
      debug.append("─────────────────────────────────────────────────────────\n");
      int numToShow = Math.min(10, rawPredictions.size());
      for (int i = 0; i < numToShow; i++)
      {
        CandidateWord candidate = rawPredictions.get(i);
        debug.append(String.format("#%d: \"%s\" (NN confidence: %.4f)\n", i+1, candidate.word, candidate.confidence));
      }
      String debugMsg = debug.toString();
      Log.d(TAG, debugMsg);
      sendDebugLog(debugMsg);
    }

    List<FilteredPrediction> validPredictions = new ArrayList<>();
    StringBuilder detailedLog = debugMode ? new StringBuilder("\n📊 DETAILED FILTERING PROCESS:\n") : null;
    if (debugMode) detailedLog.append("─────────────────────────────────────────────────────────\n");

    for (CandidateWord candidate : rawPredictions)
    {
      String word = candidate.word.toLowerCase().trim();

      // Skip invalid word formats
      if (!word.matches("^[a-z]+$"))
      {
        if (debugMode) detailedLog.append(String.format("❌ \"%s\" - INVALID FORMAT (not a-z)\n", candidate.word));
        continue;
      }

      // FILTER OUT DISABLED WORDS (Dictionary Manager integration)
      if (disabledWords.contains(word))
      {
        if (debugMode) detailedLog.append(String.format("❌ \"%s\" - DISABLED by user\n", word));
        continue; // Skip disabled words from beam search
      }

      // CRITICAL OPTIMIZATION: SINGLE hash lookup (was 3 lookups!)
      WordInfo info = vocabulary.get(word);
      if (info == null)
      {
        if (debugMode) detailedLog.append(String.format("❌ \"%s\" - NOT IN VOCABULARY (not in main/custom/user dict)\n", word));
        continue; // Word not in vocabulary
      }

      // OPTIMIZATION: Tier is embedded in WordInfo (no additional lookups!)
      // v1.33+: Use configurable boost values instead of hardcoded constants
      float boost;
      String source;

      switch (info.tier)
      {
        case 2: // common (top 100)
          boost = commonBoost;  // v1.33+: configurable (default: 1.3)
          source = "common";
          break;
        case 1: // top5000
          boost = top5000Boost;  // v1.33+: configurable (default: 1.0)
          source = "top5000";
          break;
        default: // regular
          // Check frequency threshold for rare words
          float minFreq = getMinFrequency(word.length());
          if (info.frequency < minFreq)
          {
            if (debugMode) detailedLog.append(String.format("❌ \"%s\" - BELOW FREQUENCY THRESHOLD (freq=%.4f < min=%.4f for length %d)\n",
              word, info.frequency, minFreq, word.length()));
            continue; // Below threshold
          }
          boost = rarePenalty;  // v1.33+: configurable (default: 0.75)
          source = "vocabulary";
          break;
      }

      // v1.33+: Pass configurable weights to scoring function
      float score = calculateCombinedScore(candidate.confidence, info.frequency, boost, confidenceWeight, frequencyWeight);
      validPredictions.add(new FilteredPrediction(word, score, candidate.confidence, info.frequency, source));

      // DEBUG: Show successful candidates with all scoring details
      if (debugMode)
      {
        detailedLog.append(String.format("✅ \"%s\" - KEPT (tier=%d, freq=%.4f, boost=%.2fx, NN=%.4f → score=%.4f) [%s]\n",
          word, info.tier, info.frequency, boost, candidate.confidence, score, source));
      }
    }

    if (debugMode && detailedLog != null)
    {
      detailedLog.append("─────────────────────────────────────────────────────────\n");
      String detailedMsg = detailedLog.toString();
      Log.d(TAG, detailedMsg);
      sendDebugLog(detailedMsg);
    }

    // Sort by combined score (confidence + frequency)
    validPredictions.sort((a, b) -> Float.compare(b.score, a.score));

    // AUTOCORRECT FOR SWIPE: Fuzzy match top beam candidates against custom words
    // This allows "parametrek" (custom) to match "parameters" (beam output)
    // v1.33+: OPTIMIZED - uses pre-loaded config from top of method (no redundant prefs reads)
    // v1.33.1: CRITICAL FIX - removed isEmpty check and match against raw beam outputs
    if (swipeAutocorrectEnabled && context != null && !rawPredictions.isEmpty())
    {
      try
      {
        android.content.SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(context);
        // Get custom words (only SharedPreferences read in autocorrect block now)
        String customWordsJson = prefs.getString("custom_words", "{}");
        if (!customWordsJson.equals("{}"))
        {
          org.json.JSONObject jsonObj = new org.json.JSONObject(customWordsJson);
          java.util.Iterator<String> keys = jsonObj.keys();

          // For each custom word, check if it fuzzy matches any top beam candidate
          while (keys.hasNext())
          {
            String customWord = keys.next().toLowerCase();
            int customFreq = jsonObj.optInt(customWord, 1000);

            // Check top N RAW beam candidates for fuzzy match (v1.33.1: CRITICAL FIX - was using validPredictions)
            // This allows autocorrect to work even when ALL beam outputs are rejected by vocabulary filtering
            for (int i = 0; i < Math.min(maxBeamCandidates, rawPredictions.size()); i++)
            {
              String beamWord = rawPredictions.get(i).word;

              // v1.33+: Configurable fuzzy matching (uses pre-loaded params)
              if (fuzzyMatch(customWord, beamWord, charMatchThreshold, maxLengthDiff, prefixLength, minWordLength))
              {
                // Add custom word as autocorrect suggestion
                float normalizedFreq = Math.max(0.0f, (float)(customFreq - 1) / 9999.0f);
                byte tier = (customFreq >= 8000) ? (byte)2 : (byte)1;
                // v1.33+: Use configurable boost values
                float boost = (tier == 2) ? commonBoost : top5000Boost;

                // Use RAW beam candidate's confidence for scoring (v1.33.1: CRITICAL FIX - was using validPredictions)
                float confidence = rawPredictions.get(i).confidence;

                // v1.33.3: MULTIPLICATIVE SCORING - match quality dominates
                // Custom words: base_score = NN_confidence (ignore frequency)
                // final_score = base_score × (match_quality^3) × tier_boost
                float matchQuality = calculateMatchQuality(customWord, beamWord, useEditDistance);
                float matchPower = matchQuality * matchQuality * matchQuality; // Cubic
                float baseScore = confidence;  // Ignore frequency for custom words
                float score = baseScore * matchPower * boost;

                validPredictions.add(new FilteredPrediction(customWord, score, confidence, normalizedFreq, "autocorrect"));

                if (debugMode)
                {
                  String matchMsg = String.format("🔄 AUTOCORRECT: \"%s\" (custom) matches \"%s\" (beam) → added with score=%.4f\n",
                    customWord, beamWord, score);
                  Log.d(TAG, matchMsg);
                  sendDebugLog(matchMsg);
                }
                break; // Only match once per custom word
              }
            }
          }

          // Re-sort after adding autocorrect suggestions
          validPredictions.sort((a, b) -> Float.compare(b.score, a.score));
        }
      }
      catch (Exception e)
      {
        Log.e(TAG, "Failed to apply autocorrect to beam candidates", e);
      }
    }

    // MAIN DICTIONARY FUZZY MATCHING: Match rejected beam outputs against dictionary words
    // v1.33.1: NEW - allows "proxity" (beam) to match "proximity" (dict)
    // Only run if autocorrect is enabled and we have few/no valid predictions
    if (swipeAutocorrectEnabled && validPredictions.size() < 3 && !rawPredictions.isEmpty())
    {
      try
      {
        if (debugMode)
        {
          String fuzzyMsg = String.format("\n🔍 MAIN DICTIONARY FUZZY MATCHING (validPredictions=%d, trying to rescue rejected beam outputs):\n", validPredictions.size());
          Log.d(TAG, fuzzyMsg);
          sendDebugLog(fuzzyMsg);
        }

        // Check top beam candidates that were rejected by vocabulary filtering
        for (int i = 0; i < Math.min(maxBeamCandidates, rawPredictions.size()); i++)
        {
          String beamWord = rawPredictions.get(i).word.toLowerCase().trim();
          float beamConfidence = rawPredictions.get(i).confidence;

          // Skip if this beam word already passed vocabulary filtering
          if (vocabulary.containsKey(beamWord))
          {
            continue; // Already in validPredictions
          }

          // Try fuzzy matching against dictionary words of similar length
          // v1.33.2: CRITICAL FIX - find BEST match (highest score), not FIRST match
          int targetLength = beamWord.length();
          String bestMatch = null;
          float bestScore = 0.0f;
          float bestFrequency = 0.0f;
          String bestSource = null;

          for (java.util.Map.Entry<String, WordInfo> entry : vocabulary.entrySet())
          {
            String dictWord = entry.getKey();
            WordInfo info = entry.getValue();

            // Only check words of similar length (performance optimization)
            if (Math.abs(dictWord.length() - targetLength) > maxLengthDiff)
            {
              continue;
            }

            // Skip disabled words
            if (disabledWords.contains(dictWord))
            {
              continue;
            }

            // Try fuzzy matching
            if (fuzzyMatch(dictWord, beamWord, charMatchThreshold, maxLengthDiff, prefixLength, minWordLength))
            {
              // Determine tier boost for matched word
              float boost;
              String source;
              switch (info.tier)
              {
                case 2:
                  boost = commonBoost;
                  source = "dict-fuzzy-common";
                  break;
                case 1:
                  boost = top5000Boost;
                  source = "dict-fuzzy-top5k";
                  break;
                default:
                  boost = rarePenalty;
                  source = "dict-fuzzy";
                  break;
              }

              // v1.33.3: MULTIPLICATIVE SCORING - match quality dominates
              // Dict fuzzy: base_score = (0.7×NN + 0.3×freq)
              // final_score = base_score × (match_quality^3) × tier_boost
              float matchQuality = calculateMatchQuality(dictWord, beamWord, useEditDistance);
              float matchPower = matchQuality * matchQuality * matchQuality; // Cubic
              float baseScore = (0.7f * beamConfidence) + (0.3f * info.frequency);
              float score = baseScore * matchPower * boost;

              // Keep track of best match (v1.33.2: don't break on first match!)
              if (score > bestScore)
              {
                bestScore = score;
                bestMatch = dictWord;
                bestFrequency = info.frequency;
                bestSource = source;
              }
            }
          }

          // Add the best match found for this beam word (if any)
          if (bestMatch != null)
          {
            validPredictions.add(new FilteredPrediction(bestMatch, bestScore, beamConfidence, bestFrequency, bestSource));

            if (debugMode)
            {
              String matchMsg = String.format("🔄 DICT FUZZY: \"%s\" (dict) matches \"%s\" (beam #%d, NN=%.4f) → added with score=%.4f\n",
                bestMatch, beamWord, i+1, beamConfidence, bestScore);
              Log.d(TAG, matchMsg);
              sendDebugLog(matchMsg);
            }
          }
        }

        // Re-sort after adding fuzzy matches
        if (!validPredictions.isEmpty())
        {
          validPredictions.sort((a, b) -> Float.compare(b.score, a.score));
        }
      }
      catch (Exception e)
      {
        Log.e(TAG, "Failed to apply dictionary fuzzy matching", e);
      }
    }

    // CONTRACTION HANDLING: Add paired variants and modify non-paired contractions
    if (!contractionPairings.isEmpty() || !nonPairedContractions.isEmpty())
    {
      try
      {
        List<FilteredPrediction> contractionVariants = new ArrayList<>();
        List<Integer> indicesToModify = new ArrayList<>();

        // Process each prediction for contractions
        for (int i = 0; i < validPredictions.size(); i++)
        {
          FilteredPrediction pred = validPredictions.get(i);
          String word = pred.word;

          // Check for paired contractions (base word exists: "well" -> "we'll")
          if (contractionPairings.containsKey(word))
          {
            List<String> contractions = contractionPairings.get(word);
            for (String contraction : contractions)
            {
              // Add contraction variant with slightly lower score (0.95x)
              // This ensures base word appears first, followed by contraction
              // word = base (for insertion), displayText = contraction (for UI)
              float variantScore = pred.score * 0.95f;
              contractionVariants.add(new FilteredPrediction(
                word,                    // word for insertion (apostrophe-free base)
                contraction,             // displayText for UI (with apostrophe)
                variantScore,
                pred.confidence,
                pred.frequency,
                pred.source + "-contraction"
              ));

              if (debugMode)
              {
                String msg = String.format("📝 CONTRACTION PAIRING: \"%s\" → added variant \"%s\" (word=%s, display=%s, score=%.4f)\n",
                  word, contraction, word, contraction, variantScore);
                Log.d(TAG, msg);
                sendDebugLog(msg);
              }
            }
          }

          // Check for non-paired contractions (base doesn't exist: "dont" -> "don't")
          if (nonPairedContractions.containsKey(word))
          {
            // Mark this index for modification
            indicesToModify.add(i);
          }
        }

        // Modify non-paired contractions (set displayText to apostrophe version)
        for (int idx : indicesToModify)
        {
          FilteredPrediction pred = validPredictions.get(idx);
          String withoutApostrophe = pred.word;
          String withApostrophe = nonPairedContractions.get(withoutApostrophe);

          // Replace with version that has displayText set to apostrophe form
          // word stays apostrophe-free (for insertion), displayText shows apostrophe
          validPredictions.set(idx, new FilteredPrediction(
            withoutApostrophe,       // word for insertion (no apostrophe)
            withApostrophe,          // displayText for UI (with apostrophe)
            pred.score,
            pred.confidence,
            pred.frequency,
            pred.source
          ));

          if (debugMode)
          {
            String msg = String.format("📝 NON-PAIRED CONTRACTION: \"%s\" → display modified to \"%s\" (word=%s)\n",
              withoutApostrophe, withApostrophe, withoutApostrophe);
            Log.d(TAG, msg);
            sendDebugLog(msg);
          }
        }

        // Add all contraction variants
        if (!contractionVariants.isEmpty())
        {
          validPredictions.addAll(contractionVariants);
          // Re-sort after adding variants
          validPredictions.sort((a, b) -> Float.compare(b.score, a.score));
        }
      }
      catch (Exception e)
      {
        Log.e(TAG, "Failed to apply contraction modifications", e);
      }
    }

    // DEBUG: Show final ranking
    if (debugMode && !validPredictions.isEmpty())
    {
      StringBuilder ranking = new StringBuilder("\n🏆 FINAL RANKING (after combining NN + frequency):\n");
      ranking.append("─────────────────────────────────────────────────────────\n");
      int numToShow = Math.min(10, validPredictions.size());
      for (int i = 0; i < numToShow; i++)
      {
        FilteredPrediction pred = validPredictions.get(i);
        String displayInfo = pred.word.equals(pred.displayText) ? "" : " (display=\"" + pred.displayText + "\")";
        ranking.append(String.format("#%d: \"%s\"%s (score=%.4f, NN=%.4f, freq=%.4f) [%s]\n",
          i+1, pred.word, displayInfo, pred.score, pred.confidence, pred.frequency, pred.source));
      }
      ranking.append("─────────────────────────────────────────────────────────\n");
      String rankingMsg = ranking.toString();
      Log.d(TAG, rankingMsg);
      sendDebugLog(rankingMsg);
    }

    // Apply swipe-specific filtering if needed
    if (swipeStats != null && swipeStats.expectedLength > 0)
    {
      return filterByExpectedLength(validPredictions, swipeStats.expectedLength);
    }

    return validPredictions.subList(0, Math.min(validPredictions.size(), 10));
  }
  
  /**
   * Calculate combined score from NN confidence and word frequency
   * Frequency is already normalized to 0.0-1.0 range where 1.0 = most frequent
   */
  /**
   * Calculate combined score from neural network confidence and dictionary frequency
   * v1.33+: Accepts configurable weights instead of using hardcoded constants
   *
   * @param confidence NN confidence from beam search (0.0-1.0)
   * @param frequency Dictionary frequency (0.0-1.0, already normalized)
   * @param boost Tier-based boost multiplier
   * @param confidenceWeight Weight for NN confidence (default: 0.6)
   * @param frequencyWeight Weight for dictionary frequency (default: 0.4)
   */
  private float calculateCombinedScore(float confidence, float frequency, float boost,
                                      float confidenceWeight, float frequencyWeight)
  {
    // Use frequency directly - already normalized to [0,1] by loading code
    // FIXED: Previous log10 formula was inverted (rare words scored higher than common)
    float freqScore = frequency;

    // Weighted combination with boost factor (v1.33+: configurable weights)
    return (confidenceWeight * confidence + frequencyWeight * freqScore) * boost;
  }
  
  /**
   * Load word frequencies from dictionary files
   * OPTIMIZATION: Single-lookup structure with tier embedded (1 lookup instead of 3)
   */
  private void loadWordFrequencies()
  {
    // Try JSON format first (50k words with actual frequencies)
    try
    {
      InputStream inputStream = context.getAssets().open("dictionaries/en_enhanced.json");
      BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
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
      int wordCount = 0;

      // First pass: collect all words with frequencies to determine tiers
      java.util.List<java.util.Map.Entry<String, Integer>> wordFreqList = new java.util.ArrayList<>();
      while (keys.hasNext())
      {
        String word = keys.next().toLowerCase();
        if (word.matches("^[a-z]+$"))
        {
          int freq = jsonDict.getInt(word);
          wordFreqList.add(new java.util.AbstractMap.SimpleEntry<>(word, freq));
        }
      }

      // Sort by frequency descending (highest frequency first)
      java.util.Collections.sort(wordFreqList, new java.util.Comparator<java.util.Map.Entry<String, Integer>>() {
        @Override
        public int compare(java.util.Map.Entry<String, Integer> a, java.util.Map.Entry<String, Integer> b) {
          return Integer.compare(b.getValue(), a.getValue());
        }
      });

      // Second pass: assign tiers based on sorted position
      for (int i = 0; i < wordFreqList.size() && i < 150000; i++)
      {
        java.util.Map.Entry<String, Integer> entry = wordFreqList.get(i);
        String word = entry.getKey();
        int rawFreq = entry.getValue();

        // Normalize frequency from 128-255 range to 0-1 range
        float frequency = (rawFreq - 128) / 127.0f;

        // Determine tier based on sorted position
        // Tightened thresholds for 50k vocabulary (was top 5000, now top 3000)
        byte tier;
        if (i < 100) {
          tier = 2; // common (top 100)
        } else if (i < 3000) {
          tier = 1; // top3000 (6% of 50k vocab)
        } else {
          tier = 0; // regular
        }

        vocabulary.put(word, new WordInfo(frequency, tier));
        wordCount++;
      }

      Log.d(TAG, "Loaded JSON vocabulary: " + wordCount + " words with frequency tiers");
    }
    catch (Exception e)
    {
      Log.w(TAG, "JSON vocabulary not found, falling back to text format: " + e.getMessage());

      // Fall back to text format (position-based frequency)
      try
      {
        InputStream inputStream = context.getAssets().open("dictionaries/en.txt");
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

        String line;
        int wordCount = 0;
        while ((line = reader.readLine()) != null)
        {
          line = line.trim().toLowerCase();
          if (!line.isEmpty() && line.matches("^[a-z]+$"))
          {
            // Position-based frequency
            float frequency = 1.0f / (wordCount + 1.0f);

            // Determine tier based on position
            byte tier;
            if (wordCount < 100) {
              tier = 2; // common
            } else if (wordCount < 5000) {
              tier = 1; // top5000
            } else {
              tier = 0; // regular
            }

            vocabulary.put(line, new WordInfo(frequency, tier));
            wordCount++;

            if (wordCount >= 150000) break;
          }
        }

        reader.close();
        Log.d(TAG, "Loaded text vocabulary: " + wordCount + " words");
      }
      catch (IOException e2)
      {
        Log.e(TAG, "Failed to load word frequencies", e2);
        throw new RuntimeException("Could not load vocabulary", e2);
      }
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
  /**
   * Fuzzy match two words using autocorrect criteria (v1.33+: configurable):
   * - Length difference within threshold (default: ±2)
   * - Same first N characters (default: 2)
   * - At least X% of characters match (default 67%)
   *
   * @param word1 First word to compare
   * @param word2 Second word to compare
   * @param charMatchThreshold Required character match ratio (0.0-1.0)
   * @param maxLengthDiff Maximum allowed length difference (e.g., 2 allows "parameter" vs "parametrek")
   * @param prefixLength Number of prefix characters that must match exactly
   * @param minWordLength Minimum word length for fuzzy matching
   */
  private boolean fuzzyMatch(String word1, String word2, float charMatchThreshold,
                            int maxLengthDiff, int prefixLength, int minWordLength)
  {
    // Check minimum word length
    if (word1.length() < minWordLength || word2.length() < minWordLength) return false;

    // Check length difference (v1.33+: configurable, was hardcoded same-length requirement)
    int lengthDiff = Math.abs(word1.length() - word2.length());
    if (lengthDiff > maxLengthDiff) return false;

    // Check prefix match (v1.33+: configurable prefix length)
    int actualPrefixLen = Math.min(prefixLength, Math.min(word1.length(), word2.length()));
    if (actualPrefixLen > 0 && !word1.substring(0, actualPrefixLen).equals(word2.substring(0, actualPrefixLen)))
    {
      return false;
    }

    // Count matching characters at the same position
    int matches = 0;
    int maxLength = Math.max(word1.length(), word2.length());
    int minLength = Math.min(word1.length(), word2.length());

    for (int i = 0; i < minLength; i++)
    {
      if (word1.charAt(i) == word2.charAt(i))
      {
        matches++;
      }
    }

    // Calculate match ratio using shorter word length as denominator
    // This allows "parametrek" (10 chars) to match "parameter" (9 chars)
    // Example: "parametrek" vs "parameter" → 9/9 = 100% match (all chars of shorter word match)
    float matchRatio = (float)matches / minLength;
    return matchRatio >= charMatchThreshold;
  }

  /**
   * Calculate Levenshtein distance (edit distance) between two words
   * Counts minimum insertions, deletions, and substitutions needed to transform one word into another
   *
   * v1.33.6: Levenshtein distance for accurate fuzzy matching
   * Better handles insertions/deletions that shift character positions
   * Example: "swollen" vs "swolen" → distance 1 (1 deletion)
   *          "swollen" vs "swore"  → distance 4 (much worse match)
   *
   * @param s1 First word
   * @param s2 Second word
   * @return Edit distance (0 = identical, higher = more different)
   */
  private int calculateLevenshteinDistance(String s1, String s2)
  {
    int len1 = s1.length();
    int len2 = s2.length();

    // Early exit for identical strings
    if (s1.equals(s2)) return 0;

    // Early exit for empty strings
    if (len1 == 0) return len2;
    if (len2 == 0) return len1;

    // Create distance matrix
    int[][] dp = new int[len1 + 1][len2 + 1];

    // Initialize first row and column
    for (int i = 0; i <= len1; i++) dp[i][0] = i;
    for (int j = 0; j <= len2; j++) dp[0][j] = j;

    // Fill matrix using dynamic programming
    for (int i = 1; i <= len1; i++)
    {
      for (int j = 1; j <= len2; j++)
      {
        int cost = (s1.charAt(i - 1) == s2.charAt(j - 1)) ? 0 : 1;

        dp[i][j] = Math.min(
          Math.min(
            dp[i - 1][j] + 1,      // Deletion
            dp[i][j - 1] + 1),     // Insertion
            dp[i - 1][j - 1] + cost  // Substitution
        );
      }
    }

    return dp[len1][len2];
  }

  /**
   * Calculate match quality between two words using configurable algorithm
   * Supports both positional matching (legacy) and edit distance (recommended)
   * Uses TARGET (dict word) length as denominator per user requirement
   *
   * v1.33.6: Configurable fuzzy matching algorithm
   * - Positional: Count matching chars at same positions (fails on insertions/deletions)
   * - Edit Distance: Levenshtein distance (handles insertions/deletions correctly)
   *
   * v1.33.3: Multiplicative scoring - match quality dramatically affects final score
   * Example: "proximity" vs "proxibity"
   *   - Positional: 8 chars match at same positions → 8/9 = 0.889
   *   - Edit Distance: distance 1 → quality 1 - (1/9) = 0.889
   *
   * @param dictWord The dictionary word (target)
   * @param beamWord The beam search output (source)
   * @param useEditDistance If true, use Levenshtein distance; if false, use positional matching
   * @return Match quality ratio 0.0-1.0 (1.0 = perfect match)
   */
  private float calculateMatchQuality(String dictWord, String beamWord, boolean useEditDistance)
  {
    if (useEditDistance)
    {
      // Edit distance algorithm: more accurate for insertions/deletions
      int distance = calculateLevenshteinDistance(dictWord, beamWord);

      // Convert distance to quality ratio (0.0-1.0)
      // Perfect match (distance=0) → quality=1.0
      // Distance equal to word length → quality=0.0
      int maxDistance = Math.max(dictWord.length(), beamWord.length());
      return 1.0f - ((float)distance / maxDistance);
    }
    else
    {
      // Positional matching algorithm: legacy behavior
      int matches = 0;
      int minLen = Math.min(dictWord.length(), beamWord.length());

      // Count positional character matches
      for (int i = 0; i < minLen; i++)
      {
        if (dictWord.charAt(i) == beamWord.charAt(i))
        {
          matches++;
        }
      }

      // Use TARGET (dict word) length as denominator
      // This gives higher match quality when more of the target is matched
      return (float)matches / dictWord.length();
    }
  }

  /**
   * Calculate match quality using default algorithm (edit distance)
   * Wrapper for backwards compatibility
   */
  private float calculateMatchQuality(String dictWord, String beamWord)
  {
    return calculateMatchQuality(dictWord, beamWord, true); // Default to edit distance
  }

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
    public final String word;          // Word for insertion (apostrophe-free)
    public final String displayText;   // Text for UI display (with apostrophes)
    public final float score;          // Combined confidence + frequency score
    public final float confidence;     // Original NN confidence
    public final float frequency;      // Word frequency
    public final String source;        // "common", "top5000", "vocabulary", "raw"

    public FilteredPrediction(String word, float score, float confidence, float frequency, String source)
    {
      this.word = word;
      this.displayText = word;  // Default: display = word
      this.score = score;
      this.confidence = confidence;
      this.frequency = frequency;
      this.source = source;
    }

    // Constructor with explicit displayText
    public FilteredPrediction(String word, String displayText, float score, float confidence, float frequency, String source)
    {
      this.word = word;
      this.displayText = displayText;
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
            int frequency = jsonObj.optInt(word, 1000); // Raw frequency 1-10000

            // Normalize frequency to 0.0-1.0 range (1.0 = most frequent)
            // Aligns with main dictionary normalization
            float normalizedFreq = Math.max(0.0f, (float)(frequency - 1) / 9999.0f);

            // Assign tier dynamically based on frequency
            // Very high frequency (>=8000) = tier 2 (common boost)
            // Otherwise = tier 1 (top5000 boost)
            byte tier = (frequency >= 8000) ? (byte)2 : (byte)1;

            vocabulary.put(word, new WordInfo(normalizedFreq, tier));
            customCount++;

            // DEBUG: Log each custom word loaded
            if (android.util.Log.isLoggable(TAG, android.util.Log.DEBUG))
            {
              String debugMsg = String.format("  Custom word loaded: \"%s\" (freq=%d → normalized=%.4f, tier=%d)\n",
                word, frequency, normalizedFreq, tier);
              Log.d(TAG, debugMsg);
              sendDebugLog(debugMsg);
            }
          }
          String loadMsg = "Loaded " + customCount + " custom words into beam search (frequency-based tiers)";
          Log.d(TAG, loadMsg);
          sendDebugLog(loadMsg + "\n");
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
            // User dictionary words should rank HIGH - user explicitly added them
            // CRITICAL: Previous value (250 → 0.025) ranked user words at position 48,736!
            int frequency = 9000;

            // Normalize to 0-1 range (~0.90)
            float normalizedFreq = Math.max(0.0f, (float)(frequency - 1) / 9999.0f);

            // Assign tier 2 (common boost) - user words are important
            byte tier = 2;

            vocabulary.put(word, new WordInfo(normalizedFreq, tier));
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
   * Load contraction mappings for apostrophe display support
   * Loads both paired contractions (base word exists: "well" -> "we'll")
   * and non-paired contractions (base doesn't exist: "dont" -> "don't")
   */
  private void loadContractionMappings()
  {
    if (context == null)
    {
      contractionPairings = new HashMap<>();
      nonPairedContractions = new HashMap<>();
      return;
    }

    try
    {
      // Load paired contractions (base word -> list of contraction variants)
      try
      {
        InputStream inputStream = context.getAssets().open("dictionaries/contraction_pairings.json");
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        StringBuilder jsonBuilder = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null)
        {
          jsonBuilder.append(line);
        }
        reader.close();

        // Parse JSON object: { "well": [{"contraction": "we'll", "frequency": 243}], ... }
        org.json.JSONObject jsonObj = new org.json.JSONObject(jsonBuilder.toString());
        java.util.Iterator<String> keys = jsonObj.keys();
        int pairingCount = 0;

        while (keys.hasNext())
        {
          String baseWord = keys.next().toLowerCase();
          org.json.JSONArray contractionArray = jsonObj.getJSONArray(baseWord);
          List<String> contractionList = new ArrayList<>();

          for (int i = 0; i < contractionArray.length(); i++)
          {
            org.json.JSONObject contractionObj = contractionArray.getJSONObject(i);
            String contraction = contractionObj.getString("contraction").toLowerCase();
            contractionList.add(contraction);
          }

          contractionPairings.put(baseWord, contractionList);
          pairingCount += contractionList.size();
        }

        Log.d(TAG, "Loaded " + pairingCount + " paired contractions for " + contractionPairings.size() + " base words");
      }
      catch (Exception e)
      {
        Log.w(TAG, "Failed to load contraction pairings: " + e.getMessage());
        contractionPairings = new HashMap<>();
      }

      // Load non-paired contractions (without apostrophe -> with apostrophe)
      try
      {
        InputStream inputStream = context.getAssets().open("dictionaries/contractions_non_paired.json");
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        StringBuilder jsonBuilder = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null)
        {
          jsonBuilder.append(line);
        }
        reader.close();

        // Parse JSON object: { "dont": "don't", "cant": "can't", ... }
        org.json.JSONObject jsonObj = new org.json.JSONObject(jsonBuilder.toString());
        java.util.Iterator<String> keys = jsonObj.keys();

        while (keys.hasNext())
        {
          String withoutApostrophe = keys.next().toLowerCase();
          String withApostrophe = jsonObj.getString(withoutApostrophe).toLowerCase();
          nonPairedContractions.put(withoutApostrophe, withApostrophe);
        }

        Log.d(TAG, "Loaded " + nonPairedContractions.size() + " non-paired contractions");
      }
      catch (Exception e)
      {
        Log.w(TAG, "Failed to load non-paired contractions: " + e.getMessage());
        nonPairedContractions = new HashMap<>();
      }
    }
    catch (Exception e)
    {
      Log.e(TAG, "Error loading contraction mappings", e);
      contractionPairings = new HashMap<>();
      nonPairedContractions = new HashMap<>();
    }
  }

    /**
   * Send debug log message to SwipeDebugActivity if available
   * Sends broadcast to be picked up by debug activity
   */
  private void sendDebugLog(String message)
  {
    if (context == null) return;

    try
    {
      android.content.Intent intent = new android.content.Intent("juloo.keyboard2.DEBUG_LOG");
      intent.setPackage(context.getPackageName());
      intent.putExtra("log_message", message);
      context.sendBroadcast(intent);
    }
    catch (Exception e)
    {
      // Silently fail - debug activity might not be running
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