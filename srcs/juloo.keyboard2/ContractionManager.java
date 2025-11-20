package juloo.keyboard2;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * Manages contraction mappings for apostrophe insertion in predictions.
 *
 * Handles two types of contractions:
 * 1. Non-paired: apostrophe-free forms that map to single contractions
 *    Example: "dont" -> "don't", "cant" -> "can't"
 * 2. Paired: base words that have multiple contraction variants
 *    Example: "well" -> ["we'll", "well"], "id" -> ["I'd", "id"]
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.341).
 */
public class ContractionManager
{
  private static final String TAG = "ContractionManager";

  // Non-paired contractions: apostrophe-free form -> contraction with apostrophe
  // Example: "dont" -> "don't", "wholl" -> "who'll"
  private final Map<String, String> _nonPairedContractions;

  // Set of all known contractions (both non-paired and paired) for quick lookup
  // Used to identify contractions in predictions and prevent unwanted autocorrect
  private final Set<String> _knownContractions;

  private final Context _context;
  private final AssetManager _assetManager;

  /**
   * Creates a new ContractionManager.
   *
   * @param context Android context for accessing asset files
   */
  public ContractionManager(Context context)
  {
    _nonPairedContractions = new HashMap<>();
    _knownContractions = new HashSet<>();
    _context = context;
    _assetManager = context.getAssets();
  }

  /**
   * Loads contraction mappings from assets/dictionaries/.
   *
   * OPTIMIZATION v1 (perftodos2.md Todo 4): Uses binary format for faster loading.
   *
   * Strategy:
   * 1. Try binary format first (contractions.bin) - fastest
   * 2. Fall back to JSON if binary doesn't exist or fails
   *
   * Binary format is 3-5x faster than JSON parsing.
   *
   * Must be called before using isKnownContraction() or getNonPairedMapping().
   */
  public void loadMappings()
  {
    try
    {
      // Try binary format first (fastest)
      if (loadBinaryContractions())
      {
        if (BuildConfig.ENABLE_VERBOSE_LOGGING)
        {
          Log.d(TAG, "Loaded contractions from binary format");
        }
        return;
      }

      // Fall back to JSON format (slower, but always works)
      if (BuildConfig.ENABLE_VERBOSE_LOGGING)
      {
        Log.d(TAG, "Binary format not available, loading from JSON");
      }
      loadNonPairedContractions();
      loadPairedContractions();

      Log.d(TAG, String.format("Loaded %d non-paired contractions, %d total known contractions",
        _nonPairedContractions.size(), _knownContractions.size()));
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to load contraction mappings", e);
    }
  }

  /**
   * Load contractions from optimized binary format.
   *
   * OPTIMIZATION v1 (perftodos2.md Todo 4): Fast binary loading without JSON parsing.
   *
   * @return true if loaded successfully, false if binary doesn't exist or failed
   */
  private boolean loadBinaryContractions()
  {
    try
    {
      BinaryContractionLoader.ContractionData data =
        BinaryContractionLoader.loadContractions(_context, "dictionaries/contractions.bin");

      if (data == null)
      {
        return false;
      }

      _nonPairedContractions.putAll(data.nonPairedContractions);
      _knownContractions.addAll(data.knownContractions);

      return true;
    }
    catch (Exception e)
    {
      if (BuildConfig.ENABLE_VERBOSE_LOGGING)
      {
        Log.d(TAG, "Binary contractions not available: " + e.getMessage());
      }
      return false;
    }
  }

  /**
   * Checks if a word is a known contraction (has apostrophe).
   *
   * @param word Word to check (case-insensitive)
   * @return true if word is in the known contractions set
   *
   * Examples:
   * - isKnownContraction("don't") -> true
   * - isKnownContraction("we'll") -> true
   * - isKnownContraction("hello") -> false
   */
  public boolean isKnownContraction(String word)
  {
    return _knownContractions.contains(word.toLowerCase());
  }

  /**
   * Gets the contraction with apostrophe for an apostrophe-free form.
   *
   * Only works for non-paired contractions (where the apostrophe-free form
   * is not a valid English word).
   *
   * @param withoutApostrophe Apostrophe-free form (case-insensitive)
   * @return Contraction with apostrophe, or null if not found
   *
   * Examples:
   * - getNonPairedMapping("dont") -> "don't"
   * - getNonPairedMapping("wholl") -> "who'll"
   * - getNonPairedMapping("well") -> null (paired contraction)
   */
  public String getNonPairedMapping(String withoutApostrophe)
  {
    return _nonPairedContractions.get(withoutApostrophe.toLowerCase());
  }

  /**
   * Gets the number of non-paired contractions loaded.
   * Useful for testing and diagnostics.
   */
  public int getNonPairedCount()
  {
    return _nonPairedContractions.size();
  }

  /**
   * Gets the total number of known contractions (non-paired + paired).
   * Useful for testing and diagnostics.
   */
  public int getTotalKnownCount()
  {
    return _knownContractions.size();
  }

  /**
   * Generates a possessive form for a given word.
   *
   * OPTIMIZATION v5 (perftodos5.md): Rule-based possessive generation.
   * Instead of storing 1700+ possessive entries, generate them dynamically.
   *
   * Rules:
   * - Most words: add 's (cat -> cat's, dog -> dog's)
   * - Words ending in 's': add 's (Charles -> Charles's) [modern style]
   * - Never generate for pronouns/function words (handled by contractions)
   *
   * @param word Base word to make possessive
   * @return Possessive form (word + 's)
   *
   * Examples:
   * - generatePossessive("cat") -> "cat's"
   * - generatePossessive("dog") -> "dog's"
   * - generatePossessive("James") -> "James's"
   */
  public String generatePossessive(String word)
  {
    if (word == null || word.isEmpty())
    {
      return null;
    }

    String wordLower = word.toLowerCase();

    // Don't generate possessives for known contractions
    // (e.g., don't turn "don't" into "don't's")
    if (isKnownContraction(wordLower))
    {
      return null;
    }

    // Don't generate for function words/pronouns that have special contractions
    // These are already handled by the true contractions in the binary file
    Set<String> functionWords = new HashSet<>();
    functionWords.add("i");
    functionWords.add("you");
    functionWords.add("he");
    functionWords.add("she");
    functionWords.add("it");
    functionWords.add("we");
    functionWords.add("they");
    functionWords.add("who");
    functionWords.add("what");
    functionWords.add("that");
    functionWords.add("there");
    functionWords.add("here");
    functionWords.add("will");
    functionWords.add("would");
    functionWords.add("shall");
    functionWords.add("should");
    functionWords.add("can");
    functionWords.add("could");
    functionWords.add("may");
    functionWords.add("might");
    functionWords.add("must");
    functionWords.add("do");
    functionWords.add("does");
    functionWords.add("did");
    functionWords.add("is");
    functionWords.add("am");
    functionWords.add("are");
    functionWords.add("was");
    functionWords.add("were");
    functionWords.add("have");
    functionWords.add("has");
    functionWords.add("had");
    functionWords.add("let");

    if (functionWords.contains(wordLower))
    {
      return null;
    }

    // Generate possessive: word + 's
    // Modern style: even words ending in 's' get 's (James's, not James')
    return word + "'s";
  }

  /**
   * Checks if a word should have a possessive variant generated.
   *
   * OPTIMIZATION v5 (perftodos5.md): Determine if possessive makes sense.
   *
   * @param word Word to check
   * @return true if possessive should be generated
   */
  public boolean shouldGeneratePossessive(String word)
  {
    return generatePossessive(word) != null;
  }

  /**
   * Loads non-paired contractions from contractions_non_paired.json.
   *
   * Format: {"dont": "don't", "cant": "can't", ...}
   *
   * These are apostrophe-free forms that are NOT valid English words.
   * The neural network predicts "dont", we replace with "don't".
   */
  private void loadNonPairedContractions() throws Exception
  {
    InputStream inputStream = _assetManager.open("dictionaries/contractions_non_paired.json");
    String jsonString = readStream(inputStream);

    JSONObject jsonObj = new JSONObject(jsonString);
    Iterator<String> keys = jsonObj.keys();

    while (keys.hasNext())
    {
      String withoutApostrophe = keys.next();
      String withApostrophe = jsonObj.getString(withoutApostrophe);

      _nonPairedContractions.put(withoutApostrophe.toLowerCase(), withApostrophe.toLowerCase());
      _knownContractions.add(withApostrophe.toLowerCase());
    }

    Log.d(TAG, "Loaded " + _nonPairedContractions.size() + " non-paired contractions");
  }

  /**
   * Loads paired contractions from contraction_pairings.json.
   *
   * Format: {"well": [{"contraction": "we'll", "frequency": 243}], ...}
   *
   * These are base words that ARE valid English words but also have
   * contraction variants. Both forms should appear in predictions.
   * Example: "well" (adverb) and "we'll" (we will) are both valid.
   */
  private void loadPairedContractions() throws Exception
  {
    InputStream inputStream = _assetManager.open("dictionaries/contraction_pairings.json");
    String jsonString = readStream(inputStream);

    JSONObject jsonObj = new JSONObject(jsonString);
    Iterator<String> keys = jsonObj.keys();
    int pairedCount = 0;

    while (keys.hasNext())
    {
      String baseWord = keys.next();
      JSONArray contractions = jsonObj.getJSONArray(baseWord);

      for (int i = 0; i < contractions.length(); i++)
      {
        JSONObject contractionObj = contractions.getJSONObject(i);
        String contraction = contractionObj.getString("contraction");

        _knownContractions.add(contraction.toLowerCase());
        pairedCount++;
      }
    }

    Log.d(TAG, "Loaded " + pairedCount + " paired contractions");
  }

  /**
   * Reads an InputStream into a String.
   * Helper method for JSON file loading.
   */
  private String readStream(InputStream inputStream) throws Exception
  {
    BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
    StringBuilder builder = new StringBuilder();
    String line;

    while ((line = reader.readLine()) != null)
    {
      builder.append(line);
    }

    reader.close();
    return builder.toString();
  }
}
