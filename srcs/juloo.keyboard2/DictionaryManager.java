package juloo.keyboard2;

import android.content.Context;
import android.content.SharedPreferences;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Manages word dictionaries for different languages and user custom words
 */
public class DictionaryManager
{
  private final Context _context;
  private final SharedPreferences _userDictPrefs;
  private final Map<String, WordPredictor> _predictors;
  private final Set<String> _userWords;
  private String _currentLanguage;
  private WordPredictor _currentPredictor;
  
  private static final String USER_DICT_PREFS = "user_dictionary";
  private static final String USER_WORDS_KEY = "user_words";
  
  public DictionaryManager(Context context)
  {
    _context = context;
    _userDictPrefs = context.getSharedPreferences(USER_DICT_PREFS, Context.MODE_PRIVATE);
    _predictors = new HashMap<>();
    _userWords = new HashSet<>();
    loadUserWords();
    setLanguage(Locale.getDefault().getLanguage());
  }
  
  /**
   * Set the active language for prediction.
   *
   * OPTIMIZATION v3 (perftodos3.md Todo 1): Uses async loading to prevent UI freezes.
   */
  public void setLanguage(String languageCode)
  {
    if (languageCode == null)
      languageCode = "en";

    _currentLanguage = languageCode;
    final String finalLanguageCode = languageCode; // Make effectively final for inner class

    // Get or create predictor for this language
    _currentPredictor = _predictors.get(languageCode);
    if (_currentPredictor == null)
    {
      final WordPredictor predictor = new WordPredictor();
      predictor.setContext(_context); // Enable disabled words filtering

      // CRITICAL: Use async loading to prevent UI freeze during language switching
      predictor.loadDictionaryAsync(_context, finalLanguageCode, new Runnable()
      {
        @Override
        public void run()
        {
          // This runs on the main thread when loading is complete
          // CRITICAL: Activate the UserDictionaryObserver now that dictionary is loaded
          predictor.startObservingDictionaryChanges();

          android.util.Log.i("DictionaryManager", "Dictionary loaded and observer activated for: " + finalLanguageCode);
        }
      });

      _currentPredictor = predictor;
      _predictors.put(languageCode, predictor);
    }
  }
  
  /**
   * Get word predictions for the given key sequence.
   *
   * Returns empty list if dictionary is still loading.
   */
  public List<String> getPredictions(String keySequence)
  {
    if (_currentPredictor == null)
      return new ArrayList<>();

    // OPTIMIZATION v3: Return empty list while dictionary is loading asynchronously
    if (_currentPredictor.isLoading())
      return new ArrayList<>();

    List<String> predictions = _currentPredictor.predictWords(keySequence);
    
    // Add user words that match
    String lowerSequence = keySequence.toLowerCase();
    for (String userWord : _userWords)
    {
      if (userWord.toLowerCase().startsWith(lowerSequence) && !predictions.contains(userWord))
      {
        predictions.add(0, userWord); // Add at beginning
        if (predictions.size() > 5)
          predictions.remove(predictions.size() - 1);
      }
    }
    
    return predictions;
  }
  
  /**
   * Add a word to the user dictionary
   */
  public void addUserWord(String word)
  {
    if (word == null || word.isEmpty())
      return;
      
    _userWords.add(word);
    saveUserWords();
  }
  
  /**
   * Remove a word from the user dictionary
   */
  public void removeUserWord(String word)
  {
    _userWords.remove(word);
    saveUserWords();
  }
  
  /**
   * Check if a word is in the user dictionary
   */
  public boolean isUserWord(String word)
  {
    return _userWords.contains(word);
  }
  
  /**
   * Clear the user dictionary
   */
  public void clearUserDictionary()
  {
    _userWords.clear();
    saveUserWords();
  }
  
  /**
   * Load user words from preferences
   */
  private void loadUserWords()
  {
    Set<String> words = _userDictPrefs.getStringSet(USER_WORDS_KEY, new HashSet<String>());
    _userWords.clear();
    _userWords.addAll(words);
  }
  
  /**
   * Save user words to preferences
   */
  private void saveUserWords()
  {
    SharedPreferences.Editor editor = _userDictPrefs.edit();
    editor.putStringSet(USER_WORDS_KEY, new HashSet<>(_userWords));
    editor.apply();
  }
  
  /**
   * Get the current language code
   */
  public String getCurrentLanguage()
  {
    return _currentLanguage;
  }

  /**
   * Check if the current predictor is loading.
   *
   * @return true if dictionary is loading asynchronously, false otherwise
   */
  public boolean isLoading()
  {
    return _currentPredictor != null && _currentPredictor.isLoading();
  }

  /**
   * Preload dictionaries for given languages.
   *
   * OPTIMIZATION v3 (perftodos3.md Todo 1): Uses async loading for all languages.
   */
  public void preloadLanguages(String[] languageCodes)
  {
    for (final String code : languageCodes)
    {
      if (!_predictors.containsKey(code))
      {
        final WordPredictor predictor = new WordPredictor();
        predictor.setContext(_context); // Enable disabled words filtering

        // CRITICAL: Use async loading to prevent UI freeze during preloading
        predictor.loadDictionaryAsync(_context, code, new Runnable()
        {
          @Override
          public void run()
          {
            // This runs on the main thread when loading is complete
            // CRITICAL: Activate the UserDictionaryObserver for preloaded language
            predictor.startObservingDictionaryChanges();

            android.util.Log.i("DictionaryManager", "Preloaded dictionary and activated observer for: " + code);
          }
        });

        _predictors.put(code, predictor);
      }
    }
  }

}