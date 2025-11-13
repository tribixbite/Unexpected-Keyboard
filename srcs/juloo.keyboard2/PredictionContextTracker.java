package juloo.keyboard2;

import java.util.ArrayList;
import java.util.List;

/**
 * Tracks typing context for word predictions.
 *
 * Maintains state about:
 * - Current partial word being typed
 * - Previous words for context (n-gram support)
 * - Whether last input was a swipe or tap
 * - Last auto-inserted word (for smart deletion)
 * - Source of last committed text (for context-aware deletion)
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.342).
 */
public class PredictionContextTracker
{
  private static final String TAG = "PredictionContextTracker";

  // Maximum number of previous words to track for context
  private static final int MAX_CONTEXT_WORDS = 2;

  // Current partial word being typed (not yet committed to input)
  // Example: User types "hel" → _currentWord = "hel"
  private StringBuilder _currentWord;

  // Previous completed words for context (n-gram prediction)
  // Example: ["the", "quick"] for predicting next word
  // Limited to MAX_CONTEXT_WORDS (currently 2) for bigram support
  private List<String> _contextWords;

  // Track if last input was a swipe gesture (vs tap typing)
  // Used for context-aware deletion and prediction selection
  private boolean _wasLastInputSwipe;

  // Last word that was auto-inserted by prediction system
  // Used for smart deletion: if user taps suggestion, we can delete it cleanly
  private String _lastAutoInsertedWord;

  // Source of last committed text (swipe, typing, candidate selection, etc.)
  // Used for context-aware deletion behavior
  private PredictionSource _lastCommitSource;

  /**
   * Creates a new PredictionContextTracker with empty state.
   */
  public PredictionContextTracker()
  {
    _currentWord = new StringBuilder();
    _contextWords = new ArrayList<>();
    _wasLastInputSwipe = false;
    _lastAutoInsertedWord = null;
    _lastCommitSource = PredictionSource.UNKNOWN;
  }

  /**
   * Appends text to the current partial word.
   * Used when user types individual characters.
   *
   * @param text Text to append (usually single character)
   *
   * Example:
   * - appendToCurrentWord("h") → currentWord = "h"
   * - appendToCurrentWord("e") → currentWord = "he"
   * - appendToCurrentWord("l") → currentWord = "hel"
   */
  public void appendToCurrentWord(String text)
  {
    _currentWord.append(text);
  }

  /**
   * Gets the current partial word being typed.
   *
   * @return Current word string (never null, may be empty)
   */
  public String getCurrentWord()
  {
    return _currentWord.toString();
  }

  /**
   * Gets the length of current partial word.
   * Useful for checking if user is currently typing.
   *
   * @return Number of characters in current word
   */
  public int getCurrentWordLength()
  {
    return _currentWord.length();
  }

  /**
   * Clears the current partial word.
   * Called when word is completed or prediction is selected.
   */
  public void clearCurrentWord()
  {
    _currentWord.setLength(0);
  }

  /**
   * Commits a completed word and updates context.
   *
   * This method:
   * 1. Adds word to context history (for n-gram predictions)
   * 2. Maintains max context size (removes oldest if needed)
   * 3. Clears current partial word
   * 4. Tracks the source and auto-insert status
   *
   * @param word Completed word to commit
   * @param source Source of the word (swipe, typing, candidate, etc.)
   * @param autoInserted Whether this word was auto-inserted by prediction
   */
  public void commitWord(String word, PredictionSource source, boolean autoInserted)
  {
    // Update context for n-gram predictions
    _contextWords.add(word.toLowerCase());

    // Maintain max context size (oldest words removed first)
    while (_contextWords.size() > MAX_CONTEXT_WORDS)
    {
      _contextWords.remove(0);
    }

    // Clear current word (it's now committed)
    clearCurrentWord();

    // Track for smart deletion
    _lastCommitSource = source;
    _lastAutoInsertedWord = autoInserted ? word : null;
  }

  /**
   * Gets the context words for prediction.
   * Returns a copy to prevent external modification.
   *
   * @return List of previous words (max MAX_CONTEXT_WORDS)
   */
  public List<String> getContextWords()
  {
    return new ArrayList<>(_contextWords);
  }

  /**
   * Sets whether the last input was a swipe gesture.
   *
   * @param wasSwipe true if last input was swipe, false if tap typing
   */
  public void setWasLastInputSwipe(boolean wasSwipe)
  {
    _wasLastInputSwipe = wasSwipe;
  }

  /**
   * Checks if the last input was a swipe gesture.
   *
   * @return true if last input was swipe, false if tap typing
   */
  public boolean wasLastInputSwipe()
  {
    return _wasLastInputSwipe;
  }

  /**
   * Gets the last auto-inserted word.
   * Used for smart deletion: if user taps backspace after auto-insert,
   * we can delete the entire word + space.
   *
   * @return Last auto-inserted word, or null if none
   */
  public String getLastAutoInsertedWord()
  {
    return _lastAutoInsertedWord;
  }

  /**
   * Clears the last auto-inserted word tracking.
   * Called after word is deleted or new input begins.
   */
  public void clearLastAutoInsertedWord()
  {
    _lastAutoInsertedWord = null;
  }

  /**
   * Sets the last auto-inserted word.
   * Used in special cases where auto-insertion happens outside commitWord().
   *
   * @param word The word that was auto-inserted
   */
  public void setLastAutoInsertedWord(String word)
  {
    _lastAutoInsertedWord = word;
  }

  /**
   * Gets the source of the last committed text.
   *
   * @return PredictionSource enum value
   */
  public PredictionSource getLastCommitSource()
  {
    return _lastCommitSource;
  }

  /**
   * Sets the source of the last committed text.
   *
   * @param source PredictionSource enum value
   */
  public void setLastCommitSource(PredictionSource source)
  {
    _lastCommitSource = source;
  }

  /**
   * Clears all tracking state.
   * Useful for resetting state when switching input fields.
   */
  public void clearAll()
  {
    clearCurrentWord();
    _contextWords.clear();
    _wasLastInputSwipe = false;
    _lastAutoInsertedWord = null;
    _lastCommitSource = PredictionSource.UNKNOWN;
  }

  /**
   * Deletes the last character from the current word.
   * Used when user taps backspace during typing.
   * Does nothing if current word is empty.
   */
  public void deleteLastChar()
  {
    if (_currentWord.length() > 0)
    {
      _currentWord.deleteCharAt(_currentWord.length() - 1);
    }
  }

  /**
   * Gets a debug string showing current state.
   * Useful for logging and troubleshooting.
   *
   * @return Human-readable state description
   */
  public String getDebugState()
  {
    return String.format("PredictionContextTracker{currentWord='%s', contextWords=%s, wasSwipe=%b, lastAutoInsert='%s', lastSource=%s}",
      getCurrentWord(),
      _contextWords,
      _wasLastInputSwipe,
      _lastAutoInsertedWord,
      _lastCommitSource);
  }
}
