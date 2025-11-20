package juloo.keyboard2;

import android.content.Context;
import android.content.res.Resources;
import android.util.Log;
import android.view.KeyEvent;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import java.util.ArrayList;
import java.util.List;
import juloo.keyboard2.ml.SwipeMLData;

/**
 * Handles suggestion selection, prediction display, and text completion logic.
 *
 * This class centralizes all logic related to:
 * - Suggestion bar updates and auto-insertion
 * - Prediction results from neural/typing engines
 * - Autocorrect for typing and swipe predictions
 * - Context tracking updates
 * - Text replacement and deletion (Termux-aware)
 * - Regular typing prediction updates
 *
 * Responsibilities:
 * - Display predictions in suggestion bar
 * - Auto-insert top predictions after swipe
 * - Handle manual suggestion selection
 * - Apply autocorrect to typed/predicted words
 * - Manage word deletion and replacement
 * - Update context tracker with completed words
 * - Handle Termux mode special cases
 *
 * NOT included (remains in Keyboard2):
 * - InputMethodService lifecycle methods
 * - View creation and inflation
 * - Configuration management
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.361).
 */
public class SuggestionHandler
{
  private static final String TAG = "SuggestionHandler";

  private final Context _context;
  private Config _config;

  // Dependencies
  private final PredictionContextTracker _contextTracker;
  private final PredictionCoordinator _predictionCoordinator;
  private final ContractionManager _contractionManager;
  private final KeyEventHandler _keyeventhandler;
  private SuggestionBar _suggestionBar; // Non-final - updated after creation

  // Debug mode for logging
  private boolean _debugMode = false;
  private DebugLogger _debugLogger; // Interface for sending debug logs

  /**
   * Interface for sending debug logs to SwipeDebugActivity.
   * Implemented by Keyboard2 to bridge to its sendDebugLog method.
   */
  public interface DebugLogger
  {
    void sendDebugLog(String message);
  }

  /**
   * Creates a new SuggestionHandler.
   *
   * @param context Android context for resource access
   * @param config Configuration instance
   * @param contextTracker Tracks current word and context
   * @param predictionCoordinator Manages prediction engines
   * @param contractionManager Handles contraction mappings
   * @param keyeventhandler Handles key events for Termux mode
   */
  public SuggestionHandler(Context context,
                          Config config,
                          PredictionContextTracker contextTracker,
                          PredictionCoordinator predictionCoordinator,
                          ContractionManager contractionManager,
                          KeyEventHandler keyeventhandler)
  {
    _context = context;
    _config = config;
    _contextTracker = contextTracker;
    _predictionCoordinator = predictionCoordinator;
    _contractionManager = contractionManager;
    _keyeventhandler = keyeventhandler;
  }

  /**
   * Updates configuration.
   *
   * @param newConfig Updated configuration
   */
  public void setConfig(Config newConfig)
  {
    _config = newConfig;
  }

  /**
   * Sets the suggestion bar reference.
   *
   * @param suggestionBar Suggestion bar for displaying predictions
   */
  public void setSuggestionBar(SuggestionBar suggestionBar)
  {
    _suggestionBar = suggestionBar;
  }

  /**
   * Sets debug mode and logger.
   *
   * @param enabled Whether debug mode is enabled
   * @param logger Debug logger implementation
   */
  public void setDebugMode(boolean enabled, DebugLogger logger)
  {
    _debugMode = enabled;
    _debugLogger = logger;
  }

  /**
   * Sends a debug log message if debug mode is enabled.
   */
  private void sendDebugLog(String message)
  {
    if (_debugMode && _debugLogger != null)
    {
      _debugLogger.sendDebugLog(message);
    }
  }

  /**
   * Handle prediction results from async prediction handler.
   * Displays predictions in suggestion bar and auto-inserts top prediction.
   *
   * @param predictions List of predicted words
   * @param scores Confidence scores for predictions
   * @param ic InputConnection for text manipulation
   * @param editorInfo Editor info for context
   * @param resources Resources for metrics
   */
  public void handlePredictionResults(List<String> predictions,
                                     List<Integer> scores,
                                     InputConnection ic,
                                     EditorInfo editorInfo,
                                     Resources resources)
  {
    // DEBUG: Log predictions received
    sendDebugLog(String.format("Predictions received: %d\n", predictions != null ? predictions.size() : 0));
    if (predictions != null && !predictions.isEmpty())
    {
      for (int i = 0; i < Math.min(5, predictions.size()); i++)
      {
        int score = (scores != null && i < scores.size()) ? scores.get(i) : 0;
        sendDebugLog(String.format("  [%d] \"%s\" (score: %d)\n", i+1, predictions.get(i), score));
      }
    }

    if (predictions.isEmpty())
    {
      sendDebugLog("No predictions - clearing suggestions\n");
      if (_suggestionBar != null)
      {
        _suggestionBar.clearSuggestions();
      }
      return;
    }

    // OPTIMIZATION v5 (perftodos5.md): Augment predictions with possessives
    // Generate possessive forms for top predictions and add them to the list
    List<String> augmentedPredictions = new ArrayList<>(predictions);
    List<Integer> augmentedScores = new ArrayList<>(scores != null ? scores : new ArrayList<>());
    augmentPredictionsWithPossessives(augmentedPredictions, augmentedScores);

    // Update suggestion bar (scores are already integers from neural system)
    if (_suggestionBar != null)
    {
      _suggestionBar.setShowDebugScores(_config.swipe_show_debug_scores);
      _suggestionBar.setSuggestionsWithScores(augmentedPredictions, augmentedScores);

      // Auto-insert top (highest scoring) prediction immediately after swipe completes
      // This enables rapid consecutive swiping without manual taps
      String topPrediction = _suggestionBar.getTopSuggestion();
      if (topPrediction != null && !topPrediction.isEmpty())
      {
        // If manual typing in progress, add space after it (don't re-commit the text!)
        if (_contextTracker.getCurrentWordLength() > 0 && ic != null)
        {
          sendDebugLog(String.format("Manual typing in progress before swipe: \"%s\"\n", _contextTracker.getCurrentWord()));

          // IMPORTANT: Characters from manual typing are already committed via KeyEventHandler.send_text()
          // _currentWord is just a tracking buffer - the text is already in the editor!
          // We only need to add a space after the manually typed word and clear the tracking buffer
          ic.commitText(" ", 1);
          _contextTracker.clearCurrentWord();

          // Clear any previous auto-inserted word tracking since user was manually typing
          _contextTracker.clearLastAutoInsertedWord();
          _contextTracker.setLastCommitSource(PredictionSource.USER_TYPED_TAP);
        }

        // DEBUG: Log auto-insertion
        sendDebugLog(String.format("Auto-inserting top prediction: \"%s\"\n", topPrediction));

        // CRITICAL: Clear auto-inserted tracking BEFORE calling onSuggestionSelected
        // This prevents the deletion logic from removing the previous auto-inserted word
        // For consecutive swipes, we want to APPEND words, not replace them
        _contextTracker.clearLastAutoInsertedWord();
        _contextTracker.setLastCommitSource(PredictionSource.UNKNOWN); // Temporarily clear

        // onSuggestionSelected handles spacing logic (no space if first text, space otherwise)
        onSuggestionSelected(topPrediction, ic, editorInfo, resources);

        // NOW track this as auto-inserted so tapping another suggestion will replace ONLY this word
        // CRITICAL: Strip "raw:" prefix BEFORE storing (v1.33.7: fixed regex to match actual prefix format)
        String cleanPrediction = topPrediction.replaceAll("^raw:", "");
        _contextTracker.setLastAutoInsertedWord(cleanPrediction);
        _contextTracker.setLastCommitSource(PredictionSource.NEURAL_SWIPE);

        // CRITICAL: Re-display suggestions after auto-insertion
        // User can still tap a different prediction if the auto-inserted one was wrong
        _suggestionBar.setSuggestionsWithScores(predictions, scores);

        sendDebugLog("Suggestions re-displayed for correction\n");
      }
    }
    sendDebugLog("========== SWIPE COMPLETE ==========\n\n");
  }

  /**
   * Called when user selects a suggestion from the suggestion bar.
   * Handles autocorrect, text replacement, and context updates.
   *
   * @param word Selected word
   * @param ic InputConnection for text manipulation
   * @param editorInfo Editor info for app detection
   * @param resources Resources for metrics
   */
  public void onSuggestionSelected(String word,
                                  InputConnection ic,
                                  EditorInfo editorInfo,
                                  Resources resources)
  {
    // Null/empty check
    if (word == null || word.trim().isEmpty())
    {
      return;
    }

    // Check if this is a raw prediction (user explicitly selected neural network output)
    // Raw predictions should skip autocorrect
    boolean isRawPrediction = word.startsWith("raw:");

    // Strip "raw:" prefix before processing (v1.33.7: fixed regex to match actual prefix format)
    // Prefix format: "raw:word" not " [raw:0.08]"
    word = word.replaceAll("^raw:", "");

    // Check if this is a known contraction (already has apostrophes from displayText)
    // If it is, skip autocorrect to prevent fuzzy matching to wrong words
    boolean isKnownContraction = _contractionManager.isKnownContraction(word);

    // Skip autocorrect for:
    // 1. Known contractions (prevent fuzzy matching)
    // 2. Raw predictions (user explicitly selected this neural output)
    if (isKnownContraction || isRawPrediction)
    {
      if (isKnownContraction)
      {
        Log.d(TAG, String.format("KNOWN CONTRACTION: \"%s\" - skipping autocorrect", word));
      }
      if (isRawPrediction)
      {
        Log.d(TAG, String.format("RAW PREDICTION: \"%s\" - skipping autocorrect", word));
      }
    }
    else
    {
      // v1.33.7: Final autocorrect - second chance autocorrect after beam search
      // Applies when user selects/auto-inserts a prediction (even if beam autocorrect was OFF)
      // Useful for correcting vocabulary misses
      // SKIP for known contractions and raw predictions
      if (_config.swipe_final_autocorrect_enabled && _predictionCoordinator.getWordPredictor() != null)
      {
        String correctedWord = _predictionCoordinator.getWordPredictor().autoCorrect(word);

        // If autocorrect found a better match, use it
        if (!correctedWord.equals(word))
        {
          Log.d(TAG, String.format("FINAL AUTOCORRECT: \"%s\" → \"%s\"", word, correctedWord));
          word = correctedWord;
        }
      }
    }

    // Record user selection for adaptation learning
    if (_predictionCoordinator.getAdaptationManager() != null)
    {
      _predictionCoordinator.getAdaptationManager().recordSelection(word.trim());
    }

    // CRITICAL: Save swipe flag before resetting for use in spacing logic below
    boolean isSwipeAutoInsert = _contextTracker.wasLastInputSwipe();

    // Store ML data if this was a swipe prediction selection
    // Note: ML data collection is handled by InputCoordinator, not here
    // This handler only deals with suggestion selection logic

    // Reset swipe tracking
    _contextTracker.setWasLastInputSwipe(false);

    if (ic != null)
    {
      try
      {
        // Detect if we're in Termux for special handling
        boolean inTermuxApp = false;
        try
        {
          if (editorInfo != null && editorInfo.packageName != null)
          {
            inTermuxApp = editorInfo.packageName.equals("com.termux");
          }
        }
        catch (Exception e)
        {
          // Fallback: assume not Termux
        }

        // IMPORTANT: _currentWord tracks typed characters, but they're already committed to input!
        // When typing normally (not swipe), each character is committed immediately via KeyEventHandler
        // So _currentWord is just for tracking - the text is already in the editor
        // We should NOT delete _currentWord characters here because:
        // 1. They're already committed and visible
        // 2. Swipe gesture detection happens AFTER typing completes
        // 3. User expects swipe to ADD a word, not delete what they typed
        //
        // Example bug scenario:
        // - User types "i" (committed to editor, _currentWord="i")
        // - User swipes "think" (without space after "i")
        // - Old code: deletes "i", adds " think " → result: " think " (lost the "i"!)
        // - New code: keeps "i", adds " think " → result: "i think " (correct!)
        //
        // The ONLY time we should delete is when replacing an auto-inserted prediction
        // (handled below via _lastAutoInsertedWord tracking)

        // CRITICAL: If we just auto-inserted a word from neural swipe, delete it for replacement
        // This allows user to tap a different prediction instead of appending
        // Only delete if the last commit was from neural swipe (not from other sources)
        if (_contextTracker.getLastAutoInsertedWord() != null && !_contextTracker.getLastAutoInsertedWord().isEmpty() &&
            _contextTracker.getLastCommitSource() == PredictionSource.NEURAL_SWIPE)
        {
          Log.d(TAG, "REPLACE: Deleting auto-inserted word: '" + _contextTracker.getLastAutoInsertedWord() + "'");

          int deleteCount = _contextTracker.getLastAutoInsertedWord().length() + 1; // Word + trailing space
          boolean deletedLeadingSpace = false;

          if (inTermuxApp)
          {
            // TERMUX: Use backspace key events instead of InputConnection methods
            // Termux doesn't support deleteSurroundingText properly
            Log.d(TAG, "TERMUX: Using backspace key events to delete " + deleteCount + " chars");

            // Check if there's a leading space to delete
            CharSequence textBefore = ic.getTextBeforeCursor(1, 0);
            if (textBefore != null && textBefore.length() > 0 && textBefore.charAt(0) == ' ')
            {
              deleteCount++; // Include leading space
              deletedLeadingSpace = true;
            }

            // Send backspace key events
            if (_keyeventhandler != null)
            {
              for (int i = 0; i < deleteCount; i++)
              {
                _keyeventhandler.send_key_down_up(KeyEvent.KEYCODE_DEL, 0);
              }
            }
          }
          else
          {
            // NORMAL APPS: Use InputConnection methods
            CharSequence debugBefore = ic.getTextBeforeCursor(50, 0);
            Log.d(TAG, "REPLACE: Text before cursor (50 chars): '" + debugBefore + "'");
            Log.d(TAG, "REPLACE: Delete count = " + deleteCount);

            // Delete the auto-inserted word and its space
            ic.deleteSurroundingText(deleteCount, 0);

            CharSequence debugAfter = ic.getTextBeforeCursor(50, 0);
            Log.d(TAG, "REPLACE: After deleting word, text before cursor: '" + debugAfter + "'");

            // Also need to check if there was a space added before it
            CharSequence textBefore = ic.getTextBeforeCursor(1, 0);
            Log.d(TAG, "REPLACE: Checking for leading space, got: '" + textBefore + "'");
            if (textBefore != null && textBefore.length() > 0 && textBefore.charAt(0) == ' ')
            {
              Log.d(TAG, "REPLACE: Deleting leading space");
              // Delete the leading space too
              ic.deleteSurroundingText(1, 0);

              CharSequence debugFinal = ic.getTextBeforeCursor(50, 0);
              Log.d(TAG, "REPLACE: After deleting leading space: '" + debugFinal + "'");
            }
          }

          // Clear the tracking variables
          _contextTracker.clearLastAutoInsertedWord();
          _contextTracker.setLastCommitSource(PredictionSource.UNKNOWN);
        }
        // ALSO: If user is selecting a prediction during regular typing, delete the partial word
        // This handles typing "hel" then selecting "hello" - we need to delete "hel" first
        else if (_contextTracker.getCurrentWordLength() > 0 && !isSwipeAutoInsert)
        {
          Log.d(TAG, "TYPING PREDICTION: Deleting partial word: '" + _contextTracker.getCurrentWord() + "'");

          if (inTermuxApp)
          {
            // TERMUX: Use backspace key events
            Log.d(TAG, "TERMUX: Using backspace key events to delete " + _contextTracker.getCurrentWordLength() + " chars");
            if (_keyeventhandler != null)
            {
              for (int i = 0; i < _contextTracker.getCurrentWordLength(); i++)
              {
                _keyeventhandler.send_key_down_up(KeyEvent.KEYCODE_DEL, 0);
              }
            }
          }
          else
          {
            // NORMAL APPS: Use InputConnection
            ic.deleteSurroundingText(_contextTracker.getCurrentWordLength(), 0);

            CharSequence debugAfter = ic.getTextBeforeCursor(50, 0);
            Log.d(TAG, "TYPING PREDICTION: After deleting partial, text before cursor: '" + debugAfter + "'");
          }
        }

        // Add space before word if previous character isn't whitespace
        boolean needsSpaceBefore = false;
        try
        {
          CharSequence textBefore = ic.getTextBeforeCursor(1, 0);
          if (textBefore != null && textBefore.length() > 0)
          {
            char prevChar = textBefore.charAt(0);
            // Add space if previous char is not whitespace and not punctuation start
            needsSpaceBefore = !Character.isWhitespace(prevChar) && prevChar != '(' && prevChar != '[' && prevChar != '{';
          }
        }
        catch (Exception e)
        {
          // If getTextBeforeCursor fails, assume we don't need space before
          needsSpaceBefore = false;
        }

        // Commit the selected word - use Termux mode if enabled
        String textToInsert;
        if (_config.termux_mode_enabled && !isSwipeAutoInsert)
        {
          // Termux mode (non-swipe): Insert word without automatic space for better terminal compatibility
          textToInsert = needsSpaceBefore ? " " + word : word;
          Log.d(TAG, "TERMUX MODE (non-swipe): textToInsert = '" + textToInsert + "'");
        }
        else
        {
          // Normal mode OR swipe in Termux: Insert word with space after (and before if needed)
          // For swipe typing, we always add trailing spaces even in Termux mode for better UX
          textToInsert = needsSpaceBefore ? " " + word + " " : word + " ";
          Log.d(TAG, "NORMAL/SWIPE MODE: textToInsert = '" + textToInsert + "' (needsSpaceBefore=" + needsSpaceBefore + ", isSwipe=" + isSwipeAutoInsert + ")");
        }

        Log.d(TAG, "Committing text: '" + textToInsert + "' (length=" + textToInsert.length() + ")");
        ic.commitText(textToInsert, 1);

        // Track that this commit was from candidate selection (manual tap)
        // Note: Auto-insertions set this separately to NEURAL_SWIPE
        if (_contextTracker.getLastCommitSource() != PredictionSource.NEURAL_SWIPE)
        {
          _contextTracker.setLastCommitSource(PredictionSource.CANDIDATE_SELECTION);
        }
      }
      catch (Exception e)
      {
        Log.e(TAG, "Error in onSuggestionSelected", e);
      }

      // Update context with the selected word
      updateContext(word);

      // Clear current word
      // NOTE: Don't clear suggestions here - they're re-displayed after auto-insertion
      _contextTracker.clearCurrentWord();
    }
  }

  /**
   * Update context with a completed word.
   *
   * NOTE: This is a legacy helper method. New code should use
   * _contextTracker.commitWord() directly with appropriate PredictionSource.
   *
   * @param word Completed word to add to context
   */
  public void updateContext(String word)
  {
    if (word == null || word.isEmpty())
      return;

    // Use the current source from tracker, or UNKNOWN if not set
    PredictionSource source = _contextTracker.getLastCommitSource();
    if (source == null)
    {
      source = PredictionSource.UNKNOWN;
    }

    // Commit word to context tracker (not auto-inserted since this is manual update)
    _contextTracker.commitWord(word, source, false);

    // Add word to WordPredictor for language detection
    if (_predictionCoordinator.getWordPredictor() != null)
    {
      _predictionCoordinator.getWordPredictor().addWordToContext(word);
    }
  }

  /**
   * Handle regular typing predictions (non-swipe).
   * Updates predictions as user types each character.
   *
   * @param text Text being typed
   * @param ic InputConnection for text manipulation
   * @param editorInfo Editor info for app detection
   */
  public void handleRegularTyping(String text, InputConnection ic, EditorInfo editorInfo)
  {
    if (!_config.word_prediction_enabled || _predictionCoordinator.getWordPredictor() == null || _suggestionBar == null)
    {
      return;
    }

    // Track current word being typed
    if (text.length() == 1 && Character.isLetter(text.charAt(0)))
    {
      _contextTracker.appendToCurrentWord(text);
      updatePredictionsForCurrentWord();
    }
    else if (text.length() == 1 && !Character.isLetter(text.charAt(0)))
    {
      // Any non-letter character - update context and reset current word

      // If we had a word being typed, add it to context before clearing
      if (_contextTracker.getCurrentWordLength() > 0)
      {
        String completedWord = _contextTracker.getCurrentWord();

        // Auto-correct the typed word if feature is enabled
        // DISABLED in Termux app due to erratic behavior with terminal input
        boolean inTermuxApp = false;
        try
        {
          if (editorInfo != null && editorInfo.packageName != null)
          {
            inTermuxApp = editorInfo.packageName.equals("com.termux");
          }
        }
        catch (Exception e)
        {
          // Fallback: assume not Termux if detection fails
        }

        if (_config.autocorrect_enabled && _predictionCoordinator.getWordPredictor() != null && text.equals(" ") && !inTermuxApp)
        {
          String correctedWord = _predictionCoordinator.getWordPredictor().autoCorrect(completedWord);

          // If correction was made, replace the typed word
          if (!correctedWord.equals(completedWord))
          {
            if (ic != null)
            {
              // At this point:
              // - The typed word "thid" has been committed via KeyEventHandler.send_text()
              // - The space " " has ALSO been committed via handle_text_typed(" ")
              // - Editor contains "thid "
              // - We need to delete both the word AND the space, then insert corrected word + space

              // Delete the typed word + space (already committed)
              ic.deleteSurroundingText(completedWord.length() + 1, 0);

              // Insert the corrected word WITH trailing space (normal apps only)
              ic.commitText(correctedWord + " ", 1);

              // Update context with corrected word
              updateContext(correctedWord);

              // Clear current word
              _contextTracker.clearCurrentWord();

              // Show corrected word as first suggestion for easy undo
              if (_suggestionBar != null)
              {
                List<String> undoSuggestions = new ArrayList<>();
                undoSuggestions.add(completedWord); // Original word first for undo
                undoSuggestions.add(correctedWord); // Corrected word second
                List<Integer> undoScores = new ArrayList<>();
                undoScores.add(0);
                undoScores.add(0);
                _suggestionBar.setSuggestionsWithScores(undoSuggestions, undoScores);
              }

              // Reset prediction state
              if (_predictionCoordinator.getWordPredictor() != null)
              {
                _predictionCoordinator.getWordPredictor().reset();
              }

              return; // Skip normal text processing - we've handled everything
            }
          }
        }

        updateContext(completedWord);
      }

      // Reset current word
      _contextTracker.clearCurrentWord();
      if (_predictionCoordinator.getWordPredictor() != null)
      {
        _predictionCoordinator.getWordPredictor().reset();
      }
      if (_suggestionBar != null)
      {
        _suggestionBar.clearSuggestions();
      }
    }
    else if (text.length() > 1)
    {
      // Multi-character input (paste, etc) - reset
      _contextTracker.clearCurrentWord();
      if (_predictionCoordinator.getWordPredictor() != null)
      {
        _predictionCoordinator.getWordPredictor().reset();
      }
      if (_suggestionBar != null)
      {
        _suggestionBar.clearSuggestions();
      }
    }
  }

  /**
   * Handle backspace for prediction tracking.
   * Updates predictions as user deletes characters.
   */
  public void handleBackspace()
  {
    if (_contextTracker.getCurrentWordLength() > 0)
    {
      _contextTracker.deleteLastChar();
      if (_contextTracker.getCurrentWordLength() > 0)
      {
        updatePredictionsForCurrentWord();
      }
      else if (_suggestionBar != null)
      {
        _suggestionBar.clearSuggestions();
      }
    }
  }

  /**
   * Update predictions based on current partial word.
   */
  private void updatePredictionsForCurrentWord()
  {
    if (_contextTracker.getCurrentWordLength() > 0)
    {
      String partial = _contextTracker.getCurrentWord();

      // Use contextual prediction
      WordPredictor.PredictionResult result = _predictionCoordinator.getWordPredictor().predictWordsWithContext(partial, _contextTracker.getContextWords());

      if (!result.words.isEmpty() && _suggestionBar != null)
      {
        _suggestionBar.setShowDebugScores(_config.swipe_show_debug_scores);
        _suggestionBar.setSuggestionsWithScores(result.words, result.scores);
      }
    }
  }

  /**
   * Smart delete last word - deletes the last auto-inserted word or last typed word.
   * Handles edge cases to avoid deleting too much text.
   *
   * @param ic InputConnection for text manipulation
   * @param editorInfo Editor info for app detection
   */
  public void handleDeleteLastWord(InputConnection ic, EditorInfo editorInfo)
  {
    if (ic == null)
      return;

    // Check if we're in Termux - if so, use Ctrl+Backspace fallback
    boolean inTermux = false;
    try
    {
      if (editorInfo != null && editorInfo.packageName != null)
      {
        inTermux = editorInfo.packageName.equals("com.termux");
      }
    }
    catch (Exception e)
    {
      Log.e(TAG, "DELETE_LAST_WORD: Error detecting Termux", e);
    }

    // For Termux, use Ctrl+W key event which Termux handles correctly
    // Termux doesn't support InputConnection methods, but processes terminal control sequences
    if (inTermux)
    {
      Log.d(TAG, "DELETE_LAST_WORD: Using Ctrl+W (^W) for Termux");
      // Send Ctrl+W which is the standard terminal "delete word backward" sequence
      if (_keyeventhandler != null)
      {
        _keyeventhandler.send_key_down_up(KeyEvent.KEYCODE_W, KeyEvent.META_CTRL_ON | KeyEvent.META_CTRL_LEFT_ON);
      }
      // Clear tracking
      _contextTracker.clearLastAutoInsertedWord();
      _contextTracker.setLastCommitSource(PredictionSource.UNKNOWN);
      return;
    }

    // First, try to delete the last auto-inserted word if it exists
    if (_contextTracker.getLastAutoInsertedWord() != null && !_contextTracker.getLastAutoInsertedWord().isEmpty())
    {
      Log.d(TAG, "DELETE_LAST_WORD: Deleting auto-inserted word: '" + _contextTracker.getLastAutoInsertedWord() + "'");

      // Get text before cursor to verify
      CharSequence textBefore = ic.getTextBeforeCursor(100, 0);
      if (textBefore != null)
      {
        String beforeStr = textBefore.toString();

        // Check if the last auto-inserted word is actually at the end
        // Account for trailing space that swipe words have
        boolean hasTrailingSpace = beforeStr.endsWith(" ");
        String lastWord = hasTrailingSpace ? beforeStr.substring(0, beforeStr.length() - 1).trim() : beforeStr.trim();

        // Find last word in the text
        int lastSpaceIdx = lastWord.lastIndexOf(' ');
        String actualLastWord = lastSpaceIdx >= 0 ? lastWord.substring(lastSpaceIdx + 1) : lastWord;

        // Verify this matches our tracked word (case-insensitive to be safe)
        if (actualLastWord.equalsIgnoreCase(_contextTracker.getLastAutoInsertedWord()))
        {
          // Delete the word + trailing space if present
          int deleteCount = _contextTracker.getLastAutoInsertedWord().length();
          if (hasTrailingSpace)
            deleteCount += 1;

          ic.deleteSurroundingText(deleteCount, 0);
          Log.d(TAG, "DELETE_LAST_WORD: Deleted " + deleteCount + " characters");

          // Clear tracking
          _contextTracker.clearLastAutoInsertedWord();
          _contextTracker.setLastCommitSource(PredictionSource.UNKNOWN);
          return;
        }
      }

      // If verification failed, fall through to delete last word generically
      Log.d(TAG, "DELETE_LAST_WORD: Auto-inserted word verification failed, using generic delete");
    }

    // Fallback: Delete the last word before cursor (generic approach)
    CharSequence textBefore = ic.getTextBeforeCursor(100, 0);
    if (textBefore == null || textBefore.length() == 0)
    {
      Log.d(TAG, "DELETE_LAST_WORD: No text before cursor");
      return;
    }

    String beforeStr = textBefore.toString();
    int cursorPos = beforeStr.length();

    // Skip trailing whitespace
    while (cursorPos > 0 && Character.isWhitespace(beforeStr.charAt(cursorPos - 1)))
      cursorPos--;

    if (cursorPos == 0)
    {
      Log.d(TAG, "DELETE_LAST_WORD: Only whitespace before cursor");
      return;
    }

    // Find the start of the last word
    int wordStart = cursorPos;
    while (wordStart > 0 && !Character.isWhitespace(beforeStr.charAt(wordStart - 1)))
      wordStart--;

    // Calculate delete count (word + any trailing spaces we skipped)
    int deleteCount = beforeStr.length() - wordStart;

    // Safety check: don't delete more than 50 characters at once
    if (deleteCount > 50)
    {
      Log.d(TAG, "DELETE_LAST_WORD: Refusing to delete " + deleteCount + " characters (safety limit)");
      deleteCount = 50;
    }

    Log.d(TAG, "DELETE_LAST_WORD: Deleting last word (generic), count=" + deleteCount);
    ic.deleteSurroundingText(deleteCount, 0);

    // Clear tracking
    _contextTracker.clearLastAutoInsertedWord();
    _contextTracker.setLastCommitSource(PredictionSource.UNKNOWN);
  }

  /**
   * Augment predictions with possessive forms.
   *
   * OPTIMIZATION v5 (perftodos5.md): Generate possessives dynamically instead of storing 1700+ entries.
   * For each top prediction (limit to first 3-5), generate possessive form if applicable.
   *
   * @param predictions List of predictions to augment (modified in-place)
   * @param scores List of scores corresponding to predictions (modified in-place)
   */
  private void augmentPredictionsWithPossessives(List<String> predictions, List<Integer> scores)
  {
    if (predictions == null || predictions.isEmpty())
    {
      return;
    }

    // Generate possessives for top 3 predictions only (avoid clutter)
    int limit = Math.min(3, predictions.size());
    List<String> possessivesToAdd = new ArrayList<>();
    List<Integer> possessiveScores = new ArrayList<>();

    for (int i = 0; i < limit; i++)
    {
      String word = predictions.get(i);
      String possessive = _contractionManager.generatePossessive(word);

      if (possessive != null)
      {
        // Don't add if possessive already exists in predictions
        boolean alreadyExists = false;
        for (String pred : predictions)
        {
          if (pred.equalsIgnoreCase(possessive))
          {
            alreadyExists = true;
            break;
          }
        }

        if (!alreadyExists)
        {
          possessivesToAdd.add(possessive);
          // Slightly lower score than base word (base word is more common)
          int baseScore = (i < scores.size()) ? scores.get(i) : 128;
          possessiveScores.add(baseScore - 10); // 10 points lower than base
        }
      }
    }

    // Add possessives to the end of predictions list
    if (!possessivesToAdd.isEmpty())
    {
      predictions.addAll(possessivesToAdd);
      scores.addAll(possessiveScores);

      if (BuildConfig.ENABLE_VERBOSE_LOGGING)
      {
        Log.d(TAG, String.format("Added %d possessive forms to predictions", possessivesToAdd.size()));
      }
    }
  }
}
