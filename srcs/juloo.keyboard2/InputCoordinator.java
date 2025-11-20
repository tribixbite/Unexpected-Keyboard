package juloo.keyboard2;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Resources;
import android.view.KeyEvent;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import juloo.keyboard2.ml.SwipeMLData;

/**
 * Coordinates all text input operations including typing, backspace, word deletion,
 * swipe typing, and suggestion selection.
 *
 * This class centralizes input handling logic that was previously in Keyboard2.java.
 * It manages:
 * - Regular typing with word predictions
 * - Autocorrection during typing
 * - Backspace and smart word deletion
 * - Swipe typing gesture recognition and prediction
 * - Suggestion selection and text insertion
 * - ML data collection for swipe training
 *
 * Dependencies:
 * - PredictionContextTracker: Tracks typing context
 * - PredictionCoordinator: Manages prediction engines
 * - ContractionManager: Handles contraction mappings
 * - SuggestionBar: Displays predictions to user
 * - Keyboard2View: For keyboard dimensions
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.350).
 */
public class InputCoordinator
{
  private static final String TAG = "InputCoordinator";

  private final Context _context;
  private Config _config;

  // Dependencies
  private final PredictionContextTracker _contextTracker;
  private final PredictionCoordinator _predictionCoordinator;
  private final ContractionManager _contractionManager;
  private SuggestionBar _suggestionBar; // Non-final - updated in onStartInputView
  private final Keyboard2View _keyboardView;
  private final KeyEventHandler _keyeventhandler;

  // Swipe ML data collection
  private SwipeMLData _currentSwipeData;

  /**
   * Creates a new InputCoordinator.
   *
   * @param context Android context
   * @param config Configuration instance
   * @param contextTracker Prediction context tracker
   * @param predictionCoordinator Prediction engine coordinator
   * @param contractionManager Contraction mappings manager
   * @param suggestionBar Suggestion bar for displaying predictions
   * @param keyboardView Keyboard view for dimensions
   * @param keyeventhandler Key event handler for sending special key events
   */
  public InputCoordinator(
    Context context,
    Config config,
    PredictionContextTracker contextTracker,
    PredictionCoordinator predictionCoordinator,
    ContractionManager contractionManager,
    SuggestionBar suggestionBar,
    Keyboard2View keyboardView,
    KeyEventHandler keyeventhandler)
  {
    _context = context;
    _config = config;
    _contextTracker = contextTracker;
    _predictionCoordinator = predictionCoordinator;
    _contractionManager = contractionManager;
    _suggestionBar = suggestionBar;
    _keyboardView = keyboardView;
    _keyeventhandler = keyeventhandler;
    _currentSwipeData = null;
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
   * Updates suggestion bar reference.
   * Called when suggestion bar is created in onStartInputView.
   *
   * @param suggestionBar Suggestion bar instance
   */
  public void setSuggestionBar(SuggestionBar suggestionBar)
  {
    _suggestionBar = suggestionBar;
  }

  /**
   * Resets swipe data tracking.
   * Called when starting new input or switching apps.
   */
  public void resetSwipeData()
  {
    _currentSwipeData = null;
  }

  /**
   * Gets current swipe ML data for storage.
   * @return Current swipe data or null if no swipe in progress
   */
  public SwipeMLData getCurrentSwipeData()
  {
    return _currentSwipeData;
  }

  /**
   * Handle prediction results from async swipe typing prediction.
   * Called when neural network predictions are ready.
   */
  public void handlePredictionResults(List<String> predictions, List<Integer> scores, InputConnection ic, EditorInfo editorInfo, Resources resources)
  {
    // TODO: Re-enable debug logging
    // sendDebugLog(String.format("Predictions received: %d\n", predictions != null ? predictions.size() : 0));

    if (predictions == null || predictions.isEmpty())
    {
      if (_suggestionBar != null)
      {
        _suggestionBar.clearSuggestions();
      }
      return;
    }

    // Update suggestion bar
    if (_suggestionBar != null)
    {
      _suggestionBar.setShowDebugScores(_config.swipe_show_debug_scores);
      _suggestionBar.setSuggestionsWithScores(predictions, scores);

      // Auto-insert top prediction immediately after swipe completes
      String topPrediction = _suggestionBar.getTopSuggestion();
      if (topPrediction != null && !topPrediction.isEmpty())
      {
        // If manual typing in progress, add space after it
        if (_contextTracker.getCurrentWordLength() > 0 && ic != null)
        {
          ic.commitText(" ", 1);
          _contextTracker.clearCurrentWord();
          _contextTracker.clearLastAutoInsertedWord();
          _contextTracker.setLastCommitSource(PredictionSource.USER_TYPED_TAP);
        }

        // Clear tracking before selection to prevent deletion
        _contextTracker.clearLastAutoInsertedWord();
        _contextTracker.setLastCommitSource(PredictionSource.UNKNOWN);

        // Insert the top prediction
        onSuggestionSelected(topPrediction, ic, editorInfo, resources);

        // Track as auto-inserted for replacement
        String cleanPrediction = topPrediction.replaceAll("^raw:", "");
        _contextTracker.setLastAutoInsertedWord(cleanPrediction);
        _contextTracker.setLastCommitSource(PredictionSource.NEURAL_SWIPE);

        // Re-display suggestions after auto-insertion
        _suggestionBar.setSuggestionsWithScores(predictions, scores);
      }
    }
  }

  /**
   * Updates context with a completed word.
   * Commits the word to context tracker and adds to word predictor.
   *
   * @param word Completed word to add to context
   */
  private void updateContext(String word)
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
   * Updates predictions for the current partial word being typed.
   * Uses contextual prediction with previous words.
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
  public void onSuggestionSelected(String word, InputConnection ic, EditorInfo editorInfo, Resources resources)
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
    // v1.32.341: Use ContractionManager for lookup
    boolean isKnownContraction = _contractionManager.isKnownContraction(word);

    // Skip autocorrect for:
    // 1. Known contractions (prevent fuzzy matching)
    // 2. Raw predictions (user explicitly selected this neural output)
    if (isKnownContraction || isRawPrediction)
    {
      if (isKnownContraction)
      {
        android.util.Log.d("Keyboard2", String.format("KNOWN CONTRACTION: \"%s\" - skipping autocorrect", word));
      }
      if (isRawPrediction)
      {
        android.util.Log.d("Keyboard2", String.format("RAW PREDICTION: \"%s\" - skipping autocorrect", word));
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
          android.util.Log.d("Keyboard2", String.format("FINAL AUTOCORRECT: \"%s\" → \"%s\"", word, correctedWord));
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
    if (isSwipeAutoInsert && _currentSwipeData != null && _predictionCoordinator.getMlDataStore() != null)
    {
      // Create a new ML data object with the selected word
      android.util.DisplayMetrics metrics = resources.getDisplayMetrics();
      SwipeMLData mlData = new SwipeMLData(word, "user_selection",
                                           metrics.widthPixels, metrics.heightPixels,
                                           _keyboardView.getHeight());
      
      // Copy trace points from the temporary data
      for (SwipeMLData.TracePoint point : _currentSwipeData.getTracePoints())
      {
        // Add points with their original normalized values and timestamps
        // Since they're already normalized, we need to denormalize then renormalize
        // to ensure proper storage
        float rawX = point.x * metrics.widthPixels;
        float rawY = point.y * metrics.heightPixels;
        // Reconstruct approximate timestamp (this is a limitation of the current design)
        long timestamp = System.currentTimeMillis() - 1000 + point.tDeltaMs;
        mlData.addRawPoint(rawX, rawY, timestamp);
      }
      
      // Copy registered keys
      for (String key : _currentSwipeData.getRegisteredKeys())
      {
        mlData.addRegisteredKey(key);
      }

      // Store the ML data
      _predictionCoordinator.getMlDataStore().storeSwipeData(mlData);

    }
    
    // Reset swipe tracking
    _contextTracker.setWasLastInputSwipe(false);
    _currentSwipeData = null;

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
            _contextTracker.getLastCommitSource() ==PredictionSource.NEURAL_SWIPE)
        {
          android.util.Log.d("Keyboard2", "REPLACE: Deleting auto-inserted word: '" + _contextTracker.getLastAutoInsertedWord() +"'");

          int deleteCount = _contextTracker.getLastAutoInsertedWord().length() + 1; // Word + trailing space
          boolean deletedLeadingSpace = false;

          if (inTermuxApp)
          {
            // TERMUX: Use backspace key events instead of InputConnection methods
            // Termux doesn't support deleteSurroundingText properly
            android.util.Log.d("Keyboard2", "TERMUX: Using backspace key events to delete " + deleteCount + " chars");

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
            android.util.Log.d("Keyboard2", "REPLACE: Text before cursor (50 chars): '" + debugBefore + "'");
            android.util.Log.d("Keyboard2", "REPLACE: Delete count = " + deleteCount);

            // Delete the auto-inserted word and its space
            ic.deleteSurroundingText(deleteCount, 0);

            CharSequence debugAfter = ic.getTextBeforeCursor(50, 0);
            android.util.Log.d("Keyboard2", "REPLACE: After deleting word, text before cursor: '" + debugAfter + "'");

            // Also need to check if there was a space added before it
            CharSequence textBefore = ic.getTextBeforeCursor(1, 0);
            android.util.Log.d("Keyboard2", "REPLACE: Checking for leading space, got: '" + textBefore + "'");
            if (textBefore != null && textBefore.length() > 0 && textBefore.charAt(0) == ' ')
            {
              android.util.Log.d("Keyboard2", "REPLACE: Deleting leading space");
              // Delete the leading space too
              ic.deleteSurroundingText(1, 0);

              CharSequence debugFinal = ic.getTextBeforeCursor(50, 0);
              android.util.Log.d("Keyboard2", "REPLACE: After deleting leading space: '" + debugFinal + "'");
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
          android.util.Log.d("Keyboard2", "TYPING PREDICTION: Deleting partial word: '" + _contextTracker.getCurrentWord() + "'");

          if (inTermuxApp)
          {
            // TERMUX: Use backspace key events
            android.util.Log.d("Keyboard2", "TERMUX: Using backspace key events to delete " + _contextTracker.getCurrentWordLength() + " chars");
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
            android.util.Log.d("Keyboard2", "TYPING PREDICTION: After deleting partial, text before cursor: '" + debugAfter + "'");
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
          android.util.Log.d("Keyboard2", "TERMUX MODE (non-swipe): textToInsert = '" + textToInsert + "'");
        }
        else
        {
          // Normal mode OR swipe in Termux: Insert word with space after (and before if needed)
          // For swipe typing, we always add trailing spaces even in Termux mode for better UX
          textToInsert = needsSpaceBefore ? " " + word + " " : word + " ";
          android.util.Log.d("Keyboard2", "NORMAL/SWIPE MODE: textToInsert = '" + textToInsert + "' (needsSpaceBefore=" + needsSpaceBefore + ", isSwipe=" + isSwipeAutoInsert + ")");
        }

        android.util.Log.d("Keyboard2", "Committing text: '" + textToInsert + "' (length=" + textToInsert.length() + ")");
        ic.commitText(textToInsert, 1);

        // Track that this commit was from candidate selection (manual tap)
        // Note: Auto-insertions set this separately to NEURAL_SWIPE
        if (_contextTracker.getLastCommitSource() !=PredictionSource.NEURAL_SWIPE)
        {
          _contextTracker.setLastCommitSource(PredictionSource.CANDIDATE_SELECTION);
        }
      }
      catch (Exception e)
      {
      }

      // Update context with the selected word
      updateContext(word);

      // Clear current word
      // NOTE: Don't clear suggestions here - they're re-displayed after auto-insertion
      _contextTracker.clearCurrentWord();
    }
  }
  
  /**
   * Handle regular typing predictions (non-swipe)
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
   * Handle backspace for prediction tracking
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
   * Update predictions based on current partial word
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
      android.util.Log.e("Keyboard2", "DELETE_LAST_WORD: Error detecting Termux", e);
    }

    // For Termux, use Ctrl+W key event which Termux handles correctly
    // Termux doesn't support InputConnection methods, but processes terminal control sequences
    if (inTermux)
    {
      android.util.Log.d("Keyboard2", "DELETE_LAST_WORD: Using Ctrl+W (^W) for Termux");
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
      android.util.Log.d("Keyboard2", "DELETE_LAST_WORD: Deleting auto-inserted word: '" + _contextTracker.getLastAutoInsertedWord() +"'");

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
          android.util.Log.d("Keyboard2", "DELETE_LAST_WORD: Deleted " + deleteCount + " characters");

          // Clear tracking
          _contextTracker.clearLastAutoInsertedWord();
          _contextTracker.setLastCommitSource(PredictionSource.UNKNOWN);
          return;
        }
      }

      // If verification failed, fall through to delete last word generically
      android.util.Log.d("Keyboard2", "DELETE_LAST_WORD: Auto-inserted word verification failed, using generic delete");
    }

    // Fallback: Delete the last word before cursor (generic approach)
    CharSequence textBefore = ic.getTextBeforeCursor(100, 0);
    if (textBefore == null || textBefore.length() == 0)
    {
      android.util.Log.d("Keyboard2", "DELETE_LAST_WORD: No text before cursor");
      return;
    }

    String beforeStr = textBefore.toString();
    int cursorPos = beforeStr.length();

    // Skip trailing whitespace
    while (cursorPos > 0 && Character.isWhitespace(beforeStr.charAt(cursorPos - 1)))
      cursorPos--;

    if (cursorPos == 0)
    {
      android.util.Log.d("Keyboard2", "DELETE_LAST_WORD: Only whitespace before cursor");
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
      android.util.Log.d("Keyboard2", "DELETE_LAST_WORD: Refusing to delete " + deleteCount + " characters (safety limit)");
      deleteCount = 50;
    }

    android.util.Log.d("Keyboard2", "DELETE_LAST_WORD: Deleting last word (generic), count=" + deleteCount);
    ic.deleteSurroundingText(deleteCount, 0);

    // Clear tracking
    _contextTracker.clearLastAutoInsertedWord();
    _contextTracker.setLastCommitSource(PredictionSource.UNKNOWN);
  }

  /**
   * Calculate dynamic keyboard height based on user settings (like calibration page)
   * Supports orientation, foldable devices, and user height preferences
   */
  private float calculateDynamicKeyboardHeight()
  {
    try {
      // Get screen dimensions
      android.util.DisplayMetrics metrics = new android.util.DisplayMetrics();
      android.view.WindowManager wm = (android.view.WindowManager) _context.getSystemService(Context.WINDOW_SERVICE);
      wm.getDefaultDisplay().getMetrics(metrics);

      // Check foldable state
      FoldStateTracker foldTracker = new FoldStateTracker(_context);
      boolean foldableUnfolded = foldTracker.isUnfolded();

      // Check orientation
      boolean isLandscape = _context.getResources().getConfiguration().orientation ==
                            android.content.res.Configuration.ORIENTATION_LANDSCAPE;

      // Get user height preference (same logic as calibration)
      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(_context);
      int keyboardHeightPref;
      
      if (isLandscape) {
        String key = foldableUnfolded ? "keyboard_height_landscape_unfolded" : "keyboard_height_landscape";
        keyboardHeightPref = prefs.getInt(key, 50);
      } else {
        String key = foldableUnfolded ? "keyboard_height_unfolded" : "keyboard_height";
        keyboardHeightPref = prefs.getInt(key, 35);
      }
      
      // Calculate dynamic height
      float keyboardHeightPercent = keyboardHeightPref / 100.0f;
      float calculatedHeight = metrics.heightPixels * keyboardHeightPercent;

      return calculatedHeight;
      
    } catch (Exception e) {
      // Fallback to view height
      return _keyboardView.getHeight();
    }
  }
  
  /**
   * Get user keyboard height percentage for logging
   */
  public void handleSwipeTyping(List<KeyboardData.Key> swipedKeys,
                                List<android.graphics.PointF> swipePath,
                                List<Long> timestamps,
                                InputConnection ic,
                                EditorInfo editorInfo,
                                Resources resources)
  {
    // Clear auto-inserted word tracking when new swipe starts
    _contextTracker.clearLastAutoInsertedWord();

    // TODO: Re-enable debug logging by passing debugLogger interface
    // DEBUG: Log swipe start
    // sendDebugLog("\n========== NEW SWIPE ==========\n");
    // sendDebugLog(String.format("Path points: %d, Keys detected: %d\n",
    //     swipePath != null ? swipePath.size() : 0,
    //     swipedKeys != null ? swipedKeys.size() : 0));

    // DEBUG: Log keyboard dimensions and first/last path points
    // if (_keyboardView != null && swipePath != null && swipePath.size() > 0)
    // {
    //   sendDebugLog(String.format("Keyboard dimensions: %dx%d\n",
    //       _keyboardView.getWidth(), _keyboardView.getHeight()));
    //   android.graphics.PointF first = swipePath.get(0);
    //   android.graphics.PointF last = swipePath.get(swipePath.size() - 1);
    //   sendDebugLog(String.format("Path: (%.1f, %.1f) → (%.1f, %.1f)\n",
    //       first.x, first.y, last.x, last.y));
    //
    //   // Calculate and log sampling rate
    //   if (timestamps != null && timestamps.size() > 1)
    //   {
    //     long totalTime = timestamps.get(timestamps.size() - 1) - timestamps.get(0);
    //     float samplingHz = (timestamps.size() - 1) * 1000.0f / totalTime;
    //     sendDebugLog(String.format("Sampling rate: %.1f Hz (%.0fms total)\n",
    //         samplingHz, (float)totalTime));
    //   }
    // }

    if (!_config.swipe_typing_enabled)
    {
      return;
    }

    // OPTIMIZATION v1.32.529: Ensure neural engine is loaded before first swipe
    // If not loaded in onCreate (rare edge case), lazy-load synchronously now
    _predictionCoordinator.ensureNeuralEngineReady();

    if (_predictionCoordinator.getNeuralEngine() == null)
    {
      // Fallback to word predictor if engine not initialized
      if (_predictionCoordinator.getWordPredictor() == null)
      {
        return;
      }

      // Ensure prediction engines are initialized (lazy initialization)
      _predictionCoordinator.ensureInitialized();

      // Neural engine dimensions and key positions already set in onStartInputView
    }
    
    // Mark that last input was a swipe for ML data collection
    _contextTracker.setWasLastInputSwipe(true);
    
    // Prepare ML data (will be saved if user selects a prediction)
    android.util.DisplayMetrics metrics = resources.getDisplayMetrics();
    _currentSwipeData = new SwipeMLData("", "user_selection",
                                        metrics.widthPixels, metrics.heightPixels,
                                        _keyboardView.getHeight());
    
    // Add swipe path points with timestamps
    if (swipePath != null && timestamps != null && swipePath.size() == timestamps.size())
    {
      for (int i = 0; i < swipePath.size(); i++)
      {
        android.graphics.PointF point = swipePath.get(i);
        long timestamp = timestamps.get(i);
        _currentSwipeData.addRawPoint(point.x, point.y, timestamp);
      }
    }
      
    // Build key sequence from swiped keys for ML data ONLY
    // NOTE: This is gesture tracker's detection - neural network will recalculate independently
    StringBuilder gestureTrackerKeys = new StringBuilder();
    for (KeyboardData.Key key : swipedKeys)
    {
      if (key != null && key.keys[0] != null)
      {
        KeyValue kv = key.keys[0];
        if (kv.getKind() == KeyValue.Kind.Char)
        {
          char c = kv.getChar();
          gestureTrackerKeys.append(c);
          // Add to ML data
          if (_currentSwipeData != null)
          {
            _currentSwipeData.addRegisteredKey(String.valueOf(c));
          }
        }
      }
    }

    // TODO: Re-enable debug logging
    // DEBUG: Log gesture tracker's detection (for comparison)
    // sendDebugLog(String.format("Gesture tracker keys: \"%s\" (%d keys filtered from %d path points)\n",
    //     gestureTrackerKeys.toString(), swipedKeys.size(), swipePath != null ? swipePath.size() : 0));

    // TODO: Re-enable file logging
    // Log to file for analysis
    // if (_logWriter != null && gestureTrackerKeys.length() > 0)
    // {
    //   try
    //   {
    //     _logWriter.write("[" + new java.util.Date() + "] Swipe: " + gestureTrackerKeys.toString() + "\n");
    //     _logWriter.flush();
    //   }
    //   catch (IOException e)
    //   {
    //   }
    // }

    if (swipePath != null && !swipePath.isEmpty())
    {
      // Create SwipeInput exactly like SwipeCalibrationActivity (empty swipedKeys)
      // This ensures neural system handles key detection internally for consistency
      // The neural network will recalculate keys from the full path without filtering
      SwipeInput swipeInput = new SwipeInput(swipePath != null ? swipePath : new ArrayList<>(),
                                            timestamps != null ? timestamps : new ArrayList<>(),
                                            new ArrayList<>()); // Empty - neural recalculates keys
      
      // UNIFIED PREDICTION STRATEGY: All predictions wait for gesture completion
      // This matches SwipeCalibrationActivity behavior and eliminates premature predictions

      // Cancel any pending predictions first
      if (_predictionCoordinator.getAsyncPredictionHandler() != null)
      {
        _predictionCoordinator.getAsyncPredictionHandler().cancelPendingPredictions();
      }
      
      // Request predictions asynchronously - always done on gesture completion
      // which matches the calibration activity's ACTION_UP behavior
      if (_predictionCoordinator.getAsyncPredictionHandler() != null)
      {
        _predictionCoordinator.getAsyncPredictionHandler().requestPredictions(swipeInput, new AsyncPredictionHandler.PredictionCallback()
        {
          @Override
          public void onPredictionsReady(List<String> predictions, List<Integer> scores)
          {
            // Process predictions on UI thread
            handlePredictionResults(predictions, scores, ic, editorInfo, resources);
          }
          
          @Override
          public void onPredictionError(String error)
          {
            // Clear suggestions on error
            if (_suggestionBar != null)
            {
              _suggestionBar.clearSuggestions();
            }
          }
        });
      }
      else
      {
        // Fallback to synchronous prediction if async handler not available
        long startTime = System.currentTimeMillis();
        PredictionResult result = _predictionCoordinator.getNeuralEngine().predict(swipeInput);
      long predictionTime = System.currentTimeMillis() - startTime;
      List<String> predictions = result.words;
      
      if (predictions.size() > 0)
      {
      }
      else
      {
      }

      // TODO: Re-enable file logging
      // Log predictions to file
      // if (_logWriter != null)
      // {
      //   try
      //   {
      //     _logWriter.write("  Predictions: " + predictions + " (" + predictionTime + "ms)\n");
      //     _logWriter.write("  Scores: " + result.scores + "\n");
      //     _logWriter.flush();
      //   }
      //   catch (IOException e)
      //   {
      //   }
      // }
      
      // Show suggestions in the bar
      if (_suggestionBar != null && !predictions.isEmpty())
      {
        _suggestionBar.setShowDebugScores(_config.swipe_show_debug_scores);
        _suggestionBar.setSuggestionsWithScores(predictions, result.scores);
        
        // Auto-commit the first suggestion if confidence is high
        if (predictions.size() > 0)
        {
          // For now, just show suggestions - user can tap to select
          // Could auto-commit the first word here if desired
        }
      }
      else
      {
      }
      } // Close fallback else block
    }
    else
    {
    }
  }

}
