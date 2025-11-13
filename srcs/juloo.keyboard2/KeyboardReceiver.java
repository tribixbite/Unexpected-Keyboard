package juloo.keyboard2;

import android.content.Context;
import android.content.Intent;
import android.os.Build.VERSION;
import android.os.Handler;
import android.os.IBinder;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.InputConnection;
import android.view.inputmethod.InputMethodManager;

/**
 * Handles keyboard events and state changes for Keyboard2.
 *
 * This class centralizes logic for:
 * - Keyboard event handling (special keys, layout switching)
 * - View state management (shift, compose, selection)
 * - Layout switching (text, numeric, emoji, clipboard)
 * - Input method switching
 * - Clipboard and emoji pane management
 *
 * Responsibilities:
 * - Handle special key events (CONFIG, SWITCH_TEXT, SWITCH_NUMERIC, etc.)
 * - Manage keyboard view state updates
 * - Coordinate with managers for layout, clipboard, and input operations
 * - Bridge between KeyEventHandler and Keyboard2
 *
 * NOT included (remains in Keyboard2):
 * - InputMethodService lifecycle methods
 * - Manager initialization
 * - Configuration management
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.368).
 */
public class KeyboardReceiver implements KeyEventHandler.IReceiver
{
  private final Context _context;
  private final Keyboard2 _keyboard2;
  private final Keyboard2View _keyboardView;
  private final LayoutManager _layoutManager;
  private final ClipboardManager _clipboardManager;
  private final PredictionContextTracker _contextTracker;
  private final InputCoordinator _inputCoordinator;
  private final SubtypeManager _subtypeManager;
  private final Handler _handler;

  // View references
  private ViewGroup _emojiPane;
  private ViewGroup _contentPaneContainer;

  /**
   * Creates a new KeyboardReceiver.
   *
   * @param context Android context
   * @param keyboard2 Reference to Keyboard2 service for lifecycle methods
   * @param keyboardView Main keyboard view
   * @param layoutManager Layout manager
   * @param clipboardManager Clipboard manager
   * @param contextTracker Prediction context tracker
   * @param inputCoordinator Input coordinator
   * @param subtypeManager Subtype manager
   * @param handler Handler for UI operations
   */
  public KeyboardReceiver(Context context, Keyboard2 keyboard2, Keyboard2View keyboardView,
                         LayoutManager layoutManager, ClipboardManager clipboardManager,
                         PredictionContextTracker contextTracker, InputCoordinator inputCoordinator,
                         SubtypeManager subtypeManager, Handler handler)
  {
    _context = context;
    _keyboard2 = keyboard2;
    _keyboardView = keyboardView;
    _layoutManager = layoutManager;
    _clipboardManager = clipboardManager;
    _contextTracker = contextTracker;
    _inputCoordinator = inputCoordinator;
    _subtypeManager = subtypeManager;
    _handler = handler;
  }

  /**
   * Sets references to emoji pane and content pane container.
   * These are created later in Keyboard2 lifecycle.
   *
   * @param emojiPane Emoji pane view
   * @param contentPaneContainer Container for emoji/clipboard panes
   */
  public void setViewReferences(ViewGroup emojiPane, ViewGroup contentPaneContainer)
  {
    _emojiPane = emojiPane;
    _contentPaneContainer = contentPaneContainer;
  }

  @Override
  public void handle_event_key(KeyValue.Event ev)
  {
    switch (ev)
    {
      case CONFIG:
        Intent intent = new Intent(_context, SettingsActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        _context.startActivity(intent);
        break;

      case SWITCH_TEXT:
        _keyboardView.setKeyboard(_layoutManager.clearSpecialLayout());
        break;

      case SWITCH_NUMERIC:
        _keyboardView.setKeyboard(_layoutManager.loadNumpad(R.xml.numeric));
        break;

      case SWITCH_EMOJI:
        if (_emojiPane == null)
          _emojiPane = (ViewGroup)_keyboard2.inflate_view(R.layout.emoji_pane);

        // Show emoji pane in content container (keyboard stays visible below)
        if (_contentPaneContainer != null)
        {
          _contentPaneContainer.removeAllViews();
          _contentPaneContainer.addView(_emojiPane);
          _contentPaneContainer.setVisibility(View.VISIBLE);
        }
        else
        {
          // Fallback for when predictions disabled (no container)
          _keyboard2.setInputView(_emojiPane);
        }
        break;

      case SWITCH_CLIPBOARD:
        // Get clipboard pane from manager (lazy initialization)
        ViewGroup clipboardPane = _clipboardManager.getClipboardPane(_keyboard2.getLayoutInflater());

        // Reset search mode and clear any previous search when showing clipboard pane
        _clipboardManager.resetSearchOnShow();

        // Show clipboard pane in content container (keyboard stays visible below)
        if (_contentPaneContainer != null)
        {
          _contentPaneContainer.removeAllViews();
          _contentPaneContainer.addView(clipboardPane);
          _contentPaneContainer.setVisibility(View.VISIBLE);
        }
        else
        {
          // Fallback for when predictions disabled (no container)
          _keyboard2.setInputView(clipboardPane);
        }
        break;

      case SWITCH_BACK_EMOJI:
      case SWITCH_BACK_CLIPBOARD:
        // Exit clipboard search mode when switching back
        _clipboardManager.resetSearchOnHide();

        // Hide content pane (keyboard remains visible)
        if (_contentPaneContainer != null)
        {
          _contentPaneContainer.setVisibility(View.GONE);
        }
        else
        {
          // Fallback for when predictions disabled
          _keyboard2.setInputView(_keyboardView);
        }
        break;

      case CHANGE_METHOD_PICKER:
        _subtypeManager.getInputMethodManager().showInputMethodPicker();
        break;

      case CHANGE_METHOD_AUTO:
        if (VERSION.SDK_INT < 28)
          _subtypeManager.getInputMethodManager().switchToLastInputMethod(_keyboard2.getConnectionToken());
        else
          _keyboard2.switchToNextInputMethod(false);
        break;

      case ACTION:
        InputConnection conn = _keyboard2.getCurrentInputConnection();
        if (conn != null)
          conn.performEditorAction(_keyboard2.actionId);
        break;

      case SWITCH_FORWARD:
        _keyboardView.setKeyboard(_layoutManager.incrTextLayout(1));
        break;

      case SWITCH_BACKWARD:
        _keyboardView.setKeyboard(_layoutManager.incrTextLayout(-1));
        break;

      case SWITCH_GREEKMATH:
        _keyboardView.setKeyboard(_layoutManager.loadNumpad(R.xml.greekmath));
        break;

      case CAPS_LOCK:
        set_shift_state(true, true);
        break;

      case SWITCH_VOICE_TYPING:
        if (!VoiceImeSwitcher.switch_to_voice_ime(_keyboard2, _subtypeManager.getInputMethodManager(),
              Config.globalPrefs()))
          _keyboard2.getConfig().shouldOfferVoiceTyping = false;
        break;

      case SWITCH_VOICE_TYPING_CHOOSER:
        VoiceImeSwitcher.choose_voice_ime(_keyboard2, _subtypeManager.getInputMethodManager(),
            Config.globalPrefs());
        break;
    }
  }

  @Override
  public void set_shift_state(boolean state, boolean lock)
  {
    _keyboardView.set_shift_state(state, lock);
  }

  @Override
  public void set_compose_pending(boolean pending)
  {
    _keyboardView.set_compose_pending(pending);
  }

  @Override
  public void selection_state_changed(boolean selection_is_ongoing)
  {
    _keyboardView.set_selection_state(selection_is_ongoing);
  }

  @Override
  public InputConnection getCurrentInputConnection()
  {
    return _keyboard2.getCurrentInputConnection();
  }

  @Override
  public Handler getHandler()
  {
    return _handler;
  }

  @Override
  public void handle_text_typed(String text)
  {
    // Reset swipe tracking when regular typing occurs
    _contextTracker.setWasLastInputSwipe(false);
    _inputCoordinator.resetSwipeData();
    _keyboard2.handleRegularTyping(text);
  }

  @Override
  public void handle_backspace()
  {
    _keyboard2.handleBackspace();
  }

  @Override
  public void handle_delete_last_word()
  {
    _keyboard2.handleDeleteLastWord();
  }

  @Override
  public boolean isClipboardSearchMode()
  {
    return _clipboardManager.isInSearchMode();
  }

  @Override
  public void appendToClipboardSearch(String text)
  {
    _clipboardManager.appendToSearch(text);
  }

  @Override
  public void backspaceClipboardSearch()
  {
    _clipboardManager.deleteFromSearch();
  }

  @Override
  public void exitClipboardSearchMode()
  {
    _clipboardManager.clearSearch();
  }
}
