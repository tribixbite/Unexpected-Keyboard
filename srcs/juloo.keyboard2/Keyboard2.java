package juloo.keyboard2;

import android.annotation.TargetApi;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.res.Resources;
import android.inputmethodservice.InputMethodService;
import android.os.Build.VERSION;
import android.os.Handler;
import android.os.IBinder;
import android.text.InputType;
import android.util.Log;
import android.util.LogPrinter;
import android.util.TypedValue;
import android.view.*;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import android.view.inputmethod.InputMethodInfo;
import android.view.inputmethod.InputMethodManager;
import android.view.inputmethod.InputMethodSubtype;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.HashMap;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import juloo.keyboard2.ml.SwipeMLData;
import juloo.keyboard2.ml.SwipeMLDataStore;
import juloo.keyboard2.prefs.LayoutsPreference;

public class Keyboard2 extends InputMethodService
  implements SharedPreferences.OnSharedPreferenceChangeListener,
  SuggestionBar.OnSuggestionSelectedListener,
  ConfigChangeListener
{
  // Unified prediction strategy: All predictions wait for gesture completion
  // to match SwipeCalibrationActivity behavior and eliminate premature predictions
  private Keyboard2View _keyboardView;
  private KeyEventHandler _keyeventhandler;
  /** If not 'null', the layout to use instead of [_config.current_layout]. */
  private KeyboardData _currentSpecialLayout;
  /** Layout associated with the currently selected locale. Not 'null'. */
  private KeyboardData _localeTextLayout;
  private ViewGroup _emojiPane = null;
  private FrameLayout _contentPaneContainer = null; // Container for emoji/clipboard panes
  public int actionId; // Action performed by the Action key.
  private Handler _handler;

  // Clipboard management (v1.32.349: extracted to ClipboardManager)
  private ClipboardManager _clipboardManager;

  // Configuration management (v1.32.345: extracted to ConfigurationManager)
  private ConfigurationManager _configManager;
  private Config _config; // Cached reference from _configManager, updated by ConfigChangeListener

  // Prediction coordination (v1.32.346: extracted to PredictionCoordinator)
  private PredictionCoordinator _predictionCoordinator;

  // UI components (remain in Keyboard2 for view integration)
  private SuggestionBar _suggestionBar;
  private LinearLayout _inputViewContainer;
  private BufferedWriter _logWriter = null;

  // Prediction context tracking (v1.32.342: extracted to PredictionContextTracker)
  private PredictionContextTracker _contextTracker;

  // Contraction mappings for apostrophe insertion (v1.32.341: extracted to ContractionManager)
  private ContractionManager _contractionManager;

  // Input coordination (v1.32.350: extracted to InputCoordinator)
  private InputCoordinator _inputCoordinator;

  // Debug mode for swipe pipeline logging
  private boolean _debugMode = false;
  private android.content.BroadcastReceiver _debugModeReceiver;

  /** Layout currently visible before it has been modified. */
  KeyboardData current_layout_unmodified()
  {
    if (_currentSpecialLayout != null)
      return _currentSpecialLayout;
    KeyboardData layout = null;
    int layout_i = _config.get_current_layout();
    if (layout_i >= _config.layouts.size())
      layout_i = 0;
    if (layout_i < _config.layouts.size())
      layout = _config.layouts.get(layout_i);
    if (layout == null)
      layout = _localeTextLayout;
    return layout;
  }

  /** Layout currently visible. */
  KeyboardData current_layout()
  {
    if (_currentSpecialLayout != null)
      return _currentSpecialLayout;
    return LayoutModifier.modify_layout(current_layout_unmodified());
  }

  void setTextLayout(int l)
  {
    _config.set_current_layout(l);
    _currentSpecialLayout = null;
    _keyboardView.setKeyboard(current_layout());
  }

  void incrTextLayout(int delta)
  {
    int s = _config.layouts.size();
    setTextLayout((_config.get_current_layout() + delta + s) % s);
  }

  void setSpecialLayout(KeyboardData l)
  {
    _currentSpecialLayout = l;
    _keyboardView.setKeyboard(l);
  }

  KeyboardData loadLayout(int layout_id)
  {
    return KeyboardData.load(getResources(), layout_id);
  }

  /** Load a layout that contains a numpad. */
  KeyboardData loadNumpad(int layout_id)
  {
    return LayoutModifier.modify_numpad(KeyboardData.load(getResources(), layout_id),
        current_layout_unmodified());
  }

  KeyboardData loadPinentry(int layout_id)
  {
    return LayoutModifier.modify_pinentry(KeyboardData.load(getResources(), layout_id),
        current_layout_unmodified());
  }

  @Override
  public void onCreate()
  {
    super.onCreate();
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
    _handler = new Handler(getMainLooper());
    _keyeventhandler = new KeyEventHandler(this.new Receiver());

    // Create FoldStateTracker for device fold state monitoring
    FoldStateTracker foldStateTracker = new FoldStateTracker(this);

    // Initialize global config for KeyEventHandler
    Config.initGlobalConfig(prefs, getResources(), _keyeventhandler, foldStateTracker.isUnfolded());

    // Initialize configuration manager (v1.32.345: extracted configuration management)
    _configManager = new ConfigurationManager(this, Config.globalConfig(), foldStateTracker);
    _config = _configManager.getConfig(); // Cache reference for convenience
    _configManager.registerConfigChangeListener(this); // Register for config change notifications

    // Register ConfigurationManager as SharedPreferences listener
    prefs.registerOnSharedPreferenceChangeListener(_configManager);

    // Check if we're the default IME and remind user if not
    checkAndPromptDefaultIME();
    _keyboardView = (Keyboard2View)inflate_view(R.layout.keyboard);
    _keyboardView.reset();
    Logs.set_debug_logs(getResources().getBoolean(R.bool.debug_logs));
    ClipboardHistoryService.on_startup(this, _keyeventhandler);

    // Fold state change callback is handled by ConfigurationManager

    // Load contraction mappings for apostrophe insertion (v1.32.341: extracted to ContractionManager)
    _contractionManager = new ContractionManager(this);
    _contractionManager.loadMappings();

    // Initialize clipboard manager (v1.32.349: extracted to ClipboardManager)
    _clipboardManager = new ClipboardManager(this, _config);

    // Initialize prediction context tracker (v1.32.342: extracted to PredictionContextTracker)
    _contextTracker = new PredictionContextTracker();

    // Initialize prediction coordinator (v1.32.346: extracted prediction engine management)
    _predictionCoordinator = new PredictionCoordinator(this, _config);

    // Initialize input coordinator (v1.32.350: extracted input handling logic)
    // Note: _suggestionBar will be set later in onStartInputView
    _inputCoordinator = new InputCoordinator(
      this,
      _config,
      _contextTracker,
      _predictionCoordinator,
      _contractionManager,
      null, // _suggestionBar created later in onStartInputView
      _keyboardView,
      _keyeventhandler
    );

    if (_config.word_prediction_enabled || _config.swipe_typing_enabled)
    {
      _predictionCoordinator.initialize();

      // Set swipe typing components on keyboard view if swipe is enabled
      if (_config.swipe_typing_enabled && _predictionCoordinator.isSwipeTypingAvailable())
      {
        android.util.Log.d("Keyboard2", "Neural engine initialized - dimensions and key positions will be set after layout");
        _keyboardView.setSwipeTypingComponents(_predictionCoordinator.getWordPredictor(), this);
      }
    }

    // Initialize log writer for swipe analysis
    try
    {
      _logWriter = new BufferedWriter(new FileWriter("/data/data/com.termux/files/home/swipe_log.txt", true));
      _logWriter.write("\n=== Keyboard2 Started: " + new java.util.Date() + " ===\n");
      _logWriter.flush();
    }
    catch (IOException e)
    {
    }

    // Register broadcast receiver for debug mode control
    _debugModeReceiver = new android.content.BroadcastReceiver()
    {
      @Override
      public void onReceive(android.content.Context context, android.content.Intent intent)
      {
        if ("juloo.keyboard2.SET_DEBUG_MODE".equals(intent.getAction()))
        {
          _debugMode = intent.getBooleanExtra("debug_enabled", false);
          if (_debugMode)
          {
            sendDebugLog("=== Debug mode enabled ===\n");
          }
        }
      }
    };
    android.content.IntentFilter debugFilter = new android.content.IntentFilter("juloo.keyboard2.SET_DEBUG_MODE");
    registerReceiver(_debugModeReceiver, debugFilter, android.content.Context.RECEIVER_NOT_EXPORTED);
  }

  @Override
  public void onDestroy() {
    super.onDestroy();

    _configManager.getFoldStateTracker().close();

    // Cleanup clipboard listener
    ClipboardHistoryService.on_shutdown();

    // Cleanup clipboard manager
    if (_clipboardManager != null)
    {
      _clipboardManager.cleanup();
    }

    // Cleanup prediction coordinator
    if (_predictionCoordinator != null)
    {
      _predictionCoordinator.shutdown();
    }

    // Unregister debug mode receiver
    if (_debugModeReceiver != null)
    {
      try
      {
        unregisterReceiver(_debugModeReceiver);
      }
      catch (Exception e)
      {
        // Already unregistered
      }
      _debugModeReceiver = null;
    }
  }

  /**
   * Send debug log message to SwipeDebugActivity if debug mode is enabled.
   * Only logs when debug mode is active (SwipeDebugActivity is open).
   */
  private void sendDebugLog(String message)
  {
    if (!_debugMode) return;

    try
    {
      android.content.Intent intent = new android.content.Intent(SwipeDebugActivity.ACTION_DEBUG_LOG);
      intent.setPackage(getPackageName());  // Explicit package for broadcast
      intent.putExtra(SwipeDebugActivity.EXTRA_LOG_MESSAGE, message);
      sendBroadcast(intent);
    }
    catch (Exception e)
    {
      // Silently fail if debug activity is not available
    }
  }

  private List<InputMethodSubtype> getEnabledSubtypes(InputMethodManager imm)
  {
    String pkg = getPackageName();
    for (InputMethodInfo imi : imm.getEnabledInputMethodList())
      if (imi.getPackageName().equals(pkg))
        return imm.getEnabledInputMethodSubtypeList(imi, true);
    return Arrays.asList();
  }

  @TargetApi(12)
  private ExtraKeys extra_keys_of_subtype(InputMethodSubtype subtype)
  {
    String extra_keys = subtype.getExtraValueOf("extra_keys");
    String script = subtype.getExtraValueOf("script");
    if (extra_keys != null)
      return ExtraKeys.parse(script, extra_keys);
    return ExtraKeys.EMPTY;
  }

  private void refreshAccentsOption(InputMethodManager imm, List<InputMethodSubtype> enabled_subtypes)
  {
    List<ExtraKeys> extra_keys = new ArrayList<ExtraKeys>();
    for (InputMethodSubtype s : enabled_subtypes)
      extra_keys.add(extra_keys_of_subtype(s));
    _config.extra_keys_subtype = ExtraKeys.merge(extra_keys);
  }

  InputMethodManager get_imm()
  {
    return (InputMethodManager)getSystemService(INPUT_METHOD_SERVICE);
  }

  @TargetApi(12)
  private InputMethodSubtype defaultSubtypes(InputMethodManager imm, List<InputMethodSubtype> enabled_subtypes)
  {
    if (VERSION.SDK_INT < 24)
      return imm.getCurrentInputMethodSubtype();
    // Android might return a random subtype, for example, the first in the
    // list alphabetically.
    InputMethodSubtype current_subtype = imm.getCurrentInputMethodSubtype();
    if (current_subtype == null)
      return null;
    for (InputMethodSubtype s : enabled_subtypes)
      if (s.getLanguageTag().equals(current_subtype.getLanguageTag()))
        return s;
    return null;
  }

  private void refreshSubtypeImm()
  {
    InputMethodManager imm = get_imm();
    _config.shouldOfferVoiceTyping = true;
    KeyboardData default_layout = null;
    _config.extra_keys_subtype = null;
    if (VERSION.SDK_INT >= 12)
    {
      List<InputMethodSubtype> enabled_subtypes = getEnabledSubtypes(imm);
      InputMethodSubtype subtype = defaultSubtypes(imm, enabled_subtypes);
      if (subtype != null)
      {
        String s = subtype.getExtraValueOf("default_layout");
        if (s != null)
          default_layout = LayoutsPreference.layout_of_string(getResources(), s);
        refreshAccentsOption(imm, enabled_subtypes);
      }
    }
    if (default_layout == null)
      default_layout = loadLayout(R.xml.latn_qwerty_us);
    _localeTextLayout = default_layout;
  }

  private String actionLabel_of_imeAction(int action)
  {
    int res;
    switch (action)
    {
      case EditorInfo.IME_ACTION_NEXT: res = R.string.key_action_next; break;
      case EditorInfo.IME_ACTION_DONE: res = R.string.key_action_done; break;
      case EditorInfo.IME_ACTION_GO: res = R.string.key_action_go; break;
      case EditorInfo.IME_ACTION_PREVIOUS: res = R.string.key_action_prev; break;
      case EditorInfo.IME_ACTION_SEARCH: res = R.string.key_action_search; break;
      case EditorInfo.IME_ACTION_SEND: res = R.string.key_action_send; break;
      case EditorInfo.IME_ACTION_UNSPECIFIED:
      case EditorInfo.IME_ACTION_NONE:
      default: return null;
    }
    return getResources().getString(res);
  }

  private void refresh_action_label(EditorInfo info)
  {
    // First try to look at 'info.actionLabel', if it isn't set, look at
    // 'imeOptions'.
    if (info.actionLabel != null)
    {
      _config.actionLabel = info.actionLabel.toString();
      actionId = info.actionId;
      _config.swapEnterActionKey = false;
    }
    else
    {
      int action = info.imeOptions & EditorInfo.IME_MASK_ACTION;
      _config.actionLabel = actionLabel_of_imeAction(action); // Might be null
      actionId = action;
      _config.swapEnterActionKey =
        (info.imeOptions & EditorInfo.IME_FLAG_NO_ENTER_ACTION) == 0;
    }
  }

  /** Might re-create the keyboard view. [_keyboardView.setKeyboard()] and
      [setInputView()] must be called soon after. */
  private void refresh_config()
  {
    // Delegate to ConfigurationManager, which will trigger listener callbacks
    _configManager.refresh(getResources());
  }

  // ConfigChangeListener implementation (v1.32.345)

  /**
   * Called when configuration has been refreshed.
   * Updates local config reference and propagates to components.
   */
  @Override
  public void onConfigChanged(Config newConfig)
  {
    // Update cached reference
    _config = newConfig;

    // Refresh subtitle IME
    refreshSubtypeImm();

    // Update clipboard manager config
    if (_clipboardManager != null)
    {
      _clipboardManager.setConfig(_config);
    }

    // Update prediction coordinator config
    if (_predictionCoordinator != null)
    {
      _predictionCoordinator.setConfig(_config);
    }

    // Update input coordinator config (v1.32.350)
    if (_inputCoordinator != null)
    {
      _inputCoordinator.setConfig(_config);
    }

    // Reset keyboard view
    if (_keyboardView != null)
    {
      _keyboardView.reset();
    }
  }

  /**
   * Called when theme has changed.
   * Re-creates keyboard views with new theme.
   */
  @Override
  public void onThemeChanged(int oldTheme, int newTheme)
  {
    // Recreate views with new theme
    _keyboardView = (Keyboard2View)inflate_view(R.layout.keyboard);
    _emojiPane = null;

    // Clean up clipboard manager views for theme change
    if (_clipboardManager != null)
    {
      _clipboardManager.cleanup();
    }

    setInputView(_keyboardView);
  }

  private KeyboardData refresh_special_layout(EditorInfo info)
  {
    switch (info.inputType & InputType.TYPE_MASK_CLASS)
    {
      case InputType.TYPE_CLASS_NUMBER:
      case InputType.TYPE_CLASS_PHONE:
      case InputType.TYPE_CLASS_DATETIME:
        if (_config.selected_number_layout == NumberLayout.PIN)
          return loadPinentry(R.xml.pin);
        else if (_config.selected_number_layout == NumberLayout.NUMBER)
          return loadNumpad(R.xml.numeric);
      default:
        break;
    }
    return null;
  }

  @Override
  public void onStartInputView(EditorInfo info, boolean restarting)
  {
    refresh_config();

    // Auto-close clipboard pane when switching to new app/field
    // Prevents confusing UX where clipboard briefly shows then keyboard closes
    if (_contentPaneContainer != null && _contentPaneContainer.getVisibility() == View.VISIBLE)
    {
      _contentPaneContainer.setVisibility(View.GONE);
      // Also reset search mode state
      _clipboardManager.resetSearchOnHide();
    }

    refresh_action_label(info);
    _currentSpecialLayout = refresh_special_layout(info);
    _keyboardView.setKeyboard(current_layout());
    _keyeventhandler.started(info);
    
    // Re-initialize word prediction components if settings have changed
    if (_config.word_prediction_enabled || _config.swipe_typing_enabled)
    {
      // Ensure prediction engines are initialized (lazy initialization)
      _predictionCoordinator.ensureInitialized();

      // Set keyboard dimensions for neural engine if available
      if (_config.swipe_typing_enabled && _predictionCoordinator.getNeuralEngine() != null && _keyboardView != null)
      {
        _predictionCoordinator.getNeuralEngine().setKeyboardDimensions(_keyboardView.getWidth(), _keyboardView.getHeight());
        _keyboardView.setSwipeTypingComponents(_predictionCoordinator.getWordPredictor(), this);
      }
      
      // Create suggestion bar if needed
      if (_suggestionBar == null)
      {
        _inputViewContainer = new LinearLayout(this);
        _inputViewContainer.setOrientation(LinearLayout.VERTICAL);
        
        // Get theme from keyboard view if available
        Theme theme = _keyboardView != null ? _keyboardView.getTheme() : null;
        _suggestionBar = theme != null ? new SuggestionBar(this, theme) : new SuggestionBar(this);
        _suggestionBar.setOnSuggestionSelectedListener(this);
        _suggestionBar.setOpacity(_config.suggestion_bar_opacity);

        // Update InputCoordinator with suggestion bar reference (v1.32.350)
        if (_inputCoordinator != null)
        {
          _inputCoordinator.setSuggestionBar(_suggestionBar);
        }

        // Wrap SuggestionBar in HorizontalScrollView for scrollable predictions
        android.widget.HorizontalScrollView scrollView = new android.widget.HorizontalScrollView(this);
        scrollView.setHorizontalScrollBarEnabled(false); // Hide scrollbar
        scrollView.setFillViewport(false); // Don't stretch content
        LinearLayout.LayoutParams scrollParams = new LinearLayout.LayoutParams(
          LinearLayout.LayoutParams.MATCH_PARENT,
          (int)TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 40,
            getResources().getDisplayMetrics()));
        scrollView.setLayoutParams(scrollParams);

        // Set SuggestionBar to wrap_content width for scrolling
        LinearLayout.LayoutParams suggestionParams = new LinearLayout.LayoutParams(
          LinearLayout.LayoutParams.WRAP_CONTENT,
          LinearLayout.LayoutParams.MATCH_PARENT);
        _suggestionBar.setLayoutParams(suggestionParams);

        scrollView.addView(_suggestionBar);
        _inputViewContainer.addView(scrollView);

        // Add content pane container (for clipboard/emoji) between suggestion bar and keyboard
        // This stays hidden until user opens clipboard or emoji pane
        // Height is based on user config (default 30% of screen height)
        _contentPaneContainer = new FrameLayout(this);
        int screenHeight = getResources().getDisplayMetrics().heightPixels;
        int paneHeight = (screenHeight * _config.clipboard_pane_height_percent) / 100;
        _contentPaneContainer.setLayoutParams(new LinearLayout.LayoutParams(
          LinearLayout.LayoutParams.MATCH_PARENT,
          paneHeight));
        _contentPaneContainer.setVisibility(View.GONE); // Hidden by default
        _inputViewContainer.addView(_contentPaneContainer);

        _inputViewContainer.addView(_keyboardView);
      }

      setInputView(_inputViewContainer != null ? _inputViewContainer : _keyboardView);

      // CRITICAL: Set correct keyboard dimensions for CGR after view is laid out
      if (_predictionCoordinator.getNeuralEngine() != null && _keyboardView != null) {
        _keyboardView.getViewTreeObserver().addOnGlobalLayoutListener(
          new android.view.ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
              // Ensure we have valid dimensions
              if (_keyboardView.getWidth() > 0 && _keyboardView.getHeight() > 0) {
                
                // Use dynamic keyboard dimensions based on user settings (like calibration)
                float keyboardWidth = _keyboardView.getWidth();
                float keyboardHeight = calculateDynamicKeyboardHeight();
                
                _predictionCoordinator.getNeuralEngine().setKeyboardDimensions(keyboardWidth, keyboardHeight);

                // CRITICAL: Set real key positions for 100% accurate coordinate mapping
                setNeuralKeyboardLayout();

                // Remove the listener to avoid repeated calls
                _keyboardView.getViewTreeObserver().removeOnGlobalLayoutListener(this);
              }
            }
          }
        );
      }
    }
    else
    {
      // Clean up if predictions are disabled
      // Note: _wordPredictor is managed by PredictionCoordinator
      // CGR recognizer cleanup handled by SwipeTypingEngine
      _suggestionBar = null;
      _inputViewContainer = null;
      setInputView(_keyboardView);
    }
    
    Logs.debug_startup_input_view(info, _config);
  }

  @Override
  public void setInputView(View v)
  {
    ViewParent parent = v.getParent();
    if (parent != null && parent instanceof ViewGroup)
      ((ViewGroup)parent).removeView(v);
    super.setInputView(v);
    updateSoftInputWindowLayoutParams();
    v.requestApplyInsets();
  }


  @Override
  public void updateFullscreenMode() {
    super.updateFullscreenMode();
    updateSoftInputWindowLayoutParams();
  }

  private void updateSoftInputWindowLayoutParams() {
    final Window window = getWindow().getWindow();
    // On API >= 35, Keyboard2View behaves as edge-to-edge
    // APIs 30 to 34 have visual artifact when edge-to-edge is enabled
    if (VERSION.SDK_INT >= 35)
    {
      WindowManager.LayoutParams wattrs = window.getAttributes();
      wattrs.layoutInDisplayCutoutMode =
        WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS;
      // Allow to draw behind system bars
      wattrs.setFitInsetsTypes(0);
      window.setDecorFitsSystemWindows(false);
    }
    updateLayoutHeightOf(window, ViewGroup.LayoutParams.MATCH_PARENT);
    final View inputArea = window.findViewById(android.R.id.inputArea);

    updateLayoutHeightOf(
            (View) inputArea.getParent(),
            isFullscreenMode()
                    ? ViewGroup.LayoutParams.MATCH_PARENT
                    : ViewGroup.LayoutParams.WRAP_CONTENT);
    updateLayoutGravityOf((View) inputArea.getParent(), Gravity.BOTTOM);

  }

  private static void updateLayoutHeightOf(final Window window, final int layoutHeight) {
    final WindowManager.LayoutParams params = window.getAttributes();
    if (params != null && params.height != layoutHeight) {
      params.height = layoutHeight;
      window.setAttributes(params);
    }
  }

  private static void updateLayoutHeightOf(final View view, final int layoutHeight) {
    final ViewGroup.LayoutParams params = view.getLayoutParams();
    if (params != null && params.height != layoutHeight) {
      params.height = layoutHeight;
      view.setLayoutParams(params);
    }
  }

  private static void updateLayoutGravityOf(final View view, final int layoutGravity) {
    final ViewGroup.LayoutParams lp = view.getLayoutParams();
    if (lp instanceof LinearLayout.LayoutParams) {
      final LinearLayout.LayoutParams params = (LinearLayout.LayoutParams) lp;
      if (params.gravity != layoutGravity) {
        params.gravity = layoutGravity;
        view.setLayoutParams(params);
      }
    } else if (lp instanceof FrameLayout.LayoutParams) {
      final FrameLayout.LayoutParams params = (FrameLayout.LayoutParams) lp;
      if (params.gravity != layoutGravity) {
        params.gravity = layoutGravity;
        view.setLayoutParams(params);
      }
    }
  }

  @Override
  public void onCurrentInputMethodSubtypeChanged(InputMethodSubtype subtype)
  {
    refreshSubtypeImm();
    _keyboardView.setKeyboard(current_layout());
  }

  @Override
  public void onUpdateSelection(int oldSelStart, int oldSelEnd, int newSelStart, int newSelEnd, int candidatesStart, int candidatesEnd)
  {
    super.onUpdateSelection(oldSelStart, oldSelEnd, newSelStart, newSelEnd, candidatesStart, candidatesEnd);
    _keyeventhandler.selection_updated(oldSelStart, newSelStart);
    if ((oldSelStart == oldSelEnd) != (newSelStart == newSelEnd))
      _keyboardView.set_selection_state(newSelStart != newSelEnd);
  }

  @Override
  public void onFinishInputView(boolean finishingInput)
  {
    super.onFinishInputView(finishingInput);
    _keyboardView.reset();
  }

  @Override
  public void onSharedPreferenceChanged(SharedPreferences _prefs, String _key)
  {
    // NOTE: ConfigurationManager is the primary SharedPreferences listener and handles
    // config refresh. This method handles additional UI updates.

    // Update keyboard layout
    if (_keyboardView != null)
    {
      _keyboardView.setKeyboard(current_layout());
    }

    // Update suggestion bar opacity if it exists
    if (_suggestionBar != null)
    {
      _suggestionBar.setOpacity(_config.suggestion_bar_opacity);
    }

    // Update neural predictor when model-related settings change
    // (This is redundant with onConfigChanged but kept for explicit model reloading)
    if (_key != null && (_key.equals("neural_custom_encoder_uri") ||
                        _key.equals("neural_custom_decoder_uri") ||
                        _key.equals("neural_model_version") ||
                        _key.equals("neural_user_max_seq_length")))
    {
      if (_predictionCoordinator.getNeuralEngine() != null)
      {
        _predictionCoordinator.getNeuralEngine().setConfig(_config);
        Log.d("Keyboard2", "Neural model setting changed: " + _key + " - engine config updated");
      }
    }
  }

  @Override
  public boolean onEvaluateFullscreenMode()
  {
    /* Entirely disable fullscreen mode. */
    return false;
  }

  /** Not static */
  public class Receiver implements KeyEventHandler.IReceiver
  {
    public void handle_event_key(KeyValue.Event ev)
    {
      switch (ev)
      {
        case CONFIG:
          Intent intent = new Intent(Keyboard2.this, SettingsActivity.class);
          intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
          startActivity(intent);
          break;

        case SWITCH_TEXT:
          _currentSpecialLayout = null;
          _keyboardView.setKeyboard(current_layout());
          break;

        case SWITCH_NUMERIC:
          setSpecialLayout(loadNumpad(R.xml.numeric));
          break;

        case SWITCH_EMOJI:
          if (_emojiPane == null)
            _emojiPane = (ViewGroup)inflate_view(R.layout.emoji_pane);

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
            setInputView(_emojiPane);
          }
          break;

        case SWITCH_CLIPBOARD:
          // Get clipboard pane from manager (lazy initialization)
          ViewGroup clipboardPane = _clipboardManager.getClipboardPane(getLayoutInflater());

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
            setInputView(clipboardPane);
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
            setInputView(_keyboardView);
          }
          break;

        case CHANGE_METHOD_PICKER:
          get_imm().showInputMethodPicker();
          break;

        case CHANGE_METHOD_AUTO:
          if (VERSION.SDK_INT < 28)
            get_imm().switchToLastInputMethod(getConnectionToken());
          else
            switchToNextInputMethod(false);
          break;

        case ACTION:
          InputConnection conn = getCurrentInputConnection();
          if (conn != null)
            conn.performEditorAction(actionId);
          break;

        case SWITCH_FORWARD:
          incrTextLayout(1);
          break;

        case SWITCH_BACKWARD:
          incrTextLayout(-1);
          break;

        case SWITCH_GREEKMATH:
          setSpecialLayout(loadNumpad(R.xml.greekmath));
          break;

        case CAPS_LOCK:
          set_shift_state(true, true);
          break;

        case SWITCH_VOICE_TYPING:
          if (!VoiceImeSwitcher.switch_to_voice_ime(Keyboard2.this, get_imm(),
                Config.globalPrefs()))
            _config.shouldOfferVoiceTyping = false;
          break;

        case SWITCH_VOICE_TYPING_CHOOSER:
          VoiceImeSwitcher.choose_voice_ime(Keyboard2.this, get_imm(),
              Config.globalPrefs());
          break;
      }
    }

    public void set_shift_state(boolean state, boolean lock)
    {
      _keyboardView.set_shift_state(state, lock);
    }

    public void set_compose_pending(boolean pending)
    {
      _keyboardView.set_compose_pending(pending);
    }

    public void selection_state_changed(boolean selection_is_ongoing)
    {
      _keyboardView.set_selection_state(selection_is_ongoing);
    }

    public InputConnection getCurrentInputConnection()
    {
      return Keyboard2.this.getCurrentInputConnection();
    }

    public Handler getHandler()
    {
      return _handler;
    }
    
    public void handle_text_typed(String text)
    {
      // Reset swipe tracking when regular typing occurs
      _contextTracker.setWasLastInputSwipe(false);
      _inputCoordinator.resetSwipeData();
      handleRegularTyping(text);
    }
    
    public void handle_backspace()
    {
      Keyboard2.this.handleBackspace();
    }

    public void handle_delete_last_word()
    {
      Keyboard2.this.handleDeleteLastWord();
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

  private IBinder getConnectionToken()
  {
    return getWindow().getWindow().getAttributes().token;
  }

  // v1.32.349: showDateFilterDialog() method removed - functionality moved to ClipboardManager class

  // SuggestionBar.OnSuggestionSelectedListener implementation
  /**
   * Update context with a completed word
   *
   * NOTE: This is a legacy helper method. New code should use
   * _contextTracker.commitWord() directly with appropriate PredictionSource.
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
   * Handle prediction results from async prediction handler
   */
  private void handlePredictionResults(List<String> predictions, List<Integer> scores)
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

    // Update suggestion bar (scores are already integers from neural system)
    if (_suggestionBar != null)
    {
      _suggestionBar.setShowDebugScores(_config.swipe_show_debug_scores);
      _suggestionBar.setSuggestionsWithScores(predictions, scores);

      // Auto-insert top (highest scoring) prediction immediately after swipe completes
      // This enables rapid consecutive swiping without manual taps
      String topPrediction = _suggestionBar.getTopSuggestion();
      if (topPrediction != null && !topPrediction.isEmpty())
      {
        InputConnection ic = getCurrentInputConnection();

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
        onSuggestionSelected(topPrediction);

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
  
  @Override
  public void onSuggestionSelected(String word)
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
    SwipeMLData currentSwipeData = _inputCoordinator.getCurrentSwipeData();
    if (isSwipeAutoInsert && currentSwipeData != null && _predictionCoordinator.getMlDataStore() != null)
    {
      // Create a new ML data object with the selected word
      android.util.DisplayMetrics metrics = getResources().getDisplayMetrics();
      SwipeMLData mlData = new SwipeMLData(word, "user_selection",
                                           metrics.widthPixels, metrics.heightPixels,
                                           _keyboardView.getHeight());

      // Copy trace points from the temporary data
      for (SwipeMLData.TracePoint point : currentSwipeData.getTracePoints())
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
      for (String key : currentSwipeData.getRegisteredKeys())
      {
        mlData.addRegisteredKey(key);
      }

      // Store the ML data
      _predictionCoordinator.getMlDataStore().storeSwipeData(mlData);

    }

    // Reset swipe tracking
    _contextTracker.setWasLastInputSwipe(false);
    _inputCoordinator.resetSwipeData();
    
    InputConnection ic = getCurrentInputConnection();
    if (ic != null)
    {
      try
      {
        // Detect if we're in Termux for special handling
        boolean inTermuxApp = false;
        try
        {
          EditorInfo editorInfo = getCurrentInputEditorInfo();
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
  public void handleRegularTyping(String text)
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
          EditorInfo editorInfo = getCurrentInputEditorInfo();
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
            InputConnection conn = getCurrentInputConnection();
            if (conn != null)
            {
              // At this point:
              // - The typed word "thid" has been committed via KeyEventHandler.send_text()
              // - The space " " has ALSO been committed via handle_text_typed(" ")
              // - Editor contains "thid "
              // - We need to delete both the word AND the space, then insert corrected word + space

              // Delete the typed word + space (already committed)
              conn.deleteSurroundingText(completedWord.length() + 1, 0);

              // Insert the corrected word WITH trailing space (normal apps only)
              conn.commitText(correctedWord + " ", 1);

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
   * Smart delete last word - deletes the last auto-inserted word or last typed word
   * Handles edge cases to avoid deleting too much text
   */
  public void handleDeleteLastWord()
  {
    InputConnection ic = getCurrentInputConnection();
    if (ic == null)
      return;

    // Check if we're in Termux - if so, use Ctrl+Backspace fallback
    boolean inTermux = false;
    try
    {
      EditorInfo editorInfo = getCurrentInputEditorInfo();
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
      android.view.WindowManager wm = (android.view.WindowManager) getSystemService(WINDOW_SERVICE);
      wm.getDefaultDisplay().getMetrics(metrics);
      
      // Check foldable state
      FoldStateTracker foldTracker = new FoldStateTracker(this);
      boolean foldableUnfolded = foldTracker.isUnfolded();
      
      // Check orientation
      boolean isLandscape = getResources().getConfiguration().orientation == 
                            android.content.res.Configuration.ORIENTATION_LANDSCAPE;
      
      // Get user height preference (same logic as calibration)
      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
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
  private int getUserKeyboardHeightPercent()
  {
    try {
      FoldStateTracker foldTracker = new FoldStateTracker(this);
      boolean foldableUnfolded = foldTracker.isUnfolded();
      boolean isLandscape = getResources().getConfiguration().orientation == 
                            android.content.res.Configuration.ORIENTATION_LANDSCAPE;
      
      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
      
      if (isLandscape) {
        String key = foldableUnfolded ? "keyboard_height_landscape_unfolded" : "keyboard_height_landscape";
        return prefs.getInt(key, 50);
      } else {
        String key = foldableUnfolded ? "keyboard_height_unfolded" : "keyboard_height";
        return prefs.getInt(key, 35);
      }
    } catch (Exception e) {
      return 35; // Default
    }
  }
  
  // Called by Keyboard2View when swipe typing completes
  public void handleSwipeTyping(List<KeyboardData.Key> swipedKeys,
                                List<android.graphics.PointF> swipePath,
                                List<Long> timestamps)
  {
    // v1.32.350: Delegated to InputCoordinator
    InputConnection ic = getCurrentInputConnection();
    EditorInfo editorInfo = getCurrentInputEditorInfo();
    Resources resources = getResources();
    _inputCoordinator.handleSwipeTyping(swipedKeys, swipePath, timestamps, ic, editorInfo, resources);
  }

  private View inflate_view(int layout)
  {
    return View.inflate(new ContextThemeWrapper(this, _config.theme), layout, null);
  }
  
  /**
   * CGR Prediction Integration Methods
   * These methods are called by the EnhancedSwipeGestureRecognizer to display predictions
   */
  
  /**
   * Update swipe predictions by checking keyboard view for CGR results
   */
  public void updateCGRPredictions()
  {
    if (_suggestionBar != null && _keyboardView != null)
    {
      List<String> cgrPredictions = _keyboardView.getCGRPredictions();
      if (!cgrPredictions.isEmpty())
      {
        _suggestionBar.setSuggestions(cgrPredictions);
      }
    }
  }
  
  /**
   * Check and update CGR predictions (call this periodically or on swipe events)
   */
  public void checkCGRPredictions()
  {
    if (_keyboardView != null && _suggestionBar != null)
    {
      // Enable always visible mode to prevent UI flickering
      _suggestionBar.setAlwaysVisible(true);
      
      List<String> cgrPredictions = _keyboardView.getCGRPredictions();
      boolean areFinal = _keyboardView.areCGRPredictionsFinal();
      
      if (!cgrPredictions.isEmpty())
      {
        _suggestionBar.setSuggestions(cgrPredictions);
      }
      else
      {
        // Show empty suggestions but keep bar visible
        _suggestionBar.setSuggestions(new ArrayList<>());
      }
    }
  }
  
  /**
   * Update swipe predictions in real-time during gesture (legacy method)
   */
  public void updateSwipePredictions(List<String> predictions)
  {
    if (_suggestionBar != null && predictions != null && !predictions.isEmpty())
    {
      _suggestionBar.setSuggestions(predictions);
    }
  }
  
  /**
   * Complete swipe predictions after gesture ends (legacy method)
   */
  public void completeSwipePredictions(List<String> finalPredictions)
  {
    if (_suggestionBar != null && finalPredictions != null && !finalPredictions.isEmpty())
    {
      _suggestionBar.setSuggestions(finalPredictions);
    }
    else
    {
    }
  }
  
  /**
   * Clear swipe predictions (legacy method)
   */
  public void clearSwipePredictions()
  {
    if (_suggestionBar != null)
    {
      // Don't actually clear - just show empty suggestions to keep bar visible
      _suggestionBar.setSuggestions(new ArrayList<>());
    }
  }

  /**
   * Extract key positions from keyboard layout and set them on neural engine.
   * CRITICAL for neural swipe typing - without this, key detection fails completely!
   */
  private void setNeuralKeyboardLayout()
  {
    if (_predictionCoordinator.getNeuralEngine() == null || _keyboardView == null)
    {
      android.util.Log.w("Keyboard2", "Cannot set neural layout - engine or view is null");
      return;
    }

    java.util.Map<Character, android.graphics.PointF> keyPositions = extractKeyPositionsFromLayout();

    if (keyPositions != null && !keyPositions.isEmpty())
    {
      _predictionCoordinator.getNeuralEngine().setRealKeyPositions(keyPositions);
      android.util.Log.d("Keyboard2", "Set " + keyPositions.size() + " key positions on neural engine");

      // Debug output only when debug mode is active
      if (_debugMode)
      {
        sendDebugLog(String.format(">>> Neural engine: %d key positions set\n", keyPositions.size()));

        // Log sample positions
        if (keyPositions.containsKey('q') && keyPositions.containsKey('a') && keyPositions.containsKey('z'))
        {
          android.graphics.PointF qPos = keyPositions.get('q');
          android.graphics.PointF aPos = keyPositions.get('a');
          android.graphics.PointF zPos = keyPositions.get('z');
          sendDebugLog(String.format(">>> Samples: q=(%.0f,%.0f) a=(%.0f,%.0f) z=(%.0f,%.0f)\n",
            qPos.x, qPos.y, aPos.x, aPos.y, zPos.x, zPos.y));
        }
      }
    }
    else
    {
      android.util.Log.e("Keyboard2", "Failed to extract key positions from layout");
    }
  }

  /**
   * Extract character key positions from the keyboard layout using reflection.
   * Returns a map of character -> center point (in pixels), or null on error.
   */
  private java.util.Map<Character, android.graphics.PointF> extractKeyPositionsFromLayout()
  {
    try
    {
      // Use reflection to access keyboard data from view
      java.lang.reflect.Field keyboardField = _keyboardView.getClass().getDeclaredField("_keyboard");
      keyboardField.setAccessible(true);
      KeyboardData keyboard = (KeyboardData) keyboardField.get(_keyboardView);

      if (keyboard == null)
      {
        android.util.Log.w("Keyboard2", "Keyboard data is null after reflection");
        return null;
      }

      // Get view dimensions
      float keyboardWidth = _keyboardView.getWidth();
      float keyboardHeight = _keyboardView.getHeight();

      if (keyboardWidth == 0 || keyboardHeight == 0)
      {
        android.util.Log.w("Keyboard2", "Keyboard dimensions are zero");
        return null;
      }

      // Calculate scale factors (layout units -> pixels)
      float scaleX = keyboardWidth / keyboard.keysWidth;
      float scaleY = keyboardHeight / keyboard.keysHeight;

      // Extract center positions of all character keys
      java.util.Map<Character, android.graphics.PointF> keyPositions = new java.util.HashMap<>();
      float currentY = 0;

      for (KeyboardData.Row row : keyboard.rows)
      {
        currentY += row.shift * scaleY;
        float centerY = currentY + (row.height * scaleY / 2.0f);
        float currentX = 0;

        for (KeyboardData.Key key : row.keys)
        {
          currentX += key.shift * scaleX;

          // Only process character keys
          if (key.keys != null && key.keys.length > 0 && key.keys[0] != null)
          {
            KeyValue kv = key.keys[0];
            if (kv.getKind() == KeyValue.Kind.Char)
            {
              char c = kv.getChar();
              float centerX = currentX + (key.width * scaleX / 2.0f);
              keyPositions.put(c, new android.graphics.PointF(centerX, centerY));
            }
          }

          currentX += key.width * scaleX;
        }

        currentY += row.height * scaleY;
      }

      return keyPositions;
    }
    catch (Exception e)
    {
      android.util.Log.e("Keyboard2", "Failed to extract key positions", e);
      return null;
    }
  }

  // Removed reloadCGRParameters method - causing crashes

  /**
   * Check if this keyboard is set as the default IME.
   * If not, show a non-intrusive notification to help user enable it.
   * Only shown once per app launch to avoid annoyance.
   */
  private void checkAndPromptDefaultIME()
  {
    try
    {
      // Get preference to track if we've already shown the prompt this session
      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
      boolean hasPromptedThisSession = prefs.getBoolean("ime_prompt_shown_this_session", false);

      if (hasPromptedThisSession)
      {
        return; // Already prompted, don't annoy the user
      }

      // Check if we're the default IME
      InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
      if (imm == null)
      {
        return;
      }

      String defaultIme = android.provider.Settings.Secure.getString(
          getContentResolver(),
          android.provider.Settings.Secure.DEFAULT_INPUT_METHOD
      );

      String ourIme = getPackageName() + "/" + getClass().getName();

      if (!ourIme.equals(defaultIme))
      {
        // We're not the default - show helpful toast
        _handler.postDelayed(new Runnable()
        {
          @Override
          public void run()
          {
            android.widget.Toast.makeText(
                Keyboard2.this,
                "Set Unexpected Keyboard as default in Settings → System → Languages & input → On-screen keyboard",
                android.widget.Toast.LENGTH_LONG
            ).show();
          }
        }, 2000); // Delay 2 seconds so it doesn't interfere with startup

        // Mark that we've shown the prompt this session
        prefs.edit().putBoolean("ime_prompt_shown_this_session", true).apply();
      }
    }
    catch (Exception e)
    {
      android.util.Log.e("Keyboard2", "Error checking default IME", e);
    }
  }

  // v1.32.341: loadContractionMappings() method removed - functionality moved to ContractionManager class
}
