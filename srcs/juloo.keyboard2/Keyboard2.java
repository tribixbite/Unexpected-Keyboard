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

  // Layout management (v1.32.363: extracted to LayoutManager)
  private LayoutManager _layoutManager;

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

  // Suggestion handling (v1.32.361: extracted to SuggestionHandler)
  private SuggestionHandler _suggestionHandler;

  // Neural layout helper (v1.32.362: extracted to NeuralLayoutHelper)
  private NeuralLayoutHelper _neuralLayoutHelper;

  // Debug mode for swipe pipeline logging
  private boolean _debugMode = false;
  private android.content.BroadcastReceiver _debugModeReceiver;

  /**
   * Layout currently visible before it has been modified.
   * (v1.32.363: Delegated to LayoutManager)
   */
  KeyboardData current_layout_unmodified()
  {
    return _layoutManager.current_layout_unmodified();
  }

  /**
   * Layout currently visible.
   * (v1.32.363: Delegated to LayoutManager)
   */
  KeyboardData current_layout()
  {
    return _layoutManager.current_layout();
  }

  /**
   * Set text layout by index.
   * (v1.32.363: Delegated to LayoutManager)
   */
  void setTextLayout(int l)
  {
    KeyboardData layout = _layoutManager.setTextLayout(l);
    _keyboardView.setKeyboard(layout);
  }

  /**
   * Cycle to next/previous text layout.
   * (v1.32.363: Delegated to LayoutManager)
   */
  void incrTextLayout(int delta)
  {
    KeyboardData layout = _layoutManager.incrTextLayout(delta);
    _keyboardView.setKeyboard(layout);
  }

  /**
   * Set special layout (numeric, emoji, etc.).
   * (v1.32.363: Delegated to LayoutManager)
   */
  void setSpecialLayout(KeyboardData l)
  {
    KeyboardData layout = _layoutManager.setSpecialLayout(l);
    _keyboardView.setKeyboard(layout);
  }

  /**
   * Load a layout from resources.
   * (v1.32.363: Delegated to LayoutManager)
   */
  KeyboardData loadLayout(int layout_id)
  {
    return _layoutManager.loadLayout(layout_id);
  }

  /**
   * Load a layout that contains a numpad.
   * (v1.32.363: Delegated to LayoutManager)
   */
  KeyboardData loadNumpad(int layout_id)
  {
    return _layoutManager.loadNumpad(layout_id);
  }

  /**
   * Load a pinentry layout.
   * (v1.32.363: Delegated to LayoutManager)
   */
  KeyboardData loadPinentry(int layout_id)
  {
    return _layoutManager.loadPinentry(layout_id);
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

    // Initialize suggestion handler (v1.32.361: extracted suggestion/prediction logic)
    _suggestionHandler = new SuggestionHandler(
      this,
      _config,
      _contextTracker,
      _predictionCoordinator,
      _contractionManager,
      _keyeventhandler
    );

    // Initialize neural layout helper (v1.32.362: extracted neural/layout utility methods)
    _neuralLayoutHelper = new NeuralLayoutHelper(
      this,
      _config,
      _predictionCoordinator
    );
    _neuralLayoutHelper.setKeyboardView(_keyboardView);

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

          // Propagate debug mode to SuggestionHandler (v1.32.361)
          if (_suggestionHandler != null)
          {
            _suggestionHandler.setDebugMode(_debugMode, _debugLoggerImpl);
          }

          // Propagate debug mode to NeuralLayoutHelper (v1.32.362)
          if (_neuralLayoutHelper != null)
          {
            _neuralLayoutHelper.setDebugMode(_debugMode, new NeuralLayoutHelper.DebugLogger()
            {
              @Override
              public void sendDebugLog(String message)
              {
                Keyboard2.this.sendDebugLog(message);
              }
            });
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

  /**
   * DebugLogger implementation for SuggestionHandler.
   */
  private final SuggestionHandler.DebugLogger _debugLoggerImpl = new SuggestionHandler.DebugLogger()
  {
    @Override
    public void sendDebugLog(String message)
    {
      Keyboard2.this.sendDebugLog(message);
    }
  };

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
      default_layout = KeyboardData.load(getResources(), R.xml.latn_qwerty_us);

    // Set locale layout on LayoutManager (v1.32.363)
    if (_layoutManager != null)
    {
      _layoutManager.setLocaleTextLayout(default_layout);
    }
    else
    {
      // First call - initialize LayoutManager with default layout
      _layoutManager = new LayoutManager(this, _config, default_layout);
    }
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

    // Update suggestion handler config (v1.32.361)
    if (_suggestionHandler != null)
    {
      _suggestionHandler.setConfig(_config);
    }

    // Update neural layout helper config (v1.32.362)
    if (_neuralLayoutHelper != null)
    {
      _neuralLayoutHelper.setConfig(_config);
    }

    // Update layout manager config (v1.32.363)
    if (_layoutManager != null)
    {
      _layoutManager.setConfig(_config);
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

  /**
   * Determine special layout based on input type.
   * (v1.32.363: Delegated to LayoutManager)
   */
  private KeyboardData refresh_special_layout(EditorInfo info)
  {
    return _layoutManager.refresh_special_layout(info);
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

    // Set special layout if needed (v1.32.363: use LayoutManager)
    KeyboardData specialLayout = refresh_special_layout(info);
    if (specialLayout != null)
    {
      _layoutManager.setSpecialLayout(specialLayout);
    }
    else
    {
      _layoutManager.clearSpecialLayout();
    }

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

        // Update SuggestionHandler with suggestion bar reference (v1.32.361)
        if (_suggestionHandler != null)
        {
          _suggestionHandler.setSuggestionBar(_suggestionBar);
        }

        // Update NeuralLayoutHelper with suggestion bar reference (v1.32.362)
        if (_neuralLayoutHelper != null)
        {
          _neuralLayoutHelper.setSuggestionBar(_suggestionBar);
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
          _keyboardView.setKeyboard(_layoutManager.clearSpecialLayout());
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
   * (v1.32.361: Delegated to SuggestionHandler)
   *
   * NOTE: This is a legacy helper method. New code should use
   * _contextTracker.commitWord() directly with appropriate PredictionSource.
   */
  private void updateContext(String word)
  {
    if (_suggestionHandler != null)
    {
      _suggestionHandler.updateContext(word);
    }
  }
  
  /**
   * Handle prediction results from async prediction handler
   * (v1.32.361: Delegated to SuggestionHandler)
   */
  private void handlePredictionResults(List<String> predictions, List<Integer> scores)
  {
    if (_suggestionHandler != null)
    {
      InputConnection ic = getCurrentInputConnection();
      EditorInfo editorInfo = getCurrentInputEditorInfo();
      Resources resources = getResources();
      _suggestionHandler.handlePredictionResults(predictions, scores, ic, editorInfo, resources);
    }
  }
  
  /**
   * Called when user selects a suggestion from suggestion bar.
   * (v1.32.361: Delegated to SuggestionHandler, but handles ML data collection here)
   */
  @Override
  public void onSuggestionSelected(String word)
  {
    // Store ML data if this was a swipe prediction selection (done in Keyboard2 not SuggestionHandler)
    boolean isSwipeAutoInsert = _contextTracker.wasLastInputSwipe();
    SwipeMLData currentSwipeData = _inputCoordinator.getCurrentSwipeData();
    if (isSwipeAutoInsert && currentSwipeData != null && _predictionCoordinator.getMlDataStore() != null)
    {
      // Strip "raw:" prefix before storing ML data
      String cleanWord = word.replaceAll("^raw:", "");

      // Create a new ML data object with the selected word
      android.util.DisplayMetrics metrics = getResources().getDisplayMetrics();
      SwipeMLData mlData = new SwipeMLData(cleanWord, "user_selection",
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

    // Reset swipe data after ML collection
    _inputCoordinator.resetSwipeData();

    // Delegate to SuggestionHandler
    if (_suggestionHandler != null)
    {
      InputConnection ic = getCurrentInputConnection();
      EditorInfo editorInfo = getCurrentInputEditorInfo();
      Resources resources = getResources();
      _suggestionHandler.onSuggestionSelected(word, ic, editorInfo, resources);
    }
  }
  
  /**
   * Handle regular typing predictions (non-swipe)
   * (v1.32.361: Delegated to SuggestionHandler)
   */
  public void handleRegularTyping(String text)
  {
    if (_suggestionHandler != null)
    {
      InputConnection ic = getCurrentInputConnection();
      EditorInfo editorInfo = getCurrentInputEditorInfo();
      _suggestionHandler.handleRegularTyping(text, ic, editorInfo);
    }
  }
  
  /**
   * Handle backspace for prediction tracking
   * (v1.32.361: Delegated to SuggestionHandler)
   */
  public void handleBackspace()
  {
    if (_suggestionHandler != null)
    {
      _suggestionHandler.handleBackspace();
    }
  }
  
  // v1.32.361: updatePredictionsForCurrentWord() method removed - functionality moved to SuggestionHandler class

  /**
   * Smart delete last word - deletes the last auto-inserted word or last typed word.
   * (v1.32.361: Delegated to SuggestionHandler)
   */
  public void handleDeleteLastWord()
  {
    if (_suggestionHandler != null)
    {
      InputConnection ic = getCurrentInputConnection();
      EditorInfo editorInfo = getCurrentInputEditorInfo();
      _suggestionHandler.handleDeleteLastWord(ic, editorInfo);
    }
  }

  /**
   * Calculate dynamic keyboard height based on user settings.
   * (v1.32.362: Delegated to NeuralLayoutHelper)
   */
  private float calculateDynamicKeyboardHeight()
  {
    if (_neuralLayoutHelper != null)
    {
      return _neuralLayoutHelper.calculateDynamicKeyboardHeight();
    }
    return _keyboardView != null ? _keyboardView.getHeight() : 0;
  }

  /**
   * Get user keyboard height percentage for logging.
   * (v1.32.362: Delegated to NeuralLayoutHelper)
   */
  private int getUserKeyboardHeightPercent()
  {
    if (_neuralLayoutHelper != null)
    {
      return _neuralLayoutHelper.getUserKeyboardHeightPercent();
    }
    return 35; // Default
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
   * (v1.32.362: Delegated to NeuralLayoutHelper)
   */

  /**
   * Update swipe predictions by checking keyboard view for CGR results.
   */
  public void updateCGRPredictions()
  {
    if (_neuralLayoutHelper != null)
    {
      _neuralLayoutHelper.updateCGRPredictions();
    }
  }

  /**
   * Check and update CGR predictions (call this periodically or on swipe events).
   */
  public void checkCGRPredictions()
  {
    if (_neuralLayoutHelper != null)
    {
      _neuralLayoutHelper.checkCGRPredictions();
    }
  }

  /**
   * Update swipe predictions in real-time during gesture (legacy method).
   */
  public void updateSwipePredictions(List<String> predictions)
  {
    if (_neuralLayoutHelper != null)
    {
      _neuralLayoutHelper.updateSwipePredictions(predictions);
    }
  }

  /**
   * Complete swipe predictions after gesture ends (legacy method).
   */
  public void completeSwipePredictions(List<String> finalPredictions)
  {
    if (_neuralLayoutHelper != null)
    {
      _neuralLayoutHelper.completeSwipePredictions(finalPredictions);
    }
  }

  /**
   * Clear swipe predictions (legacy method).
   */
  public void clearSwipePredictions()
  {
    if (_neuralLayoutHelper != null)
    {
      _neuralLayoutHelper.clearSwipePredictions();
    }
  }

  /**
   * Extract key positions from keyboard layout and set them on neural engine.
   * CRITICAL for neural swipe typing - without this, key detection fails completely!
   * (v1.32.362: Delegated to NeuralLayoutHelper)
   */
  private void setNeuralKeyboardLayout()
  {
    if (_neuralLayoutHelper != null)
    {
      _neuralLayoutHelper.setNeuralKeyboardLayout();
    }
  }

  // v1.32.362: extractKeyPositionsFromLayout() method removed - functionality moved to NeuralLayoutHelper class

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
