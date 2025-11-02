package juloo.keyboard2;

import android.annotation.TargetApi;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
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
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
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
  SuggestionBar.OnSuggestionSelectedListener
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
  private ViewGroup _clipboard_pane = null;
  public int actionId; // Action performed by the Action key.
  private Handler _handler;

  private Config _config;

  private FoldStateTracker _foldStateTracker;
  
  // Swipe typing components
  private DictionaryManager _dictionaryManager;
  private WordPredictor _wordPredictor;
  private NeuralSwipeTypingEngine _neuralEngine;
  private AsyncPredictionHandler _asyncPredictionHandler;
  private SuggestionBar _suggestionBar;
  private LinearLayout _inputViewContainer;
  private StringBuilder _currentWord = new StringBuilder();
  private List<String> _contextWords = new ArrayList<>(); // Track previous words for context
  private BufferedWriter _logWriter = null;
  
  // ML data collection
  private SwipeMLDataStore _mlDataStore;
  private SwipeMLData _currentSwipeData;
  private boolean _wasLastInputSwipe = false;

  // Track auto-inserted word for replacement
  private String _lastAutoInsertedWord = null;

  // Track source of last committed text for context-aware deletion
  private PredictionSource _lastCommitSource = PredictionSource.UNKNOWN;

  // User adaptation
  private UserAdaptationManager _adaptationManager;

  // Contraction mappings for apostrophe insertion
  private Map<String, String> _nonPairedContractions = new HashMap<>();

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
    _foldStateTracker = new FoldStateTracker(this);
    Config.initGlobalConfig(prefs, getResources(), _keyeventhandler, _foldStateTracker.isUnfolded());
    prefs.registerOnSharedPreferenceChangeListener(this);
    _config = Config.globalConfig();

    // Check if we're the default IME and remind user if not
    checkAndPromptDefaultIME();
    _keyboardView = (Keyboard2View)inflate_view(R.layout.keyboard);
    _keyboardView.reset();
    Logs.set_debug_logs(getResources().getBoolean(R.bool.debug_logs));
    ClipboardHistoryService.on_startup(this, _keyeventhandler);
    _foldStateTracker.setChangedCallback(() -> { refresh_config(); });
    
    // Initialize ML data store
    _mlDataStore = SwipeMLDataStore.getInstance(this);
    
    // CGR parameter reloading will be handled through existing preference listener
    
    // Initialize user adaptation manager
    _adaptationManager = UserAdaptationManager.getInstance(this);

    // Load contraction mappings for apostrophe insertion
    loadContractionMappings();

    // KeyboardSwipeRecognizer is now handled through SwipeTypingEngine
    
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
    
    // Initialize word prediction components (for both swipe and regular typing)
    if (_config.word_prediction_enabled || _config.swipe_typing_enabled)
    {
      _dictionaryManager = new DictionaryManager(this);
      _dictionaryManager.setLanguage("en");
      _wordPredictor = new WordPredictor();
      _wordPredictor.setContext(this); // Enable disabled words filtering
      _wordPredictor.setConfig(_config);
      _wordPredictor.setUserAdaptationManager(_adaptationManager);
      _wordPredictor.loadDictionary(this, "en");
      
      // Initialize neural predictor for swipe typing only
      if (_config.swipe_typing_enabled)
      {
        _neuralEngine = new NeuralSwipeTypingEngine(this, _config);
        
        // Initialize async prediction handler
        _asyncPredictionHandler = new AsyncPredictionHandler(_neuralEngine);
        
        // CGR recognizer doesn't need separate calibration data loading
        
        // Note: Key positions will be set in onStartInputView after layout is complete
        android.util.Log.d("Keyboard2", "Neural engine initialized - dimensions and key positions will be set after layout");

        _keyboardView.setSwipeTypingComponents(_wordPredictor, this);
      }
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

    _foldStateTracker.close();

    // Cleanup async prediction handler
    if (_asyncPredictionHandler != null)
    {
      _asyncPredictionHandler.shutdown();
      _asyncPredictionHandler = null;
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
    int prev_theme = _config.theme;
    _config.refresh(getResources(), _foldStateTracker.isUnfolded());
    refreshSubtypeImm();
    
    // Update swipe engine config if it exists
    if (_neuralEngine != null)
    {
      _neuralEngine.setConfig(_config);
    }
    if (_wordPredictor != null)
    {
      _wordPredictor.setConfig(_config);
    }
    // Refreshing the theme config requires re-creating the views
    if (prev_theme != _config.theme)
    {
      _keyboardView = (Keyboard2View)inflate_view(R.layout.keyboard);
      _emojiPane = null;
      _clipboard_pane = null;
      setInputView(_keyboardView);
    }
    _keyboardView.reset();
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
    refresh_action_label(info);
    _currentSpecialLayout = refresh_special_layout(info);
    _keyboardView.setKeyboard(current_layout());
    _keyeventhandler.started(info);
    
    // Re-initialize word prediction components if settings have changed
    if (_config.word_prediction_enabled || _config.swipe_typing_enabled)
    {
      // Initialize predictors if not already initialized
      if (_wordPredictor == null)
      {
        _dictionaryManager = new DictionaryManager(this);
        _dictionaryManager.setLanguage("en");
        _wordPredictor = new WordPredictor();
        _wordPredictor.setContext(this); // Enable disabled words filtering
        _wordPredictor.setConfig(_config);
        _wordPredictor.loadDictionary(this, "en");
      }
      
      if (_config.swipe_typing_enabled && _neuralEngine == null)
      {
        _neuralEngine = new NeuralSwipeTypingEngine(this, _config);
        
        // Initialize async prediction handler
        _asyncPredictionHandler = new AsyncPredictionHandler(_neuralEngine);
        
        // CGR recognizer doesn't need separate calibration data loading
        
        // Set keyboard dimensions
        if (_keyboardView != null)
        {
          _neuralEngine.setKeyboardDimensions(_keyboardView.getWidth(), _keyboardView.getHeight());
        }
        
        _keyboardView.setSwipeTypingComponents(_wordPredictor, this);
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
        _inputViewContainer.addView(_keyboardView);
      }
      
      setInputView(_inputViewContainer != null ? _inputViewContainer : _keyboardView);
      
      // CRITICAL: Set correct keyboard dimensions for CGR after view is laid out
      if (_neuralEngine != null && _keyboardView != null) {
        _keyboardView.getViewTreeObserver().addOnGlobalLayoutListener(
          new android.view.ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
              // Ensure we have valid dimensions
              if (_keyboardView.getWidth() > 0 && _keyboardView.getHeight() > 0) {
                
                // Use dynamic keyboard dimensions based on user settings (like calibration)
                float keyboardWidth = _keyboardView.getWidth();
                float keyboardHeight = calculateDynamicKeyboardHeight();
                
                _neuralEngine.setKeyboardDimensions(keyboardWidth, keyboardHeight);

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
      _wordPredictor = null;
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
    refresh_config();
    _keyboardView.setKeyboard(current_layout());
    // Update suggestion bar opacity if it exists
    if (_suggestionBar != null)
    {
      _suggestionBar.setOpacity(_config.suggestion_bar_opacity);
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
          setInputView(_emojiPane);
          break;

        case SWITCH_CLIPBOARD:
          if (_clipboard_pane == null)
            _clipboard_pane = (ViewGroup)inflate_view(R.layout.clipboard_pane);
          setInputView(_clipboard_pane);
          break;

        case SWITCH_BACK_EMOJI:
        case SWITCH_BACK_CLIPBOARD:
          setInputView(_keyboardView);
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
      _wasLastInputSwipe = false;
      _currentSwipeData = null;
      handleRegularTyping(text);
    }
    
    public void handle_backspace()
    {
      Keyboard2.this.handleBackspace();
    }
  }

  private IBinder getConnectionToken()
  {
    return getWindow().getWindow().getAttributes().token;
  }
  
  // SuggestionBar.OnSuggestionSelectedListener implementation
  /**
   * Update context with a completed word
   */
  private void updateContext(String word)
  {
    if (word == null || word.isEmpty())
      return;
    
    // Add word to context
    _contextWords.add(word.toLowerCase());
    
    // Keep only last 2 words for bigram context
    while (_contextWords.size() > 2)
    {
      _contextWords.remove(0);
    }
    
    // Add word to WordPredictor for language detection
    if (_wordPredictor != null)
    {
      _wordPredictor.addWordToContext(word);
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
        if (_currentWord.length() > 0 && ic != null)
        {
          sendDebugLog(String.format("Manual typing in progress before swipe: \"%s\"\n", _currentWord));

          // IMPORTANT: Characters from manual typing are already committed via KeyEventHandler.send_text()
          // _currentWord is just a tracking buffer - the text is already in the editor!
          // We only need to add a space after the manually typed word and clear the tracking buffer
          ic.commitText(" ", 1);
          _currentWord = new StringBuilder();

          // Clear any previous auto-inserted word tracking since user was manually typing
          _lastAutoInsertedWord = null;
          _lastCommitSource = PredictionSource.USER_TYPED_TAP;
        }

        // DEBUG: Log auto-insertion
        sendDebugLog(String.format("Auto-inserting top prediction: \"%s\"\n", topPrediction));

        // CRITICAL: Clear auto-inserted tracking BEFORE calling onSuggestionSelected
        // This prevents the deletion logic from removing the previous auto-inserted word
        // For consecutive swipes, we want to APPEND words, not replace them
        _lastAutoInsertedWord = null;
        _lastCommitSource = PredictionSource.UNKNOWN; // Temporarily clear

        // onSuggestionSelected handles spacing logic (no space if first text, space otherwise)
        onSuggestionSelected(topPrediction);

        // NOW track this as auto-inserted so tapping another suggestion will replace ONLY this word
        // CRITICAL: Strip "raw:" prefix BEFORE storing (v1.33.7: fixed regex to match actual prefix format)
        String cleanPrediction = topPrediction.replaceAll("^raw:", "");
        _lastAutoInsertedWord = cleanPrediction;
        _lastCommitSource = PredictionSource.NEURAL_SWIPE;

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

    // Strip "raw:" prefix before processing (v1.33.7: fixed regex to match actual prefix format)
    // Prefix format: "raw:word" not " [raw:0.08]"
    word = word.replaceAll("^raw:", "");

    // Apply contraction mapping BEFORE final autocorrect
    // This converts apostrophe-free forms (wholl, dont) to proper contractions (who'll, don't)
    // CRITICAL: Must happen BEFORE autocorrect to prevent fuzzy matching
    if (_nonPairedContractions.containsKey(word.toLowerCase()))
    {
      String withApostrophe = _nonPairedContractions.get(word.toLowerCase());
      android.util.Log.d("Keyboard2", String.format("CONTRACTION MAPPING: \"%s\" → \"%s\"", word, withApostrophe));
      word = withApostrophe;
    }

    // v1.33.7: Final autocorrect - second chance autocorrect after beam search
    // Applies when user selects/auto-inserts a prediction (even if beam autocorrect was OFF)
    // Useful for correcting raw predictions or vocabulary misses
    if (_config.swipe_final_autocorrect_enabled && _wordPredictor != null)
    {
      String correctedWord = _wordPredictor.autoCorrect(word);

      // If autocorrect found a better match, use it
      if (!correctedWord.equals(word))
      {
        android.util.Log.d("Keyboard2", String.format("FINAL AUTOCORRECT: \"%s\" → \"%s\"", word, correctedWord));
        word = correctedWord;
      }
    }

    // Record user selection for adaptation learning
    if (_adaptationManager != null)
    {
      _adaptationManager.recordSelection(word.trim());
    }

    // CRITICAL: Save swipe flag before resetting for use in spacing logic below
    boolean isSwipeAutoInsert = _wasLastInputSwipe;

    // Store ML data if this was a swipe prediction selection
    if (isSwipeAutoInsert && _currentSwipeData != null && _mlDataStore != null)
    {
      // Create a new ML data object with the selected word
      android.util.DisplayMetrics metrics = getResources().getDisplayMetrics();
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
      _mlDataStore.storeSwipeData(mlData);
      
    }
    
    // Reset swipe tracking
    _wasLastInputSwipe = false;
    _currentSwipeData = null;
    
    InputConnection ic = getCurrentInputConnection();
    if (ic != null)
    {
      try
      {
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
        if (_lastAutoInsertedWord != null && !_lastAutoInsertedWord.isEmpty() &&
            _lastCommitSource == PredictionSource.NEURAL_SWIPE)
        {
          android.util.Log.d("Keyboard2", "REPLACE: Deleting auto-inserted word: '" + _lastAutoInsertedWord + "'");

          // Get text before cursor to understand what we're deleting
          CharSequence debugBefore = ic.getTextBeforeCursor(50, 0);
          android.util.Log.d("Keyboard2", "REPLACE: Text before cursor (50 chars): '" + debugBefore + "'");

          // Calculate how many characters to delete (word + space after it)
          // CRITICAL: Swipe words ALWAYS get a trailing space, even in Termux mode!
          // See line 1068: "For swipe typing, we always add trailing spaces even in Termux mode"
          int deleteCount = _lastAutoInsertedWord.length() + 1; // Word + trailing space

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

          // Clear the tracking variables
          _lastAutoInsertedWord = null;
          _lastCommitSource = PredictionSource.UNKNOWN;
        }
        // ALSO: If user is selecting a prediction during regular typing, delete the partial word
        // This handles typing "hel" then selecting "hello" - we need to delete "hel" first
        else if (_currentWord.length() > 0 && !isSwipeAutoInsert)
        {
          android.util.Log.d("Keyboard2", "TYPING PREDICTION: Deleting partial word: '" + _currentWord + "'");

          // Delete the characters that were typed (they're already in the editor)
          ic.deleteSurroundingText(_currentWord.length(), 0);

          CharSequence debugAfter = ic.getTextBeforeCursor(50, 0);
          android.util.Log.d("Keyboard2", "TYPING PREDICTION: After deleting partial, text before cursor: '" + debugAfter + "'");
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
        if (_lastCommitSource != PredictionSource.NEURAL_SWIPE)
        {
          _lastCommitSource = PredictionSource.CANDIDATE_SELECTION;
        }
      }
      catch (Exception e)
      {
      }

      // Update context with the selected word
      updateContext(word);

      // Clear current word
      // NOTE: Don't clear suggestions here - they're re-displayed after auto-insertion
      _currentWord.setLength(0);
    }
  }
  
  /**
   * Handle regular typing predictions (non-swipe)
   */
  public void handleRegularTyping(String text)
  {
    if (!_config.word_prediction_enabled || _wordPredictor == null || _suggestionBar == null)
    {
      return;
    }
      
    
    // Track current word being typed
    if (text.length() == 1 && Character.isLetter(text.charAt(0)))
    {
      _currentWord.append(text);
      updatePredictionsForCurrentWord();
    }
    else if (text.length() == 1 && !Character.isLetter(text.charAt(0)))
    {
      // Any non-letter character - update context and reset current word

      // If we had a word being typed, add it to context before clearing
      if (_currentWord.length() > 0)
      {
        String completedWord = _currentWord.toString();

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

        if (_config.autocorrect_enabled && _wordPredictor != null && text.equals(" ") && !inTermuxApp)
        {
          String correctedWord = _wordPredictor.autoCorrect(completedWord);

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
              _currentWord.setLength(0);

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
              if (_wordPredictor != null)
              {
                _wordPredictor.reset();
              }

              return; // Skip normal text processing - we've handled everything
            }
          }
        }

        updateContext(completedWord);
      }

      // Reset current word
      _currentWord.setLength(0);
      if (_wordPredictor != null)
      {
        _wordPredictor.reset();
      }
      if (_suggestionBar != null)
      {
        _suggestionBar.clearSuggestions();
      }
    }
    else if (text.length() > 1)
    {
      // Multi-character input (paste, etc) - reset
      _currentWord.setLength(0);
      if (_wordPredictor != null)
      {
        _wordPredictor.reset();
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
    if (_currentWord.length() > 0)
    {
      _currentWord.deleteCharAt(_currentWord.length() - 1);
      if (_currentWord.length() > 0)
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
    if (_currentWord.length() > 0)
    {
      String partial = _currentWord.toString();
      
      // Use contextual prediction
      WordPredictor.PredictionResult result = _wordPredictor.predictWordsWithContext(partial, _contextWords);
      
      if (!result.words.isEmpty() && _suggestionBar != null)
      {
        _suggestionBar.setShowDebugScores(_config.swipe_show_debug_scores);
        _suggestionBar.setSuggestionsWithScores(result.words, result.scores);
      }
    }
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
    // Clear auto-inserted word tracking when new swipe starts
    _lastAutoInsertedWord = null;

    // DEBUG: Log swipe start
    sendDebugLog("\n========== NEW SWIPE ==========\n");
    sendDebugLog(String.format("Path points: %d, Keys detected: %d\n",
        swipePath != null ? swipePath.size() : 0,
        swipedKeys != null ? swipedKeys.size() : 0));

    // DEBUG: Log keyboard dimensions and first/last path points
    if (_keyboardView != null && swipePath != null && swipePath.size() > 0)
    {
      sendDebugLog(String.format("Keyboard dimensions: %dx%d\n",
          _keyboardView.getWidth(), _keyboardView.getHeight()));
      android.graphics.PointF first = swipePath.get(0);
      android.graphics.PointF last = swipePath.get(swipePath.size() - 1);
      sendDebugLog(String.format("Path: (%.1f, %.1f) → (%.1f, %.1f)\n",
          first.x, first.y, last.x, last.y));

      // Calculate and log sampling rate
      if (timestamps != null && timestamps.size() > 1)
      {
        long totalTime = timestamps.get(timestamps.size() - 1) - timestamps.get(0);
        float samplingHz = (timestamps.size() - 1) * 1000.0f / totalTime;
        sendDebugLog(String.format("Sampling rate: %.1f Hz (%.0fms total)\n",
            samplingHz, (float)totalTime));
      }
    }

    if (!_config.swipe_typing_enabled)
    {
      return;
    }
    
    if (_neuralEngine == null)
    {
      // Fallback to word predictor if engine not initialized
      if (_wordPredictor == null)
      {
        return;
      }
      // Initialize engine on the fly
      _neuralEngine = new NeuralSwipeTypingEngine(this, _config);

      // Initialize async handler if not already done
      if (_asyncPredictionHandler == null)
      {
        _asyncPredictionHandler = new AsyncPredictionHandler(_neuralEngine);
      }

      // Neural engine dimensions and key positions already set in onStartInputView
    }
    
    // Mark that last input was a swipe for ML data collection
    _wasLastInputSwipe = true;
    
    // Prepare ML data (will be saved if user selects a prediction)
    android.util.DisplayMetrics metrics = getResources().getDisplayMetrics();
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

    // DEBUG: Log gesture tracker's detection (for comparison)
    sendDebugLog(String.format("Gesture tracker keys: \"%s\" (%d keys filtered from %d path points)\n",
        gestureTrackerKeys.toString(), swipedKeys.size(), swipePath != null ? swipePath.size() : 0));

    // Log to file for analysis
    if (_logWriter != null && gestureTrackerKeys.length() > 0)
    {
      try
      {
        _logWriter.write("[" + new java.util.Date() + "] Swipe: " + gestureTrackerKeys.toString() + "\n");
        _logWriter.flush();
      }
      catch (IOException e)
      {
      }
    }

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
      if (_asyncPredictionHandler != null)
      {
        _asyncPredictionHandler.cancelPendingPredictions();
      }
      
      // Request predictions asynchronously - always done on gesture completion
      // which matches the calibration activity's ACTION_UP behavior
      if (_asyncPredictionHandler != null)
      {
        _asyncPredictionHandler.requestPredictions(swipeInput, new AsyncPredictionHandler.PredictionCallback()
        {
          @Override
          public void onPredictionsReady(List<String> predictions, List<Integer> scores)
          {
            // Process predictions on UI thread
            handlePredictionResults(predictions, scores);
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
        PredictionResult result = _neuralEngine.predict(swipeInput);
      long predictionTime = System.currentTimeMillis() - startTime;
      List<String> predictions = result.words;
      
      if (predictions.size() > 0)
      {
      }
      else
      {
      }
      
      // Log predictions to file
      if (_logWriter != null)
      {
        try
        {
          _logWriter.write("  Predictions: " + predictions + " (" + predictionTime + "ms)\n");
          _logWriter.write("  Scores: " + result.scores + "\n");
          _logWriter.flush();
        }
        catch (IOException e)
        {
        }
      }
      
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
    if (_neuralEngine == null || _keyboardView == null)
    {
      android.util.Log.w("Keyboard2", "Cannot set neural layout - engine or view is null");
      return;
    }

    java.util.Map<Character, android.graphics.PointF> keyPositions = extractKeyPositionsFromLayout();

    if (keyPositions != null && !keyPositions.isEmpty())
    {
      _neuralEngine.setRealKeyPositions(keyPositions);
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

  /**
   * Load contraction mappings from JSON file.
   * Maps apostrophe-free forms to their proper contraction forms.
   * Example: "dont" -> "don't", "wholl" -> "who'll"
   */
  private void loadContractionMappings()
  {
    try
    {
      java.io.InputStream inputStream = getAssets().open("dictionaries/contractions_non_paired.json");
      java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.InputStreamReader(inputStream));
      StringBuilder jsonBuilder = new StringBuilder();
      String line;
      while ((line = reader.readLine()) != null)
      {
        jsonBuilder.append(line);
      }
      reader.close();

      // Parse JSON and populate map
      org.json.JSONObject jsonObj = new org.json.JSONObject(jsonBuilder.toString());
      java.util.Iterator<String> keys = jsonObj.keys();
      while (keys.hasNext())
      {
        String withoutApostrophe = keys.next();
        String withApostrophe = jsonObj.getString(withoutApostrophe);
        _nonPairedContractions.put(withoutApostrophe.toLowerCase(), withApostrophe.toLowerCase());
      }

      Log.d("Keyboard2", "Loaded " + _nonPairedContractions.size() + " contraction mappings");
    }
    catch (Exception e)
    {
      Log.e("Keyboard2", "Failed to load contraction mappings", e);
    }
  }
}
