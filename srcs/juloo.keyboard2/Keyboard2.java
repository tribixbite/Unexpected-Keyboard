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

  // Subtype management (v1.32.365: extracted to SubtypeManager)
  private SubtypeManager _subtypeManager;

  // Event handling (v1.32.368: extracted to KeyboardReceiver)
  private KeyboardReceiver _receiver;

  // KeyEventHandler bridge (v1.32.390: extracted to KeyEventReceiverBridge)
  private KeyEventReceiverBridge _receiverBridge;

  // ML data collection (v1.32.370: extracted to MLDataCollector)
  private MLDataCollector _mlDataCollector;

  // Debug logging management (v1.32.384: extracted to DebugLoggingManager)
  private DebugLoggingManager _debugLoggingManager;

  // Config propagation (v1.32.386: extracted to ConfigPropagator)
  private ConfigPropagator _configPropagator;

  // Suggestion/prediction bridge (v1.32.406: extracted to SuggestionBridge)
  private SuggestionBridge _suggestionBridge;

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

    // Create bridge for KeyEventHandler to KeyboardReceiver delegation (v1.32.390)
    // Receiver will be initialized later and set on the bridge
    _receiverBridge = KeyEventReceiverBridge.create(this, _handler);
    _keyeventhandler = new KeyEventHandler(_receiverBridge);

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

    // Initialize all managers (v1.32.388: extracted to ManagerInitializer)
    ManagerInitializer.InitializationResult managers =
        ManagerInitializer.create(this, _config, _keyboardView, _keyeventhandler).initialize();

    _contractionManager = managers.getContractionManager();
    _clipboardManager = managers.getClipboardManager();
    _contextTracker = managers.getContextTracker();
    _predictionCoordinator = managers.getPredictionCoordinator();
    _inputCoordinator = managers.getInputCoordinator();
    _suggestionHandler = managers.getSuggestionHandler();
    _neuralLayoutHelper = managers.getNeuralLayoutHelper();
    _mlDataCollector = managers.getMlDataCollector();

    // Initialize suggestion bridge (v1.32.406: extracted to SuggestionBridge)
    _suggestionBridge = SuggestionBridge.create(
      this,
      _suggestionHandler,
      _mlDataCollector,
      _inputCoordinator,
      _contextTracker,
      _predictionCoordinator,
      _keyboardView
    );

    // Initialize prediction components if enabled (v1.32.405: extracted to PredictionInitializer)
    PredictionInitializer.create(_config, _predictionCoordinator, _keyboardView, this)
        .initializeIfEnabled();

    // Initialize debug logging manager (v1.32.384)
    _debugLoggingManager = new DebugLoggingManager(this, getPackageName());
    _debugLoggingManager.initializeLogWriter();

    // Initialize propagators (v1.32.396: extracted propagator initialization)
    // Creates and registers DebugModePropagator, builds ConfigPropagator with all managers
    PropagatorInitializer.InitializationResult propagators = PropagatorInitializer.create(
      _suggestionHandler,
      _neuralLayoutHelper,
      _debugLoggerImpl,
      _debugLoggingManager,
      _clipboardManager,
      _predictionCoordinator,
      _inputCoordinator,
      _layoutManager,
      _keyboardView,
      _subtypeManager
    ).initialize();

    _configPropagator = propagators.getConfigPropagator();

    // Register broadcast receiver for debug mode control (v1.32.384: delegated to DebugLoggingManager)
    _debugLoggingManager.registerDebugModeReceiver(this);
  }

  @Override
  public void onDestroy() {
    super.onDestroy();

    // Cleanup all managers (v1.32.404: extracted to CleanupHandler)
    CleanupHandler.create(
      this,
      _configManager,
      _clipboardManager,
      _predictionCoordinator,
      _debugLoggingManager
    ).cleanup();
  }

  /**
   * Send debug log message to SwipeDebugActivity if debug mode is enabled.
   * (v1.32.384: Delegated to DebugLoggingManager)
   */
  private void sendDebugLog(String message)
  {
    if (_debugLoggingManager != null)
    {
      _debugLoggingManager.sendDebugLog(message);
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

  /**
   * Gets InputMethodManager.
   * (v1.32.365: Delegated to SubtypeManager)
   */
  InputMethodManager get_imm()
  {
    return _subtypeManager.getInputMethodManager();
  }

  /**
   * Refreshes IME subtype settings and initializes managers.
   * (v1.32.365: Simplified by delegating to SubtypeManager)
   */
  private void refreshSubtypeImm()
  {
    // Initialize SubtypeManager if needed (v1.32.365)
    if (_subtypeManager == null)
    {
      _subtypeManager = new SubtypeManager(this);
    }

    // Refresh subtype and get default layout (v1.32.365: delegated to SubtypeManager)
    KeyboardData default_layout = _subtypeManager.refreshSubtype(_config, getResources());
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

  /**
   * Refresh action label configuration from EditorInfo.
   *
   * v1.32.379: EditorInfo parsing extracted to EditorInfoHelper (Kotlin).
   * Extracts action label, action ID, and Enter/Action key swap behavior.
   */
  private void refresh_action_label(EditorInfo info)
  {
    EditorInfoHelper.EditorActionInfo actionInfo =
        EditorInfoHelper.extractActionInfo(info, getResources());

    _config.actionLabel = actionInfo.getActionLabel();
    actionId = actionInfo.getActionId();
    _config.swapEnterActionKey = actionInfo.getSwapEnterActionKey();
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

    // Propagate config to all managers (v1.32.386: delegated to ConfigPropagator)
    if (_configPropagator != null)
    {
      _configPropagator.propagateConfig(_config, getResources());
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

    // Initialize KeyboardReceiver if needed (v1.32.397: extracted to ReceiverInitializer)
    // Lazy initialization: creates receiver on first call, returns existing on subsequent calls
    _receiver = ReceiverInitializer.create(
      this,
      this,
      _keyboardView,
      _layoutManager,
      _clipboardManager,
      _contextTracker,
      _inputCoordinator,
      _subtypeManager,
      _handler,
      _receiverBridge
    ).initializeIfNeeded(_receiver);

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

    // Setup prediction views (v1.32.400: extracted prediction/swipe setup logic)
    // Handles initialization, suggestion bar creation, neural engine dimensions, and cleanup
    PredictionViewSetup.SetupResult predictionSetup = PredictionViewSetup.create(
      this,
      _config,
      _keyboardView,
      _predictionCoordinator,
      _inputCoordinator,
      _suggestionHandler,
      _neuralLayoutHelper,
      _receiver,
      _emojiPane
    ).setupPredictionViews(_suggestionBar, _inputViewContainer, _contentPaneContainer);

    // Update components from setup result
    _suggestionBar = predictionSetup.getSuggestionBar();
    _inputViewContainer = predictionSetup.getInputViewContainer();
    _contentPaneContainer = predictionSetup.getContentPaneContainer();
    setInputView(predictionSetup.getInputView());
    
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

  /**
   * Updates soft input window layout parameters for IME.
   *
   * v1.32.375: Window layout management extracted to WindowLayoutUtils (Kotlin).
   * Configures edge-to-edge display, window height, input area height, and gravity.
   */
  private void updateSoftInputWindowLayoutParams() {
    final Window window = getWindow().getWindow();
    final View inputArea = window.findViewById(android.R.id.inputArea);
    WindowLayoutUtils.updateSoftInputWindowLayoutParams(window, inputArea, isFullscreenMode());
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
  // v1.32.368: Receiver inner class removed - functionality moved to KeyboardReceiver class

  /**
   * Gets connection token for IME operations.
   * (v1.32.368: Made public for KeyboardReceiver)
   */
  public IBinder getConnectionToken()
  {
    return getWindow().getWindow().getAttributes().token;
  }

  /**
   * Gets current configuration.
   * (v1.32.368: Added for KeyboardReceiver)
   */
  public Config getConfig()
  {
    return _config;
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
   * (v1.32.406: Delegated to SuggestionBridge)
   */
  private void handlePredictionResults(List<String> predictions, List<Integer> scores)
  {
    _suggestionBridge.handlePredictionResults(predictions, scores);
  }
  
  /**
   * Called when user selects a suggestion from suggestion bar.
   * (v1.32.370: ML data collection delegated to MLDataCollector)
   * (v1.32.406: Delegated to SuggestionBridge)
   */
  @Override
  public void onSuggestionSelected(String word)
  {
    _suggestionBridge.onSuggestionSelected(word);
  }
  
  /**
   * Handle regular typing predictions (non-swipe)
   * (v1.32.361: Delegated to SuggestionHandler)
   * (v1.32.406: Delegated to SuggestionBridge)
   */
  public void handleRegularTyping(String text)
  {
    _suggestionBridge.handleRegularTyping(text);
  }
  
  /**
   * Handle backspace for prediction tracking
   * (v1.32.361: Delegated to SuggestionHandler)
   * (v1.32.406: Delegated to SuggestionBridge)
   */
  public void handleBackspace()
  {
    _suggestionBridge.handleBackspace();
  }
  
  // v1.32.361: updatePredictionsForCurrentWord() method removed - functionality moved to SuggestionHandler class

  /**
   * Smart delete last word - deletes the last auto-inserted word or last typed word.
   * (v1.32.361: Delegated to SuggestionHandler)
   * (v1.32.406: Delegated to SuggestionBridge)
   */
  public void handleDeleteLastWord()
  {
    _suggestionBridge.handleDeleteLastWord();
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

  /**
   * Inflates a view with the current theme.
   * (v1.32.368: Made public for KeyboardReceiver)
   */
  public View inflate_view(int layout)
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
   *
   * v1.32.377: IME status checking extracted to IMEStatusHelper (Kotlin).
   */
  private void checkAndPromptDefaultIME()
  {
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
    IMEStatusHelper.checkAndPromptDefaultIME(
        this, _handler, prefs, getPackageName(), getClass().getName()
    );
  }

  // v1.32.341: loadContractionMappings() method removed - functionality moved to ContractionManager class
}
