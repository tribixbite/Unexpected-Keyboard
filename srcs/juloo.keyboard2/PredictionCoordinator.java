package juloo.keyboard2;

import android.content.Context;
import android.util.Log;
import juloo.keyboard2.ml.SwipeMLDataStore;
import java.util.List;

/**
 * Coordinates prediction engines and manages prediction lifecycle.
 *
 * This class centralizes the management of:
 * - DictionaryManager (dictionary loading and management)
 * - WordPredictor (typing predictions and context)
 * - NeuralSwipeTypingEngine (swipe typing ML model)
 * - AsyncPredictionHandler (asynchronous prediction processing)
 *
 * Responsibilities:
 * - Initialize and configure prediction engines
 * - Coordinate predictions from multiple sources
 * - Manage engine lifecycle (shutdown, cleanup)
 * - Provide unified interface for prediction requests
 *
 * NOT included (remains in Keyboard2):
 * - SuggestionBar UI integration
 * - InputConnection text insertion
 * - Auto-insertion logic
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.346).
 */
public class PredictionCoordinator
{
  private static final String TAG = "PredictionCoordinator";

  private final Context _context;
  private Config _config;

  // Prediction engines
  private DictionaryManager _dictionaryManager;
  private WordPredictor _wordPredictor;
  private NeuralSwipeTypingEngine _neuralEngine;
  private AsyncPredictionHandler _asyncPredictionHandler;
  private volatile boolean _isInitializingNeuralEngine = false; // v1.32.529: Track initialization state

  // Supporting services
  private SwipeMLDataStore _mlDataStore;
  private UserAdaptationManager _adaptationManager;

  // Debug logging
  private NeuralSwipeTypingEngine.DebugLogger _debugLogger;

  /**
   * Creates a new PredictionCoordinator.
   *
   * @param context Android context for resource access
   * @param config Configuration instance
   */
  public PredictionCoordinator(Context context, Config config)
  {
    _context = context;
    _config = config;
  }

  /**
   * Initializes prediction engines based on configuration.
   * Should be called during keyboard startup.
   */
  public void initialize()
  {
    // Initialize ML data store
    _mlDataStore = SwipeMLDataStore.getInstance(_context);

    // Initialize user adaptation manager
    _adaptationManager = UserAdaptationManager.getInstance(_context);

    // Initialize dictionary manager and word predictor
    initializeWordPredictor();

    // Initialize neural engine if swipe typing is enabled
    if (_config.swipe_typing_enabled)
    {
      // FIX: Initialize neural engine on background thread to avoid blocking startup
      new Thread(() -> {
          initializeNeuralEngine();
      }).start();
    }
  }

  /**
   * Initializes word predictor for typing predictions.
   */
  private void initializeWordPredictor()
  {
    _dictionaryManager = new DictionaryManager(_context);
    _dictionaryManager.setLanguage("en");

    _wordPredictor = new WordPredictor();
    _wordPredictor.setContext(_context); // Enable disabled words filtering
    _wordPredictor.setConfig(_config);
    _wordPredictor.setUserAdaptationManager(_adaptationManager);
    
    // FIX: Load dictionary asynchronously to prevent Main Thread blocking during startup
    // This prevents ANRs when the keyboard initializes
    Log.d(TAG, "Starting async dictionary loading...");
    _wordPredictor.loadDictionaryAsync(_context, "en", () -> {
        Log.d(TAG, "Dictionary loaded successfully");
    });

    // OPTIMIZATION: Start observing dictionary changes for automatic updates
    _wordPredictor.startObservingDictionaryChanges();

    Log.d(TAG, "WordPredictor initialized with automatic update observation");
  }

  /**
   * Initializes neural engine for swipe typing.
   * OPTIMIZATION v1.32.529: Removed synchronized as it's now protected by double-checked locking
   * in ensureNeuralEngineReady and initialize
   */
  private void initializeNeuralEngine()
  {
    // Skip if already initialized or initializing
    if (_neuralEngine != null || _isInitializingNeuralEngine)
    {
      return;
    }

    try
    {
      _isInitializingNeuralEngine = true;
      _neuralEngine = new NeuralSwipeTypingEngine(_context, _config);

      // Set debug logger before initialization so logs appear during model loading
      if (_debugLogger != null)
      {
        _neuralEngine.setDebugLogger(_debugLogger);
        Log.d(TAG, "Debug logger set on neural engine");
      }

      // CRITICAL: Call initialize() to actually load the ONNX models
      boolean success = _neuralEngine.initialize();
      if (!success)
      {
        Log.e(TAG, "Neural engine initialization returned false");
        _neuralEngine = null;
        _asyncPredictionHandler = null;
        return;
      }

      // Initialize async prediction handler
      _asyncPredictionHandler = new AsyncPredictionHandler(_neuralEngine);

      Log.d(TAG, "NeuralSwipeTypingEngine initialized successfully");
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to initialize neural engine", e);
      _neuralEngine = null;
      _asyncPredictionHandler = null;
    }
    finally
    {
      _isInitializingNeuralEngine = false;
    }
  }

  /**
   * Ensures neural engine is initialized before use.
   * OPTIMIZATION v1.32.529: Double-checked locking to prevent Main Thread blocking
   * This allows check to be fast if already initialized, without acquiring lock.
   */
  public void ensureNeuralEngineReady()
  {
    if (_config.swipe_typing_enabled && _neuralEngine == null && !_isInitializingNeuralEngine)
    {
      synchronized(this)
      {
        if (_neuralEngine == null && !_isInitializingNeuralEngine)
        {
          Log.d(TAG, "Lazy-loading neural engine on first swipe...");
          initializeNeuralEngine();
        }
      }
    }
  }

  /**
   * Sets the debug logger for neural engine logging.
   * Should be called before initialize() for model loading logs.
   *
   * @param logger Debug logger implementation that sends to SwipeDebugActivity
   */
  public void setDebugLogger(NeuralSwipeTypingEngine.DebugLogger logger)
  {
    _debugLogger = logger;

    // Also set on existing engine if already initialized
    if (_neuralEngine != null)
    {
      _neuralEngine.setDebugLogger(logger);
      Log.d(TAG, "Debug logger updated on existing neural engine");
    }
  }

  /**
   * Ensures word predictor is initialized (lazy initialization).
   * Called when predictions are first requested.
   */
  public void ensureInitialized()
  {
    if (_wordPredictor == null)
    {
      initializeWordPredictor();
    }

    if (_config.swipe_typing_enabled && _neuralEngine == null)
    {
      initializeNeuralEngine();
    }
  }

  /**
   * Updates configuration and propagates to engines.
   *
   * @param newConfig Updated configuration
   */
  public void setConfig(Config newConfig)
  {
    _config = newConfig;

    // Update neural engine config if it exists
    if (_neuralEngine != null)
    {
      _neuralEngine.setConfig(_config);
    }

    // Update word predictor config if it exists
    if (_wordPredictor != null)
    {
      _wordPredictor.setConfig(_config);
    }
  }

  /**
   * Gets the WordPredictor instance.
   *
   * @return WordPredictor for typing predictions, or null if not initialized
   */
  public WordPredictor getWordPredictor()
  {
    return _wordPredictor;
  }

  /**
   * Gets the NeuralSwipeTypingEngine instance.
   *
   * @return Neural engine for swipe predictions, or null if not initialized
   */
  public NeuralSwipeTypingEngine getNeuralEngine()
  {
    return _neuralEngine;
  }

  /**
   * Gets the AsyncPredictionHandler instance.
   *
   * @return Async handler for background predictions, or null if not initialized
   */
  public AsyncPredictionHandler getAsyncPredictionHandler()
  {
    return _asyncPredictionHandler;
  }

  /**
   * Gets the DictionaryManager instance.
   *
   * @return Dictionary manager, or null if not initialized
   */
  public DictionaryManager getDictionaryManager()
  {
    return _dictionaryManager;
  }

  /**
   * Gets the SwipeMLDataStore instance.
   *
   * @return ML data store for swipe training data, or null if not initialized
   */
  public SwipeMLDataStore getMlDataStore()
  {
    return _mlDataStore;
  }

  /**
   * Gets the UserAdaptationManager instance.
   *
   * @return User adaptation manager for learning user preferences, or null if not initialized
   */
  public UserAdaptationManager getAdaptationManager()
  {
    return _adaptationManager;
  }

  /**
   * Checks if swipe typing is available.
   *
   * @return true if neural engine is initialized and ready
   */
  public boolean isSwipeTypingAvailable()
  {
    return _neuralEngine != null;
  }

  /**
   * Checks if word prediction is available.
   *
   * @return true if word predictor is initialized and ready
   */
  public boolean isWordPredictionAvailable()
  {
    return _wordPredictor != null;
  }

  /**
   * Shuts down all prediction engines and cleans up resources.
   * Should be called during keyboard shutdown.
   */
  public void shutdown()
  {
    // Shutdown async prediction handler
    if (_asyncPredictionHandler != null)
    {
      _asyncPredictionHandler.shutdown();
      _asyncPredictionHandler = null;
    }

    // Stop observing dictionary changes
    if (_wordPredictor != null)
    {
      _wordPredictor.stopObservingDictionaryChanges();
    }

    // Clean up engines
    _neuralEngine = null;
    _wordPredictor = null;
    _dictionaryManager = null;

    Log.d(TAG, "PredictionCoordinator shutdown complete");
  }

  /**
   * Gets a debug string showing current state.
   * Useful for logging and troubleshooting.
   *
   * @return Human-readable state description
   */
  public String getDebugState()
  {
    return String.format("PredictionCoordinator{wordPredictor=%s, neuralEngine=%s, asyncHandler=%s}",
      _wordPredictor != null ? "initialized" : "null",
      _neuralEngine != null ? "initialized" : "null",
      _asyncPredictionHandler != null ? "initialized" : "null");
  }
}
