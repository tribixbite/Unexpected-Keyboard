package juloo.keyboard2;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Resources;
import java.util.ArrayList;
import java.util.List;

/**
 * Manages keyboard configuration and notifies listeners of changes.
 *
 * This class centralizes configuration management and uses the observer pattern
 * to decouple config refresh from config propagation. Components that need to
 * respond to config changes should implement ConfigChangeListener.
 *
 * Responsibilities:
 * - Own and manage the Config instance
 * - Own and manage the FoldStateTracker
 * - Listen to SharedPreferences changes
 * - Refresh config when needed
 * - Notify registered listeners of changes
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.345).
 */
public class ConfigurationManager implements SharedPreferences.OnSharedPreferenceChangeListener
{
  private static final String TAG = "ConfigurationManager";

  private final Context _context;
  private Config _config;
  private FoldStateTracker _foldStateTracker;
  private final List<ConfigChangeListener> _listeners;

  /**
   * Creates a new ConfigurationManager.
   *
   * NOTE: Config parameter should be the global Config instance (from Config.globalConfig()).
   * ConfigurationManager will manage its lifecycle but not create it.
   *
   * @param context Android context for accessing resources
   * @param config Pre-initialized Config instance (typically from Config.globalConfig())
   * @param foldStateTracker Pre-initialized FoldStateTracker instance
   */
  public ConfigurationManager(Context context, Config config, FoldStateTracker foldStateTracker)
  {
    _context = context;
    _config = config;
    _foldStateTracker = foldStateTracker;
    _listeners = new ArrayList<>();

    // Set up fold state callback to refresh config when device is folded/unfolded
    _foldStateTracker.setChangedCallback(() -> {
      refresh(context.getResources());
    });
  }

  /**
   * Gets the current Config instance.
   *
   * @return Current Config object (never null)
   */
  public Config getConfig()
  {
    return _config;
  }

  /**
   * Gets the FoldStateTracker instance.
   *
   * @return FoldStateTracker for monitoring device fold state
   */
  public FoldStateTracker getFoldStateTracker()
  {
    return _foldStateTracker;
  }

  /**
   * Registers a listener for config changes.
   *
   * @param listener ConfigChangeListener to be notified of changes
   */
  public void registerConfigChangeListener(ConfigChangeListener listener)
  {
    if (listener != null && !_listeners.contains(listener))
    {
      _listeners.add(listener);
    }
  }

  /**
   * Unregisters a previously registered listener.
   *
   * @param listener ConfigChangeListener to remove
   */
  public void unregisterConfigChangeListener(ConfigChangeListener listener)
  {
    _listeners.remove(listener);
  }

  /**
   * Refreshes the configuration from SharedPreferences.
   *
   * This method:
   * 1. Captures the previous theme
   * 2. Refreshes the Config object
   * 3. Notifies all registered listeners
   * 4. Sends theme change notification if theme changed
   *
   * @param res Resources for config loading
   */
  public void refresh(Resources res)
  {
    int prevTheme = _config.theme;

    // Refresh config from SharedPreferences
    _config.refresh(res, _foldStateTracker.isUnfolded());

    // Notify listeners of config change
    for (ConfigChangeListener listener : _listeners)
    {
      listener.onConfigChanged(_config);
    }

    // Special notification for theme changes (requires view recreation)
    if (prevTheme != _config.theme)
    {
      for (ConfigChangeListener listener : _listeners)
      {
        listener.onThemeChanged(prevTheme, _config.theme);
      }
    }
  }

  /**
   * Called when a SharedPreference changes.
   *
   * This is the entry point for all preference changes. It triggers
   * a config refresh which will notify all registered listeners.
   *
   * @param prefs SharedPreferences that changed
   * @param key Key of the preference that changed (may be null)
   */
  @Override
  public void onSharedPreferenceChanged(SharedPreferences prefs, String key)
  {
    // Refresh config and notify listeners
    refresh(_context.getResources());
  }

  /**
   * Gets a debug string showing current state.
   * Useful for logging and troubleshooting.
   *
   * @return Human-readable state description
   */
  public String getDebugState()
  {
    return String.format("ConfigurationManager{theme=%d, listeners=%d, isUnfolded=%b}",
      _config.theme,
      _listeners.size(),
      _foldStateTracker.isUnfolded());
  }
}
