package juloo.keyboard2;

import android.content.Context;
import android.content.res.Resources;
import android.view.inputmethod.EditorInfo;
import android.text.InputType;

/**
 * Manages keyboard layout selection, switching, and loading.
 *
 * This class centralizes logic for:
 * - Current layout tracking (text layouts and special layouts)
 * - Layout switching (text, numeric, emoji, clipboard, etc.)
 * - Layout navigation (forward/backward cycling through text layouts)
 * - Layout loading with modifiers (numpad, pinentry)
 * - Special layout determination based on input type
 *
 * Responsibilities:
 * - Track current text layout and special layout state
 * - Provide current layout (with or without modifiers)
 * - Load layouts from resources with appropriate modifications
 * - Determine special layouts based on EditorInfo input type
 * - Navigate between text layouts
 *
 * NOT included (remains in Keyboard2):
 * - View updates (setting keyboard on Keyboard2View)
 * - InputMethodService lifecycle methods
 * - Configuration management (reads from Config but doesn't modify)
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.363).
 */
public class LayoutManager
{
  private static final String TAG = "LayoutManager";

  private final Context _context;
  private Config _config;

  // Layout state
  private KeyboardData _currentSpecialLayout;
  private KeyboardData _localeTextLayout;

  /**
   * Creates a new LayoutManager.
   *
   * @param context Android context for resource access
   * @param config Configuration instance
   * @param localeTextLayout Initial locale-specific text layout
   */
  public LayoutManager(Context context, Config config, KeyboardData localeTextLayout)
  {
    _context = context;
    _config = config;
    _localeTextLayout = localeTextLayout;
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
   * Sets the locale text layout (default layout for typing).
   *
   * @param layout Locale-specific text layout
   */
  public void setLocaleTextLayout(KeyboardData layout)
  {
    _localeTextLayout = layout;
  }

  /**
   * Gets the current special layout (or null if showing text layout).
   *
   * @return Current special layout, or null
   */
  public KeyboardData getCurrentSpecialLayout()
  {
    return _currentSpecialLayout;
  }

  /**
   * Layout currently visible before it has been modified.
   *
   * @return Unmodified current layout
   */
  public KeyboardData current_layout_unmodified()
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

  /**
   * Layout currently visible (with modifiers applied).
   *
   * @return Current layout with modifications
   */
  public KeyboardData current_layout()
  {
    if (_currentSpecialLayout != null)
      return _currentSpecialLayout;
    return LayoutModifier.modify_layout(current_layout_unmodified());
  }

  /**
   * Sets the current text layout index and clears special layout.
   *
   * @param layoutIndex Index of text layout in config.layouts
   * @return The new current layout (for updating view)
   */
  public KeyboardData setTextLayout(int layoutIndex)
  {
    _config.set_current_layout(layoutIndex);
    _currentSpecialLayout = null;
    return current_layout();
  }

  /**
   * Cycles to next/previous text layout.
   *
   * @param delta +1 for forward, -1 for backward
   * @return The new current layout (for updating view)
   */
  public KeyboardData incrTextLayout(int delta)
  {
    int s = _config.layouts.size();
    int newIndex = (_config.get_current_layout() + delta + s) % s;
    return setTextLayout(newIndex);
  }

  /**
   * Sets a special layout (numeric, emoji, etc.).
   *
   * @param layout Special layout to display
   * @return The special layout (for updating view)
   */
  public KeyboardData setSpecialLayout(KeyboardData layout)
  {
    _currentSpecialLayout = layout;
    return layout;
  }

  /**
   * Clears special layout and returns to text layout.
   *
   * @return The current text layout (for updating view)
   */
  public KeyboardData clearSpecialLayout()
  {
    _currentSpecialLayout = null;
    return current_layout();
  }

  /**
   * Load a layout from resources.
   *
   * @param layoutId Resource ID of layout XML
   * @return Loaded layout
   */
  public KeyboardData loadLayout(int layoutId)
  {
    return KeyboardData.load(_context.getResources(), layoutId);
  }

  /**
   * Load a layout that contains a numpad, modified with current layout keys.
   *
   * @param layoutId Resource ID of layout XML
   * @return Loaded and modified numpad layout
   */
  public KeyboardData loadNumpad(int layoutId)
  {
    return LayoutModifier.modify_numpad(
        KeyboardData.load(_context.getResources(), layoutId),
        current_layout_unmodified());
  }

  /**
   * Load a pinentry layout, modified with current layout keys.
   *
   * @param layoutId Resource ID of layout XML
   * @return Loaded and modified pinentry layout
   */
  public KeyboardData loadPinentry(int layoutId)
  {
    return LayoutModifier.modify_pinentry(
        KeyboardData.load(_context.getResources(), layoutId),
        current_layout_unmodified());
  }

  /**
   * Determine special layout based on input type (number, phone, datetime).
   * Returns null if no special layout is needed.
   *
   * @param info EditorInfo from input field
   * @return Special layout, or null for normal text input
   */
  public KeyboardData refresh_special_layout(EditorInfo info)
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

  /**
   * Gets current text layout index.
   *
   * @return Current layout index
   */
  public int getCurrentLayoutIndex()
  {
    return _config.get_current_layout();
  }

  /**
   * Gets total number of text layouts.
   *
   * @return Number of layouts
   */
  public int getLayoutCount()
  {
    return _config.layouts.size();
  }
}
