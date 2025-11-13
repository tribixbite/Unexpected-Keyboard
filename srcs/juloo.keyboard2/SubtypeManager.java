package juloo.keyboard2;

import android.annotation.TargetApi;
import android.content.Context;
import android.content.res.Resources;
import android.os.Build.VERSION;
import android.view.inputmethod.InputMethodInfo;
import android.view.inputmethod.InputMethodManager;
import android.view.inputmethod.InputMethodSubtype;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import juloo.keyboard2.prefs.LayoutsPreference;

/**
 * Manages IME subtypes, locale layouts, and extra keys.
 *
 * This class centralizes logic for:
 * - Getting enabled IME subtypes for this keyboard
 * - Extracting extra keys (accents) from subtypes
 * - Determining default subtype based on system settings
 * - Refreshing locale layout based on current subtype
 * - Managing extra keys configuration
 *
 * Responsibilities:
 * - Query InputMethodManager for enabled subtypes
 * - Parse subtype extra values (default_layout, extra_keys, script)
 * - Update Config with merged extra keys from all enabled subtypes
 * - Determine locale-specific default layout
 * - Handle Android version differences (API 12+, 24+)
 *
 * NOT included (remains in Keyboard2):
 * - InputMethodService lifecycle methods
 * - LayoutManager updates (caller updates after getting layout)
 * - Configuration persistence (SubtypeManager reads/writes to Config)
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.365).
 */
public class SubtypeManager
{
  private static final String TAG = "SubtypeManager";

  private final Context _context;
  private final InputMethodManager _imm;

  /**
   * Creates a new SubtypeManager.
   *
   * @param context Android context for system services and resources
   */
  public SubtypeManager(Context context)
  {
    _context = context;
    _imm = (InputMethodManager) context.getSystemService(Context.INPUT_METHOD_SERVICE);
  }

  /**
   * Gets list of enabled subtypes for this keyboard.
   *
   * @return List of enabled subtypes, or empty list if none found
   */
  public List<InputMethodSubtype> getEnabledSubtypes()
  {
    String pkg = _context.getPackageName();
    for (InputMethodInfo imi : _imm.getEnabledInputMethodList())
    {
      if (imi.getPackageName().equals(pkg))
      {
        return _imm.getEnabledInputMethodSubtypeList(imi, true);
      }
    }
    return Arrays.asList();
  }

  /**
   * Extracts extra keys from a subtype.
   *
   * @param subtype Input method subtype
   * @return ExtraKeys parsed from subtype, or EMPTY if none
   */
  @TargetApi(12)
  public ExtraKeys extra_keys_of_subtype(InputMethodSubtype subtype)
  {
    String extra_keys = subtype.getExtraValueOf("extra_keys");
    String script = subtype.getExtraValueOf("script");
    if (extra_keys != null)
    {
      return ExtraKeys.parse(script, extra_keys);
    }
    return ExtraKeys.EMPTY;
  }

  /**
   * Refreshes accent options by merging extra keys from all enabled subtypes.
   *
   * @param enabled_subtypes List of enabled subtypes
   * @return Merged ExtraKeys from all subtypes
   */
  public ExtraKeys refreshAccentsOption(List<InputMethodSubtype> enabled_subtypes)
  {
    List<ExtraKeys> extra_keys = new ArrayList<ExtraKeys>();
    for (InputMethodSubtype s : enabled_subtypes)
    {
      extra_keys.add(extra_keys_of_subtype(s));
    }
    return ExtraKeys.merge(extra_keys);
  }

  /**
   * Gets the default subtype based on current system settings.
   * On Android 7.0+ (API 24), matches by language tag to avoid random selection.
   *
   * @param enabled_subtypes List of enabled subtypes
   * @return Default subtype, or null if none found
   */
  @TargetApi(12)
  public InputMethodSubtype defaultSubtypes(List<InputMethodSubtype> enabled_subtypes)
  {
    if (VERSION.SDK_INT < 24)
    {
      return _imm.getCurrentInputMethodSubtype();
    }

    // Android might return a random subtype, for example, the first in the
    // list alphabetically.
    InputMethodSubtype current_subtype = _imm.getCurrentInputMethodSubtype();
    if (current_subtype == null)
    {
      return null;
    }

    for (InputMethodSubtype s : enabled_subtypes)
    {
      if (s.getLanguageTag().equals(current_subtype.getLanguageTag()))
      {
        return s;
      }
    }
    return null;
  }

  /**
   * Refreshes subtype settings and returns the appropriate default layout.
   * Updates config with voice typing availability and extra keys.
   *
   * @param config Config to update with extra keys
   * @param resources Resources for loading layouts
   * @return Default layout for current subtype, or null to use fallback
   */
  public KeyboardData refreshSubtype(Config config, Resources resources)
  {
    config.shouldOfferVoiceTyping = true;
    KeyboardData default_layout = null;
    config.extra_keys_subtype = null;

    if (VERSION.SDK_INT >= 12)
    {
      List<InputMethodSubtype> enabled_subtypes = getEnabledSubtypes();
      InputMethodSubtype subtype = defaultSubtypes(enabled_subtypes);

      if (subtype != null)
      {
        String s = subtype.getExtraValueOf("default_layout");
        if (s != null)
        {
          default_layout = LayoutsPreference.layout_of_string(resources, s);
        }
        config.extra_keys_subtype = refreshAccentsOption(enabled_subtypes);
      }
    }

    return default_layout;
  }

  /**
   * Gets InputMethodManager instance.
   *
   * @return InputMethodManager for IME operations
   */
  public InputMethodManager getInputMethodManager()
  {
    return _imm;
  }
}
