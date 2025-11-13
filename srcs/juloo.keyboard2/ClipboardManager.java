package juloo.keyboard2;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.DatePicker;
import android.widget.RadioButton;
import android.widget.Switch;
import android.widget.TextView;

/**
 * Manages clipboard pane and clipboard history search functionality.
 *
 * This class centralizes the management of:
 * - Clipboard pane view lifecycle
 * - Clipboard search mode and search box
 * - Date filter dialog
 *
 * Responsibilities:
 * - Initialize and inflate clipboard pane views
 * - Manage clipboard search mode state
 * - Handle search text modification (append, delete, clear)
 * - Show and configure date filter dialog
 *
 * NOT included (remains in Keyboard2):
 * - Content pane container management (shared with emoji pane)
 * - Input view switching and lifecycle
 * - Key event routing during search mode
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.349).
 */
public class ClipboardManager
{
  private static final String TAG = "ClipboardManager";

  private final Context _context;
  private Config _config;

  // Clipboard views
  private ViewGroup _clipboardPane;
  private TextView _clipboardSearchBox;
  private ClipboardHistoryView _clipboardHistoryView;

  // Search state
  private boolean _searchMode;

  /**
   * Creates a new ClipboardManager.
   *
   * @param context Android context for inflating views
   * @param config Configuration instance
   */
  public ClipboardManager(Context context, Config config)
  {
    _context = context;
    _config = config;
    _searchMode = false;
  }

  /**
   * Gets or creates the clipboard pane view.
   * Performs lazy initialization on first call.
   *
   * @param layoutInflater LayoutInflater for inflating views
   * @return Clipboard pane ViewGroup
   */
  public ViewGroup getClipboardPane(LayoutInflater layoutInflater)
  {
    if (_clipboardPane == null)
    {
      // Inflate clipboard pane layout
      _clipboardPane = (ViewGroup)layoutInflater.inflate(R.layout.clipboard_pane, null);

      // Get search box and history view references
      _clipboardSearchBox = (TextView)_clipboardPane.findViewById(R.id.clipboard_search);
      _clipboardHistoryView = (ClipboardHistoryView)_clipboardPane.findViewById(R.id.clipboard_history_view);

      // Set up search box click listener
      if (_clipboardSearchBox != null)
      {
        _clipboardSearchBox.setOnClickListener(new View.OnClickListener() {
          @Override
          public void onClick(View v) {
            _searchMode = true;
            _clipboardSearchBox.setHint("Type on keyboard below...");
            _clipboardSearchBox.requestFocus();
          }
        });
      }

      // Set up date filter icon
      View dateFilterIcon = _clipboardPane.findViewById(R.id.clipboard_date_filter);
      if (dateFilterIcon != null)
      {
        dateFilterIcon.setOnClickListener(new View.OnClickListener() {
          @Override
          public void onClick(View v) {
            showDateFilterDialog(v);
          }
        });
      }
    }

    return _clipboardPane;
  }

  /**
   * Checks if clipboard search mode is active.
   *
   * @return true if in search mode
   */
  public boolean isInSearchMode()
  {
    return _searchMode;
  }

  /**
   * Appends text to clipboard search box and updates filter.
   *
   * @param text Text to append
   */
  public void appendToSearch(String text)
  {
    if (_clipboardSearchBox != null && _clipboardHistoryView != null)
    {
      // Append to current search text
      CharSequence current = _clipboardSearchBox.getText();
      String newText = current + text;
      _clipboardSearchBox.setText(newText);

      // Update history view filter
      _clipboardHistoryView.setSearchFilter(newText);
    }
  }

  /**
   * Deletes last character from clipboard search box and updates filter.
   */
  public void deleteFromSearch()
  {
    if (_clipboardSearchBox != null && _clipboardHistoryView != null)
    {
      CharSequence current = _clipboardSearchBox.getText();

      // Delete last character
      if (current.length() > 0)
      {
        String newText = current.subSequence(0, current.length() - 1).toString();
        _clipboardSearchBox.setText(newText);

        // Update history view filter
        _clipboardHistoryView.setSearchFilter(newText);
      }
    }
  }

  /**
   * Clears clipboard search and exits search mode.
   */
  public void clearSearch()
  {
    _searchMode = false;
    if (_clipboardSearchBox != null)
    {
      _clipboardSearchBox.setText("");
      _clipboardSearchBox.setHint("Tap to search...");
    }
    if (_clipboardHistoryView != null)
    {
      _clipboardHistoryView.setSearchFilter("");
    }
  }

  /**
   * Resets search state when showing clipboard pane.
   * Clears any previous search and exits search mode.
   */
  public void resetSearchOnShow()
  {
    _searchMode = false;
    if (_clipboardSearchBox != null)
    {
      _clipboardSearchBox.setText("");
      _clipboardSearchBox.setHint("Tap to search...");
    }
    if (_clipboardHistoryView != null)
    {
      _clipboardHistoryView.setSearchFilter("");
    }
  }

  /**
   * Resets search state when hiding clipboard pane.
   * Exits search mode and clears search text.
   */
  public void resetSearchOnHide()
  {
    _searchMode = false;
    if (_clipboardSearchBox != null)
    {
      _clipboardSearchBox.setText("");
      _clipboardSearchBox.setHint("Tap to search...");
    }
  }

  /**
   * Shows the date filter dialog for filtering clipboard entries by date.
   *
   * @param anchorView View to anchor the dialog window token
   */
  public void showDateFilterDialog(View anchorView)
  {
    // Use dark theme for dialog to match keyboard theme
    android.view.ContextThemeWrapper themedContext = new android.view.ContextThemeWrapper(
      _context, android.R.style.Theme_DeviceDefault_Dialog);

    final View dialogView = LayoutInflater.from(themedContext).inflate(
      R.layout.clipboard_date_filter_dialog, null);

    final Switch enabledSwitch = dialogView.findViewById(R.id.date_filter_enabled);
    final android.widget.RadioGroup modeGroup = dialogView.findViewById(R.id.date_filter_mode);
    final RadioButton beforeRadio = dialogView.findViewById(R.id.date_filter_before);
    final RadioButton afterRadio = dialogView.findViewById(R.id.date_filter_after);
    final DatePicker datePicker = dialogView.findViewById(R.id.date_picker);
    final View modeContainer = dialogView.findViewById(R.id.date_filter_mode_container);
    final View pickerContainer = dialogView.findViewById(R.id.date_picker_container);

    // Get current filter state from ClipboardHistoryView
    final boolean isFilterEnabled = _clipboardHistoryView != null && _clipboardHistoryView.isDateFilterEnabled();
    final boolean isBeforeMode = _clipboardHistoryView != null && _clipboardHistoryView.isDateFilterBefore();

    enabledSwitch.setChecked(isFilterEnabled);
    if (isBeforeMode) {
      beforeRadio.setChecked(true);
    } else {
      afterRadio.setChecked(true);
    }

    // Set initial visibility based on enabled state
    modeContainer.setVisibility(isFilterEnabled ? View.VISIBLE : View.GONE);
    pickerContainer.setVisibility(isFilterEnabled ? View.VISIBLE : View.GONE);

    // Toggle visibility when enable switch changes
    enabledSwitch.setOnCheckedChangeListener(new android.widget.CompoundButton.OnCheckedChangeListener() {
      @Override
      public void onCheckedChanged(android.widget.CompoundButton buttonView, boolean isChecked) {
        modeContainer.setVisibility(isChecked ? View.VISIBLE : View.GONE);
        pickerContainer.setVisibility(isChecked ? View.VISIBLE : View.GONE);
      }
    });

    // Get current filter date or default to today
    java.util.Calendar cal = java.util.Calendar.getInstance();
    if (_clipboardHistoryView != null && _clipboardHistoryView.getDateFilterTimestamp() > 0) {
      cal.setTimeInMillis(_clipboardHistoryView.getDateFilterTimestamp());
    }
    datePicker.updateDate(cal.get(java.util.Calendar.YEAR),
                          cal.get(java.util.Calendar.MONTH),
                          cal.get(java.util.Calendar.DAY_OF_MONTH));

    android.app.AlertDialog dialog = new android.app.AlertDialog.Builder(themedContext)
      .setTitle("Filter by Date")
      .setView(dialogView)
      .create();

    // Set up button click handlers
    dialogView.findViewById(R.id.date_filter_clear).setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        if (_clipboardHistoryView != null) {
          _clipboardHistoryView.clearDateFilter();
        }
        dialog.dismiss();
      }
    });

    dialogView.findViewById(R.id.date_filter_cancel).setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        dialog.dismiss();
      }
    });

    dialogView.findViewById(R.id.date_filter_apply).setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        if (_clipboardHistoryView != null) {
          boolean enabled = enabledSwitch.isChecked();
          boolean isBefore = beforeRadio.isChecked();

          if (enabled) {
            // Get selected date at start of day (00:00:00)
            java.util.Calendar selectedCal = java.util.Calendar.getInstance();
            selectedCal.set(datePicker.getYear(), datePicker.getMonth(), datePicker.getDayOfMonth(), 0, 0, 0);
            selectedCal.set(java.util.Calendar.MILLISECOND, 0);
            long timestamp = selectedCal.getTimeInMillis();

            _clipboardHistoryView.setDateFilter(timestamp, isBefore);
          } else {
            _clipboardHistoryView.clearDateFilter();
          }
        }
        dialog.dismiss();
      }
    });

    Utils.show_dialog_on_ime(dialog, anchorView.getWindowToken());
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
   * Cleans up resources.
   * Should be called during keyboard shutdown.
   */
  public void cleanup()
  {
    _clipboardPane = null;
    _clipboardSearchBox = null;
    _clipboardHistoryView = null;
    _searchMode = false;
  }

  /**
   * Gets a debug string showing current state.
   * Useful for logging and troubleshooting.
   *
   * @return Human-readable state description
   */
  public String getDebugState()
  {
    return String.format("ClipboardManager{clipboardPane=%s, searchMode=%b}",
      _clipboardPane != null ? "initialized" : "null",
      _searchMode);
  }
}
