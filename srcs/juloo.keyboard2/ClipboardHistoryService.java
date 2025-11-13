package juloo.keyboard2;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.os.Build.VERSION;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public final class ClipboardHistoryService
{
  /** Start the service on startup and start listening to clipboard changes.
   *  IMPORTANT: This should be called from InputMethodService.onCreate() to ensure
   *  system-wide clipboard monitoring for the entire service lifetime. */
  public static void on_startup(Context ctx, ClipboardPasteCallback cb)
  {
    ClipboardHistoryService service = get_service(ctx);
    if (service != null)
    {
      service._paste_callback = cb;
      // Register listener immediately on service startup for system-wide monitoring
      service.registerClipboardListener();
    }
  }

  /** Cleanup and unregister listener. Call from InputMethodService.onDestroy(). */
  public static void on_shutdown()
  {
    if (_service != null)
    {
      _service.unregisterClipboardListener();
    }
  }

  /** Start the service if it hasn't been started before. Returns [null] if the
      feature is unsupported. */
  public static ClipboardHistoryService get_service(Context ctx)
  {
    if (VERSION.SDK_INT <= 11)
      return null;
    if (_service == null)
      _service = new ClipboardHistoryService(ctx);
    return _service;
  }

  public static void set_history_enabled(boolean e)
  {
    Config.globalConfig().set_clipboard_history_enabled(e);
    if (_service == null)
      return;
    if (e)
    {
      // Re-enable: add current clip and re-register listener if needed
      _service.add_current_clip();
      _service.registerClipboardListener();
    }
    // NOTE: When disabling, we DO NOT clear history data
    // This preserves user data and allows re-enabling without data loss
    // History will simply stop recording new clipboard changes
  }

  /** Send the given string to the editor. */
  public static void paste(String clip)
  {
    if (_service != null && _service._paste_callback != null)
      _service._paste_callback.paste_from_clipboard_pane(clip);
    else
      android.util.Log.w("ClipboardHistory", "Cannot paste - callback not initialized");
  }

  /** Clipboard history is persistently stored in SQLite database and survives app restarts.
      Entries expire after HISTORY_TTL_MS unless pinned. The configurable size limit
      (clipboard_history_limit) controls maximum entries (0 = unlimited). */
  /** Time in ms until history entries expire.
   *  Set to 7 days to maintain useful history across app updates and restarts.
   *  Use pinning for permanent entries. */
  public static final long HISTORY_TTL_MS = 7 * 24 * 60 * 60 * 1000L; // 7 days

  static ClipboardHistoryService _service = null;

  Context _context;
  ClipboardManager _cm;
  ClipboardDatabase _database;
  ClipboardPasteCallback _paste_callback = null;
  OnClipboardHistoryChange _listener = null;
  boolean _isListenerRegistered = false;

  ClipboardHistoryService(Context ctx)
  {
    _context = ctx.getApplicationContext();
    _database = ClipboardDatabase.getInstance(_context);
    _cm = (ClipboardManager)_context.getSystemService(Context.CLIPBOARD_SERVICE);

    // Clean up expired entries on startup
    _database.cleanupExpiredEntries();

    // Note: Listener registration is deferred to attemptToRegisterListener()
    // which will be called from on_startup() and can be retried when keyboard gains focus
  }

  /**
   * Register clipboard listener for system-wide monitoring.
   * On Android 10+, being the default IME grants clipboard access even when keyboard is hidden.
   * This listener persists for the entire InputMethodService lifetime.
   * Should be called ONCE from InputMethodService.onCreate().
   */
  public void registerClipboardListener()
  {
    if (_isListenerRegistered || _cm == null)
      return;

    // On Android 10+ (API 29+), being default IME grants system-wide clipboard access
    if (VERSION.SDK_INT >= 29 && !isDefaultIme())
    {
      android.util.Log.w("ClipboardHistory", "Clipboard access requires this keyboard to be set as default input method");
      // User notification will be handled by settings UI showing clipboard status
      return;
    }

    try
    {
      _cm.addPrimaryClipChangedListener(this.new SystemListener());
      _isListenerRegistered = true;
      android.util.Log.i("ClipboardHistory", "Clipboard listener registered for system-wide monitoring");

      // Add current clip in case it changed while listener was not active
      add_current_clip();
    }
    catch (SecurityException e)
    {
      _isListenerRegistered = false;
      android.util.Log.e("ClipboardHistory", "Clipboard access denied: " + e.getMessage());
    }
    catch (Exception e)
    {
      _isListenerRegistered = false;
      android.util.Log.e("ClipboardHistory", "Failed to register clipboard listener", e);
    }
  }

  /**
   * Unregister clipboard listener. Call from InputMethodService.onDestroy().
   */
  public void unregisterClipboardListener()
  {
    if (!_isListenerRegistered || _cm == null)
      return;

    try
    {
      // Note: We cannot remove a specific listener instance, so this may not work as expected
      // The listener will be automatically cleaned up when the service process is destroyed
      android.util.Log.i("ClipboardHistory", "Clipboard listener cleanup on service destroy");
      _isListenerRegistered = false;
    }
    catch (Exception e)
    {
      android.util.Log.e("ClipboardHistory", "Error cleaning up clipboard listener", e);
    }
  }

  /**
   * Check if this keyboard is set as the default input method.
   * Required for clipboard access on Android 10+.
   */
  private boolean isDefaultIme()
  {
    try
    {
      String defaultIme = android.provider.Settings.Secure.getString(
          _context.getContentResolver(),
          android.provider.Settings.Secure.DEFAULT_INPUT_METHOD
      );
      return defaultIme != null && defaultIme.startsWith(_context.getPackageName());
    }
    catch (Exception e)
    {
      android.util.Log.e("ClipboardHistory", "Failed to check default IME status", e);
      return false;
    }
  }

  /**
   * Get clipboard feature status for user feedback.
   * Returns status message indicating if clipboard monitoring is active.
   */
  public String getClipboardStatus()
  {
    if (!Config.globalConfig().clipboard_history_enabled)
      return "Clipboard history disabled in settings";

    if (!_isListenerRegistered)
    {
      if (VERSION.SDK_INT >= 29 && !isDefaultIme())
        return "Clipboard access requires setting this keyboard as default input method";
      return "Clipboard monitoring inactive - open keyboard to activate";
    }

    int activeEntries = _database.getActiveEntryCount();
    return String.format("Clipboard monitoring active (%d entries)", activeEntries);
  }

  public List<String> clear_expired_and_get_history()
  {
    // Clean up expired entries and return active ones
    _database.cleanupExpiredEntries();
    return _database.getActiveClipboardEntries();
  }

  /** This will call [on_clipboard_history_change]. */
  public void remove_history_entry(String clip)
  {
    // Check if this is the most recent clipboard entry
    List<String> currentHistory = _database.getActiveClipboardEntries();
    boolean isCurrentClip = !currentHistory.isEmpty() && currentHistory.get(0).equals(clip);

    // If removing the current clipboard, clear the system clipboard
    if (isCurrentClip)
    {
      try
      {
        if (VERSION.SDK_INT >= 28)
          _cm.clearPrimaryClip();
        else
          _cm.setPrimaryClip(ClipData.newPlainText("", ""));
      }
      catch (SecurityException e)
      {
        // Android 10+ may deny clipboard access when app is not in focus
        android.util.Log.d("ClipboardHistory", "Cannot clear clipboard (app not in focus): " + e.getMessage());
      }
    }

    // Remove from database
    boolean removed = _database.removeClipboardEntry(clip);
    if (removed && _listener != null)
      _listener.on_clipboard_history_change();
  }

  /** Add clipboard entries to the history, skipping consecutive duplicates and
      empty strings. */
  public void add_clip(String clip)
  {
    if (!Config.globalConfig().clipboard_history_enabled)
      return;

    if (clip == null || clip.trim().isEmpty())
      return;

    // Check maximum item size limit
    int maxSizeKb = Config.globalConfig().clipboard_max_item_size_kb;
    if (maxSizeKb > 0)
    {
      try
      {
        int sizeBytes = clip.getBytes(java.nio.charset.StandardCharsets.UTF_8).length;
        int maxSizeBytes = maxSizeKb * 1024;

        if (sizeBytes > maxSizeBytes)
        {
          // Item exceeds size limit - reject and notify user
          android.util.Log.w("ClipboardHistory", "Clipboard item too large: " + sizeBytes + " bytes (limit: " + maxSizeBytes + " bytes)");

          // Show toast notification to user
          String message = String.format("Clipboard item too large (%d KB). Limit is %d KB.",
              sizeBytes / 1024, maxSizeKb);
          android.widget.Toast.makeText(_context, message, android.widget.Toast.LENGTH_LONG).show();
          return; // Don't add to clipboard history
        }
      }
      catch (Exception e)
      {
        android.util.Log.e("ClipboardHistory", "Error checking clipboard item size: " + e.getMessage());
        // Continue with add if size check fails
      }
    }

    // Calculate expiry time
    long expiryTime = System.currentTimeMillis() + HISTORY_TTL_MS;

    // Add to database (handles duplicate detection automatically)
    boolean added = _database.addClipboardEntry(clip, expiryTime);
    
    if (added)
    {
      // Apply size limits if configured
      int maxHistorySize = Config.globalConfig().clipboard_history_limit;
      if (maxHistorySize > 0)
      {
        _database.applySizeLimit(maxHistorySize);
      }
      
      if (_listener != null)
        _listener.on_clipboard_history_change();
    }
  }

  public void clear_history()
  {
    _database.clearAllEntries();
    if (_listener != null)
      _listener.on_clipboard_history_change();
  }

  public void set_on_clipboard_history_change(OnClipboardHistoryChange l) { _listener = l; }
  
  /** Pin or unpin a clipboard entry to prevent it from expiring */
  public void set_pinned_status(String clip, boolean isPinned)
  {
    boolean updated = _database.setPinnedStatus(clip, isPinned);
    if (updated && _listener != null)
      _listener.on_clipboard_history_change();
  }

  /** Get all pinned clipboard entries */
  public List<String> get_pinned_entries()
  {
    return _database.getPinnedEntries();
  }

  /** Get statistics about clipboard storage */
  public String getStorageStats()
  {
    ClipboardDatabase.StorageStats stats = _database.getStorageStats();

    // Format size in human-readable format (KB/MB)
    String activeSize = formatBytes(stats.activeSizeBytes);
    String pinnedSize = formatBytes(stats.pinnedSizeBytes);

    // Build multi-line summary with active and pinned breakdown
    StringBuilder sb = new StringBuilder();
    sb.append(String.format("%d active entries (%s)", stats.activeEntries, activeSize));

    if (stats.pinnedEntries > 0)
    {
      sb.append(String.format("\n%d pinned (%s)", stats.pinnedEntries, pinnedSize));
    }

    return sb.toString();
  }

  /** Format bytes into human-readable string (KB or MB) */
  private String formatBytes(long bytes)
  {
    if (bytes < 1024)
    {
      return bytes + " B";
    }
    else if (bytes < 1024 * 1024)
    {
      return String.format("%.1f KB", bytes / 1024.0);
    }
    else
    {
      return String.format("%.2f MB", bytes / (1024.0 * 1024.0));
    }
  }

  public static interface OnClipboardHistoryChange
  {
    public void on_clipboard_history_change();
  }

  /** Add what is currently in the system clipboard into the history. */
  void add_current_clip()
  {
    try
    {
      ClipData clip = _cm.getPrimaryClip();
      if (clip == null)
        return;
      int count = clip.getItemCount();
      for (int i = 0; i < count; i++)
      {
        CharSequence text = clip.getItemAt(i).getText();
        if (text != null)
          add_clip(text.toString());
      }
    }
    catch (SecurityException e)
    {
      // Android 10+ denies clipboard access when app is not in focus
      // This is expected behavior - we can only access clipboard when keyboard is visible
      android.util.Log.d("ClipboardHistoryService", "Clipboard access denied (app not in focus): " + e.getMessage());
    }
  }

  final class SystemListener implements ClipboardManager.OnPrimaryClipChangedListener
  {
    public SystemListener() {}

    @Override
    public void onPrimaryClipChanged()
    {
      add_current_clip();
    }
  }

  // HistoryEntry class removed - now using SQLite database storage

  public interface ClipboardPasteCallback
  {
    public void paste_from_clipboard_pane(String content);
  }
}
