package juloo.keyboard2;

/**
 * Data class representing a clipboard entry with content and timestamp
 */
public class ClipboardEntry
{
  public final String content;
  public final long timestamp; // Unix timestamp in milliseconds

  public ClipboardEntry(String content, long timestamp)
  {
    this.content = content;
    this.timestamp = timestamp;
  }

  /**
   * Format timestamp as relative time (e.g., "2h ago", "Yesterday")
   */
  public String getRelativeTime()
  {
    long now = System.currentTimeMillis();
    long diff = now - timestamp;

    long seconds = diff / 1000;
    long minutes = seconds / 60;
    long hours = minutes / 60;
    long days = hours / 24;

    if (seconds < 60)
      return "Just now";
    else if (minutes < 60)
      return minutes + "m ago";
    else if (hours < 24)
      return hours + "h ago";
    else if (days == 1)
      return "Yesterday";
    else if (days < 7)
      return days + "d ago";
    else
      return formatDate();
  }

  /**
   * Format timestamp as date string (e.g., "Nov 12")
   */
  public String formatDate()
  {
    java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat("MMM d", java.util.Locale.getDefault());
    return sdf.format(new java.util.Date(timestamp));
  }

  /**
   * Get formatted text with timestamp appended
   * Returns SpannableString with timestamp in secondary color
   */
  public android.text.SpannableString getFormattedText(android.content.Context context)
  {
    String timeStr = " · " + getRelativeTime();
    String fullText = content + timeStr;

    android.text.SpannableString spannable = new android.text.SpannableString(fullText);

    // Get theme-aware colors using context theme
    android.util.TypedValue typedValue = new android.util.TypedValue();
    android.content.res.Resources.Theme theme = context.getTheme();

    // Get colorLabel for main text (theme-aware)
    theme.resolveAttribute(android.R.attr.textColorPrimary, typedValue, true);
    int primaryColor = typedValue.data;

    // Get colorSubLabel for timestamp (theme-aware)
    theme.resolveAttribute(android.R.attr.textColorSecondary, typedValue, true);
    int secondaryColor = typedValue.data;

    // Apply primary color to content portion
    spannable.setSpan(
      new android.text.style.ForegroundColorSpan(primaryColor),
      0,
      content.length(),
      android.text.Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
    );

    // Apply secondary color to timestamp portion
    spannable.setSpan(
      new android.text.style.ForegroundColorSpan(secondaryColor),
      content.length(),
      fullText.length(),
      android.text.Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
    );

    return spannable;
  }
}
