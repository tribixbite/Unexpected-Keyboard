package juloo.keyboard2;

import android.content.Context;
import android.graphics.Color;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import java.util.ArrayList;
import java.util.List;

/**
 * View component that displays word suggestions above the keyboard
 */
public class SuggestionBar extends LinearLayout
{
  private final List<TextView> _suggestionViews;
  private OnSuggestionSelectedListener _listener;
  private List<String> _currentSuggestions;
  private int _selectedIndex = -1;
  
  public interface OnSuggestionSelectedListener
  {
    void onSuggestionSelected(String word);
  }
  
  public SuggestionBar(Context context)
  {
    this(context, null);
  }
  
  public SuggestionBar(Context context, AttributeSet attrs)
  {
    super(context, attrs);
    _suggestionViews = new ArrayList<>();
    _currentSuggestions = new ArrayList<>();
    initialize(context);
  }
  
  private void initialize(Context context)
  {
    setOrientation(HORIZONTAL);
    setGravity(Gravity.CENTER_VERTICAL);
    setBackgroundColor(Color.parseColor("#E0E0E0"));
    
    int padding = dpToPx(context, 8);
    setPadding(padding, padding, padding, padding);
    
    // Create suggestion text views
    for (int i = 0; i < 5; i++)
    {
      TextView suggestionView = createSuggestionView(context, i);
      _suggestionViews.add(suggestionView);
      
      // Add divider except for the last item
      if (i < 4)
      {
        View divider = createDivider(context);
        addView(divider);
      }
    }
  }
  
  private TextView createSuggestionView(Context context, final int index)
  {
    TextView textView = new TextView(context);
    LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
      0, ViewGroup.LayoutParams.MATCH_PARENT, 1.0f);
    textView.setLayoutParams(params);
    textView.setGravity(Gravity.CENTER);
    textView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
    textView.setTextColor(Color.BLACK);
    textView.setPadding(dpToPx(context, 8), 0, dpToPx(context, 8), 0);
    textView.setMaxLines(1);
    textView.setClickable(true);
    textView.setFocusable(true);
    
    // Set click listener
    textView.setOnClickListener(new OnClickListener()
    {
      @Override
      public void onClick(View v)
      {
        if (index < _currentSuggestions.size() && _listener != null)
        {
          _listener.onSuggestionSelected(_currentSuggestions.get(index));
        }
      }
    });
    
    addView(textView);
    return textView;
  }
  
  private View createDivider(Context context)
  {
    View divider = new View(context);
    LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
      dpToPx(context, 1), ViewGroup.LayoutParams.MATCH_PARENT);
    params.setMargins(0, dpToPx(context, 4), 0, dpToPx(context, 4));
    divider.setLayoutParams(params);
    divider.setBackgroundColor(Color.parseColor("#CCCCCC"));
    return divider;
  }
  
  /**
   * Update the displayed suggestions
   */
  public void setSuggestions(List<String> suggestions)
  {
    _currentSuggestions.clear();
    if (suggestions != null)
    {
      _currentSuggestions.addAll(suggestions);
    }
    
    // Update text views
    for (int i = 0; i < _suggestionViews.size(); i++)
    {
      TextView textView = _suggestionViews.get(i);
      if (i < _currentSuggestions.size())
      {
        String suggestion = _currentSuggestions.get(i);
        textView.setText(suggestion);
        textView.setVisibility(View.VISIBLE);
        
        // Highlight first suggestion
        if (i == 0)
        {
          textView.setTypeface(Typeface.DEFAULT_BOLD);
          textView.setTextColor(Color.parseColor("#1976D2"));
        }
        else
        {
          textView.setTypeface(Typeface.DEFAULT);
          textView.setTextColor(Color.BLACK);
        }
      }
      else
      {
        textView.setText("");
        textView.setVisibility(View.GONE);
      }
    }
    
    // Show or hide the entire bar based on suggestions
    setVisibility(_currentSuggestions.isEmpty() ? View.GONE : View.VISIBLE);
  }
  
  /**
   * Clear all suggestions
   */
  public void clearSuggestions()
  {
    setSuggestions(null);
  }
  
  /**
   * Set the listener for suggestion selection
   */
  public void setOnSuggestionSelectedListener(OnSuggestionSelectedListener listener)
  {
    _listener = listener;
  }
  
  /**
   * Get the currently displayed suggestions
   */
  public List<String> getCurrentSuggestions()
  {
    return new ArrayList<>(_currentSuggestions);
  }
  
  /**
   * Convert dp to pixels
   */
  private int dpToPx(Context context, int dp)
  {
    float density = context.getResources().getDisplayMetrics().density;
    return Math.round(dp * density);
  }
}