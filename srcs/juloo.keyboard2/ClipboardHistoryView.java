package juloo.keyboard2;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.util.AttributeSet;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class ClipboardHistoryView extends NonScrollListView
  implements ClipboardHistoryService.OnClipboardHistoryChange
{
  List<ClipboardEntry> _history;
  List<ClipboardEntry> _filteredHistory;
  String _searchFilter = "";
  ClipboardHistoryService _service;
  ClipboardEntriesAdapter _adapter;
  // Track expanded state: position -> isExpanded
  java.util.Map<Integer, Boolean> _expandedStates = new java.util.HashMap<>();
  // Date filter state
  boolean _dateFilterEnabled = false;
  boolean _dateFilterBefore = false; // true = before date, false = after date
  long _dateFilterTimestamp = 0;

  public ClipboardHistoryView(Context ctx, AttributeSet attrs)
  {
    super(ctx, attrs);
    _history = Collections.EMPTY_LIST;
    _filteredHistory = Collections.EMPTY_LIST;
    _adapter = this.new ClipboardEntriesAdapter();
    _service = ClipboardHistoryService.get_service(ctx);
    if (_service != null)
    {
      _service.set_on_clipboard_history_change(this);
      _history = _service.clear_expired_and_get_history();
      _filteredHistory = _history;
    }
    setAdapter(_adapter);
  }

  /** Filter clipboard history by search text */
  public void setSearchFilter(String filter)
  {
    _searchFilter = filter == null ? "" : filter.toLowerCase();
    applyFilter();
  }

  private void applyFilter()
  {
    List<ClipboardEntry> filtered = new ArrayList<ClipboardEntry>();

    // Apply both search and date filters
    for (ClipboardEntry item : _history)
    {
      // Apply search filter
      if (!_searchFilter.isEmpty() && !item.content.toLowerCase().contains(_searchFilter))
        continue;

      // Apply date filter
      if (_dateFilterEnabled)
      {
        if (_dateFilterBefore)
        {
          // Show entries before the selected date
          if (item.timestamp >= _dateFilterTimestamp)
            continue;
        }
        else
        {
          // Show entries after the selected date
          if (item.timestamp < _dateFilterTimestamp)
            continue;
        }
      }

      filtered.add(item);
    }

    // If no filters are active, show all history
    if (_searchFilter.isEmpty() && !_dateFilterEnabled)
    {
      _filteredHistory = _history;
    }
    else
    {
      _filteredHistory = filtered;
    }

    _adapter.notifyDataSetChanged();
    invalidate();
  }

  /** The history entry at index [pos] is removed from the history and added to
      the list of pinned clipboards. */
  public void pin_entry(int pos)
  {
    String clip = _filteredHistory.get(pos).content;

    // Set pinned status in database instead of removing
    _service.set_pinned_status(clip, true);

    // Notify pin view to refresh
    ClipboardPinView v = (ClipboardPinView)((ViewGroup)getParent().getParent()).findViewById(R.id.clipboard_pin_view);
    if (v != null)
    {
      v.refresh_pinned_items();
    }
  }

  /** Send the specified entry to the editor. */
  public void paste_entry(int pos)
  {
    ClipboardHistoryService.paste(_filteredHistory.get(pos).content);
  }

  @Override
  public void on_clipboard_history_change()
  {
    update_data();
  }

  @Override
  protected void onWindowVisibilityChanged(int visibility)
  {
    if (visibility == View.VISIBLE)
      update_data();
  }

  void update_data()
  {
    _history = _service.clear_expired_and_get_history();
    applyFilter(); // Reapply current search filter
  }

  /** Date filter methods */
  public boolean isDateFilterEnabled()
  {
    return _dateFilterEnabled;
  }

  public boolean isDateFilterBefore()
  {
    return _dateFilterBefore;
  }

  public long getDateFilterTimestamp()
  {
    return _dateFilterTimestamp;
  }

  public void setDateFilter(long timestamp, boolean isBefore)
  {
    _dateFilterEnabled = true;
    _dateFilterTimestamp = timestamp;
    _dateFilterBefore = isBefore;
    applyFilter();
  }

  public void clearDateFilter()
  {
    _dateFilterEnabled = false;
    _dateFilterTimestamp = 0;
    _dateFilterBefore = false;
    applyFilter();
  }

  class ClipboardEntriesAdapter extends BaseAdapter
  {
    public ClipboardEntriesAdapter() {}

    @Override
    public int getCount() { return _filteredHistory.size(); }
    @Override
    public Object getItem(int pos) { return _filteredHistory.get(pos); }
    @Override
    public long getItemId(int pos) { return _filteredHistory.get(pos).hashCode(); }

    @Override
    public View getView(final int pos, View v, ViewGroup _parent)
    {
      if (v == null)
        v = View.inflate(getContext(), R.layout.clipboard_history_entry, null);

      final ClipboardEntry entry = _filteredHistory.get(pos);
      final String text = entry.content;
      final TextView textView = (TextView)v.findViewById(R.id.clipboard_entry_text);
      final View expandButton = v.findViewById(R.id.clipboard_entry_expand);

      // Set text with timestamp appended
      textView.setText(entry.getFormattedText(getContext()));

      // Check if text contains newlines (multi-line)
      final boolean isMultiLine = text.contains("\n");
      final boolean isExpanded = _expandedStates.containsKey(pos) && _expandedStates.get(pos);

      // Set maxLines based on expanded state (applies to all entries)
      if (isExpanded)
      {
        textView.setMaxLines(Integer.MAX_VALUE);
        textView.setEllipsize(null);
      }
      else
      {
        textView.setMaxLines(1);
        textView.setEllipsize(android.text.TextUtils.TruncateAt.END);
      }

      // Show expand button only for multi-line entries
      if (isMultiLine)
      {
        expandButton.setVisibility(View.VISIBLE);
        expandButton.setRotation(isExpanded ? 180 : 0);

        // Handle expand button click for multi-line entries
        expandButton.setOnClickListener(new View.OnClickListener()
        {
          @Override
          public void onClick(View v)
          {
            _expandedStates.put(pos, !isExpanded);
            notifyDataSetChanged();
          }
        });
      }
      else
      {
        expandButton.setVisibility(View.GONE);
      }

      // Make text clickable to expand/collapse (all entries)
      textView.setOnClickListener(new View.OnClickListener()
      {
        @Override
        public void onClick(View v)
        {
          _expandedStates.put(pos, !isExpanded);
          notifyDataSetChanged();
        }
      });

      v.findViewById(R.id.clipboard_entry_addpin).setOnClickListener(
          new View.OnClickListener()
          {
            @Override
            public void onClick(View v) { pin_entry(pos); }
          });
      v.findViewById(R.id.clipboard_entry_paste).setOnClickListener(
          new View.OnClickListener()
          {
            @Override
            public void onClick(View v) { paste_entry(pos); }
          });
      return v;
    }
  }
}
