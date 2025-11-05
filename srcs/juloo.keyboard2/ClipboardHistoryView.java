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
  List<String> _history;
  List<String> _filteredHistory;
  String _searchFilter = "";
  ClipboardHistoryService _service;
  ClipboardEntriesAdapter _adapter;

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
    if (_searchFilter.isEmpty())
    {
      _filteredHistory = _history;
    }
    else
    {
      List<String> filtered = new ArrayList<String>();
      for (String item : _history)
      {
        if (item.toLowerCase().contains(_searchFilter))
        {
          filtered.add(item);
        }
      }
      _filteredHistory = filtered;
    }
    _adapter.notifyDataSetChanged();
    invalidate();
  }

  /** The history entry at index [pos] is removed from the history and added to
      the list of pinned clipboards. */
  public void pin_entry(int pos)
  {
    String clip = _filteredHistory.get(pos);

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
    ClipboardHistoryService.paste(_filteredHistory.get(pos));
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
      ((TextView)v.findViewById(R.id.clipboard_entry_text))
        .setText(_filteredHistory.get(pos));
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
