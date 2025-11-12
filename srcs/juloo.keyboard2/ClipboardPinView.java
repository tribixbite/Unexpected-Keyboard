package juloo.keyboard2;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.util.AttributeSet;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONException;

public final class ClipboardPinView extends MaxHeightListView
{
  List<String> _entries;
  ClipboardPinEntriesAdapter _adapter;
  ClipboardHistoryService _service;
  // Track expanded state: position -> isExpanded
  java.util.Map<Integer, Boolean> _expandedStates = new java.util.HashMap<>();

  public ClipboardPinView(Context ctx, AttributeSet attrs)
  {
    super(ctx, attrs);
    _entries = new ArrayList<String>();
    _service = ClipboardHistoryService.get_service(ctx);
    _adapter = this.new ClipboardPinEntriesAdapter();
    setAdapter(_adapter);
    refresh_pinned_items();
  }

  /** Refresh pinned items from database */
  public void refresh_pinned_items()
  {
    if (_service != null)
    {
      _entries = _service.get_pinned_entries();
      _adapter.notifyDataSetChanged();
      invalidate();

      // Set minimum height on parent ScrollView if 2+ items exist
      // This ensures 2 entries are visible without scrolling
      updateParentMinHeight();
    }
  }

  /** Update parent ScrollView minHeight based on item count */
  private void updateParentMinHeight()
  {
    ViewGroup parent = (ViewGroup)getParent();
    if (parent != null)
    {
      if (_entries.size() >= 2)
      {
        // Set minHeight to show 2 entries (approximately 200dp per entry)
        int minHeightPx = (int)(400 * getResources().getDisplayMetrics().density);
        parent.setMinimumHeight(minHeightPx);
      }
      else
      {
        // Clear minHeight when less than 2 items
        parent.setMinimumHeight(0);
      }
    }
  }

  /** Remove the entry at index [pos] entirely from database. */
  public void remove_entry(int pos)
  {
    if (pos < 0 || pos >= _entries.size())
      return;

    String clip = _entries.get(pos);

    // Delete entirely from database
    if (_service != null)
    {
      _service.remove_history_entry(clip);
    }

    refresh_pinned_items();
  }

  /** Send the specified entry to the editor. */
  public void paste_entry(int pos)
  {
    ClipboardHistoryService.paste(_entries.get(pos));
  }

  @Override
  protected void onWindowVisibilityChanged(int visibility)
  {
    if (visibility == View.VISIBLE)
      refresh_pinned_items();
  }

  class ClipboardPinEntriesAdapter extends BaseAdapter
  {
    public ClipboardPinEntriesAdapter() {}

    @Override
    public int getCount() { return _entries.size(); }
    @Override
    public Object getItem(int pos) { return _entries.get(pos); }
    @Override
    public long getItemId(int pos) { return _entries.get(pos).hashCode(); }

    @Override
    public View getView(final int pos, View v, ViewGroup _parent)
    {
      if (v == null)
        v = View.inflate(getContext(), R.layout.clipboard_pin_entry, null);

      final String text = _entries.get(pos);
      final TextView textView = (TextView)v.findViewById(R.id.clipboard_pin_text);
      final View expandButton = v.findViewById(R.id.clipboard_pin_expand);

      textView.setText(text);

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

      v.findViewById(R.id.clipboard_pin_paste).setOnClickListener(
          new View.OnClickListener()
          {
            @Override
            public void onClick(View v) { paste_entry(pos); }
          });
      v.findViewById(R.id.clipboard_pin_remove).setOnClickListener(
          new View.OnClickListener()
          {
            @Override
            public void onClick(View v)
            {
              AlertDialog d = new AlertDialog.Builder(getContext())
                .setTitle(R.string.clipboard_remove_confirm)
                .setPositiveButton(R.string.clipboard_remove_confirmed,
                    new DialogInterface.OnClickListener(){
                      public void onClick(DialogInterface _dialog, int _which)
                      {
                        remove_entry(pos);
                      }
                    })
                .setNegativeButton(android.R.string.cancel, null)
                .create();
              Utils.show_dialog_on_ime(d, v.getWindowToken());
            }
          });
      return v;
    }
  }
}
