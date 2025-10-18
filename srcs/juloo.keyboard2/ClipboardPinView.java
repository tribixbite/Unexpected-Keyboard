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

public final class ClipboardPinView extends NonScrollListView
{
  List<String> _entries;
  ClipboardPinEntriesAdapter _adapter;
  ClipboardHistoryService _service;

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
    }
  }

  /** Remove the entry at index [pos] and unpin in database. */
  public void remove_entry(int pos)
  {
    if (pos < 0 || pos >= _entries.size())
      return;

    String clip = _entries.get(pos);

    // Unpin in database
    if (_service != null)
    {
      _service.set_pinned_status(clip, false);
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
      ((TextView)v.findViewById(R.id.clipboard_pin_text))
        .setText(_entries.get(pos));
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
