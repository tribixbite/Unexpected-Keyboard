package juloo.keyboard2;

import android.content.Context;
import android.util.AttributeSet;
import android.view.View;
import android.widget.ListView;

/**
 * A scrollable ListView that limits its maximum height.
 * Used for pinned clipboard items - shows max 2 items with internal scrolling.
 */
public class MaxHeightListView extends ListView
{
  private int _maxHeight = -1;

  public MaxHeightListView(Context context)
  {
    super(context);
  }

  public MaxHeightListView(Context context, AttributeSet attrs)
  {
    super(context, attrs);
    init(context, attrs);
  }

  public MaxHeightListView(Context context, AttributeSet attrs, int defStyle)
  {
    super(context, attrs, defStyle);
    init(context, attrs);
  }

  private void init(Context context, AttributeSet attrs)
  {
    if (attrs != null)
    {
      // Read maxHeight from XML attributes
      for (int i = 0; i < attrs.getAttributeCount(); i++)
      {
        String name = attrs.getAttributeName(i);
        if ("maxHeight".equals(name))
        {
          _maxHeight = attrs.getAttributeIntValue(i, -1);
          break;
        }
      }
    }
  }

  public void setMaxHeight(int maxHeight)
  {
    _maxHeight = maxHeight;
    requestLayout();
  }

  @Override
  protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec)
  {
    if (_maxHeight > 0)
    {
      // Limit height to maxHeight
      int heightMode = View.MeasureSpec.getMode(heightMeasureSpec);
      int heightSize = View.MeasureSpec.getSize(heightMeasureSpec);

      if (heightMode == View.MeasureSpec.UNSPECIFIED || heightSize > _maxHeight)
      {
        heightMeasureSpec = View.MeasureSpec.makeMeasureSpec(_maxHeight, View.MeasureSpec.AT_MOST);
      }
    }

    super.onMeasure(widthMeasureSpec, heightMeasureSpec);
  }
}
