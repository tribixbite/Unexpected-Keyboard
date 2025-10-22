package juloo.keyboard2;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

/**
 * Debug activity for swipe typing pipeline analysis.
 * Displays real-time logging of every step in the swipe prediction process.
 */
public class SwipeDebugActivity extends Activity
{
  public static final String ACTION_DEBUG_LOG = "juloo.keyboard2.DEBUG_LOG";
  public static final String EXTRA_LOG_MESSAGE = "log_message";

  private TextView _logOutput;
  private ScrollView _logScroll;
  private EditText _inputText;
  private Button _copyButton;
  private Button _clearButton;

  private StringBuilder _logBuffer = new StringBuilder();

  private BroadcastReceiver _logReceiver = new BroadcastReceiver()
  {
    @Override
    public void onReceive(Context context, Intent intent)
    {
      if (ACTION_DEBUG_LOG.equals(intent.getAction()))
      {
        String message = intent.getStringExtra(EXTRA_LOG_MESSAGE);
        if (message != null)
        {
          appendLog(message);
        }
      }
    }
  };

  @Override
  protected void onCreate(Bundle savedInstanceState)
  {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.swipe_debug_activity);

    _logOutput = findViewById(R.id.log_output);
    _logScroll = findViewById(R.id.log_scroll);
    _inputText = findViewById(R.id.input_text);
    _copyButton = findViewById(R.id.copy_button);
    _clearButton = findViewById(R.id.clear_button);

    _copyButton.setOnClickListener(new View.OnClickListener()
    {
      @Override
      public void onClick(View v)
      {
        copyLogsToClipboard();
      }
    });

    _clearButton.setOnClickListener(new View.OnClickListener()
    {
      @Override
      public void onClick(View v)
      {
        clearLogs();
      }
    });

    // Request focus for input text
    _inputText.requestFocus();
    _inputText.setFocusableInTouchMode(true);

    // Prevent log output from stealing focus when scrolling
    _logScroll.setDescendantFocusability(android.view.ViewGroup.FOCUS_BEFORE_DESCENDANTS);
    _logOutput.setFocusable(false);

    // Register broadcast receiver for debug logs
    IntentFilter filter = new IntentFilter(ACTION_DEBUG_LOG);
    registerReceiver(_logReceiver, filter, Context.RECEIVER_NOT_EXPORTED);

    // Enable debug mode
    setDebugMode(true);

    appendLog("=== Swipe Debug Session Started ===\n");
    appendLog("Start swiping in the text field above to see pipeline logs.\n\n");
  }

  @Override
  protected void onDestroy()
  {
    super.onDestroy();

    // Disable debug mode
    setDebugMode(false);

    // Unregister broadcast receiver
    try
    {
      unregisterReceiver(_logReceiver);
    }
    catch (Exception e)
    {
      // Already unregistered
    }
  }

  private void appendLog(String message)
  {
    runOnUiThread(new Runnable()
    {
      @Override
      public void run()
      {
        _logBuffer.append(message);
        _logOutput.setText(_logBuffer.toString());

        // Auto-scroll to bottom
        _logScroll.post(new Runnable()
        {
          @Override
          public void run()
          {
            _logScroll.fullScroll(View.FOCUS_DOWN);
          }
        });
      }
    });
  }

  private void clearLogs()
  {
    _logBuffer.setLength(0);
    _logOutput.setText("Logs cleared. Waiting for swipe input...\n");
    Toast.makeText(this, "Logs cleared", Toast.LENGTH_SHORT).show();
  }

  private void copyLogsToClipboard()
  {
    ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
    ClipData clip = ClipData.newPlainText("Swipe Debug Logs", _logBuffer.toString());
    clipboard.setPrimaryClip(clip);
    Toast.makeText(this, "Logs copied to clipboard", Toast.LENGTH_SHORT).show();
  }

  private void setDebugMode(boolean enabled)
  {
    // Broadcast debug mode state to keyboard service
    Intent intent = new Intent("juloo.keyboard2.SET_DEBUG_MODE");
    intent.setPackage(getPackageName());  // Explicit package for broadcast
    intent.putExtra("debug_enabled", enabled);
    sendBroadcast(intent);
  }
}
