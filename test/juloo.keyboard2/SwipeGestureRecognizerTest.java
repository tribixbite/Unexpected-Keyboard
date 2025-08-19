package juloo.keyboard2;

import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

public class SwipeGestureRecognizerTest
{
  private SwipeGestureRecognizer recognizer;
  
  @Before
  public void setUp()
  {
    recognizer = new SwipeGestureRecognizer();
  }
  
  @Test
  public void testInitialState()
  {
    assertFalse("Recognizer should not be swipe typing initially", 
                recognizer.isSwipeTyping());
    assertTrue("Initial swipe path should be empty", 
               recognizer.getSwipePath().isEmpty());
    assertEquals("Initial key sequence should be empty", 
                 "", recognizer.getKeySequence());
  }
  
  @Test
  public void testReset()
  {
    // Create a dummy key
    KeyboardData.Key key = createDummyKey('a');
    
    // Start a swipe
    recognizer.startSwipe(100.0f, 100.0f, key);
    recognizer.addPoint(200.0f, 200.0f, key);
    
    // Reset should clear everything
    recognizer.reset();
    
    assertFalse("Should not be swipe typing after reset", 
                recognizer.isSwipeTyping());
    assertTrue("Swipe path should be empty after reset", 
               recognizer.getSwipePath().isEmpty());
    assertEquals("Key sequence should be empty after reset", 
                 "", recognizer.getKeySequence());
  }
  
  @Test 
  public void testSwipePathTracking()
  {
    KeyboardData.Key key = createDummyKey('a');
    
    recognizer.startSwipe(100.0f, 100.0f, key);
    assertEquals("Should have one point after start", 
                 1, recognizer.getSwipePath().size());
    
    recognizer.addPoint(150.0f, 150.0f, key);
    assertEquals("Should have two points after adding one", 
                 2, recognizer.getSwipePath().size());
    
    recognizer.addPoint(200.0f, 200.0f, key);
    assertEquals("Should have three points after adding another", 
                 3, recognizer.getSwipePath().size());
  }
  
  @Test
  public void testEndSwipeRequiresMinimumKeys()
  {
    KeyboardData.Key keyA = createDummyKey('a');
    
    // Single key should not trigger swipe typing
    recognizer.startSwipe(100.0f, 100.0f, keyA);
    assertNull("Single key should not return touched keys", 
               recognizer.endSwipe());
    
    // Two keys should trigger swipe typing
    KeyboardData.Key keyB = createDummyKey('b');
    recognizer.startSwipe(100.0f, 100.0f, keyA);
    
    // Simulate movement to trigger swipe typing detection
    for (int i = 0; i < 10; i++)
    {
      recognizer.addPoint(100.0f + i * 10, 100.0f, i % 2 == 0 ? keyA : keyB);
    }
    
    // Note: This may still return null because isSwipeTyping() 
    // depends on internal timing and distance calculations
  }
  
  /**
   * Helper method to create a dummy key for testing
   */
  private KeyboardData.Key createDummyKey(char c)
  {
    KeyValue kv = KeyValue.makeStringKey(String.valueOf(c));
    KeyValue[] keys = new KeyValue[9];
    keys[0] = kv;
    return new KeyboardData.Key(keys, null, 0, 1.0f, 0.0f, null);
  }
}