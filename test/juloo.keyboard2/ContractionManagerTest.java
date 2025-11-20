package juloo.keyboard2;

import static org.junit.Assert.*;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.RuntimeEnvironment;
import android.content.Context;

/**
 * Unit tests for ContractionManager possessive generation.
 * 
 * Tests the hybrid contraction system (v1.32.544):
 * - True contractions loaded from binary
 * - Possessives generated dynamically
 */
@RunWith(RobolectricTestRunner.class)
public class ContractionManagerTest
{
  private ContractionManager _manager;
  private Context _context;

  @Before
  public void setUp()
  {
    _context = RuntimeEnvironment.application;
    _manager = new ContractionManager(_context);
    _manager.loadMappings();
  }

  @Test
  public void testGeneratePossessive_RegularNouns()
  {
    // Regular nouns should generate possessives
    assertEquals("cat's", _manager.generatePossessive("cat"));
    assertEquals("dog's", _manager.generatePossessive("dog"));
    assertEquals("house's", _manager.generatePossessive("house"));
    assertEquals("book's", _manager.generatePossessive("book"));
  }

  @Test
  public void testGeneratePossessive_NounsEndingInS()
  {
    // Modern style: words ending in 's' get 's (not just ')
    assertEquals("James's", _manager.generatePossessive("James"));
    assertEquals("Charles's", _manager.generatePossessive("Charles"));
    assertEquals("boss's", _manager.generatePossessive("boss"));
  }

  @Test
  public void testGeneratePossessive_Pronouns()
  {
    // Pronouns should NOT generate possessives (handled by true contractions)
    assertNull(_manager.generatePossessive("i"));
    assertNull(_manager.generatePossessive("you"));
    assertNull(_manager.generatePossessive("he"));
    assertNull(_manager.generatePossessive("she"));
    assertNull(_manager.generatePossessive("it"));
    assertNull(_manager.generatePossessive("we"));
    assertNull(_manager.generatePossessive("they"));
  }

  @Test
  public void testGeneratePossessive_FunctionWords()
  {
    // Function words should NOT generate possessives
    assertNull(_manager.generatePossessive("will"));
    assertNull(_manager.generatePossessive("would"));
    assertNull(_manager.generatePossessive("can"));
    assertNull(_manager.generatePossessive("could"));
    assertNull(_manager.generatePossessive("do"));
    assertNull(_manager.generatePossessive("does"));
    assertNull(_manager.generatePossessive("is"));
    assertNull(_manager.generatePossessive("are"));
    assertNull(_manager.generatePossessive("have"));
    assertNull(_manager.generatePossessive("has"));
  }

  @Test
  public void testGeneratePossessive_KnownContractions()
  {
    // Known contractions should NOT generate possessives
    // (don't → don't's is invalid)
    assertNull(_manager.generatePossessive("don't"));
    assertNull(_manager.generatePossessive("won't"));
    assertNull(_manager.generatePossessive("can't"));
    assertNull(_manager.generatePossessive("we'll"));
  }

  @Test
  public void testGeneratePossessive_EmptyAndNull()
  {
    assertNull(_manager.generatePossessive(""));
    assertNull(_manager.generatePossessive(null));
  }

  @Test
  public void testGeneratePossessive_CaseInsensitive()
  {
    // Should work regardless of case
    assertEquals("Cat's", _manager.generatePossessive("Cat"));
    assertEquals("DOG's", _manager.generatePossessive("DOG"));
    assertEquals("MiXeD's", _manager.generatePossessive("MiXeD"));
  }

  @Test
  public void testShouldGeneratePossessive()
  {
    // Regular nouns
    assertTrue(_manager.shouldGeneratePossessive("cat"));
    assertTrue(_manager.shouldGeneratePossessive("dog"));
    
    // Pronouns
    assertFalse(_manager.shouldGeneratePossessive("i"));
    assertFalse(_manager.shouldGeneratePossessive("you"));
    
    // Contractions
    assertFalse(_manager.shouldGeneratePossessive("don't"));
    assertFalse(_manager.shouldGeneratePossessive("won't"));
    
    // Empty/null
    assertFalse(_manager.shouldGeneratePossessive(""));
    assertFalse(_manager.shouldGeneratePossessive(null));
  }

  @Test
  public void testTrueContractions_StillLoaded()
  {
    // Verify true contractions are still loaded from binary
    assertTrue(_manager.isKnownContraction("don't"));
    assertTrue(_manager.isKnownContraction("won't"));
    assertTrue(_manager.isKnownContraction("can't"));
    assertTrue(_manager.isKnownContraction("we'll"));
    assertTrue(_manager.isKnownContraction("i'm"));
    assertTrue(_manager.isKnownContraction("you're"));
  }

  @Test
  public void testNonPairedContractions_StillWork()
  {
    // Verify non-paired contractions mapping still works
    assertEquals("don't", _manager.getNonPairedMapping("dont"));
    assertEquals("can't", _manager.getNonPairedMapping("cant"));
    assertEquals("won't", _manager.getNonPairedMapping("wont"));
  }
}
