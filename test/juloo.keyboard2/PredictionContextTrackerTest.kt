package juloo.keyboard2

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Comprehensive test suite for PredictionContextTracker.
 *
 * Tests cover:
 * - Current word tracking (append, get, clear, delete)
 * - Context word history (commit, max size, retrieval)
 * - Swipe gesture tracking
 * - Auto-inserted word tracking
 * - Commit source tracking
 * - Complete state management
 * - Edge cases and boundary conditions
 */
class PredictionContextTrackerTest {

    private lateinit var tracker: PredictionContextTracker

    @Before
    fun setUp() {
        tracker = PredictionContextTracker()
    }

    // Current Word Tests

    @Test
    fun testGetCurrentWord_initially_isEmpty() {
        // Assert
        assertEquals("", tracker.getCurrentWord())
        assertEquals(0, tracker.getCurrentWordLength())
    }

    @Test
    fun testAppendToCurrentWord_withSingleChar_appendsChar() {
        // Act
        tracker.appendToCurrentWord("h")

        // Assert
        assertEquals("h", tracker.getCurrentWord())
        assertEquals(1, tracker.getCurrentWordLength())
    }

    @Test
    fun testAppendToCurrentWord_withMultipleChars_buildsWord() {
        // Act
        tracker.appendToCurrentWord("h")
        tracker.appendToCurrentWord("e")
        tracker.appendToCurrentWord("l")
        tracker.appendToCurrentWord("l")
        tracker.appendToCurrentWord("o")

        // Assert
        assertEquals("hello", tracker.getCurrentWord())
        assertEquals(5, tracker.getCurrentWordLength())
    }

    @Test
    fun testAppendToCurrentWord_withString_appendsEntireString() {
        // Act
        tracker.appendToCurrentWord("hello")

        // Assert
        assertEquals("hello", tracker.getCurrentWord())
        assertEquals(5, tracker.getCurrentWordLength())
    }

    @Test
    fun testClearCurrentWord_resetsWord() {
        // Arrange
        tracker.appendToCurrentWord("hello")

        // Act
        tracker.clearCurrentWord()

        // Assert
        assertEquals("", tracker.getCurrentWord())
        assertEquals(0, tracker.getCurrentWordLength())
    }

    @Test
    fun testDeleteLastChar_removesLastCharacter() {
        // Arrange
        tracker.appendToCurrentWord("hello")

        // Act
        tracker.deleteLastChar()

        // Assert
        assertEquals("hell", tracker.getCurrentWord())
        assertEquals(4, tracker.getCurrentWordLength())
    }

    @Test
    fun testDeleteLastChar_withEmptyWord_doesNothing() {
        // Act
        tracker.deleteLastChar()

        // Assert
        assertEquals("", tracker.getCurrentWord())
        assertEquals(0, tracker.getCurrentWordLength())
    }

    @Test
    fun testDeleteLastChar_multipleDeletes_removesAllChars() {
        // Arrange
        tracker.appendToCurrentWord("hi")

        // Act
        tracker.deleteLastChar()
        tracker.deleteLastChar()
        tracker.deleteLastChar() // Extra delete on empty

        // Assert
        assertEquals("", tracker.getCurrentWord())
        assertEquals(0, tracker.getCurrentWordLength())
    }

    // Context Word Tests

    @Test
    fun testGetContextWords_initially_isEmpty() {
        // Assert
        assertTrue(tracker.getContextWords().isEmpty())
    }

    @Test
    fun testCommitWord_addsToContext() {
        // Act
        tracker.commitWord("hello", PredictionSource.USER_TYPED_TAP, false)

        // Assert
        val context = tracker.getContextWords()
        assertEquals(1, context.size)
        assertEquals("hello", context[0])
    }

    @Test
    fun testCommitWord_lowercasesWord() {
        // Act
        tracker.commitWord("HELLO", PredictionSource.USER_TYPED_TAP, false)

        // Assert
        val context = tracker.getContextWords()
        assertEquals("hello", context[0])
    }

    @Test
    fun testCommitWord_multipleWords_maintainsOrder() {
        // Act
        tracker.commitWord("the", PredictionSource.USER_TYPED_TAP, false)
        tracker.commitWord("quick", PredictionSource.USER_TYPED_TAP, false)
        tracker.commitWord("brown", PredictionSource.USER_TYPED_TAP, false)

        // Assert
        val context = tracker.getContextWords()
        assertEquals(2, context.size) // MAX_CONTEXT_WORDS = 2
        assertEquals("quick", context[0])
        assertEquals("brown", context[1])
    }

    @Test
    fun testCommitWord_exceedsMaxContext_removesOldest() {
        // Act
        tracker.commitWord("first", PredictionSource.USER_TYPED_TAP, false)
        tracker.commitWord("second", PredictionSource.USER_TYPED_TAP, false)
        tracker.commitWord("third", PredictionSource.USER_TYPED_TAP, false)
        tracker.commitWord("fourth", PredictionSource.USER_TYPED_TAP, false)

        // Assert
        val context = tracker.getContextWords()
        assertEquals(2, context.size)
        assertEquals("third", context[0])
        assertEquals("fourth", context[1])
        assertFalse(context.contains("first"))
        assertFalse(context.contains("second"))
    }

    @Test
    fun testCommitWord_clearsCurrentWord() {
        // Arrange
        tracker.appendToCurrentWord("hello")

        // Act
        tracker.commitWord("hello", PredictionSource.USER_TYPED_TAP, false)

        // Assert
        assertEquals("", tracker.getCurrentWord())
        assertEquals(0, tracker.getCurrentWordLength())
    }

    @Test
    fun testGetContextWords_returnsImmutableCopy() {
        // Arrange
        tracker.commitWord("hello", PredictionSource.USER_TYPED_TAP, false)
        val context1 = tracker.getContextWords()

        // Act - modify returned list
        val mutableContext = context1.toMutableList()
        mutableContext.add("world")

        // Assert - original context unchanged
        val context2 = tracker.getContextWords()
        assertEquals(1, context2.size)
        assertEquals("hello", context2[0])
    }

    // Swipe Tracking Tests

    @Test
    fun testWasLastInputSwipe_initially_isFalse() {
        // Assert
        assertFalse(tracker.wasLastInputSwipe())
    }

    @Test
    fun testSetWasLastInputSwipe_withTrue_setsFlag() {
        // Act
        tracker.setWasLastInputSwipe(true)

        // Assert
        assertTrue(tracker.wasLastInputSwipe())
    }

    @Test
    fun testSetWasLastInputSwipe_withFalse_clearsFlag() {
        // Arrange
        tracker.setWasLastInputSwipe(true)

        // Act
        tracker.setWasLastInputSwipe(false)

        // Assert
        assertFalse(tracker.wasLastInputSwipe())
    }

    @Test
    fun testSetWasLastInputSwipe_toggleMultipleTimes() {
        // Act & Assert
        tracker.setWasLastInputSwipe(true)
        assertTrue(tracker.wasLastInputSwipe())

        tracker.setWasLastInputSwipe(false)
        assertFalse(tracker.wasLastInputSwipe())

        tracker.setWasLastInputSwipe(true)
        assertTrue(tracker.wasLastInputSwipe())
    }

    // Auto-Inserted Word Tests

    @Test
    fun testGetLastAutoInsertedWord_initially_isNull() {
        // Assert
        assertNull(tracker.getLastAutoInsertedWord())
    }

    @Test
    fun testCommitWord_withAutoInserted_tracksWord() {
        // Act
        tracker.commitWord("hello", PredictionSource.NEURAL_SWIPE, true)

        // Assert
        assertEquals("hello", tracker.getLastAutoInsertedWord())
    }

    @Test
    fun testCommitWord_withoutAutoInsert_clearsTracking() {
        // Arrange
        tracker.commitWord("hello", PredictionSource.NEURAL_SWIPE, true)

        // Act
        tracker.commitWord("world", PredictionSource.USER_TYPED_TAP, false)

        // Assert
        assertNull(tracker.getLastAutoInsertedWord())
    }

    @Test
    fun testSetLastAutoInsertedWord_setsWord() {
        // Act
        tracker.setLastAutoInsertedWord("test")

        // Assert
        assertEquals("test", tracker.getLastAutoInsertedWord())
    }

    @Test
    fun testClearLastAutoInsertedWord_clearsWord() {
        // Arrange
        tracker.setLastAutoInsertedWord("test")

        // Act
        tracker.clearLastAutoInsertedWord()

        // Assert
        assertNull(tracker.getLastAutoInsertedWord())
    }

    // Commit Source Tests

    @Test
    fun testGetLastCommitSource_initially_isUnknown() {
        // Assert
        assertEquals(PredictionSource.UNKNOWN, tracker.getLastCommitSource())
    }

    @Test
    fun testCommitWord_tracksSource() {
        // Act
        tracker.commitWord("hello", PredictionSource.NEURAL_SWIPE, false)

        // Assert
        assertEquals(PredictionSource.NEURAL_SWIPE, tracker.getLastCommitSource())
    }

    @Test
    fun testSetLastCommitSource_setsSource() {
        // Act
        tracker.setLastCommitSource(PredictionSource.CANDIDATE_SELECTION)

        // Assert
        assertEquals(PredictionSource.CANDIDATE_SELECTION, tracker.getLastCommitSource())
    }

    @Test
    fun testCommitWord_multipleCommits_tracksLatestSource() {
        // Act
        tracker.commitWord("first", PredictionSource.USER_TYPED_TAP, false)
        tracker.commitWord("second", PredictionSource.NEURAL_SWIPE, true)
        tracker.commitWord("third", PredictionSource.CANDIDATE_SELECTION, false)

        // Assert
        assertEquals(PredictionSource.CANDIDATE_SELECTION, tracker.getLastCommitSource())
    }

    // Clear All Tests

    @Test
    fun testClearAll_resetsAllState() {
        // Arrange - set all state
        tracker.appendToCurrentWord("typing")
        tracker.commitWord("first", PredictionSource.USER_TYPED_TAP, false)
        tracker.commitWord("second", PredictionSource.NEURAL_SWIPE, true)
        tracker.setWasLastInputSwipe(true)

        // Act
        tracker.clearAll()

        // Assert
        assertEquals("", tracker.getCurrentWord())
        assertEquals(0, tracker.getCurrentWordLength())
        assertTrue(tracker.getContextWords().isEmpty())
        assertFalse(tracker.wasLastInputSwipe())
        assertNull(tracker.getLastAutoInsertedWord())
        assertEquals(PredictionSource.UNKNOWN, tracker.getLastCommitSource())
    }

    @Test
    fun testClearAll_withEmptyState_doesNotCrash() {
        // Act - should not throw
        tracker.clearAll()

        // Assert
        assertEquals("", tracker.getCurrentWord())
        assertTrue(tracker.getContextWords().isEmpty())
    }

    // Integration Tests

    @Test
    fun testTypingWorkflow_completeWord() {
        // Simulate typing "hello"
        tracker.appendToCurrentWord("h")
        tracker.appendToCurrentWord("e")
        tracker.appendToCurrentWord("l")
        tracker.appendToCurrentWord("l")
        tracker.appendToCurrentWord("o")

        assertEquals("hello", tracker.getCurrentWord())
        assertEquals(5, tracker.getCurrentWordLength())

        // Commit word (user pressed space)
        tracker.commitWord("hello", PredictionSource.USER_TYPED_TAP, false)

        assertEquals("", tracker.getCurrentWord())
        assertEquals(1, tracker.getContextWords().size)
        assertEquals("hello", tracker.getContextWords()[0])
        assertEquals(PredictionSource.USER_TYPED_TAP, tracker.getLastCommitSource())
    }

    @Test
    fun testSwipeWorkflow_autoInsert() {
        // Simulate swipe gesture
        tracker.setWasLastInputSwipe(true)

        // Prediction system auto-inserts "hello"
        tracker.commitWord("hello", PredictionSource.NEURAL_SWIPE, true)

        assertTrue(tracker.wasLastInputSwipe())
        assertEquals("hello", tracker.getLastAutoInsertedWord())
        assertEquals(PredictionSource.NEURAL_SWIPE, tracker.getLastCommitSource())
        assertEquals(1, tracker.getContextWords().size)
    }

    @Test
    fun testBackspaceWorkflow_duringTyping() {
        // Type "hello" then delete 2 chars
        tracker.appendToCurrentWord("hello")
        tracker.deleteLastChar()
        tracker.deleteLastChar()

        assertEquals("hel", tracker.getCurrentWord())
        assertEquals(3, tracker.getCurrentWordLength())
    }

    @Test
    fun testContextBuilding_multipleSentences() {
        // Type "the quick"
        tracker.commitWord("the", PredictionSource.USER_TYPED_TAP, false)
        tracker.commitWord("quick", PredictionSource.USER_TYPED_TAP, false)

        // Context should have ["the", "quick"]
        val context1 = tracker.getContextWords()
        assertEquals(2, context1.size)
        assertEquals("the", context1[0])
        assertEquals("quick", context1[1])

        // Type "brown" (should drop "the")
        tracker.commitWord("brown", PredictionSource.USER_TYPED_TAP, false)

        val context2 = tracker.getContextWords()
        assertEquals(2, context2.size)
        assertEquals("quick", context2[0])
        assertEquals("brown", context2[1])
    }

    @Test
    fun testMixedInputWorkflow_tapAndSwipe() {
        // Tap typing
        tracker.appendToCurrentWord("the")
        tracker.commitWord("the", PredictionSource.USER_TYPED_TAP, false)
        assertFalse(tracker.wasLastInputSwipe())

        // Swipe input
        tracker.setWasLastInputSwipe(true)
        tracker.commitWord("quick", PredictionSource.NEURAL_SWIPE, true)
        assertTrue(tracker.wasLastInputSwipe())
        assertEquals("quick", tracker.getLastAutoInsertedWord())

        // Context should have both
        val context = tracker.getContextWords()
        assertEquals(2, context.size)
        assertEquals("the", context[0])
        assertEquals("quick", context[1])
    }

    // Debug State Tests

    @Test
    fun testGetDebugState_withEmptyState_returnsValidString() {
        // Act
        val debug = tracker.getDebugState()

        // Assert
        assertNotNull(debug)
        assertTrue(debug.contains("PredictionContextTracker"))
        assertTrue(debug.contains("currentWord=''"))
        assertTrue(debug.contains("wasSwipe=false"))
    }

    @Test
    fun testGetDebugState_withFullState_includesAllInfo() {
        // Arrange
        tracker.appendToCurrentWord("typing")
        tracker.commitWord("hello", PredictionSource.NEURAL_SWIPE, true)
        tracker.setWasLastInputSwipe(true)

        // Act
        val debug = tracker.getDebugState()

        // Assert
        assertTrue(debug.contains("currentWord='typing'"))
        assertTrue(debug.contains("contextWords="))
        assertTrue(debug.contains("wasSwipe=true"))
        assertTrue(debug.contains("lastAutoInsert='hello'"))
        assertTrue(debug.contains("lastSource="))
    }

    // Edge Cases

    @Test
    fun testAppendToCurrentWord_withEmptyString_doesNotCrash() {
        // Act
        tracker.appendToCurrentWord("")

        // Assert
        assertEquals("", tracker.getCurrentWord())
    }

    @Test
    fun testCommitWord_withEmptyString_addsToContext() {
        // Act
        tracker.commitWord("", PredictionSource.USER_TYPED_TAP, false)

        // Assert
        assertEquals(1, tracker.getContextWords().size)
        assertEquals("", tracker.getContextWords()[0])
    }

    @Test
    fun testCommitWord_withWhitespace_preservesInContext() {
        // Act
        tracker.commitWord("  hello  ", PredictionSource.USER_TYPED_TAP, false)

        // Assert
        // Note: commitWord lowercases but doesn't trim
        assertEquals("  hello  ", tracker.getContextWords()[0])
    }

    @Test
    fun testMultipleClearCurrentWord_doesNotCrash() {
        // Act
        tracker.clearCurrentWord()
        tracker.clearCurrentWord()
        tracker.clearCurrentWord()

        // Assert
        assertEquals("", tracker.getCurrentWord())
    }
}
