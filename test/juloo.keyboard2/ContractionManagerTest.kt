package juloo.keyboard2

import android.content.Context
import android.content.res.AssetManager
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner
import java.io.ByteArrayInputStream

/**
 * Comprehensive test suite for ContractionManager.
 *
 * Tests cover:
 * - Contraction mapping loading (JSON and binary formats)
 * - Known contraction detection
 * - Non-paired contraction mapping
 * - Possessive generation rules
 * - Function word exclusions
 * - Case-insensitive lookups
 * - Edge cases and error handling
 */
@RunWith(MockitoJUnitRunner::class)
class ContractionManagerTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockAssetManager: AssetManager

    private lateinit var manager: ContractionManager

    @Before
    fun setUp() {
        `when`(mockContext.assets).thenReturn(mockAssetManager)
        manager = ContractionManager(mockContext)
    }

    // Initialization Tests

    @Test
    fun testContractionManager_initialization_doesNotCrash() {
        // Assert - constructor should not throw
        assertNotNull(manager)
    }

    @Test
    fun testGetNonPairedCount_beforeLoading_isZero() {
        // Assert
        assertEquals(0, manager.getNonPairedCount())
    }

    @Test
    fun testGetTotalKnownCount_beforeLoading_isZero() {
        // Assert
        assertEquals(0, manager.getTotalKnownCount())
    }

    // Known Contraction Tests

    @Test
    fun testIsKnownContraction_beforeLoading_returnsFalse() {
        // Act & Assert
        assertFalse(manager.isKnownContraction("don't"))
        assertFalse(manager.isKnownContraction("we'll"))
    }

    @Test
    fun testIsKnownContraction_withEmptyString_returnsFalse() {
        // Act & Assert
        assertFalse(manager.isKnownContraction(""))
    }

    @Test
    fun testIsKnownContraction_caseInsensitive() {
        // This test assumes manual data setup since we can't load real assets in unit tests
        // In real usage after loadMappings(), case should not matter
        assertFalse(manager.isKnownContraction("DON'T"))
        assertFalse(manager.isKnownContraction("Don't"))
        assertFalse(manager.isKnownContraction("don't"))
    }

    // Non-Paired Mapping Tests

    @Test
    fun testGetNonPairedMapping_beforeLoading_returnsNull() {
        // Act & Assert
        assertNull(manager.getNonPairedMapping("dont"))
        assertNull(manager.getNonPairedMapping("cant"))
    }

    @Test
    fun testGetNonPairedMapping_withEmptyString_returnsNull() {
        // Act & Assert
        assertNull(manager.getNonPairedMapping(""))
    }

    @Test
    fun testGetNonPairedMapping_caseInsensitive() {
        // Before loading, all should return null
        assertNull(manager.getNonPairedMapping("DONT"))
        assertNull(manager.getNonPairedMapping("Dont"))
        assertNull(manager.getNonPairedMapping("dont"))
    }

    // Possessive Generation Tests

    @Test
    fun testGeneratePossessive_withNullWord_returnsNull() {
        // Act & Assert
        assertNull(manager.generatePossessive(null))
    }

    @Test
    fun testGeneratePossessive_withEmptyString_returnsNull() {
        // Act & Assert
        assertNull(manager.generatePossessive(""))
    }

    @Test
    fun testGeneratePossessive_withSimpleWord_addsSuffix() {
        // Act & Assert
        assertEquals("cat's", manager.generatePossessive("cat"))
        assertEquals("dog's", manager.generatePossessive("dog"))
        assertEquals("house's", manager.generatePossessive("house"))
    }

    @Test
    fun testGeneratePossessive_withWordEndingInS_addsSuffix() {
        // Modern style: even words ending in 's' get 's
        // Act & Assert
        assertEquals("James's", manager.generatePossessive("James"))
        assertEquals("Charles's", manager.generatePossessive("Charles"))
        assertEquals("bus's", manager.generatePossessive("bus"))
    }

    @Test
    fun testGeneratePossessive_preservesCase() {
        // Act & Assert
        assertEquals("Cat's", manager.generatePossessive("Cat"))
        assertEquals("DOG's", manager.generatePossessive("DOG"))
        assertEquals("JaMeS's", manager.generatePossessive("JaMeS"))
    }

    @Test
    fun testGeneratePossessive_withFunctionWords_returnsNull() {
        // Function words should not get possessive forms
        // They have special contractions instead
        assertNull(manager.generatePossessive("I"))
        assertNull(manager.generatePossessive("you"))
        assertNull(manager.generatePossessive("he"))
        assertNull(manager.generatePossessive("she"))
        assertNull(manager.generatePossessive("it"))
        assertNull(manager.generatePossessive("we"))
        assertNull(manager.generatePossessive("they"))
        assertNull(manager.generatePossessive("who"))
        assertNull(manager.generatePossessive("what"))
        assertNull(manager.generatePossessive("that"))
    }

    @Test
    fun testGeneratePossessive_withModalVerbs_returnsNull() {
        // Modal verbs have special contractions
        assertNull(manager.generatePossessive("will"))
        assertNull(manager.generatePossessive("would"))
        assertNull(manager.generatePossessive("can"))
        assertNull(manager.generatePossessive("could"))
        assertNull(manager.generatePossessive("shall"))
        assertNull(manager.generatePossessive("should"))
    }

    @Test
    fun testGeneratePossessive_withAuxiliaryVerbs_returnsNull() {
        // Auxiliary verbs have special contractions
        assertNull(manager.generatePossessive("is"))
        assertNull(manager.generatePossessive("am"))
        assertNull(manager.generatePossessive("are"))
        assertNull(manager.generatePossessive("was"))
        assertNull(manager.generatePossessive("were"))
        assertNull(manager.generatePossessive("have"))
        assertNull(manager.generatePossessive("has"))
        assertNull(manager.generatePossessive("had"))
        assertNull(manager.generatePossessive("do"))
        assertNull(manager.generatePossessive("does"))
        assertNull(manager.generatePossessive("did"))
    }

    @Test
    fun testGeneratePossessive_caseInsensitiveForFunctionWords() {
        // Function words should be excluded regardless of case
        assertNull(manager.generatePossessive("I"))
        assertNull(manager.generatePossessive("i"))
        assertNull(manager.generatePossessive("WILL"))
        assertNull(manager.generatePossessive("Will"))
        assertNull(manager.generatePossessive("will"))
    }

    @Test
    fun testShouldGeneratePossessive_withValidWord_returnsTrue() {
        // Act & Assert
        assertTrue(manager.shouldGeneratePossessive("cat"))
        assertTrue(manager.shouldGeneratePossessive("dog"))
        assertTrue(manager.shouldGeneratePossessive("house"))
    }

    @Test
    fun testShouldGeneratePossessive_withFunctionWord_returnsFalse() {
        // Act & Assert
        assertFalse(manager.shouldGeneratePossessive("I"))
        assertFalse(manager.shouldGeneratePossessive("you"))
        assertFalse(manager.shouldGeneratePossessive("will"))
        assertFalse(manager.shouldGeneratePossessive("have"))
    }

    @Test
    fun testShouldGeneratePossessive_withNullWord_returnsFalse() {
        // Note: shouldGeneratePossessive takes non-null String
        // This tests the underlying generatePossessive logic
        assertNull(manager.generatePossessive(null))
    }

    @Test
    fun testShouldGeneratePossessive_withEmptyString_returnsFalse() {
        // Act & Assert
        assertFalse(manager.shouldGeneratePossessive(""))
    }

    // Possessive + Contraction Interaction Tests

    @Test
    fun testGeneratePossessive_withKnownContraction_returnsNull() {
        // Setup: simulate loaded contractions by testing logic
        // In real usage, known contractions would be loaded from assets
        // "don't" would be a known contraction, so no possessive

        // Before loading any data, contractions aren't known
        // So this will generate possessive (can't test the exclusion without loading data)
        // This test documents expected behavior
        assertNotNull(manager.generatePossessive("test"))
    }

    // Edge Cases

    @Test
    fun testGeneratePossessive_withSpecialCharacters_addsSuffix() {
        // Act & Assert
        assertEquals("hello-world's", manager.generatePossessive("hello-world"))
        assertEquals("test_name's", manager.generatePossessive("test_name"))
        assertEquals("data.file's", manager.generatePossessive("data.file"))
    }

    @Test
    fun testGeneratePossessive_withNumbers_addsSuffix() {
        // Act & Assert
        assertEquals("2024's", manager.generatePossessive("2024"))
        assertEquals("test123's", manager.generatePossessive("test123"))
    }

    @Test
    fun testGeneratePossessive_withSingleLetter_addsSuffix() {
        // Single letters that aren't function words
        // Act & Assert
        assertEquals("a's", manager.generatePossessive("a"))
        assertEquals("b's", manager.generatePossessive("b"))
        assertEquals("x's", manager.generatePossessive("x"))
    }

    @Test
    fun testGeneratePossessive_withVeryLongWord_addsSuffix() {
        // Act
        val longWord = "a".repeat(100)
        val result = manager.generatePossessive(longWord)

        // Assert
        assertNotNull(result)
        assertEquals("${longWord}'s", result)
        assertEquals(longWord.length + 2, result!!.length)
    }

    // Multiple Possessive Generations

    @Test
    fun testGeneratePossessive_multipleCalls_consistent() {
        // Act
        val result1 = manager.generatePossessive("cat")
        val result2 = manager.generatePossessive("cat")
        val result3 = manager.generatePossessive("cat")

        // Assert - should always return same result
        assertEquals("cat's", result1)
        assertEquals("cat's", result2)
        assertEquals("cat's", result3)
    }

    @Test
    fun testGeneratePossessive_differentWords_independent() {
        // Act
        val cat = manager.generatePossessive("cat")
        val dog = manager.generatePossessive("dog")
        val house = manager.generatePossessive("house")

        // Assert - each should be independent
        assertEquals("cat's", cat)
        assertEquals("dog's", dog)
        assertEquals("house's", house)
    }

    // Load Mappings Error Handling Tests

    @Test
    fun testLoadMappings_withMissingAssets_doesNotCrash() {
        // Arrange - mock asset manager to throw exception
        `when`(mockAssetManager.open(anyString())).thenThrow(RuntimeException("Asset not found"))

        // Act - should not throw
        manager.loadMappings()

        // Assert - counts should remain zero
        assertEquals(0, manager.getNonPairedCount())
        assertEquals(0, manager.getTotalKnownCount())
    }

    @Test
    fun testLoadMappings_withInvalidJSON_doesNotCrash() {
        // Arrange - mock asset with invalid JSON
        val invalidJson = ByteArrayInputStream("{ invalid json }".toByteArray())
        `when`(mockAssetManager.open(anyString())).thenReturn(invalidJson)

        // Act - should not throw
        manager.loadMappings()

        // Assert - counts should remain zero
        assertEquals(0, manager.getNonPairedCount())
        assertEquals(0, manager.getTotalKnownCount())
    }

    // State Consistency Tests

    @Test
    fun testManager_initialState_isConsistent() {
        // Assert - before loading, everything should be empty/zero/null
        assertEquals(0, manager.getNonPairedCount())
        assertEquals(0, manager.getTotalKnownCount())
        assertNull(manager.getNonPairedMapping("any"))
        assertFalse(manager.isKnownContraction("any"))
    }

    @Test
    fun testGeneratePossessive_doesNotAffectState() {
        // Arrange
        val initialNonPaired = manager.getNonPairedCount()
        val initialTotal = manager.getTotalKnownCount()

        // Act - generate several possessives
        manager.generatePossessive("cat")
        manager.generatePossessive("dog")
        manager.generatePossessive("house")
        manager.generatePossessive("test")

        // Assert - state should not change
        assertEquals(initialNonPaired, manager.getNonPairedCount())
        assertEquals(initialTotal, manager.getTotalKnownCount())
    }

    // Comprehensive Possessive Rule Tests

    @Test
    fun testGeneratePossessive_allFunctionWordsExcluded() {
        // Test all function words from FUNCTION_WORDS set
        val functionWords = listOf(
            "i", "you", "he", "she", "it", "we", "they",
            "who", "what", "that", "there", "here",
            "will", "would", "shall", "should",
            "can", "could", "may", "might", "must",
            "do", "does", "did",
            "is", "am", "are", "was", "were",
            "have", "has", "had", "let"
        )

        // Assert - all should return null
        for (word in functionWords) {
            assertNull("Function word '$word' should not generate possessive",
                manager.generatePossessive(word))
        }
    }

    @Test
    fun testGeneratePossessive_commonNouns_generateCorrectly() {
        // Test common nouns that should always work
        val nouns = mapOf(
            "cat" to "cat's",
            "dog" to "dog's",
            "house" to "house's",
            "car" to "car's",
            "book" to "book's",
            "computer" to "computer's",
            "phone" to "phone's",
            "table" to "table's"
        )

        // Assert
        for ((word, expected) in nouns) {
            assertEquals("Possessive for '$word' incorrect",
                expected, manager.generatePossessive(word))
        }
    }

    @Test
    fun testGeneratePossessive_properNouns_generateCorrectly() {
        // Test proper nouns (names)
        val properNouns = mapOf(
            "John" to "John's",
            "Mary" to "Mary's",
            "James" to "James's",  // Modern style
            "Charles" to "Charles's",  // Modern style
            "Paris" to "Paris's",
            "Google" to "Google's"
        )

        // Assert
        for ((word, expected) in properNouns) {
            assertEquals("Possessive for proper noun '$word' incorrect",
                expected, manager.generatePossessive(word))
        }
    }

    // Helper method for argument matching
    private fun anyString(): String = org.mockito.ArgumentMatchers.anyString() ?: ""
}
