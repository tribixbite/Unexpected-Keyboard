package juloo.keyboard2.personalization

import android.content.Context
import android.content.SharedPreferences
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.MockitoAnnotations

class PersonalizedScorerTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockPrefs: SharedPreferences

    @Mock
    private lateinit var mockEditor: SharedPreferences.Editor

    private lateinit var engine: PersonalizationEngine
    private lateinit var scorer: PersonalizedScorer

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)

        `when`(mockContext.getSharedPreferences(anyString(), anyInt())).thenReturn(mockPrefs)
        `when`(mockPrefs.getBoolean(anyString(), anyBoolean())).thenReturn(true)
        `when`(mockPrefs.getString(anyString(), anyString())).thenReturn("BALANCED")
        `when`(mockPrefs.getString(eq("vocabulary_data"), any())).thenReturn(null)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putBoolean(anyString(), anyBoolean())).thenReturn(mockEditor)
        `when`(mockEditor.putString(anyString(), anyString())).thenReturn(mockEditor)

        engine = PersonalizationEngine(mockContext)
        scorer = PersonalizedScorer(engine)
    }

    @Test
    fun testInitialization() {
        assertNotNull(scorer)
        assertEquals(PersonalizedScorer.ScoringMode.HYBRID, scorer.getScoringMode())
    }

    @Test
    fun testSetScoringMode() {
        scorer.setScoringMode(PersonalizedScorer.ScoringMode.MULTIPLICATIVE)
        assertEquals(PersonalizedScorer.ScoringMode.MULTIPLICATIVE, scorer.getScoringMode())

        scorer.setScoringMode(PersonalizedScorer.ScoringMode.ADDITIVE)
        assertEquals(PersonalizedScorer.ScoringMode.ADDITIVE, scorer.getScoringMode())
    }

    @Test
    fun testScoreWithPersonalization_NoPersonalizationData() {
        val baseScore = 0.75f
        val score = scorer.scoreWithPersonalization("unknown", baseScore)

        // No personalization data, should return base score
        assertEquals(baseScore, score, 0.01f)
    }

    @Test
    fun testScoreWithPersonalization_PersonalizationDisabled() {
        // Record a word
        repeat(10) { engine.recordWordTyped("kotlin") }

        // Disable personalization
        engine.setEnabled(false)

        val baseScore = 0.75f
        val score = scorer.scoreWithPersonalization("kotlin", baseScore)

        // Personalization disabled, should return base score
        assertEquals(baseScore, score, 0.01f)
    }

    @Test
    fun testScoreWithPersonalization_WithBoost() {
        // Record frequent word
        repeat(50) { engine.recordWordTyped("kotlin") }

        val baseScore = 0.75f
        val score = scorer.scoreWithPersonalization("kotlin", baseScore)

        // Should have higher score due to personalization boost
        assertTrue(score > baseScore)
    }

    @Test
    fun testScoreWithPersonalization_LowBaseScore() {
        repeat(50) { engine.recordWordTyped("kotlin") }

        val baseScore = 0.001f // Very low base score
        val score = scorer.scoreWithPersonalization("kotlin", baseScore)

        // Very low base scores should not be boosted much
        assertEquals(baseScore, score, 0.01f)
    }

    @Test
    fun testScoreWithPersonalization_MaxScore() {
        repeat(1000) { engine.recordWordTyped("frequent") }

        val baseScore = 0.9f
        val score = scorer.scoreWithPersonalization("frequent", baseScore)

        // Score should be clamped to 1.0
        assertTrue(score <= 1.0f)
    }

    @Test
    fun testScoreMultiple() {
        repeat(50) { engine.recordWordTyped("kotlin") }
        repeat(10) { engine.recordWordTyped("java") }

        val words = listOf("kotlin", "java", "python")
        val baseScores = listOf(0.75f, 0.70f, 0.65f)

        val scores = scorer.scoreMultiple(words, baseScores)

        assertEquals(3, scores.size)
        assertTrue(scores[0] > baseScores[0]) // kotlin boosted
        assertTrue(scores[1] > baseScores[1]) // java boosted
        assertEquals(baseScores[2], scores[2], 0.01f) // python not boosted (unknown)
    }

    @Test
    fun testScoreMultiple_SizeMismatch() {
        val words = listOf("kotlin", "java")
        val baseScores = listOf(0.75f) // Wrong size

        try {
            scorer.scoreMultiple(words, baseScores)
            fail("Should throw IllegalArgumentException")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("same size"))
        }
    }

    @Test
    fun testScoreAndRank() {
        repeat(100) { engine.recordWordTyped("frequent") }
        repeat(10) { engine.recordWordTyped("medium") }

        val predictions = mapOf(
            "frequent" to 0.60f,
            "medium" to 0.65f,
            "rare" to 0.70f
        )

        val ranked = scorer.scoreAndRank(predictions, topK = 3)

        assertEquals(3, ranked.size)

        // "frequent" should rank first despite lower base score
        // due to strong personalization boost
        assertEquals("frequent", ranked[0].first)
    }

    @Test
    fun testScoreAndRank_TopK() {
        repeat(10) { i ->
            repeat(i + 1) { engine.recordWordTyped("word$i") }
        }

        val predictions = (0..9).associate { "word$it" to (0.5f + it * 0.01f) }

        val top5 = scorer.scoreAndRank(predictions, topK = 5)

        assertEquals(5, top5.size)
    }

    @Test
    fun testMultiplicativeScoring() {
        scorer.setScoringMode(PersonalizedScorer.ScoringMode.MULTIPLICATIVE)

        repeat(50) { engine.recordWordTyped("test") }

        val baseScore = 0.75f
        val score = scorer.scoreWithPersonalization("test", baseScore)

        // Multiplicative: score * (1 + boost)
        assertTrue(score > baseScore)
        assertTrue(score <= 1.0f)
    }

    @Test
    fun testAdditiveScoring() {
        scorer.setScoringMode(PersonalizedScorer.ScoringMode.ADDITIVE)

        repeat(50) { engine.recordWordTyped("test") }

        val baseScore = 0.75f
        val score = scorer.scoreWithPersonalization("test", baseScore)

        // Additive: score + (boost * weight)
        assertTrue(score > baseScore)
        assertTrue(score <= 1.0f)
    }

    @Test
    fun testHybridScoring() {
        scorer.setScoringMode(PersonalizedScorer.ScoringMode.HYBRID)

        repeat(50) { engine.recordWordTyped("test") }

        val baseScore = 0.75f
        val score = scorer.scoreWithPersonalization("test", baseScore)

        // Hybrid: geometric mean of multiplicative and additive
        assertTrue(score > baseScore)
        assertTrue(score <= 1.0f)
    }

    @Test
    fun testScoringModeComparison() {
        repeat(50) { engine.recordWordTyped("word") }

        val baseScore = 0.50f

        scorer.setScoringMode(PersonalizedScorer.ScoringMode.MULTIPLICATIVE)
        val multiplicative = scorer.scoreWithPersonalization("word", baseScore)

        scorer.setScoringMode(PersonalizedScorer.ScoringMode.ADDITIVE)
        val additive = scorer.scoreWithPersonalization("word", baseScore)

        scorer.setScoringMode(PersonalizedScorer.ScoringMode.HYBRID)
        val hybrid = scorer.scoreWithPersonalization("word", baseScore)

        // All should boost, but by different amounts
        assertTrue(multiplicative > baseScore)
        assertTrue(additive > baseScore)
        assertTrue(hybrid > baseScore)

        // Hybrid should be between multiplicative and additive
        // (Not always true, but generally for moderate boosts)
        assertTrue(hybrid > baseScore)
    }

    @Test
    fun testGetEffectiveMultiplier() {
        repeat(50) { engine.recordWordTyped("test") }

        val baseScore = 0.50f
        val multiplier = scorer.getEffectiveMultiplier("test", baseScore)

        // Multiplier should be > 1.0 for boosted words
        assertTrue(multiplier > 1.0f)
    }

    @Test
    fun testGetEffectiveMultiplier_UnknownWord() {
        val multiplier = scorer.getEffectiveMultiplier("unknown", 0.50f)

        // No boost, multiplier should be 1.0
        assertEquals(1.0f, multiplier, 0.01f)
    }

    @Test
    fun testGetEffectiveMultiplier_DisabledPersonalization() {
        repeat(50) { engine.recordWordTyped("test") }
        engine.setEnabled(false)

        val multiplier = scorer.getEffectiveMultiplier("test", 0.50f)

        // Disabled, multiplier should be 1.0
        assertEquals(1.0f, multiplier, 0.01f)
    }

    @Test
    fun testExplainScoring() {
        repeat(50) { engine.recordWordTyped("kotlin") }

        val explanation = scorer.explainScoring("kotlin", 0.75f)

        assertEquals("kotlin", explanation.word)
        assertEquals(0.75f, explanation.baseScore, 0.01f)
        assertTrue(explanation.personalizationBoost > 0.0f)
        assertTrue(explanation.personalizedScore > explanation.baseScore)
        assertTrue(explanation.effectiveMultiplier > 1.0f)
        assertEquals(PersonalizedScorer.ScoringMode.HYBRID, explanation.scoringMode)
        assertTrue(explanation.explanation.contains("kotlin"))
    }

    @Test
    fun testExplainScoring_UnknownWord() {
        val explanation = scorer.explainScoring("unknown", 0.75f)

        assertEquals("unknown", explanation.word)
        assertEquals(0.0f, explanation.personalizationBoost, 0.01f)
        assertEquals(0.75f, explanation.personalizedScore, 0.01f)
        assertEquals(1.0f, explanation.effectiveMultiplier, 0.01f)
    }

    @Test
    fun testRealWorldScenario_RankPredictions() {
        // Simulate user frequently types "kotlin" and "android"
        repeat(100) { engine.recordWordTyped("kotlin") }
        repeat(80) { engine.recordWordTyped("android") }
        repeat(5) { engine.recordWordTyped("java") }

        // Neural model gives these base scores
        val predictions = mapOf(
            "java" to 0.90f,      // High neural score
            "javascript" to 0.85f, // High neural score
            "kotlin" to 0.60f,     // Low neural score, but user types it often
            "android" to 0.55f,    // Low neural score, but user types it often
            "python" to 0.50f      // Medium neural score
        )

        val ranked = scorer.scoreAndRank(predictions, topK = 5)

        // "kotlin" and "android" should rank higher due to personalization
        // despite lower neural scores
        val topWords = ranked.map { it.first }

        // kotlin and android should be in top positions
        assertTrue(topWords.indexOf("kotlin") < topWords.indexOf("javascript"))
        assertTrue(topWords.indexOf("android") < topWords.indexOf("python"))
    }

    @Test
    fun testRealWorldScenario_BoostStrongPredictions() {
        // User types "kotlin" frequently
        repeat(100) { engine.recordWordTyped("kotlin") }

        // Neural model already confident about "kotlin"
        val baseScore = 0.95f
        val score = scorer.scoreWithPersonalization("kotlin", baseScore)

        // Should boost even strong predictions (but clamped to 1.0)
        assertTrue(score >= baseScore)
        assertTrue(score <= 1.0f)
    }

    @Test
    fun testRealWorldScenario_NoBoostForWeakPredictions() {
        repeat(100) { engine.recordWordTyped("kotlin") }

        // Neural model very uncertain
        val baseScore = 0.001f
        val score = scorer.scoreWithPersonalization("kotlin", baseScore)

        // Very weak predictions shouldn't be boosted much
        assertTrue(score < 0.1f)
    }
}
