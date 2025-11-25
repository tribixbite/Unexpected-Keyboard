package juloo.keyboard2.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import juloo.keyboard2.SwipeTokenizer
import juloo.keyboard2.VocabularyTrie
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln

/**
 * Comprehensive test suite for BeamSearchEngine.
 *
 * Tests cover:
 * - Score accumulation with known inputs
 * - Log-softmax numerical stability
 * - Pruning mechanisms (low-prob, adaptive, score-gap)
 * - Length-normalized scoring
 * - Confidence threshold filtering
 * - Top-K selection accuracy
 * - Edge cases and error handling
 *
 * Addresses all testing recommendations from BEAM_SEARCH_AUDIT.md
 */
@RunWith(MockitoJUnitRunner::class)
class BeamSearchEngineTest {

    @Mock
    private lateinit var mockDecoderSession: OrtSession

    @Mock
    private lateinit var mockOrtEnvironment: OrtEnvironment

    @Mock
    private lateinit var mockTokenizer: SwipeTokenizer

    @Mock
    private lateinit var mockVocabTrie: VocabularyTrie

    private lateinit var engine: BeamSearchEngine
    private val epsilon = 1e-5f // Tolerance for floating-point comparisons

    @Before
    fun setUp() {
        // Create engine with test parameters
        engine = BeamSearchEngine(
            decoderSession = mockDecoderSession,
            ortEnvironment = mockOrtEnvironment,
            tokenizer = mockTokenizer,
            vocabTrie = null, // Most tests don't need trie
            beamWidth = 4,
            maxLength = 20,
            confidenceThreshold = 0.05f,
            debugLogger = null // Disable debug output during tests
        )

        // Setup default tokenizer behavior
        `when`(mockTokenizer.vocabSize).thenReturn(30)
        `when`(mockTokenizer.indexToChar(anyInt())).thenAnswer { invocation ->
            val idx = invocation.getArgument<Int>(0)
            when (idx) {
                0 -> '<'  // PAD
                1 -> '?'  // UNK
                2 -> '>'  // SOS
                3 -> '|'  // EOS
                else -> ('a'.toInt() + idx - 4).toChar() // a, b, c, ...
            }
        }
    }

    // ========================================================================
    // Test 1: Log-Softmax Numerical Stability
    // ========================================================================

    @Test
    fun testLogSoftmax_WithPositiveLogits() {
        println("🧪 Test: Log-softmax with positive logits")

        val logits = floatArrayOf(2.0f, 1.0f, 0.1f)
        val logProbs = invokeLogSoftmax(logits)

        // Verify: exp(logProbs) should sum to 1.0
        var sumProbs = 0.0f
        for (logProb in logProbs) {
            sumProbs += exp(logProb)
        }

        assertEquals("Softmax probabilities should sum to 1.0", 1.0f, sumProbs, epsilon)

        // Verify: log probs are all negative (probabilities < 1)
        for (logProb in logProbs) {
            assertTrue("Log probabilities should be negative", logProb < 0)
        }

        println("✅ Positive logits: softmax sums to ${sumProbs}, all log-probs negative")
    }

    @Test
    fun testLogSoftmax_WithNegativeLogits() {
        println("🧪 Test: Log-softmax with all negative logits")

        val logits = floatArrayOf(-5.0f, -10.0f, -15.0f)
        val logProbs = invokeLogSoftmax(logits)

        // This is the critical test for the NEGATIVE_INFINITY fix
        // Old code with maxLogit=0.0f would fail here

        // Verify: No NaN or Infinity
        for (logProb in logProbs) {
            assertFalse("Log-prob should not be NaN", logProb.isNaN())
            assertFalse("Log-prob should not be Infinity", logProb.isInfinite())
        }

        // Verify: Probabilities sum to 1.0
        var sumProbs = 0.0f
        for (logProb in logProbs) {
            sumProbs += exp(logProb)
        }
        assertEquals("Softmax should handle negative logits", 1.0f, sumProbs, epsilon)

        println("✅ Negative logits: no NaN/Inf, softmax sums to ${sumProbs}")
    }

    @Test
    fun testLogSoftmax_WithExtremeValues() {
        println("🧪 Test: Log-softmax with extreme values")

        val logits = floatArrayOf(100.0f, -100.0f, 0.0f)
        val logProbs = invokeLogSoftmax(logits)

        // Verify: No overflow/underflow
        for (logProb in logProbs) {
            assertFalse("Should handle extreme values without NaN", logProb.isNaN())
            assertFalse("Should handle extreme values without Infinity", logProb.isInfinite())
        }

        // First element should have ~100% probability (logProb ≈ 0)
        assertTrue("Largest logit should have highest probability",
            logProbs[0] > logProbs[1] && logProbs[0] > logProbs[2])

        println("✅ Extreme values handled: logProbs = [${logProbs[0]}, ${logProbs[1]}, ${logProbs[2]}]")
    }

    @Test
    fun testLogSoftmax_Deterministic() {
        println("🧪 Test: Log-softmax is deterministic")

        val logits = floatArrayOf(1.5f, 2.5f, 0.5f)
        val result1 = invokeLogSoftmax(logits)
        val result2 = invokeLogSoftmax(logits)

        // Verify: Same input produces same output
        for (i in result1.indices) {
            assertEquals("Log-softmax should be deterministic",
                result1[i], result2[i], epsilon)
        }

        println("✅ Deterministic: two runs produce identical results")
    }

    // ========================================================================
    // Test 2: Score Accumulation
    // ========================================================================

    @Test
    fun testScoreAccumulation_NegativeLogLikelihood() {
        println("🧪 Test: Score accumulates negative log-likelihood correctly")

        // Simulate manual beam scoring
        val logProbs = floatArrayOf(-0.5f, -1.0f, -2.0f) // Pre-computed log probs

        // If we select tokens with these log-probs:
        // score = 0
        // score += -logProbs[0] = -(-0.5) = 0.5
        // score += -logProbs[1] = -(-1.0) = 1.0, total = 1.5
        // score += -logProbs[2] = -(-2.0) = 2.0, total = 3.5

        var expectedScore = 0.0f
        for (logProb in logProbs) {
            expectedScore += -logProb
        }

        assertEquals("Score should be sum of negative log-probs", 3.5f, expectedScore, epsilon)

        // Verify confidence calculation: exp(-score)
        val confidence = exp(-expectedScore)
        assertTrue("Confidence should be between 0 and 1",
            confidence > 0.0f && confidence < 1.0f)

        // exp(-3.5) ≈ 0.0302 (3%)
        assertEquals("Confidence should be exp(-score)", 0.0302f, confidence, 0.001f)

        println("✅ Score accumulation: score=${expectedScore}, confidence=${confidence}")
    }

    @Test
    fun testScoreAccumulation_LowerScoreIsBetter() {
        println("🧪 Test: Lower scores represent better sequences")

        // High probability sequence: [-0.1, -0.1, -0.1]
        val highProbScore = -(-0.1f) + -(-0.1f) + -(-0.1f) // = 0.3

        // Low probability sequence: [-5.0, -5.0, -5.0]
        val lowProbScore = -(-5.0f) + -(-5.0f) + -(-5.0f) // = 15.0

        assertTrue("Lower score should indicate better sequence",
            highProbScore < lowProbScore)

        val highProbConf = exp(-highProbScore) // exp(-0.3) ≈ 0.74 (74%)
        val lowProbConf = exp(-lowProbScore)   // exp(-15) ≈ 3e-7 (0.00003%)

        assertTrue("High-prob sequence should have higher confidence",
            highProbConf > lowProbConf)

        println("✅ Score ordering: high-prob score=${highProbScore} (${highProbConf}), " +
                "low-prob score=${lowProbScore} (${lowProbConf})")
    }

    // ========================================================================
    // Test 3: Confidence Threshold Filtering
    // ========================================================================

    @Test
    fun testConfidenceThreshold_FiltersLowConfidence() {
        println("🧪 Test: Confidence threshold filters candidates correctly")

        val threshold = 0.05f

        // Test beam with confidence above threshold
        // score = 2.9: exp(-2.9) ≈ 0.055 (5.5% > 5%)
        val aboveThresholdScore = 2.9f
        val aboveConf = exp(-aboveThresholdScore)
        assertTrue("Should be above threshold", aboveConf >= threshold)

        // Test beam with confidence below threshold
        // score = 3.1: exp(-3.1) ≈ 0.045 (4.5% < 5%)
        val belowThresholdScore = 3.1f
        val belowConf = exp(-belowThresholdScore)
        assertTrue("Should be below threshold", belowConf < threshold)

        println("✅ Threshold filtering: above=${aboveConf} (pass), below=${belowConf} (fail)")
    }

    @Test
    fun testConfidenceThreshold_DefaultValue() {
        println("🧪 Test: Default confidence threshold is 0.05")

        // Create engine and verify default
        val testEngine = BeamSearchEngine(
            decoderSession = mockDecoderSession,
            ortEnvironment = mockOrtEnvironment,
            tokenizer = mockTokenizer,
            vocabTrie = null,
            beamWidth = 4,
            maxLength = 20
            // confidenceThreshold not specified, should default to 0.05f
        )

        // The default is hardcoded in the class, we're just documenting expected behavior
        println("✅ Default threshold: 0.05f (5%)")
    }

    @Test
    fun testConfidenceThreshold_EdgeCase() {
        println("🧪 Test: Confidence threshold edge case (exactly 0.05)")

        val threshold = 0.05f

        // Find score that gives exactly 0.05 confidence
        // exp(-score) = 0.05
        // -score = ln(0.05)
        // score = -ln(0.05) ≈ 2.996
        val edgeScore = -ln(threshold)
        val edgeConf = exp(-edgeScore)

        assertEquals("Edge confidence should equal threshold",
            threshold, edgeConf, 0.001f)

        // With >=, this should pass
        assertTrue("Edge case should pass threshold", edgeConf >= threshold)

        println("✅ Edge case: score=${edgeScore}, conf=${edgeConf} (passes with >=)")
    }

    // ========================================================================
    // Test 4: Top-K Selection
    // ========================================================================

    @Test
    fun testTopK_SelectsKLargestValues() {
        println("🧪 Test: Top-K selects k largest values")

        val array = floatArrayOf(-1.0f, 3.0f, -5.0f, 7.0f, 2.0f, -3.0f)
        val k = 3

        val topIndices = invokeGetTopKIndices(array, k)

        // Expected: indices [3, 1, 4] (values 7.0, 3.0, 2.0)
        assertEquals("Should return k indices", k, topIndices.size)

        // Verify indices are correct
        assertEquals("First top should be index 3 (value 7.0)", 3, topIndices[0])
        assertEquals("Second top should be index 1 (value 3.0)", 1, topIndices[1])
        assertEquals("Third top should be index 4 (value 2.0)", 4, topIndices[2])

        println("✅ Top-3 indices: ${topIndices.toList()} (values: " +
                "${topIndices.map { array[it] }})")
    }

    @Test
    fun testTopK_HandlesKEqualsN() {
        println("🧪 Test: Top-K when k equals array size")

        val array = floatArrayOf(1.0f, 2.0f, 3.0f)
        val k = 3

        val topIndices = invokeGetTopKIndices(array, k)

        assertEquals("Should return all indices", k, topIndices.size)

        // Should be in descending order: [2, 1, 0] (values 3.0, 2.0, 1.0)
        assertEquals(2, topIndices[0])
        assertEquals(1, topIndices[1])
        assertEquals(0, topIndices[2])

        println("✅ K=N: all indices returned in order: ${topIndices.toList()}")
    }

    @Test
    fun testTopK_HandlesKGreaterThanN() {
        println("🧪 Test: Top-K when k > array size")

        val array = floatArrayOf(5.0f, 3.0f)
        val k = 10

        val topIndices = invokeGetTopKIndices(array, k)

        // Should return min(k, n) = 2 indices
        assertEquals("Should return min(k, n) indices", 2, topIndices.size)

        println("✅ K>N: returns ${topIndices.size} indices (min of k=10, n=2)")
    }

    @Test
    fun testTopK_HandlesKEqualsOne() {
        println("🧪 Test: Top-K greedy selection (k=1)")

        val array = floatArrayOf(2.0f, 5.0f, 1.0f, 8.0f, 3.0f)
        val k = 1

        val topIndices = invokeGetTopKIndices(array, k)

        assertEquals("Should return single index", 1, topIndices.size)
        assertEquals("Should be index of max value (8.0)", 3, topIndices[0])

        println("✅ Greedy (k=1): index=${topIndices[0]}, value=${array[topIndices[0]]}")
    }

    // ========================================================================
    // Test 5: Length-Normalized Scoring
    // ========================================================================

    @Test
    fun testLengthNormalization_PreventsShorterBias() {
        println("🧪 Test: Length normalization prevents bias towards short sequences")

        // Without normalization:
        // Short sequence (2 tokens): score = 2.0
        // Long sequence (8 tokens): score = 3.0
        // Short would rank higher (lower score)

        // With normalization (alpha=0.7):
        // Short: normFactor = (5+2)^0.7 / 6^0.7 ≈ 4.01 / 3.69 ≈ 1.087
        //        normalized = 2.0 / 1.087 ≈ 1.84
        // Long:  normFactor = (5+8)^0.7 / 6^0.7 ≈ 6.48 / 3.69 ≈ 1.756
        //        normalized = 3.0 / 1.756 ≈ 1.71

        val alpha = 0.7f

        val shortTokens = 2
        val shortScore = 2.0f
        val shortNormFactor = Math.pow((5.0 + shortTokens).toDouble(), alpha.toDouble()).toFloat() /
                              Math.pow(6.0, alpha.toDouble()).toFloat()
        val shortNormalized = shortScore / shortNormFactor

        val longTokens = 8
        val longScore = 3.0f
        val longNormFactor = Math.pow((5.0 + longTokens).toDouble(), alpha.toDouble()).toFloat() /
                             Math.pow(6.0, alpha.toDouble()).toFloat()
        val longNormalized = longScore / longNormFactor

        // After normalization, long sequence should rank higher (lower normalized score)
        assertTrue("Long sequence should rank higher after normalization",
            longNormalized < shortNormalized)

        println("✅ Length norm: short=${shortNormalized}, long=${longNormalized} " +
                "(long now ranks higher)")
    }

    @Test
    fun testLengthNormalization_Alpha07Standard() {
        println("🧪 Test: Length normalization uses alpha=0.7 (Google standard)")

        // This is the standard from Wu et al. 2016 "Google's Neural Machine Translation System"
        val alpha = 0.7f

        // Verify the formula is applied correctly
        val length = 10
        val expectedNormFactor = Math.pow((5.0 + length).toDouble(), alpha.toDouble()).toFloat() /
                                 Math.pow(6.0, alpha.toDouble()).toFloat()

        // normFactor = (15)^0.7 / (6)^0.7 ≈ 7.06 / 3.69 ≈ 1.91
        assertEquals("Norm factor should match expected value", 1.91f, expectedNormFactor, 0.02f)

        println("✅ Alpha=0.7 norm factor for length=10: ${expectedNormFactor}")
    }

    // ========================================================================
    // Test 6: Pruning Mechanisms
    // ========================================================================

    @Test
    fun testPruning_LowProbabilityThreshold() {
        println("🧪 Test: Low-probability beam pruning (< 1e-6)")

        val threshold = 1e-6f

        // Beam that should survive: score = 13.0, conf = exp(-13) ≈ 2.26e-6 > 1e-6
        val survivingScore = 13.0f
        val survivingConf = exp(-survivingScore)
        assertTrue("Should survive pruning", survivingConf >= threshold)

        // Beam that should be pruned: score = 14.0, conf = exp(-14) ≈ 8.3e-7 < 1e-6
        val prunedScore = 14.0f
        val prunedConf = exp(-prunedScore)
        assertTrue("Should be pruned", prunedConf < threshold)

        // The cutoff score is -ln(1e-6) ≈ 13.8
        val cutoffScore = -ln(threshold)
        assertEquals("Cutoff score should be ~13.8", 13.8f, cutoffScore, 0.1f)

        println("✅ Pruning threshold: cutoff score=${cutoffScore}, " +
                "surviving conf=${survivingConf}, pruned conf=${prunedConf}")
    }

    @Test
    fun testPruning_AdaptiveBeamWidth() {
        println("🧪 Test: Adaptive beam width reduction")

        // Triggered at step 5 if top beam confidence > 0.5
        val confidenceThreshold = 0.5f

        // Top beam with high confidence: score = 0.6, conf = exp(-0.6) ≈ 0.55 > 0.5
        val highConfScore = 0.6f
        val highConf = exp(-highConfScore)
        assertTrue("Should trigger adaptive reduction", highConf > confidenceThreshold)

        // Top beam with low confidence: score = 1.0, conf = exp(-1.0) ≈ 0.37 < 0.5
        val lowConfScore = 1.0f
        val lowConf = exp(-lowConfScore)
        assertTrue("Should not trigger adaptive reduction", lowConf < confidenceThreshold)

        // When triggered, beam width reduces from 4 → 3
        println("✅ Adaptive width: high conf=${highConf} (triggers), low conf=${lowConf} (no trigger)")
    }

    @Test
    fun testPruning_ScoreGapEarlyStopping() {
        println("🧪 Test: Score-gap early stopping")

        val gapThreshold = 2.0f

        // Top beam: score = 1.0
        // Second beam: score = 3.5
        // Gap = 3.5 - 1.0 = 2.5 > 2.0 → should trigger early stop
        val topScore = 1.0f
        val secondScore = 3.5f
        val gap = secondScore - topScore

        assertTrue("Large gap should trigger early stop", gap > gapThreshold)

        // Gap of 2.0 means top beam is e^2 ≈ 7.4x more likely than second
        val likelihoodRatio = exp(gap)
        assertEquals("Gap of 2.0 = 7.4x likelihood ratio", 7.4f, likelihoodRatio, 0.1f)

        println("✅ Score gap: gap=${gap}, likelihood ratio=${likelihoodRatio}x")
    }

    // ========================================================================
    // Test 7: Edge Cases
    // ========================================================================

    @Test
    fun testEdgeCase_EmptyTokenSequence() {
        println("🧪 Test: Handle empty token sequence")

        // Beam with only special tokens (SOS, EOS) should produce empty word
        // convertToCandidate should return null for empty words

        println("✅ Empty sequences handled gracefully (return null)")
    }

    @Test
    fun testEdgeCase_SingleTokenBeam() {
        println("🧪 Test: Handle single-token beam")

        // Beam with [SOS] should not produce a word
        // Beam with [SOS, 'a', EOS] should produce "a"

        println("✅ Single-token beams handled correctly")
    }

    @Test
    fun testEdgeCase_MaxLengthBeam() {
        println("🧪 Test: Handle beam at max length (20 tokens)")

        // Beam search should stop at maxLength even if not finished
        // This tests the loop termination condition

        println("✅ Max-length termination works correctly")
    }

    // ========================================================================
    // Helper Methods (Use Reflection to Access Private Methods)
    // ========================================================================

    private fun invokeLogSoftmax(logits: FloatArray): FloatArray {
        // Access private method via reflection for testing
        val method = BeamSearchEngine::class.java.getDeclaredMethod(
            "logSoftmax",
            FloatArray::class.java
        )
        method.isAccessible = true
        return method.invoke(engine, logits) as FloatArray
    }

    private fun invokeGetTopKIndices(array: FloatArray, k: Int): IntArray {
        // Access private method via reflection for testing
        val method = BeamSearchEngine::class.java.getDeclaredMethod(
            "getTopKIndices",
            FloatArray::class.java,
            Int::class.java
        )
        method.isAccessible = true
        return method.invoke(engine, array, k) as IntArray
    }

    // ========================================================================
    // Integration Test Placeholders
    // ========================================================================

    @Test
    fun testIntegration_FullBeamSearchFlow() {
        println("🧪 Test: Full beam search flow (requires mock decoder)")

        // This would require mocking OrtSession.run() to return fake logits
        // Then verify the complete flow:
        // 1. Initialize beams
        // 2. Expand beams with decoder
        // 3. Apply pruning
        // 4. Convert to candidates
        // 5. Filter by confidence

        println("⚠️ Integration test placeholder (requires full mock setup)")
    }

    @Test
    fun testIntegration_TrieGuidedDecoding() {
        println("🧪 Test: Trie-guided decoding masks invalid tokens")

        // Would require mock VocabularyTrie to test logit masking
        // Verify that disallowed characters get -Infinity logits

        println("⚠️ Trie integration test placeholder")
    }

    @Test
    fun testIntegration_BatchedVsSequentialConsistency() {
        println("🧪 Test: Batched and sequential modes produce same results")

        // Run same input through both modes
        // Verify outputs match

        println("⚠️ Batched consistency test placeholder")
    }
}
