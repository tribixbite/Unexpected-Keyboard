package juloo.keyboard2

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment

/**
 * Builder for ONNX decoder input parameters
 * Handles both builtin v2 models (single target_mask) and custom models (separate padding/causal masks)
 *
 * @since v1.32.429
 */
class DecoderInputBuilder(
    private val ortEnvironment: OrtEnvironment,
    private val usesSeparateMasks: Boolean
) {
    companion object {
        private const val PAD_IDX = 0
        private const val DECODER_SEQ_LENGTH = 35
    }

    /**
     * Build decoder input map with appropriate mask configuration
     *
     * @param batchedMemory Encoder memory output [num_beams, seq_len, hidden_dim]
     * @param targetTokens Target token IDs [num_beams, seq_length]
     * @param batchedTokens Raw token array for padding mask detection
     * @param batchedTargetMask Combined target mask for v2 builtin (only used if !usesSeparateMasks)
     * @param srcMask Source sequence mask [num_beams, src_seq_len]
     * @return Map of input name to tensor for ONNX inference
     */
    fun buildInputs(
        batchedMemory: OnnxTensor,
        targetTokens: OnnxTensor,
        batchedTokens: Array<LongArray>,
        batchedTargetMask: Array<BooleanArray>?,
        srcMask: OnnxTensor
    ): Map<String, OnnxTensor> {
        val decoderInputs = HashMap<String, OnnxTensor>()

        // Common inputs for all model types
        decoderInputs["memory"] = batchedMemory
        decoderInputs["target_tokens"] = targetTokens
        decoderInputs["src_mask"] = srcMask

        // Model-specific mask inputs
        if (usesSeparateMasks) {
            // Custom models: Use separate padding and causal masks
            createSeparateMasks(batchedTokens, decoderInputs)
        } else {
            // Builtin v2: Use combined target_mask
            if (batchedTargetMask == null) {
                throw IllegalArgumentException("batchedTargetMask required for builtin v2 model")
            }
            decoderInputs["target_mask"] = OnnxTensor.createTensor(ortEnvironment, batchedTargetMask)
        }

        return decoderInputs
    }

    /**
     * Create separate padding and causal masks for custom models
     */
    private fun createSeparateMasks(
        batchedTokens: Array<LongArray>,
        decoderInputs: HashMap<String, OnnxTensor>
    ) {
        val numActiveBeams = batchedTokens.size

        // Padding mask: True where tokens are PAD (0), False elsewhere
        val paddingMask = Array(numActiveBeams) { b ->
            BooleanArray(DECODER_SEQ_LENGTH) { i ->
                batchedTokens[b][i] == PAD_IDX.toLong()
            }
        }

        // Causal mask: True in upper triangle to prevent attending to future positions
        // For custom models, we use all-false since they may handle causality internally
        val causalMask = Array(numActiveBeams) {
            BooleanArray(DECODER_SEQ_LENGTH) { false }
        }

        decoderInputs["target_padding_mask"] = OnnxTensor.createTensor(ortEnvironment, paddingMask)
        decoderInputs["target_causal_mask"] = OnnxTensor.createTensor(ortEnvironment, causalMask)
    }

    /**
     * Create proper causal mask (upper triangular matrix)
     * Currently unused but available for future use if custom models require it
     */
    private fun createCausalMask(seqLength: Int): Array<BooleanArray> {
        return Array(seqLength) { i ->
            BooleanArray(seqLength) { j ->
                j > i // True in upper triangle (mask future positions)
            }
        }
    }
}
