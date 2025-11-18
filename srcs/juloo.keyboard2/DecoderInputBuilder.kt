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
     * CRITICAL: Custom models expect FLOAT tensors, not boolean
     * Uses actual token array length instead of hardcoded DECODER_SEQ_LENGTH
     */
    private fun createSeparateMasks(
        batchedTokens: Array<LongArray>,
        decoderInputs: HashMap<String, OnnxTensor>
    ) {
        val numActiveBeams = batchedTokens.size
        val actualSeqLength = batchedTokens[0].size  // Use actual sequence length from tokens

        // Padding mask: 1.0 where tokens are PAD (0), 0.0 elsewhere (FLOAT tensor)
        val paddingMask = Array(numActiveBeams) { b ->
            FloatArray(actualSeqLength) { i ->
                if (batchedTokens[b][i] == PAD_IDX.toLong()) 1.0f else 0.0f
            }
        }

        // Causal mask: -inf in upper triangle to prevent attending to future positions (FLOAT tensor)
        // Use 0.0 on diagonal and lower triangle, negative infinity on upper triangle
        val causalMask = Array(numActiveBeams) { _ ->
            FloatArray(actualSeqLength * actualSeqLength) { idx ->
                val i = idx / actualSeqLength
                val j = idx % actualSeqLength
                if (j > i) Float.NEGATIVE_INFINITY else 0.0f
            }
        }

        // Reshape causal mask to [num_beams, seq_len, seq_len]
        val causalMask3D = Array(numActiveBeams) { b ->
            Array(actualSeqLength) { i ->
                FloatArray(actualSeqLength) { j ->
                    causalMask[b][i * actualSeqLength + j]
                }
            }
        }

        decoderInputs["target_padding_mask"] = OnnxTensor.createTensor(ortEnvironment, paddingMask)
        decoderInputs["target_causal_mask"] = OnnxTensor.createTensor(ortEnvironment, causalMask3D)
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
