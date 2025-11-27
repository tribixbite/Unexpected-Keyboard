// Test the actual logic that's failing
public class test_logic {
    public static void main(String[] args) {
        System.out.println("Testing tensor casting logic...");
        
        // Simulate what ONNX returns
        float[][][] logits3D = new float[1][20][30];
        
        // Fill with test data
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 30; j++) {
                logits3D[0][i][j] = (float)(Math.random());
            }
        }
        
        // Test the exact operations from beam search
        Object logitsValue = logits3D; // This is what tensor.getValue() returns
        int tokenPosition = 0;
        int vocabSize = 30;
        
        System.out.println("Logits object type: " + logitsValue.getClass().getName());
        
        try {
            // This is my current fix - direct 3D cast
            float[][][] logits3DCast = (float[][][]) logitsValue;
            System.out.println("✅ 3D cast successful");
            
            // Extract position logits
            if (tokenPosition < logits3DCast[0].length) {
                float[] positionLogits = logits3DCast[0][tokenPosition];
                System.out.println("✅ Position extraction successful, length: " + positionLogits.length);
                
                // Copy to result array
                float[] relevantLogits = new float[vocabSize];
                System.arraycopy(positionLogits, 0, relevantLogits, 0, vocabSize);
                System.out.println("✅ Array copy successful");
                
                // Show first few values
                System.out.print("First 5 logits: ");
                for (int i = 0; i < 5; i++) {
                    System.out.printf("%.3f ", relevantLogits[i]);
                }
                System.out.println();
                
            } else {
                System.err.println("❌ Token position out of bounds");
            }
            
        } catch (ClassCastException e) {
            System.err.println("❌ 3D cast failed: " + e.getMessage());
            
            // Check what's actually happening
            if (logitsValue instanceof float[]) {
                System.out.println("Object is actually float[] (flat)");
            } else if (logitsValue instanceof float[][][]) {
                System.out.println("Object is float[][][] but cast failed??");
            } else {
                System.out.println("Object is: " + logitsValue.getClass().getName());
            }
        }
        
        System.out.println("\nThis logic should work in the neural system.");
        System.out.println("If ClassCastException persists, there's old code still doing wrong cast somewhere.");
    }
}