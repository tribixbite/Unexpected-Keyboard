// Minimal test to validate 3D tensor handling
// javac -cp ~/.gradle/caches/transforms-4/485e19b84cfae8d5bdd16762731ed50b/transformed/onnxruntime-android-1.20.0-api.jar minimal_test.java

import ai.onnxruntime.*;
import java.io.*;
import java.nio.*;

public class minimal_test {
    public static void main(String[] args) {
        System.out.println("Testing 3D tensor operations...");
        
        try {
            // Create a fake 3D tensor to test casting
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            
            // Create 3D array [1, 3, 5] 
            float[][][] test3D = new float[1][3][5];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 5; j++) {
                    test3D[0][i][j] = i * 5 + j; // Fill with test data
                }
            }
            
            OnnxTensor tensor = OnnxTensor.createTensor(env, test3D);
            System.out.println("Created tensor shape: " + java.util.Arrays.toString(tensor.getInfo().getShape()));
            
            // Now test getValue() and casting
            Object value = tensor.getValue();
            System.out.println("Tensor value type: " + value.getClass().getName());
            
            // Test the exact casting that's failing
            try {
                float[][][] result3D = (float[][][]) value;
                System.out.println("✅ 3D cast successful");
                System.out.println("Dimensions: [" + result3D.length + "][" + result3D[0].length + "][" + result3D[0][0].length + "]");
                
                // Test accessing position like beam search
                float[] position0 = result3D[0][0];
                float[] position1 = result3D[0][1];
                System.out.println("Position 0 logits: " + java.util.Arrays.toString(position0));
                System.out.println("Position 1 logits: " + java.util.Arrays.toString(position1));
                
            } catch (ClassCastException e) {
                System.err.println("❌ 3D cast failed: " + e.getMessage());
                
                // Try flat cast
                try {
                    float[] resultFlat = (float[]) value;
                    System.out.println("Flat cast successful, length: " + resultFlat.length);
                } catch (ClassCastException e2) {
                    System.err.println("❌ Flat cast also failed: " + e2.getMessage());
                }
            }
            
            tensor.close();
            
        } catch (Exception e) {
            System.err.println("Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}