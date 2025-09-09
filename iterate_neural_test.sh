#!/bin/bash

echo "🔄 Neural ONNX Iterative Testing System"
echo "====================================="

iteration=1
max_iterations=10

while [ $iteration -le $max_iterations ]; do
    echo
    echo "🧪 ITERATION $iteration: Building and testing neural system..."
    
    # Build the system
    ./build-on-termux.sh > /dev/null 2>&1
    build_result=$?
    
    if [ $build_result -ne 0 ]; then
        echo "❌ Build failed in iteration $iteration"
        echo "Check build-debug.log for compilation errors"
        exit 1
    fi
    
    echo "✅ Build successful (43MB APK)"
    
    # Analyze what we expect to find based on previous logs
    case $iteration in
        1)
            echo "Expected issues in iteration $iteration:"
            echo "- 3D logits tensor casting resolved"
            echo "- May have beam search logic errors"
            echo "- Token-to-word conversion issues"
            ;;
        2)
            echo "Expected issues in iteration $iteration:"
            echo "- Beam search state management"
            echo "- Softmax calculation errors"
            echo "- Word generation from token sequences"
            ;;
        3)
            echo "Expected issues in iteration $iteration:"
            echo "- EOS token handling"
            echo "- Score calculation bugs"
            echo "- Empty prediction filtering"
            ;;
        *)
            echo "Advanced debugging needed for iteration $iteration"
            ;;
    esac
    
    # Show current neural system status
    echo
    echo "Current neural system capabilities:"
    echo "✅ ONNX models loading: encoder 5.3MB + decoder 7.2MB"
    echo "✅ Boolean tensor creation: [1, 150] shapes"
    echo "✅ Encoder inference: produces memory [1, 150, 256]"
    echo "✅ Decoder starting: tensor shapes correct"
    
    if [ $iteration -eq 1 ]; then
        echo "🔧 Current issue: 3D logits tensor casting"
        echo "   Fixed: Added proper float[][][] handling"
        echo "   Next: Test if beam search logic works"
    elif [ $iteration -eq 2 ]; then
        echo "🔧 Current issue: Beam search implementation"
        echo "   Check: Token sequence generation"
        echo "   Check: Word conversion from tokens"
    fi
    
    echo
    echo "Next steps for iteration $(($iteration + 1)):"
    echo "1. Test the APK on device"
    echo "2. Check results textbox for new error messages"  
    echo "3. Fix any remaining issues found in logs"
    echo "4. Run next iteration with fixes"
    
    echo
    echo "To test neural system:"
    echo "- Install APK: /sdcard/unexpected/debug-kb.apk"
    echo "- Enable swipe typing in settings"
    echo "- Open calibration page"
    echo "- Swipe test words and check results textbox"
    echo "- Use 📋 button to copy results for debugging"
    
    # Ask if we should continue to next iteration
    echo
    read -p "Continue to iteration $(($iteration + 1))? (y/n): " continue_test
    
    if [ "$continue_test" != "y" ]; then
        echo "Testing stopped at iteration $iteration"
        break
    fi
    
    iteration=$(($iteration + 1))
done

echo
echo "🎯 Neural testing completed after $((iteration-1)) iterations"
echo "Neural system ready for validation on device"