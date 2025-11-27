# Neural Swipe Typing - User Guide

## Overview

Unexpected Keyboard now includes advanced neural network-powered swipe typing with privacy-first design, performance monitoring, and smart model management.

**Version**: v1.32.904+
**Status**: Production Ready
**Privacy**: Local-only by default, full user control

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Privacy & Data Controls](#privacy--data-controls)
3. [Performance Monitoring](#performance-monitoring)
4. [Model Management](#model-management)
5. [A/B Testing](#ab-testing)
6. [Custom Models](#custom-models)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Enabling Neural Swipe Typing

1. Open **Settings** → **Neural ML & Swipe Typing**
2. Enable **"Neural Swipe Prediction (ONNX)"**
3. The keyboard will load the built-in model automatically

**First-time setup takes ~500ms to load models**
**Subsequent predictions are instant (<50ms)**

### Basic Usage

1. **Swipe** your finger across the keyboard letters to spell a word
2. Watch predictions appear in the suggestion bar
3. Tap a prediction to insert it
4. The system learns from your selections

**Tips**:
- Swipe through the approximate letter positions
- No need to be precise - the neural network handles ambiguity
- Lift your finger to complete the gesture
- Predictions show confidence scores when enabled

---

## Privacy & Data Controls

### Overview

All data collection is **opt-in** and **local-only by default**. You have complete control over what data is collected and how it's used.

### Accessing Privacy Settings

**Settings** → **Neural ML & Swipe Typing** → **🔒 Privacy & Data**

### Consent Management

#### Granting Consent

1. Tap **"Data Collection Consent"**
2. Read the explanation of what data will be collected
3. Tap **"Grant Consent"** to enable data collection
4. You can revoke consent at any time

#### Revoking Consent

1. Tap **"Data Collection Consent"**
2. Choose:
   - **"Revoke & Delete"**: Removes consent AND deletes all collected data
   - **"Revoke Only"**: Stops collection but preserves existing data

### Data Collection Controls

#### What Data Can Be Collected

1. **Swipe Data** (gesture paths):
   - Touch coordinates during swipes
   - Key sequences detected
   - Used for improving swipe recognition

2. **Performance Data** (statistics):
   - Prediction accuracy (top-1, top-3)
   - Inference latency
   - Model load time

3. **Error Logs** (debugging):
   - Model loading errors
   - Prediction failures
   - Crash reports

**Each type can be enabled/disabled independently**

#### Enabling/Disabling Collection

Navigate to **Privacy & Data** → **Data Collection Controls**:

- ✅ **Collect Swipe Data** (default: ON if consent granted)
- ✅ **Collect Performance Data** (default: ON if consent granted)
- ❌ **Collect Error Logs** (default: OFF)

### Privacy Settings

Navigate to **Privacy & Data** → **Privacy Settings**:

1. **Anonymize Data** (default: ON)
   - Removes device identifiers
   - Strips timestamps to hour granularity
   - Normalizes screen dimensions

2. **Local-Only Training** (default: ON)
   - All data stays on your device
   - No cloud synchronization
   - Complete control

3. **Allow Data Export** (default: OFF)
   - Enable to export data for external analysis
   - Data exported as JSON files
   - Useful for training custom models

4. **Allow Model Sharing** (default: OFF)
   - Enable to share trained models with other devices
   - Models remain local unless explicitly exported

### Data Retention

Navigate to **Privacy & Data** → **Data Retention**:

1. **Retention Period**:
   - 7 days
   - 30 days
   - 90 days (default)
   - 180 days
   - 365 days
   - Never delete

2. **Auto-Delete Old Data** (default: ON)
   - Automatically removes data older than retention period
   - Runs daily at midnight
   - Can be disabled if you want manual control

3. **Delete All Data Now**:
   - Immediately deletes all collected ML data
   - Requires confirmation
   - Cannot be undone

### Privacy Audit Trail

View all privacy-related actions:

**Privacy & Data** → **📜 Privacy Audit Trail**

Shows:
- Consent grants/revocations
- Setting changes
- Data deletions
- Timestamps for all actions

Keeps the last 50 entries.

### Exporting Privacy Settings

**Privacy & Data** → **💾 Export Privacy Settings**

- Exports all privacy settings as JSON
- Useful for backup or compliance
- Copies to clipboard for easy sharing

---

## Performance Monitoring

### Viewing Performance Stats

**Settings** → **Neural ML & Swipe Typing** → **📊 Performance Statistics**

### Metrics Displayed

1. **Usage Statistics**:
   - Total predictions made
   - Total selections
   - Days tracked

2. **Performance**:
   - Average inference time (ms)
   - Model load time (ms)
   - Memory usage

3. **Accuracy**:
   - Top-1 accuracy (%)
   - Top-3 accuracy (%)
   - Prediction success rate

### Understanding Accuracy

- **Top-1**: How often your desired word was the first prediction
- **Top-3**: How often your desired word was in the top 3 predictions

**Good targets**:
- Top-1: >70%
- Top-3: >85%

### Resetting Statistics

**Performance Statistics** → **🔄 Reset Statistics**

- Clears all performance data
- Starts fresh tracking
- Useful after major model updates

---

## Model Management

### Model Versioning

The keyboard automatically tracks model versions and their performance.

### Viewing Model Status

**Settings** → **Neural ML & Swipe Typing** → **🔄 Model Version & Rollback**

Shows:
- Current model version
- Success/failure counts
- Last success/failure timestamps
- Model health status

### Automatic Rollback

If a new model performs poorly, the system can automatically rollback to the previous version.

#### Rollback Settings

1. **Auto-Rollback Enabled** (default: ON)
   - Automatically switches to previous model if failures exceed threshold
   - Threshold: 3 consecutive failures
   - Cooldown: 5 minutes between rollbacks

2. **Pin Current Version**:
   - Prevents automatic rollbacks
   - Use when you trust a specific model
   - Can be unpinned at any time

### Manual Model Management

1. **View Previous Version**:
   - Shows metadata about the previous model
   - Displays success rate

2. **Manual Rollback**:
   - Force switch to previous model
   - Requires at least 2 model versions

3. **Reset Version History**:
   - Clears all version tracking data
   - Starts fresh with current model

---

## A/B Testing

### Overview

Compare two models side-by-side to determine which performs better.

### Accessing A/B Tests

**Settings** → **Neural ML & Swipe Typing** → **🧪 A/B Testing**

### Creating a Test

1. **Create New Test**:
   - Name: Descriptive name (e.g., "V1 vs V2")
   - Control: Baseline model (e.g., "builtin_v1")
   - Variant: Test model (e.g., "custom_v2")
   - Traffic Split: Percentage of swipes using variant (10-90%)

2. **Activate Test**:
   - Starts collecting comparison data
   - Randomly assigns swipes to control or variant
   - Assignment is consistent per user session

### Viewing Test Results

**A/B Testing** → **View Test Status**

Shows:
- Predictions made
- Win rates (which model's prediction was selected)
- Accuracy metrics (top-1, top-3)
- Average latency
- Statistical significance

### Understanding Results

- **Win Rate**: How often each model's prediction was chosen
- **Top-1 Accuracy**: Prediction accuracy for each model
- **Latency**: Inference speed comparison
- **Significance**: Whether results are statistically valid (requires ≥100 samples)

### Stopping a Test

1. **Deactivate Test**: Stops data collection
2. **View Final Report**: See complete results
3. **Apply Winner**: Manually switch to better performing model

---

## Custom Models

### Overview

Load your own trained ONNX models for swipe prediction.

### Model Requirements

1. **Encoder Model**:
   - Input: Trajectory coordinates
   - Output: Encoded representation
   - Format: ONNX (.onnx file)

2. **Decoder Model**:
   - Input: Encoded representation + previous tokens
   - Output: Next token probabilities
   - Format: ONNX (.onnx file)

### Loading Custom Models

**Settings** → **Neural ML & Swipe Typing** → **🧠 Neural Model Settings**

1. **Select Custom Encoder**:
   - Tap **"Select Custom Encoder Model"**
   - Choose .onnx file from file picker
   - Filename and size displayed

2. **Select Custom Decoder**:
   - Tap **"Select Custom Decoder Model"**
   - Choose .onnx file from file picker
   - Filename and size displayed

3. **Model Loading**:
   - Models load automatically on selection
   - Status updates in real-time
   - Interface auto-detected

### Supported Model Interfaces

The system supports two model interfaces:

1. **Built-in V2** (recommended):
   - Single `target_mask` input
   - Combined padding and causal masking

2. **Custom Models**:
   - Separate `target_padding_mask` and `target_causal_mask`
   - More flexible for custom architectures

**Interface is auto-detected** - no configuration needed.

### Training Your Own Models

See [ML Training Guide](ML_TRAINING_GUIDE.md) for:
- Data export instructions
- Training script usage
- Model conversion to ONNX
- Optimization techniques

### Reverting to Built-in Model

**Neural Model Settings** → **🔄 Reset to Built-in Model**

- Removes custom model selections
- Reloads default built-in models
- Preserves all collected data

---

## Troubleshooting

### Swipe Predictions Not Appearing

**Checklist**:
1. ✅ Neural swipe prediction enabled in settings
2. ✅ Models loaded successfully (check status in settings)
3. ✅ Swiping across multiple letters (minimum 2 keys)
4. ✅ Swipe length ≥100px
5. ✅ Gesture recognized as swipe (not multi-touch)

**Solutions**:
- Check **Model Status** in settings
- Try **Reset to Built-in Model** if using custom models
- View logs with **Show Debug Scores** enabled

### Low Prediction Accuracy

**Common Causes**:
1. Insufficient training data
2. Model not suited for your typing style
3. Custom model not properly trained

**Solutions**:
- Grant data collection consent (improves over time)
- Try built-in model first
- Check **Performance Statistics** for accuracy metrics
- Consider **A/B testing** different models

### High Inference Latency

**Target**: <50ms average
**Acceptable**: <100ms

**If latency is high**:
1. Check device performance (older devices may be slower)
2. Verify model sizes (encoder <6MB, decoder <8MB recommended)
3. Try built-in models (already optimized)
4. Check for background processes consuming CPU

**Performance Statistics** shows average inference time.

### Model Loading Errors

**Symptoms**:
- "Model failed to load" message
- No predictions appearing

**Solutions**:
1. **For built-in models**:
   - Reinstall app
   - Clear app data (loses collected data!)
   - Report issue on GitHub

2. **For custom models**:
   - Verify .onnx file format
   - Check file permissions
   - Try smaller model files
   - Validate model architecture matches requirements
   - Reset to built-in model

### Data Collection Not Working

**Checklist**:
1. ✅ Privacy consent granted
2. ✅ Data collection toggles enabled
3. ✅ Storage permissions granted

**Solutions**:
- Check **Privacy Status** in settings
- Grant consent if not already done
- Enable specific data collection types
- Verify **Auto-Delete** not set too aggressively

### A/B Test Not Producing Results

**Requirements**:
- Minimum 100 predictions for significance
- Both models must be loaded successfully
- Test must be activated

**Solutions**:
- Continue normal usage to collect more data
- Check test is activated
- Verify both models in **Model Status**

---

## Advanced Topics

### Model Comparison Metrics

When comparing models, the system uses:

1. **Win Rate** (primary):
   - Which model's prediction was selected
   - 60%+ win rate indicates clear winner

2. **Top-1 Accuracy** (tiebreaker):
   - Prediction quality metric
   - Used when win rates are close

3. **Latency** (final tiebreaker):
   - Inference speed
   - Lower is better
   - Used when accuracy is equal

### Privacy Best Practices

1. **Keep Local-Only Training enabled** unless you have a specific reason
2. **Use anonymization** to protect sensitive data
3. **Set appropriate retention periods** for your usage patterns
4. **Review audit trail** periodically
5. **Export privacy settings** for backup

### Performance Optimization

1. **Use built-in models** for best performance (already optimized)
2. **Enable caching** (automatic)
3. **Grant storage permissions** for faster model loading
4. **Keep only necessary data** with auto-delete
5. **Monitor performance statistics** to track improvements

---

## Getting Help

### Reporting Issues

**GitHub**: https://github.com/Julow/Unexpected-Keyboard/issues

Include:
- App version (Settings → About)
- Android version
- Device model
- Steps to reproduce
- Logs (if available)

### Feature Requests

Use GitHub Discussions for:
- Model improvement suggestions
- New privacy features
- Performance enhancement ideas

### Privacy Concerns

For privacy-related questions:
- Review **Privacy Status** and **Audit Trail** in settings
- Check [PRIVACY_POLICY.md](PRIVACY_POLICY.md)
- Contact maintainers via GitHub

---

## FAQ

### Q: Is my data sent to the cloud?

**A**: No, by default all data stays on your device with **Local-Only Training** enabled. Data is only exported if you explicitly enable **Allow Data Export** and manually export files.

### Q: Can I use the keyboard without data collection?

**A**: Yes! Data collection requires consent. The neural swipe typing works with the built-in model without any data collection.

### Q: How much storage does data collection use?

**A**: Typically <10MB for 90 days of data with standard usage. Auto-delete keeps it under control.

### Q: Can I train my own models?

**A**: Yes! Export your data, use the training scripts (see ML Training Guide), and load custom ONNX models.

### Q: How do I know which model is better?

**A**: Use **A/B Testing** to compare models side-by-side with real usage data. The system calculates win rates and statistical significance.

### Q: What happens if a model update breaks predictions?

**A**: **Auto-Rollback** automatically switches back to the previous working model after 3 consecutive failures.

### Q: How can I see what data is collected?

**A**: Enable **Allow Data Export**, export data via settings, and review the JSON files. All data formats are documented.

### Q: Does the built-in model improve over time?

**A**: The built-in model is static. Improvement requires:
- Collecting usage data (with consent)
- Training updated models
- Loading new models via custom model feature

---

## Version History

- **v1.32.904**: Complete Phase 6 test coverage
- **v1.32.903**: Privacy controls (Phase 6.5)
- **v1.32.900-901**: Rollback capability (Phase 6.4)
- **v1.32.899**: A/B testing framework (Phase 6.3)
- **v1.32.897-898**: Model versioning (Phase 6.2)
- **v1.32.896**: Performance monitoring (Phase 6.1)
- **v1.32.431**: Custom model support (Phase 5.1)

---

## Credits

**Neural Swipe Typing** developed as part of Unexpected Keyboard by:
- Architecture & Implementation: Claude Code
- Testing & Integration: Community Contributors
- Privacy Design: GDPR/CCPA Compliance Standards

Licensed under GPL-3.0

---

*Last Updated: 2025-11-27*
*For developers: See [ML_TRAINING_GUIDE.md](ML_TRAINING_GUIDE.md) and [ARCHITECTURE.md](ARCHITECTURE.md)*
