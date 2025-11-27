# Privacy Policy - Neural Swipe Typing

**Effective Date**: 2025-11-27
**Version**: 1.0
**App Version**: v1.32.903+

---

## Overview

Unexpected Keyboard with Neural Swipe Typing is committed to protecting your privacy. This privacy policy explains how we handle data in the neural machine learning features.

**Core Principle**: Your data stays on your device by default. All data collection is opt-in and under your complete control.

---

## Data Collection Philosophy

### Five Privacy Principles

1. **Consent First**: No data collection without your explicit consent
2. **Transparency**: Clear explanations of what, why, and how
3. **User Control**: Easy opt-out and data deletion
4. **Data Minimization**: Only collect necessary data
5. **Local by Default**: Everything stays on your device

---

## What Data Can Be Collected

### 1. Swipe Gesture Data

**Purpose**: Improve swipe recognition accuracy

**What is collected**:
- Touch coordinates (x, y) during swipe gestures
- Timestamps of each touch point
- Key sequence detected from gesture
- Device screen dimensions (for normalization)
- Selected prediction (what word you chose)

**What is NOT collected**:
- Content of typed text
- Individual key presses outside of swipes
- Passwords or sensitive fields
- App names or context

**Storage**: SQLite database on device (`swipe_ml_data.db`)

**Retention**: Default 90 days, configurable (7-365 days or never)

### 2. Performance Statistics

**Purpose**: Monitor prediction quality and system performance

**What is collected**:
- Prediction accuracy metrics (top-1, top-3)
- Inference latency measurements
- Model load times
- Success/failure counts
- First statistics timestamp

**What is NOT collected**:
- Actual predictions or text
- User behavior patterns
- App usage context

**Storage**: SharedPreferences (`neural_performance_stats`)

**Retention**: Persistent until manually reset

### 3. Error Logs (Optional, Default: OFF)

**Purpose**: Debug model loading and prediction failures

**What is collected**:
- Error messages from ONNX runtime
- Stack traces (when available)
- Model file paths (if custom models fail to load)
- Timestamp of errors

**What is NOT collected**:
- User input or text
- System information beyond error context
- Network requests or API calls

**Storage**: Android Logcat (temporary) and optionally to file

**Retention**: Logs rotate automatically, controlled by system

---

## Data Usage

### How Collected Data is Used

1. **Swipe Data**:
   - Train improved neural network models
   - Analyze gesture patterns for algorithm optimization
   - Debug recognition failures
   - Export for external training (only if you enable and export)

2. **Performance Stats**:
   - Display in settings UI for transparency
   - Track model improvements over time
   - Identify performance regressions
   - Inform A/B testing decisions

3. **Error Logs**:
   - Troubleshoot model loading issues
   - Fix bugs in prediction pipeline
   - Improve error handling

### How Data is NOT Used

- ❌ **Never** shared with third parties
- ❌ **Never** transmitted over network (unless you manually export and share)
- ❌ **Never** used for advertising or analytics
- ❌ **Never** combined with other apps' data
- ❌ **Never** used to identify individuals

---

## Consent Management

### Granting Consent

To enable data collection:

1. Open **Settings** → **Neural ML & Swipe Typing** → **Privacy & Data**
2. Tap **"Data Collection Consent"**
3. Read the consent dialog explaining data collection
4. Tap **"Grant Consent"** to approve

**What happens**:
- Consent flag set in SharedPreferences
- Timestamp recorded for audit trail
- Consent version stored (for future policy updates)
- Data collection begins on next swipe

### Revoking Consent

To disable data collection:

1. Open **Privacy & Data** → **Data Collection Consent**
2. Tap **"Revoke & Delete"** or **"Revoke Only"**

**"Revoke & Delete"**:
- Stops all data collection immediately
- Deletes all collected swipe data
- Deletes all performance statistics
- Records revocation in audit trail

**"Revoke Only"**:
- Stops data collection immediately
- Preserves existing data for your reference
- You can delete manually later

### Consent Versioning

If we update the privacy policy or data collection practices:
- Current consent version: **1**
- Future updates will show consent dialog again
- You choose whether to accept new terms
- No data collection under new terms without re-consent

---

## Data Control Features

### Granular Collection Controls

Even with consent granted, you control exactly what is collected:

**Settings** → **Privacy & Data** → **Data Collection Controls**

- **Collect Swipe Data** (default: ON)
- **Collect Performance Data** (default: ON)
- **Collect Error Logs** (default: OFF)

Each toggle is independent. Changes take effect immediately.

### Privacy Settings

**Settings** → **Privacy & Data** → **Privacy Settings**

1. **Anonymize Data** (default: ON):
   - Removes device identifiers
   - Strips precise timestamps (rounds to hour)
   - Normalizes screen dimensions
   - Removes build fingerprints

2. **Local-Only Training** (default: ON):
   - All data stays on device
   - No cloud synchronization
   - No external transmission
   - Complete control

3. **Allow Data Export** (default: OFF):
   - Enables manual export via settings
   - Data exported as JSON files
   - You control when and where exported
   - Useful for training custom models

4. **Allow Model Sharing** (default: OFF):
   - Enables sharing of trained model files
   - Models remain local unless explicitly shared
   - No automatic upload or sync
   - Requires manual export

### Data Retention

**Settings** → **Privacy & Data** → **Data Retention**

**Retention Period** (default: 90 days):
- 7 days (minimal)
- 30 days (short)
- 90 days (recommended)
- 180 days (extended)
- 365 days (yearly)
- Never delete (manual control)

**Auto-Delete** (default: ON):
- Automatically removes data older than retention period
- Runs daily at midnight (local time)
- Logs deletions in audit trail
- Can be disabled for manual control

**Delete All Data Now**:
- Immediately deletes all ML data
- Requires confirmation dialog
- Cannot be undone
- Logs in audit trail

---

## Data Security

### On-Device Storage

- **SQLite database**: Standard Android SQLite with file permissions
- **SharedPreferences**: Android encrypted storage (on supported devices)
- **File permissions**: App-private directory, inaccessible to other apps
- **No cloud backup**: Data excluded from Android Auto Backup

### Access Control

- **App sandbox**: Android security model isolates app data
- **No network access**: Neural features don't use internet permission
- **No external storage**: Data never written to SD card or shared storage
- **Root protection**: Standard Android security applies

### Data Deletion

- **Revoke consent**: Immediate deletion via settings
- **Uninstall app**: All data deleted automatically
- **Clear app data**: System settings deletes all data
- **Auto-delete**: Scheduled cleanup per retention policy

---

## Compliance

### GDPR (European Union)

**Legal Basis**: Consent (Article 6(1)(a))

**Your Rights**:
1. **Right to Access**: View all collected data via export feature
2. **Right to Rectification**: Delete incorrect data and re-collect
3. **Right to Erasure**: Delete all data via "Delete All Data Now"
4. **Right to Restrict Processing**: Disable collection toggles
5. **Right to Data Portability**: Export data as JSON
6. **Right to Object**: Revoke consent at any time
7. **Right to Withdraw Consent**: Revoke consent in settings

**Data Controller**: User (you) - data stays on your device
**Data Processor**: App (Unexpected Keyboard)
**Data Transfer**: None - all local processing

### CCPA (California)

**Consumer Rights**:
1. **Right to Know**: Privacy settings show all collected data types
2. **Right to Delete**: "Delete All Data Now" feature
3. **Right to Opt-Out**: Revoke consent or disable collection toggles
4. **Right to Non-Discrimination**: All features work without consent

**Sale of Personal Information**: We do NOT sell personal information

**Business Purpose**: Improve keyboard functionality (local only)

### General Compliance

- ✅ Data minimization (only necessary data)
- ✅ Purpose limitation (used only for stated purposes)
- ✅ Storage limitation (retention periods enforced)
- ✅ Integrity and confidentiality (Android security)
- ✅ Accountability (audit trail of all actions)

---

## Privacy Audit Trail

### What is Logged

All privacy-related actions are logged with timestamps:

- Consent grants and revocations
- Setting changes (collection toggles, retention, etc.)
- Data exports
- Data deletions (manual and auto-delete)
- Privacy policy views

### Viewing Audit Trail

**Settings** → **Privacy & Data** → **Privacy Audit Trail**

Shows:
- Action type
- Timestamp
- Description
- Last 50 entries

**Audit trail is not deletable** (except via "Delete All Data Now")

---

## Children's Privacy

Unexpected Keyboard does not knowingly collect data from children under 13 (COPPA) or 16 (GDPR).

**Parental Guidance**:
- Review privacy settings before child uses keyboard
- Consider disabling all data collection
- Use "Local-Only Training" mode
- Supervise model loading and export features

---

## Third-Party Services

### ONNX Runtime

**Purpose**: Neural network inference engine
**Data Access**: Only processes data in-memory, no persistence
**Privacy**: Local computation only, no network access
**License**: MIT License
**Website**: https://onnxruntime.ai/

### No Other Third Parties

- No analytics services
- No crash reporting services
- No advertising networks
- No cloud services
- No telemetry

---

## Changes to Privacy Policy

### Notification

If we update this privacy policy:

1. **Version number** will increment
2. **Consent version** may increment (requires re-consent)
3. **Effective date** will update
4. **Changelog** will be added to document

### Consent Updates

For material changes:
- Consent dialog will reappear in app
- You choose whether to accept new terms
- No data collection under new policy without re-consent
- Old data remains subject to old policy terms

---

## Data Breach Notification

### Prevention

- All data stored locally (no cloud breaches)
- Android app sandbox security
- No network transmission
- Standard Android encryption

### In Case of Device Loss

If you lose your device:

1. **Data is protected** by Android lock screen
2. **Remote wipe** via Find My Device clears all app data
3. **No cloud sync** means data only on lost device
4. **Consider encryption** on Android 7.0+ (enabled by default)

### Reporting Security Issues

If you discover a security vulnerability:

**Email**: security@unexpected-keyboard.org (if available)
**GitHub**: https://github.com/Julow/Unexpected-Keyboard/security

Please provide:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (optional)

---

## Contact Information

### Questions About Privacy

- **GitHub Issues**: https://github.com/Julow/Unexpected-Keyboard/issues
- **Documentation**: See [NEURAL_SWIPE_GUIDE.md](NEURAL_SWIPE_GUIDE.md)

### Data Subject Requests

For GDPR/CCPA requests:

1. **Access**: Use "Export Privacy Settings" in app
2. **Delete**: Use "Delete All Data Now" in app
3. **Portability**: Use data export feature
4. **Questions**: Open GitHub issue with [Privacy] tag

---

## Jurisdiction

This app is developed as open-source software. Privacy practices comply with:

- **GDPR** (European Union)
- **CCPA** (California, USA)
- **PIPEDA** (Canada)
- **General best practices** for data protection

Specific legal requirements vary by location. This policy aims to meet the strictest standards globally.

---

## Open Source Transparency

### Code Availability

All privacy-related code is open source:

**Repository**: https://github.com/Julow/Unexpected-Keyboard
**License**: GPL-3.0

**Key Files**:
- `PrivacyManager.kt` - Privacy controls and consent management
- `MLDataCollector.kt` - Data collection with privacy checks
- `NeuralPerformanceStats.kt` - Performance monitoring
- `SwipeMLDataStore.kt` - Data storage and export

**Audit Welcome**: Community code review encouraged

---

## Summary

**Your data, your choice, your device.**

- ✅ **Local-only by default** - Nothing leaves your device
- ✅ **Opt-in consent** - You decide if/what to collect
- ✅ **Full transparency** - View all collected data
- ✅ **Easy deletion** - One-tap data removal
- ✅ **Open source** - Verify our privacy claims in code

**Questions?** See [NEURAL_SWIPE_GUIDE.md](NEURAL_SWIPE_GUIDE.md) or open a GitHub issue.

---

*Last Updated: 2025-11-27*
*Version: 1.0*
*Effective: All versions v1.32.903 and later*
