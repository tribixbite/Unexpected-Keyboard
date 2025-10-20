# GitHub Actions Keystore Setup

## Problem
GitHub Actions builds were creating APKs with different signing keys each time, causing `INSTALL_FAILED_UPDATE_INCOMPATIBLE` errors for testers trying to update.

## Solution
Upload your local debug keystore to GitHub Secrets so all builds use the same signing key.

## Steps

### 1. Copy the Encrypted Keystore Content

The file `debug.keystore.asc` has been created. Copy its contents:

```bash
cat debug.keystore.asc
```

### 2. Add to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `DEBUG_KEYSTORE`
5. Value: Paste the entire contents of `debug.keystore.asc`
6. Click **Add secret**

### 3. Verify the Fix

The workflow file `.github/workflows/build.yml` has been updated to:
- Restore the debug keystore from GitHub Secrets before building
- Use the same keystore environment variables as `make-apk.yml`

After adding the secret, the next GitHub Actions build will:
✅ Use the same signing key as your local builds
✅ Allow testers to update without uninstalling
✅ Maintain signature consistency across all builds

## For Testers (One-Time)

Since previous builds used different keys, testers need to:

1. **Uninstall** the old version:
   - Settings → Apps → Unexpected Keyboard (Debug) → Uninstall

2. **Install** the new version from GitHub releases:
   - Download the latest APK
   - Install it
   - Re-enable the keyboard in system settings

3. **Future updates** will work normally without uninstalling!

## Keystore Details

- **Location**: `debug.keystore`
- **Encrypted**: `debug.keystore.asc` (gpg-encrypted with passphrase "debug0")
- **Alias**: debug
- **Password**: debug0
- **Validity**: Until January 2053
- **Algorithm**: SHA256withRSA (2048-bit)

## Verification

To verify the keystore fingerprint:

```bash
keytool -list -v -keystore debug.keystore -storepass debug0 | grep SHA256
```

Expected output:
```
SHA256: 0F:37:81:FF:2F:E2:4B:CC:12:EF:EF:41:6F:CD:99:FF:AB:0F:7F:10:46:7F:5C:B3:83:B8:62:F7:3A:7D:E6:92
```

All builds (local and GitHub) should show this same SHA256 fingerprint.
