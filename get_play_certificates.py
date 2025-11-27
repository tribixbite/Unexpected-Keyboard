#!/usr/bin/env python3
"""
Google Play Console Certificate Downloader
Downloads signing certificates from Google Play Console API
"""

import json
import sys
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# Scopes required for Google Play Console API
SCOPES = ['https://www.googleapis.com/auth/androidpublisher']

def get_play_console_service():
    """Authenticate and return Google Play Console API service."""
    print("Authenticating with Google Play Console...")
    
    # Try to use Application Default Credentials first
    try:
        from google.auth import default
        credentials, project = default(scopes=SCOPES)
        credentials.refresh(Request())
        print(f"Using default credentials for project: {project}")
        return build('androidpublisher', 'v3', credentials=credentials)
    except Exception as e:
        print(f"Default credentials failed: {e}")
        print("You need to set up Google Play Console API credentials.")
        print("\nTo get your credentials:")
        print("1. Go to Google Cloud Console: https://console.cloud.google.com/")
        print("2. Enable Google Play Android Developer API")
        print("3. Create a service account with Google Play Console access")
        print("4. Download the JSON key file")
        print("5. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        return None

def list_app_certificates(service, package_name):
    """List all certificates for the given package."""
    try:
        result = service.edits().certificates().list(
            packageName=package_name
        ).execute()
        return result.get('certificates', [])
    except Exception as e:
        print(f"Error getting certificates: {e}")
        return []

def main():
    package_name = "juloo.keyboard2"  # Your app's package name
    
    service = get_play_console_service()
    if not service:
        return
    
    print(f"\nGetting certificates for package: {package_name}")
    certificates = list_app_certificates(service, package_name)
    
    if not certificates:
        print("No certificates found. This could mean:")
        print("1. The app hasn't been uploaded to Google Play Console yet")
        print("2. You don't have the right permissions")
        print("3. Google Play App Signing is not enabled")
        return
    
    print(f"\nFound {len(certificates)} certificate(s):")
    for i, cert in enumerate(certificates):
        print(f"\nCertificate {i+1}:")
        print(f"  SHA-1: {cert.get('sha1', 'N/A')}")
        print(f"  SHA-256: {cert.get('sha256', 'N/A')}")
        print(f"  Certificate Type: {cert.get('certificateType', 'N/A')}")
    
    # Save certificates to file
    with open('play_certificates.json', 'w') as f:
        json.dump(certificates, f, indent=2)
    print(f"\nCertificates saved to: play_certificates.json")

if __name__ == '__main__':
    main()