"""One-time helper: produce a YouTube OAuth refresh token for headless uploads.

Usage:
    1. Create OAuth 2.0 Desktop client credentials in Google Cloud Console
       (APIs & Services → Credentials → Create Credentials → OAuth client ID → Desktop app).
    2. Download the JSON, save it as `client_secret.json` in this directory.
    3. Run:  python scripts/generate_youtube_token.py
    4. A browser will open — grant access. The refresh token is printed to the console.
    5. Add to your .env or GitHub Secrets:
         YOUTUBE_CLIENT_ID=...
         YOUTUBE_CLIENT_SECRET=...
         YOUTUBE_REFRESH_TOKEN=...
"""
from __future__ import annotations

import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def main() -> int:
    here = Path(__file__).resolve().parent
    secret = here / "client_secret.json"
    if not secret.exists():
        print(f"ERROR: place your OAuth client JSON at {secret}", file=sys.stderr)
        return 1

    flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")

    print("\n=== Add these to your .env / GitHub Secrets ===")
    print(f"YOUTUBE_CLIENT_ID={creds.client_id}")
    print(f"YOUTUBE_CLIENT_SECRET={creds.client_secret}")
    print(f"YOUTUBE_REFRESH_TOKEN={creds.refresh_token}")
    print()
    if not creds.refresh_token:
        print(
            "WARNING: no refresh_token returned. Revoke the app's access at "
            "https://myaccount.google.com/permissions and re-run.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
