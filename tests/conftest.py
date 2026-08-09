"""Shared test configuration that keeps unit tests independent of local secrets."""

import os

os.environ.setdefault("EXAONE_API_KEY", "test-exaone-api-key")
os.environ.setdefault("UPSTAGE_API_KEY", "test-upstage-api-key")
