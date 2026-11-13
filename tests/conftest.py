"""Shared pytest configuration for isolated local test execution.

Temporary storage and deterministic test-only credentials are established
before application imports. No production configuration or study records are
required by these fixtures.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


# Several test modules import app.config at collection time. Establish one
# isolated configuration before any module import so the suite does not depend
# on alphabetical collection order.
_SUITE_DATA = Path(tempfile.mkdtemp(prefix="tta-study-suite-"))
os.environ.update(
    {
        "STUDY_DATA_DIR": str(_SUITE_DATA),
        "STUDY_TOKEN_SECRET": "test-token-secret",
        "STUDY_CODE_PEPPER": "test-code-pepper",
        "STUDY_IP_PEPPER": "test-ip-pepper",
        "STUDY_ADMIN_TOKEN": "test-admin-token",
        "COMPENSATION_VAULT_KEY": (
            "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA="
        ),
        "CONSENT_EVIDENCE_SIGNING_KEY": (
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        ),
        "FORMAL_STUDY_ENABLED": "0",
        "AGENT_MODE": "mock",
        "PUBLIC_BASE_URL": "http://testserver",
        "STUDY_INSTANCE_ID": "test",
        "PUBLIC_ENROLLMENT_ENABLED": "1",
    }
)
