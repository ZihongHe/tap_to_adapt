"""Command-line utility: provision simulation pipeline roster v10.

This module supports the study's local tools and test fixtures. Inspect the
entry point and explicit path arguments before running data-writing operations.
"""

from provision_simulation_pipeline_roster_v8 import main


if __name__ == "__main__":
    main(
        expected_protocol_version="three-day-online-2026-07-18-v10.3",
        roster_version="v10",
    )
