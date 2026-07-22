from scripts.render_home_scan_flyby import (
    flyby_status_marker,
    summarize_frame_stats,
)


def test_failed_frame_audit_never_emits_success_marker() -> None:
    frame_signal = summarize_frame_stats(
        [
            {
                "nonblack_fraction": 0.0,
                "mean_luma": 0.0,
                "clipped_fraction": 0.0,
            }
        ]
    )

    assert frame_signal["pass"] is False
    assert flyby_status_marker(frame_signal["pass"]) == "GAUSSIAN_FLYBY_FAIL"


def test_passing_frame_audit_emits_success_marker() -> None:
    frame_signal = summarize_frame_stats(
        [
            {
                "nonblack_fraction": 0.5,
                "mean_luma": 0.25,
                "clipped_fraction": 0.0,
            }
        ]
    )

    assert frame_signal["pass"] is True
    assert flyby_status_marker(frame_signal["pass"]) == "GAUSSIAN_FLYBY_OK"
