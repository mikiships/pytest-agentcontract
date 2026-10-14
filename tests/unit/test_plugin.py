"""Behavioral tests for the pytest plugin fixtures."""

import pytest

pytest_plugins = ["pytester"]


def test_replay_skips_when_cassette_is_missing(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_missing_cassette(ac_replay_engine):
            raise AssertionError("missing cassette must skip before the test runs")
        """
    )

    result = pytester.runpytest("--ac-replay", "-rs")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(
        ["*No cassette found at*test_missing_cassette.agentrun.json*"]
    )
