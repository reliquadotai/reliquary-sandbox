import pytest
from episode_fixtures import episode_settings
from pydantic import ValidationError

from reliquary_sandbox_service.settings import Settings


def test_episodes_are_off_by_default():
    settings = Settings(tokens={"evals": "e" * 32})
    assert settings.episodes_enabled is False
    assert settings.episode_allow_unquoted_disk is False


def test_complete_episode_configuration_is_accepted(tmp_path):
    settings, _ = episode_settings(tmp_path)
    assert settings.episode_capacity == 4
    assert settings.episode_max_disk_bytes == 10 * 1024**3


@pytest.mark.parametrize("override", [
    {"machine_id": None},
    {"machine_key_file": None},
    {"validator_public_keys": {}},
    {"validator_public_keys": {"v-1": "not-a-key"}},
    {"episode_images": ["registry.example/task:latest"]},
    {"episode_images": []},
    {"episode_envs": {"fake-env": "no entry point"}},
    {"heartbeat_url": "http://validator.example/heartbeat"},
    {"machine_id": "bad id with spaces"},
    {"episode_images": ["a@sha256:zzz"]},
    {"episode_envs": {"bad env!": "fake_episode_env:sandbox_task"}},
    {"machine_key_file": "/nonexistent/key.pem"},
])
def test_incomplete_episode_configuration_is_refused(tmp_path, override):
    with pytest.raises(ValidationError):
        episode_settings(tmp_path, **override)


def test_heartbeat_url_may_carry_a_path(tmp_path):
    settings, _ = episode_settings(tmp_path, heartbeat_url="https://validator.example/sandbox/heartbeat/")
    assert settings.heartbeat_url == "https://validator.example/sandbox/heartbeat"


def test_argument_cap_is_fixed_under_the_kernel_argv_element_limit():
    from reliquary_sandbox.observation import MAX_ARGUMENT_BYTES
    from reliquary_sandbox_service.settings import Settings

    assert MAX_ARGUMENT_BYTES == 120 * 1024 < 128 * 1024  # MAX_ARG_STRLEN
    assert "episode_max_argument_bytes" not in Settings.model_fields


@pytest.mark.parametrize("override", [
    {"episode_helper_busybox": None},
    {"episode_helper_busybox": "/nonexistent/busybox"},
    {"episode_helper_busybox_sha256": None},
    {"episode_helper_busybox_sha256": "abc"},
])
def test_the_helper_busybox_must_exist_and_be_pinned(tmp_path, override):
    with pytest.raises(ValidationError):
        episode_settings(tmp_path, **override)


def test_the_grading_box_start_bound_stays_below_the_retry_margin(tmp_path):
    from reliquary_sandbox_service.episodes.grader import RETRY_MARGIN_S

    assert episode_settings(tmp_path)[0].episode_grading_create_timeout_s == 90
    with pytest.raises(ValidationError):
        episode_settings(tmp_path, episode_grading_create_timeout_s=RETRY_MARGIN_S + 1)


def test_the_transcript_cap_is_eight_mib_at_most():
    from reliquary_sandbox_service.settings import MAX_TRANSCRIPT_BYTES

    assert MAX_TRANSCRIPT_BYTES == 8 * 1024**2
    settings = Settings(tokens={"evals": "e" * 32})
    assert settings.episode_max_transcript_bytes == MAX_TRANSCRIPT_BYTES
    with pytest.raises(ValidationError):
        Settings(tokens={"evals": "e" * 32}, episode_max_transcript_bytes=MAX_TRANSCRIPT_BYTES + 1)


def test_the_transcript_cap_must_leave_room_for_one_more_call_and_the_final():
    from reliquary_sandbox_service.settings import transcript_reserve_bytes

    reserve = transcript_reserve_bytes(65536)
    assert reserve >= 6 * 65536
    with pytest.raises(ValidationError):
        Settings(tokens={"evals": "e" * 32}, episode_output_cap_bytes=2 * 1024**2)
    with pytest.raises(ValidationError):
        Settings(tokens={"evals": "e" * 32}, episode_max_transcript_bytes=reserve)


def test_verifiers_entries_and_their_options_are_accepted(tmp_path):
    settings, _ = episode_settings(
        tmp_path, episode_envs={"fake-env": "fake_episode_env:sandbox_task",
                                "toy": "verifiers:toy-bridge-env==unpackaged"},
        episode_env_options={"toy": {"tools": ["bash"]}})
    assert settings.episode_envs["toy"] == "verifiers:toy-bridge-env==unpackaged"
    assert settings.episode_env_options == {"toy": {"tools": ["bash"]}}


@pytest.mark.parametrize("override", [
    {"episode_envs": {"toy": "verifiers:toy-bridge-env"}},
    {"episode_envs": {"toy": "verifiers:toy-bridge-env>=1"}},
    {"episode_envs": {"toy": "verifiers:org/toy==1"}},
    {"episode_envs": {"toy": "verifiers:toyenv"}},
    {"episode_envs": {"toy": "verifiers:toy.env:sandbox_task"}},
    {"episode_env_options": {"fake-env": {"tools": ["bash"]}}},
    {"episode_env_options": {"absent": {}}},
])
def test_malformed_verifiers_entries_and_stray_options_are_refused(tmp_path, override):
    with pytest.raises(ValidationError):
        episode_settings(tmp_path, **override)


def test_a_verifiers_host_has_a_scoring_slot_per_concurrent_episode(tmp_path):
    """Every episode may grade at once: fewer slots than the capacity would abort grades
    that find none free. A host without a verifiers entry starts no scoring process."""
    toy = {"toy": "verifiers:toy-bridge-env==unpackaged"}
    assert episode_settings(tmp_path, episode_envs=toy, episode_capacity=8,
                            scoring_slots=8)[0].scoring_slots == 8
    with pytest.raises(ValidationError, match="scoring_slots"):
        episode_settings(tmp_path, episode_envs=toy, episode_capacity=8, scoring_slots=7)
    legacy, _ = episode_settings(tmp_path, episode_capacity=8, scoring_slots=7)
    assert legacy.scoring_slots == 7
