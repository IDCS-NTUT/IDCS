"""Conservative idle proof for configuration changes; no command authority."""


def configuration_engaged(safety, last_intent_active: bool) -> bool:
    return bool(last_intent_active or not (
        safety.valid and safety.sample_age_ms is not None and safety.sample_age_ms <= 750
        and safety.manual_active and not safety.auto_allowed))
