use std::collections::HashMap;

pub const KIMI_DEFAULT_MODEL: &str = "kimi-for-coding";

/// Wire model ids the Kimi coding endpoint serves. Anything outside this list
/// is rejected before it reaches the upstream, which answers unknown ids with
/// a default model rather than an error — silently running the wrong model.
pub const KIMI_WIRE_MODELS: &[&str] = &[
    "kimi-for-coding",
    "kimi-for-coding-highspeed",
    "k3",
    "k3-256k",
];

static ALIAS_TARGETS: once_cell::sync::Lazy<HashMap<&'static str, &'static str>> =
    once_cell::sync::Lazy::new(|| {
        let mut m = HashMap::new();
        m.insert("haiku", KIMI_DEFAULT_MODEL);
        m.insert("claude-haiku-4-5", KIMI_DEFAULT_MODEL);
        m.insert("claude-haiku-4-5-20251001", KIMI_DEFAULT_MODEL);
        m.insert("sonnet", KIMI_DEFAULT_MODEL);
        m.insert("claude-sonnet-4-6", KIMI_DEFAULT_MODEL);
        m.insert("claude-sonnet-5", KIMI_DEFAULT_MODEL);
        m.insert("opus", KIMI_DEFAULT_MODEL);
        m.insert("claude-opus-4-7", KIMI_DEFAULT_MODEL);
        m.insert("claude-opus-4-8", KIMI_DEFAULT_MODEL);
        m.insert("fable", KIMI_DEFAULT_MODEL);
        m.insert("claude-fable-5", KIMI_DEFAULT_MODEL);
        m.insert("kimi-for-coding", KIMI_DEFAULT_MODEL);
        m
    });

/// Real Kimi ids pass through untouched; Anthropic-style aliases (used when
/// Kimi is the configured alias provider) fall back to the default model.
pub fn resolve_model(model: &str) -> String {
    if KIMI_WIRE_MODELS.contains(&model) {
        return model.to_string();
    }
    ALIAS_TARGETS
        .get(model)
        .copied()
        .unwrap_or(KIMI_DEFAULT_MODEL)
        .to_string()
}

pub fn assert_allowed_model(model: &str) -> Result<(), ModelNotAllowedError> {
    if !KIMI_WIRE_MODELS.contains(&model) {
        return Err(ModelNotAllowedError {
            model: model.to_string(),
        });
    }
    Ok(())
}

#[derive(Debug)]
pub struct ModelNotAllowedError {
    pub model: String,
}

impl std::fmt::Display for ModelNotAllowedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Model not allowed: {}", self.model)
    }
}

impl std::error::Error for ModelNotAllowedError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_haiku_to_default() {
        assert_eq!(resolve_model("haiku"), KIMI_DEFAULT_MODEL);
    }

    #[test]
    fn resolve_opus_4_8_to_default() {
        assert_eq!(resolve_model("claude-opus-4-8"), KIMI_DEFAULT_MODEL);
    }

    #[test]
    fn resolve_claude_5_aliases_to_default() {
        for model in ["claude-sonnet-5", "fable", "claude-fable-5"] {
            assert_eq!(resolve_model(model), KIMI_DEFAULT_MODEL);
        }
    }

    #[test]
    fn resolve_unknown_to_default() {
        assert_eq!(resolve_model("unknown-model"), KIMI_DEFAULT_MODEL);
    }

    #[test]
    fn resolve_kimi_for_coding() {
        assert_eq!(resolve_model("kimi-for-coding"), KIMI_DEFAULT_MODEL);
    }

    #[test]
    fn assert_allowed_accepts_default() {
        assert!(assert_allowed_model(KIMI_DEFAULT_MODEL).is_ok());
    }

    #[test]
    fn assert_allowed_rejects_other() {
        assert!(assert_allowed_model("kimi-k2.6").is_err());
    }

    #[test]
    fn every_wire_model_passes_through_unchanged() {
        for model in KIMI_WIRE_MODELS {
            assert_eq!(&resolve_model(model), model);
            assert!(assert_allowed_model(model).is_ok(), "{model} rejected");
        }
    }

    #[test]
    fn k3_is_not_collapsed_to_the_coding_model() {
        assert_eq!(resolve_model("k3"), "k3");
        assert_eq!(resolve_model("k3-256k"), "k3-256k");
        assert_eq!(
            resolve_model("kimi-for-coding-highspeed"),
            "kimi-for-coding-highspeed"
        );
    }
}
