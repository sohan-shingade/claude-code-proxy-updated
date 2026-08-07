use crate::{
    anthropic::{json_error, schema::MessagesRequest},
    config::AliasProvider,
    provider::{CliHandlers, Provider, RequestContext},
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use axum::{http::StatusCode, response::Response};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, LazyLock};

pub const ANTHROPIC_STYLE_ALIASES: &[&str] = &["haiku", "sonnet", "opus", "fable"];

pub const ANTHROPIC_MODELS: &[&str] = &[
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
];

pub const CURSOR_PREFIXES: &[&str] = &["cursor:", "cursor-plan:", "cursor-ask:"];

const CURSOR_LEGACY_MODELS: &[&str] = &[
    "cursor",
    "cursor-agent",
    "cursor-composer",
    "cursor-composer-fast",
    "cursor-plan",
    "cursor-ask",
    "composer-2.5",
    "composer-2.5-fast",
];

pub(crate) const CODEX_MODELS: &[&str] = &[
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
];

pub(crate) const CODEX_MODEL_ALIASES: &[&str] = &["gpt-5", "gpt-5-mini", "gpt-5-codex"];

pub(crate) const KIMI_MODELS: &[&str] = &["kimi-for-coding", "kimi-k2.6", "k2.6"];
/// The Kimi aliases all collapse to the same wire model, so only the canonical
/// id earns a picker entry; the rest stay routable when typed explicitly.
pub(crate) const KIMI_PICKER_MODELS: &[&str] = &["kimi-for-coding"];
pub(crate) const GROK_MODELS: &[&str] = &["grok-composer-2.5-fast", "grok-4.5"];

pub struct Registry {
    alias_provider: AliasProvider,
    anthropic_passthrough_enabled: bool,
    models: BTreeMap<String, Vec<String>>,
    handlers: BTreeMap<String, Arc<dyn Provider>>,
}

impl Registry {
    pub fn new(alias_provider: AliasProvider) -> Self {
        Self::with_anthropic_passthrough(alias_provider, false)
    }

    pub fn with_anthropic_passthrough(
        alias_provider: AliasProvider,
        anthropic_passthrough_enabled: bool,
    ) -> Self {
        let mut models: BTreeMap<String, Vec<String>> = BTreeMap::new();
        if anthropic_passthrough_enabled {
            models.insert(
                "anthropic".into(),
                ANTHROPIC_MODELS
                    .iter()
                    .map(|model| (*model).to_string())
                    .collect(),
            );
        }
        models.insert("codex".into(), expand_codex_models());
        models.insert(
            "kimi".into(),
            KIMI_MODELS.iter().map(|m| (*m).to_string()).collect(),
        );
        models.insert("cursor".into(), build_cursor_models());
        models.insert(
            "grok".into(),
            GROK_MODELS
                .iter()
                .map(|model| (*model).to_string())
                .collect(),
        );

        let mut handlers = BTreeMap::new();
        for (name, entries) in &models {
            let handler: Arc<dyn Provider> = match name.as_str() {
                "anthropic" => Arc::new(crate::providers::anthropic::AnthropicProvider::new()),
                "codex" => Arc::new(crate::providers::codex::CodexProvider::new()),
                "kimi" => Arc::new(crate::providers::kimi::KimiProvider::new()),
                "cursor" => Arc::new(crate::providers::cursor::CursorProvider::new()),
                "grok" => Arc::new(crate::providers::grok::GrokProvider::new()),
                _ => Arc::new(PlaceholderProvider::new(name, entries.clone())),
            };
            handlers.insert(name.clone(), handler);
        }

        Self {
            alias_provider,
            anthropic_passthrough_enabled,
            models,
            handlers,
        }
    }

    pub fn with_default_alias() -> Self {
        Self::with_anthropic_passthrough(
            crate::config::alias_provider(),
            crate::config::anthropic_passthrough_enabled(),
        )
    }

    pub fn list_provider_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.handlers.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    pub fn provider(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.handlers.get(name).cloned()
    }

    pub fn supported_models_for(&self, provider: &str) -> Vec<String> {
        let mut models = self.models.get(provider).cloned().unwrap_or_default();
        if provider == self.alias_provider.as_str() {
            for alias in ANTHROPIC_STYLE_ALIASES {
                if !models.iter().any(|value| value == alias) {
                    models.push((*alias).to_string());
                }
            }
            if !self.anthropic_passthrough_enabled {
                for model in ANTHROPIC_MODELS {
                    if !models.iter().any(|value| value == model) {
                        models.push((*model).to_string());
                    }
                }
            }
        }
        models.sort_unstable();
        models
    }

    pub fn all_supported_models(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for provider in self.handlers.keys() {
            for model in self.supported_models_for(provider) {
                out.push((model, provider.clone()));
            }
        }
        out
    }

    pub fn grouped_models(&self) -> BTreeMap<String, Vec<String>> {
        let mut out = BTreeMap::new();
        for provider in self.handlers.keys() {
            out.insert(provider.clone(), self.supported_models_for(provider));
        }
        out
    }

    pub fn catalog_models(&self) -> Vec<(String, String)> {
        let mut models = self
            .all_supported_models()
            .into_iter()
            .filter(|(model, _)| !ANTHROPIC_STYLE_ALIASES.contains(&model.as_str()))
            .filter(|(model, provider)| {
                provider != "kimi" || KIMI_PICKER_MODELS.contains(&model.as_str())
            })
            .map(|(model, provider)| (discovery_model_id(&provider, &model), provider))
            .filter(|(id, _)| is_gateway_discovery_id(id))
            .collect::<Vec<_>>();
        models.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
        models.dedup_by(|left, right| left.0 == right.0);
        models
    }

    pub fn provider_for_model(
        &self,
        raw_model: &str,
        session_affinity: Option<&AliasProvider>,
    ) -> Option<Arc<dyn Provider>> {
        let normalized = normalize_incoming_model(raw_model);
        if is_codex_discovery_model(&normalized) {
            return self.handlers.get("codex").cloned();
        }
        if self.anthropic_passthrough_enabled && normalized.starts_with("claude-") {
            return self.handlers.get("anthropic").cloned();
        }
        if is_anthropic_alias(&normalized) || ANTHROPIC_MODELS.contains(&normalized.as_str()) {
            let target = session_affinity.unwrap_or(&self.alias_provider);
            return self.handlers.get(target.as_str()).cloned();
        }
        if is_cursor_model(&normalized) {
            return self.handlers.get("cursor").cloned();
        }

        for (name, models) in &self.models {
            if models.iter().any(|candidate| candidate == &normalized) {
                return self.handlers.get(name).cloned();
            }
        }

        None
    }

    pub fn unknown_model_message(&self) -> String {
        let mut parts = Vec::new();
        for (provider, models) in self.grouped_models() {
            let mut models = models;
            models.sort_unstable();
            parts.push(format!("{}: {}", provider, models.join(", ")));
        }
        format!("Supported: {}.", parts.join("; "))
    }
}

pub fn normalize_incoming_model(model: &str) -> String {
    let suffix = "[1m]";
    let stripped = if model.len() >= suffix.len() && model.to_ascii_lowercase().ends_with(suffix) {
        &model[..model.len() - suffix.len()]
    } else {
        model
    };
    normalize_gateway_discovery_model(stripped)
}

pub fn is_anthropic_alias(model: &str) -> bool {
    ANTHROPIC_STYLE_ALIASES.contains(&model)
}

pub fn is_cursor_model(model: &str) -> bool {
    if CURSOR_LEGACY_MODELS.contains(&model) {
        return true;
    }

    CURSOR_PREFIXES
        .iter()
        .any(|prefix| model.starts_with(prefix))
}

struct PlaceholderProvider {
    name: &'static str,
    models: Vec<String>,
}

impl PlaceholderProvider {
    fn new(name: &str, models: Vec<String>) -> Self {
        let name = match name {
            "anthropic" => "anthropic",
            "codex" => "codex",
            "kimi" => "kimi",
            "cursor" => "cursor",
            "grok" => "grok",
            _ => "codex",
        };
        Self { name, models }
    }
}

#[async_trait]
impl Provider for PlaceholderProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn supported_models(&self) -> Vec<String> {
        self.models.clone()
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        match self.name {
            "anthropic" => &ANTHROPIC_CLI,
            "codex" => &CODEX_CLI,
            "kimi" => &KIMI_CLI,
            "cursor" => &CURSOR_CLI,
            "grok" => &GROK_CLI,
            _ => &CODEX_CLI,
        }
    }

    async fn handle_messages(&self, _body: MessagesRequest, ctx: RequestContext) -> Response {
        placeholder_provider_response("messages", &ctx.provider)
    }

    async fn handle_count_tokens(&self, _body: MessagesRequest, ctx: RequestContext) -> Response {
        placeholder_provider_response("count_tokens", &ctx.provider)
    }
}

fn placeholder_provider_response(route: &str, provider: &str) -> Response {
    let _ = route;
    json_error(
        StatusCode::NOT_IMPLEMENTED,
        "unsupported_provider_error",
        format!("provider '{}' is not yet implemented", provider),
    )
}

#[derive(Clone, Copy)]
struct PlaceholderCli {
    provider: &'static str,
}

impl CliHandlers for PlaceholderCli {
    fn login(&self) -> Result<()> {
        Err(anyhow!("{}: browser login not supported", self.provider))
    }

    fn device(&self) -> Result<()> {
        Err(anyhow!("{}: device login not supported", self.provider))
    }

    fn status(&self) -> Result<()> {
        use serde_json::Value;
        let path = crate::paths::provider_auth_file(self.provider);
        let legacy = crate::paths::provider_legacy_auth_file(self.provider);
        if crate::auth::load_auth_file_with_legacy::<Value>(&path, &legacy).is_some() {
            Ok(())
        } else {
            Err(anyhow!("Not authenticated"))
        }
    }

    fn logout(&self) -> Result<()> {
        let path = crate::paths::provider_auth_file(self.provider);
        let legacy = crate::paths::provider_legacy_auth_file(self.provider);
        let _ = crate::auth::delete_auth_file(&path, &legacy);
        Ok(())
    }
}

const ANTHROPIC_CLI: PlaceholderCli = PlaceholderCli {
    provider: "anthropic",
};
const CODEX_CLI: PlaceholderCli = PlaceholderCli { provider: "codex" };
const KIMI_CLI: PlaceholderCli = PlaceholderCli { provider: "kimi" };
const CURSOR_CLI: PlaceholderCli = PlaceholderCli { provider: "cursor" };
const GROK_CLI: PlaceholderCli = PlaceholderCli { provider: "grok" };

fn expand_codex_models() -> Vec<String> {
    let mut set = HashSet::new();
    let mut out = Vec::new();
    for model in CODEX_MODELS {
        if set.insert((*model).to_string()) {
            out.push((*model).to_string());
        }
        let fast = format!("{model}-fast");
        if set.insert(fast.clone()) {
            out.push(fast);
        }
    }
    for model in CODEX_MODEL_ALIASES {
        if set.insert((*model).to_string()) {
            out.push((*model).to_string());
        }
    }
    out.sort_unstable();
    out
}

fn build_cursor_models() -> Vec<String> {
    let mut out: Vec<String> = CURSOR_LEGACY_MODELS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    out.sort_unstable();
    out
}

fn discovery_model_id(provider: &str, model: &str) -> String {
    match provider {
        "codex" | "kimi" => gateway_discovery_model_id(model),
        _ => model.to_string(),
    }
}

fn normalize_gateway_discovery_model(model: &str) -> String {
    if let Some(raw) = codex_discovery_model_target(model) {
        return raw.to_string();
    }
    if let Some(raw) = kimi_discovery_model_target(model) {
        return raw.to_string();
    }
    model.to_string()
}

fn is_codex_discovery_model(model: &str) -> bool {
    codex_discovery_model_target(model).is_some()
}

fn is_gateway_discovery_id(model: &str) -> bool {
    model.starts_with("claude") || model.starts_with("anthropic")
}

fn gateway_discovery_model_id(model: &str) -> String {
    if model.starts_with("claude") || model.starts_with("anthropic") {
        return model.to_string();
    }
    format!("claude-{}", model.replace('.', "-"))
}

/// Reverse of `gateway_discovery_model_id`, derived from the same model lists so
/// the two directions cannot drift. Accepts both the hyphenized picker id
/// (`claude-gpt-5-6-sol`) and the dotted spelling (`claude-gpt-5.6-sol`).
fn codex_discovery_model_target(model: &str) -> Option<&'static str> {
    static TARGETS: LazyLock<HashMap<String, String>> = LazyLock::new(|| {
        let mut map = HashMap::new();
        for upstream in expand_codex_models() {
            map.insert(gateway_discovery_model_id(&upstream), upstream.clone());
            map.insert(format!("claude-{upstream}"), upstream);
        }
        map
    });
    TARGETS.get(model).map(String::as_str)
}

/// Kimi's counterpart to `codex_discovery_model_target`. Kept separate so that
/// `is_codex_discovery_model` — which routes straight to the Codex handler —
/// never matches a Kimi picker id.
fn kimi_discovery_model_target(model: &str) -> Option<&'static str> {
    static TARGETS: LazyLock<HashMap<String, String>> = LazyLock::new(|| {
        let mut map = HashMap::new();
        for upstream in KIMI_MODELS.iter().map(|model| (*model).to_string()) {
            map.insert(gateway_discovery_model_id(&upstream), upstream.clone());
            map.insert(format!("claude-{upstream}"), upstream);
        }
        map
    });
    TARGETS.get(model).map(String::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_model_trims_hint() {
        assert_eq!(normalize_incoming_model("gpt-5.4-fast[1m]"), "gpt-5.4-fast");
        assert_eq!(normalize_incoming_model("gpt-5.4-fast"), "gpt-5.4-fast");
        assert_eq!(normalize_incoming_model("claude-gpt-5[1m]"), "gpt-5");
        assert_eq!(
            normalize_incoming_model("claude-gpt-5-6-sol"),
            "gpt-5.6-sol"
        );
        assert_eq!(
            normalize_incoming_model("claude-gpt-5.6-sol"),
            "gpt-5.6-sol"
        );
        assert_eq!(
            normalize_incoming_model("claude-gpt-5-3-codex-spark-fast"),
            "gpt-5.3-codex-spark-fast"
        );
    }

    #[test]
    fn every_codex_model_round_trips_through_discovery_ids() {
        for model in expand_codex_models() {
            let picker_id = gateway_discovery_model_id(&model);
            assert!(
                picker_id.starts_with("claude"),
                "{picker_id} would be filtered out of the /model picker"
            );
            assert_eq!(normalize_incoming_model(&picker_id), model);
        }
    }

    #[test]
    fn every_kimi_model_round_trips_through_discovery_ids() {
        for model in KIMI_MODELS {
            let picker_id = gateway_discovery_model_id(model);
            assert!(
                picker_id.starts_with("claude"),
                "{picker_id} would be filtered out of the /model picker"
            );
            assert_eq!(&normalize_incoming_model(&picker_id), model);
        }
    }

    #[test]
    fn kimi_picker_id_routes_to_kimi_not_codex_or_anthropic() {
        let registry = Registry::with_anthropic_passthrough(AliasProvider::Codex, true);
        for id in ["claude-kimi-for-coding", "claude-kimi-for-coding[1m]"] {
            let provider = registry.provider_for_model(id, None);
            assert_eq!(
                provider.expect("provider").name(),
                "kimi",
                "{id} must reach the Kimi handler"
            );
        }
    }

    #[test]
    fn kimi_appears_once_in_the_catalog() {
        let registry = Registry::with_anthropic_passthrough(AliasProvider::Codex, true);
        let kimi: Vec<_> = registry
            .catalog_models()
            .into_iter()
            .filter(|(_, provider)| provider == "kimi")
            .collect();
        assert_eq!(
            kimi,
            vec![("claude-kimi-for-coding".to_string(), "kimi".to_string())]
        );
    }

    #[test]
    fn alias_routes_to_configured_provider() {
        let registry = Registry::new(AliasProvider::Kimi);
        let p = registry.provider_for_model("haiku", None);
        assert!(p.is_some());
        assert_eq!(p.expect("provider").name(), "kimi");
    }

    #[test]
    fn opus_4_8_routes_to_configured_provider() {
        let registry = Registry::new(AliasProvider::Codex);
        let p = registry.provider_for_model("claude-opus-4-8", None);
        assert!(p.is_some());
        assert_eq!(p.expect("provider").name(), "codex");
    }

    #[test]
    fn claude_5_aliases_route_to_configured_provider() {
        let registry = Registry::new(AliasProvider::Codex);
        for model in ["claude-sonnet-5", "fable", "claude-fable-5"] {
            let p = registry.provider_for_model(model, None);
            assert!(p.is_some(), "{model} should route to a provider");
            assert_eq!(p.expect("provider").name(), "codex");
        }
    }

    #[test]
    fn claude_prefix_routes_to_anthropic_when_passthrough_enabled() {
        let registry = Registry::with_anthropic_passthrough(AliasProvider::Codex, true);
        let p = registry.provider_for_model("claude-opus-4-8", None);
        assert!(p.is_some());
        assert_eq!(p.expect("provider").name(), "anthropic");
    }

    #[test]
    fn claude_gpt_picker_models_route_to_codex() {
        let registry = Registry::with_anthropic_passthrough(AliasProvider::Codex, true);
        let p = registry.provider_for_model("claude-gpt-5-mini", None);
        assert!(p.is_some());
        assert_eq!(p.expect("provider").name(), "codex");
    }

    #[test]
    fn codex_catalog_uses_hyphenized_picker_ids() {
        let registry = Registry::with_anthropic_passthrough(AliasProvider::Codex, true);
        let catalog = registry.catalog_models();
        assert!(catalog.iter().any(|(id, _)| id == "claude-gpt-5-6-sol"));
        assert!(catalog.iter().any(|(id, _)| id == "claude-gpt-5-mini"));
        assert!(!catalog.iter().any(|(id, _)| id == "claude-gpt-5.6-sol"));
    }

    #[test]
    fn catalog_never_double_prefixes_anthropic_ids() {
        for passthrough in [false, true] {
            let registry = Registry::with_anthropic_passthrough(AliasProvider::Codex, passthrough);
            let catalog = registry.catalog_models();
            assert!(
                catalog
                    .iter()
                    .all(|(id, _)| !id.starts_with("claude-claude-")),
                "passthrough={passthrough}: {catalog:?}"
            );
            assert!(catalog.iter().any(|(id, _)| id == "claude-sonnet-5"));
        }
    }

    #[test]
    fn catalog_ids_are_unique() {
        let registry = Registry::with_anthropic_passthrough(AliasProvider::Codex, true);
        let catalog = registry.catalog_models();
        let mut ids: Vec<_> = catalog.iter().map(|(id, _)| id.clone()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), catalog.len());
    }

    #[test]
    fn cursor_prefix_routes() {
        let registry = Registry::new(AliasProvider::Codex);
        assert_eq!(
            registry
                .provider_for_model("cursor:gpt-5.5", None)
                .unwrap()
                .name(),
            "cursor"
        );
        assert_eq!(
            registry
                .provider_for_model("cursor-plan:gpt-5.5", None)
                .unwrap()
                .name(),
            "cursor"
        );
        assert_eq!(
            registry
                .provider_for_model("cursor-ask:gpt-5.5", None)
                .unwrap()
                .name(),
            "cursor"
        );
    }
}
