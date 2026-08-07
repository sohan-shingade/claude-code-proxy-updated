use crate::{
    anthropic::json_error,
    logging::{Logger, REDACT_KEYS, create_logger},
    monitor::{EndpointKind, MonitorHandle},
    project,
    provider::RequestContext,
    providers::codex::translate::model_allowlist::resolve_model_request,
    registry::{Registry, normalize_incoming_model},
    session::{self, SessionState},
    traffic::{TrafficCaptureOptions, create_traffic_capture},
};
use axum::{
    Json, Router,
    body::Body,
    extract::{Request as AxumRequest, State},
    http::{Request, StatusCode},
    middleware::{Next, from_fn_with_state},
    response::Response,
    routing::{get, post},
};
use http_body_util::{BodyExt, StreamBody};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use std::fs::{self, File};
use std::future::Future;
use std::io::Write;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use uuid::Uuid;

pub const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;

pub struct ServerConfig {
    pub bind_address: String,
    pub port: u16,
    pub monitor: Option<MonitorHandle>,
}

pub async fn serve(config: ServerConfig) -> anyhow::Result<()> {
    serve_inner(config, std::future::pending::<()>()).await
}

pub async fn serve_with_shutdown(
    config: ServerConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    serve_inner(config, shutdown).await
}

async fn serve_inner(
    config: ServerConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let auth_token = crate::config::proxy_auth_token();
    validate_bind_auth(&config.bind_address, auth_token.as_deref())?;
    let listener = bind_proxy_listener(&config.bind_address, config.port).await?;
    serve_listener_with_options(
        listener,
        config.monitor,
        auth_token,
        Some(config.bind_address),
        shutdown,
    )
    .await
}

pub fn validate_bind_auth(bind_address: &str, auth_token: Option<&str>) -> anyhow::Result<()> {
    let ip = bind_address
        .parse::<IpAddr>()
        .map_err(|err| anyhow::anyhow!("invalid proxy bind address {bind_address:?}: {err}"))?;
    if !ip.is_loopback() && auth_token.is_none() {
        anyhow::bail!(
            "refusing non-loopback bind address {bind_address:?} without CCP_PROXY_AUTH_TOKEN"
        );
    }
    Ok(())
}

pub async fn bind_proxy_listener(bind_address: &str, port: u16) -> anyhow::Result<TcpListener> {
    let ip = bind_address
        .parse::<std::net::IpAddr>()
        .map_err(|err| anyhow::anyhow!("invalid proxy bind address {bind_address:?}: {err}"))?;
    let addr = std::net::SocketAddr::new(ip, port);
    TcpListener::bind(addr)
        .await
        .map_err(|err| anyhow::anyhow!("failed to bind proxy listener on {addr}: {err}"))
}

pub async fn serve_listener(
    listener: TcpListener,
    monitor: Option<MonitorHandle>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let bind_address = listener.local_addr()?.ip().to_string();
    serve_listener_with_options(
        listener,
        monitor,
        crate::config::proxy_auth_token(),
        Some(bind_address),
        shutdown,
    )
    .await
}

async fn serve_listener_with_options(
    listener: TcpListener,
    monitor: Option<MonitorHandle>,
    auth_token: Option<String>,
    picker_bind_address: Option<String>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let local_addr = listener.local_addr()?;
    validate_bind_auth(&local_addr.ip().to_string(), auth_token.as_deref())?;
    let port = local_addr.port();
    create_logger("server").info(
        "server listening",
        Some(serde_json::Map::from_iter([
            ("port".to_string(), json!(port)),
            (
                "bindAddress".to_string(),
                json!(local_addr.ip().to_string()),
            ),
            (
                "logDir".to_string(),
                json!(
                    crate::paths::log_file()
                        .parent()
                        .map(|path| path.display().to_string())
                ),
            ),
        ])),
    );
    let app = app_with_options(
        Arc::new(Registry::with_default_alias()),
        monitor,
        auth_token,
    );
    let picker_bind_address = picker_bind_address.unwrap_or_else(|| local_addr.ip().to_string());
    seed_picker_caches_on_start(&picker_bind_address, port);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

/// Seed Claude Code's `/model` picker cache(s) whenever the proxy starts, so
/// a stale or clobbered `gateway-models.json` self-heals without a manual
/// `seed-picker` run. The proxy is necessarily running before Claude Code
/// launches, which makes startup the one reliable hook point.
fn seed_picker_caches_on_start(bind_address: &str, port: u16) {
    if !crate::config::picker_seed_on_start() {
        return;
    }
    let log = create_logger("picker");
    let registry = Registry::with_default_alias();
    let base_url = crate::config::picker_base_url(bind_address, port);
    for outcome in crate::picker::seed_claude_picker_caches(&registry, &base_url) {
        let mut fields = serde_json::Map::from_iter([
            (
                "configDir".to_string(),
                json!(outcome.config_dir.display().to_string()),
            ),
            (
                "cacheFile".to_string(),
                json!(outcome.cache_file.display().to_string()),
            ),
            ("baseUrl".to_string(), json!(&base_url)),
        ]);
        match outcome.result {
            Ok(crate::picker::SeedResult::Written { models }) => {
                fields.insert("models".to_string(), json!(models));
                log.info("picker_cache_seeded", Some(fields));
            }
            Ok(crate::picker::SeedResult::Unchanged { models }) => {
                fields.insert("models".to_string(), json!(models));
                log.info("picker_cache_unchanged", Some(fields));
            }
            Ok(crate::picker::SeedResult::SkippedMissingDir) => {
                log.info("picker_cache_skipped_missing_dir", Some(fields));
            }
            Err(err) => {
                fields.insert("error".to_string(), json!(err.to_string()));
                log.warn("picker_cache_seed_failed", Some(fields));
            }
        }
    }
}

pub fn app(registry: Arc<Registry>) -> Router {
    app_with_options(registry, None, None)
}

pub fn app_with_auth(registry: Arc<Registry>, auth_token: Option<String>) -> Router {
    app_with_options(registry, None, auth_token)
}

pub fn app_with_monitor(registry: Arc<Registry>, monitor: Option<MonitorHandle>) -> Router {
    app_with_options(registry, monitor, None)
}

fn app_with_options(
    registry: Arc<Registry>,
    monitor: Option<MonitorHandle>,
    auth_token: Option<String>,
) -> Router {
    let state = Arc::new(AppState { registry, monitor });
    let v1 = Router::new()
        .route("/models", get(handler_models))
        .route("/messages", post(handler_messages))
        .route("/messages/count_tokens", post(handler_count_tokens));
    Router::new()
        .route("/healthz", get(healthz))
        .nest("/v1", v1)
        .fallback(fallback_handler)
        .layer(from_fn_with_state(auth_token, require_proxy_auth))
        .with_state(state)
}

#[derive(Clone)]
struct AppState {
    registry: Arc<Registry>,
    monitor: Option<MonitorHandle>,
}

async fn healthz() -> Json<serde_json::Value> {
    Json(json!({ "ok": true }))
}

async fn require_proxy_auth(
    State(expected): State<Option<String>>,
    req: AxumRequest,
    next: Next,
) -> Response {
    let Some(expected) = expected.as_deref() else {
        return next.run(req).await;
    };
    if !req.uri().path().starts_with("/v1/") {
        return next.run(req).await;
    }
    let mut req = req;
    let bearer_matches = req
        .headers()
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        == Some(expected);
    let api_key_matches = req
        .headers()
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        == Some(expected);
    if bearer_matches {
        req.headers_mut().remove("authorization");
        return next.run(req).await;
    }
    if api_key_matches {
        req.headers_mut().remove("x-api-key");
        return next.run(req).await;
    }
    json_error(
        StatusCode::UNAUTHORIZED,
        "authentication_error",
        "Missing or invalid proxy authentication token",
    )
}

async fn handler_messages(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    dispatch_request(state, req, false).await
}

async fn handler_models(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
) -> Json<serde_json::Value> {
    let log = create_logger("server");
    let auth_mode = if req.headers().contains_key("authorization") {
        "authorization"
    } else if req.headers().contains_key("x-api-key") {
        "x-api-key"
    } else {
        "none"
    };
    log.info(
        "model_discovery_request",
        Some(serde_json::Map::from_iter([
            ("path".to_string(), json!(req.uri().path())),
            ("query".to_string(), json!(redacted_query(req.uri()))),
            ("authMode".to_string(), json!(auth_mode)),
        ])),
    );
    let models = crate::registry::apply_pinned_order(
        state.registry.catalog_models(),
        &crate::config::picker_pinned(),
    )
    .into_iter()
    .map(|(id, provider)| {
        json!({
            "type": "model",
            "id": id,
            "display_name": provider_display_name(&provider, &id),
            "created_at": "2026-01-01T00:00:00Z",
        })
    })
    .collect::<Vec<_>>();
    let first_id = models
        .first()
        .and_then(|model| model["id"].as_str())
        .map(str::to_string);
    let last_id = models
        .last()
        .and_then(|model| model["id"].as_str())
        .map(str::to_string);
    Json(json!({
        "data": models,
        "has_more": false,
        "first_id": first_id,
        "last_id": last_id,
    }))
}

async fn handler_count_tokens(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    dispatch_request(state, req, true).await
}

async fn dispatch_request(
    state: Arc<AppState>,
    req: Request<Body>,
    count_tokens: bool,
) -> Response {
    let started_at = Instant::now();
    let log = create_logger("server");
    let req_id = Uuid::new_v4().to_string();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();
    let path = uri.path().to_string();
    let query = redacted_query(&uri);
    let endpoint = if count_tokens {
        EndpointKind::CountTokens
    } else {
        EndpointKind::Messages
    };
    log.info(
        "request",
        Some(serde_json::Map::from_iter([
            ("reqId".to_string(), json!(&req_id)),
            ("method".to_string(), json!(method.as_str())),
            ("path".to_string(), json!(&path)),
            ("query".to_string(), json!(&query)),
        ])),
    );
    let session_id = req
        .headers()
        .get("x-claude-code-session-id")
        .and_then(|value| value.to_str().ok())
        .map(std::string::ToString::to_string);
    if let Some(monitor) = state.monitor.as_ref() {
        monitor.request_started(&req_id, session_id.clone(), None, endpoint);
    }
    let request_guard = RequestMonitorGuard::new(state.monitor.clone(), req_id.clone());
    let now = current_millis();
    let body_bytes = match axum::body::to_bytes(req.into_body(), MAX_REQUEST_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(err) => {
            let too_large = std::error::Error::source(&err)
                .is_some_and(|source| source.is::<http_body_util::LengthLimitError>());
            let (status, message) = if too_large {
                (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    format!("Request body exceeds the {MAX_REQUEST_BODY_BYTES}-byte limit"),
                )
            } else {
                (
                    StatusCode::BAD_REQUEST,
                    format!("Invalid request body: {err}"),
                )
            };
            let response = json_error(status, "invalid_request_error", message);
            log_request_completed(
                &log,
                RequestLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    status: response.status(),
                    started_at,
                },
            );
            let (response, details) = record_failed_response(
                &log,
                FailedResponseLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    started_at,
                },
                response,
            )
            .await;
            monitor_failed(
                state.monitor.as_ref(),
                &req_id,
                Some(response.status()),
                details
                    .as_ref()
                    .map(|details| details.message.as_str())
                    .unwrap_or("Invalid JSON"),
            );
            return response;
        }
    };

    let mut body: crate::anthropic::schema::MessagesRequest = match parse_json_body(&body_bytes) {
        Ok(body) => body,
        Err(response) => {
            let status = response.status();
            log_request_completed(
                &log,
                RequestLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    status: response.status(),
                    started_at,
                },
            );
            let (response, details) = record_failed_response(
                &log,
                FailedResponseLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    started_at,
                },
                *response,
            )
            .await;
            monitor_failed(
                state.monitor.as_ref(),
                &req_id,
                Some(status),
                details
                    .as_ref()
                    .map(|details| details.message.as_str())
                    .unwrap_or("Invalid JSON"),
            );
            return response;
        }
    };

    if let Some(project) = project::name_from_request(
        body.extra.get("system"),
        body.messages.iter().rev().map(|message| &message.content),
    ) && let Some(monitor) = state.monitor.as_ref()
    {
        monitor.project_resolved(&req_id, project);
    }

    let model = match body.model.as_deref() {
        Some(model) => model,
        None => {
            let response = json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!(
                    "Missing \"model\" in request body. {}",
                    state.registry.unknown_model_message()
                ),
            );
            log_request_completed(
                &log,
                RequestLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    status: response.status(),
                    started_at,
                },
            );
            let (response, details) = record_failed_response(
                &log,
                FailedResponseLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    started_at,
                },
                response,
            )
            .await;
            monitor_failed(
                state.monitor.as_ref(),
                &req_id,
                Some(response.status()),
                details
                    .as_ref()
                    .map(|details| details.message.as_str())
                    .unwrap_or("Missing model"),
            );
            return response;
        }
    };

    let normalized_model = normalize_incoming_model(model);
    body.model = Some(normalized_model.clone());
    let session_state = if let Some(session_id) = session_id.as_deref() {
        session::existing_session(Some(session_id), now)
    } else {
        None
    };

    let provider = state.registry.provider_for_model(
        &normalized_model,
        session_state
            .as_ref()
            .and_then(|state| state.affinity_provider.as_ref()),
    );

    let provider = match provider {
        Some(provider) => provider,
        None => {
            log.warn(
                "unknown model",
                Some(serde_json::Map::from_iter([
                    ("reqId".to_string(), json!(&req_id)),
                    ("model".to_string(), json!(&normalized_model)),
                ])),
            );
            let response = json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!(
                    "Unknown model \"{normalized_model}\". {}",
                    state.registry.unknown_model_message()
                ),
            );
            log_request_completed(
                &log,
                RequestLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: Some(&normalized_model),
                    count_tokens,
                    status: response.status(),
                    started_at,
                },
            );
            let (response, details) = record_failed_response(
                &log,
                FailedResponseLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: Some(&normalized_model),
                    count_tokens,
                    started_at,
                },
                response,
            )
            .await;
            monitor_failed(
                state.monitor.as_ref(),
                &req_id,
                Some(response.status()),
                details
                    .as_ref()
                    .map(|details| details.message.as_str())
                    .unwrap_or("Unknown model"),
            );
            return response;
        }
    };

    let effort = crate::providers::translate_shared::read_effort(&body)
        .ok()
        .flatten()
        .map(str::to_string);
    let current = session::record_session_request(
        session_id.as_deref(),
        session_state.as_ref(),
        provider.name(),
        &normalized_model,
        now,
    );
    if let Some(monitor) = state.monitor.as_ref() {
        if let Some(current) = current.as_ref() {
            monitor.session_sequence_resolved(&req_id, current.seq);
        }
        monitor.provider_selected(&req_id, provider.name(), &normalized_model, effort);
    }

    let traffic = create_traffic_capture(TrafficCaptureOptions {
        req_id: req_id.clone(),
        session_id: session_id.clone(),
        session_seq: current.as_ref().map(|s| s.seq),
        provider: Some(provider.name().to_string()),
        state_dir_override: None,
    })
    .map(Arc::new);

    if let Some(capture) = traffic.as_ref() {
        if let Some(monitor) = state.monitor.as_ref() {
            monitor.traffic_capture_path(&req_id, capture.root().to_path_buf());
        }
        capture.write_json(
            "000-metadata",
            &json!({
                "reqId": &req_id,
                "sessionId": &session_id,
                "sessionSeq": current.as_ref().map(|s| s.seq),
                "kind": if count_tokens { "count_tokens" } else { "messages" },
                "provider": provider.name(),
                "model": &normalized_model,
                "method": method.as_str(),
                "path": &path,
                "query": &query,
                "headers": headers_to_record(&headers),
            }),
        );
        capture.write_json(
            "010-anthropic-request",
            &serde_json::to_value(&body).unwrap_or_else(|_| json!({})),
        );
    }

    let context = RequestContext {
        req_id: req_id.clone(),
        session_id,
        session_seq: current.map(|s| s.seq),
        provider: provider.name().to_string(),
        request_headers: headers,
        traffic,
        monitor: state.monitor.clone(),
    };

    let response = if count_tokens {
        provider.handle_count_tokens(body, context).await
    } else {
        provider.handle_messages(body, context).await
    };
    log_request_completed(
        &log,
        RequestLogContext {
            req_id: &req_id,
            provider: Some(provider.name()),
            model: Some(&normalized_model),
            count_tokens,
            status: response.status(),
            started_at,
        },
    );
    let status = response.status();
    if status.is_success() {
        return monitor_response_body(response, request_guard);
    }

    let (response, details) = record_failed_response(
        &log,
        FailedResponseLogContext {
            req_id: &req_id,
            provider: Some(provider.name()),
            model: Some(&normalized_model),
            count_tokens,
            started_at,
        },
        response,
    )
    .await;
    if let Some(details) = details.as_ref() {
        monitor_failed(
            state.monitor.as_ref(),
            &req_id,
            Some(status),
            details.message.as_str(),
        );
    } else {
        monitor_failed(
            state.monitor.as_ref(),
            &req_id,
            Some(status),
            format!("HTTP {}", status.as_u16()),
        );
    }
    response
}

fn provider_display_name(provider: &str, id: &str) -> String {
    match provider {
        "anthropic" => id.to_string(),
        "codex" => display_codex_name(id),
        "kimi" => display_kimi_name(id),
        "grok" => format!("{id} (Grok)"),
        "cursor" => format!("{id} (Cursor)"),
        _ => id.to_string(),
    }
}

/// Human-readable picker label for a Kimi model. The upstream ids say nothing
/// about which generation they are (`kimi-for-coding` is K2.7), so the labels
/// come from the `display_name` the Kimi models endpoint reports, plus the
/// context window, which is the main reason to pick one over another.
fn display_kimi_name(id: &str) -> String {
    match normalize_incoming_model(id).as_str() {
        "kimi-for-coding" => "Kimi K2.7 Coding (256K)".to_string(),
        "kimi-for-coding-highspeed" => "Kimi K2.7 Coding Highspeed (256K)".to_string(),
        "k3" => "Kimi K3 (1M)".to_string(),
        "k3-256k" => "Kimi K3 (256K)".to_string(),
        other => format!("{other} (Kimi)"),
    }
}

/// Body for Claude Code's `<config-dir>/cache/gateway-models.json` picker
/// cache. Claude Code normally writes this file itself after fetching
/// `/v1/models` at startup, but that fetch only runs when
/// `ANTHROPIC_AUTH_TOKEN` (or an API key) is set — which switches Claude Code
/// off claude.ai login and disables its connectors. Seeding the cache
/// directly gives the /model picker the same catalog without the token.
/// `base_url` must exactly match the `ANTHROPIC_BASE_URL` Claude Code runs
/// with, and `fetched_at_ms` is a Unix timestamp in milliseconds.
pub fn picker_cache_json(registry: &Registry, base_url: &str, fetched_at_ms: u64) -> Value {
    let models = crate::registry::apply_pinned_order(
        registry.catalog_models(),
        &crate::config::picker_pinned(),
    )
    .into_iter()
    .map(|(id, provider)| {
        json!({
            "id": id,
            "display_name": provider_display_name(&provider, &id),
        })
    })
    .collect::<Vec<_>>();
    json!({
        "baseUrl": base_url,
        "fetchedAt": fetched_at_ms,
        "models": models,
    })
}

/// Human-readable picker label derived from the upstream id, e.g.
/// `gpt-5.3-codex-spark-fast` -> `GPT-5.3 Codex Spark Fast`. Rolling aliases
/// additionally show what they currently resolve to.
fn display_codex_name(id: &str) -> String {
    let model = normalize_incoming_model(id);
    let Some(rest) = model.strip_prefix("gpt-") else {
        return model;
    };
    let mut name = String::from("GPT-");
    for (index, word) in rest.split('-').enumerate() {
        if index == 0 {
            name.push_str(word);
            continue;
        }
        name.push(' ');
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            name.extend(first.to_uppercase());
            name.push_str(chars.as_str());
        }
    }
    if crate::registry::CODEX_MODEL_ALIASES.contains(&model.as_str()) {
        let resolved = resolve_model_request(&model).model;
        name.push_str(&format!(" (Codex {resolved})"));
    }
    name
}

fn monitor_response_body(response: Response, guard: RequestMonitorGuard) -> Response {
    let status = response.status();
    let (parts, body) = response.into_parts();
    let stream =
        futures_util::stream::unfold((body, guard), move |(mut body, mut guard)| async move {
            match body.frame().await {
                Some(Ok(frame)) => Some((Ok(frame), (body, guard))),
                Some(Err(err)) => {
                    guard.failed(status, err.to_string());
                    Some((Err(err), (body, guard)))
                }
                None => {
                    guard.completed(status);
                    None
                }
            }
        });
    Response::from_parts(parts, Body::new(StreamBody::new(stream)))
}

struct RequestLogContext<'a> {
    req_id: &'a str,
    provider: Option<&'a str>,
    model: Option<&'a str>,
    count_tokens: bool,
    status: StatusCode,
    started_at: Instant,
}

fn log_request_completed(log: &Logger, ctx: RequestLogContext<'_>) {
    log.info(
        "request_completed",
        Some(serde_json::Map::from_iter([
            ("reqId".to_string(), json!(ctx.req_id)),
            ("provider".to_string(), json!(ctx.provider)),
            ("model".to_string(), json!(ctx.model)),
            ("countTokens".to_string(), json!(ctx.count_tokens)),
            ("status".to_string(), json!(ctx.status.as_u16())),
            (
                "ms".to_string(),
                json!(ctx.started_at.elapsed().as_millis()),
            ),
        ])),
    );
}

struct FailedResponseLogContext<'a> {
    req_id: &'a str,
    provider: Option<&'a str>,
    model: Option<&'a str>,
    count_tokens: bool,
    started_at: Instant,
}

struct FailedResponseDetails {
    message: String,
}

async fn record_failed_response(
    log: &Logger,
    ctx: FailedResponseLogContext<'_>,
    response: Response,
) -> (Response, Option<FailedResponseDetails>) {
    if response.status().is_success() {
        return (response, None);
    }

    let status = response.status();
    let (parts, body) = response.into_parts();
    let bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            log.info(
                "request_failed",
                Some(serde_json::Map::from_iter([
                    ("reqId".to_string(), json!(ctx.req_id)),
                    ("provider".to_string(), json!(ctx.provider)),
                    ("model".to_string(), json!(ctx.model)),
                    ("countTokens".to_string(), json!(ctx.count_tokens)),
                    ("status".to_string(), json!(status.as_u16())),
                    (
                        "ms".to_string(),
                        json!(ctx.started_at.elapsed().as_millis()),
                    ),
                    ("bodyReadError".to_string(), json!(err.to_string())),
                ])),
            );
            return (Response::from_parts(parts, Body::empty()), None);
        }
    };

    let response_body = response_body_value(&bytes);
    let message = error_message_from_response(&response_body)
        .unwrap_or_else(|| format!("HTTP {}", status.as_u16()));
    let document = json!({
        "reqId": ctx.req_id,
        "provider": ctx.provider,
        "model": ctx.model,
        "countTokens": ctx.count_tokens,
        "status": status.as_u16(),
        "elapsedMs": ctx.started_at.elapsed().as_millis(),
        "message": message,
        "response": response_body,
    });
    let error_file = write_error_capture(ctx.req_id, &redact_error_value(document));

    let mut fields = serde_json::Map::from_iter([
        ("reqId".to_string(), json!(ctx.req_id)),
        ("provider".to_string(), json!(ctx.provider)),
        ("model".to_string(), json!(ctx.model)),
        ("countTokens".to_string(), json!(ctx.count_tokens)),
        ("status".to_string(), json!(status.as_u16())),
        (
            "ms".to_string(),
            json!(ctx.started_at.elapsed().as_millis()),
        ),
        ("message".to_string(), json!(message)),
    ]);
    if let Some(path) = error_file.as_ref() {
        fields.insert("errorFile".to_string(), json!(path.display().to_string()));
    }
    log.info("request_failed", Some(fields));

    (
        Response::from_parts(parts, Body::from(bytes)),
        Some(FailedResponseDetails { message }),
    )
}

fn response_body_value(bytes: &[u8]) -> Value {
    match serde_json::from_slice::<Value>(bytes) {
        Ok(value) => json!({ "json": value }),
        Err(_) => json!({ "text": String::from_utf8_lossy(bytes) }),
    }
}

fn error_message_from_response(response_body: &Value) -> Option<String> {
    response_body
        .get("json")
        .and_then(|body| body.get("error"))
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| {
            response_body
                .get("text")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
        })
        .map(std::string::ToString::to_string)
}

fn write_error_capture(req_id: &str, document: &Value) -> Option<PathBuf> {
    let dir = crate::paths::state_dir().join("errors");
    fs::create_dir_all(&dir).ok()?;
    set_mode(&dir, 0o700);
    let path = dir.join(format!(
        "{}-{}.json",
        current_millis(),
        sanitize_path_part(req_id)
    ));
    let mut file = File::create(&path).ok()?;
    set_mode(&path, 0o600);
    let payload = serde_json::to_vec_pretty(document).ok()?;
    file.write_all(&payload).ok()?;
    file.write_all(b"\n").ok()?;
    Some(path)
}

fn sanitize_path_part(raw: &str) -> String {
    let sanitized: String = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized
    }
}

fn redact_error_value(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(redact_error_value).collect()),
        Value::Object(fields) => {
            let mut out = Map::new();
            for (key, value) in fields {
                if REDACT_KEYS.contains(&key.to_lowercase().as_str()) {
                    out.insert(key, redact_error_key(value));
                } else {
                    out.insert(key, redact_error_value(value));
                }
            }
            Value::Object(out)
        }
        value => value,
    }
}

fn redact_error_key(value: Value) -> Value {
    match value {
        Value::String(value) => Value::String(format!("[redacted len={}]", value.len())),
        _ => Value::String("[redacted]".to_string()),
    }
}

struct RequestMonitorGuard {
    monitor: Option<MonitorHandle>,
    req_id: String,
}

impl RequestMonitorGuard {
    fn new(monitor: Option<MonitorHandle>, req_id: String) -> Self {
        Self { monitor, req_id }
    }

    fn completed(&mut self, status: StatusCode) {
        if let Some(monitor) = self.monitor.take() {
            monitor.request_completed(&self.req_id, status.as_u16(), None, None);
        }
    }

    fn failed(&mut self, status: StatusCode, error: String) {
        if let Some(monitor) = self.monitor.take() {
            monitor.request_failed(&self.req_id, Some(status.as_u16()), error);
        }
    }
}

impl Drop for RequestMonitorGuard {
    fn drop(&mut self) {
        if let Some(monitor) = self.monitor.as_ref() {
            monitor.request_abandoned(&self.req_id, "Request future ended before completion");
        }
    }
}

fn monitor_failed(
    monitor: Option<&MonitorHandle>,
    req_id: &str,
    status: Option<StatusCode>,
    error: impl Into<String>,
) {
    if let Some(monitor) = monitor {
        monitor.request_failed(req_id, status.map(|status| status.as_u16()), error);
    }
}

fn headers_to_record(headers: &http::HeaderMap) -> Value {
    let mut out = Map::new();
    for (key, value) in headers {
        if let Ok(raw) = value.to_str() {
            out.insert(key.as_str().to_string(), Value::String(raw.to_string()));
        }
    }
    Value::Object(out)
}

fn redacted_query(uri: &http::Uri) -> Value {
    let mut out = Map::new();
    let Some(query) = uri.query() else {
        return Value::Object(out);
    };
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        let key = key.into_owned();
        let lower = key.to_lowercase();
        let value = if REDACT_KEYS.contains(&lower.as_str()) {
            Value::String(format!("[redacted len={}]", value.len()))
        } else {
            Value::String(value.into_owned())
        };
        out.insert(key, value);
    }
    Value::Object(out)
}

fn parse_json_body<T>(body: &[u8]) -> Result<T, Box<Response>>
where
    T: DeserializeOwned,
{
    if body.is_empty() {
        return Err(Box::new(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Invalid JSON: empty body",
        )));
    }

    serde_json::from_slice::<T>(body).map_err(|err| {
        Box::new(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("Invalid JSON: {err}"),
        ))
    })
}

async fn fallback_handler(method: axum::http::Method, uri: axum::http::Uri) -> Response {
    json_error(
        StatusCode::NOT_FOUND,
        "not_found",
        format!("No route for {method} {}", uri.path()),
    )
}

fn current_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn set_mode(path: &Path, mode: u32) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = fs::metadata(path) {
            let mut perm = meta.permissions();
            perm.set_mode(mode);
            let _ = fs::set_permissions(path, perm);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (path, mode);
    }
}

#[allow(dead_code)]
fn _unused(session_state: Option<&SessionState>) {
    let _ = session_state;
}
