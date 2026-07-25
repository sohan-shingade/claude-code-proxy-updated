use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use claude_code_proxy::{
    monitor::{MonitorHandle, RequestStatus},
    registry::Registry,
    server::{
        MAX_REQUEST_BODY_BYTES, app, app_with_auth, app_with_monitor, bind_proxy_listener,
        validate_bind_auth,
    },
};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::util::ServiceExt;

fn body_string(json: &str) -> Body {
    Body::from(json.to_string())
}

#[tokio::test]
async fn bind_error_names_address_and_port() {
    let occupied = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = occupied.local_addr().unwrap().port();

    let err = bind_proxy_listener("127.0.0.1", port)
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains(&format!("127.0.0.1:{port}")));
    assert!(err.contains("failed to bind proxy listener"));
}

#[tokio::test]
async fn configurable_bind_address_accepts_all_interfaces() {
    let listener = bind_proxy_listener("0.0.0.0", 0).await.unwrap();
    assert_eq!(listener.local_addr().unwrap().ip().to_string(), "0.0.0.0");
}

#[test]
fn non_loopback_bind_requires_proxy_auth() {
    let err = validate_bind_auth("0.0.0.0", None).unwrap_err().to_string();
    assert!(err.contains("CCP_PROXY_AUTH_TOKEN"));
    validate_bind_auth("0.0.0.0", Some("secret")).unwrap();
    validate_bind_auth("127.0.0.1", None).unwrap();
    validate_bind_auth("::1", None).unwrap();
}

#[tokio::test]
async fn invalid_bind_address_is_actionable() {
    let err = bind_proxy_listener("not-an-ip", 18765)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("invalid proxy bind address"));
    assert!(err.contains("not-an-ip"));
}

#[tokio::test]
async fn healthz_returns_ok() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    assert_eq!(body, json!({"ok": true}));
}

#[tokio::test]
async fn proxy_auth_protects_v1_but_not_healthz() {
    let app = app_with_auth(
        Arc::new(Registry::with_default_alias()),
        Some("secret".to_string()),
    );

    let unauthorized = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", "bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

    let unknown_v1 = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/not-a-route")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unknown_v1.status(), StatusCode::UNAUTHORIZED);

    for (header, value) in [("authorization", "Bearer secret"), ("x-api-key", "secret")] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header(header, value)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    let health = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
}

#[tokio::test]
async fn oversized_request_body_returns_413() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .body(Body::from(vec![b'x'; MAX_REQUEST_BODY_BYTES + 1]))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn models_endpoint_returns_merged_catalog() {
    let app = app(Arc::new(Registry::with_anthropic_passthrough(
        claude_code_proxy::config::AliasProvider::Codex,
        true,
    )));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    assert_eq!(body["has_more"], false);
    let data = body["data"].as_array().unwrap();
    assert!(data.iter().any(|entry| entry["id"] == "claude-sonnet-5"));
    assert!(data.iter().any(|entry| entry["id"] == "claude-gpt-5"));
    assert!(data.iter().any(|entry| entry["id"] == "claude-gpt-5-mini"));
    assert!(data.iter().any(|entry| entry["id"] == "claude-gpt-5-codex"));
    assert!(data.iter().any(|entry| entry["id"] == "claude-gpt-5-6-sol"));
    assert!(
        data.iter()
            .any(|entry| entry["display_name"] == "GPT-5.6 Sol")
    );
    assert!(data.iter().all(|entry| entry["type"] == "model"));
    assert!(data.iter().all(|entry| entry["display_name"].is_string()));
    // Claude Code's /model picker drops ids without a claude/anthropic prefix.
    assert!(data.iter().all(|entry| {
        entry["id"]
            .as_str()
            .map(|id| id.starts_with("claude") || id.starts_with("anthropic"))
            .unwrap_or(false)
    }));
    assert_eq!(body["first_id"], data.first().unwrap()["id"]);
    assert_eq!(body["last_id"], data.last().unwrap()["id"]);
}

#[tokio::test]
async fn invalid_json_request_is_json_error() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .body(body_string("{"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    let error_type = value["error"]["type"].as_str().unwrap_or("");
    assert_eq!(error_type, "invalid_request_error");
}

#[tokio::test]
async fn empty_body_is_invalid_json() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn unknown_model_returns_400_with_summary() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"messages":[{"role":"user","content":"hello"}],"model":"not-a-model"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    let message = body["error"]["message"].as_str().unwrap_or("");
    assert!(message.contains("Unknown model \"not-a-model\""));
    assert!(message.contains("Supported:"));
}

#[tokio::test]
async fn missing_model_returns_400() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    let error_type = body["error"]["type"].as_str().unwrap_or("");
    assert_eq!(error_type, "invalid_request_error");
}

#[tokio::test]
async fn known_model_reaches_codex_provider() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // Codex provider is now concrete, so it should attempt auth before returning 501
    let status = response.status();
    assert!(
        status != StatusCode::NOT_IMPLEMENTED,
        "codex should no longer be a placeholder provider"
    );
}

#[tokio::test]
async fn count_tokens_routes_to_provider() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // Codex provider is now concrete, so count_tokens should succeed
    let status = response.status();
    assert!(
        status != StatusCode::NOT_IMPLEMENTED,
        "count_tokens should no longer return 501 for codex models"
    );
}

#[tokio::test]
async fn context_window_hint_is_removed_before_provider_dispatch() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"model":"gpt-5.6-luna[1m]","messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn unknown_routes_use_anthropic_not_found_error() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/nope")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    assert_eq!(body["type"].as_str().unwrap_or(""), "error");
}

#[tokio::test]
async fn monitor_records_successful_request_events() {
    let monitor = MonitorHandle::new(10);
    let app = app_with_monitor(
        Arc::new(Registry::with_default_alias()),
        Some(monitor.clone()),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .header("x-claude-code-session-id", "project-session")
                .body(body_string(
                    r##"{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.177.45c"},{"type":"text","text":"You are a Claude agent, built on Anthropic's Claude Agent SDK.","cache_control":{"type":"ephemeral"}},{"type":"text","text":"\nYou are an interactive agent.\n\n# Environment\nYou have been invoked in the following environment: \n - Primary working directory: /projects/example\n - Is a git repository: true","cache_control":{"type":"ephemeral"}}],"output_config":{"effort":"high"}}"##,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let state = monitor.snapshot();
    assert_eq!(state.active.len(), 1);
    assert!(state.recent.is_empty());

    let _body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let state = monitor.snapshot();
    assert!(state.active.is_empty());
    assert_eq!(state.recent.len(), 1);
    assert_eq!(state.recent[0].status, RequestStatus::Completed);
    assert_eq!(state.recent[0].http_status, Some(200));
    assert_eq!(
        state.recent[0].session_id.as_deref(),
        Some("project-session")
    );
    assert!(state.recent[0].session_seq.is_some());
    assert_eq!(state.recent[0].project.as_deref(), Some("example"));
    assert_eq!(state.sessions[0].project.as_deref(), Some("example"));
    assert_eq!(state.recent[0].provider.as_deref(), Some("codex"));
    assert_eq!(state.recent[0].model.as_deref(), Some("gpt-5.4"));
    assert_eq!(state.recent[0].effort.as_deref(), Some("high"));
    assert!(state.recent[0].input_tokens.is_some());
}

#[tokio::test]
async fn monitor_records_invalid_json_failure() {
    let monitor = MonitorHandle::new(10);
    let app = app_with_monitor(
        Arc::new(Registry::with_default_alias()),
        Some(monitor.clone()),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .body(body_string("{"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let state = monitor.snapshot();
    assert!(state.active.is_empty());
    assert_eq!(state.recent[0].status, RequestStatus::Failed);
    assert_eq!(state.recent[0].http_status, Some(400));
    let error = state.recent[0].error.as_deref().unwrap_or("");
    assert!(error.starts_with("Invalid JSON:"));
}

#[tokio::test]
async fn monitor_records_unknown_model_failure() {
    let monitor = MonitorHandle::new(10);
    let app = app_with_monitor(
        Arc::new(Registry::with_default_alias()),
        Some(monitor.clone()),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"messages":[{"role":"user","content":"hello"}],"model":"not-a-model"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let state = monitor.snapshot();
    assert!(state.active.is_empty());
    assert_eq!(state.recent[0].status, RequestStatus::Failed);
    assert_eq!(state.recent[0].http_status, Some(400));
    let error = state.recent[0].error.as_deref().unwrap_or("");
    assert!(error.starts_with("Unknown model \"not-a-model\""));
    assert!(error.contains("Supported:"));
}
