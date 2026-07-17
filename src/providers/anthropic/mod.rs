use anyhow::Result;
use async_trait::async_trait;
use axum::body::Body;
use axum::response::Response;
use futures_util::TryStreamExt;
use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use serde::Serialize;
use serde_json::json;
use std::path::PathBuf;

use crate::anthropic::error::json_error;
use crate::anthropic::schema::MessagesRequest;
use crate::config;
use crate::provider::{CliHandlers, Provider, RequestContext};
use crate::registry;

const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicProvider {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .http2_adaptive_window(true)
            .build()
            .expect("anthropic client");
        Self {
            client,
            base_url: config::anthropic_base_url(),
        }
    }

    async fn forward_request<T: Serialize>(
        &self,
        path: &str,
        body: &T,
        ctx: &RequestContext,
        want_stream: bool,
    ) -> Response {
        let headers = match build_upstream_headers(&ctx.request_headers, want_stream) {
            Ok(headers) => headers,
            Err(response) => return response,
        };

        if let Some(monitor) = ctx.monitor.as_ref() {
            if let Some(model) = request_model(body) {
                monitor.model_resolved(&ctx.req_id, model);
            }
            monitor.upstream_started(&ctx.req_id);
        }
        if let Some(capture) = ctx.traffic.as_ref() {
            capture.write_json(
                "020-anthropic-upstream-request",
                &json!({
                    "url": format!("{}{}", self.base_url, path),
                    "stream": want_stream,
                }),
            );
        }

        let upstream = match self
            .client
            .post(format!("{}{}", self.base_url, path))
            .headers(headers)
            .json(body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(err) => {
                return json_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    format!("Anthropic passthrough request failed: {err}"),
                );
            }
        };

        if want_stream {
            response_from_upstream_stream(upstream)
        } else {
            response_from_upstream_buffered(upstream).await
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn supported_models(&self) -> Vec<String> {
        registry::ANTHROPIC_MODELS
            .iter()
            .map(|model| (*model).to_string())
            .collect()
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &ANTHROPIC_CLI
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        self.forward_request("/v1/messages", &body, &ctx, body.stream)
            .await
    }

    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        self.forward_request("/v1/messages/count_tokens", &body, &ctx, false)
            .await
    }
}

#[derive(Clone, Copy)]
struct AnthropicCli;

impl CliHandlers for AnthropicCli {
    fn login(&self) -> Result<()> {
        anyhow::bail!("anthropic passthrough uses existing Anthropic credentials")
    }

    fn device(&self) -> Result<()> {
        anyhow::bail!("anthropic passthrough uses existing Anthropic credentials")
    }

    fn status(&self) -> Result<()> {
        if config::anthropic_oauth_token().is_some()
            || config::anthropic_auth_token().is_some()
            || config::anthropic_api_key().is_some()
        {
            Ok(())
        } else {
            anyhow::bail!("No proxy-side Anthropic credential is configured")
        }
    }

    fn logout(&self) -> Result<()> {
        Ok(())
    }
}

const ANTHROPIC_CLI: AnthropicCli = AnthropicCli;

fn build_upstream_headers(
    incoming: &HeaderMap,
    want_stream: bool,
) -> std::result::Result<HeaderMap, Response> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        ACCEPT,
        HeaderValue::from_static(if want_stream {
            "text/event-stream"
        } else {
            "application/json"
        }),
    );

    copy_header(
        incoming,
        &mut headers,
        HeaderName::from_static("anthropic-version"),
        Some(DEFAULT_ANTHROPIC_VERSION),
    );
    copy_header(
        incoming,
        &mut headers,
        HeaderName::from_static("anthropic-beta"),
        None,
    );

    if config::anthropic_forward_auth_headers()
        && copy_incoming_auth_headers(incoming, &mut headers)
    {
        return Ok(headers);
    }

    if let Some(token) = config::anthropic_oauth_token()
        .or_else(config::anthropic_auth_token)
        .or_else(read_claude_credentials_access_token)
    {
        let value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|err| {
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                format!("Invalid Anthropic bearer token: {err}"),
            )
        })?;
        headers.insert(AUTHORIZATION, value);
        return Ok(headers);
    }

    if let Some(api_key) = config::anthropic_api_key() {
        let value = HeaderValue::from_str(&api_key).map_err(|err| {
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                format!("Invalid Anthropic API key: {err}"),
            )
        })?;
        headers.insert(HeaderName::from_static("x-api-key"), value);
        return Ok(headers);
    }

    Err(json_error(
        StatusCode::UNAUTHORIZED,
        "authentication_error",
        "Anthropic passthrough is enabled but no Anthropic credential is available. Set CLAUDE_CODE_OAUTH_TOKEN, CCP_ANTHROPIC_AUTH_TOKEN, or CCP_ANTHROPIC_API_KEY.",
    ))
}

fn copy_header(
    incoming: &HeaderMap,
    outgoing: &mut HeaderMap,
    name: HeaderName,
    default_value: Option<&'static str>,
) {
    if let Some(value) = incoming.get(&name) {
        outgoing.insert(name, value.clone());
        return;
    }
    if let Some(default_value) = default_value {
        outgoing.insert(name, HeaderValue::from_static(default_value));
    }
}

fn copy_incoming_auth_headers(incoming: &HeaderMap, outgoing: &mut HeaderMap) -> bool {
    if let Some(value) = incoming.get(AUTHORIZATION)
        && !is_placeholder_auth(value)
    {
        outgoing.insert(AUTHORIZATION, value.clone());
        return true;
    }

    let api_key_name = HeaderName::from_static("x-api-key");
    if let Some(value) = incoming.get(&api_key_name)
        && !is_placeholder_auth(value)
    {
        outgoing.insert(api_key_name, value.clone());
        return true;
    }

    false
}

fn is_placeholder_auth(value: &HeaderValue) -> bool {
    value
        .to_str()
        .ok()
        .map(|raw| {
            let normalized = raw.trim();
            normalized.eq_ignore_ascii_case("unused")
                || normalized.eq_ignore_ascii_case("bearer unused")
        })
        .unwrap_or(false)
}

async fn response_from_upstream_buffered(upstream: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        builder = builder.header(name, value);
    }
    let bytes = match upstream.bytes().await {
        Ok(bytes) => bytes,
        Err(err) => {
            return json_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("Failed to read Anthropic response: {err}"),
            );
        }
    };
    builder.body(Body::from(bytes)).unwrap_or_else(|err| {
        json_error(
            StatusCode::BAD_GATEWAY,
            "api_error",
            format!("Failed to build Anthropic response: {err}"),
        )
    })
}

fn response_from_upstream_stream(upstream: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        builder = builder.header(name, value);
    }
    let stream = upstream
        .bytes_stream()
        .map_err(std::io::Error::other);
    builder
        .body(Body::from_stream(stream))
        .unwrap_or_else(|err| {
            json_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("Failed to build Anthropic stream response: {err}"),
            )
        })
}

fn request_model<T: Serialize>(body: &T) -> Option<String> {
    let value = serde_json::to_value(body).ok()?;
    value.get("model")?.as_str().map(str::to_string)
}

fn read_claude_credentials_access_token() -> Option<String> {
    let path = std::env::var_os("CCP_CLAUDE_CREDENTIALS_PATH")
        .map(PathBuf::from)
        .or_else(default_claude_credentials_path)?;
    let raw = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .pointer("/claudeAiOauth/accessToken")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn default_claude_credentials_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".claude").join(".credentials.json"))
}
