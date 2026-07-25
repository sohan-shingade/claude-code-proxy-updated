//! Seeding of Claude Code's `/model` picker cache
//! (`<claude-config-dir>/cache/gateway-models.json`).
//!
//! Claude Code re-reads this file at most once per session and only trusts it
//! when the recorded `baseUrl` exactly matches its `ANTHROPIC_BASE_URL`.
//! Writes must be atomic: Claude Code memoizes a failed parse as "no gateway
//! models" for the whole session, so a session that reads a half-written file
//! would silently lose every gateway model until restart.

use crate::registry::Registry;
use crate::server::picker_cache_json;
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const CACHE_FILE_NAME: &str = "gateway-models.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeedResult {
    Written { models: usize },
    Unchanged { models: usize },
    SkippedMissingDir,
}

#[derive(Debug)]
pub struct SeedOutcome {
    pub config_dir: PathBuf,
    pub cache_file: PathBuf,
    pub result: anyhow::Result<SeedResult>,
}

pub fn cache_file_for(claude_config_dir: &Path) -> PathBuf {
    claude_config_dir.join("cache").join(CACHE_FILE_NAME)
}

/// Claude Code config dirs to seed, in priority order:
/// `picker.claudeConfigDirs` from the proxy config (or the
/// `CCP_PICKER_CLAUDE_CONFIG_DIRS` env override), else `$CLAUDE_CONFIG_DIR`,
/// else `~/.claude`.
pub fn claude_config_dirs() -> Vec<PathBuf> {
    if let Some(dirs) = crate::config::picker_claude_config_dirs()
        && !dirs.is_empty()
    {
        return dirs;
    }
    if let Some(dir) = std::env::var_os("CLAUDE_CONFIG_DIR") {
        return vec![PathBuf::from(dir)];
    }
    match std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        Some(home) => vec![PathBuf::from(home).join(".claude")],
        None => Vec::new(),
    }
}

/// Seed every resolved Claude Code config dir. Dirs that don't exist are
/// skipped rather than created: an absent dir means Claude Code was never
/// run with it, and creating one would leave a broken half-install behind.
pub fn seed_claude_picker_caches(registry: &Registry, base_url: &str) -> Vec<SeedOutcome> {
    let fetched_at_ms = current_millis();
    claude_config_dirs()
        .into_iter()
        .map(|config_dir| seed_one(registry, base_url, &config_dir, fetched_at_ms))
        .collect()
}

pub fn seed_one(
    registry: &Registry,
    base_url: &str,
    claude_config_dir: &Path,
    fetched_at_ms: u64,
) -> SeedOutcome {
    let cache_file = cache_file_for(claude_config_dir);
    let result = if claude_config_dir.is_dir() {
        write_cache(registry, base_url, &cache_file, fetched_at_ms)
    } else {
        Ok(SeedResult::SkippedMissingDir)
    };
    SeedOutcome {
        config_dir: claude_config_dir.to_path_buf(),
        cache_file,
        result,
    }
}

fn write_cache(
    registry: &Registry,
    base_url: &str,
    cache_file: &Path,
    fetched_at_ms: u64,
) -> anyhow::Result<SeedResult> {
    let cache = picker_cache_json(registry, base_url, fetched_at_ms);
    let models = cache["models"].as_array().map_or(0, Vec::len);
    if cache_matches(cache_file, &cache) {
        return Ok(SeedResult::Unchanged { models });
    }
    write_atomic(cache_file, &cache)?;
    Ok(SeedResult::Written { models })
}

/// True when the existing cache already has this `baseUrl` and model list
/// (`fetchedAt` is ignored — Claude Code never reads it). Skipping the write
/// keeps mtimes stable and avoids replacing the file out from under a
/// concurrently starting Claude Code session for no reason.
fn cache_matches(cache_file: &Path, next: &Value) -> bool {
    let Ok(raw) = fs::read_to_string(cache_file) else {
        return false;
    };
    let Ok(existing) = serde_json::from_str::<Value>(&raw) else {
        return false;
    };
    existing["baseUrl"] == next["baseUrl"] && existing["models"] == next["models"]
}

/// Write via a same-directory temp file and rename, so readers only ever see
/// the old or the new complete file. Claude Code writes this file with mode
/// 0600; match it.
fn write_atomic(cache_file: &Path, value: &Value) -> anyhow::Result<()> {
    let dir = cache_file
        .parent()
        .ok_or_else(|| anyhow::anyhow!("cache file {} has no parent dir", cache_file.display()))?;
    fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(".{}.tmp.{}", CACHE_FILE_NAME, std::process::id()));
    let mut payload = serde_json::to_vec_pretty(value)?;
    payload.push(b'\n');
    {
        let mut options = fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&tmp)?;
        file.write_all(&payload)?;
        file.sync_all()?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // OpenOptions mode only applies on create; force 0600 in case a
        // stale temp file from a previous run survived with other bits.
        let _ = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600));
    }
    fs::rename(&tmp, cache_file).inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })?;
    Ok(())
}

fn current_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AliasProvider;

    fn registry() -> Registry {
        Registry::new(AliasProvider::Codex)
    }

    #[test]
    fn seeds_new_cache_file_with_0600() {
        let dir = tempfile::TempDir::new().unwrap();
        let outcome = seed_one(&registry(), "http://localhost:18765", dir.path(), 42);

        assert!(matches!(
            outcome.result.as_ref().unwrap(),
            SeedResult::Written { models } if *models > 0
        ));
        let raw = fs::read_to_string(&outcome.cache_file).unwrap();
        let cache: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(cache["baseUrl"], "http://localhost:18765");
        assert_eq!(cache["fetchedAt"], 42);
        assert!(
            cache["models"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["id"] == "claude-gpt-5-6-sol")
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&outcome.cache_file)
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn second_seed_is_unchanged_and_preserves_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let registry = registry();
        seed_one(&registry, "http://localhost:18765", dir.path(), 1)
            .result
            .unwrap();
        let before = fs::read_to_string(cache_file_for(dir.path())).unwrap();

        let outcome = seed_one(&registry, "http://localhost:18765", dir.path(), 2);

        assert!(matches!(
            outcome.result.unwrap(),
            SeedResult::Unchanged { .. }
        ));
        assert_eq!(
            fs::read_to_string(cache_file_for(dir.path())).unwrap(),
            before,
            "unchanged seed must not rewrite the file"
        );
    }

    #[test]
    fn base_url_change_rewrites_cache() {
        let dir = tempfile::TempDir::new().unwrap();
        let registry = registry();
        seed_one(&registry, "http://localhost:18765", dir.path(), 1)
            .result
            .unwrap();

        let outcome = seed_one(&registry, "http://localhost:9999", dir.path(), 2);

        assert!(matches!(
            outcome.result.unwrap(),
            SeedResult::Written { .. }
        ));
        let raw = fs::read_to_string(cache_file_for(dir.path())).unwrap();
        let cache: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(cache["baseUrl"], "http://localhost:9999");
    }

    #[test]
    fn corrupt_cache_is_replaced() {
        let dir = tempfile::TempDir::new().unwrap();
        let cache_file = cache_file_for(dir.path());
        fs::create_dir_all(cache_file.parent().unwrap()).unwrap();
        fs::write(&cache_file, "{ not json").unwrap();

        let outcome = seed_one(&registry(), "http://localhost:18765", dir.path(), 1);

        assert!(matches!(
            outcome.result.unwrap(),
            SeedResult::Written { .. }
        ));
        let raw = fs::read_to_string(&cache_file).unwrap();
        assert!(serde_json::from_str::<Value>(&raw).is_ok());
    }

    #[test]
    fn missing_claude_config_dir_is_skipped() {
        let dir = tempfile::TempDir::new().unwrap();
        let missing = dir.path().join("does-not-exist");

        let outcome = seed_one(&registry(), "http://localhost:18765", &missing, 1);

        assert!(matches!(
            outcome.result.unwrap(),
            SeedResult::SkippedMissingDir
        ));
        assert!(!missing.exists(), "seeding must not create the config dir");
    }
}
