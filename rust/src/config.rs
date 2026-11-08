//! Configuration from environment variables (see README.md for the list).

use std::{str::FromStr, time::Duration};

use anyhow::{Context, Result};

use crate::mqtt::{HaSettings, MqttSettings};

pub struct Config {
    pub stream_url: String,
    pub capture_timeout: Duration,
    pub warmup_frames: u32,
    pub interval_running: Duration,
    pub interval_idle: Duration,
    /// Consecutive failed readings before the entities go unavailable.
    pub failures_before_unavailable: u32,
    pub mqtt: MqttSettings,
    pub ha: HaSettings,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let device_id = var_or("DEVICE_ID", "washing_machine");
        Ok(Self {
            stream_url: required("RTSP_URL")?,
            capture_timeout: Duration::from_secs(parse_or("CAPTURE_TIMEOUT_SECS", 10)?),
            warmup_frames: parse_or("WARMUP_FRAMES", 5)?,
            interval_running: Duration::from_secs(parse_or("INTERVAL_RUNNING_SECS", 30)?),
            interval_idle: Duration::from_secs(parse_or("INTERVAL_IDLE_SECS", 60)?),
            failures_before_unavailable: parse_or("FAILURES_BEFORE_UNAVAILABLE", 3)?,
            mqtt: MqttSettings {
                host: var_or("MQTT_HOST", "localhost"),
                port: parse_or("MQTT_PORT", 1883)?,
                username: optional("MQTT_USERNAME"),
                password: optional("MQTT_PASSWORD"),
                client_id: var_or("MQTT_CLIENT_ID", &format!("{device_id}_monitor")),
            },
            ha: HaSettings {
                discovery_prefix: var_or("HA_DISCOVERY_PREFIX", "homeassistant"),
                base_topic: var_or("MQTT_BASE_TOPIC", &device_id),
                device_name: var_or("DEVICE_NAME", "Washing machine"),
                device_id,
            },
        })
    }
}

fn optional(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn required(name: &str) -> Result<String> {
    optional(name).with_context(|| format!("environment variable {name} is required"))
}

fn var_or(name: &str, default: &str) -> String {
    optional(name).unwrap_or_else(|| default.to_string())
}

fn parse_or<T: FromStr>(name: &str, default: T) -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    optional(name)
        .map(|v| {
            v.parse()
                .with_context(|| format!("invalid value for {name}: {v:?}"))
        })
        .unwrap_or(Ok(default))
}
