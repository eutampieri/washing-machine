//! MQTT publishing with Home Assistant discovery.
//!
//! Topics (with the default base topic `washing_machine`):
//! * `washing_machine/availability` — `online` / `offline` (retained, also the
//!   last will, so Home Assistant marks the entities unavailable if we die);
//! * `washing_machine/state` — retained JSON, see [`State`];
//! * `<discovery prefix>/<component>/<device id>/<entity>/config` — retained
//!   discovery configs, re-sent whenever Home Assistant comes back online.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant, SystemTime},
};

use log::{debug, info, warn};
use rumqttc::{Client, Event, LastWill, MqttOptions, Packet, QoS};
use serde::Serialize;
use serde_json::{json, Value};

use crate::recognise::{Phase, Reading};

pub struct MqttSettings {
    pub host: String,
    pub port: u16,
    pub username: Option<String>,
    pub password: Option<String>,
    pub client_id: String,
}

pub struct HaSettings {
    pub discovery_prefix: String,
    pub base_topic: String,
    pub device_id: String,
    pub device_name: String,
}

impl HaSettings {
    fn availability_topic(&self) -> String {
        format!("{}/availability", self.base_topic)
    }

    fn state_topic(&self) -> String {
        format!("{}/state", self.base_topic)
    }

    fn ha_status_topic(&self) -> String {
        format!("{}/status", self.discovery_prefix)
    }

    /// (topic, payload) of every discovery message.
    fn discovery(&self) -> Vec<(String, Value)> {
        let device = json!({
            "identifiers": [self.device_id],
            "name": self.device_name,
            "model": "Camera-based washing machine monitor",
            "sw_version": env!("CARGO_PKG_VERSION"),
        });
        let origin = json!({
            "name": env!("CARGO_PKG_NAME"),
            "sw_version": env!("CARGO_PKG_VERSION"),
            "support_url": "https://github.com/eutampieri/washing-machine",
        });
        let entity = |component: &str, key: &str, extra: Value| {
            let mut config = json!({
                "unique_id": format!("{}_{key}", self.device_id),
                "state_topic": self.state_topic(),
                "availability_topic": self.availability_topic(),
                "device": device,
                "origin": origin,
            });
            config
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            (
                format!(
                    "{}/{component}/{}/{key}/config",
                    self.discovery_prefix, self.device_id
                ),
                config,
            )
        };
        // Templates rendering `None` (JSON null) set the entity to unknown.
        vec![
            entity(
                "binary_sensor",
                "running",
                json!({
                    "name": "Running",
                    "device_class": "running",
                    "value_template": "{{ value_json.running }}",
                    "payload_on": "ON",
                    "payload_off": "OFF",
                }),
            ),
            entity(
                "sensor",
                "phase",
                json!({
                    "name": "Phase",
                    "device_class": "enum",
                    "options": Phase::ALL.map(Phase::label),
                    "icon": "mdi:washing-machine",
                    "value_template": "{{ value_json.phase }}",
                }),
            ),
            entity(
                "sensor",
                "remaining_time",
                json!({
                    "name": "Remaining time",
                    "device_class": "duration",
                    "unit_of_measurement": "min",
                    "icon": "mdi:timer-sand",
                    "value_template": "{{ value_json.remaining_minutes }}",
                }),
            ),
            entity(
                "sensor",
                "end_time",
                json!({
                    "name": "End time",
                    "device_class": "timestamp",
                    "value_template": "{{ value_json.end_time }}",
                }),
            ),
        ]
    }
}

/// Payload of the state topic.
#[derive(Debug, PartialEq, Serialize)]
pub struct State {
    pub running: &'static str,
    pub phase: Option<&'static str>,
    pub remaining_minutes: Option<u16>,
    /// RFC 3339, UTC.
    pub end_time: Option<String>,
}

impl State {
    pub fn new(reading: &Reading, now: SystemTime) -> Self {
        Self {
            running: if reading.running { "ON" } else { "OFF" },
            phase: reading.phase.map(Phase::label),
            remaining_minutes: reading.remaining_minutes,
            end_time: reading.remaining_minutes.map(|m| {
                let end = now + Duration::from_secs(u64::from(m) * 60);
                humantime::format_rfc3339_seconds(end).to_string()
            }),
        }
    }
}

pub struct Publisher {
    client: Client,
    ha: Arc<HaSettings>,
    available: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
    event_loop: JoinHandle<()>,
}

impl Publisher {
    /// Starts the MQTT connection in a background thread; it reconnects on
    /// its own and re-announces everything on every (re)connection.
    pub fn start(mqtt: MqttSettings, ha: HaSettings) -> Self {
        let ha = Arc::new(ha);
        let mut options = MqttOptions::new(mqtt.client_id, mqtt.host, mqtt.port);
        options.set_keep_alive(Duration::from_secs(30));
        if let Some(username) = mqtt.username {
            options.set_credentials(username, mqtt.password.unwrap_or_default());
        }
        options.set_last_will(LastWill::new(
            ha.availability_topic(),
            "offline",
            QoS::AtLeastOnce,
            true,
        ));
        let (client, mut connection) = Client::new(options, 64);

        let available = Arc::new(AtomicBool::new(true));
        let stopping = Arc::new(AtomicBool::new(false));
        let event_loop = {
            let (client, ha, available, stopping) = (
                client.clone(),
                ha.clone(),
                available.clone(),
                stopping.clone(),
            );
            std::thread::spawn(move || {
                for event in connection.iter() {
                    match event {
                        Ok(Event::Incoming(Packet::ConnAck(_))) => {
                            info!("connected to the MQTT broker");
                            announce(&client, &ha, available.load(Ordering::SeqCst));
                        }
                        Ok(Event::Incoming(Packet::Publish(p)))
                            if p.topic == ha.ha_status_topic() && &p.payload[..] == b"online" =>
                        {
                            info!("Home Assistant restarted, re-announcing");
                            announce(&client, &ha, available.load(Ordering::SeqCst));
                        }
                        Ok(event) => debug!("MQTT event: {event:?}"),
                        Err(_) if stopping.load(Ordering::SeqCst) => break,
                        Err(e) => {
                            warn!("MQTT connection error: {e}; retrying in 5 s");
                            std::thread::sleep(Duration::from_secs(5));
                        }
                    }
                }
            })
        };
        Self {
            client,
            ha,
            available,
            stopping,
            event_loop,
        }
    }

    pub fn publish_state(&self, state: &State) {
        let payload = serde_json::to_vec(state).expect("state serialises");
        publish(&self.client, self.ha.state_topic(), payload, true);
    }

    /// Marks the entities (un)available, e.g. while the camera is unreachable.
    pub fn set_available(&self, available: bool) {
        if self.available.swap(available, Ordering::SeqCst) != available {
            publish(
                &self.client,
                self.ha.availability_topic(),
                availability(available),
                true,
            );
        }
    }

    /// Publishes `offline` and disconnects cleanly.
    pub fn shutdown(self) {
        self.stopping.store(true, Ordering::SeqCst);
        publish(
            &self.client,
            self.ha.availability_topic(),
            availability(false),
            true,
        );
        if let Err(e) = self.client.disconnect() {
            warn!("MQTT disconnect failed: {e}");
        }
        let deadline = Instant::now() + Duration::from_secs(3);
        while !self.event_loop.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

fn availability(available: bool) -> &'static str {
    if available {
        "online"
    } else {
        "offline"
    }
}

/// Non-blocking publish: never stalls the event loop or the capture loop
/// while the broker is unreachable.
fn publish(client: &Client, topic: String, payload: impl Into<Vec<u8>>, retain: bool) {
    if let Err(e) = client.try_publish(topic.clone(), QoS::AtLeastOnce, retain, payload) {
        warn!("cannot queue MQTT message for {topic}: {e}");
    }
}

fn announce(client: &Client, ha: &HaSettings, available: bool) {
    if let Err(e) = client.try_subscribe(ha.ha_status_topic(), QoS::AtLeastOnce) {
        warn!("cannot subscribe to {}: {e}", ha.ha_status_topic());
    }
    for (topic, config) in ha.discovery() {
        publish(
            client,
            topic,
            serde_json::to_vec(&config).expect("config serialises"),
            true,
        );
    }
    publish(
        client,
        ha.availability_topic(),
        availability(available),
        true,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recognise::Digit;

    fn reading(running: bool, phase: Option<Phase>, minutes: Option<u16>) -> Reading {
        Reading {
            running,
            phase,
            remaining_minutes: minutes,
            digits: [Digit {
                value: None,
                confidence: 0.0,
            }; 3],
            display_match: 1.0,
            bar_match: 1.0,
            alignment: crate::align::Alignment {
                matches: 0,
                inliers: 0,
                panel_shift: 0.0,
                warped: false,
            },
        }
    }

    #[test]
    fn state_payload() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_733_738_109);
        let state = State::new(&reading(true, Some(Phase::Rinse), Some(20)), now);
        assert_eq!(
            serde_json::to_value(&state).unwrap(),
            json!({"running": "ON", "phase": "Risciacquo", "remaining_minutes": 20,
                   "end_time": "2024-12-09T10:15:09Z"})
        );
        let state = State::new(&reading(false, None, None), now);
        assert_eq!(
            serde_json::to_value(&state).unwrap(),
            json!({"running": "OFF", "phase": null, "remaining_minutes": null, "end_time": null})
        );
    }

    #[test]
    fn discovery_topics() {
        let ha = HaSettings {
            discovery_prefix: "homeassistant".into(),
            base_topic: "washing_machine".into(),
            device_id: "wm".into(),
            device_name: "Washing machine".into(),
        };
        let discovery = ha.discovery();
        let topics: Vec<_> = discovery.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(
            topics,
            [
                "homeassistant/binary_sensor/wm/running/config",
                "homeassistant/sensor/wm/phase/config",
                "homeassistant/sensor/wm/remaining_time/config",
                "homeassistant/sensor/wm/end_time/config",
            ]
        );
        for (_, config) in &discovery {
            assert_eq!(config["state_topic"], "washing_machine/state");
            assert_eq!(config["availability_topic"], "washing_machine/availability");
            assert_eq!(config["device"]["identifiers"][0], "wm");
        }
        assert_eq!(discovery[1].1["options"].as_array().unwrap().len(), 6);
    }
}
