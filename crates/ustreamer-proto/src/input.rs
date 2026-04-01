/// Input event types sent from browser to server.
///
/// Binary wire format: 1-byte type tag followed by type-specific payload.

#[derive(Debug, Clone, Copy)]
pub enum InputEvent {
    /// Continuous pointer movement (sent as unreliable datagram).
    PointerMove {
        /// Normalized X coordinate (0.0–1.0).
        x: f32,
        /// Normalized Y coordinate (0.0–1.0).
        y: f32,
        /// Button bitmask (same as PointerEvent.buttons).
        buttons: u8,
        /// Client-side timestamp (ms) for RTT measurement.
        timestamp_ms: u32,
    },

    /// Discrete pointer button press (sent reliably).
    PointerDown {
        button: u8,
        x: f32,
        y: f32,
    },

    /// Discrete pointer button release (sent reliably).
    PointerUp {
        button: u8,
        x: f32,
        y: f32,
    },

    /// Scroll / wheel event (unreliable).
    Scroll {
        delta_x: f32,
        delta_y: f32,
        mode: ScrollMode,
    },

    /// Key press (reliable).
    KeyDown { code: u16 },

    /// Key release (reliable).
    KeyUp { code: u16 },
}

#[derive(Debug, Clone, Copy)]
pub enum ScrollMode {
    Pixels = 0,
    Lines = 1,
    Pages = 2,
}

// Wire format type tags
const TAG_POINTER_MOVE: u8 = 0x01;
const TAG_POINTER_DOWN: u8 = 0x02;
const TAG_POINTER_UP: u8 = 0x03;
const TAG_SCROLL: u8 = 0x04;
const TAG_KEY_DOWN: u8 = 0x10;
const TAG_KEY_UP: u8 = 0x11;

impl InputEvent {
    /// Serialize to compact binary format.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            InputEvent::PointerMove { x, y, buttons, timestamp_ms } => {
                let mut buf = Vec::with_capacity(14);
                buf.push(TAG_POINTER_MOVE);
                buf.push(*buttons);
                buf.extend_from_slice(&x.to_le_bytes());
                buf.extend_from_slice(&y.to_le_bytes());
                buf.extend_from_slice(&timestamp_ms.to_le_bytes());
                buf
            }
            InputEvent::PointerDown { button, x, y } => {
                let mut buf = Vec::with_capacity(10);
                buf.push(TAG_POINTER_DOWN);
                buf.push(*button);
                buf.extend_from_slice(&x.to_le_bytes());
                buf.extend_from_slice(&y.to_le_bytes());
                buf
            }
            InputEvent::PointerUp { button, x, y } => {
                let mut buf = Vec::with_capacity(10);
                buf.push(TAG_POINTER_UP);
                buf.push(*button);
                buf.extend_from_slice(&x.to_le_bytes());
                buf.extend_from_slice(&y.to_le_bytes());
                buf
            }
            InputEvent::Scroll { delta_x, delta_y, mode } => {
                let mut buf = Vec::with_capacity(10);
                buf.push(TAG_SCROLL);
                buf.extend_from_slice(&delta_x.to_le_bytes());
                buf.extend_from_slice(&delta_y.to_le_bytes());
                buf.push(*mode as u8);
                buf
            }
            InputEvent::KeyDown { code } => {
                vec![TAG_KEY_DOWN, (*code >> 8) as u8, *code as u8]
            }
            InputEvent::KeyUp { code } => {
                vec![TAG_KEY_UP, (*code >> 8) as u8, *code as u8]
            }
        }
    }

    /// Deserialize from compact binary format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, InputEventError> {
        if data.is_empty() {
            return Err(InputEventError::Empty);
        }

        match data[0] {
            TAG_POINTER_MOVE if data.len() >= 14 => Ok(InputEvent::PointerMove {
                buttons: data[1],
                x: f32::from_le_bytes(data[2..6].try_into().unwrap()),
                y: f32::from_le_bytes(data[6..10].try_into().unwrap()),
                timestamp_ms: u32::from_le_bytes(data[10..14].try_into().unwrap()),
            }),
            TAG_POINTER_DOWN if data.len() >= 10 => Ok(InputEvent::PointerDown {
                button: data[1],
                x: f32::from_le_bytes(data[2..6].try_into().unwrap()),
                y: f32::from_le_bytes(data[6..10].try_into().unwrap()),
            }),
            TAG_POINTER_UP if data.len() >= 10 => Ok(InputEvent::PointerUp {
                button: data[1],
                x: f32::from_le_bytes(data[2..6].try_into().unwrap()),
                y: f32::from_le_bytes(data[6..10].try_into().unwrap()),
            }),
            TAG_SCROLL if data.len() >= 10 => Ok(InputEvent::Scroll {
                delta_x: f32::from_le_bytes(data[1..5].try_into().unwrap()),
                delta_y: f32::from_le_bytes(data[5..9].try_into().unwrap()),
                mode: match data[9] {
                    1 => ScrollMode::Lines,
                    2 => ScrollMode::Pages,
                    _ => ScrollMode::Pixels,
                },
            }),
            TAG_KEY_DOWN if data.len() >= 3 => Ok(InputEvent::KeyDown {
                code: ((data[1] as u16) << 8) | data[2] as u16,
            }),
            TAG_KEY_UP if data.len() >= 3 => Ok(InputEvent::KeyUp {
                code: ((data[1] as u16) << 8) | data[2] as u16,
            }),
            tag => Err(InputEventError::UnknownTag(tag)),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum InputEventError {
    #[error("empty input buffer")]
    Empty,
    #[error("unknown input event tag: 0x{0:02x}")]
    UnknownTag(u8),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_pointer_move() {
        let event = InputEvent::PointerMove {
            x: 0.5,
            y: 0.75,
            buttons: 1,
            timestamp_ms: 12345,
        };
        let bytes = event.to_bytes();
        let decoded = InputEvent::from_bytes(&bytes).unwrap();
        match decoded {
            InputEvent::PointerMove { x, y, buttons, timestamp_ms } => {
                assert!((x - 0.5).abs() < f32::EPSILON);
                assert!((y - 0.75).abs() < f32::EPSILON);
                assert_eq!(buttons, 1);
                assert_eq!(timestamp_ms, 12345);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn roundtrip_key_down() {
        let event = InputEvent::KeyDown { code: 0x0041 };
        let bytes = event.to_bytes();
        let decoded = InputEvent::from_bytes(&bytes).unwrap();
        match decoded {
            InputEvent::KeyDown { code } => assert_eq!(code, 0x0041),
            _ => panic!("wrong variant"),
        }
    }
}
