//! Browser input event bridge.
//!
//! Decodes compact binary input events from the browser and maps them
//! to abstract application actions. Consumers define their own action types
//! and mapping logic; this crate provides the raw event decode and a
//! default mapper with common 3D/2D interaction patterns.

use ustreamer_proto::input::InputEvent;

/// Abstract application action produced by the input mapper.
/// These cover common interaction patterns for 3D/2D applications.
/// Consumers can extend or replace these with domain-specific actions.
#[derive(Debug, Clone)]
pub enum AppAction {
    /// Camera orbit / rotation (e.g., left-drag in a 3D viewport).
    Rotate { dx: f32, dy: f32 },
    /// Camera zoom (e.g., scroll or middle-drag).
    Zoom { delta: f32 },
    /// Camera pan / translate.
    Pan { dx: f32, dy: f32 },
    /// Discrete scroll through items (e.g., layers, frames, slices).
    ScrollStep { delta: i32 },
    /// Two-axis parameter adjustment via drag (e.g., brightness/contrast).
    DragAdjust { dx: f32, dy: f32 },
    /// Raw pointer position update (for cursor overlay rendering).
    PointerUpdate { x: f32, y: f32 },
}

/// Maps raw browser input events to application actions based on the current interaction mode.
pub struct InputMapper {
    mode: InteractionMode,
    last_x: f32,
    last_y: f32,
    buttons: u8,
}

/// Interaction modes that determine how pointer drags are interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionMode {
    Rotate,
    Pan,
    Zoom,
    Scroll,
    /// Two-axis parameter adjustment via drag.
    DragAdjust,
}

impl Default for InputMapper {
    fn default() -> Self {
        Self {
            mode: InteractionMode::Rotate,
            last_x: 0.0,
            last_y: 0.0,
            buttons: 0,
        }
    }
}

impl InputMapper {
    pub fn set_mode(&mut self, mode: InteractionMode) {
        self.mode = mode;
    }

    /// Process a raw input event and return zero or more application actions.
    pub fn process(&mut self, event: &InputEvent) -> Vec<AppAction> {
        match event {
            InputEvent::PointerMove { x, y, buttons, .. } => {
                let dx = x - self.last_x;
                let dy = y - self.last_y;
                self.last_x = *x;
                self.last_y = *y;
                self.buttons = *buttons;

                let mut actions = vec![AppAction::PointerUpdate { x: *x, y: *y }];

                // Left button drag → mode-dependent action
                if *buttons & 1 != 0 {
                    match self.mode {
                        InteractionMode::Rotate => {
                            actions.push(AppAction::Rotate { dx, dy });
                        }
                        InteractionMode::Pan => {
                            actions.push(AppAction::Pan { dx, dy });
                        }
                        InteractionMode::Zoom => {
                            actions.push(AppAction::Zoom { delta: dy });
                        }
                        InteractionMode::DragAdjust => {
                            actions.push(AppAction::DragAdjust { dx, dy });
                        }
                        InteractionMode::Scroll => {}
                    }
                }

                actions
            }
            InputEvent::Scroll { delta_y, .. } => {
                vec![AppAction::ScrollStep {
                    delta: if *delta_y > 0.0 { 1 } else { -1 },
                }]
            }
            InputEvent::PointerDown { x, y, .. } => {
                self.last_x = *x;
                self.last_y = *y;
                vec![]
            }
            _ => vec![],
        }
    }
}
