//! Cognitive Rules - Procedural Logic for Geometric Reasoning
//!
//! Rules define conditions and actions that operate on the geometric state,
//! allowing programmable responses to detected patterns.

use crate::geometry::{GeometryCore, GeometryMode, TrinityComponent, Rotation4D};
use super::TrinityState;
use serde::{Serialize, Deserialize};

/// A condition that can be evaluated against the current state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleCondition {
    /// Dialectic tension exceeds threshold
    TensionAbove(f64),
    /// Dialectic tension below threshold
    TensionBelow(f64),
    /// Synthesis coherence exceeds threshold
    CoherenceAbove(f64),
    /// System is in resonance
    IsResonant,
    /// Specific component is dominant
    ComponentDominant(TrinityComponent),
    /// Alpha-Beta distance exceeds threshold
    DistanceAbove(f64),
    /// Alpha-Beta distance below threshold
    DistanceBelow(f64),
    /// Frame count modulo (periodic trigger)
    EveryNFrames(u64),
    /// Synthesis rate exceeds threshold
    SynthesisRateAbove(f64),
    /// Compound condition: all must be true
    All(Vec<RuleCondition>),
    /// Compound condition: any must be true
    Any(Vec<RuleCondition>),
    /// Negation
    Not(Box<RuleCondition>),
    /// Always true
    Always,
}

/// An action to take when a rule fires
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleAction {
    /// Switch geometry mode
    SetMode(GeometryMode),
    /// Apply rotation in specified plane
    ApplyRotation { plane: RotationPlane, angle: f64 },
    /// Set rotation speed
    SetRotationSpeed([f64; 6]),
    /// Scale rotation speed
    ScaleRotationSpeed(f64),
    /// Log a message
    Log(String),
    /// Trigger synthesis check
    ForceSynthesisCheck,
    /// Reset to initial state
    Reset,
    /// Execute multiple actions
    Sequence(Vec<RuleAction>),
    /// No action
    None,
}

/// Rotation planes in 4D
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RotationPlane {
    XY,
    XZ,
    XW,
    YZ,
    YW,
    ZW,
}

/// A cognitive rule with condition and action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveRule {
    pub name: String,
    pub condition: RuleCondition,
    pub action: RuleAction,
    pub priority: i32,
    pub enabled: bool,
    /// Cooldown frames after firing
    pub cooldown: u64,
    /// Last frame this rule fired
    #[serde(skip)]
    last_fired: u64,
}

impl CognitiveRule {
    pub fn new(name: impl Into<String>, condition: RuleCondition, action: RuleAction) -> Self {
        Self {
            name: name.into(),
            condition,
            action,
            priority: 0,
            enabled: true,
            cooldown: 0,
            last_fired: 0,
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_cooldown(mut self, frames: u64) -> Self {
        self.cooldown = frames;
        self
    }

    /// Check if this rule's condition is met
    fn evaluate(&self, state: &TrinityState, frame: u64) -> bool {
        if !self.enabled {
            return false;
        }

        // Check cooldown
        if frame < self.last_fired + self.cooldown {
            return false;
        }

        self.evaluate_condition(&self.condition, state, frame)
    }

    fn evaluate_condition(&self, cond: &RuleCondition, state: &TrinityState, frame: u64) -> bool {
        match cond {
            RuleCondition::TensionAbove(t) => state.dialectic_tension() > *t,
            RuleCondition::TensionBelow(t) => state.dialectic_tension() < *t,
            RuleCondition::CoherenceAbove(c) => state.synthesis_coherence() > *c,
            RuleCondition::IsResonant => state.is_resonant(),
            RuleCondition::ComponentDominant(comp) => {
                match comp {
                    TrinityComponent::Alpha => state.alpha.dominant,
                    TrinityComponent::Beta => state.beta.dominant,
                    TrinityComponent::Gamma => state.gamma.dominant,
                }
            }
            RuleCondition::DistanceAbove(d) => state.alpha_beta_distance > *d,
            RuleCondition::DistanceBelow(d) => state.alpha_beta_distance < *d,
            RuleCondition::EveryNFrames(n) => frame % n == 0,
            RuleCondition::SynthesisRateAbove(r) => state.synthesis_rate(100) > *r,
            RuleCondition::All(conditions) => {
                conditions.iter().all(|c| self.evaluate_condition(c, state, frame))
            }
            RuleCondition::Any(conditions) => {
                conditions.iter().any(|c| self.evaluate_condition(c, state, frame))
            }
            RuleCondition::Not(c) => !self.evaluate_condition(c, state, frame),
            RuleCondition::Always => true,
        }
    }
}

/// The rule engine manages and executes cognitive rules
pub struct RuleEngine {
    rules: Vec<CognitiveRule>,
    frame: u64,
}

impl RuleEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            rules: Vec::new(),
            frame: 0,
        };

        // Add default rules
        engine.add_default_rules();

        engine
    }

    /// Add default cognitive rules
    fn add_default_rules(&mut self) {
        // Rule: High tension triggers mode expansion
        self.add_rule(
            CognitiveRule::new(
                "expand_on_tension",
                RuleCondition::TensionAbove(0.8),
                RuleAction::SetMode(GeometryMode::Expanded600),
            )
            .with_priority(10)
            .with_cooldown(120)
        );

        // Rule: Low tension returns to Trinity mode
        self.add_rule(
            CognitiveRule::new(
                "contract_on_low_tension",
                RuleCondition::All(vec![
                    RuleCondition::TensionBelow(0.2),
                    RuleCondition::Not(Box::new(RuleCondition::IsResonant)),
                ]),
                RuleAction::SetMode(GeometryMode::Trinity),
            )
            .with_priority(5)
            .with_cooldown(120)
        );

        // Rule: Resonance boosts rotation
        self.add_rule(
            CognitiveRule::new(
                "resonance_boost",
                RuleCondition::IsResonant,
                RuleAction::ScaleRotationSpeed(1.5),
            )
            .with_priority(3)
            .with_cooldown(60)
        );
    }

    /// Add a new rule
    pub fn add_rule(&mut self, rule: CognitiveRule) {
        self.rules.push(rule);
        // Sort by priority (higher first)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Remove a rule by name
    pub fn remove_rule(&mut self, name: &str) {
        self.rules.retain(|r| r.name != name);
    }

    /// Enable or disable a rule by name
    pub fn set_rule_enabled(&mut self, name: &str, enabled: bool) {
        for rule in &mut self.rules {
            if rule.name == name {
                rule.enabled = enabled;
            }
        }
    }

    /// Execute all applicable rules
    pub fn execute(&mut self, geometry: &mut GeometryCore, state: &TrinityState, _delta_time: f64) {
        self.frame += 1;

        for rule in &mut self.rules {
            if rule.evaluate(state, self.frame) {
                log::trace!("Rule '{}' fired", rule.name);
                rule.last_fired = self.frame;
                Self::execute_action(&rule.action, geometry);
            }
        }
    }

    /// Execute a single action
    fn execute_action(action: &RuleAction, geometry: &mut GeometryCore) {
        match action {
            RuleAction::SetMode(mode) => {
                geometry.set_mode(*mode);
            }
            RuleAction::ApplyRotation { plane, angle } => {
                let rotation = match plane {
                    RotationPlane::XY => Rotation4D::simple_xy(*angle),
                    RotationPlane::XZ => Rotation4D::simple_xz(*angle),
                    RotationPlane::XW => Rotation4D::simple_xw(*angle),
                    RotationPlane::YZ => Rotation4D::simple_yz(*angle),
                    RotationPlane::YW => Rotation4D::simple_yw(*angle),
                    RotationPlane::ZW => Rotation4D::simple_zw(*angle),
                };
                geometry.cell24_mut().rotate(rotation);
            }
            RuleAction::SetRotationSpeed(speeds) => {
                geometry.set_rotation_speed(*speeds);
            }
            RuleAction::ScaleRotationSpeed(factor) => {
                // This would need access to current speeds
                log::debug!("Would scale rotation by {}", factor);
            }
            RuleAction::Log(msg) => {
                log::info!("Rule log: {}", msg);
            }
            RuleAction::ForceSynthesisCheck => {
                let _ = geometry.check_synthesis();
            }
            RuleAction::Reset => {
                geometry.cell24_mut().set_rotation(Rotation4D::identity());
            }
            RuleAction::Sequence(actions) => {
                for a in actions {
                    Self::execute_action(a, geometry);
                }
            }
            RuleAction::None => {}
        }
    }

    /// Get current rule count
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// List all rules
    pub fn rules(&self) -> &[CognitiveRule] {
        &self.rules
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_creation() {
        let rule = CognitiveRule::new(
            "test",
            RuleCondition::Always,
            RuleAction::None,
        );
        assert_eq!(rule.name, "test");
        assert!(rule.enabled);
    }

    #[test]
    fn test_rule_engine_default_rules() {
        let engine = RuleEngine::new();
        assert!(engine.rule_count() >= 3);
    }

    #[test]
    fn test_compound_conditions() {
        let cond = RuleCondition::All(vec![
            RuleCondition::TensionAbove(0.5),
            RuleCondition::CoherenceAbove(0.5),
        ]);

        let rule = CognitiveRule::new("compound", cond, RuleAction::None);
        assert_eq!(rule.name, "compound");
    }
}
