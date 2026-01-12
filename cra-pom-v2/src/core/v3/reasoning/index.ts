/**
 * PPP v3 Reasoning Module
 *
 * Provides verified reasoning capabilities with audit trails.
 */

export {
  VerifiedReasoner,
  getVerifiedReasoner,
  resetVerifiedReasoner,
  type ReasoningStep,
  type ReasoningOperation,
  type ReasoningSession,
  type SignedReasoningStep,
  type Conclusion,
  type SignedConclusion,
  type ReasoningQuery,
  type ReasoningResult,
} from './verified-reasoner';
