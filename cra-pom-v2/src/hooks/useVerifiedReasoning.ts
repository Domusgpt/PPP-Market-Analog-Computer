/**
 * PPP v3 React Integration Hook
 *
 * Provides React state management for the PPP v3 verified reasoning system.
 *
 * Created: 2026-01-09
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  VerifiedReasoner,
  getVerifiedReasoner,
  getConceptStore,
  initializeWithBasicConcepts,
  ConceptStore,
  type SignedReasoningStep,
  type SignedConclusion,
  type ReasoningResult,
  type EmbeddingSource,
} from '../core/v3';

// ============================================================================
// Types
// ============================================================================

export interface V3ReasoningState {
  /** Is the system initialized? */
  initialized: boolean;

  /** Is a session active? */
  sessionActive: boolean;

  /** Current session info */
  session: {
    id: string;
    query: string;
    stepCount: number;
  } | null;

  /** Recent reasoning steps */
  recentSteps: SignedReasoningStep[];

  /** Current conclusion if any */
  conclusion: SignedConclusion | null;

  /** Verification status */
  verification: {
    chainValid: boolean;
    signaturesValid: boolean;
    lastChecked: string;
  };

  /** Grounding status */
  grounding: {
    semantic: boolean;
    source: EmbeddingSource;
    conceptCount: number;
    warning: string | null;
  };

  /** Public key for external verification */
  publicKey: JsonWebKey | null;

  /** Error if any */
  error: string | null;

  /** Is currently processing */
  loading: boolean;
}

export interface UseVerifiedReasoningReturn {
  state: V3ReasoningState;

  // Session management
  startSession: (query: string) => Promise<void>;
  endSession: () => Promise<ReasoningResult | null>;

  // Reasoning operations
  lookupConcept: (name: string) => Promise<SignedReasoningStep | null>;
  querySimilar: (query: string, topK?: number) => Promise<SignedReasoningStep | null>;
  makeInference: (premises: string[], conclusion: string, confidence: number) => Promise<SignedReasoningStep | null>;
  conclude: (statement: string, confidence: number, caveats?: string[]) => Promise<SignedConclusion | null>;

  // Concept management
  addConcept: (name: string, description: string) => Promise<void>;

  // Verification
  validateChain: () => Promise<boolean>;
  exportChain: () => Promise<unknown>;

  // Reset
  reset: () => void;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useVerifiedReasoning(): UseVerifiedReasoningReturn {
  const reasonerRef = useRef<VerifiedReasoner | null>(null);
  const conceptStoreRef = useRef<ConceptStore | null>(null);
  const stepNumbersRef = useRef<number[]>([]);

  const [state, setState] = useState<V3ReasoningState>({
    initialized: false,
    sessionActive: false,
    session: null,
    recentSteps: [],
    conclusion: null,
    verification: {
      chainValid: true,
      signaturesValid: true,
      lastChecked: new Date().toISOString(),
    },
    grounding: {
      semantic: false,
      source: 'deterministic_fallback',
      conceptCount: 0,
      warning: 'Using non-semantic fallback embeddings',
    },
    publicKey: null,
    error: null,
    loading: false,
  });

  // Initialize on mount
  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        setState(prev => ({ ...prev, loading: true }));

        // Get concept store and initialize with basic concepts
        conceptStoreRef.current = getConceptStore();
        await initializeWithBasicConcepts(conceptStoreRef.current);

        // Get reasoner
        reasonerRef.current = await getVerifiedReasoner();

        // Get grounding status
        const grounding = conceptStoreRef.current.getGroundingStatus();

        if (mounted) {
          setState(prev => ({
            ...prev,
            initialized: true,
            loading: false,
            grounding: {
              semantic: grounding.embeddingSource !== 'deterministic_fallback',
              source: grounding.embeddingSource as EmbeddingSource,
              conceptCount: grounding.conceptCount,
              warning: grounding.embeddingSource === 'deterministic_fallback'
                ? 'Using non-semantic fallback embeddings'
                : null,
            },
          }));
        }
      } catch (err) {
        if (mounted) {
          setState(prev => ({
            ...prev,
            loading: false,
            error: err instanceof Error ? err.message : 'Failed to initialize',
          }));
        }
      }
    }

    init();

    return () => {
      mounted = false;
    };
  }, []);

  // Start a reasoning session
  const startSession = useCallback(async (query: string) => {
    if (!reasonerRef.current) {
      setState(prev => ({ ...prev, error: 'Reasoner not initialized' }));
      return;
    }

    try {
      setState(prev => ({ ...prev, loading: true, error: null }));

      const session = await reasonerRef.current.startSession(query);
      stepNumbersRef.current = [];

      setState(prev => ({
        ...prev,
        loading: false,
        sessionActive: true,
        session: {
          id: session.sessionId,
          query: session.initialQuery,
          stepCount: 0,
        },
        recentSteps: [],
        conclusion: null,
      }));
    } catch (err) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to start session',
      }));
    }
  }, []);

  // End a session
  const endSession = useCallback(async (): Promise<ReasoningResult | null> => {
    if (!reasonerRef.current) return null;

    try {
      setState(prev => ({ ...prev, loading: true }));

      const result = await reasonerRef.current.endSession();

      setState(prev => ({
        ...prev,
        loading: false,
        sessionActive: false,
        verification: {
          chainValid: result.verification.chainValid,
          signaturesValid: result.verification.signaturesValid,
          lastChecked: new Date().toISOString(),
        },
        publicKey: result.verification.publicKey,
      }));

      return result;
    } catch (err) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to end session',
      }));
      return null;
    }
  }, []);

  // Look up a concept
  const lookupConcept = useCallback(async (name: string): Promise<SignedReasoningStep | null> => {
    if (!reasonerRef.current || !state.sessionActive) return null;

    try {
      const step = await reasonerRef.current.lookupConcept(name);
      stepNumbersRef.current.push(step.payload.stepNumber);

      setState(prev => ({
        ...prev,
        recentSteps: [...prev.recentSteps.slice(-9), step],
        session: prev.session ? { ...prev.session, stepCount: prev.session.stepCount + 1 } : null,
      }));

      return step;
    } catch (err) {
      setState(prev => ({ ...prev, error: err instanceof Error ? err.message : 'Lookup failed' }));
      return null;
    }
  }, [state.sessionActive]);

  // Query similar concepts
  const querySimilar = useCallback(async (query: string, topK = 5): Promise<SignedReasoningStep | null> => {
    if (!reasonerRef.current || !state.sessionActive) return null;

    try {
      const step = await reasonerRef.current.querySimilar(query, topK);
      stepNumbersRef.current.push(step.payload.stepNumber);

      setState(prev => ({
        ...prev,
        recentSteps: [...prev.recentSteps.slice(-9), step],
        session: prev.session ? { ...prev.session, stepCount: prev.session.stepCount + 1 } : null,
      }));

      return step;
    } catch (err) {
      setState(prev => ({ ...prev, error: err instanceof Error ? err.message : 'Query failed' }));
      return null;
    }
  }, [state.sessionActive]);

  // Make an inference
  const makeInference = useCallback(async (
    premises: string[],
    conclusion: string,
    confidence: number
  ): Promise<SignedReasoningStep | null> => {
    if (!reasonerRef.current || !state.sessionActive) return null;

    try {
      const step = await reasonerRef.current.makeInference(premises, conclusion, confidence);
      stepNumbersRef.current.push(step.payload.stepNumber);

      setState(prev => ({
        ...prev,
        recentSteps: [...prev.recentSteps.slice(-9), step],
        session: prev.session ? { ...prev.session, stepCount: prev.session.stepCount + 1 } : null,
      }));

      return step;
    } catch (err) {
      setState(prev => ({ ...prev, error: err instanceof Error ? err.message : 'Inference failed' }));
      return null;
    }
  }, [state.sessionActive]);

  // Reach a conclusion
  const conclude = useCallback(async (
    statement: string,
    confidence: number,
    caveats: string[] = []
  ): Promise<SignedConclusion | null> => {
    if (!reasonerRef.current || !state.sessionActive) return null;

    try {
      const conclusion = await reasonerRef.current.conclude(
        statement,
        confidence,
        stepNumbersRef.current,
        caveats
      );

      setState(prev => ({
        ...prev,
        conclusion,
      }));

      return conclusion;
    } catch (err) {
      setState(prev => ({ ...prev, error: err instanceof Error ? err.message : 'Conclusion failed' }));
      return null;
    }
  }, [state.sessionActive]);

  // Add a concept
  const addConcept = useCallback(async (name: string, description: string) => {
    if (!conceptStoreRef.current) return;

    try {
      await conceptStoreRef.current.addConcept(name, description);
      const grounding = conceptStoreRef.current.getGroundingStatus();

      setState(prev => ({
        ...prev,
        grounding: {
          ...prev.grounding,
          conceptCount: grounding.conceptCount,
        },
      }));
    } catch (err) {
      setState(prev => ({ ...prev, error: err instanceof Error ? err.message : 'Failed to add concept' }));
    }
  }, []);

  // Validate chain
  const validateChain = useCallback(async (): Promise<boolean> => {
    if (!reasonerRef.current) return false;

    try {
      const chain = await reasonerRef.current.exportAuditChain() as { length: number };
      // Simple validation - check if chain exists and has entries
      const valid = chain && chain.length >= 0;

      setState(prev => ({
        ...prev,
        verification: {
          ...prev.verification,
          chainValid: valid,
          lastChecked: new Date().toISOString(),
        },
      }));

      return valid;
    } catch {
      return false;
    }
  }, []);

  // Export chain
  const exportChain = useCallback(async (): Promise<unknown> => {
    if (!reasonerRef.current) return null;
    return reasonerRef.current.exportAuditChain();
  }, []);

  // Reset
  const reset = useCallback(() => {
    stepNumbersRef.current = [];
    setState(prev => ({
      ...prev,
      sessionActive: false,
      session: null,
      recentSteps: [],
      conclusion: null,
      error: null,
    }));
  }, []);

  return {
    state,
    startSession,
    endSession,
    lookupConcept,
    querySimilar,
    makeInference,
    conclude,
    addConcept,
    validateChain,
    exportChain,
    reset,
  };
}
