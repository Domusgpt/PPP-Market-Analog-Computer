/**
 * PPP v3 Visualization Bridge
 *
 * Connects the verified reasoning engine with the HypercubeCore visualization.
 *
 * This bridge:
 * - Translates reasoning steps into visual representations
 * - Provides real-time updates for the polytope visualization
 * - Shows audit chain status
 * - Visualizes concept relationships in the 24-cell geometry
 */

import type {
  ReasoningSession,
  SignedReasoningStep,
  SignedConclusion,
  ReasoningOperation,
} from '../reasoning';
import type { ConceptStore, StoredConcept } from '../embeddings';
import type { VerificationClient } from '../isolation';

// ============================================================================
// Types
// ============================================================================

/**
 * Visual state for rendering
 */
export interface VisualizationState {
  /** Current session info */
  session: {
    id: string;
    query: string;
    stepCount: number;
    isActive: boolean;
  } | null;

  /** Current step being visualized */
  currentStep: {
    number: number;
    operation: ReasoningOperation;
    description: string;
    confidence: number;
    signed: boolean;
  } | null;

  /** Concept nodes for the visualization */
  conceptNodes: ConceptNode[];

  /** Edges between concepts */
  conceptEdges: ConceptEdge[];

  /** Audit chain status */
  auditStatus: {
    valid: boolean;
    chainLength: number;
    lastHash: string;
  };

  /** Grounding status */
  grounding: {
    semantic: boolean;
    source: string;
    warning?: string;
  };
}

/**
 * A concept node for visualization
 */
export interface ConceptNode {
  id: string;
  name: string;
  /** Position in 4D space (will be projected) */
  position4D: [number, number, number, number];
  /** Role in current reasoning */
  role: 'input' | 'output' | 'intermediate' | 'retrieved' | 'inactive';
  /** Confidence/strength */
  strength: number;
  /** Is this semantically grounded? */
  semantic: boolean;
}

/**
 * An edge between concepts
 */
export interface ConceptEdge {
  from: string;
  to: string;
  /** Type of relationship */
  type: 'similarity' | 'inference' | 'binding' | 'composition';
  /** Strength of relationship */
  strength: number;
}

/**
 * Event emitted by the bridge
 */
export type VisualizationEvent =
  | { type: 'SESSION_STARTED'; session: ReasoningSession }
  | { type: 'STEP_ADDED'; step: SignedReasoningStep }
  | { type: 'CONCLUSION_REACHED'; conclusion: SignedConclusion }
  | { type: 'SESSION_ENDED'; summary: string }
  | { type: 'CHAIN_VALIDATED'; valid: boolean }
  | { type: 'GROUNDING_CHANGED'; semantic: boolean; source: string }
  | { type: 'STATE_UPDATED'; state: VisualizationState };

type EventListener = (event: VisualizationEvent) => void;

// ============================================================================
// Implementation
// ============================================================================

export class VisualizationBridge {
  private state: VisualizationState;
  private listeners: Set<EventListener> = new Set();
  private conceptStore: ConceptStore | null = null;
  private verificationClient: VerificationClient | null = null;

  constructor() {
    this.state = this.createEmptyState();
  }

  /**
   * Connect to concept store and verification client
   */
  connect(conceptStore: ConceptStore, verificationClient: VerificationClient): void {
    this.conceptStore = conceptStore;
    this.verificationClient = verificationClient;

    // Update grounding status
    const grounding = conceptStore.getGroundingStatus();
    this.state.grounding = {
      semantic: grounding.embeddingSource !== 'deterministic_fallback',
      source: grounding.embeddingSource,
      warning:
        grounding.embeddingSource === 'deterministic_fallback'
          ? 'Using non-semantic fallback embeddings'
          : undefined,
    };

    this.emitStateUpdate();
  }

  /**
   * Get current visualization state
   */
  getState(): VisualizationState {
    return { ...this.state };
  }

  /**
   * Subscribe to visualization events
   */
  subscribe(listener: EventListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Handle session start
   */
  onSessionStart(session: ReasoningSession): void {
    this.state.session = {
      id: session.sessionId,
      query: session.initialQuery,
      stepCount: 0,
      isActive: true,
    };

    this.state.currentStep = null;
    this.state.conceptNodes = [];
    this.state.conceptEdges = [];

    this.emit({ type: 'SESSION_STARTED', session });
    this.emitStateUpdate();
  }

  /**
   * Handle reasoning step
   */
  onReasoningStep(step: SignedReasoningStep): void {
    if (this.state.session) {
      this.state.session.stepCount++;
    }

    this.state.currentStep = {
      number: step.payload.stepNumber,
      operation: step.payload.operation,
      description: step.payload.description,
      confidence: step.payload.confidence,
      signed: true,
    };

    // Update concept nodes based on operation
    this.updateConceptsFromStep(step);

    this.emit({ type: 'STEP_ADDED', step });
    this.emitStateUpdate();
  }

  /**
   * Handle conclusion
   */
  onConclusion(conclusion: SignedConclusion): void {
    this.state.currentStep = {
      number: -1,
      operation: 'CONCLUSION',
      description: conclusion.payload.statement,
      confidence: conclusion.payload.confidence,
      signed: true,
    };

    this.emit({ type: 'CONCLUSION_REACHED', conclusion });
    this.emitStateUpdate();
  }

  /**
   * Handle session end
   */
  onSessionEnd(summary: string): void {
    if (this.state.session) {
      this.state.session.isActive = false;
    }

    this.emit({ type: 'SESSION_ENDED', summary });
    this.emitStateUpdate();
  }

  /**
   * Update audit status
   */
  async updateAuditStatus(): Promise<void> {
    if (!this.verificationClient) return;

    const validation = await this.verificationClient.validateChain();
    const exported = await this.verificationClient.exportChain();

    this.state.auditStatus = {
      valid: validation.valid,
      chainLength: validation.length,
      lastHash: exported.headHash.substring(0, 16) + '...',
    };

    this.emit({ type: 'CHAIN_VALIDATED', valid: validation.valid });
    this.emitStateUpdate();
  }

  /**
   * Add a concept to the visualization
   */
  addConceptNode(concept: StoredConcept, role: ConceptNode['role']): void {
    // Check if already exists
    const existing = this.state.conceptNodes.find((n) => n.id === concept.id);
    if (existing) {
      existing.role = role;
      existing.strength = 1.0;
      this.emitStateUpdate();
      return;
    }

    // Generate 4D position from embedding
    const position4D = this.embeddingTo4D(concept.embedding);

    this.state.conceptNodes.push({
      id: concept.id,
      name: concept.name,
      position4D,
      role,
      strength: 1.0,
      semantic: concept.embeddingSource !== 'deterministic_fallback',
    });

    this.emitStateUpdate();
  }

  /**
   * Add an edge between concepts
   */
  addConceptEdge(
    fromId: string,
    toId: string,
    type: ConceptEdge['type'],
    strength: number
  ): void {
    // Check if edge already exists
    const existing = this.state.conceptEdges.find(
      (e) => e.from === fromId && e.to === toId
    );
    if (existing) {
      existing.strength = strength;
      this.emitStateUpdate();
      return;
    }

    this.state.conceptEdges.push({
      from: fromId,
      to: toId,
      type,
      strength,
    });

    this.emitStateUpdate();
  }

  /**
   * Fade inactive concepts
   */
  fadeInactiveConcepts(decayRate = 0.1): void {
    for (const node of this.state.conceptNodes) {
      if (node.role === 'inactive') {
        node.strength = Math.max(0, node.strength - decayRate);
      }
    }

    // Remove completely faded nodes
    this.state.conceptNodes = this.state.conceptNodes.filter((n) => n.strength > 0);

    // Remove edges with missing nodes
    const nodeIds = new Set(this.state.conceptNodes.map((n) => n.id));
    this.state.conceptEdges = this.state.conceptEdges.filter(
      (e) => nodeIds.has(e.from) && nodeIds.has(e.to)
    );

    this.emitStateUpdate();
  }

  /**
   * Reset visualization state
   */
  reset(): void {
    this.state = this.createEmptyState();
    this.emitStateUpdate();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private createEmptyState(): VisualizationState {
    return {
      session: null,
      currentStep: null,
      conceptNodes: [],
      conceptEdges: [],
      auditStatus: {
        valid: true,
        chainLength: 0,
        lastHash: '',
      },
      grounding: {
        semantic: false,
        source: 'none',
      },
    };
  }

  private emit(event: VisualizationEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (err) {
        console.error('Visualization listener error:', err);
      }
    }
  }

  private emitStateUpdate(): void {
    this.emit({ type: 'STATE_UPDATED', state: this.getState() });
  }

  private updateConceptsFromStep(step: SignedReasoningStep): void {
    if (!this.conceptStore) return;

    // Mark all current nodes as inactive
    for (const node of this.state.conceptNodes) {
      if (node.role !== 'inactive') {
        node.role = 'inactive';
      }
    }

    // Add input concepts
    for (const input of step.payload.inputs) {
      const concept = this.conceptStore.getConceptByName(input);
      if (concept) {
        this.addConceptNode(concept, 'input');
      }
    }

    // Add output concepts
    for (const output of step.payload.outputs) {
      // Try to parse output format like "Found: name - description"
      const match = output.match(/^(?:Found|Nearest):\s*(\w+)/);
      if (match) {
        const concept = this.conceptStore.getConceptByName(match[1]);
        if (concept) {
          this.addConceptNode(concept, 'output');
        }
      }
    }

    // Add edges based on operation type
    if (step.payload.operation === 'SIMILARITY_QUERY') {
      // Connect query to results
      const inputNodes = this.state.conceptNodes.filter((n) => n.role === 'input');
      const outputNodes = this.state.conceptNodes.filter((n) => n.role === 'output');

      for (const input of inputNodes) {
        for (const output of outputNodes) {
          this.addConceptEdge(input.id, output.id, 'similarity', step.payload.confidence);
        }
      }
    } else if (step.payload.operation === 'INFERENCE') {
      // Connect premises to conclusion
      const inputNodes = this.state.conceptNodes.filter((n) => n.role === 'input');
      const outputNodes = this.state.conceptNodes.filter((n) => n.role === 'output');

      for (const input of inputNodes) {
        for (const output of outputNodes) {
          this.addConceptEdge(input.id, output.id, 'inference', step.payload.confidence);
        }
      }
    }
  }

  /**
   * Project embedding to 4D position
   *
   * Uses PCA-like dimensionality reduction to map high-dimensional
   * embedding to 4D space for visualization on the 24-cell.
   */
  private embeddingTo4D(embedding: Float32Array): [number, number, number, number] {
    if (embedding.length < 4) {
      return [0, 0, 0, 0];
    }

    // Simple projection: use first 4 principal components
    // (In a full implementation, we'd use actual PCA)

    // Divide embedding into 4 chunks and average each
    const chunkSize = Math.floor(embedding.length / 4);
    const result: [number, number, number, number] = [0, 0, 0, 0];

    for (let i = 0; i < 4; i++) {
      let sum = 0;
      for (let j = 0; j < chunkSize; j++) {
        sum += embedding[i * chunkSize + j];
      }
      result[i] = sum / chunkSize;
    }

    // Normalize to unit hypersphere
    const norm = Math.sqrt(result.reduce((s, v) => s + v * v, 0));
    if (norm > 0) {
      for (let i = 0; i < 4; i++) {
        result[i] /= norm;
      }
    }

    return result;
  }
}

// ============================================================================
// React Hook
// ============================================================================

/**
 * Hook state for React components
 */
export interface UseVisualizationBridgeState {
  state: VisualizationState;
  isConnected: boolean;
}

/**
 * Create a visualization bridge instance
 */
export function createVisualizationBridge(): VisualizationBridge {
  return new VisualizationBridge();
}

// ============================================================================
// Singleton
// ============================================================================

let globalBridge: VisualizationBridge | null = null;

export function getVisualizationBridge(): VisualizationBridge {
  if (!globalBridge) {
    globalBridge = new VisualizationBridge();
  }
  return globalBridge;
}

export function resetVisualizationBridge(): void {
  globalBridge = null;
}
