/**
 * TopologyController - Metamorphic manifold selector.
 *
 * Dynamically inflates topology from Simplex -> Hypercube -> 24-Cell
 * based on ambiguity/tension metrics.
 */

import {
    TopologyProvider,
    TopologyStage,
    Vector4D,
    ConvexityResult
} from '../../types/index.js';

import { simplex5 } from './Simplex5.js';
import { hypercube8 } from './Hypercube8.js';
import { lattice24Provider } from './Lattice24Provider.js';

export interface TopologyControllerConfig {
    readonly deflateThreshold: number;
    readonly hypercubeThreshold: number;
    readonly cell24Threshold: number;
    readonly minStageDurationMs: number;
}

export interface TopologyTransitionEvent {
    readonly from: TopologyStage;
    readonly to: TopologyStage;
    readonly reason: string;
    readonly tensionScore: number;
    readonly timestamp: number;
}

const DEFAULT_CONFIG: TopologyControllerConfig = {
    deflateThreshold: 0.2,
    hypercubeThreshold: 0.5,
    cell24Threshold: 0.8,
    minStageDurationMs: 1500
};

export class TopologyController {
    private _stage: TopologyStage = 'SIMPLEX';
    private _active: TopologyProvider = simplex5;
    private _lastTransitionAt = Date.now();
    private _tensionScore = 0;

    constructor(private readonly config: TopologyControllerConfig = DEFAULT_CONFIG) {}

    get stage(): TopologyStage {
        return this._stage;
    }

    get activeLattice(): TopologyProvider {
        return this._active;
    }

    get tensionScore(): number {
        return this._tensionScore;
    }

    updateTension(tensionScore: number): TopologyTransitionEvent | null {
        this._tensionScore = tensionScore;
        return this._maybeTransition();
    }

    evaluate(point: Vector4D, kNearest?: number): ConvexityResult {
        return this._active.checkConvexity(point, kNearest);
    }

    private _maybeTransition(): TopologyTransitionEvent | null {
        const now = Date.now();
        if (now - this._lastTransitionAt < this.config.minStageDurationMs) {
            return null;
        }

        let nextStage = this._stage;
        let reason = '';

        if (this._stage === 'SIMPLEX' && this._tensionScore > this.config.hypercubeThreshold) {
            nextStage = 'HYPERCUBE';
            reason = `tension>${this.config.hypercubeThreshold}`;
        } else if (
            this._stage === 'HYPERCUBE' &&
            this._tensionScore > this.config.cell24Threshold
        ) {
            nextStage = 'CELL24';
            reason = `tension>${this.config.cell24Threshold}`;
        } else if (this._tensionScore < this.config.deflateThreshold) {
            nextStage = 'SIMPLEX';
            reason = `tension<${this.config.deflateThreshold}`;
        }

        if (nextStage === this._stage) {
            return null;
        }

        const previousStage = this._stage;
        this._stage = nextStage;
        this._active = this._stage === 'SIMPLEX'
            ? simplex5
            : this._stage === 'HYPERCUBE'
                ? hypercube8
                : lattice24Provider;

        this._lastTransitionAt = now;

        return {
            from: previousStage,
            to: nextStage,
            reason,
            tensionScore: this._tensionScore,
            timestamp: now
        };
    }

}

export const defaultTopologyController = new TopologyController();
