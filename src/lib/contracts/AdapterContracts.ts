import type { RawApiTick } from '../fusion/StereoscopicFeed';
import type { TimeBinderConfig } from '../temporal/TimeBinder';

/**
 * PPP Adapter Contract
 *
 * Adapters normalize external sources into RawApiTick and report ingest metrics.
 */
export interface PPPAdapter {
  connect(): Promise<void> | void;
  disconnect(): Promise<void> | void;
  ingest?(tick: RawApiTick): void;
  onTick?(callback: (tick: RawApiTick) => void): () => void;
  metrics?(): Record<string, number | string | boolean>;
}

/**
 * PPP Core configuration boundary
 */
export interface PPPCoreConfig {
  timeBinder?: Partial<TimeBinderConfig>;
  smoothing?: boolean;
}
