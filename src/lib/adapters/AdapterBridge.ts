import type { PPPAdapter } from '../contracts/AdapterContracts';
import type { RawApiTick } from '../fusion/StereoscopicFeed';
import { StereoscopicFeed } from '../fusion/StereoscopicFeed';

export interface AdapterBridge {
  stop: () => void;
}

/**
 * Connect a PPPAdapter to a StereoscopicFeed.
 *
 * Adapters that expose onTick will stream ticks into the feed automatically.
 */
export function connectAdapterToFeed(
  adapter: PPPAdapter,
  feed: StereoscopicFeed
): AdapterBridge {
  const handler = (tick: RawApiTick) => feed.ingest(tick);
  const unsubscribe = adapter.onTick ? adapter.onTick(handler) : () => {};

  return {
    stop: () => {
      unsubscribe();
      adapter.disconnect();
    }
  };
}
