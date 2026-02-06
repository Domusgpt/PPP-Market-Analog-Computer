import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { PPPAdapter } from '../src/lib/contracts/AdapterContracts';
import type { RawApiTick } from '../src/lib/fusion/StereoscopicFeed';
import { StereoscopicFeed } from '../src/lib/fusion/StereoscopicFeed';
import { connectAdapterToFeed } from '../src/lib/adapters/AdapterBridge';

class TestAdapter implements PPPAdapter {
  private callback?: (tick: RawApiTick) => void;
  public disconnected = false;

  connect(): void {}

  disconnect(): void {
    this.disconnected = true;
  }

  onTick(callback: (tick: RawApiTick) => void): () => void {
    this.callback = callback;
    return () => {
      if (this.callback === callback) {
        this.callback = undefined;
      }
    };
  }

  emit(tick: RawApiTick): void {
    this.callback?.(tick);
  }
}

test('connectAdapterToFeed forwards ticks and disconnects', () => {
  const adapter = new TestAdapter();
  const feed = new StereoscopicFeed({ timeBinder: { latencyBuffer: 0 } });
  const bridge = connectAdapterToFeed(adapter, feed);

  adapter.emit({
    symbol: 'TEST',
    price: 1,
    bid: 1,
    ask: 1,
    volume: 1,
    timestamp: Date.now()
  });

  const metrics = feed.getMetrics();
  assert.equal(metrics.ticksReceived, 1);

  bridge.stop();
  assert.equal(adapter.disconnected, true);
});
