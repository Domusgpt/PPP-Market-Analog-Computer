import { connectAdapterToFeed } from '../src/lib/adapters/AdapterBridge.ts';
import { MarketQuoteAdapter } from '../src/lib/adapters/MarketQuoteAdapter.ts';
import { StereoscopicFeed } from '../src/lib/fusion/StereoscopicFeed.ts';

const feed = new StereoscopicFeed({ timeBinder: { latencyBuffer: 50 }, smoothing: true });
const adapter = new MarketQuoteAdapter({ source: 'sample' });
const bridge = connectAdapterToFeed(adapter, feed);

adapter.ingestQuote({
  symbol: 'DEMO',
  bid: 100.1,
  ask: 100.3,
  last: 100.2,
  volume: 500,
  bidSize: 200,
  askSize: 300,
  timestamp: Date.now()
});

const frame = feed.frame();
console.log('Stereoscopic frame snapshot:', {
  timestamp: frame.timestamp,
  price: frame.leftEye.price,
  rotation: frame.rightEye.smoothedRotation
});

bridge.stop();
