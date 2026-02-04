# PPP Adapter Usage (Market-First)

## Purpose
Provide ready-to-use integration guidance for PPP adapters with common app stacks (Node, browser, and service workers). The focus is market/asset quote ingestion via `MarketQuoteAdapter`, while keeping `HemocOddsAdapter` as a legacy path.

---

## MarketQuoteAdapter (Primary)

### What it emits
`MarketQuoteAdapter` normalizes market quotes into `RawApiTick`:
- `price`: last (or mid)
- `bid`, `ask`
- `volume`
- `channels`: `[mid, spread, imbalance, bidSize, askSize, last, volume]`

### Node / Server (REST / WebSocket)
```ts
import { MarketQuoteAdapter, StereoscopicFeed } from '../src/lib';

const feed = new StereoscopicFeed({
  timeBinder: { latencyBuffer: 50 },
  smoothing: true
});

const adapter = new MarketQuoteAdapter({
  source: 'alpaca-rest',
  onTick: tick => feed.ingestTick(tick)
});

adapter.ingestQuote({
  symbol: 'AAPL',
  bid: 225.41,
  ask: 225.44,
  last: 225.43,
  volume: 1200,
  bidSize: 500,
  askSize: 700,
  timestamp: Date.now()
});
```

### Adapter Bridge Helper
```ts
import { MarketQuoteAdapter, StereoscopicFeed, connectAdapterToFeed } from '../src/lib';

const feed = new StereoscopicFeed({ timeBinder: { latencyBuffer: 50 } });
const adapter = new MarketQuoteAdapter({ source: 'alpaca-rest' });

const bridge = connectAdapterToFeed(adapter, feed);

// later, when shutting down:
bridge.stop();
```

### Browser (WebSocket)
```ts
import { MarketQuoteAdapter, StereoscopicFeed } from '../src/lib';

const feed = new StereoscopicFeed({ timeBinder: { latencyBuffer: 50 } });
const adapter = new MarketQuoteAdapter({
  source: 'polygon-ws',
  onTick: tick => feed.ingestTick(tick)
});

const ws = new WebSocket('wss://example.com/quotes');
ws.onmessage = event => {
  const quote = JSON.parse(event.data);
  adapter.ingestQuote({
    symbol: quote.symbol,
    bid: quote.bid,
    ask: quote.ask,
    last: quote.last,
    volume: quote.volume,
    bidSize: quote.bidSize,
    askSize: quote.askSize,
    timestamp: quote.timestamp
  });
};
```

---

## HemocOddsAdapter (Legacy)
For backwards compatibility with the HEMOC odds demo.

```ts
import { HemocOddsAdapter } from '../src/lib';

const adapter = new HemocOddsAdapter({
  onTick: tick => console.log('normalized', tick)
});

adapter.ingestOdds({
  pinnacle: [-150, 140],
  draftkings: [-145, 125],
  fanduel: [-155, 135],
  betmgm: [-140, 120],
  timestamp: Date.now()
});
```

---

## Integration Notes
- **TimeBinder** expects timestamps in milliseconds; use `Date.now()` when upstream data lacks a timestamp.
- Use `StereoscopicFeed` for phase-locked frames and crosshair synchronization.
- For production, prefer market feeds (stocks/crypto/forex) to maintain asset-centric focus.
