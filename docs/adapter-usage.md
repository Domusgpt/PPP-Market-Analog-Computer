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

#### Channel mapping
| Index | Field | Description |
| --- | --- | --- |
| 0 | mid | Midpoint between bid and ask. |
| 1 | spread | `ask - bid` spread. |
| 2 | imbalance | `(bidSize - askSize) / (bidSize + askSize)` when sizes are present. |
| 3 | bidSize | Bid size (defaults to 0). |
| 4 | askSize | Ask size (defaults to 0). |
| 5 | last | Last trade price (falls back to mid). |
| 6 | volume | Volume (defaults to 0). |

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

### Mapping from Provider Fields
```ts
import { MarketQuoteAdapter } from '../src/lib';

const adapter = new MarketQuoteAdapter();
const tick = adapter.ingestFrom(
  { symbol: 'TSLA', b: '240.1', a: '240.3', last: 240.2, volume: '9000', ts: 1700000001000 },
  { symbol: 'symbol', bid: 'b', ask: 'a', last: 'last', volume: 'volume', timestamp: 'ts' }
);
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
- Adapter `metrics()` reports adapter name, ticks ingested, and last timestamp for simple health checks.
- Use `resetMetrics()` before test runs or when rotating sessions to clear counters.
