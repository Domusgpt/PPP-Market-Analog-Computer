import assert from 'node:assert/strict';
import { test } from 'node:test';

import { HemocOddsAdapter, MarketQuoteAdapter } from '../src/lib/adapters';

test('MarketQuoteAdapter maps quotes into RawApiTick channels', () => {
  const adapter = new MarketQuoteAdapter();
  const tick = adapter.ingestQuote({
    symbol: 'MSFT',
    bid: 100,
    ask: 102,
    last: 101,
    volume: 250,
    bidSize: 40,
    askSize: 60,
    timestamp: 1700000000000
  });

  assert.equal(tick.symbol, 'MSFT');
  assert.equal(tick.price, 101);
  assert.equal(tick.bid, 100);
  assert.equal(tick.ask, 102);
  assert.equal(tick.volume, 250);
  assert.equal(tick.timestamp, 1700000000000);
  assert.ok(Array.isArray(tick.channels));
  assert.equal(tick.channels?.length, 7);
  assert.equal(tick.channels?.[0], 101);
  assert.equal(tick.channels?.[1], 2);
  assert.equal(tick.channels?.[2], (40 - 60) / (40 + 60));

  const metrics = adapter.metrics();
  assert.equal(metrics.adapter, 'MarketQuoteAdapter');
  assert.equal(metrics.ticks, 1);
});

test('MarketQuoteAdapter maps records with field maps', () => {
  const adapter = new MarketQuoteAdapter();
  const tick = adapter.ingestFrom(
    {
      symbol: 'TSLA',
      b: '240.1',
      a: '240.3',
      last: 240.2,
      volume: '9000',
      bidSize: 10,
      askSize: 12,
      ts: 1700000001000
    },
    {
      symbol: 'symbol',
      bid: 'b',
      ask: 'a',
      last: 'last',
      volume: 'volume',
      bidSize: 'bidSize',
      askSize: 'askSize',
      timestamp: 'ts'
    }
  );

  assert.equal(tick.symbol, 'TSLA');
  assert.equal(tick.bid, 240.1);
  assert.equal(tick.ask, 240.3);
  assert.equal(tick.price, 240.2);
  assert.equal(tick.volume, 9000);
  assert.equal(tick.timestamp, 1700000001000);
});

test('HemocOddsAdapter emits consensus-based ticks', () => {
  const adapter = new HemocOddsAdapter();
  const tick = adapter.ingestOdds({
    pinnacle: [-150, 140],
    draftkings: [-145, 125],
    fanduel: [-155, 135],
    betmgm: [-140, 120],
    timestamp: 1700000000000
  });

  assert.equal(tick.symbol, 'hemoc-odds');
  assert.equal(tick.timestamp, 1700000000000);
  assert.ok(Array.isArray(tick.channels));
  assert.equal(tick.channels?.length, 9);
  assert.ok(tick.price > 0 && tick.price < 1);
  assert.ok(tick.bid > 0 && tick.bid < 1);
  assert.ok(tick.ask > 0 && tick.ask < 1);

  const metrics = adapter.metrics();
  assert.equal(metrics.adapter, 'HemocOddsAdapter');
  assert.equal(metrics.ticks, 1);
});
