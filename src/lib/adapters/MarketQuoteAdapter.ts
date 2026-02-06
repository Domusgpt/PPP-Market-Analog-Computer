import type { PPPAdapter } from '../contracts/AdapterContracts';
import type { RawApiTick } from '../fusion/StereoscopicFeed';

export interface MarketQuoteInput {
  symbol: string;
  bid?: number;
  ask?: number;
  last?: number;
  volume?: number;
  bidSize?: number;
  askSize?: number;
  timestamp?: number;
}

export interface MarketQuoteAdapterOptions {
  onTick?: (tick: RawApiTick) => void;
  source?: string;
}

export class MarketQuoteAdapter implements PPPAdapter {
  private onTickCallback?: (tick: RawApiTick) => void;
  private readonly source: string;
  private ticks = 0;
  private lastTimestamp: number | null = null;

  constructor(options: MarketQuoteAdapterOptions = {}) {
    this.onTickCallback = options.onTick;
    this.source = options.source ?? 'market-quote';
  }

  connect(): void {
    // No-op: adapter is push-driven via ingestQuote().
  }

  disconnect(): void {
    // No-op: adapter is stateless.
  }

  ingestQuote(input: MarketQuoteInput): RawApiTick {
    const bid = input.bid ?? input.last ?? 0;
    const ask = input.ask ?? input.last ?? bid;
    const mid = (bid + ask) / 2;
    const spread = ask - bid;
    const volume = input.volume ?? 0;
    const bidSize = input.bidSize ?? 0;
    const askSize = input.askSize ?? 0;
    const imbalance = (bidSize + askSize) > 0 ? (bidSize - askSize) / (bidSize + askSize) : 0;

    const tick: RawApiTick = {
      symbol: input.symbol,
      price: input.last ?? mid,
      bid,
      ask,
      volume,
      timestamp: input.timestamp,
      channels: [
        mid,
        spread,
        imbalance,
        bidSize,
        askSize,
        input.last ?? mid,
        volume
      ]
    };

    this.ticks += 1;
    this.lastTimestamp = input.timestamp ?? null;
    this.onTickCallback?.(tick);
    return tick;
  }

  onTick(callback: (tick: RawApiTick) => void): () => void {
    this.onTickCallback = callback;
    return () => {
      if (this.onTickCallback === callback) {
        this.onTickCallback = undefined;
      }
    };
  }

  metrics(): Record<string, number | string | boolean> {
    return {
      source: this.source,
      adapter: 'MarketQuoteAdapter',
      ticks: this.ticks,
      lastTimestamp: this.lastTimestamp ?? 'n/a'
    };
  }
}
