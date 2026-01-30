import type { PPPAdapter } from '../contracts/AdapterContracts';
import type { RawApiTick } from '../fusion/StereoscopicFeed';

export interface HemocOddsInput {
  pinnacle: [number, number];
  draftkings: [number, number];
  fanduel: [number, number];
  betmgm: [number, number];
  timestamp?: number;
}

export interface HemocOddsAdapterOptions {
  onTick?: (tick: RawApiTick) => void;
  source?: string;
}

export class HemocOddsAdapter implements PPPAdapter {
  private onTickCallback?: (tick: RawApiTick) => void;
  private readonly source: string;

  constructor(options: HemocOddsAdapterOptions = {}) {
    this.onTickCallback = options.onTick;
    this.source = options.source ?? 'hemoc-odds';
  }

  connect(): void {
    // No-op: adapter is push-driven via ingestOdds().
  }

  disconnect(): void {
    // No-op: adapter is stateless.
  }

  ingestOdds(input: HemocOddsInput): RawApiTick {
    const probs = {
      pinnacle: [this.americanToProb(input.pinnacle[0]), this.americanToProb(input.pinnacle[1])],
      draftkings: [this.americanToProb(input.draftkings[0]), this.americanToProb(input.draftkings[1])],
      fanduel: [this.americanToProb(input.fanduel[0]), this.americanToProb(input.fanduel[1])],
      betmgm: [this.americanToProb(input.betmgm[0]), this.americanToProb(input.betmgm[1])]
    };

    const consensusHome = this.average([probs.pinnacle[0], probs.draftkings[0], probs.fanduel[0], probs.betmgm[0]]);
    const consensusAway = this.average([probs.pinnacle[1], probs.draftkings[1], probs.fanduel[1], probs.betmgm[1]]);

    const vigs = {
      pinnacle: probs.pinnacle[0] + probs.pinnacle[1] - 1,
      draftkings: probs.draftkings[0] + probs.draftkings[1] - 1,
      fanduel: probs.fanduel[0] + probs.fanduel[1] - 1,
      betmgm: probs.betmgm[0] + probs.betmgm[1] - 1
    };

    const maxDeviation = Math.max(
      Math.abs(probs.pinnacle[0] - consensusHome),
      Math.abs(probs.draftkings[0] - consensusHome),
      Math.abs(probs.fanduel[0] - consensusHome),
      Math.abs(probs.betmgm[0] - consensusHome)
    );

    const avgVig = this.average(Object.values(vigs));
    const edgeScore = maxDeviation / (avgVig + 0.01);

    const tick: RawApiTick = {
      symbol: this.source,
      price: consensusHome,
      bid: consensusHome,
      ask: consensusAway,
      volume: avgVig,
      timestamp: input.timestamp,
      channels: [
        consensusHome,
        consensusAway,
        maxDeviation,
        avgVig,
        edgeScore,
        vigs.pinnacle,
        vigs.draftkings,
        vigs.fanduel,
        vigs.betmgm
      ]
    };

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
    return { source: this.source, adapter: 'HemocOddsAdapter' };
  }

  private americanToProb(odds: number): number {
    return odds > 0 ? 100 / (odds + 100) : Math.abs(odds) / (Math.abs(odds) + 100);
  }

  private average(values: number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }
}
