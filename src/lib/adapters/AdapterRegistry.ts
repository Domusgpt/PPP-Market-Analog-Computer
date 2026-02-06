import type { PPPAdapter } from '../contracts/AdapterContracts';

export class AdapterRegistry {
  private adapters: Map<string, PPPAdapter> = new Map();

  register(name: string, adapter: PPPAdapter): void {
    this.adapters.set(name, adapter);
  }

  get(name: string): PPPAdapter | undefined {
    return this.adapters.get(name);
  }

  list(): string[] {
    return Array.from(this.adapters.keys());
  }

  clear(): void {
    this.adapters.clear();
  }
}
