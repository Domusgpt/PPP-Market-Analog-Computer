import type { PPPAdapter } from '../contracts/AdapterContracts';

export class AdapterRegistry {
  private adapters: Map<string, PPPAdapter> = new Map();

  register(name: string, adapter: PPPAdapter): void {
    this.adapters.set(name, adapter);
  }

  unregister(name: string): void {
    this.adapters.delete(name);
  }

  get(name: string): PPPAdapter | undefined {
    return this.adapters.get(name);
  }

  forEach(callback: (adapter: PPPAdapter, name: string) => void): void {
    this.adapters.forEach((adapter, name) => callback(adapter, name));
  }

  list(): string[] {
    return Array.from(this.adapters.keys());
  }

  clear(): void {
    this.adapters.clear();
  }
}
