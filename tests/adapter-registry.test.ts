import assert from 'node:assert/strict';
import { test } from 'node:test';

import { AdapterRegistry } from '../src/lib/adapters/AdapterRegistry';
import { MarketQuoteAdapter } from '../src/lib/adapters/MarketQuoteAdapter';

test('AdapterRegistry stores and lists adapters', () => {
  const registry = new AdapterRegistry();
  registry.register('primary', new MarketQuoteAdapter({ source: 'alpaca' }));

  assert.equal(registry.list().length, 1);
  assert.ok(registry.get('primary'));

  let visited = '';
  registry.forEach((_adapter, name) => {
    visited = name;
  });
  assert.equal(visited, 'primary');

  registry.unregister('primary');
  assert.equal(registry.list().length, 0);

  registry.clear();
  assert.equal(registry.list().length, 0);
});
