/**
 * PPP v3 Hash Chain Tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { CryptoService } from './crypto-service';
import { SignedHashChain, createSignedHashChain } from './hash-chain';

describe('SignedHashChain', () => {
  let crypto: CryptoService;
  let chain: SignedHashChain;

  beforeEach(async () => {
    crypto = new CryptoService();
    await crypto.initialize();
    chain = await createSignedHashChain(crypto);
  });

  describe('initialization', () => {
    it('should create chain with genesis entry', () => {
      expect(chain.length).toBe(1);
    });

    it('should have genesis entry at index 0', () => {
      const genesis = chain.getEntry(0);
      expect(genesis).toBeDefined();
      expect(genesis!.payload.index).toBe(0);
      expect(genesis!.payload.operationType).toBe('GENESIS');
    });

    it('should throw if initialized twice', async () => {
      const newChain = new SignedHashChain(crypto);
      await newChain.initialize();
      await expect(newChain.initialize()).rejects.toThrow('already initialized');
    });
  });

  describe('append', () => {
    it('should add entries to chain', async () => {
      await chain.append('TEST', { value: 1 });
      await chain.append('TEST', { value: 2 });
      expect(chain.length).toBe(3); // genesis + 2
    });

    it('should maintain sequential indices', async () => {
      await chain.append('OP1', {});
      await chain.append('OP2', {});
      await chain.append('OP3', {});

      for (let i = 0; i < chain.length; i++) {
        expect(chain.getEntry(i)!.payload.index).toBe(i);
      }
    });

    it('should link entries via previous hash', async () => {
      const genesis = chain.getEntry(0);
      const entry1 = await chain.append('TEST', { msg: 'first' });
      const entry2 = await chain.append('TEST', { msg: 'second' });

      expect(entry1.payload.previousHash).toBe(genesis!.proof.dataHash);
      expect(entry2.payload.previousHash).toBe(entry1.proof.dataHash);
    });

    it('should update head hash after append', async () => {
      const initialHead = chain.getHeadHash();
      await chain.append('TEST', {});
      const newHead = chain.getHeadHash();
      expect(newHead).not.toBe(initialHead);
    });

    it('should sign each entry', async () => {
      const entry = await chain.append('SIGNED_OP', { data: 'test' });
      expect(entry.proof.signature).toBeDefined();
      expect(entry.proof.publicKeyJwk).toBeDefined();
    });
  });

  describe('validation', () => {
    it('should validate empty chain', async () => {
      const emptyChain = new SignedHashChain(crypto);
      const result = await emptyChain.validate();
      expect(result.valid).toBe(true);
    });

    it('should validate intact chain', async () => {
      await chain.append('OP1', { a: 1 });
      await chain.append('OP2', { b: 2 });
      await chain.append('OP3', { c: 3 });

      const result = await chain.validate();
      expect(result.valid).toBe(true);
      expect(result.details.hashChainIntact).toBe(true);
      expect(result.details.signaturesValid).toBe(true);
      expect(result.details.sequenceCorrect).toBe(true);
    });

    it('should detect tampered entry', async () => {
      await chain.append('TEST', { original: true });

      // Tamper with the entry
      const entry = chain.getEntry(1)!;
      (entry.payload as { data: { tampered?: boolean } }).data.tampered = true;

      const result = await chain.validate();
      expect(result.valid).toBe(false);
      expect(result.brokenAt).toBe(1);
    });
  });

  describe('export and external verification', () => {
    it('should export chain data', async () => {
      await chain.append('TEST', { value: 42 });

      const exported = chain.export();
      expect(exported.version).toBe('3.0.0');
      expect(exported.publicKey).toBeDefined();
      expect(exported.entries.length).toBe(2);
      expect(exported.headHash).toBe(chain.getHeadHash());
    });

    it('should verify exported chain', async () => {
      await chain.append('OP1', { x: 1 });
      await chain.append('OP2', { y: 2 });

      const exported = chain.export();
      const result = await SignedHashChain.verifyExported(exported);

      expect(result.valid).toBe(true);
      expect(result.length).toBe(3);
    });

    it('should detect tampered exported chain', async () => {
      await chain.append('TEST', { amount: 100 });

      const exported = chain.export();
      // Tamper with exported data
      (exported.entries[1].payload as { data: { amount: number } }).data.amount = 999;

      const result = await SignedHashChain.verifyExported(exported);
      expect(result.valid).toBe(false);
    });

    it('should detect broken hash chain in export', async () => {
      await chain.append('TEST1', {});
      await chain.append('TEST2', {});

      const exported = chain.export();
      // Break the hash chain link
      exported.entries[1].payload.previousHash = 'broken-hash';

      const result = await SignedHashChain.verifyExported(exported);
      expect(result.valid).toBe(false);
      expect(result.details.hashChainIntact).toBe(false);
    });
  });

  describe('getRecent', () => {
    it('should return last N entries', async () => {
      await chain.append('A', {});
      await chain.append('B', {});
      await chain.append('C', {});
      await chain.append('D', {});

      const recent = chain.getRecent(2);
      expect(recent.length).toBe(2);
      expect(recent[0].payload.operationType).toBe('C');
      expect(recent[1].payload.operationType).toBe('D');
    });

    it('should return all if count exceeds length', () => {
      const recent = chain.getRecent(100);
      expect(recent.length).toBe(1); // Just genesis
    });
  });
});
