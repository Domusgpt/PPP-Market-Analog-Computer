/**
 * PPP v3 Crypto Service Tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { CryptoService, getCryptoService, resetCryptoService } from './crypto-service';

describe('CryptoService', () => {
  let crypto: CryptoService;

  beforeEach(async () => {
    crypto = new CryptoService();
    await crypto.initialize();
  });

  describe('initialization', () => {
    it('should initialize successfully', () => {
      expect(crypto.isInitialized()).toBe(true);
    });

    it('should provide public key after initialization', () => {
      const publicKey = crypto.getPublicKeyJwk();
      expect(publicKey).toBeDefined();
      expect(publicKey.kty).toBe('EC');
      expect(publicKey.crv).toBe('P-256');
    });

    it('should throw if not initialized', () => {
      const uninit = new CryptoService();
      expect(() => uninit.getPublicKeyJwk()).toThrow('CryptoService not initialized');
    });
  });

  describe('hashing', () => {
    it('should produce 256-bit hash', async () => {
      const result = await crypto.hashString('hello world');
      expect(result.bytes.length).toBe(32); // 256 bits = 32 bytes
      expect(result.hex.length).toBe(64); // 32 bytes = 64 hex chars
    });

    it('should produce consistent hashes', async () => {
      const hash1 = await crypto.hashString('test data');
      const hash2 = await crypto.hashString('test data');
      expect(hash1.hex).toBe(hash2.hex);
    });

    it('should produce different hashes for different inputs', async () => {
      const hash1 = await crypto.hashString('data1');
      const hash2 = await crypto.hashString('data2');
      expect(hash1.hex).not.toBe(hash2.hex);
    });

    it('should hash objects deterministically', async () => {
      const obj1 = { b: 2, a: 1 };
      const obj2 = { a: 1, b: 2 };
      const hash1 = await crypto.hashObject(obj1);
      const hash2 = await crypto.hashObject(obj2);
      expect(hash1.hex).toBe(hash2.hex);
    });
  });

  describe('signing', () => {
    it('should produce valid signature', async () => {
      const message = 'test message';
      const signature = await crypto.signString(message);
      expect(signature.base64).toBeDefined();
      expect(signature.data.length).toBeGreaterThan(0);
    });

    it('should throw if not initialized', async () => {
      const uninit = new CryptoService();
      await expect(uninit.signString('test')).rejects.toThrow();
    });
  });

  describe('verification', () => {
    it('should verify valid signature', async () => {
      const message = 'test message';
      const encoder = new TextEncoder();
      const data = encoder.encode(message);

      const signature = await crypto.signString(message);
      const publicKey = crypto.getPublicKeyJwk();

      const valid = await crypto.verifyWithJwk(data, signature.base64, publicKey);
      expect(valid).toBe(true);
    });

    it('should reject tampered data', async () => {
      const message = 'original message';
      const encoder = new TextEncoder();

      const signature = await crypto.signString(message);
      const publicKey = crypto.getPublicKeyJwk();

      const tamperedData = encoder.encode('tampered message');
      const valid = await crypto.verifyWithJwk(tamperedData, signature.base64, publicKey);
      expect(valid).toBe(false);
    });

    it('should reject invalid signature', async () => {
      const message = 'test message';
      const encoder = new TextEncoder();
      const data = encoder.encode(message);
      const publicKey = crypto.getPublicKeyJwk();

      const fakeSignature = btoa('invalid signature data');
      const valid = await crypto.verifyWithJwk(data, fakeSignature, publicKey);
      expect(valid).toBe(false);
    });
  });

  describe('signData', () => {
    it('should create signed data wrapper', async () => {
      const payload = { action: 'test', value: 42 };
      const signed = await crypto.signData(payload);

      expect(signed.payload).toEqual(payload);
      expect(signed.proof.dataHash).toBeDefined();
      expect(signed.proof.signature).toBeDefined();
      expect(signed.proof.publicKeyJwk).toBeDefined();
      expect(signed.proof.timestamp).toBeDefined();
      expect(signed.proof.algorithm).toBe('ECDSA-P256-SHA256');
    });

    it('should verify signed data successfully', async () => {
      const payload = { message: 'hello' };
      const signed = await crypto.signData(payload);

      const result = await crypto.verifySignedData(signed);
      expect(result.valid).toBe(true);
      expect(result.checks.hashMatches).toBe(true);
      expect(result.checks.signatureValid).toBe(true);
      expect(result.checks.timestampReasonable).toBe(true);
    });

    it('should detect tampered payload', async () => {
      const payload = { value: 100 };
      const signed = await crypto.signData(payload);

      // Tamper with payload
      signed.payload.value = 999;

      const result = await crypto.verifySignedData(signed);
      expect(result.valid).toBe(false);
      expect(result.checks.hashMatches).toBe(false);
      expect(result.error).toContain('tampered');
    });
  });
});

describe('getCryptoService singleton', () => {
  beforeEach(() => {
    resetCryptoService();
  });

  it('should return initialized service', async () => {
    const service = await getCryptoService();
    expect(service.isInitialized()).toBe(true);
  });

  it('should return same instance', async () => {
    const service1 = await getCryptoService();
    const service2 = await getCryptoService();
    expect(service1).toBe(service2);
  });
});
