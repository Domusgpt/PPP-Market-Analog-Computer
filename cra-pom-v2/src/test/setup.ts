/**
 * Vitest Setup File
 *
 * Provides polyfills for browser APIs in Node.js environment.
 */

import { webcrypto } from 'crypto';

// Polyfill Web Crypto API
if (typeof globalThis.crypto === 'undefined') {
  // @ts-expect-error - Node's webcrypto is compatible but typed differently
  globalThis.crypto = webcrypto;
}

// Polyfill TextEncoder/TextDecoder (should already exist in Node 18+)
if (typeof globalThis.TextEncoder === 'undefined') {
  const { TextEncoder, TextDecoder } = await import('util');
  globalThis.TextEncoder = TextEncoder;
  // @ts-expect-error - Type compatibility
  globalThis.TextDecoder = TextDecoder;
}

// Mock Worker for tests (Web Workers don't exist in Node)
class MockWorker {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;

  private messageQueue: unknown[] = [];
  private keypair: CryptoKeyPair | null = null;

  constructor(_scriptUrl: string | URL) {
    // Initialize mock worker with crypto keys
    this.initKeys();
  }

  private async initKeys() {
    this.keypair = await crypto.subtle.generateKey(
      { name: 'ECDSA', namedCurve: 'P-256' },
      true, // extractable for testing
      ['sign', 'verify']
    );
  }

  async postMessage(message: { type: string; id: string; payload?: unknown }) {
    // Simulate worker processing
    await this.processMessage(message);
  }

  private async processMessage(message: { type: string; id: string; payload?: unknown }) {
    if (!this.keypair) {
      await this.initKeys();
    }

    let response: unknown;

    switch (message.type) {
      case 'GET_PUBLIC_KEY':
        response = {
          type: 'PUBLIC_KEY',
          id: message.id,
          publicKey: await crypto.subtle.exportKey('jwk', this.keypair!.publicKey),
        };
        break;

      case 'SIGN':
        const payload = message.payload as { data: string };
        const data = new TextEncoder().encode(payload.data);
        const signature = await crypto.subtle.sign(
          { name: 'ECDSA', hash: 'SHA-256' },
          this.keypair!.privateKey,
          data
        );
        response = {
          type: 'SIGNATURE',
          id: message.id,
          signature: btoa(String.fromCharCode(...new Uint8Array(signature))),
        };
        break;

      case 'GET_ATTESTATION':
        response = {
          type: 'ATTESTATION',
          id: message.id,
          attestation: {
            workerIsolation: true,
            keyNonExtractable: false, // In mock, keys are extractable
            algorithm: 'ECDSA-P256-SHA256',
            createdAt: new Date().toISOString(),
            workerId: 'mock-worker-test',
          },
        };
        break;

      default:
        response = {
          type: 'ERROR',
          id: message.id,
          error: `Unknown message type: ${message.type}`,
        };
    }

    // Deliver response asynchronously
    setTimeout(() => {
      if (this.onmessage) {
        this.onmessage({ data: response } as MessageEvent);
      }
    }, 0);
  }

  terminate() {
    // Cleanup
  }
}

// @ts-expect-error - Mock Worker for testing
globalThis.Worker = MockWorker;

// Mock URL.createObjectURL for worker scripts
if (typeof URL.createObjectURL === 'undefined') {
  URL.createObjectURL = (_blob: Blob) => 'blob:mock-worker-url';
}

if (typeof URL.revokeObjectURL === 'undefined') {
  URL.revokeObjectURL = (_url: string) => {};
}

export {};
