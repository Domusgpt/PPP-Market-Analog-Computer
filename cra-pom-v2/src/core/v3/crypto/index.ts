/**
 * PPP v3 Cryptographic Module
 *
 * Exports all cryptographic primitives for the PPP v3 architecture.
 *
 * TRUST BOUNDARY:
 * These services should run in an isolated context (Web Worker)
 * that the LLM cannot directly access. The main thread should only
 * interact with them through a message-passing API.
 */

export {
  CryptoService,
  getCryptoService,
  resetCryptoService,
  type KeyPair,
  type ExportedKeyPair,
  type Signature,
  type HashResult,
  type SignedData,
} from './crypto-service';

export {
  SignedHashChain,
  createSignedHashChain,
  type ChainEntry,
  type SignedChainEntry,
  type ChainValidationResult,
} from './hash-chain';
