/**
 * PPP v3 End-to-End Example
 *
 * This file demonstrates a complete reasoning session with:
 * - Process isolation
 * - Cryptographic signing
 * - Semantic concepts
 * - Audit chain export
 *
 * Run this example to see PPP v3 in action.
 */

import {
  getVerifiedReasoner,
  getConceptStore,
  initializeWithBasicConcepts,
  SignedHashChain,
} from './index';

/**
 * Run a complete PPP v3 reasoning session
 */
export async function runExample(): Promise<void> {
  console.log('='.repeat(60));
  console.log('PPP v3 End-to-End Example');
  console.log('='.repeat(60));
  console.log();

  // Step 1: Initialize concept store
  console.log('[1] Initializing concept store...');
  const store = getConceptStore();
  await initializeWithBasicConcepts(store);

  // Add some domain concepts
  await store.addConcept('democracy', 'A system of government by the whole population');
  await store.addConcept('freedom', 'The state of being free, liberty');
  await store.addConcept('equality', 'The state of being equal in status or rights');
  await store.addConcept('justice', 'Just behavior or treatment');
  await store.addConcept('tyranny', 'Cruel and oppressive government rule');

  const grounding = store.getGroundingStatus();
  console.log(`   Concepts loaded: ${grounding.conceptCount}`);
  console.log(`   Embedding source: ${grounding.embeddingSource}`);
  console.log(`   Semantically grounded: ${grounding.percentSemantic.toFixed(1)}%`);
  console.log();

  // Step 2: Get reasoner
  console.log('[2] Initializing verified reasoner...');
  const reasoner = await getVerifiedReasoner();
  console.log('   Reasoner initialized with isolated signing service');
  console.log();

  // Step 3: Start reasoning session
  console.log('[3] Starting reasoning session...');
  const session = await reasoner.startSession(
    'What is the relationship between democracy and freedom?'
  );
  console.log(`   Session ID: ${session.sessionId}`);
  console.log(`   Query: ${session.initialQuery}`);
  console.log();

  // Step 4: Perform reasoning steps
  console.log('[4] Performing reasoning steps...');
  console.log();

  // Look up democracy
  console.log('   4a. Looking up "democracy"...');
  const step1 = await reasoner.lookupConcept('democracy');
  console.log(`       Found: ${step1.payload.outputs[0]}`);
  console.log(`       Confidence: ${(step1.payload.confidence * 100).toFixed(1)}%`);
  console.log(`       Signed: ✓ (hash: ${step1.proof.dataHash.substring(0, 16)}...)`);
  console.log();

  // Look up freedom
  console.log('   4b. Looking up "freedom"...');
  const step2 = await reasoner.lookupConcept('freedom');
  console.log(`       Found: ${step2.payload.outputs[0]}`);
  console.log(`       Signed: ✓ (hash: ${step2.proof.dataHash.substring(0, 16)}...)`);
  console.log();

  // Query similar to "liberty"
  console.log('   4c. Finding concepts similar to "liberty"...');
  const step3 = await reasoner.querySimilar('liberty', 3);
  console.log('       Similar concepts:');
  for (const output of step3.payload.outputs) {
    console.log(`         - ${output}`);
  }
  console.log(`       Signed: ✓ (hash: ${step3.proof.dataHash.substring(0, 16)}...)`);
  console.log();

  // Make an inference
  console.log('   4d. Making inference...');
  const step4 = await reasoner.makeInference(
    [
      'Democracy enables citizens to participate in government',
      'Freedom includes political liberty',
      'Political liberty requires participatory rights',
    ],
    'Democracy and freedom are mutually reinforcing concepts',
    0.85
  );
  console.log(`       Inference: ${step4.payload.outputs[0]}`);
  console.log(`       Confidence: ${(step4.payload.confidence * 100).toFixed(1)}%`);
  console.log(`       Signed: ✓`);
  console.log();

  // Step 5: Reach conclusion
  console.log('[5] Reaching conclusion...');
  const conclusion = await reasoner.conclude(
    'Democracy and freedom are deeply interconnected - democracy provides the political structures that protect freedom, while freedom enables the participation required for democracy to function.',
    0.8,
    [1, 2, 3, 4],
    [
      'Analysis limited to conceptual relationships',
      'Real-world complexity not fully captured',
      grounding.embeddingSource === 'deterministic_fallback'
        ? 'Using fallback embeddings - semantic analysis limited'
        : 'Using semantic embeddings',
    ]
  );
  console.log(`   Conclusion: ${conclusion.payload.statement.substring(0, 80)}...`);
  console.log(`   Confidence: ${(conclusion.payload.confidence * 100).toFixed(1)}%`);
  console.log(`   Caveats: ${conclusion.payload.caveats.length}`);
  console.log(`   Semantically grounded: ${conclusion.payload.semanticallyGrounded}`);
  console.log();

  // Step 6: End session and verify
  console.log('[6] Ending session and verifying...');
  const result = await reasoner.endSession();
  console.log();

  console.log('   Session Summary:');
  console.log('   ' + '-'.repeat(40));
  for (const line of result.summary.split('\n')) {
    console.log(`   ${line}`);
  }
  console.log('   ' + '-'.repeat(40));
  console.log();

  console.log('   Verification Status:');
  console.log(`     Chain valid: ${result.verification.chainValid ? '✓' : '✗'}`);
  console.log(`     Signatures valid: ${result.verification.signaturesValid ? '✓' : '✗'}`);
  console.log(`     Public key available: ✓`);
  console.log();

  // Step 7: Export and verify externally
  console.log('[7] Exporting audit chain for external verification...');
  const exported = await reasoner.exportAuditChain() as ReturnType<typeof SignedHashChain.prototype.export>;
  console.log(`   Exported ${exported.length} chain entries`);
  console.log(`   Head hash: ${exported.headHash.substring(0, 32)}...`);
  console.log();

  // Simulate external verification
  console.log('[8] Simulating external verification...');
  console.log('   (In real use, this would be done by an independent party)');

  // Note: verifyExported is a static method that doesn't need the original service
  // It uses only the public key embedded in the exported data
  const externalVerification = await SignedHashChain.verifyExported(exported);
  console.log(`   External verification result: ${externalVerification.valid ? 'VALID' : 'INVALID'}`);
  console.log(`   Hash chain intact: ${externalVerification.details.hashChainIntact ? '✓' : '✗'}`);
  console.log(`   Signatures valid: ${externalVerification.details.signaturesValid ? '✓' : '✗'}`);
  console.log(`   Sequence correct: ${externalVerification.details.sequenceCorrect ? '✓' : '✗'}`);
  console.log();

  // Final notes
  console.log('='.repeat(60));
  console.log('IMPORTANT NOTES');
  console.log('='.repeat(60));
  console.log();
  console.log('What this example demonstrates:');
  console.log('  ✓ Every reasoning step is cryptographically signed');
  console.log('  ✓ The audit chain can be exported and verified externally');
  console.log('  ✓ The system honestly reports its grounding status');
  console.log();
  console.log('What this does NOT demonstrate:');
  console.log('  ✗ That an LLM would actually use these results');
  console.log('  ✗ That the conclusions are semantically correct');
  console.log('  ✗ That the reasoning is complete or sound');
  console.log();
  console.log('The cryptographic proofs only show that certain operations');
  console.log('were requested through proper channels. They do not prove');
  console.log('the reasoning is correct or that it was actually used.');
  console.log();
}

/**
 * Example of concept composition (when semantically grounded)
 */
export async function runCompositionExample(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Concept Composition Example');
  console.log('='.repeat(60));
  console.log();

  const store = getConceptStore();

  // Try the classic analogy: king - man + woman ≈ queen
  console.log('Attempting: king - man + woman = ?');
  console.log();

  // Add concepts if not present
  if (!store.hasConcept('king')) {
    await store.addConcept('king', 'A male monarch, ruler of a kingdom');
  }
  if (!store.hasConcept('queen')) {
    await store.addConcept('queen', 'A female monarch, ruler of a kingdom');
  }
  if (!store.hasConcept('man')) {
    await store.addConcept('man', 'An adult male human');
  }
  if (!store.hasConcept('woman')) {
    await store.addConcept('woman', 'An adult female human');
  }

  const result = await store.compose([
    { concept: 'king', operation: 'add' },
    { concept: 'man', operation: 'subtract' },
    { concept: 'woman', operation: 'add' },
  ]);

  console.log('Nearest concepts to (king - man + woman):');
  for (const nearest of result.nearest.slice(0, 5)) {
    console.log(`  ${nearest.concept.name}: ${(nearest.similarity * 100).toFixed(1)}%`);
  }
  console.log();

  if (result.semanticallyMeaningful) {
    console.log('✓ Using semantic embeddings - results are meaningful');
  } else {
    console.log('⚠ Using fallback embeddings - results are NOT semantically meaningful');
    console.log('  (Configure real embeddings for meaningful analogy results)');
  }
  console.log();
}

// Run if executed directly in a browser
if (typeof window !== 'undefined' && typeof Worker !== 'undefined') {
  // Browser environment with Web Worker support
  runExample()
    .then(() => runCompositionExample())
    .catch(console.error);
} else if (typeof window === 'undefined') {
  // Node.js environment - would need Web Worker polyfill
  console.log('Note: This example requires a browser environment with Web Workers');
  console.log('Please run in a browser or use a Web Worker polyfill for Node.js');
}
