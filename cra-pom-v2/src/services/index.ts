// File: src/services/index.ts
// Service exports

export {
  LLMService,
  DemoLLMService,
  buildGeometricContext,
  parseCommands,
  formatContextForPrompt,
  GEOMETRIC_COGNITION_SYSTEM_PROMPT,
  type LLMProviderConfig,
  type ChatMessage,
  type GeometricContext,
  type SimulationCommand,
  type LLMResponse,
} from './llm';
