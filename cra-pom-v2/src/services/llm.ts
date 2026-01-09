// File: src/services/llm.ts
// LLM Integration Service - Connects geometric cognition to language models

import type { GeometricState, ConvexityResult } from '../core/geometry';
import type { CoherenceMetrics } from '../core/analysis';
import type { TrajectoryMode } from '../components/SettingsPanel';

/**
 * LLM Provider configuration
 */
export interface LLMProviderConfig {
  provider: 'anthropic' | 'openai' | 'ollama' | 'custom';
  apiKey?: string;
  baseUrl?: string;
  model: string;
}

/**
 * Message in the conversation
 */
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  geometricContext?: GeometricContext;
}

/**
 * Geometric context passed to LLM
 */
export interface GeometricContext {
  position: { w: number; x: number; y: number; z: number };
  convexityStatus: string;
  coherenceMetrics: {
    overall: number;
    spinorAlignment: number;
    goldenResonance: number;
    stability: number;
  };
  trajectoryMode: string;
  stepCount: number;
}

/**
 * Commands the LLM can issue to control the simulation
 */
export interface SimulationCommand {
  type: 'set_mode' | 'inject_entropy' | 'reset' | 'toggle_run' | 'step' | 'set_projection';
  payload?: unknown;
}

/**
 * LLM response with potential commands
 */
export interface LLMResponse {
  message: string;
  commands?: SimulationCommand[];
  reasoning?: string;
}

/**
 * Build geometric context for LLM
 */
export function buildGeometricContext(
  state: GeometricState,
  convexity: ConvexityResult | null,
  coherence: CoherenceMetrics | null,
  mode: TrajectoryMode,
  stepCount: number
): GeometricContext {
  return {
    position: state.position,
    convexityStatus: convexity?.status ?? 'unknown',
    coherenceMetrics: {
      overall: coherence?.overallCoherence ?? 0,
      spinorAlignment: coherence?.spinorAlignment ?? 0,
      goldenResonance: coherence?.goldenResonance ?? 0,
      stability: coherence?.stabilityIndex ?? 0,
    },
    trajectoryMode: mode,
    stepCount,
  };
}

/**
 * System prompt for geometric cognition
 */
export const GEOMETRIC_COGNITION_SYSTEM_PROMPT = `You are a geometric cognition assistant operating within a 4D quaternion-based cognitive manifold. You have access to real-time geometric state information from a 24-cell polytope (D4 lattice) visualization.

GEOMETRIC CONTEXT:
- Position: A unit quaternion (w, x, y, z) representing the current cognitive state
- Convexity Status: SAFE (inside), WARNING (near boundary), VIOLATION (outside L1 norm ≤ 2)
- Coherence Metrics:
  - Overall: Combined cognitive coherence (0-1)
  - Spinor Alignment: How aligned the isoclinic rotation axes are
  - Golden Resonance: Proximity to φ (golden ratio) patterns
  - Stability: Trajectory smoothness and predictability
- Trajectory Mode: Current exploration pattern (free, spiral, boundary, ergodic, oscillate)

CAPABILITIES:
You can issue commands to control the simulation by including them in your response using this format:
[COMMAND: type=value]

Available commands:
- [COMMAND: mode=free|spiral|boundary|ergodic|oscillate] - Change trajectory mode
- [COMMAND: entropy=random] - Inject random entropy
- [COMMAND: entropy=w,x,y,z] - Inject specific entropy vector
- [COMMAND: reset] - Reset to origin
- [COMMAND: run] - Start simulation
- [COMMAND: pause] - Pause simulation
- [COMMAND: step] - Single step

RESPONSE GUIDELINES:
1. Interpret the geometric state metaphorically for cognitive/reasoning contexts
2. Use coherence metrics to gauge "clarity of thought"
3. Suggest mode changes based on the user's goals
4. Monitor for violations as potential "boundary conditions" being reached
5. Reference the golden ratio (φ ≈ 1.618) when discussing harmonic patterns

The user may ask about the current state, request changes, or discuss abstract concepts using the geometric framework as a metaphor.`;

/**
 * Parse commands from LLM response
 */
export function parseCommands(response: string): SimulationCommand[] {
  const commands: SimulationCommand[] = [];
  const commandRegex = /\[COMMAND:\s*(\w+)=([^\]]+)\]/gi;
  let match;

  while ((match = commandRegex.exec(response)) !== null) {
    const [, type, value] = match;

    switch (type.toLowerCase()) {
      case 'mode':
        if (['free', 'spiral', 'boundary', 'ergodic', 'oscillate'].includes(value)) {
          commands.push({ type: 'set_mode', payload: value });
        }
        break;
      case 'entropy':
        if (value === 'random') {
          commands.push({ type: 'inject_entropy', payload: 'random' });
        } else {
          const parts = value.split(',').map(Number);
          if (parts.length === 4 && parts.every((n) => !isNaN(n))) {
            commands.push({ type: 'inject_entropy', payload: parts });
          }
        }
        break;
      case 'reset':
        commands.push({ type: 'reset' });
        break;
      case 'run':
        commands.push({ type: 'toggle_run', payload: true });
        break;
      case 'pause':
        commands.push({ type: 'toggle_run', payload: false });
        break;
      case 'step':
        commands.push({ type: 'step' });
        break;
    }
  }

  return commands;
}

/**
 * Format geometric context for display
 */
export function formatContextForPrompt(context: GeometricContext): string {
  return `
[GEOMETRIC STATE]
Position: (${context.position.w.toFixed(3)}, ${context.position.x.toFixed(3)}, ${context.position.y.toFixed(3)}, ${context.position.z.toFixed(3)})
Status: ${context.convexityStatus}
Step: ${context.stepCount}
Mode: ${context.trajectoryMode}

[COHERENCE]
Overall: ${(context.coherenceMetrics.overall * 100).toFixed(1)}%
Spinor Alignment: ${(context.coherenceMetrics.spinorAlignment * 100).toFixed(1)}%
Golden Resonance: ${(context.coherenceMetrics.goldenResonance * 100).toFixed(1)}%
Stability: ${(context.coherenceMetrics.stability * 100).toFixed(1)}%
`.trim();
}

/**
 * LLM Service class
 */
export class LLMService {
  private config: LLMProviderConfig;
  private conversationHistory: ChatMessage[] = [];

  constructor(config: LLMProviderConfig) {
    this.config = config;
  }

  /**
   * Update configuration
   */
  setConfig(config: Partial<LLMProviderConfig>) {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get conversation history
   */
  getHistory(): ChatMessage[] {
    return [...this.conversationHistory];
  }

  /**
   * Clear conversation history
   */
  clearHistory() {
    this.conversationHistory = [];
  }

  /**
   * Add message to history
   */
  addMessage(message: Omit<ChatMessage, 'id' | 'timestamp'>): ChatMessage {
    const fullMessage: ChatMessage = {
      ...message,
      id: `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
      timestamp: Date.now(),
    };
    this.conversationHistory.push(fullMessage);
    return fullMessage;
  }

  /**
   * Send message and get response
   */
  async sendMessage(
    userMessage: string,
    geometricContext: GeometricContext
  ): Promise<LLMResponse> {
    // Add user message with context
    this.addMessage({
      role: 'user',
      content: userMessage,
      geometricContext,
    });

    try {
      const response = await this.callProvider(userMessage, geometricContext);

      // Parse commands from response
      const commands = parseCommands(response);

      // Clean response (remove command markers for display)
      const cleanResponse = response.replace(/\[COMMAND:[^\]]+\]/gi, '').trim();

      // Add assistant message
      this.addMessage({
        role: 'assistant',
        content: cleanResponse,
      });

      return {
        message: cleanResponse,
        commands: commands.length > 0 ? commands : undefined,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new Error(`LLM request failed: ${errorMessage}`);
    }
  }

  /**
   * Call the configured provider
   */
  private async callProvider(
    userMessage: string,
    context: GeometricContext
  ): Promise<string> {
    const contextString = formatContextForPrompt(context);
    const fullMessage = `${contextString}\n\n${userMessage}`;

    switch (this.config.provider) {
      case 'anthropic':
        return this.callAnthropic(fullMessage);
      case 'openai':
        return this.callOpenAI(fullMessage);
      case 'ollama':
        return this.callOllama(fullMessage);
      case 'custom':
        return this.callCustom(fullMessage);
      default:
        throw new Error(`Unknown provider: ${this.config.provider}`);
    }
  }

  /**
   * Call Anthropic API
   */
  private async callAnthropic(message: string): Promise<string> {
    if (!this.config.apiKey) {
      throw new Error('Anthropic API key required');
    }

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.config.apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: this.config.model || 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        system: GEOMETRIC_COGNITION_SYSTEM_PROMPT,
        messages: [
          ...this.conversationHistory.slice(0, -1).map((m) => ({
            role: m.role === 'user' ? 'user' : 'assistant',
            content: m.content,
          })),
          { role: 'user', content: message },
        ],
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Anthropic API error: ${error}`);
    }

    const data = await response.json();
    return data.content[0].text;
  }

  /**
   * Call OpenAI API
   */
  private async callOpenAI(message: string): Promise<string> {
    if (!this.config.apiKey) {
      throw new Error('OpenAI API key required');
    }

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({
        model: this.config.model || 'gpt-4',
        messages: [
          { role: 'system', content: GEOMETRIC_COGNITION_SYSTEM_PROMPT },
          ...this.conversationHistory.slice(0, -1).map((m) => ({
            role: m.role,
            content: m.content,
          })),
          { role: 'user', content: message },
        ],
        max_tokens: 1024,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenAI API error: ${error}`);
    }

    const data = await response.json();
    return data.choices[0].message.content;
  }

  /**
   * Call Ollama API (local)
   */
  private async callOllama(message: string): Promise<string> {
    const baseUrl = this.config.baseUrl || 'http://localhost:11434';

    const response = await fetch(`${baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.config.model || 'llama2',
        messages: [
          { role: 'system', content: GEOMETRIC_COGNITION_SYSTEM_PROMPT },
          ...this.conversationHistory.slice(0, -1).map((m) => ({
            role: m.role,
            content: m.content,
          })),
          { role: 'user', content: message },
        ],
        stream: false,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Ollama API error: ${error}`);
    }

    const data = await response.json();
    return data.message.content;
  }

  /**
   * Call custom API endpoint
   */
  private async callCustom(message: string): Promise<string> {
    if (!this.config.baseUrl) {
      throw new Error('Custom API base URL required');
    }

    const response = await fetch(this.config.baseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.config.apiKey ? { Authorization: `Bearer ${this.config.apiKey}` } : {}),
      },
      body: JSON.stringify({
        model: this.config.model,
        system: GEOMETRIC_COGNITION_SYSTEM_PROMPT,
        messages: [
          ...this.conversationHistory.slice(0, -1).map((m) => ({
            role: m.role,
            content: m.content,
          })),
          { role: 'user', content: message },
        ],
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Custom API error: ${error}`);
    }

    const data = await response.json();
    // Try common response formats
    return data.content || data.message || data.response || data.text || JSON.stringify(data);
  }
}

/**
 * Demo mode - simulates LLM responses for testing
 */
export class DemoLLMService extends LLMService {
  constructor() {
    super({ provider: 'custom', model: 'demo', baseUrl: '' });
  }

  async sendMessage(
    userMessage: string,
    geometricContext: GeometricContext
  ): Promise<LLMResponse> {
    // Add user message
    this.addMessage({
      role: 'user',
      content: userMessage,
      geometricContext,
    });

    // Generate contextual response
    const response = this.generateDemoResponse(userMessage, geometricContext);

    // Parse commands
    const commands = parseCommands(response);
    const cleanResponse = response.replace(/\[COMMAND:[^\]]+\]/gi, '').trim();

    // Add assistant message
    this.addMessage({
      role: 'assistant',
      content: cleanResponse,
    });

    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 500 + Math.random() * 500));

    return {
      message: cleanResponse,
      commands: commands.length > 0 ? commands : undefined,
    };
  }

  private generateDemoResponse(message: string, ctx: GeometricContext): string {
    const lowerMessage = message.toLowerCase();

    // Status queries
    if (lowerMessage.includes('status') || lowerMessage.includes('state') || lowerMessage.includes('where')) {
      return `The cognitive manifold is currently at quaternion position (${ctx.position.w.toFixed(3)}, ${ctx.position.x.toFixed(3)}, ${ctx.position.y.toFixed(3)}, ${ctx.position.z.toFixed(3)}) with ${ctx.convexityStatus} convexity status.

Overall coherence is at ${(ctx.coherenceMetrics.overall * 100).toFixed(1)}%, with spinor alignment at ${(ctx.coherenceMetrics.spinorAlignment * 100).toFixed(1)}%. ${
        ctx.coherenceMetrics.overall > 0.7
          ? 'The system is in a highly coherent state, indicating clear cognitive alignment.'
          : ctx.coherenceMetrics.overall > 0.4
          ? 'Moderate coherence suggests room for optimization.'
          : 'Low coherence may indicate need for entropy injection or mode change.'
      }`;
    }

    // Mode changes
    if (lowerMessage.includes('explore') || lowerMessage.includes('ergodic')) {
      return `Switching to ergodic mode for maximum manifold coverage. This will inject periodic entropy to ensure the trajectory explores the full 24-cell structure.

[COMMAND: mode=ergodic]

Ergodic exploration maximizes coverage while maintaining convexity constraints. Watch for golden ratio resonance patterns as the trajectory evolves.`;
    }

    if (lowerMessage.includes('boundary') || lowerMessage.includes('edge') || lowerMessage.includes('limit')) {
      return `Engaging boundary exploration mode. The system will navigate near the L1 norm = 2 surface, testing the limits of the convexity constraint.

[COMMAND: mode=boundary]

This mode is useful for understanding the constraint boundaries of the cognitive manifold. Stay alert for WARNING states.`;
    }

    if (lowerMessage.includes('spiral') || lowerMessage.includes('outward')) {
      return `Activating spiral trajectory mode. The system will gradually expand outward in a golden spiral pattern.

[COMMAND: mode=spiral]

The spiral follows φ-based proportions, creating aesthetically harmonious paths through the 24-cell.`;
    }

    if (lowerMessage.includes('oscillate') || lowerMessage.includes('rhythm')) {
      return `Setting oscillation mode for periodic rhythmic motion between vertices.

[COMMAND: mode=oscillate]

Oscillation creates standing wave patterns in the manifold, useful for studying resonance phenomena.`;
    }

    if (lowerMessage.includes('free') || lowerMessage.includes('natural')) {
      return `Returning to free evolution mode - natural isoclinic rotation with golden ratio axes.

[COMMAND: mode=free]

Free mode allows the inherent geometry of Clifford translations to guide the trajectory.`;
    }

    // Entropy injection
    if (lowerMessage.includes('entropy') || lowerMessage.includes('random') || lowerMessage.includes('shake')) {
      return `Injecting entropy into the system to perturb the trajectory and explore new regions of the manifold.

[COMMAND: entropy=random]

Entropy injection can break out of local patterns and improve ergodicity.`;
    }

    // Reset
    if (lowerMessage.includes('reset') || lowerMessage.includes('restart') || lowerMessage.includes('origin')) {
      return `Resetting the cognitive manifold to origin state (1, 0, 0, 0) - the identity quaternion.

[COMMAND: reset]

The system will begin fresh from the scalar position on the 3-sphere.`;
    }

    // Run/pause
    if (lowerMessage.includes('start') || lowerMessage.includes('run') || lowerMessage.includes('go')) {
      return `Starting continuous simulation. The trajectory will evolve according to the current mode.

[COMMAND: run]

Watch the coherence metrics to monitor system behavior.`;
    }

    if (lowerMessage.includes('stop') || lowerMessage.includes('pause') || lowerMessage.includes('halt')) {
      return `Pausing the simulation. The geometric state is preserved.

[COMMAND: pause]

You can use single-step mode to advance manually.`;
    }

    // Coherence queries
    if (lowerMessage.includes('coherence') || lowerMessage.includes('clarity') || lowerMessage.includes('aligned')) {
      const assessment = ctx.coherenceMetrics.overall > 0.8 ? 'excellent' :
                        ctx.coherenceMetrics.overall > 0.6 ? 'good' :
                        ctx.coherenceMetrics.overall > 0.4 ? 'moderate' : 'low';

      return `Current coherence assessment: ${assessment}

- Overall Coherence: ${(ctx.coherenceMetrics.overall * 100).toFixed(1)}%
- Spinor Alignment: ${(ctx.coherenceMetrics.spinorAlignment * 100).toFixed(1)}%
- Golden Resonance: ${(ctx.coherenceMetrics.goldenResonance * 100).toFixed(1)}%
- Stability Index: ${(ctx.coherenceMetrics.stability * 100).toFixed(1)}%

${ctx.coherenceMetrics.goldenResonance > 0.7
  ? 'Strong golden ratio patterns detected - the trajectory is harmonically aligned.'
  : 'Consider ergodic mode to improve golden resonance through broader exploration.'}`;
    }

    // Help
    if (lowerMessage.includes('help') || lowerMessage.includes('what can') || lowerMessage.includes('commands')) {
      return `I can help you navigate and understand the geometric cognition manifold. Here's what I can do:

**Query State:**
- Ask about current status, position, or coherence metrics

**Control Modes:**
- "explore" or "ergodic" - Maximize manifold coverage
- "boundary" - Test convexity limits
- "spiral" - Golden spiral outward motion
- "oscillate" - Rhythmic vertex-to-vertex motion
- "free" - Natural isoclinic rotation

**Actions:**
- "entropy" or "shake" - Inject randomness
- "reset" - Return to origin
- "start/run" - Begin continuous evolution
- "pause/stop" - Halt simulation

Current mode: ${ctx.trajectoryMode}
Current status: ${ctx.convexityStatus}`;
    }

    // Golden ratio
    if (lowerMessage.includes('golden') || lowerMessage.includes('phi') || lowerMessage.includes('φ')) {
      return `The golden ratio φ ≈ 1.618 is fundamental to this system's geometry. The isoclinic rotation axes are aligned along φ-based directions, creating inherently harmonic trajectories.

Current Golden Resonance: ${(ctx.coherenceMetrics.goldenResonance * 100).toFixed(1)}%

${ctx.coherenceMetrics.goldenResonance > 0.6
  ? 'The trajectory is exhibiting strong φ-proportioned patterns!'
  : 'Try spiral mode to enhance golden ratio alignment: [COMMAND: mode=spiral]'}`;
    }

    // Default contextual response
    return `I'm observing the geometric cognition manifold at step ${ctx.stepCount}.

Current state:
- Position: (${ctx.position.w.toFixed(2)}, ${ctx.position.x.toFixed(2)}, ${ctx.position.y.toFixed(2)}, ${ctx.position.z.toFixed(2)})
- Convexity: ${ctx.convexityStatus}
- Coherence: ${(ctx.coherenceMetrics.overall * 100).toFixed(0)}%
- Mode: ${ctx.trajectoryMode}

You can ask me about the geometric state, request mode changes, or inject entropy. Try asking "help" for available commands.`;
  }
}

export default LLMService;
