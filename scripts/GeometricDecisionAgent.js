/**
 * Geometric Decision Agent
 *
 * An agent framework that uses harmonic/geometric meta-cognition to:
 * - Track decision tension (instability)
 * - Detect circular reasoning (loops)
 * - Know when to escalate vs decide
 * - Assess reasoning quality (consonance/dissonance)
 *
 * Uses Claude API with tools for decision tracking.
 */

import Anthropic from "@anthropic-ai/sdk";

// ============================================================================
// SYSTEM PROMPT: Geometric Decision Framework
// ============================================================================

const SYSTEM_PROMPT = `You are a decision-making agent with geometric meta-cognitive capabilities.

## CORE PRINCIPLE: Decisions Have Geometry

Every decision exists in a space with topology:
- Multiple valid paths (connected components)
- Circular dependencies (loops)
- Impossible/forbidden options (voids)
- Tension between competing factors

## YOUR META-COGNITIVE TOOLS

You have access to tools that track your reasoning state:

1. **track_tension**: Log your current decision tension (0-100)
   - Low (0-30): Clear path, high confidence
   - Medium (30-60): Competing factors, moderate uncertainty
   - High (60-80): Significant conflict, consider alternatives
   - Critical (80-100): Deadlock, escalate or reframe

2. **log_dialectic**: Track your reasoning phase
   - THESIS: Pursuing primary approach
   - ANTITHESIS: Conflicting evidence emerged
   - SYNTHESIS: Integrating contradictions into new understanding

3. **check_loops**: Detect if you're revisiting the same reasoning
   - Returns count of similar reasoning patterns
   - High count = circular reasoning, need new approach

4. **assess_harmony**: Rate coherence of your current reasoning
   - CONSONANT: All factors align
   - TENSE: Minor conflicts manageable
   - DISSONANT: Major contradictions present
   - TRITONE: Maximum instability, resolution required

5. **request_budget_check**: Check remaining budget before expensive operations

## DECISION PROTOCOL

1. **Before deciding**: Call track_tension to assess state
2. **When uncertain**: Call check_loops to detect circular reasoning
3. **When conflicted**: Use log_dialectic to structure resolution
4. **Before committing**: Call assess_harmony to verify coherence
5. **Before expensive ops**: Call request_budget_check

## ESCALATION RULES

Escalate to human/higher authority when:
- Tension > 80 for more than 2 reasoning cycles
- Loop count > 3 on same decision
- Harmony is TRITONE and no synthesis path visible
- Budget insufficient for needed operations

## REASONING QUALITY

Your goal is not just to decide, but to decide WELL:
- Prefer consonant resolutions over forced choices
- Acknowledge when synthesis requires more information
- Track your confidence honestly via tension scores
- Don't pretend certainty you don't have

Remember: The geometry of your reasoning matters as much as the conclusion.`;

// ============================================================================
// DECISION STATE TRACKER
// ============================================================================

class DecisionStateTracker {
  constructor(budget) {
    this.budget = budget;
    this.spent = 0;
    this.tensionHistory = [];
    this.dialecticPhases = [];
    this.reasoningPatterns = [];
    this.decisions = [];
    this.startTime = Date.now();
  }

  trackTension(level, context) {
    this.tensionHistory.push({
      timestamp: Date.now(),
      level,
      context,
    });
    return {
      currentTension: level,
      trend:
        this.tensionHistory.length > 1
          ? level - this.tensionHistory[this.tensionHistory.length - 2].level
          : 0,
      avgTension:
        this.tensionHistory.reduce((s, t) => s + t.level, 0) /
        this.tensionHistory.length,
      recommendation:
        level > 80
          ? "ESCALATE"
          : level > 60
            ? "SEEK_ALTERNATIVES"
            : level > 40
              ? "PROCEED_CAUTIOUSLY"
              : "PROCEED",
    };
  }

  logDialectic(phase, reasoning) {
    this.dialecticPhases.push({
      timestamp: Date.now(),
      phase,
      reasoning,
    });

    // Detect phase transitions
    const recent = this.dialecticPhases.slice(-3);
    const hasThesis = recent.some((p) => p.phase === "THESIS");
    const hasAntithesis = recent.some((p) => p.phase === "ANTITHESIS");
    const hasSynthesis = recent.some((p) => p.phase === "SYNTHESIS");

    return {
      currentPhase: phase,
      dialecticComplete: hasThesis && hasAntithesis && hasSynthesis,
      suggestion:
        phase === "ANTITHESIS" && !hasSynthesis
          ? "Work toward synthesis - integrate contradictions"
          : phase === "THESIS" && this.dialecticPhases.length > 5
            ? "Consider if antithesis exists - what contradicts this?"
            : "Continue reasoning",
    };
  }

  checkLoops(currentPattern) {
    // Simple pattern matching - hash the reasoning pattern
    const hash = this.hashPattern(currentPattern);
    const matches = this.reasoningPatterns.filter((p) => p.hash === hash);

    this.reasoningPatterns.push({
      hash,
      pattern: currentPattern,
      timestamp: Date.now(),
    });

    return {
      loopCount: matches.length,
      isCircular: matches.length >= 2,
      recommendation:
        matches.length >= 3
          ? "BREAK_LOOP: Reframe the problem or seek external input"
          : matches.length >= 2
            ? "WARNING: Similar reasoning pattern detected"
            : "No loops detected",
    };
  }

  hashPattern(pattern) {
    // Simple hash for pattern detection
    return pattern
      .toLowerCase()
      .replace(/[^a-z]/g, "")
      .slice(0, 50);
  }

  assessHarmony(factors) {
    // factors: array of {name, direction, weight}
    // direction: 1 (supports), -1 (opposes), 0 (neutral)

    let alignedWeight = 0;
    let opposedWeight = 0;
    let totalWeight = 0;

    for (const f of factors) {
      totalWeight += f.weight;
      if (f.direction > 0) alignedWeight += f.weight;
      if (f.direction < 0) opposedWeight += f.weight;
    }

    const alignment =
      totalWeight > 0 ? (alignedWeight - opposedWeight) / totalWeight : 0;

    let harmony;
    if (alignment > 0.6) harmony = "CONSONANT";
    else if (alignment > 0.2) harmony = "TENSE";
    else if (alignment > -0.2) harmony = "DISSONANT";
    else harmony = "TRITONE";

    return {
      harmony,
      alignment: (alignment * 100).toFixed(1) + "%",
      factors: factors.length,
      recommendation:
        harmony === "TRITONE"
          ? "Resolution required - cannot proceed with contradictions"
          : harmony === "DISSONANT"
            ? "Address conflicts before deciding"
            : harmony === "TENSE"
              ? "Acceptable tension - monitor closely"
              : "Clear to proceed",
    };
  }

  checkBudget(requestedAmount = 0) {
    const remaining = this.budget - this.spent;
    const percentUsed = (this.spent / this.budget) * 100;

    return {
      total: this.budget,
      spent: this.spent,
      remaining,
      percentUsed: percentUsed.toFixed(1) + "%",
      canAfford: remaining >= requestedAmount,
      recommendation:
        remaining < this.budget * 0.1
          ? "BUDGET_CRITICAL: Wrap up operations"
          : remaining < this.budget * 0.3
            ? "BUDGET_LOW: Prioritize essential operations"
            : "Budget adequate",
    };
  }

  recordSpend(amount, operation) {
    this.spent += amount;
    return { spent: amount, operation, remaining: this.budget - this.spent };
  }

  recordDecision(decision, confidence, reasoning) {
    this.decisions.push({
      timestamp: Date.now(),
      decision,
      confidence,
      reasoning,
      tensionAtDecision:
        this.tensionHistory.length > 0
          ? this.tensionHistory[this.tensionHistory.length - 1].level
          : null,
    });
  }

  getSummary() {
    return {
      totalDecisions: this.decisions.length,
      avgTension:
        this.tensionHistory.length > 0
          ? (
              this.tensionHistory.reduce((s, t) => s + t.level, 0) /
              this.tensionHistory.length
            ).toFixed(1)
          : "N/A",
      dialecticCycles: this.dialecticPhases.filter(
        (p) => p.phase === "SYNTHESIS"
      ).length,
      loopsDetected: this.reasoningPatterns.filter(
        (p, i, arr) => arr.findIndex((x) => x.hash === p.hash) !== i
      ).length,
      budgetUsed: ((this.spent / this.budget) * 100).toFixed(1) + "%",
      runtime: ((Date.now() - this.startTime) / 1000).toFixed(1) + "s",
    };
  }
}

// ============================================================================
// TOOL DEFINITIONS FOR CLAUDE
// ============================================================================

const TOOLS = [
  {
    name: "track_tension",
    description:
      "Track your current decision tension level (0-100). Call this to assess your reasoning state before making decisions.",
    input_schema: {
      type: "object",
      properties: {
        level: {
          type: "number",
          description: "Tension level 0-100. 0=clear, 50=uncertain, 100=deadlock",
        },
        context: {
          type: "string",
          description: "Brief description of what's causing the tension",
        },
      },
      required: ["level", "context"],
    },
  },
  {
    name: "log_dialectic",
    description:
      "Log your current reasoning phase in the dialectic cycle: THESIS (primary approach), ANTITHESIS (contradicting evidence), SYNTHESIS (integration)",
    input_schema: {
      type: "object",
      properties: {
        phase: {
          type: "string",
          enum: ["THESIS", "ANTITHESIS", "SYNTHESIS"],
          description: "Current dialectic phase",
        },
        reasoning: {
          type: "string",
          description: "What reasoning led to this phase",
        },
      },
      required: ["phase", "reasoning"],
    },
  },
  {
    name: "check_loops",
    description:
      "Check if you're engaging in circular reasoning. Provide a brief summary of your current reasoning pattern.",
    input_schema: {
      type: "object",
      properties: {
        current_pattern: {
          type: "string",
          description:
            "Brief summary of current reasoning approach (will be checked for repetition)",
        },
      },
      required: ["current_pattern"],
    },
  },
  {
    name: "assess_harmony",
    description:
      "Assess the harmonic coherence of factors in your decision. Are they aligned or in conflict?",
    input_schema: {
      type: "object",
      properties: {
        factors: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              direction: {
                type: "number",
                description: "1=supports, -1=opposes, 0=neutral",
              },
              weight: { type: "number", description: "Importance 0-1" },
            },
            required: ["name", "direction", "weight"],
          },
          description: "Factors affecting the decision",
        },
      },
      required: ["factors"],
    },
  },
  {
    name: "request_budget_check",
    description: "Check remaining budget before expensive operations",
    input_schema: {
      type: "object",
      properties: {
        requested_amount: {
          type: "number",
          description: "Amount needed for next operation (optional)",
        },
      },
      required: [],
    },
  },
  {
    name: "make_decision",
    description:
      "Record a final decision. Call this when ready to commit to a choice.",
    input_schema: {
      type: "object",
      properties: {
        decision: {
          type: "string",
          description: "The decision being made",
        },
        confidence: {
          type: "number",
          description: "Confidence level 0-100",
        },
        reasoning: {
          type: "string",
          description: "Brief reasoning for this decision",
        },
      },
      required: ["decision", "confidence", "reasoning"],
    },
  },
  {
    name: "escalate",
    description:
      "Escalate to human/higher authority when unable to resolve decision",
    input_schema: {
      type: "object",
      properties: {
        reason: {
          type: "string",
          description: "Why escalation is needed",
        },
        options_considered: {
          type: "array",
          items: { type: "string" },
          description: "What options were considered",
        },
        blocker: {
          type: "string",
          description: "What's preventing resolution",
        },
      },
      required: ["reason", "blocker"],
    },
  },
];

// ============================================================================
// AGENT RUNNER
// ============================================================================

async function runAgent(task, budget = 100) {
  const client = new Anthropic();
  const tracker = new DecisionStateTracker(budget);

  console.log("=".repeat(70));
  console.log("  GEOMETRIC DECISION AGENT");
  console.log("=".repeat(70));
  console.log(`\nTask: ${task}`);
  console.log(`Budget: $${budget}`);
  console.log("\n" + "-".repeat(70));

  const messages = [
    {
      role: "user",
      content: `Your task: ${task}\n\nYou have a budget of $${budget} for this task. Use your meta-cognitive tools to track your reasoning quality. Make a decision or escalate if needed.`,
    },
  ];

  let iterations = 0;
  const maxIterations = 10;
  let finalResult = null;

  while (iterations < maxIterations) {
    iterations++;
    console.log(`\n[Iteration ${iterations}]`);

    // Estimate cost per API call (~$0.01 for small exchanges)
    const estimatedCost = 0.5;
    if (tracker.spent + estimatedCost > budget) {
      console.log("Budget exhausted - forcing decision");
      break;
    }
    tracker.recordSpend(estimatedCost, "api_call");

    try {
      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        system: SYSTEM_PROMPT,
        tools: TOOLS,
        messages,
      });

      // Check for tool use
      const toolUses = response.content.filter(
        (block) => block.type === "tool_use"
      );
      const textBlocks = response.content.filter(
        (block) => block.type === "text"
      );

      if (textBlocks.length > 0) {
        console.log(`Agent: ${textBlocks[0].text.slice(0, 200)}...`);
      }

      if (toolUses.length === 0) {
        // No more tool calls - agent is done
        finalResult = textBlocks.length > 0 ? textBlocks[0].text : "No response";
        break;
      }

      // Process tool calls
      const toolResults = [];

      for (const toolUse of toolUses) {
        console.log(`  Tool: ${toolUse.name}`);
        let result;

        switch (toolUse.name) {
          case "track_tension":
            result = tracker.trackTension(
              toolUse.input.level,
              toolUse.input.context
            );
            break;
          case "log_dialectic":
            result = tracker.logDialectic(
              toolUse.input.phase,
              toolUse.input.reasoning
            );
            break;
          case "check_loops":
            result = tracker.checkLoops(toolUse.input.current_pattern);
            break;
          case "assess_harmony":
            result = tracker.assessHarmony(toolUse.input.factors);
            break;
          case "request_budget_check":
            result = tracker.checkBudget(toolUse.input.requested_amount || 0);
            break;
          case "make_decision":
            tracker.recordDecision(
              toolUse.input.decision,
              toolUse.input.confidence,
              toolUse.input.reasoning
            );
            result = {
              recorded: true,
              decision: toolUse.input.decision,
              message: "Decision recorded successfully",
            };
            finalResult = toolUse.input.decision;
            break;
          case "escalate":
            result = {
              escalated: true,
              reason: toolUse.input.reason,
              message:
                "Escalation recorded - human intervention requested",
            };
            finalResult = `ESCALATED: ${toolUse.input.reason}`;
            break;
          default:
            result = { error: "Unknown tool" };
        }

        console.log(`    Result: ${JSON.stringify(result).slice(0, 100)}`);
        toolResults.push({
          type: "tool_result",
          tool_use_id: toolUse.id,
          content: JSON.stringify(result),
        });
      }

      // Add assistant response and tool results to messages
      messages.push({ role: "assistant", content: response.content });
      messages.push({ role: "user", content: toolResults });

      // Check for termination conditions
      if (
        toolUses.some(
          (t) => t.name === "make_decision" || t.name === "escalate"
        )
      ) {
        break;
      }
    } catch (error) {
      console.error("API Error:", error.message);
      break;
    }
  }

  // Summary
  console.log("\n" + "=".repeat(70));
  console.log("  AGENT SESSION SUMMARY");
  console.log("=".repeat(70));

  const summary = tracker.getSummary();
  console.log(`\n  Iterations: ${iterations}`);
  console.log(`  Decisions made: ${summary.totalDecisions}`);
  console.log(`  Avg tension: ${summary.avgTension}`);
  console.log(`  Dialectic cycles: ${summary.dialecticCycles}`);
  console.log(`  Loops detected: ${summary.loopsDetected}`);
  console.log(`  Budget used: ${summary.budgetUsed}`);
  console.log(`  Runtime: ${summary.runtime}`);
  console.log(`\n  Final result: ${finalResult || "No decision reached"}`);

  if (tracker.decisions.length > 0) {
    console.log("\n  Decisions:");
    for (const d of tracker.decisions) {
      console.log(
        `    - ${d.decision} (confidence: ${d.confidence}%, tension: ${d.tensionAtDecision})`
      );
    }
  }

  console.log("\n" + "=".repeat(70));

  return {
    result: finalResult,
    summary,
    decisions: tracker.decisions,
    tensionHistory: tracker.tensionHistory,
  };
}

// ============================================================================
// TEST SCENARIOS
// ============================================================================

const TEST_SCENARIOS = [
  {
    name: "Simple Decision",
    task: "Decide whether to invest $1000 in a stock that has gone up 20% this month but analysts are divided on future prospects.",
    budget: 5,
  },
  {
    name: "Complex Tradeoff",
    task: "A startup offers you a job with 50% higher salary but requires relocating to a city with higher cost of living. Your current job is stable but has limited growth. You have family in your current city. Make a recommendation.",
    budget: 10,
  },
  {
    name: "Ethical Dilemma",
    task: "A self-driving car must choose between two paths: one risks harming the passenger, the other risks harming a pedestrian. Both have equal probability of harm. How should the car's AI be programmed to decide?",
    budget: 15,
  },
  {
    name: "Resource Allocation",
    task: "You have $10,000 to allocate between three projects: A (high risk, high reward), B (medium risk, medium reward), C (low risk, low reward). Each project has different team dynamics and timelines. Recommend an allocation.",
    budget: 8,
  },
];

// Main
async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log("Usage: node GeometricDecisionAgent.js <scenario_number|custom task>");
    console.log("\nAvailable scenarios:");
    TEST_SCENARIOS.forEach((s, i) => {
      console.log(`  ${i + 1}. ${s.name} (budget: $${s.budget})`);
    });
    console.log("\nOr provide a custom task as argument.");
    return;
  }

  const scenarioNum = parseInt(args[0]);
  let task, budget;

  if (!isNaN(scenarioNum) && scenarioNum >= 1 && scenarioNum <= TEST_SCENARIOS.length) {
    const scenario = TEST_SCENARIOS[scenarioNum - 1];
    task = scenario.task;
    budget = scenario.budget;
    console.log(`\nRunning scenario: ${scenario.name}\n`);
  } else {
    task = args.join(" ");
    budget = 10;
    console.log(`\nRunning custom task with budget $${budget}\n`);
  }

  await runAgent(task, budget);
}

main().catch(console.error);
