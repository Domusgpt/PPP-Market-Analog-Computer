/**
 * MCP Agent Tools for Harmonic Alpha - Market Larynx Integration
 *
 * This module provides tools for fetching market sentiment using Voyage AI embeddings
 * and feeding them into the Rust/WASM Market Larynx engine.
 *
 * Usage:
 * const tools = new MarketSentimentTools({ voyageApiKey: 'your-key' });
 * const sentiment = await tools.fetchMarketSentiment('Bitcoin price analysis');
 * engine.set_market_sentiment_embedding(sentiment);
 */

const VOYAGE_API_URL = 'https://api.voyageai.com/v1/embeddings';
const DEFAULT_MODEL = 'voyage-large-2';

/**
 * Configuration for Market Sentiment Tools
 * @typedef {Object} MarketSentimentConfig
 * @property {string} voyageApiKey - Voyage AI API key
 * @property {string} [model] - Embedding model to use (default: voyage-large-2)
 * @property {number} [cacheTimeout] - Cache timeout in milliseconds (default: 60000)
 * @property {boolean} [enableCache] - Whether to enable caching (default: true)
 */

/**
 * Market sentiment result
 * @typedef {Object} SentimentResult
 * @property {Float64Array} embedding - The raw embedding vector
 * @property {number} sentiment - Normalized sentiment (0-1)
 * @property {string} regime - Interpreted market regime
 * @property {number} confidence - Confidence score
 * @property {string} query - Original query
 * @property {number} timestamp - Timestamp of the fetch
 */

/**
 * MCP Tool definitions for agent registration
 */
export const MCP_TOOL_DEFINITIONS = {
    fetchMarketSentiment: {
        name: 'fetchMarketSentiment',
        description: 'Fetch market sentiment from news/analysis text using Voyage AI embeddings. Returns a sentiment embedding that can be fed into the Market Larynx engine.',
        inputSchema: {
            type: 'object',
            properties: {
                query: {
                    type: 'string',
                    description: 'The market news, analysis, or query text to analyze'
                },
                context: {
                    type: 'string',
                    description: 'Optional context about the asset or market (e.g., "Bitcoin", "S&P500")',
                    default: ''
                }
            },
            required: ['query']
        }
    },
    fetchMultiSourceSentiment: {
        name: 'fetchMultiSourceSentiment',
        description: 'Aggregate sentiment from multiple news sources or queries',
        inputSchema: {
            type: 'object',
            properties: {
                queries: {
                    type: 'array',
                    items: { type: 'string' },
                    description: 'Array of news headlines or analysis texts'
                },
                weights: {
                    type: 'array',
                    items: { type: 'number' },
                    description: 'Optional weights for each query (default: equal weights)'
                }
            },
            required: ['queries']
        }
    },
    analyzePriceSentimentDivergence: {
        name: 'analyzePriceSentimentDivergence',
        description: 'Analyze divergence between price movement and sentiment (key for crash detection)',
        inputSchema: {
            type: 'object',
            properties: {
                price: {
                    type: 'number',
                    description: 'Current normalized price (0-1)'
                },
                sentimentQuery: {
                    type: 'string',
                    description: 'Current market news/sentiment text'
                }
            },
            required: ['price', 'sentimentQuery']
        }
    }
};

/**
 * Market Sentiment reference embeddings for regime classification
 * These are pre-computed embeddings for canonical sentiment phrases
 */
const SENTIMENT_ANCHORS = {
    bullish: 'Market showing strong bullish momentum with increasing volume and positive sentiment across all indicators.',
    bearish: 'Market exhibiting bearish divergence with declining volume and negative sentiment spreading through sectors.',
    neutral: 'Market trading sideways in consolidation with mixed signals and uncertain direction.',
    crash: 'Market in freefall with extreme fear, capitulation selling, and systemic risk concerns.',
    euphoria: 'Market experiencing euphoric buying with extreme greed and unsustainable valuations.'
};

/**
 * Simple in-memory cache for embeddings
 */
class EmbeddingCache {
    constructor(timeout = 60000) {
        this.cache = new Map();
        this.timeout = timeout;
    }

    get(key) {
        const entry = this.cache.get(key);
        if (!entry) return null;
        if (Date.now() - entry.timestamp > this.timeout) {
            this.cache.delete(key);
            return null;
        }
        return entry.value;
    }

    set(key, value) {
        this.cache.set(key, {
            value,
            timestamp: Date.now()
        });
    }

    clear() {
        this.cache.clear();
    }
}

/**
 * Market Sentiment Tools class
 */
export class MarketSentimentTools {
    /**
     * @param {MarketSentimentConfig} config
     */
    constructor(config = {}) {
        this.apiKey = config.voyageApiKey || process.env.VOYAGE_API_KEY || '';
        this.model = config.model || DEFAULT_MODEL;
        this.enableCache = config.enableCache !== false;
        this.cache = new EmbeddingCache(config.cacheTimeout || 60000);

        // Pre-computed anchor embeddings (populated on first use)
        this.anchorEmbeddings = null;
    }

    /**
     * Check if the tools are properly configured
     */
    isConfigured() {
        return Boolean(this.apiKey);
    }

    /**
     * Get embedding from Voyage AI
     * @param {string} text - Text to embed
     * @returns {Promise<Float64Array>} Embedding vector
     */
    async getEmbedding(text) {
        if (!text || typeof text !== 'string') {
            throw new Error('Text must be a non-empty string');
        }

        // Check cache
        if (this.enableCache) {
            const cached = this.cache.get(text);
            if (cached) return cached;
        }

        if (!this.apiKey) {
            // Return a deterministic fallback embedding for testing
            console.warn('Voyage API key not configured, using fallback embedding');
            return this._generateFallbackEmbedding(text);
        }

        const response = await fetch(VOYAGE_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            },
            body: JSON.stringify({
                model: this.model,
                input: text
            })
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`Voyage AI API error: ${response.status} - ${error}`);
        }

        const data = await response.json();
        const embedding = new Float64Array(data.data[0].embedding);

        // Cache the result
        if (this.enableCache) {
            this.cache.set(text, embedding);
        }

        return embedding;
    }

    /**
     * Generate a deterministic fallback embedding based on text content
     * Used when API key is not available (for testing/development)
     */
    _generateFallbackEmbedding(text) {
        const dim = 1024; // Standard embedding dimension
        const embedding = new Float64Array(dim);

        // Use text characteristics to generate pseudo-embedding
        const lower = text.toLowerCase();

        // Sentiment keywords and their weights
        const bullishKeywords = ['bullish', 'rally', 'surge', 'growth', 'positive', 'gains', 'up', 'buy', 'long'];
        const bearishKeywords = ['bearish', 'crash', 'fall', 'decline', 'negative', 'losses', 'down', 'sell', 'short'];
        const neutralKeywords = ['stable', 'sideways', 'consolidation', 'mixed', 'uncertain'];

        let bullishScore = bullishKeywords.filter(k => lower.includes(k)).length / bullishKeywords.length;
        let bearishScore = bearishKeywords.filter(k => lower.includes(k)).length / bearishKeywords.length;
        let neutralScore = neutralKeywords.filter(k => lower.includes(k)).length / neutralKeywords.length;

        // Normalize
        const total = bullishScore + bearishScore + neutralScore || 1;
        bullishScore /= total;
        bearishScore /= total;
        neutralScore /= total;

        // Generate pseudo-random embedding based on text hash and sentiment
        let hash = 0;
        for (let i = 0; i < text.length; i++) {
            hash = ((hash << 5) - hash) + text.charCodeAt(i);
            hash = hash & hash;
        }

        for (let i = 0; i < dim; i++) {
            // Combine hash-based pseudo-random with sentiment bias
            const pseudoRandom = Math.sin(hash * (i + 1)) * 0.5;
            const sentimentBias = (bullishScore - bearishScore) * 0.3;
            embedding[i] = pseudoRandom + sentimentBias * Math.cos(i * 0.1);
        }

        return embedding;
    }

    /**
     * Initialize anchor embeddings for regime classification
     */
    async initializeAnchors() {
        if (this.anchorEmbeddings) return;

        this.anchorEmbeddings = {};
        for (const [regime, text] of Object.entries(SENTIMENT_ANCHORS)) {
            this.anchorEmbeddings[regime] = await this.getEmbedding(text);
        }
    }

    /**
     * Compute cosine similarity between two embeddings
     */
    cosineSimilarity(a, b) {
        if (a.length !== b.length) {
            throw new Error('Embeddings must have same dimension');
        }

        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Classify sentiment into market regime based on anchor similarity
     */
    async classifyRegime(embedding) {
        await this.initializeAnchors();

        const similarities = {};
        let maxSim = -Infinity;
        let maxRegime = 'neutral';

        for (const [regime, anchor] of Object.entries(this.anchorEmbeddings)) {
            const sim = this.cosineSimilarity(embedding, anchor);
            similarities[regime] = sim;
            if (sim > maxSim) {
                maxSim = sim;
                maxRegime = regime;
            }
        }

        return {
            regime: maxRegime,
            confidence: (maxSim + 1) / 2, // Normalize to 0-1
            similarities
        };
    }

    /**
     * MCP Tool: Fetch market sentiment from query text
     * @param {string} query - Market news or analysis text
     * @param {string} [context] - Optional context
     * @returns {Promise<SentimentResult>}
     */
    async fetchMarketSentiment(query, context = '') {
        const fullQuery = context ? `${context}: ${query}` : query;
        const embedding = await this.getEmbedding(fullQuery);
        const classification = await this.classifyRegime(embedding);

        // Convert regime to sentiment score
        const regimeToSentiment = {
            bullish: 0.8,
            euphoria: 0.95,
            neutral: 0.5,
            bearish: 0.2,
            crash: 0.05
        };

        const sentiment = regimeToSentiment[classification.regime] || 0.5;

        return {
            embedding,
            sentiment,
            regime: classification.regime,
            confidence: classification.confidence,
            query: fullQuery,
            timestamp: Date.now()
        };
    }

    /**
     * MCP Tool: Fetch aggregated sentiment from multiple sources
     * @param {string[]} queries - Array of news/analysis texts
     * @param {number[]} [weights] - Optional weights for each query
     * @returns {Promise<SentimentResult>}
     */
    async fetchMultiSourceSentiment(queries, weights = null) {
        if (!queries || queries.length === 0) {
            throw new Error('At least one query is required');
        }

        // Default to equal weights
        const w = weights || queries.map(() => 1 / queries.length);
        if (w.length !== queries.length) {
            throw new Error('Weights array must match queries length');
        }

        // Normalize weights
        const totalWeight = w.reduce((a, b) => a + b, 0);
        const normalizedWeights = w.map(x => x / totalWeight);

        // Get all embeddings
        const embeddings = await Promise.all(queries.map(q => this.getEmbedding(q)));

        // Compute weighted average embedding
        const dim = embeddings[0].length;
        const aggregatedEmbedding = new Float64Array(dim);

        for (let i = 0; i < dim; i++) {
            let sum = 0;
            for (let j = 0; j < embeddings.length; j++) {
                sum += embeddings[j][i] * normalizedWeights[j];
            }
            aggregatedEmbedding[i] = sum;
        }

        const classification = await this.classifyRegime(aggregatedEmbedding);

        const regimeToSentiment = {
            bullish: 0.8,
            euphoria: 0.95,
            neutral: 0.5,
            bearish: 0.2,
            crash: 0.05
        };

        return {
            embedding: aggregatedEmbedding,
            sentiment: regimeToSentiment[classification.regime] || 0.5,
            regime: classification.regime,
            confidence: classification.confidence,
            query: `[Aggregated from ${queries.length} sources]`,
            timestamp: Date.now()
        };
    }

    /**
     * MCP Tool: Analyze price-sentiment divergence (crash detection)
     * @param {number} price - Normalized price (0-1)
     * @param {string} sentimentQuery - Current sentiment text
     * @returns {Promise<Object>} Divergence analysis
     */
    async analyzePriceSentimentDivergence(price, sentimentQuery) {
        const sentimentResult = await this.fetchMarketSentiment(sentimentQuery);

        const divergence = Math.abs(price - sentimentResult.sentiment);
        const tension = divergence * (1 + (1 - sentimentResult.confidence) * 0.5);

        // High divergence with bearish sentiment = crash risk
        const crashRisk = sentimentResult.regime === 'bearish' || sentimentResult.regime === 'crash'
            ? tension * 1.5
            : tension;

        return {
            price,
            sentiment: sentimentResult.sentiment,
            regime: sentimentResult.regime,
            divergence,
            tension: Math.min(tension, 1),
            crashRisk: Math.min(crashRisk, 1),
            recommendation: this._getRecommendation(price, sentimentResult.sentiment, crashRisk),
            embedding: sentimentResult.embedding,
            timestamp: Date.now()
        };
    }

    /**
     * Get trading recommendation based on analysis
     */
    _getRecommendation(price, sentiment, crashRisk) {
        if (crashRisk > 0.7) {
            return 'HIGH_ALERT: Significant crash risk detected. Consider reducing exposure.';
        }
        if (crashRisk > 0.5) {
            return 'WARNING: Elevated risk. Monitor closely for further divergence.';
        }
        if (price > sentiment + 0.3) {
            return 'CAUTION: Price ahead of sentiment. Potential overextension.';
        }
        if (sentiment > price + 0.3) {
            return 'OPPORTUNITY: Sentiment leading price. Potential upside.';
        }
        return 'STABLE: Price and sentiment aligned. Normal conditions.';
    }

    /**
     * Register tools with an MCP server
     * @param {Object} mcpServer - MCP server instance
     */
    registerWithMCP(mcpServer) {
        if (!mcpServer || typeof mcpServer.registerTool !== 'function') {
            console.warn('Invalid MCP server provided');
            return;
        }

        // Register fetchMarketSentiment
        mcpServer.registerTool(
            MCP_TOOL_DEFINITIONS.fetchMarketSentiment,
            async (params) => {
                const result = await this.fetchMarketSentiment(params.query, params.context);
                return {
                    content: [{
                        type: 'text',
                        text: JSON.stringify(result, (key, value) =>
                            value instanceof Float64Array ? Array.from(value) : value
                        , 2)
                    }]
                };
            }
        );

        // Register fetchMultiSourceSentiment
        mcpServer.registerTool(
            MCP_TOOL_DEFINITIONS.fetchMultiSourceSentiment,
            async (params) => {
                const result = await this.fetchMultiSourceSentiment(params.queries, params.weights);
                return {
                    content: [{
                        type: 'text',
                        text: JSON.stringify(result, (key, value) =>
                            value instanceof Float64Array ? Array.from(value) : value
                        , 2)
                    }]
                };
            }
        );

        // Register analyzePriceSentimentDivergence
        mcpServer.registerTool(
            MCP_TOOL_DEFINITIONS.analyzePriceSentimentDivergence,
            async (params) => {
                const result = await this.analyzePriceSentimentDivergence(params.price, params.sentimentQuery);
                return {
                    content: [{
                        type: 'text',
                        text: JSON.stringify(result, (key, value) =>
                            value instanceof Float64Array ? Array.from(value) : value
                        , 2)
                    }]
                };
            }
        );
    }
}

/**
 * Bridge to connect MCP tools with WASM Market Larynx
 */
export class MarketLarynxBridge {
    /**
     * @param {MarketSentimentTools} tools - MCP tools instance
     * @param {Object} wasmEngine - WebEngine WASM instance
     */
    constructor(tools, wasmEngine) {
        this.tools = tools;
        this.engine = wasmEngine;
        this.updateInterval = null;
        this.lastUpdate = 0;
    }

    /**
     * Initialize the bridge and enable market mode
     */
    initialize() {
        if (this.engine && typeof this.engine.enable_market_mode === 'function') {
            this.engine.enable_market_mode();
        }
    }

    /**
     * Update market state with new data
     * @param {number} price - Current price (normalized 0-1)
     * @param {string} newsQuery - Current market news/analysis
     */
    async update(price, newsQuery) {
        // Set price in WASM engine
        if (this.engine && typeof this.engine.set_market_price === 'function') {
            this.engine.set_market_price(price);
        }

        // Get sentiment from Voyage AI
        const sentimentResult = await this.tools.fetchMarketSentiment(newsQuery);

        // Feed embedding to WASM engine
        if (this.engine && typeof this.engine.set_market_sentiment_embedding === 'function') {
            this.engine.set_market_sentiment_embedding(Array.from(sentimentResult.embedding));
        }

        this.lastUpdate = Date.now();

        return {
            price,
            sentiment: sentimentResult.sentiment,
            regime: sentimentResult.regime,
            tension: this.engine?.get_market_tension?.() || 0,
            gammaActive: this.engine?.is_market_gamma_active?.() || false
        };
    }

    /**
     * Start automatic updates at given interval
     * @param {Function} priceProvider - Async function returning current price
     * @param {Function} newsProvider - Async function returning current news
     * @param {number} intervalMs - Update interval in milliseconds
     */
    startAutoUpdate(priceProvider, newsProvider, intervalMs = 5000) {
        this.stopAutoUpdate();

        this.updateInterval = setInterval(async () => {
            try {
                const price = await priceProvider();
                const news = await newsProvider();
                await this.update(price, news);
            } catch (error) {
                console.error('Market Larynx auto-update error:', error);
            }
        }, intervalMs);
    }

    /**
     * Stop automatic updates
     */
    stopAutoUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    /**
     * Get current market state from WASM engine
     */
    getState() {
        if (!this.engine) return null;

        return {
            tension: this.engine.get_market_tension?.() || 0,
            regime: this.engine.get_market_regime?.() || 'Unknown',
            gammaActive: this.engine.is_market_gamma_active?.() || false,
            crashProbability: this.engine.get_crash_probability?.() || 0,
            frequencies: this.engine.get_market_sonification_frequencies?.() || [110, 165, 220],
            lastUpdate: this.lastUpdate
        };
    }

    /**
     * Shutdown the bridge
     */
    shutdown() {
        this.stopAutoUpdate();
        if (this.engine && typeof this.engine.disable_market_mode === 'function') {
            this.engine.disable_market_mode();
        }
    }
}

export default MarketSentimentTools;
