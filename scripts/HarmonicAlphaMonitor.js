/**
 * Harmonic Alpha Automated Market Monitor
 *
 * Real-time API integration for automated market monitoring and event reactions.
 * Connects to live data feeds and triggers alerts/actions based on Harmonic Alpha signals.
 *
 * APIs Used:
 * - CoinGecko: Bitcoin price, market cap, volume
 * - Alternative.me: Crypto Fear & Greed Index
 * - Yahoo Finance (via proxy): VIX data
 * - Optional: Discord/Slack webhooks for alerts
 *
 * @author Harmonic Alpha Research Team
 * @version 1.0.0
 * @license MIT
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    // Polling intervals (milliseconds)
    intervals: {
        price: 60000,        // 1 minute for price updates
        sentiment: 300000,   // 5 minutes for sentiment
        fullAnalysis: 900000 // 15 minutes for full Harmonic Alpha analysis
    },

    // API endpoints
    apis: {
        coingecko: {
            price: 'https://api.coingecko.com/api/v3/simple/price',
            marketData: 'https://api.coingecko.com/api/v3/coins/bitcoin',
            params: {
                ids: 'bitcoin',
                vs_currencies: 'usd',
                include_24hr_change: true,
                include_24hr_vol: true,
                include_market_cap: true
            }
        },
        fearGreed: {
            url: 'https://api.alternative.me/fng/',
            params: { limit: 1 }
        },
        // VIX requires a proxy or CORS-enabled endpoint in production
        vix: {
            // Use a serverless function or backend proxy for Yahoo Finance
            proxyUrl: '/api/vix', // Configure your own proxy
            fallbackValue: 15.0   // Fallback if API unavailable
        }
    },

    // Alert thresholds
    thresholds: {
        crashProbability: {
            warning: 0.5,
            critical: 0.7,
            extreme: 0.85
        },
        tension: {
            elevated: 0.6,
            high: 0.75,
            critical: 0.85
        },
        priceChange: {
            significant: 0.03,  // 3%
            major: 0.05,        // 5%
            crash: 0.10         // 10%
        }
    },

    // Webhook configurations (set your URLs)
    webhooks: {
        discord: process.env.DISCORD_WEBHOOK_URL || null,
        slack: process.env.SLACK_WEBHOOK_URL || null,
        telegram: {
            botToken: process.env.TELEGRAM_BOT_TOKEN || null,
            chatId: process.env.TELEGRAM_CHAT_ID || null
        },
        custom: process.env.CUSTOM_WEBHOOK_URL || null
    },

    // Logging
    logging: {
        console: true,
        file: './logs/harmonic-alpha.log',
        level: 'info' // debug, info, warn, error
    }
};

// ============================================================================
// HARMONIC ALPHA ENGINE (Embedded for standalone operation)
// ============================================================================

const MUSICAL_INTERVALS = {
    UNISON: { ratio: 1.0, consonance: 1.0, name: 'Unison' },
    PERFECT_FIFTH: { ratio: 1.5, consonance: 0.95, name: 'Perfect Fifth' },
    PERFECT_FOURTH: { ratio: 4/3, consonance: 0.9, name: 'Perfect Fourth' },
    MAJOR_THIRD: { ratio: 1.25, consonance: 0.8, name: 'Major Third' },
    MINOR_THIRD: { ratio: 1.2, consonance: 0.7, name: 'Minor Third' },
    MINOR_SECOND: { ratio: 16/15, consonance: 0.2, name: 'Minor Second' },
    TRITONE: { ratio: Math.SQRT2, consonance: 0.0, name: 'Tritone' }
};

class HarmonicAlphaEngine {
    constructor(config = {}) {
        this.config = {
            gammaThreshold: config.gammaThreshold || 0.85,
            smoothingWindow: config.smoothingWindow || 30,
            tensionDecay: config.tensionDecay || 0.02,
            sentimentSensitivity: config.sentimentSensitivity || 1.5,
            baseFrequency: config.baseFrequency || 110,
            priceRange: config.priceRange || { min: 80000, max: 120000 },
            vix52WeekRange: config.vix52WeekRange || { low: 13.38, high: 60.13 }
        };

        this.state = {
            priceAlpha: 0.5,
            sentimentBeta: 0.5,
            tension: 0,
            tensionHistory: [],
            gammaActive: false,
            lastUpdate: null,
            predictions: []
        };
    }

    normalizePrice(price) {
        const { min, max } = this.config.priceRange;
        return Math.max(0, Math.min(1, (price - min) / (max - min)));
    }

    normalizeSentiment(fearGreed, vix) {
        // Fear & Greed: 0-100 → 0-1
        const fgNorm = fearGreed / 100;

        // VIX: inverted (low VIX = high sentiment/complacency)
        const vixNorm = 1 - Math.max(0, Math.min(1, (vix - 10) / 30));

        // Weighted combination
        return fgNorm * 0.6 + vixNorm * 0.4;
    }

    getVixPercentile(vix) {
        const { low, high } = this.config.vix52WeekRange;
        return (vix - low) / (high - low);
    }

    detectDivergence(fearGreed, vix, priceChange) {
        const vixPercentile = this.getVixPercentile(vix);
        const divergences = [];

        // Divergence 1: Crypto Fear + VIX Complacency
        if (fearGreed < 50 && vixPercentile < 0.15) {
            divergences.push({
                type: 'SENTIMENT_VIX_COMPLACENCY',
                severity: (1 - vixPercentile) * 0.5,
                message: `Fear & Greed at ${fearGreed} but VIX only ${(vixPercentile * 100).toFixed(1)}% above 52-week low`
            });
        }

        // Divergence 2: Price falling + VIX falling
        if (priceChange < -0.02 && vixPercentile < 0.20) {
            divergences.push({
                type: 'FALLING_PRICE_LOW_VIX',
                severity: Math.abs(priceChange) * 2,
                message: `Price down ${(priceChange * 100).toFixed(1)}% but VIX remains low`
            });
        }

        // Divergence 3: Extreme greed + high price
        if (fearGreed > 75 && this.state.priceAlpha > 0.8) {
            divergences.push({
                type: 'EUPHORIA_WARNING',
                severity: (fearGreed / 100) * this.state.priceAlpha,
                message: `Extreme greed (${fearGreed}) at elevated prices - potential top`
            });
        }

        return divergences;
    }

    getInterval(tension) {
        if (tension < 0.15) return MUSICAL_INTERVALS.UNISON;
        if (tension < 0.30) return MUSICAL_INTERVALS.PERFECT_FIFTH;
        if (tension < 0.45) return MUSICAL_INTERVALS.PERFECT_FOURTH;
        if (tension < 0.55) return MUSICAL_INTERVALS.MAJOR_THIRD;
        if (tension < 0.65) return MUSICAL_INTERVALS.MINOR_THIRD;
        if (tension < 0.80) return MUSICAL_INTERVALS.MINOR_SECOND;
        return MUSICAL_INTERVALS.TRITONE;
    }

    getRegime(tension) {
        if (tension < 0.15) return 'Bull';
        if (tension < 0.30) return 'MildBull';
        if (tension < 0.45) return 'Neutral';
        if (tension < 0.60) return 'MildBear';
        if (tension < 0.75) return 'Bear';
        if (tension < 0.90) return 'CrashRisk';
        return 'GammaEvent';
    }

    update(price, fearGreed, vix, priceChange24h = 0) {
        // Normalize inputs
        this.state.priceAlpha = this.normalizePrice(price);
        this.state.sentimentBeta = this.normalizeSentiment(fearGreed, vix);

        // Calculate raw tension (dialectic divergence)
        const rawTension = Math.abs(this.state.priceAlpha - this.state.sentimentBeta);

        // Detect divergences and adjust sensitivity
        const divergences = this.detectDivergence(fearGreed, vix, priceChange24h);
        const divergenceMultiplier = divergences.length > 0
            ? 1 + divergences.reduce((sum, d) => sum + d.severity, 0)
            : 1.0;

        // Apply sensitivity and divergence boost
        const targetTension = Math.min(1, rawTension * this.config.sentimentSensitivity * divergenceMultiplier);

        // Smooth tension update
        const decayFactor = 0.1;
        this.state.tension = this.state.tension + (targetTension - this.state.tension) * decayFactor;

        // Update history
        this.state.tensionHistory.push(this.state.tension);
        if (this.state.tensionHistory.length > this.config.smoothingWindow) {
            this.state.tensionHistory.shift();
        }

        // Calculate smoothed tension
        const smoothedTension = this.state.tensionHistory.reduce((a, b) => a + b, 0)
            / this.state.tensionHistory.length;

        // Check gamma threshold
        if (smoothedTension > this.config.gammaThreshold && !this.state.gammaActive) {
            this.state.gammaActive = true;
        } else if (smoothedTension < this.config.gammaThreshold * 0.7 && this.state.gammaActive) {
            this.state.gammaActive = false;
        }

        // Get musical mapping
        const interval = this.getInterval(smoothedTension);
        const regime = this.getRegime(smoothedTension);

        // Calculate crash probability
        const crashProbability = Math.min(1,
            smoothedTension * 0.6 +
            (divergences.length > 0 ? 0.2 : 0) +
            (this.state.gammaActive ? 0.2 : 0)
        );

        this.state.lastUpdate = new Date().toISOString();

        return {
            timestamp: this.state.lastUpdate,
            inputs: {
                price,
                fearGreed,
                vix,
                priceChange24h
            },
            normalized: {
                priceAlpha: this.state.priceAlpha,
                sentimentBeta: this.state.sentimentBeta
            },
            analysis: {
                tension: this.state.tension,
                smoothedTension,
                regime,
                crashProbability,
                gammaActive: this.state.gammaActive
            },
            musical: {
                interval: interval.name,
                ratio: interval.ratio,
                consonance: interval.consonance,
                frequencies: [
                    this.config.baseFrequency,
                    this.config.baseFrequency * interval.ratio,
                    this.config.baseFrequency * interval.ratio * interval.ratio
                ]
            },
            divergences,
            vixAnalysis: {
                percentile: this.getVixPercentile(vix),
                complacent: this.getVixPercentile(vix) < 0.15
            }
        };
    }

    generatePrediction(analysisResult) {
        const { crashProbability, regime, gammaActive } = analysisResult.analysis;
        const { divergences } = analysisResult;

        let riskLevel, outlook, timeframe, confidence;

        if (crashProbability > 0.7 || gammaActive) {
            riskLevel = 'HIGH';
            outlook = 'BEARISH - Significant correction likely';
            timeframe = '1-5 days';
            confidence = divergences.length > 0 ? 'High' : 'Medium-High';
        } else if (crashProbability > 0.5) {
            riskLevel = 'ELEVATED';
            outlook = 'CAUTIOUS - Increased downside risk';
            timeframe = '1-2 weeks';
            confidence = 'Medium-High';
        } else if (crashProbability > 0.3) {
            riskLevel = 'MODERATE';
            outlook = 'NEUTRAL - Monitor for changes';
            timeframe = '2-4 weeks';
            confidence = 'Medium';
        } else {
            riskLevel = 'LOW';
            outlook = 'STABLE - No immediate crash signals';
            timeframe = 'Ongoing';
            confidence = 'Medium';
        }

        const prediction = {
            timestamp: new Date().toISOString(),
            validUntil: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
            riskLevel,
            outlook,
            timeframe,
            confidence,
            crashProbability,
            regime,
            keyFactors: divergences.map(d => d.message),
            verification: this.generateVerificationCriteria(riskLevel, analysisResult.inputs.price)
        };

        this.state.predictions.push(prediction);
        return prediction;
    }

    generateVerificationCriteria(riskLevel, currentPrice) {
        switch (riskLevel) {
            case 'HIGH':
                return {
                    correct: `Price drops >5% (below $${(currentPrice * 0.95).toLocaleString()}) within 5 days`,
                    incorrect: `Price stays above $${(currentPrice * 0.95).toLocaleString()}`
                };
            case 'ELEVATED':
                return {
                    correct: `Price drops >3% (below $${(currentPrice * 0.97).toLocaleString()}) within 2 weeks`,
                    incorrect: `Price stays above $${(currentPrice * 0.97).toLocaleString()}`
                };
            default:
                return {
                    correct: `Price stays within ±5% for 1 week`,
                    incorrect: `Price drops >5% unexpectedly`
                };
        }
    }
}

// ============================================================================
// API DATA FETCHERS
// ============================================================================

class MarketDataFetcher {
    constructor(config) {
        this.config = config;
        this.cache = {
            price: null,
            sentiment: null,
            lastFetch: {}
        };
    }

    async fetchBitcoinPrice() {
        try {
            const params = new URLSearchParams(this.config.apis.coingecko.params);
            const url = `${this.config.apis.coingecko.price}?${params}`;

            const response = await fetch(url);
            if (!response.ok) throw new Error(`CoinGecko API error: ${response.status}`);

            const data = await response.json();

            this.cache.price = {
                price: data.bitcoin.usd,
                change24h: data.bitcoin.usd_24h_change / 100,
                volume24h: data.bitcoin.usd_24h_vol,
                marketCap: data.bitcoin.usd_market_cap,
                timestamp: new Date().toISOString()
            };

            return this.cache.price;
        } catch (error) {
            console.error('Error fetching Bitcoin price:', error.message);
            return this.cache.price; // Return cached data
        }
    }

    async fetchFearGreedIndex() {
        try {
            const response = await fetch(this.config.apis.fearGreed.url);
            if (!response.ok) throw new Error(`Fear & Greed API error: ${response.status}`);

            const data = await response.json();

            this.cache.sentiment = {
                value: parseInt(data.data[0].value),
                classification: data.data[0].value_classification,
                timestamp: data.data[0].timestamp,
                fetchedAt: new Date().toISOString()
            };

            return this.cache.sentiment;
        } catch (error) {
            console.error('Error fetching Fear & Greed:', error.message);
            return this.cache.sentiment;
        }
    }

    async fetchVIX() {
        // VIX requires a backend proxy due to CORS
        // This is a placeholder - implement your own proxy
        try {
            if (this.config.apis.vix.proxyUrl) {
                const response = await fetch(this.config.apis.vix.proxyUrl);
                if (response.ok) {
                    const data = await response.json();
                    return data.vix;
                }
            }
            // Return fallback or last known value
            return this.config.apis.vix.fallbackValue;
        } catch (error) {
            console.error('Error fetching VIX:', error.message);
            return this.config.apis.vix.fallbackValue;
        }
    }

    async fetchAllData() {
        const [priceData, sentimentData, vix] = await Promise.all([
            this.fetchBitcoinPrice(),
            this.fetchFearGreedIndex(),
            this.fetchVIX()
        ]);

        return {
            price: priceData?.price || 0,
            priceChange24h: priceData?.change24h || 0,
            volume24h: priceData?.volume24h || 0,
            fearGreed: sentimentData?.value || 50,
            fearGreedClass: sentimentData?.classification || 'Unknown',
            vix: vix,
            fetchedAt: new Date().toISOString()
        };
    }
}

// ============================================================================
// ALERT SYSTEM
// ============================================================================

class AlertSystem {
    constructor(config) {
        this.config = config;
        this.alertHistory = [];
        this.cooldowns = {};
    }

    async sendDiscordAlert(message, level = 'info') {
        if (!this.config.webhooks.discord) return false;

        const colors = {
            info: 0x3498db,
            warning: 0xf39c12,
            critical: 0xe74c3c,
            success: 0x2ecc71
        };

        const payload = {
            embeds: [{
                title: '🎵 Harmonic Alpha Alert',
                description: message.text,
                color: colors[level] || colors.info,
                fields: message.fields || [],
                timestamp: new Date().toISOString(),
                footer: { text: 'Harmonic Alpha Market Monitor' }
            }]
        };

        try {
            const response = await fetch(this.config.webhooks.discord, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            return response.ok;
        } catch (error) {
            console.error('Discord alert failed:', error.message);
            return false;
        }
    }

    async sendSlackAlert(message, level = 'info') {
        if (!this.config.webhooks.slack) return false;

        const emojis = {
            info: ':information_source:',
            warning: ':warning:',
            critical: ':rotating_light:',
            success: ':white_check_mark:'
        };

        const payload = {
            blocks: [
                {
                    type: 'header',
                    text: { type: 'plain_text', text: `${emojis[level]} Harmonic Alpha Alert` }
                },
                {
                    type: 'section',
                    text: { type: 'mrkdwn', text: message.text }
                },
                ...(message.fields ? [{
                    type: 'section',
                    fields: message.fields.map(f => ({
                        type: 'mrkdwn',
                        text: `*${f.name}:* ${f.value}`
                    }))
                }] : [])
            ]
        };

        try {
            const response = await fetch(this.config.webhooks.slack, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            return response.ok;
        } catch (error) {
            console.error('Slack alert failed:', error.message);
            return false;
        }
    }

    async sendTelegramAlert(message, level = 'info') {
        const { botToken, chatId } = this.config.webhooks.telegram;
        if (!botToken || !chatId) return false;

        const emojis = { info: 'ℹ️', warning: '⚠️', critical: '🚨', success: '✅' };

        const text = `${emojis[level]} *Harmonic Alpha Alert*\n\n${message.text}` +
            (message.fields ? '\n\n' + message.fields.map(f => `*${f.name}:* ${f.value}`).join('\n') : '');

        try {
            const url = `https://api.telegram.org/bot${botToken}/sendMessage`;
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chat_id: chatId,
                    text: text,
                    parse_mode: 'Markdown'
                })
            });
            return response.ok;
        } catch (error) {
            console.error('Telegram alert failed:', error.message);
            return false;
        }
    }

    async sendAlert(message, level = 'info') {
        // Check cooldown (prevent spam)
        const cooldownKey = `${level}-${message.type || 'general'}`;
        const cooldownMs = level === 'critical' ? 300000 : 900000; // 5 min for critical, 15 min otherwise

        if (this.cooldowns[cooldownKey] && Date.now() - this.cooldowns[cooldownKey] < cooldownMs) {
            return false; // Still in cooldown
        }

        this.cooldowns[cooldownKey] = Date.now();

        // Log alert
        const alert = {
            timestamp: new Date().toISOString(),
            level,
            message: message.text,
            fields: message.fields
        };
        this.alertHistory.push(alert);

        if (this.config.logging.console) {
            const prefix = { info: '📊', warning: '⚠️', critical: '🚨', success: '✅' };
            console.log(`${prefix[level] || '📊'} [${level.toUpperCase()}] ${message.text}`);
        }

        // Send to all configured channels
        const results = await Promise.all([
            this.sendDiscordAlert(message, level),
            this.sendSlackAlert(message, level),
            this.sendTelegramAlert(message, level)
        ]);

        return results.some(r => r);
    }

    createPredictionAlert(prediction, analysis) {
        const levelMap = { HIGH: 'critical', ELEVATED: 'warning', MODERATE: 'info', LOW: 'info' };

        return {
            type: 'prediction',
            text: `**${prediction.riskLevel} RISK** - ${prediction.outlook}`,
            fields: [
                { name: 'Crash Probability', value: `${(prediction.crashProbability * 100).toFixed(1)}%`, inline: true },
                { name: 'Regime', value: prediction.regime, inline: true },
                { name: 'Timeframe', value: prediction.timeframe, inline: true },
                { name: 'Consonance', value: analysis.musical.consonance.toFixed(2), inline: true },
                { name: 'Interval', value: analysis.musical.interval, inline: true },
                { name: 'Confidence', value: prediction.confidence, inline: true },
                ...(prediction.keyFactors.length > 0
                    ? [{ name: 'Key Factors', value: prediction.keyFactors.join('\n'), inline: false }]
                    : [])
            ],
            level: levelMap[prediction.riskLevel] || 'info'
        };
    }
}

// ============================================================================
// MAIN MONITOR CLASS
// ============================================================================

class HarmonicAlphaMonitor {
    constructor(config = CONFIG) {
        this.config = config;
        this.engine = new HarmonicAlphaEngine();
        this.dataFetcher = new MarketDataFetcher(config);
        this.alertSystem = new AlertSystem(config);

        this.state = {
            running: false,
            lastAnalysis: null,
            lastPrediction: null,
            analysisHistory: [],
            intervals: {}
        };

        this.eventHandlers = {
            onAnalysis: [],
            onPrediction: [],
            onAlert: [],
            onRegimeChange: [],
            onGammaEvent: []
        };
    }

    on(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        }
        return this;
    }

    emit(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Event handler error (${event}):`, error);
                }
            });
        }
    }

    async runAnalysis() {
        try {
            // Fetch current market data
            const marketData = await this.dataFetcher.fetchAllData();

            if (!marketData.price) {
                console.warn('No price data available, skipping analysis');
                return null;
            }

            // Run Harmonic Alpha analysis
            const analysis = this.engine.update(
                marketData.price,
                marketData.fearGreed,
                marketData.vix,
                marketData.priceChange24h
            );

            // Generate prediction
            const prediction = this.engine.generatePrediction(analysis);

            // Check for regime change
            if (this.state.lastAnalysis &&
                this.state.lastAnalysis.analysis.regime !== analysis.analysis.regime) {
                this.emit('onRegimeChange', {
                    from: this.state.lastAnalysis.analysis.regime,
                    to: analysis.analysis.regime,
                    analysis
                });
            }

            // Check for gamma event
            if (analysis.analysis.gammaActive && !this.state.lastAnalysis?.analysis.gammaActive) {
                this.emit('onGammaEvent', { analysis, prediction });

                await this.alertSystem.sendAlert({
                    type: 'gamma',
                    text: '🚨 **GAMMA EVENT TRIGGERED** - Extreme market tension detected!',
                    fields: [
                        { name: 'Tension', value: analysis.analysis.smoothedTension.toFixed(3), inline: true },
                        { name: 'Regime', value: analysis.analysis.regime, inline: true }
                    ]
                }, 'critical');
            }

            // Store state
            this.state.lastAnalysis = analysis;
            this.state.lastPrediction = prediction;
            this.state.analysisHistory.push({ analysis, prediction, timestamp: new Date().toISOString() });

            // Trim history
            if (this.state.analysisHistory.length > 1000) {
                this.state.analysisHistory = this.state.analysisHistory.slice(-500);
            }

            // Emit events
            this.emit('onAnalysis', analysis);
            this.emit('onPrediction', prediction);

            // Send alerts based on thresholds
            await this.checkAndSendAlerts(analysis, prediction);

            return { analysis, prediction, marketData };
        } catch (error) {
            console.error('Analysis error:', error);
            return null;
        }
    }

    async checkAndSendAlerts(analysis, prediction) {
        const { crashProbability, regime } = analysis.analysis;
        const { thresholds } = this.config;

        // Critical alert
        if (crashProbability >= thresholds.crashProbability.extreme) {
            const alert = this.alertSystem.createPredictionAlert(prediction, analysis);
            await this.alertSystem.sendAlert(alert, 'critical');
            this.emit('onAlert', { level: 'critical', analysis, prediction });
        }
        // Warning alert
        else if (crashProbability >= thresholds.crashProbability.warning) {
            const alert = this.alertSystem.createPredictionAlert(prediction, analysis);
            await this.alertSystem.sendAlert(alert, 'warning');
            this.emit('onAlert', { level: 'warning', analysis, prediction });
        }

        // Divergence alerts
        if (analysis.divergences.length > 0) {
            for (const divergence of analysis.divergences) {
                await this.alertSystem.sendAlert({
                    type: 'divergence',
                    text: `📉 **Divergence Detected:** ${divergence.message}`,
                    fields: [
                        { name: 'Type', value: divergence.type, inline: true },
                        { name: 'Severity', value: divergence.severity.toFixed(3), inline: true }
                    ]
                }, divergence.severity > 0.4 ? 'warning' : 'info');
            }
        }
    }

    start() {
        if (this.state.running) {
            console.warn('Monitor already running');
            return this;
        }

        console.log('🎵 Starting Harmonic Alpha Monitor...');
        this.state.running = true;

        // Initial analysis
        this.runAnalysis();

        // Set up polling intervals
        this.state.intervals.fullAnalysis = setInterval(
            () => this.runAnalysis(),
            this.config.intervals.fullAnalysis
        );

        console.log(`✅ Monitor started. Analysis interval: ${this.config.intervals.fullAnalysis / 1000}s`);
        return this;
    }

    stop() {
        console.log('🛑 Stopping Harmonic Alpha Monitor...');
        this.state.running = false;

        Object.values(this.state.intervals).forEach(interval => clearInterval(interval));
        this.state.intervals = {};

        console.log('✅ Monitor stopped');
        return this;
    }

    getStatus() {
        return {
            running: this.state.running,
            lastAnalysis: this.state.lastAnalysis,
            lastPrediction: this.state.lastPrediction,
            analysisCount: this.state.analysisHistory.length,
            alertCount: this.alertSystem.alertHistory.length
        };
    }

    getHistory(limit = 100) {
        return this.state.analysisHistory.slice(-limit);
    }
}

// ============================================================================
// EXPRESS SERVER FOR WEBHOOK ENDPOINTS
// ============================================================================

function createAPIServer(monitor, port = 3000) {
    // This requires Express - for Node.js deployment
    // npm install express

    try {
        const express = require('express');
        const app = express();

        app.use(express.json());

        // Health check
        app.get('/health', (req, res) => {
            res.json({ status: 'ok', ...monitor.getStatus() });
        });

        // Current prediction
        app.get('/api/prediction', (req, res) => {
            const result = {
                prediction: monitor.state.lastPrediction,
                analysis: monitor.state.lastAnalysis,
                timestamp: new Date().toISOString()
            };
            res.json(result);
        });

        // Trigger manual analysis
        app.post('/api/analyze', async (req, res) => {
            const result = await monitor.runAnalysis();
            res.json(result);
        });

        // History
        app.get('/api/history', (req, res) => {
            const limit = parseInt(req.query.limit) || 100;
            res.json(monitor.getHistory(limit));
        });

        // Webhook receiver (for external signals)
        app.post('/webhook/signal', async (req, res) => {
            const { type, data } = req.body;
            console.log(`Received external signal: ${type}`, data);
            // Process external signals here
            res.json({ received: true, type });
        });

        app.listen(port, () => {
            console.log(`📡 API server running on port ${port}`);
        });

        return app;
    } catch (error) {
        console.log('Express not available, API server not started');
        return null;
    }
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
    HarmonicAlphaMonitor,
    HarmonicAlphaEngine,
    MarketDataFetcher,
    AlertSystem,
    createAPIServer,
    CONFIG
};

// Browser global
if (typeof window !== 'undefined') {
    window.HarmonicAlphaMonitor = HarmonicAlphaMonitor;
    window.HarmonicAlphaEngine = HarmonicAlphaEngine;
}

// Node.js CLI
if (typeof process !== 'undefined' && process.argv[1]?.includes('HarmonicAlphaMonitor')) {
    const monitor = new HarmonicAlphaMonitor();

    monitor
        .on('onPrediction', (prediction) => {
            console.log('\n📊 New Prediction:', prediction.riskLevel, '-', prediction.outlook);
        })
        .on('onGammaEvent', () => {
            console.log('\n🚨 GAMMA EVENT!');
        })
        .start();

    // Keep process running
    process.on('SIGINT', () => {
        monitor.stop();
        process.exit(0);
    });
}
