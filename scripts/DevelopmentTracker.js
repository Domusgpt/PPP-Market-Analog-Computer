const cloneHighlights = (values) => {
    if (!Array.isArray(values)) {
        return [];
    }
    return values
        .filter((item) => typeof item === 'string' && item.trim().length > 0)
        .map((item) => item.trim());
};

const normalizeEntry = (entry, fallbackSequence, fallbackIndex) => {
    const sequence = Number.isFinite(entry?.sequence) ? entry.sequence : fallbackSequence;
    const baseId = typeof entry?.id === 'string' && entry.id.trim().length
        ? entry.id.trim()
        : `session-${String(sequence || fallbackIndex).padStart(2, '0')}`;
    const title = typeof entry?.title === 'string' && entry.title.trim().length
        ? entry.title.trim()
        : `Session ${String(sequence || fallbackIndex).padStart(2, '0')}`;
    const summary = typeof entry?.summary === 'string' ? entry.summary.trim() : '';
    const analysis = typeof entry?.analysis === 'string' ? entry.analysis.trim() : '';
    const highlights = cloneHighlights(entry?.highlights);
    return {
        id: baseId,
        sequence: Number.isFinite(sequence) ? sequence : fallbackIndex,
        title,
        summary,
        highlights,
        analysis
    };
};

const cloneEntry = (entry) => ({
    id: entry.id,
    sequence: entry.sequence,
    title: entry.title,
    summary: entry.summary,
    analysis: entry.analysis,
    highlights: entry.highlights.slice()
});

export class DevelopmentTracker {
    constructor({ entries = [] } = {}) {
        this.entries = [];
        this.setEntries(entries);
    }

    setEntries(entries = []) {
        const normalized = [];
        if (Array.isArray(entries)) {
            entries.forEach((entry, index) => {
                normalized.push(normalizeEntry(entry, index + 1, index + 1));
            });
        }
        this.entries = normalized.sort((a, b) => a.sequence - b.sequence);
    }

    addEntry(entry) {
        if (!entry || typeof entry !== 'object') {
            return null;
        }
        const normalized = normalizeEntry(entry, this.entries.length + 1, this.entries.length + 1);
        const existingIndex = this.entries.findIndex((item) => item.id === normalized.id || item.sequence === normalized.sequence);
        if (existingIndex >= 0) {
            this.entries.splice(existingIndex, 1, normalized);
        } else {
            this.entries.push(normalized);
        }
        this.entries.sort((a, b) => a.sequence - b.sequence);
        return cloneEntry(normalized);
    }

    addEntries(entries = []) {
        if (!Array.isArray(entries)) {
            return;
        }
        entries.forEach((entry) => this.addEntry(entry));
    }

    getEntries() {
        return this.entries.map((entry) => cloneEntry(entry));
    }

    getLatestEntry() {
        if (!this.entries.length) {
            return null;
        }
        return cloneEntry(this.entries[this.entries.length - 1]);
    }

    getSummary() {
        const latest = this.getLatestEntry();
        return {
            sessionCount: this.entries.length,
            latestSequence: latest ? latest.sequence : 0,
            latestTitle: latest ? latest.title : '',
            latestAnalysis: latest ? latest.analysis : ''
        };
    }
}
