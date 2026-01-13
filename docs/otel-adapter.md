# OpenTelemetry Adapter Guide

PPP exposes a lightweight adapter that maps live telemetry and audit events to OpenTelemetry-compatible writers without requiring OTel dependencies in the bundle.

## Configuration

```js
window.PPP_CONFIG = {
  otel: {
    enabled: true,
    namespace: 'ppp',
    attributes: { service: 'ppp-info-site' },
    metricWriter: ({ name, value, kind, attributes }) => {
      // forward to your OTel meter (counter/gauge)
      console.log('metric', name, value, kind, attributes);
    },
    logWriter: ({ name, body, attributes }) => {
      // forward to your OTel logger
      console.log('log', name, body, attributes);
    }
  }
};
```

## Emitted Metrics

- `ppp.live.frames` (counter)
- `ppp.live.drops` (counter)
- `ppp.live.parse_errors` (counter)
- `ppp.live.reconnects` (counter)
- `ppp.live.latency_ms` (gauge)
- `ppp.live.latency_avg_ms` (gauge)

Each metric carries `mode` and `connected` attributes.

## Emitted Logs

- `ppp.live.health` for health changes.
- `ppp.audit.evidence` for evidence events.
- `ppp.audit.batch` for batch events (`stage: sealed | anchored`).

## Mapping to OTel SDKs

Use `metricWriter` to call your OTel `Counter.add()` or `Gauge.record()` APIs, and `logWriter` to emit OTel logs/spans as appropriate. The adapter intentionally stays dependency-free so you can plug in any OTel runtime.
