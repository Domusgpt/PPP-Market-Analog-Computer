// File: src/components/AuditLog.tsx
// Right panel showing live audit log with SHA-256 hashes

import { useEffect, useRef } from 'react';
import type { LogEntryCompact, AuditEventType } from '../core/trace';

export interface AuditLogProps {
  /** Recent log entries */
  entries: LogEntryCompact[];
  /** Chain head (latest hash) */
  chainHead: string;
  /** Chain is valid */
  isValid: boolean;
}

/**
 * Get color for event type
 */
function getEventColor(eventType: AuditEventType): string {
  const colors: Record<AuditEventType, string> = {
    SYSTEM_INIT: '#00ffff',
    INFERENCE_STEP: '#00ff88',
    ENTROPY_INJECT: '#ffaa00',
    CONVEXITY_WARNING: '#ffaa00',
    CONVEXITY_VIOLATION: '#ff3366',
    POSITION_RESET: '#aa88ff',
    CONSTRAINT_APPLIED: '#ff88aa',
    ROTOR_UPDATE: '#88aaff',
    PROJECTION_UPDATE: '#88ffaa',
    USER_ACTION: '#ffffff',
  };
  return colors[eventType];
}

/**
 * Get color for convexity status
 */
function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    SAFE: '#00ff88',
    WARNING: '#ffaa00',
    VIOLATION: '#ff3366',
  };
  return colors[status] ?? '#666666';
}

/**
 * Format hash for display (truncate to 8 chars)
 */
function formatHash(hash: string): string {
  return hash.substring(0, 8) + '...';
}

/**
 * Format timestamp for display
 */
function formatTimestamp(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

/**
 * Single log entry row
 */
function LogEntryRow({ entry }: { entry: LogEntryCompact }) {
  return (
    <div
      style={{
        padding: '8px 12px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        fontSize: '10px',
        fontFamily: 'monospace',
        animation: 'fadeIn 0.3s ease-out',
      }}
    >
      {/* Header row */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '4px',
        }}
      >
        <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>
          #{entry.index}
        </span>
        <span style={{ color: 'rgba(255, 255, 255, 0.6)' }}>
          {formatTimestamp(entry.timestamp)}
        </span>
      </div>

      {/* Event type */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '4px',
        }}
      >
        <span
          style={{
            color: getEventColor(entry.eventType),
            fontSize: '9px',
            padding: '2px 6px',
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '3px',
          }}
        >
          {entry.eventType}
        </span>
        <span
          style={{
            color: getStatusColor(entry.convexityStatus),
            fontSize: '9px',
          }}
        >
          {entry.convexityStatus}
        </span>
      </div>

      {/* Hash */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '2px',
        }}
      >
        <div style={{ color: 'rgba(255, 255, 255, 0.3)' }}>
          prev: {formatHash(entry.previousHash)}
        </div>
        <div
          style={{
            color: '#00ffff',
            fontSize: '11px',
            letterSpacing: '0.5px',
          }}
        >
          hash: {entry.hash.substring(0, 16)}...
        </div>
      </div>
    </div>
  );
}

/**
 * AuditLog - Displays live hash chain
 */
export function AuditLog({ entries, chainHead, isValid }: AuditLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new entries are added
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries.length]);

  return (
    <div
      style={{
        width: '260px',
        height: '100%',
        backgroundColor: '#0d0d15',
        borderLeft: '1px solid #1a1a2e',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '16px',
          borderBottom: '1px solid #1a1a2e',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '8px',
          }}
        >
          <h2
            style={{
              color: '#ffffff',
              fontSize: '12px',
              fontWeight: 'bold',
              margin: 0,
              letterSpacing: '0.5px',
            }}
          >
            TRACE Audit Log
          </h2>
          <span
            style={{
              fontSize: '9px',
              padding: '2px 6px',
              backgroundColor: isValid
                ? 'rgba(0, 255, 136, 0.2)'
                : 'rgba(255, 51, 102, 0.2)',
              color: isValid ? '#00ff88' : '#ff3366',
              borderRadius: '3px',
            }}
          >
            {isValid ? 'VALID' : 'INVALID'}
          </span>
        </div>

        {/* Chain head */}
        <div
          style={{
            fontSize: '9px',
            color: 'rgba(255, 255, 255, 0.5)',
            marginBottom: '4px',
          }}
        >
          Chain Head:
        </div>
        <div
          style={{
            fontSize: '10px',
            fontFamily: 'monospace',
            color: '#00ffff',
            wordBreak: 'break-all',
            lineHeight: '1.4',
          }}
        >
          {chainHead}
        </div>
      </div>

      {/* Log entries */}
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflow: 'auto',
        }}
      >
        {entries.length === 0 ? (
          <div
            style={{
              padding: '20px',
              textAlign: 'center',
              color: 'rgba(255, 255, 255, 0.3)',
              fontSize: '11px',
            }}
          >
            No entries yet...
          </div>
        ) : (
          entries.map((entry) => <LogEntryRow key={entry.index} entry={entry} />)
        )}
      </div>

      {/* Footer */}
      <div
        style={{
          padding: '12px 16px',
          borderTop: '1px solid #1a1a2e',
          fontSize: '9px',
          color: 'rgba(255, 255, 255, 0.4)',
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>SHA-256 Hash Chain</span>
        <span>{entries.length} entries</span>
      </div>

      {/* Animation keyframes */}
      <style>
        {`
          @keyframes fadeIn {
            from {
              opacity: 0;
              transform: translateY(-10px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}
      </style>
    </div>
  );
}

export default AuditLog;
