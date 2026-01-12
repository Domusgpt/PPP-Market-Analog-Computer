// File: src/components/ChatPanel.tsx
// Chat interface for LLM integration

import { useState, useRef, useEffect, useCallback } from 'react';
import type { ChatMessage, LLMProviderConfig, SimulationCommand } from '../services/llm';

export interface ChatPanelProps {
  messages: ChatMessage[];
  isLoading: boolean;
  isOpen: boolean;
  provider: LLMProviderConfig;
  onSendMessage: (message: string) => void;
  onCommandExecuted?: (command: SimulationCommand) => void;
  onProviderChange: (config: Partial<LLMProviderConfig>) => void;
  onClearHistory: () => void;
  onClose: () => void;
}

/**
 * Provider selector component
 */
function ProviderSelector({
  config,
  onConfigChange,
}: {
  config: LLMProviderConfig;
  onConfigChange: (config: Partial<LLMProviderConfig>) => void;
}) {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div style={{ marginBottom: '12px' }}>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
        {(['demo', 'anthropic', 'openai', 'ollama'] as const).map((provider) => (
          <button
            key={provider}
            onClick={() => onConfigChange({ provider: provider === 'demo' ? 'custom' : provider, model: provider === 'demo' ? 'demo' : config.model })}
            style={{
              flex: 1,
              padding: '6px 8px',
              fontSize: '10px',
              backgroundColor:
                (config.provider === provider) || (config.provider === 'custom' && config.model === 'demo' && provider === 'demo')
                  ? '#1a2e2e'
                  : '#1a1a2e',
              border: `1px solid ${
                (config.provider === provider) || (config.provider === 'custom' && config.model === 'demo' && provider === 'demo')
                  ? '#00ffff'
                  : '#333366'
              }`,
              borderRadius: '4px',
              color: '#ffffff',
              cursor: 'pointer',
              textTransform: 'uppercase',
            }}
          >
            {provider}
          </button>
        ))}
      </div>

      {config.provider !== 'custom' && (
        <button
          onClick={() => setShowSettings(!showSettings)}
          style={{
            width: '100%',
            padding: '6px',
            fontSize: '10px',
            backgroundColor: 'transparent',
            border: '1px dashed #333366',
            borderRadius: '4px',
            color: 'rgba(255, 255, 255, 0.5)',
            cursor: 'pointer',
          }}
        >
          {showSettings ? '▼ Hide Settings' : '▶ API Settings'}
        </button>
      )}

      {showSettings && config.provider !== 'custom' && (
        <div
          style={{
            marginTop: '8px',
            padding: '12px',
            backgroundColor: 'rgba(0, 0, 0, 0.3)',
            borderRadius: '6px',
          }}
        >
          <div style={{ marginBottom: '8px' }}>
            <label style={{ display: 'block', color: 'rgba(255, 255, 255, 0.6)', fontSize: '10px', marginBottom: '4px' }}>
              API Key
            </label>
            <input
              type="password"
              value={config.apiKey ?? ''}
              onChange={(e) => onConfigChange({ apiKey: e.target.value })}
              placeholder={`Enter ${config.provider} API key`}
              style={{
                width: '100%',
                padding: '8px',
                fontSize: '11px',
                backgroundColor: '#0d0d15',
                border: '1px solid #333366',
                borderRadius: '4px',
                color: '#ffffff',
                outline: 'none',
              }}
            />
          </div>

          {config.provider === 'ollama' && (
            <div style={{ marginBottom: '8px' }}>
              <label style={{ display: 'block', color: 'rgba(255, 255, 255, 0.6)', fontSize: '10px', marginBottom: '4px' }}>
                Base URL
              </label>
              <input
                type="text"
                value={config.baseUrl ?? 'http://localhost:11434'}
                onChange={(e) => onConfigChange({ baseUrl: e.target.value })}
                placeholder="http://localhost:11434"
                style={{
                  width: '100%',
                  padding: '8px',
                  fontSize: '11px',
                  backgroundColor: '#0d0d15',
                  border: '1px solid #333366',
                  borderRadius: '4px',
                  color: '#ffffff',
                  outline: 'none',
                }}
              />
            </div>
          )}

          <div>
            <label style={{ display: 'block', color: 'rgba(255, 255, 255, 0.6)', fontSize: '10px', marginBottom: '4px' }}>
              Model
            </label>
            <input
              type="text"
              value={config.model}
              onChange={(e) => onConfigChange({ model: e.target.value })}
              placeholder={
                config.provider === 'anthropic' ? 'claude-sonnet-4-20250514' :
                config.provider === 'openai' ? 'gpt-4' :
                config.provider === 'ollama' ? 'llama2' : 'model'
              }
              style={{
                width: '100%',
                padding: '8px',
                fontSize: '11px',
                backgroundColor: '#0d0d15',
                border: '1px solid #333366',
                borderRadius: '4px',
                color: '#ffffff',
                outline: 'none',
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Message component
 */
function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        marginBottom: '12px',
      }}
    >
      <div
        style={{
          maxWidth: '85%',
          padding: '10px 14px',
          borderRadius: isUser ? '12px 12px 0 12px' : '12px 12px 12px 0',
          backgroundColor: isUser ? 'rgba(0, 255, 136, 0.15)' : 'rgba(0, 255, 255, 0.1)',
          border: `1px solid ${isUser ? 'rgba(0, 255, 136, 0.3)' : 'rgba(0, 255, 255, 0.2)'}`,
        }}
      >
        <div
          style={{
            color: '#ffffff',
            fontSize: '12px',
            lineHeight: '1.5',
            whiteSpace: 'pre-wrap',
          }}
        >
          {message.content}
        </div>
      </div>
      <div
        style={{
          fontSize: '9px',
          color: 'rgba(255, 255, 255, 0.3)',
          marginTop: '4px',
          fontFamily: 'monospace',
        }}
      >
        {new Date(message.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
}

/**
 * Loading indicator
 */
function LoadingIndicator() {
  return (
    <div
      style={{
        display: 'flex',
        gap: '4px',
        padding: '12px',
        marginBottom: '12px',
      }}
    >
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: '#00ffff',
            animation: `pulse 1.4s ease-in-out ${i * 0.2}s infinite`,
            opacity: 0.6,
          }}
        />
      ))}
      <style>{`
        @keyframes pulse {
          0%, 80%, 100% { transform: scale(1); opacity: 0.4; }
          40% { transform: scale(1.2); opacity: 1; }
        }
      `}</style>
    </div>
  );
}

/**
 * Chat Panel Component
 */
export function ChatPanel({
  messages,
  isLoading,
  isOpen,
  provider,
  onSendMessage,
  onProviderChange,
  onClearHistory,
  onClose,
}: ChatPanelProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  const handleSubmit = useCallback(() => {
    const trimmed = input.trim();
    if (trimmed && !isLoading) {
      onSendMessage(trimmed);
      setInput('');
    }
  }, [input, isLoading, onSendMessage]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        width: '380px',
        maxHeight: '70vh',
        backgroundColor: '#0d0d15',
        border: '1px solid #1a1a2e',
        borderRadius: '12px',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
        zIndex: 1000,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '12px 16px',
          borderBottom: '1px solid #1a1a2e',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          backgroundColor: 'rgba(0, 255, 255, 0.05)',
        }}
      >
        <div>
          <h3 style={{ color: '#ffffff', fontSize: '14px', margin: 0 }}>
            Geometric Cognition Chat
          </h3>
          <p style={{ color: 'rgba(255, 255, 255, 0.5)', fontSize: '10px', margin: '2px 0 0 0' }}>
            {provider.model === 'demo' ? 'Demo Mode' : `${provider.provider} / ${provider.model}`}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={onClearHistory}
            style={{
              background: 'none',
              border: 'none',
              color: 'rgba(255, 255, 255, 0.5)',
              cursor: 'pointer',
              fontSize: '12px',
              padding: '4px 8px',
            }}
            title="Clear history"
          >
            Clear
          </button>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'rgba(255, 255, 255, 0.5)',
              cursor: 'pointer',
              fontSize: '18px',
              lineHeight: 1,
            }}
          >
            ×
          </button>
        </div>
      </div>

      {/* Provider selector */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid #1a1a2e' }}>
        <ProviderSelector config={provider} onConfigChange={onProviderChange} />
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '16px',
          minHeight: '200px',
          maxHeight: '400px',
        }}
      >
        {messages.length === 0 ? (
          <div
            style={{
              textAlign: 'center',
              color: 'rgba(255, 255, 255, 0.3)',
              fontSize: '12px',
              padding: '40px 20px',
            }}
          >
            <div style={{ fontSize: '24px', marginBottom: '12px' }}>⬡</div>
            <p>Start a conversation with the geometric cognition engine.</p>
            <p style={{ marginTop: '8px' }}>Try: "What's the current state?" or "Explore the manifold"</p>
          </div>
        ) : (
          messages.map((msg) => <Message key={msg.id} message={msg} />)
        )}
        {isLoading && <LoadingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div
        style={{
          padding: '12px 16px',
          borderTop: '1px solid #1a1a2e',
          backgroundColor: 'rgba(0, 0, 0, 0.2)',
        }}
      >
        <div style={{ display: 'flex', gap: '8px' }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the geometric state..."
            disabled={isLoading}
            rows={2}
            style={{
              flex: 1,
              padding: '10px 12px',
              fontSize: '12px',
              backgroundColor: '#1a1a2e',
              border: '1px solid #333366',
              borderRadius: '8px',
              color: '#ffffff',
              outline: 'none',
              resize: 'none',
              fontFamily: 'inherit',
            }}
          />
          <button
            onClick={handleSubmit}
            disabled={isLoading || !input.trim()}
            style={{
              padding: '10px 16px',
              backgroundColor: isLoading || !input.trim() ? '#1a1a2e' : '#00ff88',
              border: 'none',
              borderRadius: '8px',
              color: isLoading || !input.trim() ? 'rgba(255, 255, 255, 0.3)' : '#000000',
              cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: 'bold',
              transition: 'all 0.2s',
            }}
          >
            ➤
          </button>
        </div>
        <div
          style={{
            marginTop: '8px',
            fontSize: '9px',
            color: 'rgba(255, 255, 255, 0.3)',
            textAlign: 'center',
          }}
        >
          Press Enter to send • Shift+Enter for new line
        </div>
      </div>
    </div>
  );
}

/**
 * Chat toggle button
 */
export function ChatToggleButton({
  isOpen,
  hasUnread,
  onClick,
}: {
  isOpen: boolean;
  hasUnread: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        width: '56px',
        height: '56px',
        borderRadius: '50%',
        backgroundColor: isOpen ? '#333366' : '#00ffff',
        border: 'none',
        cursor: 'pointer',
        boxShadow: '0 4px 16px rgba(0, 255, 255, 0.3)',
        display: isOpen ? 'none' : 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '24px',
        transition: 'all 0.2s',
        zIndex: 999,
      }}
    >
      <span style={{ color: isOpen ? '#ffffff' : '#000000' }}>⬡</span>
      {hasUnread && (
        <div
          style={{
            position: 'absolute',
            top: '0',
            right: '0',
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            backgroundColor: '#ff3366',
            border: '2px solid #0d0d15',
          }}
        />
      )}
    </button>
  );
}

export default ChatPanel;
