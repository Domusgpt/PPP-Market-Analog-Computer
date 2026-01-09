// File: src/hooks/index.ts
// Hook exports

export { useTouchControls, type TouchControlState, type TouchControlCallbacks } from './useTouchControls';
export {
  useLocalStorage,
  useSessionSettings,
  clearAllStoredData,
  DEFAULT_SESSION_SETTINGS,
  type SessionSettings,
  type CoherenceHistoryEntry,
} from './useLocalStorage';
