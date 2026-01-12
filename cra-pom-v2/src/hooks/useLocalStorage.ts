// File: src/hooks/useLocalStorage.ts
// LocalStorage persistence for session state

import { useState, useCallback } from 'react';

const STORAGE_PREFIX = 'cra-pom-v2:';

/**
 * Hook for persisting state to localStorage
 */
export function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((prev: T) => T)) => void, () => void] {
  const fullKey = STORAGE_PREFIX + key;

  // Get initial value from storage or use default
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(fullKey);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  // Update localStorage when value changes
  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      try {
        const valueToStore = value instanceof Function ? value(storedValue) : value;
        setStoredValue(valueToStore);
        window.localStorage.setItem(fullKey, JSON.stringify(valueToStore));
      } catch (error) {
        console.warn(`Failed to save to localStorage: ${error}`);
      }
    },
    [fullKey, storedValue]
  );

  // Clear this key from storage
  const clearValue = useCallback(() => {
    try {
      window.localStorage.removeItem(fullKey);
      setStoredValue(initialValue);
    } catch (error) {
      console.warn(`Failed to clear localStorage: ${error}`);
    }
  }, [fullKey, initialValue]);

  return [storedValue, setValue, clearValue];
}

/**
 * Session settings that are persisted
 */
export interface SessionSettings {
  autoRotate: boolean;
  showTrajectory: boolean;
  showGrid: boolean;
  stepRate: number;
  trajectoryLength: number;
  projectionW: number;
  cameraDistance: number;
  fov: number;
}

export const DEFAULT_SESSION_SETTINGS: SessionSettings = {
  autoRotate: true,
  showTrajectory: true,
  showGrid: true,
  stepRate: 10,
  trajectoryLength: 200,
  projectionW: 2.5,
  cameraDistance: 4,
  fov: 300,
};

/**
 * Hook specifically for session settings
 */
export function useSessionSettings() {
  return useLocalStorage<SessionSettings>('session-settings', DEFAULT_SESSION_SETTINGS);
}

/**
 * Coherence history for sparkline
 */
export interface CoherenceHistoryEntry {
  timestamp: number;
  overall: number;
  spinor: number;
  stability: number;
}

/**
 * Clear all CRA-POM v2 data from localStorage
 */
export function clearAllStoredData(): void {
  const keysToRemove: string[] = [];
  for (let i = 0; i < window.localStorage.length; i++) {
    const key = window.localStorage.key(i);
    if (key?.startsWith(STORAGE_PREFIX)) {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach((key) => window.localStorage.removeItem(key));
}

export default useLocalStorage;
