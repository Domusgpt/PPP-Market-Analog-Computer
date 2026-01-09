// File: src/hooks/useTouchControls.ts
// Touch and mouse controls for canvas manipulation

import { useRef, useCallback, useEffect } from 'react';

export interface TouchControlState {
  rotationX: number;
  rotationY: number;
  zoom: number;
}

export interface TouchControlCallbacks {
  onRotate: (deltaX: number, deltaY: number) => void;
  onZoom: (delta: number) => void;
  onTap?: () => void;
}

/**
 * Hook for touch and mouse controls on canvas
 */
export function useTouchControls(
  canvasRef: React.RefObject<HTMLElement>,
  callbacks: TouchControlCallbacks
) {
  const isDragging = useRef(false);
  const lastPosition = useRef({ x: 0, y: 0 });
  const lastPinchDistance = useRef(0);
  const lastTapTime = useRef(0);

  // Sensitivity settings
  const ROTATION_SENSITIVITY = 0.005;
  const ZOOM_SENSITIVITY = 0.01;
  const TAP_THRESHOLD = 200; // ms
  const MOVE_THRESHOLD = 10; // px

  /**
   * Get position from mouse or touch event
   */
  const getEventPosition = useCallback(
    (e: MouseEvent | TouchEvent): { x: number; y: number } => {
      if ('touches' in e) {
        if (e.touches.length === 1) {
          return { x: e.touches[0].clientX, y: e.touches[0].clientY };
        } else if (e.touches.length === 2) {
          // Center of two touches
          return {
            x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
            y: (e.touches[0].clientY + e.touches[1].clientY) / 2,
          };
        }
      }
      return { x: (e as MouseEvent).clientX, y: (e as MouseEvent).clientY };
    },
    []
  );

  /**
   * Get pinch distance
   */
  const getPinchDistance = useCallback((e: TouchEvent): number => {
    if (e.touches.length < 2) return 0;
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }, []);

  /**
   * Handle pointer down (mouse or touch start)
   */
  const handlePointerDown = useCallback(
    (e: MouseEvent | TouchEvent) => {
      e.preventDefault();
      isDragging.current = true;
      lastPosition.current = getEventPosition(e);

      if ('touches' in e && e.touches.length === 2) {
        lastPinchDistance.current = getPinchDistance(e);
      }

      // Track for tap detection
      lastTapTime.current = Date.now();
    },
    [getEventPosition, getPinchDistance]
  );

  /**
   * Handle pointer move (mouse move or touch move)
   */
  const handlePointerMove = useCallback(
    (e: MouseEvent | TouchEvent) => {
      if (!isDragging.current) return;
      e.preventDefault();

      const pos = getEventPosition(e);

      // Handle pinch zoom
      if ('touches' in e && e.touches.length === 2) {
        const pinchDistance = getPinchDistance(e);
        if (lastPinchDistance.current > 0) {
          const delta = (pinchDistance - lastPinchDistance.current) * ZOOM_SENSITIVITY;
          callbacks.onZoom(delta);
        }
        lastPinchDistance.current = pinchDistance;
        return;
      }

      // Handle rotation drag
      const deltaX = pos.x - lastPosition.current.x;
      const deltaY = pos.y - lastPosition.current.y;

      callbacks.onRotate(deltaX * ROTATION_SENSITIVITY, deltaY * ROTATION_SENSITIVITY);

      lastPosition.current = pos;
    },
    [getEventPosition, getPinchDistance, callbacks]
  );

  /**
   * Handle pointer up (mouse up or touch end)
   */
  const handlePointerUp = useCallback(
    (e: MouseEvent | TouchEvent) => {
      const wasShortTap = Date.now() - lastTapTime.current < TAP_THRESHOLD;
      const pos = getEventPosition(e);
      const moved =
        Math.abs(pos.x - lastPosition.current.x) > MOVE_THRESHOLD ||
        Math.abs(pos.y - lastPosition.current.y) > MOVE_THRESHOLD;

      if (wasShortTap && !moved && callbacks.onTap) {
        callbacks.onTap();
      }

      isDragging.current = false;
      lastPinchDistance.current = 0;
    },
    [getEventPosition, callbacks]
  );

  /**
   * Handle mouse wheel for zoom
   */
  const handleWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      const delta = -e.deltaY * ZOOM_SENSITIVITY * 0.1;
      callbacks.onZoom(delta);
    },
    [callbacks]
  );

  /**
   * Attach event listeners
   */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Mouse events
    canvas.addEventListener('mousedown', handlePointerDown as EventListener);
    canvas.addEventListener('mousemove', handlePointerMove as EventListener);
    canvas.addEventListener('mouseup', handlePointerUp as EventListener);
    canvas.addEventListener('mouseleave', handlePointerUp as EventListener);
    canvas.addEventListener('wheel', handleWheel, { passive: false });

    // Touch events
    canvas.addEventListener('touchstart', handlePointerDown as EventListener, {
      passive: false,
    });
    canvas.addEventListener('touchmove', handlePointerMove as EventListener, {
      passive: false,
    });
    canvas.addEventListener('touchend', handlePointerUp as EventListener);
    canvas.addEventListener('touchcancel', handlePointerUp as EventListener);

    return () => {
      canvas.removeEventListener('mousedown', handlePointerDown as EventListener);
      canvas.removeEventListener('mousemove', handlePointerMove as EventListener);
      canvas.removeEventListener('mouseup', handlePointerUp as EventListener);
      canvas.removeEventListener('mouseleave', handlePointerUp as EventListener);
      canvas.removeEventListener('wheel', handleWheel);

      canvas.removeEventListener('touchstart', handlePointerDown as EventListener);
      canvas.removeEventListener('touchmove', handlePointerMove as EventListener);
      canvas.removeEventListener('touchend', handlePointerUp as EventListener);
      canvas.removeEventListener('touchcancel', handlePointerUp as EventListener);
    };
  }, [canvasRef, handlePointerDown, handlePointerMove, handlePointerUp, handleWheel]);

  return {
    isDragging: isDragging.current,
  };
}

export default useTouchControls;
