"use client";

/**
 * WebSocket connection with automatic reconnection.
 *
 * A dashboard left open on a desk will outlive any single connection: the API
 * restarts on deploy, a laptop sleeps, a network blips. So reconnection is the
 * normal case rather than error handling, and the hook treats it that way.
 *
 * Backoff is exponential with jitter. Without backoff, a restarting server gets
 * hammered by every open tab at once; without jitter, those tabs retry in
 * lockstep and arrive as a thundering herd on each attempt.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type ConnectionStatus =
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed";

const INITIAL_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 15_000;
const JITTER_RATIO = 0.3;

/** Delay before attempt `n`, capped and jittered. */
function backoffFor(attempt: number): number {
  const exponential = Math.min(
    INITIAL_BACKOFF_MS * 2 ** attempt,
    MAX_BACKOFF_MS,
  );
  const jitter = exponential * JITTER_RATIO * Math.random();
  return exponential + jitter;
}

export interface UseWebSocketOptions<T> {
  /** Full ws:// or wss:// URL. A null URL holds the hook idle. */
  url: string | null;
  /** Called for each parsed frame. */
  onMessage: (message: T) => void;
  /** Narrows a parsed frame; frames that fail are counted and dropped. */
  validate?: (value: unknown) => value is T;
  /** Set false to disconnect without unmounting. */
  enabled?: boolean;
}

export interface UseWebSocketResult {
  status: ConnectionStatus;
  /** Reconnect attempts since the last successful open. */
  attempts: number;
  /** Frames received since mount. */
  received: number;
  /** Frames dropped because they failed validation. */
  malformed: number;
  /** Force an immediate reconnect, bypassing the backoff delay. */
  reconnect: () => void;
}

export function useWebSocket<T>({
  url,
  onMessage,
  validate,
  enabled = true,
}: UseWebSocketOptions<T>): UseWebSocketResult {
  const [status, setStatus] = useState<ConnectionStatus>("closed");
  const [attempts, setAttempts] = useState(0);
  const [received, setReceived] = useState(0);
  const [malformed, setMalformed] = useState(0);

  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptRef = useRef(0);
  // The caller usually passes an inline arrow, so holding it in a ref keeps a
  // new function identity from tearing down and reopening the socket.
  const onMessageRef = useRef(onMessage);
  const validateRef = useRef(validate);
  // Sockets we closed deliberately. `onclose` fires asynchronously, so a
  // shared boolean cannot tell a stale socket's close from the current one's:
  // by the time it arrives, a new connect may already have reset the flag.
  const abandonedRef = useRef(new WeakSet<WebSocket>());

  useEffect(() => {
    onMessageRef.current = onMessage;
    validateRef.current = validate;
  }, [onMessage, validate]);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (!url || !enabled) {
      return;
    }

    clearTimer();
    setStatus(attemptRef.current === 0 ? "connecting" : "reconnecting");

    const scheduleRetry = (): void => {
      const attempt = attemptRef.current;
      attemptRef.current = attempt + 1;
      setAttempts(attempt + 1);
      setStatus("reconnecting");
      timerRef.current = setTimeout(connect, backoffFor(attempt));
    };

    let socket: WebSocket;
    try {
      socket = new WebSocket(url);
    } catch {
      // A malformed URL throws synchronously; treat it as a failed attempt so
      // the retry path still runs rather than leaving the hook stuck.
      scheduleRetry();
      return;
    }
    socketRef.current = socket;

    socket.onopen = () => {
      attemptRef.current = 0;
      setAttempts(0);
      setStatus("open");
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        setMalformed((count) => count + 1);
        return;
      }

      const check = validateRef.current;
      if (check && !check(parsed)) {
        setMalformed((count) => count + 1);
        return;
      }

      setReceived((count) => count + 1);
      onMessageRef.current(parsed as T);
    };

    socket.onerror = () => {
      // `onclose` always follows, so retry scheduling lives there to avoid
      // queueing two attempts for one failure.
    };

    socket.onclose = () => {
      if (abandonedRef.current.has(socket)) {
        // A socket we closed on purpose. Never retry, and never clear
        // `socketRef`: it may already point at the replacement.
        setStatus((current) => (socketRef.current === null ? "closed" : current));
        return;
      }
      socketRef.current = null;
      scheduleRetry();
    };
  }, [url, enabled, clearTimer]);

  const abandon = useCallback((socket: WebSocket | null) => {
    if (socket === null) {
      return;
    }
    abandonedRef.current.add(socket);
    socket.close();
  }, []);

  const reconnect = useCallback(() => {
    attemptRef.current = 0;
    setAttempts(0);
    clearTimer();
    abandon(socketRef.current);
    socketRef.current = null;
    connect();
  }, [abandon, clearTimer, connect]);

  useEffect(() => {
    if (!enabled || !url) {
      setStatus("closed");
      return;
    }

    connect();

    return () => {
      clearTimer();
      abandon(socketRef.current);
      socketRef.current = null;
    };
  }, [abandon, connect, clearTimer, enabled, url]);

  return { status, attempts, received, malformed, reconnect };
}
