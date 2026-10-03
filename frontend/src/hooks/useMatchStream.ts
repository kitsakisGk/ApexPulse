"use client";

/**
 * Folds the WebSocket frame stream into the state a dashboard renders.
 *
 * The server sends two frame rates on purpose: predictions on every scoreable
 * tick, full snapshots four times a second. This keeps them in one object so a
 * component reads a single source of truth, while the gauge still moves at the
 * faster rate — the prediction is applied on its own frame rather than waiting
 * for the next snapshot.
 *
 * Probability history is retained for the timeline chart. It is bounded: an
 * unbounded array on a match left running for an hour is a slow memory leak,
 * and nothing can usefully draw ten thousand points across a few hundred pixels.
 */

import { useCallback, useMemo, useReducer } from "react";

import type { MatchSnapshot, PredictionSummary, RoundOutcome } from "@/types/api";
import type { StreamMessage } from "@/types/events";
import { isStreamMessage } from "@/types/events";

import { useWebSocket, type ConnectionStatus } from "./useWebSocket";

export const MAX_HISTORY_POINTS = 600;
/** Roughly two and a half minutes at 4 Hz — one round, at a drawable density. */

export const MAX_ROUND_LOG = 32;
/** More rounds than a regulation match can hold, so nothing is lost in practice. */

/** One probability reading, for the timeline. */
export interface ProbabilityPoint {
  sequence: number;
  roundNumber: number;
  ctWinProbability: number;
}

export interface MatchStreamState {
  snapshot: MatchSnapshot | null;
  /** The freshest prediction, which may be newer than the snapshot. */
  prediction: PredictionSummary | null;
  history: ProbabilityPoint[];
  rounds: RoundOutcome[];
  /** Set when the match has finished; the dashboard freezes rather than blanks. */
  finished: { winner: string; scoreCt: number; scoreT: number } | null;
  serverVersion: string | null;
  modelLoaded: boolean;
  lastError: string | null;
}

const initialState: MatchStreamState = {
  snapshot: null,
  prediction: null,
  history: [],
  rounds: [],
  finished: null,
  serverVersion: null,
  modelLoaded: false,
  lastError: null,
};

/** Append to a bounded list, dropping from the front when full. */
function appendBounded<T>(items: T[], item: T, limit: number): T[] {
  const next = [...items, item];
  return next.length > limit ? next.slice(next.length - limit) : next;
}

function reduce(state: MatchStreamState, message: StreamMessage): MatchStreamState {
  switch (message.type) {
    case "welcome":
      return {
        ...state,
        serverVersion: message.server_version,
        modelLoaded: message.model_loaded,
        snapshot: message.snapshot ?? state.snapshot,
        prediction: message.snapshot?.prediction ?? state.prediction,
        lastError: null,
      };

    case "tick": {
      const { snapshot } = message;
      // A snapshot carries its own prediction, but a prediction frame may have
      // arrived since. Keep whichever is newer rather than stepping backwards.
      const prediction = snapshot.prediction ?? state.prediction;
      return { ...state, snapshot, prediction };
    }

    case "prediction": {
      const point: ProbabilityPoint = {
        sequence: message.sequence,
        roundNumber: state.snapshot?.round.number ?? 0,
        ctWinProbability: message.prediction.ct_win_probability,
      };
      return {
        ...state,
        prediction: message.prediction,
        history: appendBounded(state.history, point, MAX_HISTORY_POINTS),
      };
    }

    case "round_end":
      return {
        ...state,
        rounds: appendBounded(state.rounds, message.outcome, MAX_ROUND_LOG),
        // A finished round has no live probability; clearing it stops the gauge
        // showing a stale number through the next freezetime.
        prediction: null,
        history: [],
      };

    case "match_end":
      return {
        ...state,
        finished: {
          winner: message.winner,
          scoreCt: message.score_ct,
          scoreT: message.score_t,
        },
        prediction: null,
      };

    case "error":
      return { ...state, lastError: message.detail };
  }
}

export interface UseMatchStreamResult extends MatchStreamState {
  status: ConnectionStatus;
  attempts: number;
  received: number;
  reconnect: () => void;
}

/**
 * Subscribe to one match and expose its live state.
 *
 * @param matchId The match to watch, or null to stay idle.
 * @param baseUrl WebSocket origin; defaults to the configured one.
 */
export function useMatchStream(
  matchId: string | null,
  baseUrl?: string,
): UseMatchStreamResult {
  const [state, dispatch] = useReducer(reduce, initialState);

  const origin =
    baseUrl ?? process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";
  const url = matchId ? `${origin}/ws/${encodeURIComponent(matchId)}` : null;

  const onMessage = useCallback(
    (message: StreamMessage) => dispatch(message),
    [],
  );

  const { status, attempts, received, reconnect } = useWebSocket<StreamMessage>({
    url,
    onMessage,
    validate: isStreamMessage,
  });

  return useMemo(
    () => ({ ...state, status, attempts, received, reconnect }),
    [state, status, attempts, received, reconnect],
  );
}
