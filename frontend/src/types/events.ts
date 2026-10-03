/**
 * TypeScript mirrors of the WebSocket frame types.
 *
 * Modelled as a discriminated union on `type`, so narrowing in a switch gives
 * the correct payload shape without a cast. That is the whole reason the server
 * puts a discriminator on every frame.
 */

import type {
  MatchSnapshot,
  PredictionSummary,
  RoundOutcome,
  Team,
} from "./api";

export type MessageType =
  | "welcome"
  | "tick"
  | "prediction"
  | "round_end"
  | "match_end"
  | "error";

/** First frame after a successful connection. */
export interface WelcomeMessage {
  type: "welcome";
  /** The match being watched, or "*" for every match. */
  match_id: string;
  server_version: string;
  model_loaded: boolean;
  /** Current state, when the match already has some. */
  snapshot: MatchSnapshot | null;
}

/** A full state update. Throttled server-side to four a second. */
export interface TickMessage {
  type: "tick";
  match_id: string;
  sequence: number;
  timestamp: string;
  snapshot: MatchSnapshot;
}

/** A win-probability update, sent on every scoreable tick. */
export interface PredictionMessage {
  type: "prediction";
  match_id: string;
  sequence: number;
  timestamp: string;
  prediction: PredictionSummary;
}

export interface RoundEndMessage {
  type: "round_end";
  match_id: string;
  timestamp: string;
  outcome: RoundOutcome;
}

export interface MatchEndMessage {
  type: "match_end";
  match_id: string;
  timestamp: string;
  winner: Team;
  score_ct: number;
  score_t: number;
}

export interface ErrorMessage {
  type: "error";
  detail: string;
  match_id: string | null;
}

export type StreamMessage =
  | WelcomeMessage
  | TickMessage
  | PredictionMessage
  | RoundEndMessage
  | MatchEndMessage
  | ErrorMessage;

/**
 * Narrow an unknown parsed frame to a `StreamMessage`.
 *
 * Frames arrive as JSON from the network, so they are `unknown` until checked.
 * This validates only the discriminator, which is enough to route the frame;
 * a component reading a field the server never sends would be a contract bug,
 * and re-validating every field on every tick is work the hot path does not
 * need.
 */
export function isStreamMessage(value: unknown): value is StreamMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as { type?: unknown };
  return (
    typeof candidate.type === "string" &&
    ["welcome", "tick", "prediction", "round_end", "match_end", "error"].includes(
      candidate.type,
    )
  );
}
