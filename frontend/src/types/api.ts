/**
 * TypeScript mirrors of the ApexPulse API response models.
 *
 * Hand-written against the live OpenAPI schema rather than generated, because
 * the generated output for a pydantic schema is noisy — every optional field
 * becomes a four-way union — and this surface is small enough to keep honest by
 * hand. The field names and nullability here were read off `/openapi.json`, not
 * from memory.
 *
 * Nullable fields are typed `| null` rather than optional: the API always sends
 * the key, and `?:` would let a component forget to handle the null case.
 */

/** One of the two sides. The API sends the bare string. */
export type Team = "CT" | "T";

/** Which side the model currently favours. */
export type FavouredSide = Team | "even";

/** Lifecycle of the round in progress. */
export type RoundPhase = "freezetime" | "live" | "bomb_planted" | "over";

/** How a round was decided. */
export type RoundEndReason =
  | "ct_eliminated"
  | "t_eliminated"
  | "bomb_defused"
  | "bomb_exploded"
  | "time_expired";

export interface ScoreLine {
  ct: number;
  t: number;
}

export interface RoundSummary {
  number: number;
  phase: RoundPhase;
  seconds_remaining: number;
  bomb_planted: boolean;
  bomb_seconds_remaining: number | null;
}

export interface TeamSummary {
  alive: number;
  health: number;
  money: number;
  equipment_value: number;
  consecutive_losses: number;
}

export interface PlayerSummary {
  player_id: string;
  name: string;
  team: Team;
  health: number;
  armour: number;
  money: number;
  weapon: string | null;
  alive: boolean;
  kills: number;
  deaths: number;
}

export interface MomentumSummary {
  kills_ct: number;
  kills_t: number;
  kill_delta: number;
  window_seconds: number;
}

export interface PredictionSummary {
  ct_win_probability: number;
  t_win_probability: number;
  favoured_side: FavouredSide;
  confidence: number;
  latency_ms: number;
}

export interface MatchSnapshot {
  match_id: string;
  map_name: string;
  sequence: number;
  timestamp: string;
  score: ScoreLine;
  round: RoundSummary;
  ct: TeamSummary;
  t: TeamSummary;
  players: PlayerSummary[];
  momentum: MomentumSummary;
  /** Null when no model is loaded, or the current tick cannot be scored. */
  prediction: PredictionSummary | null;
}

export interface RoundOutcome {
  round_number: number;
  winner: Team;
  reason: RoundEndReason;
  score: ScoreLine;
  timestamp: string;
}

export interface MatchHistoryResponse {
  match_id: string;
  rounds_played: number;
  rounds: RoundOutcome[];
  ct_wins: number;
  t_wins: number;
  streak_side: Team | null;
  streak_length: number;
}

export interface MatchListItem {
  match_id: string;
  map_name: string;
  score: ScoreLine;
  round_number: number;
  phase: RoundPhase;
  ct_win_probability: number | null;
}

export interface MatchListResponse {
  count: number;
  matches: MatchListItem[];
}

export interface FeatureContribution {
  name: string;
  value: number;
  importance: number;
}

export interface ExplainedPrediction {
  match_id: string;
  round_number: number;
  ct_win_probability: number;
  favoured_side: FavouredSide;
  confidence: number;
  latency_ms: number;
  /** Ordered by training importance, most influential first. */
  features: FeatureContribution[];
}

export interface ModelInfo {
  loaded: boolean;
  trained_at: string | null;
  features: string[];
  trees: number | null;
  roc_auc: number | null;
  log_loss: number | null;
  skill_score: number | null;
  accuracy: number | null;
  train_rows: number | null;
  test_matches: number | null;
  top_features: FeatureContribution[];
}

export interface ComponentHealth {
  name: string;
  healthy: boolean;
  detail: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  components: ComponentHealth[];
}

export interface MetricsResponse {
  live_matches: number;
  predictions_served: number;
  model_loaded: boolean;
  model_trained_at: string | null;
  inference_p50_ms: number;
  inference_p95_ms: number;
  inference_p99_ms: number;
  uptime_seconds: number;
}
