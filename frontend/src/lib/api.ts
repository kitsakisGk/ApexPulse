/**
 * Typed fetch wrappers for the ApexPulse REST API.
 *
 * Thin on purpose. The dashboard's live data arrives over the WebSocket; these
 * cover the cases a socket does not — the initial match list, model metadata,
 * and a health check — so there is no need for a caching layer here.
 */

import type {
  HealthResponse,
  MatchHistoryResponse,
  MatchListResponse,
  MatchSnapshot,
  MetricsResponse,
  ModelInfo,
} from "@/types/api";

const DEFAULT_BASE_URL = "http://localhost:8000";

/** Request timeout. A dashboard should show a stale number, not hang. */
const TIMEOUT_MS = 5_000;

export function apiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_BASE_URL;
}

/** Raised when the API answers with a non-2xx status. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly path: string,
    detail: string,
  ) {
    super(`${path} returned ${status}: ${detail}`);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`${apiBaseUrl()}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { Accept: "application/json", ...init?.headers },
      // Live data must never come from a cache.
      cache: "no-store",
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => response.statusText);
      throw new ApiError(response.status, path, detail.slice(0, 200));
    }

    return (await response.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function fetchMetrics(): Promise<MetricsResponse> {
  return request<MetricsResponse>("/metrics");
}

export function fetchModel(): Promise<ModelInfo> {
  return request<ModelInfo>("/model");
}

export function fetchMatches(): Promise<MatchListResponse> {
  return request<MatchListResponse>("/matches");
}

export function fetchMatch(matchId: string): Promise<MatchSnapshot> {
  return request<MatchSnapshot>(`/matches/${encodeURIComponent(matchId)}`);
}

export function fetchMatchHistory(
  matchId: string,
): Promise<MatchHistoryResponse> {
  return request<MatchHistoryResponse>(
    `/matches/${encodeURIComponent(matchId)}/history`,
  );
}
