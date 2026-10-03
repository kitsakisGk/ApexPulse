"use client";

/**
 * Connection smoke page.
 *
 * Proves the frontend is genuinely wired to a running backend: it subscribes to
 * the demo match, reports socket status, and shows the probability moving. The
 * composed dashboard — gauge, player grid, event ticker — lands on Day 13; this
 * is the foundation it sits on.
 */

import { useMatchStream } from "@/hooks/useMatchStream";
import type { ConnectionStatus } from "@/hooks/useWebSocket";

const DEMO_MATCH_ID = "apex-demo";

const STATUS_STYLE: Record<ConnectionStatus, string> = {
  open: "bg-live/15 text-live",
  connecting: "bg-warn/15 text-warn",
  reconnecting: "bg-warn/15 text-warn",
  closed: "bg-danger/15 text-danger",
};

function StatusPill({ status, attempts }: { status: ConnectionStatus; attempts: number }) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${STATUS_STYLE[status]}`}
    >
      <span
        className={`size-1.5 rounded-full bg-current ${status === "open" ? "animate-pulse-live" : ""}`}
      />
      {status}
      {attempts > 0 && status !== "open" ? ` · attempt ${attempts}` : ""}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-base-700 bg-base-850 px-4 py-3">
      <div className="text-[0.7rem] uppercase tracking-wider text-muted">{label}</div>
      <div className="tabular mt-1 text-lg font-semibold">{value}</div>
    </div>
  );
}

export default function Home() {
  const stream = useMatchStream(DEMO_MATCH_ID);
  const { snapshot, prediction, status, attempts, received, rounds } = stream;

  const probability = prediction?.ct_win_probability ?? null;
  const percent = probability === null ? null : Math.round(probability * 1000) / 10;

  return (
    <main className="mx-auto max-w-4xl p-8">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Apex<span className="text-ct">Pulse</span>
          </h1>
          <p className="mt-1 text-sm text-muted">
            Live CS2 win probability · watching{" "}
            <code className="text-base-600">{DEMO_MATCH_ID}</code>
          </p>
        </div>
        <StatusPill status={status} attempts={attempts} />
      </header>

      <section className="mt-8">
        <div className="flex items-end justify-between text-sm">
          <span className="font-medium text-ct">Counter-Terrorists</span>
          <span className="font-medium text-t">Terrorists</span>
        </div>

        {/* The headline number. A null probability means the tick is
            unscoreable — freezetime, or no model — not a 50/50 call. */}
        <div className="mt-2 h-10 overflow-hidden rounded-lg border border-base-700 bg-base-850">
          {probability === null ? (
            <div className="flex h-full items-center justify-center text-xs text-muted">
              waiting for a scoreable tick
            </div>
          ) : (
            <div
              className="h-full bg-ct transition-[width] duration-300 ease-out"
              style={{ width: `${probability * 100}%` }}
            />
          )}
        </div>

        <div className="tabular mt-2 flex justify-between text-sm">
          <span className="text-ct">{percent === null ? "—" : `${percent.toFixed(1)}%`}</span>
          <span className="text-t">
            {percent === null ? "—" : `${(100 - percent).toFixed(1)}%`}
          </span>
        </div>
      </section>

      <section className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat
          label="score"
          value={snapshot ? `${snapshot.score.ct} – ${snapshot.score.t}` : "—"}
        />
        <Stat
          label="round"
          value={snapshot ? `${snapshot.round.number} · ${snapshot.round.phase}` : "—"}
        />
        <Stat
          label="alive"
          value={snapshot ? `${snapshot.ct.alive}v${snapshot.t.alive}` : "—"}
        />
        <Stat
          label="latency"
          value={prediction ? `${prediction.latency_ms.toFixed(2)} ms` : "—"}
        />
      </section>

      <footer className="mt-8 flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted">
        <span>frames {received}</span>
        <span>rounds logged {rounds.length}</span>
        <span>server {stream.serverVersion ?? "—"}</span>
        <span>model {stream.modelLoaded ? "loaded" : "absent"}</span>
      </footer>

      {status !== "open" ? (
        <p className="mt-6 rounded-lg border border-base-700 bg-base-850 p-4 text-sm text-muted">
          No connection. Start the API with{" "}
          <code className="text-base-600">apexpulse serve --simulate</code> and this
          page will reconnect on its own.
        </p>
      ) : null}
    </main>
  );
}
