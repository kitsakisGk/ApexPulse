"use client";

/**
 * The dashboard.
 *
 * Layout follows how a viewer reads a match: the scoreline and the probability
 * first, because that is the question; the player grid and economy next,
 * because that is the evidence; the round log last, because that is the history.
 */

import { EconomyTicker } from "@/components/EconomyTicker";
import { EventLog } from "@/components/EventLog";
import { MomentumTimeline } from "@/components/MomentumTimeline";
import { PlayerGrid } from "@/components/PlayerGrid";
import { ScoreBoard } from "@/components/ScoreBoard";
import { useMatchStream } from "@/hooks/useMatchStream";
import type { ConnectionStatus } from "@/hooks/useWebSocket";
import { WinProbabilityGauge } from "@/components/WinProbabilityGauge";

const DEMO_MATCH_ID = "apex-demo";

const STATUS_STYLE: Record<ConnectionStatus, string> = {
  open: "bg-live/15 text-live",
  connecting: "bg-warn/15 text-warn",
  reconnecting: "bg-warn/15 text-warn",
  closed: "bg-danger/15 text-danger",
};

function StatusPill({
  status,
  attempts,
}: {
  status: ConnectionStatus;
  attempts: number;
}) {
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

export default function Dashboard() {
  const stream = useMatchStream(DEMO_MATCH_ID);
  const { snapshot, prediction, history, rounds, finished, status, attempts } = stream;

  return (
    <main className="mx-auto max-w-6xl p-6 lg:p-8">
      <header className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Apex<span className="text-ct">Pulse</span>
          </h1>
          <p className="mt-0.5 text-xs text-muted">
            Live CS2 win probability ·{" "}
            <code className="text-base-600">{DEMO_MATCH_ID}</code>
          </p>
        </div>
        <StatusPill status={status} attempts={attempts} />
      </header>

      {status !== "open" && snapshot === null ? (
        <p className="mt-8 rounded-lg border border-base-700 bg-base-850 p-6 text-center text-sm text-muted">
          No connection. Start the API with{" "}
          <code className="text-base-600">apexpulse serve --simulate</code> — this
          page reconnects on its own.
        </p>
      ) : null}

      {snapshot ? (
        <>
          {/* The question: where does the round stand, and how did it get here. */}
          <section className="mt-6 grid gap-5 lg:grid-cols-[1fr_320px]">
            <div className="space-y-4">
              <ScoreBoard
                score={snapshot.score}
                round={snapshot.round}
                mapName={snapshot.map_name}
              />
              <MomentumTimeline
                history={history}
                placeholder={
                  snapshot.round.phase === "freezetime"
                    ? "freezetime — no reading yet"
                    : undefined
                }
              />
            </div>

            <div className="flex items-center justify-center rounded-lg border border-base-700 bg-base-850 p-5">
              <WinProbabilityGauge
                prediction={prediction}
                phase={snapshot.round.phase.replace("_", " ")}
              />
            </div>
          </section>

          {/* The evidence: who is alive, and what they can afford. */}
          <section className="mt-5 grid gap-5 lg:grid-cols-[1fr_320px]">
            <PlayerGrid players={snapshot.players} />
            <div className="space-y-5">
              <EconomyTicker ct={snapshot.ct} t={snapshot.t} />
              <EventLog rounds={rounds} finished={finished} />
            </div>
          </section>
        </>
      ) : null}

      <footer className="mt-8 flex flex-wrap gap-x-6 gap-y-1 border-t border-base-800 pt-4 text-[0.7rem] text-muted">
        <span>frames {stream.received}</span>
        <span>server {stream.serverVersion ?? "—"}</span>
        <span>model {stream.modelLoaded ? "loaded" : "absent"}</span>
        {snapshot ? <span>tick {snapshot.sequence}</span> : null}
      </footer>
    </main>
  );
}
