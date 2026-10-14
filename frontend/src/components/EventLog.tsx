"use client";

/**
 * Completed rounds, newest first.
 *
 * Newest-first because the interesting entry is always the one that just
 * happened, and a reader should not have to scroll to find it. Each row says
 * *how* a round ended, not only who won: a bomb detonation and a time expiry
 * are the same point on the scoreboard and completely different stories.
 */

import type { RoundEndReason, RoundOutcome } from "@/types/api";

/** Reason slugs, in the language a caster would use. */
const REASON_LABELS: Record<RoundEndReason, string> = {
  ct_eliminated: "CT eliminated",
  t_eliminated: "T eliminated",
  bomb_defused: "bomb defused",
  bomb_exploded: "bomb detonated",
  time_expired: "time expired",
};

interface EventLogProps {
  rounds: RoundOutcome[];
  /** Appended as a live entry when a match has finished. */
  finished?: { winner: string; scoreCt: number; scoreT: number } | null;
}

function RoundRow({ outcome }: { outcome: RoundOutcome }) {
  const ctWon = outcome.winner === "CT";
  const accent = ctWon ? "text-ct" : "text-t";
  const edge = ctWon ? "border-l-ct" : "border-l-t";

  return (
    <li
      className={`flex items-baseline gap-3 border-l-2 bg-base-850 px-3 py-2 ${edge} animate-slide-in`}
    >
      <span className="tabular w-8 shrink-0 text-[0.7rem] text-muted">
        R{outcome.round_number}
      </span>
      <span className={`w-6 shrink-0 text-xs font-semibold ${accent}`}>
        {outcome.winner}
      </span>
      <span className="min-w-0 flex-1 truncate text-xs text-muted">
        {REASON_LABELS[outcome.reason]}
      </span>
      <span className="tabular shrink-0 text-xs font-medium">
        {outcome.score.ct}–{outcome.score.t}
      </span>
    </li>
  );
}

export function EventLog({ rounds, finished }: EventLogProps) {
  // The server appends; a viewer wants the latest first. Reversing a copy keeps
  // the source array in arrival order for anything else reading it.
  const newestFirst = [...rounds].reverse();

  return (
    <div className="rounded-lg border border-base-700 bg-base-900">
      <div className="flex items-baseline justify-between border-b border-base-700 px-4 py-2.5">
        <span className="text-[0.7rem] uppercase tracking-wider text-muted">
          round log
        </span>
        <span className="tabular text-[0.7rem] text-muted">
          {rounds.length} played
        </span>
      </div>

      {finished ? (
        <div className="border-b border-base-700 bg-base-850 px-4 py-3 text-center">
          <div className="text-xs uppercase tracking-wider text-muted">match over</div>
          <div className="tabular mt-1 text-sm font-semibold">
            <span className={finished.winner === "CT" ? "text-ct" : "text-t"}>
              {finished.winner}
            </span>{" "}
            {finished.scoreCt}–{finished.scoreT}
          </div>
        </div>
      ) : null}

      {newestFirst.length === 0 ? (
        <div className="px-4 py-8 text-center text-xs text-muted">
          no rounds decided yet
        </div>
      ) : (
        <ul className="scroll-slim max-h-[20rem] divide-y divide-base-800 overflow-y-auto">
          {newestFirst.map((outcome) => (
            <RoundRow key={outcome.round_number} outcome={outcome} />
          ))}
        </ul>
      )}
    </div>
  );
}
