"use client";

/**
 * Score, map, and the round clock.
 *
 * The clock switches to the bomb timer once the bomb is down, because at that
 * point the round clock stops mattering — the only deadline that decides the
 * round is detonation. Showing both would ask a viewer to work out which one
 * counts.
 */

import type { RoundSummary, ScoreLine } from "@/types/api";

/** Map slugs arrive as de_mirage; render them as players name them. */
function mapLabel(mapName: string): string {
  const bare = mapName.replace(/^de_/, "");
  return bare.charAt(0).toUpperCase() + bare.slice(1);
}

function clock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  return `${minutes}:${String(total % 60).padStart(2, "0")}`;
}

interface ScoreBoardProps {
  score: ScoreLine;
  round: RoundSummary;
  mapName: string;
}

export function ScoreBoard({ score, round, mapName }: ScoreBoardProps) {
  const planted = round.bomb_planted && round.bomb_seconds_remaining !== null;
  const remaining = planted ? round.bomb_seconds_remaining! : round.seconds_remaining;

  // Under ten seconds on the bomb is the tensest moment in the game; the colour
  // change is the one place on this dashboard that deliberately demands
  // attention.
  const urgent = planted && remaining <= 10;

  return (
    <div className="rounded-lg border border-base-700 bg-base-850 px-5 py-4">
      <div className="flex items-center justify-between gap-6">
        <div className="text-center">
          <div className="text-[0.65rem] uppercase tracking-wider text-ct">CT</div>
          <div className="tabular text-3xl font-bold text-ct">{score.ct}</div>
        </div>

        <div className="flex-1 text-center">
          <div className="text-[0.65rem] uppercase tracking-wider text-muted">
            {mapLabel(mapName)} · round {round.number}
          </div>
          <div
            className={`tabular mt-0.5 text-2xl font-semibold ${
              urgent ? "text-danger animate-pulse-live" : planted ? "text-t" : ""
            }`}
          >
            {clock(remaining)}
          </div>
          <div className="mt-0.5 text-[0.65rem] uppercase tracking-wider text-muted">
            {planted ? "bomb planted" : round.phase.replace("_", " ")}
          </div>
        </div>

        <div className="text-center">
          <div className="text-[0.65rem] uppercase tracking-wider text-t">T</div>
          <div className="tabular text-3xl font-bold text-t">{score.t}</div>
        </div>
      </div>
    </div>
  );
}
