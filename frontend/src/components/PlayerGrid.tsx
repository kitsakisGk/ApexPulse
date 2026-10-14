"use client";

/**
 * Ten players, five a side, with health and equipment at a glance.
 *
 * Rows keep a fixed order and a fixed height. A grid that reorders by health or
 * collapses dead players would make a viewer re-find the player they were
 * watching on every tick, which is worse than a row that greys out in place.
 */

import type { PlayerSummary, Team } from "@/types/api";

interface PlayerGridProps {
  players: PlayerSummary[];
}

interface PlayerRowProps {
  player: PlayerSummary;
}

/** Health drives the bar colour, so a wounded player is visible without reading. */
function healthColour(health: number): string {
  if (health > 60) return "bg-live";
  if (health > 25) return "bg-warn";
  return "bg-danger";
}

/** Weapon names arrive as API slugs; render them the way players say them. */
const WEAPON_LABELS: Record<string, string> = {
  ak47: "AK-47",
  m4a4: "M4A4",
  m4a1s: "M4A1-S",
  awp: "AWP",
  sg553: "SG 553",
  aug: "AUG",
  galil: "Galil",
  famas: "FAMAS",
  ump45: "UMP-45",
  mp9: "MP9",
  mac10: "MAC-10",
  deagle: "Desert Eagle",
  p250: "P250",
  usp: "USP-S",
  glock: "Glock-18",
  knife: "Knife",
};

function weaponLabel(weapon: string | null): string {
  if (!weapon) return "—";
  return WEAPON_LABELS[weapon] ?? weapon;
}

function PlayerRow({ player }: PlayerRowProps) {
  const dead = !player.alive;
  const accent = player.team === "CT" ? "text-ct" : "text-t";

  return (
    <div
      className={`grid grid-cols-[1fr_auto] items-center gap-3 rounded-md border border-base-700 bg-base-850 px-3 py-2 transition-opacity ${
        dead ? "opacity-35" : ""
      }`}
    >
      <div className="min-w-0">
        <div className="flex items-baseline gap-2">
          <span className={`truncate text-sm font-medium ${dead ? "line-through" : accent}`}>
            {player.name}
          </span>
          <span className="tabular shrink-0 text-[0.65rem] text-muted">
            {player.kills}/{player.deaths}
          </span>
        </div>

        {/* The health bar is the row's primary signal, so it sits directly
            under the name rather than in a column a reader has to scan to. */}
        <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-base-700">
          <div
            className={`h-full transition-[width] duration-300 ${healthColour(player.health)}`}
            style={{ width: `${player.health}%` }}
          />
        </div>
      </div>

      <div className="text-right">
        <div className="tabular text-xs font-medium">
          {dead ? "dead" : `${player.health} hp`}
        </div>
        <div className="truncate text-[0.65rem] text-muted">{weaponLabel(player.weapon)}</div>
      </div>
    </div>
  );
}

function TeamColumn({ team, players }: { team: Team; players: PlayerSummary[] }) {
  const alive = players.filter((player) => player.alive).length;
  const label = team === "CT" ? "Counter-Terrorists" : "Terrorists";
  const accent = team === "CT" ? "text-ct" : "text-t";

  return (
    <div>
      <div className="flex items-baseline justify-between px-1">
        <span className={`text-xs font-semibold uppercase tracking-wider ${accent}`}>
          {label}
        </span>
        <span className="tabular text-xs text-muted">{alive} alive</span>
      </div>
      <div className="mt-2 space-y-1.5">
        {players.map((player) => (
          <PlayerRow key={player.player_id} player={player} />
        ))}
      </div>
    </div>
  );
}

export function PlayerGrid({ players }: PlayerGridProps) {
  if (players.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center rounded-lg border border-base-700 bg-base-850 text-sm text-muted">
        waiting for the first tick
      </div>
    );
  }

  const ct = players.filter((player) => player.team === "CT");
  const t = players.filter((player) => player.team === "T");

  return (
    <div className="grid gap-5 sm:grid-cols-2">
      <TeamColumn team="CT" players={ct} />
      <TeamColumn team="T" players={t} />
    </div>
  );
}
