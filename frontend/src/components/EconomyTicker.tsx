"use client";

/**
 * Money and equipment, side by side.
 *
 * Economy is the half of CS2 a scoreline hides: a team two rounds down with
 * $20,000 banked is in a better position than the score suggests, and a team
 * ahead on an eco is in a worse one. Shown as a shared bar so the comparison is
 * immediate rather than two numbers a reader has to subtract.
 */

import type { TeamSummary } from "@/types/api";

/** Per-player money at which a side can fully equip, from the game's economy. */
const FULL_BUY_PER_PLAYER = 4_000;
const PLAYERS_PER_TEAM = 5;
const FULL_BUY_TEAM = FULL_BUY_PER_PLAYER * PLAYERS_PER_TEAM;

/** The loss-bonus ladder. A longer streak means a larger payout next round. */
const LOSS_BONUS_LADDER = [1_400, 1_900, 2_400, 2_900, 3_400] as const;

function lossBonus(consecutiveLosses: number): number {
  if (consecutiveLosses <= 0) return 0;
  const index = Math.min(consecutiveLosses, LOSS_BONUS_LADDER.length) - 1;
  return LOSS_BONUS_LADDER[index]!;
}

function money(value: number): string {
  return `$${value.toLocaleString("en-US")}`;
}

interface EconomyTickerProps {
  ct: TeamSummary;
  t: TeamSummary;
}

/** A row comparing one metric across both sides, split proportionally. */
function ComparisonRow({
  label,
  ctValue,
  tValue,
  format,
}: {
  label: string;
  ctValue: number;
  tValue: number;
  format: (value: number) => string;
}) {
  const total = ctValue + tValue;
  // A zero total is a tie, not a divide-by-zero. Both sides broke is a real
  // state in CS2 and should render as an even split.
  const ctShare = total > 0 ? (ctValue / total) * 100 : 50;

  return (
    <div>
      <div className="flex items-baseline justify-between text-xs">
        <span className="tabular font-medium text-ct">{format(ctValue)}</span>
        <span className="text-[0.65rem] uppercase tracking-wider text-muted">{label}</span>
        <span className="tabular font-medium text-t">{format(tValue)}</span>
      </div>
      <div className="mt-1 flex h-1.5 overflow-hidden rounded-full bg-base-700">
        <div
          className="bg-ct transition-[width] duration-300"
          style={{ width: `${ctShare}%` }}
        />
        <div className="flex-1 bg-t" />
      </div>
    </div>
  );
}

/** Buy state, which is what the money number actually means in context. */
function BuyBadge({ team, economy }: { team: "CT" | "T"; economy: TeamSummary }) {
  const fullBuy = economy.money >= FULL_BUY_TEAM;
  const bonus = lossBonus(economy.consecutive_losses);
  const accent = team === "CT" ? "text-ct" : "text-t";

  return (
    <div className="flex items-baseline gap-2 text-[0.65rem]">
      <span className={`font-semibold uppercase tracking-wider ${accent}`}>{team}</span>
      <span className={fullBuy ? "text-live" : "text-warn"}>
        {fullBuy ? "full buy" : "limited"}
      </span>
      {economy.consecutive_losses > 0 ? (
        <span className="tabular text-muted">
          {economy.consecutive_losses}L · next {money(bonus)}
        </span>
      ) : null}
    </div>
  );
}

export function EconomyTicker({ ct, t }: EconomyTickerProps) {
  return (
    <div className="rounded-lg border border-base-700 bg-base-850 p-4">
      <div className="flex items-baseline justify-between">
        <span className="text-[0.7rem] uppercase tracking-wider text-muted">economy</span>
      </div>

      <div className="mt-3 space-y-3">
        <ComparisonRow label="money" ctValue={ct.money} tValue={t.money} format={money} />
        <ComparisonRow
          label="equipment"
          ctValue={ct.equipment_value}
          tValue={t.equipment_value}
          format={money}
        />
      </div>

      <div className="mt-4 flex justify-between border-t border-base-700 pt-3">
        <BuyBadge team="CT" economy={ct} />
        <BuyBadge team="T" economy={t} />
      </div>
    </div>
  );
}
