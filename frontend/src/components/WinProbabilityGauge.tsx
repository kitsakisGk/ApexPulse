"use client";

/**
 * The headline number.
 *
 * Drawn as an SVG arc rather than a full circle: a semicircle reads as a
 * balance between two sides, which is what a win probability is, while a full
 * ring reads as progress towards a target. The sides' colours meet at the
 * current probability, so the split itself carries the information and the
 * numeral only confirms it.
 */

import type { FavouredSide, PredictionSummary } from "@/types/api";

/** Geometry. The arc spans 180°, drawn right-to-left so CT fills from the left. */
const SIZE = 260;
const STROKE = 18;
const RADIUS = (SIZE - STROKE) / 2;
const CENTRE_Y = SIZE / 2;

/** Arc length of a semicircle, used to drive the dash offset. */
const ARC_LENGTH = Math.PI * RADIUS;

interface WinProbabilityGaugeProps {
  prediction: PredictionSummary | null;
  /** Shown under the number; "waiting" states explain themselves. */
  phase?: string;
}

function semicirclePath(): string {
  const left = STROKE / 2;
  const right = SIZE - STROKE / 2;
  return `M ${left} ${CENTRE_Y} A ${RADIUS} ${RADIUS} 0 0 1 ${right} ${CENTRE_Y}`;
}

function sideLabel(side: FavouredSide): string {
  if (side === "even") return "too close to call";
  return side === "CT" ? "Counter-Terrorists favoured" : "Terrorists favoured";
}

export function WinProbabilityGauge({ prediction, phase }: WinProbabilityGaugeProps) {
  const probability = prediction?.ct_win_probability ?? null;
  const path = semicirclePath();

  // A null probability is not 50%. Freezetime and an absent model both mean the
  // model has no opinion, and drawing an even split would invent one.
  const hasReading = prediction !== null && probability !== null;
  const ctLength = hasReading ? ARC_LENGTH * probability : 0;

  return (
    <div className="flex flex-col items-center">
      <svg
        viewBox={`0 0 ${SIZE} ${CENTRE_Y + STROKE}`}
        className="w-full max-w-[280px]"
        role="img"
        aria-label={
          hasReading
            ? `Counter-Terrorist win probability ${Math.round(probability * 100)} percent`
            : "Win probability unavailable"
        }
      >
        {/* The T side is the full arc; the CT side is drawn over it. The
            remaining track therefore belongs to T without a second element. */}
        <path
          d={path}
          fill="none"
          stroke={hasReading ? "var(--color-t)" : "var(--color-base-700)"}
          strokeWidth={STROKE}
          strokeLinecap="round"
        />
        {hasReading ? (
          <path
            d={path}
            fill="none"
            stroke="var(--color-ct)"
            strokeWidth={STROKE}
            strokeLinecap="round"
            strokeDasharray={`${ctLength} ${ARC_LENGTH}`}
            // A transition on the dash makes the arc sweep to its new value
            // rather than jumping, which at 4 Hz is the difference between a
            // gauge that reads as live and one that reads as glitching.
            style={{ transition: "stroke-dasharray 280ms ease-out" }}
          />
        ) : null}

        <text
          x={SIZE / 2}
          y={CENTRE_Y - 34}
          textAnchor="middle"
          className="tabular fill-current text-[2.75rem] font-bold"
        >
          {hasReading ? `${Math.round(probability * 100)}%` : "—"}
        </text>
        <text
          x={SIZE / 2}
          y={CENTRE_Y - 10}
          textAnchor="middle"
          className="fill-current text-[0.7rem] uppercase tracking-[0.18em] opacity-50"
        >
          CT win
        </text>
      </svg>

      <div className="mt-1 flex w-full max-w-[280px] items-baseline justify-between text-xs">
        <span className="font-semibold text-ct">CT</span>
        <span className="text-center text-muted">
          {hasReading ? sideLabel(prediction.favoured_side) : (phase ?? "no reading")}
        </span>
        <span className="font-semibold text-t">T</span>
      </div>

      {prediction ? (
        <div className="tabular mt-3 flex gap-4 text-[0.7rem] text-muted">
          <span>confidence {Math.round(prediction.confidence * 100)}%</span>
          <span>{prediction.latency_ms.toFixed(1)} ms</span>
        </div>
      ) : null}
    </div>
  );
}
