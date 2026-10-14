"use client";

/**
 * How the probability moved through the current round.
 *
 * The gauge says where the round stands; this says how it got there, which is
 * the more interesting half. A line that fell off a cliff reads as a lost duel
 * far more clearly than a number that is merely low.
 *
 * Drawn as a hand-built SVG polyline rather than with a chart library: the
 * series is one line with no axes, legend, or interaction, so a charting
 * dependency would add bundle weight for features this does not use.
 */

import { useMemo } from "react";

import type { ProbabilityPoint } from "@/hooks/useMatchStream";

const WIDTH = 100;
const HEIGHT = 32;
/** Minimum points before a line is meaningful; below this, two dots is noise. */
const MIN_POINTS = 2;

interface MomentumTimelineProps {
  history: ProbabilityPoint[];
  /** Shown when there is nothing to draw yet. */
  placeholder?: string;
}

/** Map readings onto the viewBox. Probability is already 0–1, so only x scales. */
function toPolyline(points: ProbabilityPoint[]): string {
  if (points.length === 1) {
    // A single reading has no slope; draw it as a flat segment so something is
    // visible rather than an invisible zero-length line.
    const y = HEIGHT - points[0]!.ctWinProbability * HEIGHT;
    return `0,${y} ${WIDTH},${y}`;
  }

  const step = WIDTH / (points.length - 1);
  return points
    .map((point, index) => {
      const x = index * step;
      const y = HEIGHT - point.ctWinProbability * HEIGHT;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

export function MomentumTimeline({ history, placeholder }: MomentumTimelineProps) {
  const polyline = useMemo(() => toPolyline(history), [history]);
  const latest = history.at(-1)?.ctWinProbability ?? null;

  if (history.length < MIN_POINTS) {
    return (
      <div className="flex h-[72px] items-center justify-center rounded-lg border border-base-700 bg-base-850 text-[0.7rem] text-muted">
        {placeholder ?? "building the round timeline"}
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-base-700 bg-base-850 p-3">
      <div className="flex items-baseline justify-between">
        <span className="text-[0.7rem] uppercase tracking-wider text-muted">
          round momentum
        </span>
        <span className="tabular text-[0.7rem] text-muted">
          {history.length} readings
        </span>
      </div>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        className="mt-2 h-10 w-full"
        role="img"
        aria-label="Counter-Terrorist win probability over the current round"
      >
        {/* The 50% line gives the eye a reference, so a reader can see which
            side of even the round is on without reading the axis. */}
        <line
          x1="0"
          y1={HEIGHT / 2}
          x2={WIDTH}
          y2={HEIGHT / 2}
          stroke="var(--color-base-600)"
          strokeWidth="0.4"
          strokeDasharray="2 2"
          vectorEffect="non-scaling-stroke"
        />
        <polyline
          points={polyline}
          fill="none"
          stroke={
            latest !== null && latest >= 0.5 ? "var(--color-ct)" : "var(--color-t)"
          }
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
          // The viewBox is stretched to fit, so an unscaled stroke keeps the
          // line one consistent weight instead of a distorted ribbon.
          vectorEffect="non-scaling-stroke"
        />
      </svg>

      <div className="mt-1 flex justify-between text-[0.65rem] text-muted">
        <span>round start</span>
        <span>now</span>
      </div>
    </div>
  );
}
