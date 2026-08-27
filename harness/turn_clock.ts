export type TurnEndReason = "settled" | "tool_cap" | "timeout";

export class TurnClock {
  constructor(
    private readonly maxTools = 12,
    private readonly maxSeconds = 60,
  ) {
    if (maxTools < 1 || maxSeconds <= 0) {
      throw new RangeError("turn limits must be positive");
    }
  }

  shouldStop(
    toolsUsed: number,
    elapsedSec: number,
    settled: boolean,
  ): TurnEndReason | null {
    if (settled) return "settled";
    if (toolsUsed >= this.maxTools) return "tool_cap";
    if (elapsedSec >= this.maxSeconds) return "timeout";
    return null;
  }
}

export interface PendingBash {
  readonly startedAtMs: number;
  readonly completion: Promise<unknown>;
}

export async function pollPendingBash(
  pending: PendingBash,
  nowMs: number,
  hungAfterMs = 120_000,
): Promise<"complete" | "pending" | "hung"> {
  if (nowMs - pending.startedAtMs > hungAfterMs) return "hung";
  const marker = Symbol("pending");
  const result = await Promise.race([
    pending.completion.then(() => "complete" as const),
    Promise.resolve(marker),
  ]);
  return result === marker ? "pending" : result;
}
