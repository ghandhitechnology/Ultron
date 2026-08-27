export interface JsonlSink {
  append(line: string): Promise<void>;
}

export interface PiEvent {
  readonly type: string;
  readonly timestampMs: number;
  readonly payload: Readonly<Record<string, unknown>>;
}

export function parsePiEvent(raw: unknown): PiEvent {
  if (!isRecord(raw)) throw new TypeError("Pi event must be an object");
  const { type, timestampMs, payload } = raw;
  if (
    typeof type !== "string" ||
    typeof timestampMs !== "number" ||
    !isRecord(payload)
  ) {
    throw new TypeError("Pi event has invalid fields");
  }
  return { type, timestampMs, payload };
}

export async function exportPiEvent(sink: JsonlSink, raw: unknown): Promise<void> {
  const event = parsePiEvent(raw);
  await sink.append(`${JSON.stringify(event)}\n`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
