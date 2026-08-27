import type { ExecutionEnv, GuestRole } from "./execution_env.js";

export interface SessionRequest {
  readonly role: GuestRole;
  readonly modelId: "ultron-attacker" | "ultron-defender";
  readonly systemPrompt: string;
  readonly executionEnv: ExecutionEnv;
}

export interface AgentSession {
  run(prompt: string): Promise<unknown>;
  close(): Promise<void>;
}

export type PiSessionFactory = (request: SessionRequest) => Promise<AgentSession>;

export async function createRoleSession(
  role: GuestRole,
  executionEnv: ExecutionEnv,
  systemPrompt: string,
  createSession: PiSessionFactory,
): Promise<AgentSession> {
  return createSession({
    role,
    modelId: role === "attacker" ? "ultron-attacker" : "ultron-defender",
    systemPrompt,
    executionEnv,
  });
}
