export type GuestRole = "attacker" | "defender";

export interface GuestHandle {
  readonly vmId: string;
  readonly vsockCid: number;
  readonly role: GuestRole;
  readonly linuxUser: string;
}

export interface ExecResult {
  readonly stdout: string;
  readonly stderr: string;
  readonly exitCode: number | null;
  readonly durationMs: number;
}

export interface ExecutionEnv {
  bash(command: string, opts?: { readonly timeoutMs?: number }): Promise<ExecResult>;
  read(path: string): Promise<string>;
  write(path: string, content: string): Promise<void>;
  edit(path: string, oldStr: string, newStr: string): Promise<void>;
}

export interface GuestExecClient {
  runBash(guest: GuestHandle, command: string, timeoutMs: number): Promise<ExecResult>;
  readFile(guest: GuestHandle, path: string): Promise<string>;
  writeFile(guest: GuestHandle, path: string, content: string): Promise<void>;
  editFile(
    guest: GuestHandle,
    path: string,
    oldStr: string,
    newStr: string,
  ): Promise<void>;
}

export class KvmGuestExecutionEnv implements ExecutionEnv {
  constructor(
    private readonly guest: GuestHandle,
    private readonly guestExec: GuestExecClient,
  ) {}

  bash(command: string, opts?: { readonly timeoutMs?: number }): Promise<ExecResult> {
    return this.guestExec.runBash(this.guest, command, opts?.timeoutMs ?? 55_000);
  }

  read(path: string): Promise<string> {
    return this.guestExec.readFile(this.guest, path);
  }

  write(path: string, content: string): Promise<void> {
    return this.guestExec.writeFile(this.guest, path, content);
  }

  edit(path: string, oldStr: string, newStr: string): Promise<void> {
    return this.guestExec.editFile(this.guest, path, oldStr, newStr);
  }
}
