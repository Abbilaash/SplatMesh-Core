export interface ServerStatus {
  training_active: boolean;
  training_done: boolean;
  progress: number;
  iteration: number;
  total: number;
  current_loss: number;
  eta_seconds: number;
  ply_available: boolean;
  checkpoint_ply_available: boolean;
}

export interface SystemConfig {
  backendUrl: string;
  pollIntervalMs: number;
}
