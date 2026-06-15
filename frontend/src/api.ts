const API_URL = "http://localhost:8000";

export interface PlayerInput {
  position: string;
  pace: number;
  shooting: number;
  passing: number;
  dribbling: number;
  defending: number;
  physic: number;
}

export interface PredictionOutput {
  convocated: boolean;
  probability: number;
  overall: number;
}

export async function fetchPositions(): Promise<string[]> {
  const res = await fetch(`${API_URL}/positions`);
  if (!res.ok) throw new Error("Falha ao buscar posições");
  const data = await res.json();
  return data.positions as string[];
}

export async function predict(payload: PlayerInput): Promise<PredictionOutput> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "Erro ao chamar a API");
  }
  return res.json();
}
