import { useState, useEffect } from "react";
import { fetchPositions, predict, type PlayerInput, type PredictionOutput } from "../api";
import { StatSlider } from "./StatSlider";
import { WebcamCapture } from "./WebcamCapture";

export const POSITION_NAMES: Record<string, string> = {
  GK:  "Goleiro",
  LB:  "Lateral Esquerdo",
  CB:  "Zagueiro",
  RB:  "Lateral Direito",
  CDM: "Volante",
  LM:  "Meia Esquerdo",
  CM:  "Meia",
  RM:  "Meia Direito",
  CAM: "Meia Atacante",
  LW:  "Ponta Esquerdo",
  RW:  "Ponta Direito",
  ST:  "Atacante",
};

const STATS: { key: keyof Omit<PlayerInput, "position">; label: string; abbr: string; desc: string }[] = [
  { key: "pace",      label: "Velocidade",  abbr: "PAC", desc: "Rapidez do jogador em campo — aceleração e velocidade máxima de sprint." },
  { key: "shooting",  label: "Chute",       abbr: "CHU", desc: "Qualidade das finalizações — força, precisão e variedade de chutes ao gol." },
  { key: "passing",   label: "Passe",       abbr: "PAS", desc: "Capacidade de distribuir a bola — precisão, visão de jogo e variedade de passes." },
  { key: "dribbling", label: "Drible",      abbr: "DRI", desc: "Habilidade no controle de bola e superação de adversários em situações de um contra um." },
  { key: "defending", label: "Defesa",      abbr: "DEF", desc: "Eficiência nas ações defensivas — posicionamento, interceptações e desarmes." },
  { key: "physic",    label: "Físico",      abbr: "FÍS", desc: "Condição física geral — força, resistência, duelos aéreos e equilíbrio em campo." },
];

interface PlayerFormProps {
  onResult: (result: PredictionOutput, input: PlayerInput, photo: string | null, name: string) => void;
  onPositionChange: (pos: string) => void;
}

export function PlayerForm({ onResult, onPositionChange }: PlayerFormProps) {
  const [positions, setPositions] = useState<string[]>([]);
  const [position, setPosition] = useState("");
  const [playerName, setPlayerName] = useState("");
  const [stats, setStats] = useState<Record<string, number>>({
    pace: 70, shooting: 70, passing: 70, dribbling: 70, defending: 70, physic: 70,
  });
  const [photo, setPhoto] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showStatsInfo, setShowStatsInfo] = useState(false);

  useEffect(() => {
    fetchPositions()
      .then((pos) => { setPositions(pos); setPosition(pos[0] ?? ""); onPositionChange(pos[0] ?? ""); })
      .catch(() => setError("Não foi possível conectar à API. Rode: make dev"));
  }, []);

  const handlePositionChange = (p: string) => { setPosition(p); onPositionChange(p); };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!position) return;
    setLoading(true);
    setError(null);
    try {
      const input: PlayerInput = { position, ...stats } as PlayerInput;
      const result = await predict(input);
      onResult(result, input, photo, playerName.trim() || "JOGADOR");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro desconhecido");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">

      {/* Foto + Nome — linha compacta */}
      <div className="flex flex-col gap-2 bg-blue-950/60 border border-blue-800 rounded-xl px-4 py-3">
        <span className="text-[10px] font-bold text-amber-300 tracking-widest uppercase">Foto & Nome</span>
        <WebcamCapture photo={photo} onCapture={(url) => setPhoto(url || null)} />
        <input
          type="text"
          value={playerName}
          onChange={(e) => setPlayerName(e.target.value)}
          placeholder="Nome na carta"
          maxLength={20}
          className="bg-blue-900 border border-blue-700 text-white rounded-lg px-3 py-1.5 text-sm placeholder-blue-500 focus:outline-none focus:border-amber-400"
        />
      </div>

      {/* Posicao */}
      <div className="flex flex-col gap-1.5 bg-blue-950/60 border border-blue-800 rounded-xl px-4 py-3">
        <span className="text-[10px] font-bold text-amber-300 tracking-widest uppercase">Posição</span>
        <select
          value={position}
          onChange={(e) => handlePositionChange(e.target.value)}
          className="bg-blue-900 border border-blue-700 text-white rounded-lg px-3 py-1.5 text-sm font-semibold focus:outline-none focus:border-amber-400"
        >
          {positions.map((p) => (
            <option key={p} value={p}>{p} — {POSITION_NAMES[p] ?? p}</option>
          ))}
        </select>
      </div>

      {/* Atributos */}
      <div className="flex flex-col gap-2 bg-blue-950/60 border border-blue-800 rounded-xl px-4 py-3">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-bold text-amber-300 tracking-widest uppercase">Atributos</span>
          <button
            type="button"
            onClick={() => setShowStatsInfo(true)}
            className="w-4 h-4 rounded-full border border-blue-500 text-blue-400 hover:border-amber-400 hover:text-amber-300 flex items-center justify-center text-[10px] font-bold leading-none transition-colors"
            title="O que significa cada atributo?"
          >
            i
          </button>
        </div>
        {STATS.map(({ key, label, abbr }) => (
          <StatSlider
            key={key}
            label={label}
            abbr={abbr}
            value={stats[key]}
            onChange={(v) => setStats((s) => ({ ...s, [key]: v }))}
          />
        ))}
      </div>

      {/* Modal de atributos */}
      {showStatsInfo && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
          onClick={() => setShowStatsInfo(false)}
        >
          <div
            className="bg-blue-950 border border-blue-700 rounded-2xl p-6 max-w-sm w-full shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-black text-amber-300 tracking-widest uppercase">Atributos</h2>
              <button
                type="button"
                onClick={() => setShowStatsInfo(false)}
                className="text-blue-400 hover:text-white text-lg leading-none font-bold"
              >
                ×
              </button>
            </div>
            <ul className="flex flex-col gap-3">
              {STATS.map(({ abbr, label, desc }) => (
                <li key={abbr} className="flex gap-3 items-start">
                  <span className="text-xs font-black text-amber-400 w-8 flex-shrink-0 mt-0.5">{abbr}</span>
                  <div>
                    <p className="text-xs font-bold text-white">{label}</p>
                    <p className="text-xs text-blue-300 mt-0.5 leading-snug">{desc}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {error && (
        <p className="text-red-300 text-xs bg-red-950/50 rounded-xl px-3 py-2 border border-red-800">{error}</p>
      )}

      <button
        type="submit"
        disabled={loading || !position}
        className="w-full py-2.5 bg-amber-500 hover:bg-amber-400 disabled:bg-blue-800 disabled:text-blue-500 disabled:cursor-not-allowed text-blue-950 font-black text-sm rounded-xl tracking-widest uppercase transition-colors"
      >
        {loading ? "Prevendo..." : "Gerar Carta"}
      </button>
    </form>
  );
}
