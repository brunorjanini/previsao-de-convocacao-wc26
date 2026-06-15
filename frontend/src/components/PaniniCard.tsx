import fifaIcon from "../assets/icone-fifa26.png";
import type { PredictionOutput, PlayerInput } from "../api";
import { POSITION_NAMES } from "./PlayerForm";

interface PaniniCardProps {
  result: PredictionOutput;
  input: PlayerInput;
  photo: string | null;
  playerName: string;
}

const STATS_DISPLAY = [
  { key: "pace",      abbr: "PAC" },
  { key: "shooting",  abbr: "CHU" },
  { key: "passing",   abbr: "PAS" },
  { key: "dribbling", abbr: "DRI" },
  { key: "defending", abbr: "DEF" },
  { key: "physic",    abbr: "FÍS" },
] as const;

type CardTheme = { bg: string; accentBg: string; border: string; text: string; textMuted: string; glow: string; tier: string };

function getCardTheme(overall: number): CardTheme {
  if (overall >= 85) return {
    bg: "linear-gradient(160deg, #ffe878 0%, #d4a017 35%, #ffd700 55%, #b8860b 100%)",
    accentBg: "rgba(0,0,0,0.15)", border: "border-yellow-400",
    text: "text-yellow-950", textMuted: "text-yellow-800", glow: "shadow-yellow-400/80", tier: "OURO",
  };
  if (overall >= 75) return {
    bg: "linear-gradient(160deg, #f0f0f0 0%, #c0c0c0 35%, #e8e8e8 55%, #a0a0a0 100%)",
    accentBg: "rgba(0,0,0,0.12)", border: "border-gray-300",
    text: "text-gray-900", textMuted: "text-gray-600", glow: "shadow-gray-400/80", tier: "PRATA",
  };
  if (overall >= 65) return {
    bg: "linear-gradient(160deg, #f0b870 0%, #c06010 35%, #e09040 55%, #904010 100%)",
    accentBg: "rgba(0,0,0,0.15)", border: "border-orange-400",
    text: "text-orange-950", textMuted: "text-orange-800", glow: "shadow-orange-500/80", tier: "BRONZE",
  };
  return {
    bg: "linear-gradient(160deg, #708090 0%, #3a4a55 35%, #607080 55%, #2a3a45 100%)",
    accentBg: "rgba(0,0,0,0.2)", border: "border-slate-400",
    text: "text-slate-100", textMuted: "text-slate-300", glow: "shadow-slate-600/80", tier: "COMUM",
  };
}

export function PaniniCard({ result, input, photo, playerName }: PaniniCardProps) {
  const theme = getCardTheme(result.overall);
  const probPercent = Math.round(result.probability * 100);
  const positionFull = POSITION_NAMES[input.position.toUpperCase()] ?? input.position;

  return (
    <div className="flex flex-col gap-3 animate-[fadeIn_0.4s_ease-out] w-full">
      {/* Carta portrait 3:4 */}
      <div
        className={`w-full rounded-2xl overflow-hidden border-4 ${theme.border} shadow-2xl ${theme.glow} flex flex-col`}
        style={{ background: theme.bg, aspectRatio: "3/4" }}
      >
        {/* Topo: overall + posicao + icone */}
        <div className="flex items-start justify-between px-6 pt-4 pb-3 flex-shrink-0" style={{ background: theme.accentBg }}>
          <div className="flex flex-col leading-none">
            <span className={`text-6xl font-black ${theme.text} leading-none`}>{result.overall}</span>
            <span className={`text-base font-black ${theme.text} tracking-widest mt-1`}>{input.position}</span>
            <span className={`text-[10px] font-bold ${theme.textMuted} tracking-wider mt-0.5 uppercase`}>{theme.tier}</span>
          </div>
          <img src={fifaIcon} alt="FIFA WC 2026" className="h-16 w-auto object-contain flex-shrink-0"
            style={{ filter: result.overall >= 65 ? "none" : "brightness(1.4) grayscale(0.3)" }} />
        </div>

        {/* Foto */}
        <div className="flex justify-center flex-shrink-0 py-4">
          {photo ? (
            <img src={photo} alt="Jogador"
              className={`w-32 h-32 rounded-full object-cover border-4 ${theme.border} shadow-xl`} />
          ) : (
            <div className={`w-32 h-32 rounded-full border-4 ${theme.border} shadow-xl overflow-hidden flex items-end justify-center`}
              style={{ background: "rgba(0,0,0,0.2)" }}>
              <svg viewBox="0 0 100 110" className={`w-28 h-28 ${theme.text}`} fill="currentColor" style={{ opacity: 0.5 }}>
                <circle cx="50" cy="30" r="20" />
                <ellipse cx="50" cy="95" rx="35" ry="22" />
              </svg>
            </div>
          )}
        </div>

        {/* Nome + posicao */}
        <div className="text-center px-6 flex-shrink-0">
          <p className={`font-black text-2xl tracking-widest uppercase ${theme.text} leading-tight`}>{playerName}</p>
          <p className={`text-xs font-semibold ${theme.textMuted} tracking-wider mt-1 uppercase`}>{positionFull}</p>
        </div>

        {/* Divisoria */}
        <div className="flex items-center gap-2 mx-6 my-3 flex-shrink-0">
          <div className="flex-1 h-px opacity-30" style={{ backgroundColor: "currentColor" }} />
          <span className={`text-sm ${theme.textMuted}`}>*</span>
          <div className="flex-1 h-px opacity-30" style={{ backgroundColor: "currentColor" }} />
        </div>

        {/* Stats — ocupa o espaco restante */}
        <div className="grid grid-cols-3 gap-x-2 flex-1 px-6 pb-4" style={{ background: theme.accentBg }}>
          {STATS_DISPLAY.map(({ key, abbr }) => {
            const v = input[key as keyof typeof input] as number;
            return (
              <div key={key} className="flex flex-col items-center justify-center">
                <span className={`text-2xl font-black ${theme.text} tabular-nums leading-none`}>{v}</span>
                <span className={`text-[10px] font-bold ${theme.textMuted} tracking-widest mt-0.5`}>{abbr}</span>
              </div>
            );
          })}
        </div>

        {/* Badge */}
        <div className={`flex-shrink-0 py-3 text-center font-black text-base tracking-widest ${
          result.convocated ? "bg-emerald-600 text-white" : "bg-red-700 text-white"
        }`}>
          {result.convocated ? "CONVOCADO" : "NÃO CONVOCADO"}
        </div>
      </div>

      {/* Barra de probabilidade */}
      <div className="w-full flex flex-col gap-1.5">
        <div className="flex justify-between text-xs text-blue-200">
          <span>Probabilidade de convocação</span>
          <span className="font-bold text-amber-300">{probPercent}%</span>
        </div>
        <div className="h-2.5 bg-blue-950 rounded-full overflow-hidden border border-blue-700">
          <div
            className={`h-full rounded-full transition-all duration-700 ${result.convocated ? "bg-emerald-500" : "bg-red-500"}`}
            style={{ width: `${probPercent}%` }}
          />
        </div>
        <p className="text-xs text-blue-400">
          {result.convocated
            ? "Este perfil seria convocado para a Copa 2026!"
            : "Este perfil não seria convocado para a Copa 2026."}
        </p>
      </div>
    </div>
  );
}
