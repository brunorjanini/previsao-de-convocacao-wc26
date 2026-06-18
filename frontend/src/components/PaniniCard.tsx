import { useState } from "react";
import { toPng } from "html-to-image";
import fifaIcon from "../assets/icone-fifa26.png";
import type { PredictionOutput, PlayerInput } from "../api";
import { POSITION_NAMES } from "./PlayerForm";

interface PaniniCardProps {
  result: PredictionOutput;
  input: PlayerInput;
  photo: string | null;
  playerName: string;
  cardRef: React.RefObject<HTMLDivElement | null>;
}

const STATS_DISPLAY = [
  { key: "pace", abbr: "PAC" },
  { key: "shooting", abbr: "CHU" },
  { key: "passing", abbr: "PAS" },
  { key: "dribbling", abbr: "DRI" },
  { key: "defending", abbr: "DEF" },
  { key: "physic", abbr: "FÍS" },
] as const;

type CardTheme = {
  bg: string;
  accentBg: string;
  border: string;
  text: string;
  textMuted: string;
  glow: string;
  tier: string;
};

function getCardTheme(overall: number): CardTheme {
  if (overall >= 85)
    return {
      bg: "linear-gradient(160deg, #ffe878 0%, #d4a017 35%, #ffd700 55%, #b8860b 100%)",
      accentBg: "rgba(0,0,0,0.15)",
      border: "border-yellow-400",
      text: "text-yellow-950",
      textMuted: "text-yellow-800",
      glow: "shadow-yellow-400/80",
      tier: "OURO",
    };
  if (overall >= 75)
    return {
      bg: "linear-gradient(160deg, #f0f0f0 0%, #c0c0c0 35%, #e8e8e8 55%, #a0a0a0 100%)",
      accentBg: "rgba(0,0,0,0.12)",
      border: "border-gray-300",
      text: "text-gray-900",
      textMuted: "text-gray-600",
      glow: "shadow-gray-400/80",
      tier: "PRATA",
    };
  if (overall >= 65)
    return {
      bg: "linear-gradient(160deg, #f0b870 0%, #c06010 35%, #e09040 55%, #904010 100%)",
      accentBg: "rgba(0,0,0,0.15)",
      border: "border-orange-400",
      text: "text-orange-950",
      textMuted: "text-orange-800",
      glow: "shadow-orange-500/80",
      tier: "BRONZE",
    };
  return {
    bg: "linear-gradient(160deg, #708090 0%, #3a4a55 35%, #607080 55%, #2a3a45 100%)",
    accentBg: "rgba(0,0,0,0.2)",
    border: "border-slate-400",
    text: "text-slate-100",
    textMuted: "text-slate-300",
    glow: "shadow-slate-600/80",
    tier: "COMUM",
  };
}

// ─── Modal de envio ────────────────────────────────────────────────────────────

interface SendCardModalProps {
  cardRef: React.RefObject<HTMLDivElement | null>;
  playerName: string;
  onClose: () => void;
}

function SendCardModal({ cardRef, playerName, onClose }: SendCardModalProps) {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<
    "idle" | "loading" | "success" | "error"
  >("idle");
  const [errorMsg, setErrorMsg] = useState("");

  async function handleSend() {
    if (!email || !email.includes("@")) {
      setErrorMsg("Digite um email válido.");
      return;
    }
    if (!cardRef.current) return;

    setStatus("loading");
    setErrorMsg("");

    try {
      const cardEl =
        cardRef.current.querySelector<HTMLElement>(".panini-card-root");
      const image = await toPng(cardEl ?? cardRef.current, {
        cacheBust: true,
        pixelRatio: 2,
      });

      const res = await fetch("http://localhost:8000/send-card", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, image, player_name: playerName }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail ?? "Erro ao enviar.");
      }

      setStatus("success");
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Erro inesperado.");
      setStatus("error");
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4">
      <div className="bg-blue-950 border border-blue-700 rounded-2xl shadow-2xl w-full max-w-sm p-6 flex flex-col gap-4">
        <h2 className="text-white font-black text-lg tracking-wide">
          Enviar carta por email
        </h2>

        {status === "success" ? (
          <div className="flex flex-col items-center gap-3 py-4">
            <span className="text-4xl">✅</span>
            <p className="text-emerald-400 font-semibold text-center">
              Carta enviada para <span className="text-white">{email}</span>!
            </p>
            <button
              onClick={onClose}
              className="mt-2 px-6 py-2 rounded-xl bg-blue-700 text-white font-bold hover:bg-blue-600 transition"
            >
              Fechar
            </button>
          </div>
        ) : (
          <>
            <p className="text-blue-300 text-sm">
              Informe seu email e enviaremos a carta de{" "}
              <span className="text-amber-300 font-semibold">{playerName}</span>{" "}
              como imagem.
            </p>

            <input
              type="email"
              placeholder="seu@email.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              className="w-full px-4 py-2.5 rounded-xl bg-blue-900 border border-blue-600 text-white placeholder-blue-500 focus:outline-none focus:border-amber-400 transition text-sm"
            />

            {errorMsg && <p className="text-red-400 text-xs">{errorMsg}</p>}

            <div className="flex gap-3 mt-1">
              <button
                onClick={onClose}
                className="flex-1 py-2.5 rounded-xl border border-blue-600 text-blue-300 font-semibold hover:bg-blue-800 transition text-sm"
              >
                Cancelar
              </button>
              <button
                onClick={handleSend}
                disabled={status === "loading"}
                className="flex-1 py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-blue-950 font-black transition text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {status === "loading" ? "Enviando…" : "Enviar ✉️"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ─── Carta principal ───────────────────────────────────────────────────────────

export function PaniniCard({
  result,
  input,
  photo,
  playerName,
  cardRef,
}: PaniniCardProps) {
  const [showModal, setShowModal] = useState(false);

  const theme = getCardTheme(result.overall);
  const probPercent = Math.round(result.probability * 100);
  const positionFull =
    POSITION_NAMES[input.position.toUpperCase()] ?? input.position;

  return (
    <div
      ref={cardRef}
      className="flex flex-col gap-3 animate-[fadeIn_0.4s_ease-out] w-full"
    >
      {/* ── Carta portrait 3:4 ── */}
      <div
        className={`panini-card-root w-full rounded-2xl overflow-hidden border-4 ${theme.border} shadow-2xl ${theme.glow} flex flex-col`}
        style={{ background: theme.bg, aspectRatio: "3/4" }}
      >
        {/* Topo: overall + posição + ícone */}
        <div
          className="flex items-start justify-between px-6 pt-4 pb-3 flex-shrink-0"
          style={{ background: theme.accentBg }}
        >
          <div className="flex flex-col leading-none">
            <span className={`text-6xl font-black ${theme.text} leading-none`}>
              {result.overall}
            </span>
            <span
              className={`text-base font-black ${theme.text} tracking-widest mt-1`}
            >
              {input.position}
            </span>
            <span
              className={`text-[10px] font-bold ${theme.textMuted} tracking-wider mt-0.5 uppercase`}
            >
              {theme.tier}
            </span>
          </div>
          <img
            src={fifaIcon}
            alt="FIFA WC 2026"
            className="h-16 w-auto object-contain flex-shrink-0"
            style={{
              filter:
                result.overall >= 65
                  ? "none"
                  : "brightness(1.4) grayscale(0.3)",
            }}
          />
        </div>

        {/* Foto */}
        <div className="flex justify-center flex-shrink-0 py-4">
          {photo ? (
            <img
              src={photo}
              alt="Jogador"
              className={`w-32 h-32 rounded-full object-cover border-4 ${theme.border} shadow-xl`}
            />
          ) : (
            <div
              className={`w-32 h-32 rounded-full border-4 ${theme.border} shadow-xl overflow-hidden flex items-end justify-center`}
              style={{ background: "rgba(0,0,0,0.2)" }}
            >
              <svg
                viewBox="0 0 100 110"
                className={`w-28 h-28 ${theme.text}`}
                fill="currentColor"
                style={{ opacity: 0.5 }}
              >
                <circle cx="50" cy="30" r="20" />
                <ellipse cx="50" cy="95" rx="35" ry="22" />
              </svg>
            </div>
          )}
        </div>

        {/* Nome + posição */}
        <div className="text-center px-6 flex-shrink-0">
          <p
            className={`font-black text-2xl tracking-widest uppercase ${theme.text} leading-tight`}
          >
            {playerName}
          </p>
          <p
            className={`text-xs font-semibold ${theme.textMuted} tracking-wider mt-1 uppercase`}
          >
            {positionFull}
          </p>
        </div>

        {/* Divisória */}
        <div className="flex items-center gap-2 mx-6 my-3 flex-shrink-0">
          <div
            className="flex-1 h-px opacity-30"
            style={{ backgroundColor: "currentColor" }}
          />
          <span className={`text-sm ${theme.textMuted}`}>*</span>
          <div
            className="flex-1 h-px opacity-30"
            style={{ backgroundColor: "currentColor" }}
          />
        </div>

        {/* Stats */}
        <div
          className="grid grid-cols-3 gap-x-2 flex-1 px-6 pb-4"
          style={{ background: theme.accentBg }}
        >
          {STATS_DISPLAY.map(({ key, abbr }) => {
            const v = input[key as keyof typeof input] as number;
            return (
              <div
                key={key}
                className="flex flex-col items-center justify-center"
              >
                <span
                  className={`text-2xl font-black ${theme.text} tabular-nums leading-none`}
                >
                  {v}
                </span>
                <span
                  className={`text-[10px] font-bold ${theme.textMuted} tracking-widest mt-0.5`}
                >
                  {abbr}
                </span>
              </div>
            );
          })}
        </div>

        {/* Badge convocação */}
        <div
          className={`flex-shrink-0 py-3 text-center font-black text-base tracking-widest ${
            result.convocated
              ? "bg-emerald-600 text-white"
              : "bg-red-700 text-white"
          }`}
        >
          {result.convocated ? "CONVOCADO" : "NÃO CONVOCADO"}
        </div>
      </div>

      {/* ── Barra de probabilidade ── */}
      <div className="w-full flex flex-col gap-1.5">
        <div className="flex justify-between text-xs text-blue-200">
          <span>Probabilidade de convocação</span>
          <span className="font-bold text-amber-300">{probPercent}%</span>
        </div>
        <div className="h-2.5 bg-blue-950 rounded-full overflow-hidden border border-blue-700">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              result.convocated ? "bg-emerald-500" : "bg-red-500"
            }`}
            style={{ width: `${probPercent}%` }}
          />
        </div>
        <p className="text-xs text-blue-400">
          {result.convocated
            ? "Este perfil seria convocado para a Copa 2026!"
            : "Este perfil não seria convocado para a Copa 2026."}
        </p>
      </div>

      {/* ── Botão enviar ── */}
      <button
        onClick={() => setShowModal(true)}
        className="w-full py-3 rounded-xl bg-blue-700 hover:bg-blue-600 active:bg-blue-800 text-white font-bold tracking-wide transition text-sm flex items-center justify-center gap-2"
      >
        ✉️ Enviar carta por email
      </button>

      {/* ── Modal ── */}
      {showModal && (
        <SendCardModal
          cardRef={cardRef}
          playerName={playerName}
          onClose={() => setShowModal(false)}
        />
      )}
    </div>
  );
}
