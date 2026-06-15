import { useState, useRef } from "react";
import { PlayerForm } from "./components/PlayerForm";
import { PaniniCard } from "./components/PaniniCard";
import { FootballField } from "./components/FootballField";
import fifaIcon from "./assets/icone-fifa26.png";
import type { PredictionOutput, PlayerInput } from "./api";

interface CardState {
  result: PredictionOutput;
  input: PlayerInput;
  photo: string | null;
  playerName: string;
}

export default function App() {
  const [card, setCard] = useState<CardState | null>(null);
  const [currentPosition, setCurrentPosition] = useState("");
  const cardRef = useRef<HTMLDivElement>(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-950 via-blue-900 to-green-950">
      {/* Decoracao de fundo */}
      <div className="fixed inset-0 opacity-[0.04] pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] border-4 border-white rounded-full" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[450px] h-[450px] border-2 border-white rounded-full" />
        <div className="absolute top-1/2 left-0 right-0 h-px bg-white" />
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white" />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-10">

        {/* Header */}
        <header className="flex items-center gap-5 mb-10">
          <img src={fifaIcon} alt="FIFA World Cup 2026" className="h-20 w-auto object-contain flex-shrink-0" />
          <div>
            <h1 className="text-3xl font-black text-white tracking-tight leading-tight">
              Copa do Mundo FIFA 2026
            </h1>
            <p className="text-amber-300 font-semibold text-sm mt-1 tracking-wide">
              Você seria convocado?
            </p>
            <p className="text-blue-300 text-xs mt-1">
              Preencha os atributos, tire uma foto e gere sua carta
            </p>
          </div>
        </header>

        {/* Layout: Campo | Formulario | Carta */}
        <div className="flex flex-col lg:flex-row gap-6 items-start">

          {/* Campo de futebol — coluna esquerda */}
          <div className="w-full lg:w-64 flex-shrink-0 flex flex-col gap-2">
            <p className="text-xs font-bold text-amber-300 tracking-widest uppercase">
              Posição no campo
            </p>
            <FootballField position={currentPosition} />
          </div>

          {/* Formulario — coluna central */}
          <div className="w-full lg:w-80 flex-shrink-0">
            <PlayerForm
              onResult={(result, input, photo, name) => {
                setCard({ result, input, photo, playerName: name });
                setTimeout(() => cardRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 50);
              }}
              onPositionChange={setCurrentPosition}
            />
          </div>

          {/* Carta — coluna direita */}
          <div ref={cardRef} className="flex-1 min-w-0 max-w-md">
            {card ? (
              <PaniniCard
                result={card.result}
                input={card.input}
                photo={card.photo}
                playerName={card.playerName}
              />
            ) : (
              <div className="w-full rounded-2xl border-2 border-dashed border-blue-700 bg-blue-950/30 flex flex-col items-center justify-center gap-4 py-20 px-6" style={{ aspectRatio: "4/3" }}>
                <img src={fifaIcon} alt="" className="h-14 w-auto object-contain opacity-20" />
                <p className="text-blue-400 font-semibold text-sm text-center">
                  Sua carta aparece aqui
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Rodape */}
        <footer className="mt-16 text-center text-blue-700 text-xs">
          Projeto IAC — USP · Previsão com XGBoost + EA FC 26
        </footer>
      </div>
    </div>
  );
}
