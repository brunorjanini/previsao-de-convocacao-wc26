import { useState } from "react";
import { toPng } from "html-to-image";

interface SendCardModalProps {
  cardRef: React.RefObject<HTMLDivElement>;
  playerName: string;
  onClose: () => void;
}

export function SendCardModal({
  cardRef,
  playerName,
  onClose,
}: SendCardModalProps) {
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
      // Captura apenas o card (primeiro filho do wrapper)
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
