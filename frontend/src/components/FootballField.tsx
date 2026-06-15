const POSITION_COORDS: Record<string, { x: number; y: number }> = {
  GK:  { x: 50, y: 88 },
  LB:  { x: 18, y: 72 },
  CB:  { x: 42, y: 76 },
  RB:  { x: 82, y: 72 },
  CDM: { x: 50, y: 58 },
  LM:  { x: 18, y: 44 },
  CM:  { x: 50, y: 44 },
  RM:  { x: 82, y: 44 },
  CAM: { x: 50, y: 30 },
  LW:  { x: 18, y: 22 },
  RW:  { x: 82, y: 22 },
  ST:  { x: 50, y: 14 },
};

interface FootballFieldProps {
  position: string;
}

export function FootballField({ position }: FootballFieldProps) {
  const coord = POSITION_COORDS[position.toUpperCase()];

  return (
    <div className="w-full rounded-xl overflow-hidden border border-blue-700 shadow-inner">
      <svg
        viewBox="0 0 200 280"
        className="w-full"
        style={{ background: "#1a5c2a" }}
      >
        {/* Gramado alternado */}
        {Array.from({ length: 7 }).map((_, i) => (
          <rect
            key={i}
            x={0}
            y={i * 40}
            width={200}
            height={40}
            fill={i % 2 === 0 ? "#1a5c2a" : "#1d6830"}
          />
        ))}

        {/* Linhas brancas */}
        {/* Borda do campo */}
        <rect x={10} y={8} width={180} height={264} fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth={1.5} />

        {/* Linha do meio */}
        <line x1={10} y1={140} x2={190} y2={140} stroke="rgba(255,255,255,0.7)" strokeWidth={1} />

        {/* Círculo central */}
        <circle cx={100} cy={140} r={28} fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth={1} />
        <circle cx={100} cy={140} r={1.5} fill="rgba(255,255,255,0.7)" />

        {/* Área grande — ataque (topo) */}
        <rect x={45} y={8} width={110} height={42} fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth={1} />
        {/* Área pequena — ataque */}
        <rect x={72} y={8} width={56} height={18} fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth={1} />
        {/* Gol — ataque */}
        <rect x={83} y={4} width={34} height={6} fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth={1} />
        {/* Pênalti ataque */}
        <circle cx={100} cy={36} r={1.5} fill="rgba(255,255,255,0.7)" />
        {/* Arco da área (ataque) */}
        <path d="M 68 50 A 28 28 0 0 0 132 50" fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth={1} />

        {/* Área grande — defesa (base) */}
        <rect x={45} y={230} width={110} height={42} fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth={1} />
        {/* Área pequena — defesa */}
        <rect x={72} y={254} width={56} height={18} fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth={1} />
        {/* Gol — defesa */}
        <rect x={83} y={270} width={34} height={6} fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth={1} />
        {/* Pênalti defesa */}
        <circle cx={100} cy={244} r={1.5} fill="rgba(255,255,255,0.7)" />
        {/* Arco da área (defesa) */}
        <path d="M 68 230 A 28 28 0 0 1 132 230" fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth={1} />

        {/* Marcador da posição */}
        {coord && (
          <>
            {/* Halo pulsante */}
            <circle
              cx={(coord.x / 100) * 200}
              cy={(coord.y / 100) * 280}
              r={12}
              fill="rgba(251,191,36,0.25)"
            />
            {/* Círculo da posição */}
            <circle
              cx={(coord.x / 100) * 200}
              cy={(coord.y / 100) * 280}
              r={8}
              fill="#f59e0b"
              stroke="white"
              strokeWidth={1.5}
            />
            {/* Label da posição */}
            <text
              x={(coord.x / 100) * 200}
              y={(coord.y / 100) * 280}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={6}
              fontWeight="bold"
              fill="white"
            >
              {position.toUpperCase()}
            </text>
          </>
        )}
      </svg>
    </div>
  );
}
