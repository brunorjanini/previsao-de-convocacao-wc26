interface StatSliderProps {
  label: string;
  abbr: string;
  value: number;
  onChange: (v: number) => void;
}

export function StatSlider({ label, abbr, value, onChange }: StatSliderProps) {
  const color =
    value >= 80 ? "text-emerald-400" :
    value >= 60 ? "text-yellow-400" :
    "text-red-400";

  return (
    <div className="flex items-center gap-3">
      <span className="w-10 text-xs font-bold text-amber-300 tracking-widest">{abbr}</span>
      <input
        type="range"
        min={0}
        max={99}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        title={label}
        className="flex-1 h-2 rounded-full accent-amber-400 cursor-pointer"
      />
      <span className={`w-8 text-right font-bold text-sm tabular-nums ${color}`}>
        {value}
      </span>
    </div>
  );
}
