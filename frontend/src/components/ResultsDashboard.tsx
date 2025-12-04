import { useState } from "react";
import { Eye, EyeOff, AlertCircle, CheckCircle } from "lucide-react";
import { PredictionResult } from "../App";

interface ResultsDashboardProps {
  isDarkMode: boolean;
  uploadedImage: string;
  result: PredictionResult;
}

export function ResultsDashboard({ isDarkMode, uploadedImage, result }: ResultsDashboardProps) {
  const [showOverlay, setShowOverlay] = useState(true);

  const defectColors = {
    bent: { color: "#EF4444", label: "bent" },
    color: { color: "#00E676", label: "color" },
    flip: { color: "#3B82F6", label: "flip" },
    scratch: { color: "#FCD34D", label: "scratch" },
  };

  const totalDefectPct = result.defect_ratio * 100;

  return (
    <section className="space-y-6">
      <h2 className="text-2xl flex items-center gap-2">
        <AlertCircle className="w-6 h-6 text-[#2979FF]" />
        Analysis Results
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div
          className={`rounded-2xl p-6 backdrop-blur-sm transition-all duration-300 hover:scale-[1.01] ${
            isDarkMode ? "bg-[#151C2C]/50 border border-white/10" : "bg-white border border-gray-200 shadow-lg"
          }`}
        >
          <h3 className="text-lg mb-4">Original Image</h3>
          <div className={`rounded-xl overflow-hidden border ${isDarkMode ? "border-white/10" : "border-gray-200"}`}>
            <img src={uploadedImage} alt="Original" className="w-full h-64 object-contain bg-black/5" />
          </div>
        </div>

        <div
          className={`rounded-2xl p-6 backdrop-blur-sm transition-all duration-300 hover:scale-[1.01] ${
            isDarkMode ? "bg-[#151C2C]/50 border border-white/10" : "bg-white border border-gray-200 shadow-lg"
          }`}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg">Overlay</h3>
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className={`p-2 rounded-lg transition-all duration-200 ${
                isDarkMode ? "hover:bg-white/10" : "hover:bg-gray-100"
              }`}
              title={showOverlay ? "Hide overlay" : "Show overlay"}
            >
              {showOverlay ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
            </button>
          </div>

          <div className={`rounded-xl overflow-hidden border relative ${isDarkMode ? "border-white/10" : "border-gray-200"}`}>
            <img src={uploadedImage} alt="Base" className="w-full h-64 object-contain bg-black/5" />
            {showOverlay && result.overlay_image_encoded && (
              <img
                src={result.overlay_image_encoded}
                alt="Overlay"
                className="absolute inset-0 w-full h-full object-contain mix-blend-screen"
              />
            )}
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2">
            {Object.entries(defectColors).map(([key, { color, label }]) => (
              <div key={key} className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-sm capitalize">{label}</span>
              </div>
            ))}
          </div>
        </div>

        <div
          className={`rounded-2xl p-6 backdrop-blur-sm space-y-4 ${
            isDarkMode ? "bg-[#151C2C]/50 border border-white/10" : "bg-white border border-gray-200 shadow-lg"
          }`}
        >
          <h3 className="text-lg mb-4">AI Summary</h3>

          <div
            className={`p-4 rounded-xl flex items-center justify-between ${
              result.is_defective ? "bg-red-500/10 border border-red-500/20" : "bg-green-500/10 border border-green-500/20"
            }`}
          >
            <div className="flex items-center gap-3">
              {result.is_defective ? (
                <AlertCircle className="w-6 h-6 text-red-500" />
              ) : (
                <CheckCircle className="w-6 h-6 text-green-500" />
              )}
              <div>
                <p className="text-sm opacity-70">Defect Detected</p>
                <p className={result.is_defective ? "text-red-500" : "text-green-500"}>
                  {result.is_defective ? "YES" : "NO"}
                </p>
              </div>
            </div>
          </div>

          <div className={isDarkMode ? "bg-white/5 p-4 rounded-xl" : "bg-gray-50 p-4 rounded-xl"}>
            <p className="text-sm opacity-70 mb-1">Dominant Defect Type</p>
            <p className="text-[#2979FF] capitalize">{result.dominant_defect || "None"}</p>
          </div>

          <div className={isDarkMode ? "bg-white/5 p-4 rounded-xl" : "bg-gray-50 p-4 rounded-xl"}>
            <p className="text-sm opacity-70 mb-1">Defect Coverage</p>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl text-[#EF4444]">{totalDefectPct.toFixed(2)}%</span>
              <span className="text-sm opacity-60">of surface</span>
            </div>
          </div>

          <div className="space-y-3">
            <p className="text-sm opacity-70">Per-Class Analysis</p>
            {Object.entries(result.class_pixel_percentages).map(([key, value]) => {
              const defect = defectColors[key as keyof typeof defectColors];
              return (
                <div key={key} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: defect.color }} />
                      {defect.label}
                    </span>
                    <span>{(value * 100).toFixed(2)}%</span>
                  </div>
                  <div className={`h-2 rounded-full overflow-hidden ${isDarkMode ? "bg-white/10" : "bg-gray-200"}`}>
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.min(value * 100, 100)}%`,
                        backgroundColor: defect.color,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
