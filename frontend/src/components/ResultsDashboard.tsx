import { useState } from "react";
import { Eye, EyeOff, AlertCircle, CheckCircle, Upload } from "lucide-react";
import { PredictionResult } from "../App";

interface ResultsDashboardProps {
  isDarkMode: boolean;
  uploadedImage: string;
  result: PredictionResult;
  onUploadNew: () => void;
}

export function ResultsDashboard({ isDarkMode, uploadedImage, result, onUploadNew }: ResultsDashboardProps) {
  const [showOverlay, setShowOverlay] = useState(true);

  const defectLabels: Record<string, string> = {
    color: "couleur",
    scratch: "rayure",
    none: "aucun",
    background: "arrière-plan",
  };

  const defectColors = {
    color: { color: "#00E676", label: defectLabels.color },
    scratch: { color: "#FCD34D", label: defectLabels.scratch },
  };

  const totalDefectPct = result.defect_ratio * 100;
  const imageDefectPct = typeof result.defect_ratio_image === "number" ? result.defect_ratio_image * 100 : null;
  const dominantDefectLabel =
    defectLabels[result.dominant_defect || "none"] ?? (result.dominant_defect || defectLabels.none);
  const decisionLabel =
    result.quality_decision === "REJECT" ? "REJETÉ" : result.quality_decision === "ACCEPT" ? "ACCEPTÉ" : undefined;

  return (
    <section className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <h2 className="text-2xl flex items-center gap-2">
          <AlertCircle className="w-6 h-6 text-[#2979FF]" />
          Résultats d’analyse
        </h2>
        <button
          onClick={onUploadNew}
          className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-[#2979FF] to-[#00E676] text-white shadow hover:shadow-lg transition-all duration-300"
        >
          <Upload className="w-4 h-4" />
          Importer une nouvelle image
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div
          className={`rounded-2xl p-6 backdrop-blur-sm transition-all duration-300 hover:scale-[1.01] ${
            isDarkMode ? "bg-[#151C2C]/50 border border-white/10" : "bg-white border border-gray-200 shadow-lg"
          }`}
        >
          <h3 className="text-lg mb-4">Image originale</h3>
          <div className={`rounded-xl overflow-hidden border ${isDarkMode ? "border-white/10" : "border-gray-200"}`}>
            <img src={uploadedImage} alt="Image originale" className="w-full h-64 object-contain bg-black/5" />
          </div>
        </div>

        <div
          className={`rounded-2xl p-6 backdrop-blur-sm transition-all duration-300 hover:scale-[1.01] ${
            isDarkMode ? "bg-[#151C2C]/50 border border-white/10" : "bg-white border border-gray-200 shadow-lg"
          }`}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg">Superposition</h3>
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className={`p-2 rounded-lg transition-all duration-200 ${
                isDarkMode ? "hover:bg-white/10" : "hover:bg-gray-100"
              }`}
              title={showOverlay ? "Masquer la superposition" : "Afficher la superposition"}
            >
              {showOverlay ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
            </button>
          </div>

          <div className={`rounded-xl overflow-hidden border relative ${isDarkMode ? "border-white/10" : "border-gray-200"}`}>
            <img src={uploadedImage} alt="Image de base" className="w-full h-64 object-contain bg-black/5" />
            {showOverlay && result.overlay_image_encoded && (
              <img
                src={result.overlay_image_encoded}
                alt="Superposition"
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
          <h3 className="text-lg mb-4">Synthèse IA</h3>

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
                <p className="text-sm opacity-70">Défaut détecté</p>
                <p className={result.is_defective ? "text-red-500" : "text-green-500"}>
                  {result.is_defective ? "OUI" : "NON"}
                </p>
              </div>
            </div>
          </div>

          <div className={isDarkMode ? "bg-white/5 p-4 rounded-xl" : "bg-gray-50 p-4 rounded-xl"}>
            <p className="text-sm opacity-70 mb-1">Type de défaut dominant</p>
            <p className="text-[#2979FF] capitalize">{dominantDefectLabel}</p>
          </div>

          <div className={isDarkMode ? "bg-white/5 p-4 rounded-xl" : "bg-gray-50 p-4 rounded-xl"}>
            <p className="text-sm opacity-70 mb-1">Couverture de défaut</p>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl text-[#EF4444]">{totalDefectPct.toFixed(2)}%</span>
              <span className="text-sm opacity-60">de la surface de l’écrou</span>
            </div>
            {imageDefectPct !== null && (
              <p className="mt-1 text-xs opacity-60">Couverture sur l’image : {imageDefectPct.toFixed(2)}%</p>
            )}
          </div>

          {result.quality_decision && (
            <div className={isDarkMode ? "bg-white/5 p-4 rounded-xl" : "bg-gray-50 p-4 rounded-xl"}>
              <p className="text-sm opacity-70 mb-1">Décision qualité</p>
              <p className={result.quality_decision === "REJECT" ? "text-[#EF4444]" : "text-[#00E676]"}>
                {decisionLabel || result.quality_decision}
              </p>
              {typeof result.reject_threshold_on_nut === "number" && (
                <p className="mt-1 text-xs opacity-60">
                  Seuil de rejet : {(result.reject_threshold_on_nut * 100).toFixed(0)}% de défaut sur l’écrou
                </p>
              )}
            </div>
          )}

          <div className="space-y-3">
            <p className="text-sm opacity-70">Analyse par classe</p>
            {Object.entries(result.class_pixel_percentages)
              .filter(([key]) => key in defectColors)
              .map(([key, value]) => {
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
