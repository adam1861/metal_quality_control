import { useMemo, useState } from "react";
import { FileText, Download, Check } from "lucide-react";
import { jsPDF } from "jspdf";
import { PredictionResult } from "../App";

interface PDFReportSectionProps {
  isDarkMode: boolean;
  result: PredictionResult;
  originalImage: string;
  overlayImage?: string;
}

export function PDFReportSection({ isDarkMode, result, originalImage, overlayImage }: PDFReportSectionProps) {
  const [reportOptions, setReportOptions] = useState({
    includeOriginal: true,
    includeMask: true,
    includePerClass: true,
    includeStats: true,
  });

  const [isGenerating, setIsGenerating] = useState(false);

  const defectLabelsFr: Record<string, string> = {
    color: "couleur",
    scratch: "rayure",
    none: "aucun",
  };

  const defectRows = useMemo(
    () => [
      { label: defectLabelsFr.color, value: result.class_pixel_percentages.color },
      { label: defectLabelsFr.scratch, value: result.class_pixel_percentages.scratch },
    ],
    [result]
  );

  const toggleOption = (key: keyof typeof reportOptions) => {
    setReportOptions((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleGenerateReport = async () => {
    setIsGenerating(true);
    try {
      const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
      let y = 15;

      doc.setFont("helvetica", "bold");
      doc.setFontSize(18);
      doc.text("Rapport d'inspection - Écrou métallique", 14, y);
      y += 8;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(11);
      doc.text(`Défaut détecté : ${result.is_defective ? "Oui" : "Non"}`, 14, y);
      y += 6;
      const topPct = ((result.dominant_defect_ratio ?? 0) * 100).toFixed(2);
      const dominant = result.dominant_defect || "none";
      const dominantLabel = defectLabelsFr[dominant] ?? dominant;
      doc.text(`Défaut principal : ${dominantLabel} (${topPct}%)`, 14, y);
      y += 6;
      doc.text(`Défaut sur l'écrou : ${(result.defect_ratio * 100).toFixed(2)}%`, 14, y);
      y += 6;
      if (typeof result.defect_ratio_image === "number") {
        doc.text(`Défaut sur l'image : ${(result.defect_ratio_image * 100).toFixed(2)}%`, 14, y);
        y += 6;
      }
      if (result.quality_decision) {
        const decisionLabel =
          result.quality_decision === "REJECT"
            ? "REJETÉ"
            : result.quality_decision === "ACCEPT"
            ? "ACCEPTÉ"
            : result.quality_decision;
        doc.text(`Décision : ${decisionLabel}`, 14, y);
        y += 6;
      }
      y += 4;

      if (reportOptions.includePerClass) {
        doc.text("Pourcentage de pixels par classe :", 14, y);
        y += 6;
        defectRows.forEach((row) => {
          doc.text(`- ${row.label}: ${(row.value * 100).toFixed(2)}%`, 18, y);
          y += 6;
        });
        y += 4;
      }

      if (reportOptions.includeOriginal && originalImage) {
        doc.setFont("helvetica", "bold");
        doc.text("Image originale", 14, y);
        doc.addImage(originalImage, "PNG", 14, y + 2, 80, 80);
      }
      if (reportOptions.includeMask && overlayImage) {
        doc.setFont("helvetica", "bold");
        doc.text("Superposition", 110, y);
        doc.addImage(overlayImage, "PNG", 110, y + 2, 80, 80);
      }

      doc.save("rapport_inspection_ecrou_metallique.pdf");
    } finally {
      setIsGenerating(false);
    }
  };

  const options = [
    { key: "includeOriginal", label: "Inclure l’image originale" },
    { key: "includeMask", label: "Inclure la superposition de segmentation" },
    { key: "includePerClass", label: "Inclure l’analyse par classe" },
    { key: "includeStats", label: "Inclure le tableau de statistiques des défauts" },
  ] as const;

  return (
    <section
      className={`rounded-2xl p-8 backdrop-blur-sm transition-all duration-300 ${
        isDarkMode ? "bg-[#151C2C]/50 border border-white/10" : "bg-white border border-gray-200 shadow-lg"
      }`}
    >
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <FileText className="w-6 h-6 text-[#2979FF]" />
          <h2 className="text-2xl">Génération du rapport PDF</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="opacity-70">Options du rapport</h3>
            <div className="space-y-3">
              {options.map(({ key, label }) => (
                <label
                  key={key}
                  className={`flex items-center gap-3 p-4 rounded-xl cursor-pointer transition-all duration-200 ${
                    isDarkMode ? "hover:bg-white/5 border border-white/10" : "hover:bg-gray-50 border border-gray-200"
                  }`}
                >
                  <div className="relative">
                    <input type="checkbox" checked={reportOptions[key]} onChange={() => toggleOption(key)} className="sr-only" />
                    <div
                      className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-all duration-200 ${
                        reportOptions[key]
                          ? "bg-[#00E676] border-[#00E676]"
                          : isDarkMode
                          ? "border-white/30"
                          : "border-gray-300"
                      }`}
                    >
                      {reportOptions[key] && <Check className="w-3 h-3 text-white" />}
                    </div>
                  </div>
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="opacity-70">Aperçu du rapport</h3>
            <div
              className={`rounded-xl p-6 border-2 border-dashed h-full flex flex-col items-center justify-center gap-4 ${
                isDarkMode ? "border-white/20 bg-white/5" : "border-gray-300 bg-gray-50"
              }`}
            >
              <div className={isDarkMode ? "p-4 rounded-full bg-[#2979FF]/20" : "p-4 rounded-full bg-[#2979FF]/10"}>
                <FileText className="w-12 h-12 text-[#2979FF]" />
              </div>
              <div className="text-center">
                <p className="mb-2">Rapport d’inspection professionnel</p>
                <p className={`text-sm ${isDarkMode ? "text-[#E3E9F1]/60" : "text-gray-500"}`}>
                  Format A4 · Haute résolution · Inclut les visuels
                </p>
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={handleGenerateReport}
          disabled={isGenerating}
          className="w-full py-4 rounded-xl bg-gradient-to-r from-[#2979FF] to-[#00E676] text-white shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center gap-2"
        >
          {isGenerating ? (
            <>
              <div className="w-5 h-5 border-2 border-white/30 border-top-white rounded-full animate-spin" />
              <span>Génération du rapport...</span>
            </>
          ) : (
            <>
              <Download className="w-5 h-5" />
              <span>Générer le rapport d’inspection</span>
            </>
          )}
        </button>
      </div>
    </section>
  );
}
