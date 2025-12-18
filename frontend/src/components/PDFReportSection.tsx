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

  const defectRows = useMemo(
    () => [
      { label: "bent", value: result.class_pixel_percentages.bent },
      { label: "color", value: result.class_pixel_percentages.color },
      { label: "flip", value: result.class_pixel_percentages.flip },
      { label: "scratch", value: result.class_pixel_percentages.scratch },
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
      doc.text("Metal Nut Defect Report", 14, y);
      y += 8;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(11);
      doc.text(`Defective: ${result.is_defective ? "Yes" : "No"}`, 14, y);
      y += 6;
      const topPct = ((result.dominant_defect_ratio ?? 0) * 100).toFixed(2);
      doc.text(`Top defect: ${result.dominant_defect || "none"} (${topPct}%)`, 14, y);
      y += 6;
      doc.text(`Defect on nut: ${(result.defect_ratio * 100).toFixed(2)}%`, 14, y);
      y += 6;
      if (typeof result.defect_ratio_image === "number") {
        doc.text(`Defect on image: ${(result.defect_ratio_image * 100).toFixed(2)}%`, 14, y);
        y += 6;
      }
      if (result.quality_decision) {
        doc.text(`Decision: ${result.quality_decision}`, 14, y);
        y += 6;
      }
      y += 4;

      if (reportOptions.includePerClass) {
        doc.text("Per-class pixel percentages:", 14, y);
        y += 6;
        defectRows.forEach((row) => {
          doc.text(`- ${row.label}: ${(row.value * 100).toFixed(2)}%`, 18, y);
          y += 6;
        });
        y += 4;
      }

      if (reportOptions.includeOriginal && originalImage) {
        doc.setFont("helvetica", "bold");
        doc.text("Original", 14, y);
        doc.addImage(originalImage, "PNG", 14, y + 2, 80, 80);
      }
      if (reportOptions.includeMask && overlayImage) {
        doc.setFont("helvetica", "bold");
        doc.text("Overlay", 110, y);
        doc.addImage(overlayImage, "PNG", 110, y + 2, 80, 80);
      }

      doc.save("metal_nut_defect_report.pdf");
    } finally {
      setIsGenerating(false);
    }
  };

  const options = [
    { key: "includeOriginal", label: "Include original image" },
    { key: "includeMask", label: "Include segmentation overlay" },
    { key: "includePerClass", label: "Include per-class analysis" },
    { key: "includeStats", label: "Include defect stats table" },
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
          <h2 className="text-2xl">PDF Report Generation</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="opacity-70">Report Options</h3>
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
            <h3 className="opacity-70">Report Preview</h3>
            <div
              className={`rounded-xl p-6 border-2 border-dashed h-full flex flex-col items-center justify-center gap-4 ${
                isDarkMode ? "border-white/20 bg-white/5" : "border-gray-300 bg-gray-50"
              }`}
            >
              <div className={isDarkMode ? "p-4 rounded-full bg-[#2979FF]/20" : "p-4 rounded-full bg-[#2979FF]/10"}>
                <FileText className="w-12 h-12 text-[#2979FF]" />
              </div>
              <div className="text-center">
                <p className="mb-2">Professional Inspection Report</p>
                <p className={`text-sm ${isDarkMode ? "text-[#E3E9F1]/60" : "text-gray-500"}`}>
                  A4 format · High resolution · Includes visuals
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
              <span>Generating Report...</span>
            </>
          ) : (
            <>
              <Download className="w-5 h-5" />
              <span>Generate Inspection Report</span>
            </>
          )}
        </button>
      </div>
    </section>
  );
}
