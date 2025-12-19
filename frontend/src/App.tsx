import { useState } from "react";
import { Header } from "./components/Header";
import { HeroSection } from "./components/HeroSection";
import { UploadPanel } from "./components/UploadPanel";
import { ResultsDashboard } from "./components/ResultsDashboard";
import { PDFReportSection } from "./components/PDFReportSection";
import { Footer } from "./components/Footer";

export interface PredictionResult {
  is_defective: boolean;
  defect_ratio: number;
  defect_ratio_on_nut?: number;
  defect_ratio_image?: number;
  nut_pixel_count?: number;
  defect_pixel_count_on_nut?: number;
  defect_pixel_count_image?: number;
  reject_threshold_on_nut?: number;
  quality_decision?: "ACCEPT" | "REJECT";
  class_pixel_percentages: {
    bent: number;
    color: number;
    flip: number;
    scratch: number;
  };
  class_pixel_percentages_on_nut?: {
    bent: number;
    color: number;
    flip: number;
    scratch: number;
  };
  class_pixel_percentages_image?: {
    bent: number;
    color: number;
    flip: number;
    scratch: number;
  };
  dominant_defect?: string;
  dominant_defect_ratio?: number;
  mask_encoded?: string;
  overlay_image_encoded?: string;
  image?: { width: number; height: number };
}

const API_URL = (() => {
  const envBase = (import.meta as any).env?.VITE_API_URL;
  if (envBase) return `${envBase.replace(/\/$/, "")}/predict`;
  try {
    const url = new URL(window.location.href);
    if (url.port === "5500") {
      url.port = "8000";
      url.pathname = "";
      url.search = "";
      url.hash = "";
      return `${url.origin}/predict`;
    }
    return `${url.origin}/predict`;
  } catch {
    return "http://localhost:8000/predict";
  }
})();

export default function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [advancedMode, setAdvancedMode] = useState(true);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<string>("dashboard");

  const handleImageUpload = (file: File, previewUrl: string) => {
    setUploadedFile(file);
    setUploadedImage(previewUrl);
    setResult(null);
    setError(null);
    setActiveSection("upload");
  };

  const handleAnalyze = async () => {
    if (!uploadedFile) return;
    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", uploadedFile);
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Requête échouée (${response.status})`);
      }
      const data = (await response.json()) as PredictionResult;
      setResult(data);
      setActiveSection("results");
    } catch (err: any) {
      console.error(err);
      setError((err?.message || "Échec de l’analyse.") + ` (API: ${API_URL})`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleUploadClick = () => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.onchange = (e: Event) => {
      const target = e.target as HTMLInputElement;
      const file = target.files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const result = e.target?.result as string;
          handleImageUpload(file, result);
        };
        reader.readAsDataURL(file);
      }
    };
    fileInput.click();
  };

  const navigateTo = (page: string) => {
    setActiveSection(page);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div
      className={`min-h-screen transition-colors duration-300 ${
        isDarkMode
          ? "bg-gradient-to-br from-[#0B0F19] via-[#101726] to-[#0B0F19] text-[#E3E9F1]"
          : "bg-[#F4F7FA] text-gray-900"
      }`}
    >
      <Header
        isDarkMode={isDarkMode}
        setIsDarkMode={setIsDarkMode}
        onNavigate={navigateTo}
        activeSection={activeSection}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
        {activeSection === "dashboard" && (
          <HeroSection isDarkMode={isDarkMode} onUploadClick={() => navigateTo("upload")} />
        )}

        {activeSection === "upload" && (
          <UploadPanel
            isDarkMode={isDarkMode}
            uploadedImage={uploadedImage}
            onImageUpload={handleImageUpload}
            onUploadNew={handleUploadClick}
            isAnalyzing={isAnalyzing}
            onAnalyze={handleAnalyze}
            advancedMode={advancedMode}
            setAdvancedMode={setAdvancedMode}
            error={error}
          />
        )}

        {activeSection === "results" && (
          result && uploadedImage ? (
            <>
              <ResultsDashboard
                isDarkMode={isDarkMode}
                uploadedImage={uploadedImage}
                result={result}
                onUploadNew={handleUploadClick}
              />
              <PDFReportSection
                isDarkMode={isDarkMode}
                result={result}
                originalImage={uploadedImage}
                overlayImage={result.overlay_image_encoded || undefined}
              />
            </>
          ) : (
            <div className={`rounded-2xl p-8 border ${isDarkMode ? "border-white/10 bg-white/5" : "border-gray-200 bg-white shadow"} `}>
              <h3 className="text-xl font-semibold mb-2">Aucun résultat pour le moment</h3>
              <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
                Lancez une analyse dans la page Importer pour afficher les résultats ici.
              </p>
              <button
                className="mt-4 px-4 py-2 rounded-lg bg-gradient-to-r from-[#2979FF] to-[#00E676] text-white shadow hover:shadow-lg"
                onClick={() => navigateTo("upload")}
              >
                Aller à Importer
              </button>
            </div>
          )
        )}

        {activeSection === "about" && (
          <div className={`rounded-2xl p-8 border ${isDarkMode ? "border-white/10 bg-white/5" : "border-gray-200 bg-white shadow"}`}>
            <h3 className="text-2xl font-semibold mb-3">À propos</h3>
            <p className={isDarkMode ? "text-[#E3E9F1]/80" : "text-gray-700"}>
              Détection multi-classes de défauts pour MVTec metal_nut (bent, color, flip, scratch) : préparation des
              données, entraînement U-Net, inférence via API, et visualisation des résultats dans cette interface avec
              génération de rapport PDF.
            </p>
            <div className="mt-4 space-x-3">
              <button
                className="px-4 py-2 rounded-lg bg-gradient-to-r from-[#2979FF] to-[#00E676] text-white shadow hover:shadow-lg"
                onClick={() => navigateTo("upload")}
              >
                Importer une image
              </button>
              <button
                className={`px-4 py-2 rounded-lg border ${isDarkMode ? "border-white/20 text-[#E3E9F1]" : "border-gray-300 text-gray-700"} hover:shadow`}
                onClick={() => navigateTo("results")}
              >
                Voir les résultats
              </button>
            </div>
          </div>
        )}
      </main>

      <Footer isDarkMode={isDarkMode} />
    </div>
  );
}
