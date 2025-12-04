import { Upload, Image as ImageIcon, Zap, Loader2 } from "lucide-react";

interface UploadPanelProps {
  isDarkMode: boolean;
  uploadedImage: string | null;
  onImageUpload: (file: File, imageUrl: string) => void;
  isAnalyzing: boolean;
  onAnalyze: () => void;
  advancedMode: boolean;
  setAdvancedMode: (value: boolean) => void;
  error: string | null;
}

export function UploadPanel({
  isDarkMode,
  uploadedImage,
  onImageUpload,
  isAnalyzing,
  onAnalyze,
  advancedMode,
  setAdvancedMode,
  error,
}: UploadPanelProps) {
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        const result = ev.target?.result as string;
        onImageUpload(file, result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        const result = ev.target?.result as string;
        onImageUpload(file, result);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <section
      className={`rounded-2xl p-8 backdrop-blur-sm transition-all duration-300 ${
        isDarkMode
          ? "bg-[#151C2C]/50 border border-white/10"
          : "bg-white border border-gray-200 shadow-lg"
      }`}
    >
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl flex items-center gap-2">
            <Upload className="w-6 h-6 text-[#2979FF]" />
            Upload & Processing
          </h2>

          <div className="flex items-center gap-3">
            <span className="text-sm">Advanced Mode</span>
            <button
              onClick={() => setAdvancedMode(!advancedMode)}
              className={`relative w-12 h-6 rounded-full transition-all duration-300 ${
                advancedMode
                  ? "bg-[#00E676]"
                  : isDarkMode
                  ? "bg-gray-600"
                  : "bg-gray-300"
              }`}
            >
              <div
                className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white shadow-lg transition-transform duration-300 ${
                  advancedMode ? "translate-x-6" : "translate-x-0"
                }`}
              />
            </button>
          </div>
        </div>

        {!uploadedImage ? (
          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 cursor-pointer group hover:border-[#2979FF] ${
              isDarkMode
                ? "border-white/20 bg-white/5 hover:bg-white/10"
                : "border-gray-300 bg-gray-50 hover:bg-gray-100"
            }`}
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />

            <div className="space-y-4">
              <div
                className={`inline-flex p-4 rounded-full transition-all duration-300 group-hover:scale-110 ${
                  isDarkMode ? "bg-[#2979FF]/20" : "bg-[#2979FF]/10"
                }`}
              >
                <ImageIcon className="w-12 h-12 text-[#2979FF]" />
              </div>

              <div>
                <p className="text-lg mb-2">Drag and drop your image here</p>
                <p
                  className={`text-sm ${
                    isDarkMode ? "text-[#E3E9F1]/60" : "text-gray-500"
                  }`}
                >
                  or click to browse — PNG, JPG up to 10MB
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div
              className={`rounded-xl overflow-hidden border ${
                isDarkMode ? "border-white/10" : "border-gray-200"
              }`}
            >
              <img
                src={uploadedImage}
                alt="Uploaded"
                className="w-full h-64 object-contain bg-black/5"
              />
            </div>

            <button
              onClick={onAnalyze}
              disabled={isAnalyzing}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-[#2979FF] to-[#00E676] text-white shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center gap-2"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  <span>Analyze with AI</span>
                </>
              )}
            </button>

            {isAnalyzing && (
              <div className="space-y-2">
                <div
                  className={`h-2 rounded-full overflow-hidden ${
                    isDarkMode ? "bg-white/10" : "bg-gray-200"
                  }`}
                >
                  <div className="h-full bg-gradient-to-r from-[#2979FF] to-[#00E676] animate-progress" />
                </div>
                <p
                  className={`text-sm text-center ${
                    isDarkMode ? "text-[#E3E9F1]/60" : "text-gray-500"
                  }`}
                >
                  Processing image... This may take a few seconds
                </p>
              </div>
            )}

            {error && (
              <div className="p-3 rounded-lg border border-red-400/50 bg-red-500/10 text-red-200 text-sm">
                {error}
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
