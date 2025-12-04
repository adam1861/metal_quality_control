import { Upload, Sparkles } from 'lucide-react';

interface HeroSectionProps {
  isDarkMode: boolean;
  onUploadClick: () => void;
}

export function HeroSection({ isDarkMode, onUploadClick }: HeroSectionProps) {
  return (
    <section className="relative overflow-hidden rounded-3xl">
      {/* Background with gradient mesh */}
      <div className={`relative p-12 md:p-16 lg:p-20 ${
        isDarkMode
          ? 'bg-gradient-to-br from-[#151C2C] via-[#1a2332] to-[#151C2C]'
          : 'bg-gradient-to-br from-blue-50 via-white to-blue-50'
      }`}>
        {/* Decorative elements */}
        <div className="absolute top-10 right-10 w-64 h-64 bg-[#2979FF] rounded-full opacity-10 blur-3xl animate-pulse" />
        <div className="absolute bottom-10 left-10 w-48 h-48 bg-[#00E676] rounded-full opacity-10 blur-3xl animate-pulse delay-1000" />
        
        {/* Content */}
        <div className="relative z-10 max-w-3xl mx-auto text-center space-y-6">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full backdrop-blur-sm border transition-all duration-300 hover:scale-105 ${
            isDarkMode 
              ? 'bg-white/5 border-white/10' 
              : 'bg-white/50 border-gray-200'
          }">
            <Sparkles className="w-4 h-4 text-[#00E676]" />
            <span className="text-sm">Powered by Advanced AI</span>
          </div>

          <h1 className="text-4xl md:text-5xl lg:text-6xl">
            <span className="block mb-2">Automated Industrial</span>
            <span className="bg-gradient-to-r from-[#2979FF] to-[#00E676] bg-clip-text text-transparent">
              Defect Detection
            </span>
          </h1>

          <p className={`text-lg md:text-xl max-w-2xl mx-auto ${
            isDarkMode ? 'text-[#E3E9F1]/70' : 'text-gray-600'
          }`}>
            Upload an image of a metal nut and let the AI detect surface defects in real time.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <button
              onClick={onUploadClick}
              className="group relative px-8 py-4 bg-gradient-to-r from-[#2979FF] to-[#00E676] rounded-xl text-white shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105 flex items-center gap-2"
            >
              <Upload className="w-5 h-5 group-hover:translate-y-[-2px] transition-transform" />
              <span>Upload Image</span>
              <div className="absolute inset-0 rounded-xl bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            </button>
          </div>
        </div>

        {/* Floating shapes */}
        <div className="absolute top-20 left-[10%] w-16 h-16 border-2 border-[#2979FF]/30 rounded-xl rotate-45 animate-float" />
        <div className="absolute bottom-20 right-[15%] w-12 h-12 border-2 border-[#00E676]/30 rounded-full animate-float delay-500" />
      </div>
    </section>
  );
}
