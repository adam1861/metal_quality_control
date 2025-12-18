import { Hexagon, Moon, Sun } from "lucide-react";

interface HeaderProps {
  isDarkMode: boolean;
  setIsDarkMode: (value: boolean) => void;
  onNavigate: (section: string) => void;
  activeSection: string;
}

export function Header({ isDarkMode, setIsDarkMode, onNavigate, activeSection }: HeaderProps) {
  const navItems = [
    { label: "Dashboard", id: "dashboard" },
    { label: "Upload", id: "upload" },
    { label: "Results", id: "results" },
    { label: "About", id: "about" },
  ];

  return (
    <header
      className={`sticky top-0 z-50 backdrop-blur-md transition-all duration-300 ${
        isDarkMode ? "bg-[#151C2C]/80 border-b border-white/10" : "bg-white/80 border-b border-gray-200/50"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <button
            onClick={() => onNavigate("dashboard")}
            className="flex items-center gap-3 group cursor-pointer"
          >
            <div
              className={`relative transition-transform duration-300 group-hover:rotate-180 ${
                isDarkMode ? "text-[#2979FF]" : "text-[#2979FF]"
              }`}
            >
              <Hexagon className="w-8 h-8" fill="currentColor" />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className={`w-2 h-2 rounded-full ${isDarkMode ? "bg-[#00E676]" : "bg-[#00E676]"}`} />
              </div>
            </div>
            <span className="font-semibold">Metal Nut AI Inspector</span>
          </button>

          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                  activeSection === item.id
                    ? isDarkMode
                      ? "bg-white/10 text-white"
                      : "bg-gray-100 text-gray-800"
                    : isDarkMode
                    ? "hover:bg-white/10 text-[#E3E9F1]"
                    : "hover:bg-gray-100 text-gray-700"
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              className={`relative w-14 h-7 rounded-full transition-all duration-300 ${
                isDarkMode ? "bg-[#2979FF]" : "bg-gray-300"
              }`}
            >
              <div
                className={`absolute top-1 left-1 w-5 h-5 rounded-full bg-white flex items-center justify-center shadow-lg transition-transform duration-300 ${
                  isDarkMode ? "translate-x-7" : "translate-x-0"
                }`}
              >
                {isDarkMode ? <Moon className="w-3 h-3 text-[#2979FF]" /> : <Sun className="w-3 h-3 text-gray-600" />}
              </div>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
