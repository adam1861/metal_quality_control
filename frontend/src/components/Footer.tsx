import { Github, FileText, Mail } from 'lucide-react';

interface FooterProps {
  isDarkMode: boolean;
}

export function Footer({ isDarkMode }: FooterProps) {
  const links = [
    { icon: FileText, label: 'Documentation', href: '#' },
    { icon: Github, label: 'GitHub', href: '#' },
    { icon: Mail, label: 'Contact', href: '#' },
  ];

  return (
    <footer className={`mt-20 border-t transition-colors duration-300 ${
      isDarkMode ? 'border-white/10 bg-[#151C2C]/30' : 'border-gray-200 bg-white/50'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left Section */}
          <div className="space-y-4">
            <h3 className="text-lg">Metal Nut AI Inspector</h3>
            <p className={`text-sm max-w-md ${
              isDarkMode ? 'text-[#E3E9F1]/60' : 'text-gray-600'
            }`}>
              Advanced AI-powered quality inspection system for industrial manufacturing. 
              Detect surface defects with precision and speed.
            </p>
            <p className={`text-sm ${
              isDarkMode ? 'text-[#E3E9F1]/50' : 'text-gray-500'
            }`}>
              Developed by <span className="text-[#2979FF]">Adam Lachkar</span>
            </p>
          </div>

          {/* Right Section - Links */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 sm:justify-end">
            {links.map((link) => (
              <a
                key={link.label}
                href={link.href}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                  isDarkMode
                    ? 'hover:bg-white/10 text-[#E3E9F1]'
                    : 'hover:bg-gray-100 text-gray-700'
                }`}
              >
                <link.icon className="w-4 h-4" />
                <span>{link.label}</span>
              </a>
            ))}
          </div>
        </div>

        {/* Bottom Bar */}
        <div className={`mt-8 pt-8 border-t text-center text-sm ${
          isDarkMode 
            ? 'border-white/10 text-[#E3E9F1]/50' 
            : 'border-gray-200 text-gray-500'
        }`}>
          <p>© 2025 Metal Nut AI Inspector. All rights reserved.</p>
          <p className="mt-2 text-xs">
            ⚠️ Figma Make is not meant for collecting PII or securing sensitive data.
          </p>
        </div>
      </div>
    </footer>
  );
}
