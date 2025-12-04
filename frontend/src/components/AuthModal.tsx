import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { createPortal } from "react-dom";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAuthSuccess: (token: string, email: string) => void;
  apiBase: string;
  portalTarget?: HTMLElement | null;
}

export function AuthModal({ isOpen, onClose, onAuthSuccess, apiBase, portalTarget }: AuthModalProps) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      const originalOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      window.scrollTo({ top: 0, behavior: "smooth" });
      return () => {
        document.body.style.overflow = originalOverflow;
      };
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const title = mode === "login" ? "Sign In" : "Create Account";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const endpoint = mode === "login" ? "/auth/login" : "/auth/signup";
      const res = await fetch(`${apiBase.replace(/\/$/, "")}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed (${res.status})`);
      }
      const data = (await res.json()) as { token: string; email: string };
      onAuthSuccess(data.token, data.email);
      onClose();
    } catch (err: any) {
      setError(err?.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  const content = (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm px-4">
      <div className="w-full max-w-md rounded-2xl bg-white text-gray-900 shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200">
          <div>
            <p className="text-sm text-gray-500 uppercase tracking-wide">{mode === "login" ? "Welcome back" : "Join us"}</p>
            <h2 className="text-xl font-semibold">{title}</h2>
          </div>
          <button onClick={onClose} className="p-2 rounded-full hover:bg-gray-100">
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <div className="space-y-2">
            <label className="text-sm text-gray-600">Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm text-gray-600">Password</label>
            <input
              type="password"
              required
              minLength={6}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {error && <div className="text-sm text-red-600">{error}</div>}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-[#2979FF] to-[#00E676] text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50"
          >
            {loading ? "Please wait..." : mode === "login" ? "Sign In" : "Sign Up"}
          </button>

          <p className="text-center text-sm text-gray-500">
            {mode === "login" ? "Don't have an account?" : "Already have an account?"}{" "}
            <button
              type="button"
              onClick={() => setMode(mode === "login" ? "signup" : "login")}
              className="text-blue-600 hover:underline"
            >
              {mode === "login" ? "Create one" : "Sign in"}
            </button>
          </p>
        </form>
      </div>
    </div>
  );

  if (portalTarget) {
    return createPortal(content, portalTarget);
  }
  return content;
}
