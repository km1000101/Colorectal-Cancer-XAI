import React, { createContext, useContext, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { ChatbotSidebar } from "./components/ChatbotSidebar";
import type { ExplainResult } from "./api/client";

// Context for sharing current prediction result with chatbot
interface ResultContextType {
  currentResult: ExplainResult | null;
  setCurrentResult: (result: ExplainResult | null) => void;
}

const ResultContext = createContext<ResultContextType | undefined>(undefined);

export const useResultContext = () => {
  const context = useContext(ResultContext);
  if (!context) {
    throw new Error("useResultContext must be used within Layout");
  }
  return context;
};

export const Layout: React.FC = () => {
  const [currentResult, setCurrentResult] = useState<ExplainResult | null>(null);
  const location = useLocation();
  const isHomePage = location.pathname === "/";

  const scrollToSection = (id: string) => {
    if (isHomePage) {
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } else {
      // If not on home page, navigate to home with hash
      window.location.href = `/#${id}`;
      // After navigation, scroll will be handled by Home component's useEffect
    }
  };

  const scrollToTop = () => {
    if (isHomePage) {
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else {
      window.location.href = "/";
    }
  };

  return (
    <ResultContext.Provider value={{ currentResult, setCurrentResult }}>
      <div className="app-root gradient-bg">
        <header className="top-nav">
          <div className="top-nav__brand">Colorectal Cancer XAI</div>
          <nav className="top-nav__links">
            <button
              onClick={scrollToTop}
              className="top-nav__link-button"
              type="button"
            >
              Home
            </button>
            <button
              onClick={() => scrollToSection("features")}
              className="top-nav__link-button"
              type="button"
            >
              Features
            </button>
            <button
              onClick={() => scrollToSection("about")}
              className="top-nav__link-button"
              type="button"
            >
              About
            </button>
            <button
              onClick={() => scrollToSection("analyze")}
              className="top-nav__link-button"
              type="button"
            >
              Analyze
            </button>
            <button
              onClick={() => scrollToSection("history")}
              className="top-nav__link-button"
              type="button"
            >
              History
            </button>
          </nav>
        </header>

        <main className="page-shell">
          <Outlet />
        </main>

        <ChatbotSidebar currentResult={currentResult} />
      </div>
    </ResultContext.Provider>
  );
};


