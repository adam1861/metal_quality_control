import {
  AlertCircle,
  CheckCircle,
  ChevronDown,
  Download,
  Eye,
  EyeOff,
  FileText,
  Image as ImageIcon,
  Sparkles,
  Upload,
  Zap,
} from "lucide-react";

interface HeroSectionProps {
  isDarkMode: boolean;
  onUploadClick: () => void;
}

export function HeroSection({ isDarkMode, onUploadClick }: HeroSectionProps) {
  const scrollToId = (id: string) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const problemCards = [
    {
      title: "L’attention humaine varie",
      body: "L’inspection manuelle dépend de l’attention et de la fatigue, ce qui rend les décisions moins cohérentes dans le temps.",
      Icon: EyeOff,
    },
    {
      title: "Les défauts sont subtils",
      body: "Les défauts peuvent être petits et difficiles à détecter de façon constante, surtout avec des éclairages ou des angles différents.",
      Icon: ImageIcon,
    },
    {
      title: "Pression de cadence",
      body: "Un volume de production élevé impose des décisions plus rapides, sans sacrifier la qualité d’inspection.",
      Icon: Zap,
    },
    {
      title: "Les défauts manqués coûtent cher",
      body: "Des défauts non détectés peuvent entraîner des problèmes d’assemblage et des arrêts, impactant le rendement et la livraison.",
      Icon: AlertCircle,
    },
  ] as const;

  const impactCards = [
    {
      title: "Production à fort volume",
      body: "Les lignes ont besoin de décisions répétables, rapidement, équipe après équipe.",
      Icon: Zap,
    },
    {
      title: "Coût des retouches",
      body: "Une détection tardive augmente les retouches, les rebuts et la complexité de manutention.",
      Icon: Download,
    },
    {
      title: "Risque de défaillance",
      body: "Des pièces défectueuses peuvent provoquer des défauts d’assemblage et des arrêts imprévus.",
      Icon: AlertCircle,
    },
    {
      title: "Traçabilité",
      body: "Les équipes qualité ont besoin de preuves claires : quoi, où et pourquoi.",
      Icon: FileText,
    },
  ] as const;

  const howItWorksSteps = [
    {
      title: "Import de l’image et validation",
      body: "Importez une image d’écrou métallique ; le système valide le fichier avant de lancer l’analyse.",
      Icon: Upload,
    },
    {
      title: "Prétraitement (redimensionnement / normalisation)",
      body: "L’image est redimensionnée et normalisée pour correspondre à ce que le modèle attend.",
      Icon: Sparkles,
    },
    {
      title: "Segmentation de l’écrou (focus sur la surface)",
      body: "Un masque de l’écrou est créé pour exclure le fond et rendre les métriques objectives.",
      Icon: Eye,
    },
    {
      title: "Segmentation par apprentissage profond (localisation au pixel près)",
      body: "Un modèle de segmentation prédit la classe de défaut pour chaque pixel afin de localiser précisément les zones défectueuses.",
      Icon: Zap,
    },
    {
      title: "Couverture de défaut calculée uniquement sur la surface de l’écrou",
      body: "Le % de défaut est calculé uniquement sur les pixels de l’écrou (non dilué par le fond).",
      Icon: CheckCircle,
    },
    {
      title: "Analyse par classe",
      body: "Le système rapporte la couverture par classe : rayure, couleur, inversion et plié.",
      Icon: AlertCircle,
    },
    {
      title: "Rapport PDF exportable pour la traçabilité",
      body: "Générez un rapport PDF incluant les visuels et les métriques d’inspection.",
      Icon: FileText,
    },
  ] as const;

  const defectCards = [
    {
      title: "Rayure",
      body: "Marques fines ou localisées pouvant affecter l’ajustement ou l’usure.",
      accent: "#FCD34D",
    },
    {
      title: "Couleur",
      body: "Décoloration de surface ou anomalies de matière visibles sur l’écrou.",
      accent: "#00E676",
    },
    {
      title: "Inversion",
      body: "Changement de motif de surface ou incohérence de zone détectée comme classe de défaut.",
      accent: "#3B82F6",
    },
    {
      title: "Plié",
      body: "Déformation liée à la géométrie visible sur la surface de l’écrou.",
      accent: "#EF4444",
    },
  ] as const;

  return (
    <div className="space-y-16">
      <section className="relative overflow-hidden rounded-3xl">
        <div
          className={`relative p-12 md:p-16 lg:p-20 ${
            isDarkMode
              ? "bg-gradient-to-br from-[#151C2C] via-[#1a2332] to-[#151C2C]"
              : "bg-gradient-to-br from-blue-50 via-white to-blue-50"
          }`}
        >
          <div className="absolute top-10 right-10 w-64 h-64 bg-[#2979FF] rounded-full opacity-10 blur-3xl animate-pulse" />
          <div className="absolute bottom-10 left-10 w-48 h-48 bg-[#00E676] rounded-full opacity-10 blur-3xl animate-pulse delay-1000" />

          <div className="relative z-10 max-w-4xl mx-auto text-center space-y-6" data-reveal>
            <div
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-full backdrop-blur-sm border transition-all duration-300 hover:scale-105 ${
                isDarkMode ? "bg-white/5 border-white/10" : "bg-white/50 border-gray-200"
              }`}
            >
              <Sparkles className="w-4 h-4 text-[#00E676]" />
              <span className="text-sm">Inspection qualité assistée par IA</span>
            </div>

            <h1 className="text-4xl md:text-5xl lg:text-6xl">
              <span className="block mb-2">Détection automatisée</span>
              <span className="bg-gradient-to-r from-[#2979FF] to-[#00E676] bg-clip-text text-transparent">
                des défauts industriels
              </span>
            </h1>

            <p className={`text-lg md:text-xl max-w-3xl mx-auto ${isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}`}>
              Importez une image d’un écrou métallique et obtenez une carte de défauts au pixel près, une couverture
              calculée uniquement sur la surface de l’écrou, une analyse par classe et un rapport PDF exportable pour la
              traçabilité.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
              <button
                onClick={onUploadClick}
                className="group relative px-8 py-4 bg-gradient-to-r from-[#2979FF] to-[#00E676] rounded-xl text-white shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105 flex items-center gap-2"
              >
                <Upload className="w-5 h-5 group-hover:translate-y-[-2px] transition-transform" />
                <span>Démarrer l’inspection</span>
                <div className="absolute inset-0 rounded-xl bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </button>

              <button
                onClick={() => scrollToId("how-it-works")}
                className={`px-8 py-4 rounded-xl border transition-all duration-300 hover:shadow-lg ${
                  isDarkMode ? "border-white/20 text-[#E3E9F1] hover:bg-white/5" : "border-gray-300 text-gray-700 hover:bg-white"
                }`}
              >
                Comment ça marche
              </button>
            </div>

            <button
              onClick={() => scrollToId("problem")}
              className={`inline-flex items-center justify-center gap-2 mx-auto pt-2 text-sm transition-colors ${
                isDarkMode ? "text-[#E3E9F1]/60 hover:text-[#E3E9F1]" : "text-gray-500 hover:text-gray-800"
              }`}
            >
              <span>Défiler</span>
              <ChevronDown className="w-4 h-4" />
            </button>
          </div>

          <div className="absolute top-20 left-[10%] w-16 h-16 border-2 border-[#2979FF]/30 rounded-xl rotate-45 animate-float" />
          <div className="absolute bottom-20 right-[15%] w-12 h-12 border-2 border-[#00E676]/30 rounded-full animate-float delay-500" />
        </div>
      </section>

      <section id="problem" data-reveal>
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <h2 className="text-3xl">Le problème</h2>
          <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
            L’inspection manuelle ne suit pas le rythme des exigences de cadence et de qualité. Les petits défauts sont
            faciles à manquer, et des décisions incohérentes réduisent la confiance sur la ligne.
          </p>
        </div>

        <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {problemCards.map(({ title, body, Icon }, idx) => (
            <div
              key={title}
              data-reveal
              style={{ transitionDelay: `${idx * 80}ms` }}
              className={`h-full rounded-2xl p-6 border backdrop-blur-sm transition-all duration-300 hover:scale-[1.01] ${
                isDarkMode ? "border-white/10 bg-[#151C2C]/40" : "border-gray-200 bg-white shadow"
              }`}
            >
              <div className={`inline-flex p-3 rounded-xl ${isDarkMode ? "bg-white/5" : "bg-gray-50"}`}>
                <Icon className="w-6 h-6 text-[#2979FF]" />
              </div>
              <h3 className="mt-4 text-lg">{title}</h3>
              <p className={`mt-2 text-sm ${isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}`}>{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="why-it-matters" data-reveal>
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <h2 className="text-3xl">Pourquoi c’est important</h2>
          <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
            Le contrôle qualité ne concerne pas seulement la détection : il s’agit aussi de décisions rapides, de
            critères stables et de preuves auditables.
          </p>
        </div>

        <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {impactCards.map(({ title, body, Icon }, idx) => (
            <div
              key={title}
              data-reveal
              style={{ transitionDelay: `${idx * 80}ms` }}
              className={`h-full rounded-2xl p-6 border backdrop-blur-sm ${
                isDarkMode ? "border-white/10 bg-[#151C2C]/40" : "border-gray-200 bg-white shadow"
              }`}
            >
              <div className={`inline-flex p-3 rounded-xl ${isDarkMode ? "bg-white/5" : "bg-gray-50"}`}>
                <Icon className="w-6 h-6 text-[#00E676]" />
              </div>
              <h3 className="mt-4 text-lg">{title}</h3>
              <p className={`mt-2 text-sm ${isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}`}>{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="solution" data-reveal>
        <div
          className={`rounded-3xl p-10 md:p-12 border backdrop-blur-sm ${
            isDarkMode ? "border-white/10 bg-[#151C2C]/45" : "border-gray-200 bg-white shadow"
          }`}
        >
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-10 items-start">
            <div className="space-y-3">
              <h2 className="text-3xl">Notre solution</h2>
              <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
                Détection de défauts de surface par IA pour les écrous métalliques, conçue pour des décisions
                d’inspection rapides et la traçabilité.
              </p>
            </div>

            <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className={`h-full rounded-2xl p-6 border ${isDarkMode ? "border-white/10 bg-white/5" : "border-gray-200 bg-gray-50"}`}>
                <h3 className="text-lg mb-2">Localisation au pixel près</h3>
                <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
                  Le système produit un masque de segmentation pour voir précisément où se trouve le défaut sur la
                  surface de l’écrou.
                </p>
              </div>
              <div className={`h-full rounded-2xl p-6 border ${isDarkMode ? "border-white/10 bg-white/5" : "border-gray-200 bg-gray-50"}`}>
                <h3 className="text-lg mb-2">Métrique de couverture objective</h3>
                <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
                  La couverture de défaut est calculée uniquement sur les pixels de l’écrou, afin que le fond n’atténue
                  pas le résultat.
                </p>
              </div>
              <div className={`h-full rounded-2xl p-6 border ${isDarkMode ? "border-white/10 bg-white/5" : "border-gray-200 bg-gray-50"}`}>
                <h3 className="text-lg mb-2">Répartition par classe</h3>
                <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
                  La sortie inclut la couverture par classe : rayure, couleur, inversion et plié.
                </p>
              </div>
              <div className={`h-full rounded-2xl p-6 border ${isDarkMode ? "border-white/10 bg-white/5" : "border-gray-200 bg-gray-50"}`}>
                <h3 className="text-lg mb-2">Rapport exportable</h3>
                <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
                  Générez un rapport PDF avec visuels et statistiques pour les workflows qualité et la traçabilité.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="how-it-works" data-reveal>
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <h2 className="text-3xl">Comment ça marche</h2>
          <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
            Un pipeline simple pensé pour la clarté industrielle : le système montre ce qu’il détecte, où il le détecte
            et comment la couverture finale est calculée.
          </p>
        </div>

        <div className="mt-10 grid grid-cols-1 lg:grid-cols-2 gap-6">
          {howItWorksSteps.map(({ title, body, Icon }, idx) => (
            <div
              key={title}
              data-reveal
              style={{ transitionDelay: `${idx * 60}ms` }}
              className={`h-full rounded-2xl p-6 border backdrop-blur-sm flex gap-4 ${
                isDarkMode ? "border-white/10 bg-[#151C2C]/40" : "border-gray-200 bg-white shadow"
              }`}
            >
              <div className={`mt-1 inline-flex p-3 rounded-xl ${isDarkMode ? "bg-white/5" : "bg-gray-50"}`}>
                <Icon className="w-6 h-6 text-[#2979FF]" />
              </div>
              <div className="space-y-1">
                <p className={`text-xs uppercase tracking-wide ${isDarkMode ? "text-[#E3E9F1]/50" : "text-gray-500"}`}>
                  Étape {idx + 1}
                </p>
                <h3 className="text-lg">{title}</h3>
                <p className={`text-sm ${isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}`}>{body}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section id="defects" data-reveal>
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <h2 className="text-3xl">Défauts détectés</h2>
          <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
            Le modèle prédit une classe de défaut pour chaque pixel afin de permettre la classification et la
            localisation.
          </p>
        </div>

        <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {defectCards.map(({ title, body, accent }, idx) => (
            <div
              key={title}
              data-reveal
              style={{ transitionDelay: `${idx * 80}ms` }}
              className={`h-full rounded-2xl p-6 border backdrop-blur-sm ${
                isDarkMode ? "border-white/10 bg-[#151C2C]/40" : "border-gray-200 bg-white shadow"
              }`}
            >
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: accent }} />
                <h3 className="text-lg">{title}</h3>
              </div>
              <p className={`mt-3 text-sm ${isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}`}>{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="outputs" data-reveal>
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <h2 className="text-3xl">Sorties</h2>
          <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
            L’objectif est de produire des sorties d’inspection utilisables sur le terrain : visuelles, mesurables et
            exportables.
          </p>
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { title: "Visualisation en superposition", body: "Une superposition colorée met en évidence les zones défectueuses sur l’écrou.", Icon: Eye },
            { title: "% de défaut sur la surface de l’écrou", body: "Couverture calculée uniquement sur les pixels de l’écrou (fond exclu).", Icon: CheckCircle },
            { title: "Répartition par classe", body: "Couverture par classe : rayure, couleur, inversion, plié.", Icon: AlertCircle },
            { title: "Rapport PDF", body: "Téléchargez un rapport PDF pour la traçabilité et les workflows qualité.", Icon: FileText },
          ].map(({ title, body, Icon }, idx) => (
            <div
              key={title}
              data-reveal
              style={{ transitionDelay: `${idx * 80}ms` }}
              className={`h-full rounded-2xl p-6 border backdrop-blur-sm ${
                isDarkMode ? "border-white/10 bg-[#151C2C]/40" : "border-gray-200 bg-white shadow"
              }`}
            >
              <div className={`inline-flex p-3 rounded-xl ${isDarkMode ? "bg-white/5" : "bg-gray-50"}`}>
                <Icon className="w-6 h-6 text-[#00E676]" />
              </div>
              <h3 className="mt-4 text-lg">{title}</h3>
              <p className={`mt-2 text-sm ${isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}`}>{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="benefits" data-reveal>
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <h2 className="text-3xl">Avantages</h2>
          <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
            Des résultats concrets pour les équipes de production : décisions plus rapides, critères cohérents et
            reporting amélioré.
          </p>
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { title: "Rapidité", body: "Soutient les workflows à forte cadence avec un retour d’inspection rapide.", Icon: Zap },
            { title: "Cohérence", body: "Applique les mêmes critères sur les images et les équipes.", Icon: CheckCircle },
            { title: "Traçabilité", body: "Conserve des preuves avec superpositions, métriques et rapports.", Icon: FileText },
            { title: "Rapports", body: "Rapport PDF exportable pour la qualité et la documentation.", Icon: Download },
          ].map(({ title, body, Icon }, idx) => (
            <div
              key={title}
              data-reveal
              style={{ transitionDelay: `${idx * 80}ms` }}
              className={`h-full rounded-2xl p-6 border backdrop-blur-sm ${
                isDarkMode ? "border-white/10 bg-[#151C2C]/40" : "border-gray-200 bg-white shadow"
              }`}
            >
              <div className={`inline-flex p-3 rounded-xl ${isDarkMode ? "bg-white/5" : "bg-gray-50"}`}>
                <Icon className="w-6 h-6 text-[#2979FF]" />
              </div>
              <h3 className="mt-4 text-lg">{title}</h3>
              <p className={`mt-2 text-sm ${isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}`}>{body}</p>
            </div>
          ))}
        </div>

        <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
          <button
            onClick={onUploadClick}
            className="px-8 py-4 bg-gradient-to-r from-[#2979FF] to-[#00E676] rounded-xl text-white shadow-lg hover:shadow-2xl transition-all duration-300 flex items-center gap-2"
          >
            <Upload className="w-5 h-5" />
            Démarrer l’inspection
          </button>
          <button
            onClick={() => scrollToId("faq")}
            className={`px-8 py-4 rounded-xl border transition-all duration-300 hover:shadow-lg ${
              isDarkMode ? "border-white/20 text-[#E3E9F1] hover:bg-white/5" : "border-gray-300 text-gray-700 hover:bg-white"
            }`}
          >
            FAQ
          </button>
        </div>
      </section>

      <section id="faq" data-reveal>
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <h2 className="text-3xl">FAQ</h2>
          <p className={isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}>
            Des réponses claires pour l’usage en production et les workflows qualité.
          </p>
        </div>

        <div className="mt-8 max-w-4xl mx-auto space-y-4">
          {[
            {
              q: "Comment le pourcentage de défaut est-il calculé ?",
              a: "La couverture de défaut est calculée uniquement sur la surface de l’écrou. Un masque de l’écrou exclut le fond, afin que le pourcentage ne soit pas dilué par les pixels noirs.",
            },
            {
              q: "Quels défauts sont reportés ?",
              a: "La couverture par classe est reportée pour rayure, couleur, inversion et plié. Le modèle retourne aussi une superposition pour la confirmation visuelle.",
            },
            {
              q: "Que reçois-je après l’inférence ?",
              a: "Vous obtenez la visualisation en superposition, la couverture de défaut sur la surface de l’écrou, la répartition par classe et un rapport PDF exportable pour la traçabilité.",
            },
            {
              q: "Puis-je utiliser cela dans des workflows à forte cadence ?",
              a: "L’interface est conçue pour un cycle rapide import → analyse → décision. La sortie est structurée pour une revue rapide et des critères reproductibles.",
            },
            {
              q: "L’application conserve-t-elle mes images ?",
              a: "Le workflow actuel traite l’image importée pour l’inférence et renvoie les résultats. La persistance n’est pas nécessaire pour un usage basique, sauf si vous implémentez un stockage pour la traçabilité.",
            },
          ].map((item) => (
            <details
              key={item.q}
              className={`faq-item rounded-2xl border backdrop-blur-sm overflow-hidden ${
                isDarkMode ? "border-white/10 bg-[#151C2C]/40" : "border-gray-200 bg-white shadow"
              }`}
            >
              <summary className="flex items-center justify-between gap-4 p-5 cursor-pointer">
                <span className="text-lg">{item.q}</span>
                <ChevronDown className="faq-chevron w-5 h-5 opacity-70" />
              </summary>
              <div className={`px-5 pb-5 text-sm ${isDarkMode ? "text-[#E3E9F1]/70" : "text-gray-600"}`}>{item.a}</div>
            </details>
          ))}
        </div>
      </section>
    </div>
  );
}
