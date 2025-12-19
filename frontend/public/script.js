(() => {
  const REVEAL_SELECTOR = "[data-reveal]";
  const PROCESSED_ATTR = "data-reveal-processed";

  const revealObserver =
    "IntersectionObserver" in window
      ? new IntersectionObserver(
          (entries) => {
            for (const entry of entries) {
              if (!entry.isIntersecting) continue;
              entry.target.classList.add("reveal--visible");
              revealObserver.unobserve(entry.target);
            }
          },
          { threshold: 0.15, rootMargin: "0px 0px -10% 0px" },
        )
      : null;

  function prepare(el) {
    if (!(el instanceof HTMLElement)) return;
    if (el.hasAttribute(PROCESSED_ATTR)) return;
    el.setAttribute(PROCESSED_ATTR, "true");
    el.classList.add("reveal");

    if (!revealObserver) {
      el.classList.add("reveal--visible");
      return;
    }
    revealObserver.observe(el);
  }

  function scan(root) {
    if (!(root instanceof Element)) return;
    const candidates = root.matches(REVEAL_SELECTOR)
      ? [root]
      : Array.from(root.querySelectorAll(REVEAL_SELECTOR));
    for (const el of candidates) prepare(el);
  }

  function init() {
    scan(document.documentElement);

    const mo = new MutationObserver((mutations) => {
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node instanceof Element) scan(node);
        }
      }
    });
    mo.observe(document.documentElement, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
