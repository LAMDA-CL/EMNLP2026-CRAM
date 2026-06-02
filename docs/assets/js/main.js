(function () {
  const sidebar = document.querySelector(".sidebar");
  const backdrop = document.querySelector(".sidebar-backdrop");
  const menuToggle = document.querySelector(".menu-toggle");
  const navLinks = document.querySelectorAll(".nav-link[href^='#']");

  /* —— Collapsible sidebar sections —— */
  document.querySelectorAll(".nav-section-toggle").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const section = btn.closest(".nav-section");
      if (!section) return;
      const willOpen = !section.classList.contains("open");
      section.classList.toggle("open", willOpen);
      btn.setAttribute("aria-expanded", willOpen ? "true" : "false");
    });
  });

  /* —— Smooth scroll + hash update; all nav targets must exist —— */
  function getAnchor(id) {
    if (!id) return null;
    try {
      return document.querySelector(`[id="${CSS.escape(id)}"]`);
    } catch {
      return document.getElementById(id);
    }
  }

  function scrollToId(id) {
    const el = getAnchor(id);
    if (!el) {
      console.warn("[PRISM docs] Missing section:", id);
      return false;
    }
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    history.replaceState(null, "", `#${id}`);
    return true;
  }

  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      const href = link.getAttribute("href");
      if (!href || !href.startsWith("#")) return;
      const id = href.slice(1);
      const target = getAnchor(id);
      if (!target) return;
      e.preventDefault();
      scrollToId(id);
      setActiveLink(id);
      expandSectionForLink(link);
      closeMobileSidebar();
    });
  });

  function expandSectionForLink(link) {
    const section = link.closest(".nav-section");
    if (section) {
      section.classList.add("open");
      const btn = section.querySelector(".nav-section-toggle");
      if (btn) btn.setAttribute("aria-expanded", "true");
    }
  }

  function setActiveLink(activeId) {
    navLinks.forEach((a) => {
      const id = (a.getAttribute("href") || "").slice(1);
      a.classList.toggle("active", id === activeId);
    });
  }

  /* —— Scroll spy: pick nearest visible heading —— */
  const spyTargets = [...navLinks]
    .map((a) => {
      const id = (a.getAttribute("href") || "").slice(1);
      const el = getAnchor(id);
      return el ? { id, el } : null;
    })
    .filter(Boolean);

  function updateSpy() {
    const y = window.scrollY + 100;
    let current = spyTargets[0]?.id;
    for (const { id, el } of spyTargets) {
      if (el.offsetTop <= y) current = id;
    }
    if (current) {
      setActiveLink(current);
      const active = document.querySelector(`.nav-link[href="#${current}"]`);
      if (active) expandSectionForLink(active);
    }
  }

  window.addEventListener("scroll", updateSpy, { passive: true });
  updateSpy();

  const HASH_ALIASES = { config: "configuration-reference" };

  function resolveHashId(id) {
    return HASH_ALIASES[id] || id;
  }

  /* —— Initial hash on load —— */
  if (location.hash) {
    const id = resolveHashId(location.hash.slice(1));
    setTimeout(() => {
      scrollToId(id);
      setActiveLink(id);
      const link = document.querySelector(`.nav-link[href="#${id}"]`);
      if (link) expandSectionForLink(link);
    }, 80);
  }

  /* —— Copy code buttons —— */
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const pre = btn.closest(".code-block")?.querySelector("pre");
      if (!pre) return;
      try {
        const code = pre.querySelector("code") || pre;
        await navigator.clipboard.writeText(code.innerText);
        const prev = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => {
          btn.textContent = prev;
        }, 1400);
      } catch {
        btn.textContent = "Failed";
      }
    });
  });

  /* —— Mobile sidebar —— */
  function closeMobileSidebar() {
    sidebar?.classList.remove("open");
    backdrop?.classList.remove("visible");
  }

  function openMobileSidebar() {
    sidebar?.classList.add("open");
    backdrop?.classList.add("visible");
  }

  menuToggle?.addEventListener("click", () => {
    if (sidebar?.classList.contains("open")) closeMobileSidebar();
    else openMobileSidebar();
  });

  backdrop?.addEventListener("click", closeMobileSidebar);

  /* Open first section by default; others closed unless they contain active link */
  document.querySelectorAll(".nav-section").forEach((sec, i) => {
    const btn = sec.querySelector(".nav-section-toggle");
    const hasActive = sec.querySelector(".nav-link.active");
    const open = hasActive || i === 0;
    sec.classList.toggle("open", open);
    if (btn) btn.setAttribute("aria-expanded", open ? "true" : "false");
  });
})();
