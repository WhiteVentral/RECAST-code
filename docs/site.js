for (const tabset of document.querySelectorAll("[data-tabset]")) {
  const controls = tabset.querySelector("[data-tab-controls]");
  const tabs = [...controls.querySelectorAll('[role="tab"]')];
  const panels = tabs.map((tab) => document.getElementById(tab.getAttribute("aria-controls")));

  function selectTab(index, focusTab = false) {
    tabs.forEach((tab, position) => {
      const active = index === position;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      panels[position].hidden = !active;
      panels[position].classList.toggle("is-current", active);
    });
    if (focusTab) tabs[index].focus();
  }

  tabs.forEach((tab, index) => {
    panels[index].setAttribute("role", "tabpanel");
    panels[index].setAttribute("aria-labelledby", tab.id);
    tab.addEventListener("click", () => selectTab(index));
    tab.addEventListener("keydown", (event) => {
      const next = {
        ArrowRight: (index + 1) % tabs.length,
        ArrowLeft: (index + tabs.length - 1) % tabs.length,
        Home: 0,
        End: tabs.length - 1,
      }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      selectTab(next, true);
    });
  });

  selectTab(0);
  controls.hidden = false;
  tabset.classList.add("enhanced");
}

const motionAllowed = !matchMedia("(prefers-reduced-motion: reduce)").matches;
const canObserve = "IntersectionObserver" in window;
const curve = document.querySelector("[data-evolution-animation]");

if (curve) {
  const image = curve.querySelector("[data-curve-image]");
  const playButton = curve.querySelector("[data-play-curve]");
  const animationUrl = curve.querySelector("[data-curve-gif]").href;
  let playback = 0;
  let autoplay;

  function playCurve() {
    autoplay?.disconnect();
    const url = new URL(animationUrl);
    if (playback > 0) url.searchParams.set("replay", playback);
    image.src = url.href;
    playback += 1;
    playButton.textContent = "Replay animation";
  }

  playButton.hidden = false;
  playButton.addEventListener("click", playCurve);
  if (canObserve && motionAllowed) {
    autoplay = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) playCurve();
    }, { threshold: 0.4 });
    autoplay.observe(image);
  }
}

if (canObserve && motionAllowed) {
  const reveal = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("in-view");
      reveal.unobserve(entry.target);
    }
  }, { threshold: 0.08 });

  document.querySelectorAll("[data-reveal]").forEach((element) => reveal.observe(element));
}
