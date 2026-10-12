/* Shared page chrome: masthead, tab bar and footer, the same on every page.
   Each page loads this file right after <body data-page="...">, so the header is in place before the
   page paints. To rename, reorder, add or remove a tab, edit PAGES (and add or delete the HTML file). */
(function () {
  "use strict";
  var PAGES = [
    ["overview", "index.html", "Overview"],
    ["sectors", "sectors.html", "Sectors &amp; firms"],
    ["rates", "rates.html", "Rates &amp; economy"],
    ["valuation", "valuation.html", "Valuation &amp; factors"],
    ["risk", "risk.html", "Risk models"],
  ];
  var ABOUT = ["about", "about.html", "About the data"];
  window.MM_PAGES = PAGES.concat([ABOUT]);

  var page = document.body.getAttribute("data-page") || "overview";

  // old single-page links (index.html#rates and so on) go to the new page
  if (page === "overview" && location.hash) {
    var legacy = { "#risk": "risk.html", "#sectors": "sectors.html", "#firms": "sectors.html#firms", "#rates": "rates.html",
      "#economy": "rates.html#economy", "#valuation": "valuation.html", "#factors": "valuation.html#factors",
      "#about": "about.html" };
    if (legacy[location.hash]) { location.replace(legacy[location.hash]); return; }
  }

  var tabs = PAGES.map(function (p) {
    var cur = p[0] === page;
    return '<a href="' + p[1] + '"' + (cur ? ' class="active" aria-current="page"' : "") + ">" + p[2] + "</a>";
  }).join("");

  var top =
    '<a class="skip-link" href="#main">Skip to content</a>' +
    '<header class="masthead">' +
      '<div class="wrap mast-top">' +
        "<div>" +
          '<div class="eyebrow" id="site-eyebrow">U.S. markets · end-of-day data</div>' +
          '<h1 class="site-title"><a href="index.html" id="site-title">Market Monitor</a></h1>' +
          '<p class="tagline" id="site-tagline">The U.S. stock market at a glance, with notes on how to read every number. Updated after each trading day.</p>' +
        "</div>" +
        '<div class="mast-meta">' +
          '<div id="asof" class="asof" aria-live="polite">Loading data…</div>' +
          '<a class="mast-link' + (page === "about" ? " active" : "") + '" href="about.html"' + (page === "about" ? ' aria-current="page"' : "") + ">About the data</a>" +
          '<button id="theme-toggle" class="icon-btn" type="button" aria-label="Switch between light and dark theme">' +
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>' +
          "</button>" +
        "</div>" +
      "</div>" +
      '<div class="strip-wrap"><div class="wrap"><ul class="strip" id="key-strip" aria-label="Key market numbers"></ul></div></div>' +
    "</header>" +
    '<nav class="secnav" aria-label="Pages"><div class="wrap">' + tabs + "</div></nav>" +
    '<div class="wrap"><div id="notice" class="notice" hidden></div></div>';

  var me = document.currentScript;
  if (me) me.insertAdjacentHTML("beforebegin", top);
  else document.body.insertAdjacentHTML("afterbegin", top);

  function footer() {
    if (document.querySelector("footer.site-foot")) return;
    var html =
      '<footer class="site-foot"><div class="wrap">' +
        '<p id="foot-line">Market Monitor · for educational use only, not investment advice.</p>' +
        '<p class="foot-links"><a href="about.html">About the data</a> · data updated after each U.S. trading day</p>' +
      "</div></footer>";
    var main = document.getElementById("main");
    if (main) main.insertAdjacentHTML("afterend", html);
    else document.body.insertAdjacentHTML("beforeend", html);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", footer);
  else footer();
})();
