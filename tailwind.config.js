/** Tailwind build config. Rebuild the stylesheet with scripts/build_css.sh after editing templates. */
module.exports = {
  content: ["./src/ndp/templates/**/*.html", "./src/ndp/static/*.js"],
  theme: {
    extend: {
      colors: {
        brand: { 50: "#eff6ff", 600: "#2563eb", 700: "#1d4ed8", 900: "#1e3a8a" },
      },
    },
  },
};
