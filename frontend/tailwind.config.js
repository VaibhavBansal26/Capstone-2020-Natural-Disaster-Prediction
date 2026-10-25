/** Tailwind build config. Rebuild the stylesheet with frontend/build.sh after editing templates. */
module.exports = {
  // Paths are relative to the repository root, where the build script runs.
  content: ["./src/ndp/templates/**/*.html", "./src/ndp/static/js/*.js"],
  theme: {
    extend: {
      colors: {
        brand: { 50: "#eff6ff", 600: "#2563eb", 700: "#1d4ed8", 900: "#1e3a8a" },
      },
    },
  },
};
