/** Tailwind config — matches the design tokens declared inline in templates/index.html */
module.exports = {
  darkMode: "class",
  content: ["./templates/**/*.html"],
  theme: {
    extend: {
      colors: {
        primary: "#000000",
        "on-primary": "#ffffff",
        secondary: "#585f6c",
        tertiary: "#c99a2e",
        success: "#16a34a",

        surface: "#ffffff",
        "surface-bright": "#f9f9f9",
        "surface-container-lowest": "#ffffff",
        "surface-container-low": "#f3f3f3",
        "surface-container": "#eeeeee",
        "surface-container-high": "#e8e8e8",
        "surface-variant": "#e2e2e2",

        outline: "#747878",
        "outline-variant": "#c4c7c7",
        "on-surface": "#1a1c1c",
        "on-surface-variant": "#444748",
      },
      borderRadius: { DEFAULT: "0.125rem", lg: "0.25rem" },
      spacing: {
        gutter: "24px",
        "margin-desktop": "32px",
        "margin-mobile": "16px",
        "container-max-width": "1440px",
      },
      fontFamily: {
        "headline-xl": ["Inter"],
        "headline-lg": ["Inter"],
        "headline-md": ["Inter"],
        "body-lg": ["Inter"],
        "body-md": ["Inter"],
        "body-sm": ["Inter"],
        "label-caps": ["Inter"],
        "numeric-tabular": ["Inter"],
      },
      fontSize: {
        "headline-xl": ["36px", { lineHeight: "44px", letterSpacing: "-0.02em", fontWeight: "700" }],
        "headline-lg": ["24px", { lineHeight: "32px", letterSpacing: "-0.01em", fontWeight: "600" }],
        "headline-md": ["20px", { lineHeight: "28px", fontWeight: "600" }],
        "body-lg": ["16px", { lineHeight: "24px", fontWeight: "400" }],
        "body-md": ["14px", { lineHeight: "20px", fontWeight: "400" }],
        "body-sm": ["13px", { lineHeight: "18px", fontWeight: "400" }],
        "label-caps": ["11px", { lineHeight: "16px", letterSpacing: "0.05em", fontWeight: "700" }],
        "numeric-tabular": ["14px", { lineHeight: "20px", fontWeight: "500" }],
      },
    },
  },
  plugins: [],
};
