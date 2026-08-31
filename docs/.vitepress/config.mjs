import { defineConfig } from "vitepress";

export default defineConfig({
  base: "/Scene_Classification/",
  title: "Scene Classification",
  description: "MobileNetV2 vs. DINOv2 Benchmark",

  // Disables dark mode and removes the theme switch button
  appearance: false,

  themeConfig: {
    nav: [
      { text: "Home", link: "/" },
      { text: "Project Report", link: "/project_report" },
    ],
    sidebar: [
      {
        text: "Project Overview",
        items: [
          { text: "Overview & Demo", link: "/" },
          { text: "Detailed Project Report", link: "/project_report" },
        ],
      },
    ],
    socialLinks: [
      {
        icon: "github",
        link: "https://github.com/Bhargav-Das-12/Scene_Classification",
      },
    ],
  },
});
