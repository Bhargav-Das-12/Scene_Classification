import { defineConfig } from "vitepress";

export default defineConfig({
  base: "/Scene_Classification/",
  title: "Scene Classification",
  description: "MobileNetV2 vs. DINOv2 Benchmark",

  appearance: false,
  markdown: {
    math: true,
  },
  head: [
    [
      "style",
      {},
      `
      /* Expand the icon button to make room for text */
      .VPSocialLink { 
        width: auto !important; 
        padding-right: 4px;
      }
      /* Inject the repository text */
      .VPSocialLink::after { 
        content: 'Bhargav-Das-12/Scene_Classification'; 
        margin-left: 8px; 
        font-size: 14px; 
        font-weight: 500;
        color: var(--vp-c-text-2); /* Matches VitePress's native grey text */
        transition: color 0.25s; /* Smooth hover transition */
      }
      /* Make it turn slightly darker on hover to match the icon */
      .VPSocialLink:hover::after {
        color: var(--vp-c-text-1);
      }
    `,
    ],
  ],

  themeConfig: {
    nav: [
      { text: "Project Report", link: "/" },
      { text: "Setup & Installation Guide", link: "/setup_and_installation" },
    ],
    sidebar: [
      {
        text: "Project Navigation",
        items: [
          { text: "Project Report", link: "/" },
          {
            text: "Setup & Installation Guide",
            link: "/setup_and_installation",
          },
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
