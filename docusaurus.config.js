// @ts-check

import { themes as prismThemes } from 'prism-react-renderer';

const isProd = process.env.NODE_ENV === 'production';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline:
    'An AI-native textbook on Physical AI & Humanoid Robotics with ROS 2, Gazebo, Unity, NVIDIA Isaac, and VLA integration',

  favicon: '/img/1.png',

  // GitHub Pages config
  url: 'https://Muhammad-Ahmad-Inamdar.github.io',

  // IMPORTANT: baseUrl conditional
  baseUrl: isProd
    ? '/Hackathon-Q4-AI_Robotics_Course_Book/'
    : '/',

  organizationName: 'Muhammad-Ahmad-Inamdar',
  projectName: 'Hackathon-Q4-AI_Robotics_Course_Book',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      ({
        docs: {
          sidebarPath: './sidebars.js',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/Muhammad-Ahmad-Inamdar/Hackathon-Q4-AI_Robotics_Course_Book',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },

    footer: {
      style: 'dark',
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
};

export default config;
