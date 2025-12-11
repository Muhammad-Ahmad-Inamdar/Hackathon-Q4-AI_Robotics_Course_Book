// @ts-check

import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline:
    'An AI-native textbook on Physical AI & Humanoid Robotics with ROS 2, Gazebo, Unity, NVIDIA Isaac, and VLA integration',
  favicon: '/img/1.png',

  url: 'https://Muhammad-Ahmad-Inamdar.github.io',

  baseUrl: '/Hackathon-Q4-AI_Robotics_Course_Book/',

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
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
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

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Modules',
          items: [
            { label: 'Module 1: ROS 2', to: '/docs/module-1-ros2/intro' },
            { label: 'Module 2: Digital Twin', to: '/docs/module-2-digital-twin/intro' },
            { label: 'Module 3: AI-Robot Brain', to: '/docs/module-3-ai-brain/intro' },
            { label: 'Module 4: VLA', to: '/docs/module-4-vla/intro' },
          ],
        },
        {
          title: 'Community',
          items: [
            { label: 'Stack Overflow', href: 'https://stackoverflow.com/tagged/docusaurus' },
            { label: 'Discord', href: 'https://discordapp.com/invite/docusaurus' },
            { label: 'Twitter', href: 'https://twitter.com/docusaurus' },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/Muhammad-Ahmad-Inamdar/Hackathon-Q4-AI_Robotics_Course_Book',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
};

export default config;
