import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--textbook button--lg"
            to="/docs/module-1-ros2/intro">
            Start Reading the Textbook
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="An AI-native textbook on Physical AI & Humanoid Robotics with ROS 2, Gazebo, Unity, NVIDIA Isaac, and VLA integration">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <img src={useBaseUrl('/img/ROS.jpg')} alt="ROS 2 Foundation" className={styles.featureImage} />
                  <h3>ROS 2 Foundation</h3>
                  <p>Learn the nervous system of robotics with ROS 2 communication patterns, nodes, and advanced integration.</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <img src={useBaseUrl('/img/Digital-Twins.jpg')} alt="Digital Twins" className={styles.featureImage} />
                  <h3>Digital Twins</h3>
                  <p>Master simulation environments with Gazebo and Unity integration for virtual body representation.</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <img src={useBaseUrl("/img/AI-Brain.jpg")} alt="AI-Robot Brains" className={styles.featureImage} />
                  <h3>AI-Robot Brains</h3>
                  <p>Implement cognitive systems using NVIDIA Isaac tools for perception and navigation.</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center padding-horiz--md">
                  <img src={useBaseUrl("/img/Vision-Language-Action.jpg")} alt="Vision-Language-Action" className={styles.featureImage} />
                  <h3>Vision-Language-Action</h3>
                  <p>Integrate multimodal systems with VLA for natural human-robot interaction and embodied AI.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.features} style={{padding: '2rem 0', backgroundColor: '#f6f6f6'}}>
          <div className="container">
            <div className="row">
              <div className="col col--12 text--center">
                <h2>Integrated Physical AI & Humanoid System</h2>
                <p>Comprehensive curriculum covering Vision-Language-Action systems for natural human-robot interaction</p>
              </div>
            </div>
            <div className="row padding-vert--md">
              <div className="col col--3">
                <div className="text--center">
                  <h4>Module 1</h4>
                  <p>ROS 2 - The Nervous System</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center">
                  <h4>Module 2</h4>
                  <p>Digital Twin - The Virtual Body</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center">
                  <h4>Module 3</h4>
                  <p>AI-Robot Brain - The Cognitive System</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center">
                  <h4>Module 4</h4>
                  <p>Vision-Language-Action - The Natural Interface</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}