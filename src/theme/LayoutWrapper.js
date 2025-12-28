import React from 'react';
import Layout from '@theme/Layout';

function LayoutWrapper({ children, ...props }) {
  return (
    <Layout {...props}>
      {children}
    </Layout>
  );
}

export default LayoutWrapper;