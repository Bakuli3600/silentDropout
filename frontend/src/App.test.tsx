import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders application title', () => {
  render(<App />);
  const titleElement = screen.getByText(/Silent Dropout Detection Dashboard/i);
  expect(titleElement).toBeInTheDocument();
});
