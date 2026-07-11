/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#0A0A0F',
          900: '#14141F',
          800: '#1E1E2F',
        }
      }
    },
  },
  plugins: [],
}
