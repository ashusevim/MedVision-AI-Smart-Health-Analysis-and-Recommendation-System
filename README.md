# MedVision - AI Smart Health Analysis and Recommendation System

MedVision is a modern AI-health themed web application built with Next.js, React, TypeScript, Tailwind CSS, and shadcn-style UI components.

It provides an interactive product-style experience for health analysis concepts such as symptom insights, risk awareness, monitoring visuals, and recommendation journeys.

## Overview

This project is a frontend-focused Next.js application with:

- App Router architecture
- Animated UI experience using Framer Motion
- Charting and data visualizations
- Component-driven design system
- Production-ready standalone build output

## Features

- Interactive health-focused landing interface
- Animated sections and transitions
- Metrics and chart visualizations
- Modular UI components under src/components/ui
- Responsive layout for desktop and mobile
- SEO metadata configured in app layout

## Tech Stack

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS 4
- Framer Motion
- Recharts
- Radix UI primitives
- Prisma (scripts and dependencies are present for database workflows)

## Project Structure

```text
.
|- src
|  |- app
|  |  |- layout.tsx
|  |  |- page.tsx
|  |  |- globals.css
|  |  \- api
|  |- components
|  |  \- ui
|  \- medvision.py
|- public
|- package.json
|- next.config.ts
\- README.md
```

## Getting Started

Prerequisites:

- Node.js 20 or newer
- npm
- Optional: Bun (only needed if you want to use the current production start script as-is)

1. Install dependencies

```bash
npm install
```

2. Start development server

```bash
npm run dev
```

3. Open the app

http://localhost:3000

## Available Scripts

- npm run dev  
	Starts Next.js development server on port 3000 and writes logs to dev.log

- npm run build  
	Builds production assets and prepares standalone output

- npm run start  
	Starts standalone production server using Bun and writes logs to server.log

- npm run lint  
	Runs ESLint checks

- npm run db:generate
- npm run db:push
- npm run db:migrate
- npm run db:reset

## Notes

- This repository currently focuses on frontend experience and presentation.
- Some imported UI components may need to be present in src/components/ui depending on your branch state.
- For production, ensure environment variables and backend integrations are configured if server-side features are added.

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push your branch
5. Open a pull request

## License

This project is licensed under Apache License 2.0. See LICENSE for details.

## Contact

- Project Owner: Ra'uf Fauzan Rambe
- Email: ramberauffauzan@gmail.com
- GitHub: https://github.com/RaufFauzanRambe
