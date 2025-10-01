# Repository Guidelines

## Project Structure & Module Organization
Presenton pairs a FastAPI backend with a Next.js front end. Core API modules and pytest suites live in `servers/fastapi` (fixtures under `servers/fastapi/tests`), while routes, shared components, and Redux slices sit in `servers/nextjs` (`app/`, `components/`, `store/`). Cypress specs stay in `servers/nextjs/cypress`, and root Docker assets plus `start.js` orchestrate full-stack runs; update them whenever dependencies or env variables move.

## Build, Test, and Development Commands
- `npm install` / `npm run dev -- --port 3000` from `servers/nextjs` boots the web client.
- `npm run build` and `npm run start` serve the production bundle; rerun after dependency updates.
- `npm run lint` executes the Next.js ESLint profile; clear warnings before PRs.
- `uv sync` in `servers/fastapi` installs Python deps, and `uv run python server.py --port 8000 --reload true` starts the API.
- `uv run pytest` runs backend tests; use `-k name` to scope modules.

## Coding Style & Naming Conventions
Use TypeScript with 2-space indentation, PascalCase components, and camelCase utilities; hooks start with `use`. Group Tailwind classes as layout -> spacing -> color for readability. Python code follows PEP 8 with 4-space indents; keep async helpers suffixed `_async` in `services/`. Validate request and response shapes with Zod (frontend) or Pydantic (backend) before persisting changes.

## Testing Guidelines
Add or refresh pytest coverage for any backend path you touch; name tests `test_feature_behavior`. UI work needs a Cypress spec or component-level assertion via Testing Library. Run `npm run lint` and `uv run pytest` before pushing, attach Cypress videos on failures, and flag intentional coverage gaps in the PR.

## Commit & Pull Request Guidelines
Follow Conventional Commit prefixes (`feat:`, `fix(scope):`, `perf:`) as in history. Keep commits focused and use imperative summaries. PRs must outline the problem, list changes, show test evidence (`uv run pytest`, `npm run lint`, etc.), attach UI media when relevant, and link issues or discussions.

## Configuration & Security Notes
`start.js` reads secrets from environment variables (`LLM`, `OPENAI_API_KEY`, `CAN_CHANGE_KEYS`, etc.); never embed credentials in code. Document new variables in the README and Compose files. Store shared templates or assets under `presentation-templates/` or `public/`, and add licensing notes to `NOTICE` when required.
