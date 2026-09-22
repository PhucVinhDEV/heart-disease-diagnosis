# AGENTS.md

## Scope

This file applies to the `heart-disease-diagnosis/` subproject inside a larger monorepo.

Before making architecture assumptions, read the root docs:

- `../docs/AI_Cardiovascular_PRD.md`
- `../docs/API_CONTRACT.md`
- `../docs/FSD_FRONTEND.md`
- `../docs/PLATFORM_PIPELINE_AND_PROGRESS.md`
- `../README.md`

Those documents are the source of truth for product scope and system boundaries.

## Monorepo Reality

This repository contains three main product projects plus shared docs:

- `aorticprecision-next/`: frontend client, Next.js, TypeScript, FSD-oriented
- `Cardiovascular_Be/`: public backend API, Spring Boot
- `Cardiovascular_FastAPI/`: internal Python services for model inference and OCR
- `heart-disease-diagnosis/`: standalone ML research and demo application used as a reference asset, prototype, and source of model/pipeline ideas

Do not treat `heart-disease-diagnosis/` as the whole platform.
Do not describe the monorepo as Streamlit-only.
Do not assume this subproject is the production backend contract.

## Role of This Subproject

`heart-disease-diagnosis/` is best understood as a research/prototype project that contains:

- a working Streamlit demo UI in `app/`
- local prediction pipeline logic in `src/`
- model artifacts in `models/`
- experimentation and tuning scripts in `scripts/`
- datasets and processed variants in `data/`
- project-specific docs in `docs/`

Use this subproject for:

- experimenting with model behavior
- validating preprocessing and feature-engineering ideas
- demoing local inference UX
- inspecting saved model compatibility issues
- borrowing logic that may later be ported into the platform services

Do not assume changes here automatically update:

- Spring Boot API behavior
- Next.js frontend integration
- FastAPI service contracts in the platform

## System Boundary Rules

According to root docs, the target platform architecture is:

- public client traffic goes to Spring Boot
- Spring Boot orchestrates business logic, auth, CRUD, audit, and public API
- FastAPI is internal-only for ML and OCR
- frontend consumes Spring Boot APIs, not this Streamlit app

Therefore:

- if a task is about public API, auth, patients, API keys, audit, or multi-tenant behavior, work in `Cardiovascular_Be/`
- if a task is about frontend screens, FSD slices, integration with `/v1/*`, or mock-to-real API migration, work in `aorticprecision-next/`
- if a task is about model-serving or OCR service endpoints intended for Spring Boot to call, work in `Cardiovascular_FastAPI/`
- only work in `heart-disease-diagnosis/` when the task is specifically about the prototype, training assets, local demo, or ML experimentation

## Subproject Map

- `app/streamlit_app.py`: local demo interface
- `app/model_functions.py`: compatibility path for model unpickling
- `src/pipeline.py`: model loading and multi-model inference
- `src/model_functions.py`: feature engineering utilities and classes
- `src/utils/app_utils.py`: history, plotting, PDF/report helpers
- `scripts/train_models.py`: training and tuning workflow
- `scripts/experiment_manager.py`: experiment tracking helpers
- `models/saved_models/latest/`: current saved model artifacts used by this demo
- `data/raw/`, `data/processed/`: dataset inputs and derived forms

## Change Rules for This Subproject

### 1. Preserve pickle compatibility

- Be careful when renaming classes, functions, or module paths used by saved `.pkl` files.
- Changes to `BasicFE`, `EnhancedFE`, `PolyFE`, or related feature-engineering code can break model loading.

### 2. Keep prototype logic local

- UI-only behavior belongs in `app/streamlit_app.py`.
- reusable inference behavior belongs in `src/`
- experiment logic belongs in `scripts/` or `notebooks/`

Do not add platform-specific auth, tenant, API key, or Spring-style controller abstractions here unless the task explicitly asks for prototype simulation.

### 3. Do not confuse prototype inputs with platform contract

- The fields and prediction flow here may differ from the final `/v1/predict` contract in `../docs/API_CONTRACT.md`.
- If porting logic from this project into platform services, explicitly reconcile field names, enums, DTOs, and response schema.

### 4. Keep data handling conservative

- Do not overwrite raw datasets unless explicitly requested.
- Prefer writing derived outputs to experiment or result directories.
- Keep educational/decision-support disclaimers intact.

## Task Routing

Use `heart-disease-diagnosis/` for tasks like:

- fix Streamlit demo behavior
- repair local model loading
- adjust feature engineering or experiment scripts
- inspect trained model metrics or artifacts
- improve local visualization or report generation

Do not use `heart-disease-diagnosis/` as the default target for tasks like:

- build `/v1/predict`, `/v1/auth/*`, `/v1/patients`, `/v1/api-keys`
- implement JWT, RBAC, audit log, or organization scoping
- wire Next.js screens to real backend APIs
- expose OCR or model endpoints for platform integration

Those belong to the other monorepo projects described above.

## Validation

After edits in this subproject, validate only what is relevant:

- Streamlit/demo change: `streamlit run app/streamlit_app.py`
- pipeline/model-loading change: verify `pipeline.initialize()` and prediction flow still work with `models/saved_models/latest/`
- training-script change: run the narrowest relevant script path
- docs-only change: verify paths and repository references are correct

If validation was not run, say so explicitly.
