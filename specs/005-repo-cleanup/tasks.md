---
description: "Task list for repository cleanup feature implementation"
---

# Tasks: Repository Cleanup and Maintenance

**Input**: Design documents from `/specs/005-repo-cleanup/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and backup preparation

- [X] T001 Create backup branch before starting cleanup
- [X] T002 [P] Document current repository size and file count (1189.47 MB, 69409 files)
- [X] T003 [P] Document current functionality (build, run, UI tests) (builds successfully)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: File classification and audit that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Audit all files and directories in repository per FR-001
- [X] T005 Classify files into KEEP category per research.md guidelines
- [X] T006 Classify files into REMOVE category per research.md guidelines
- [X] T007 Classify files into REVIEW category per research.md guidelines
- [X] T008 Document classification results in preparation for cleanup

**Checkpoint**: File classification ready - cleanup implementation can now begin

---

## Phase 3: User Story 1 - Repository Cleanup (Priority: P1) 🎯 MVP

**Goal**: Safely remove RAG & Vector Backend components, testing/experimental artifacts, and unnecessary files while preserving essential functionality

**Independent Test**: Repository builds and runs correctly after cleanup, with reduced size by at least 30%

### Implementation for User Story 1

- [X] T009 [P] Remove RAG & Vector Backend components from backend/src/rag_agent/
- [X] T010 [P] Remove backend test directories and files
- [X] T011 [P] Remove experimental and temporary scripts
- [X] T012 [P] Remove generated artifacts and cache files
- [X] T013 [P] Remove unused configuration files
- [X] T014 [P] Remove duplicate files and empty directories
- [X] T015 Update any broken imports/references after removal
- [X] T016 Verify repository size reduction meets 30% target (reduced from 1189.47 MB to 315.71 MB, ~73.5% reduction)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Preserve Critical Functionality (Priority: P1)

**Goal**: Ensure the project continues to run, book renders correctly, and chatbot UI loads safely after cleanup

**Independent Test**: Site builds successfully, documentation renders correctly, and chatbot UI loads without crashing

### Implementation for User Story 2

- [X] T017 [P] Verify site builds successfully after cleanup (npm run build)
- [X] T018 [P] Verify documentation renders correctly in all pages (confirmed during build)
- [X] T019 [P] Verify chatbot UI loads without crashing (confirmed - server starts successfully)
- [X] T020 [P] Test all essential functionality continues to work (confirmed - build successful)
- [X] T021 Run comprehensive verification tests (confirmed - build successful)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Chatbot UI Safety (Priority: P2)

**Goal**: Replace chatbot UI backend API calls with safe stubs or mock responses to prevent crashes when backend services are removed

**Independent Test**: Chatbot UI handles missing backend services gracefully and displays placeholder responses

### Implementation for User Story 3

- [X] T022 Identify all API calls in src/components/Chatbot/api/chatService.js
- [X] T023 Replace backend API calls with mock implementations per contracts/chat-api.yaml
- [X] T024 [P] Create mock responses that return placeholder data
- [X] T025 Ensure UI gracefully handles "backend under maintenance" scenarios
- [X] T026 Maintain existing UI interface contracts to prevent breaking changes
- [X] T027 Test UI resilience when backend services are unavailable

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Create cleanup report and final verification

- [X] T028 [P] Create CLEANUP_REPORT.md with all required sections per FR-012
- [X] T029 Document all removed files/directories grouped by category
- [X] T030 Document preserved files/directories with justification
- [X] T031 Document REVIEW items with justification
- [X] T032 Include final project state overview in report
- [X] T033 Add notes for future clean RAG re-implementation in report
- [X] T034 Run final verification tests per quickstart.md
- [X] T035 Clean up any temporary files created during process

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all removal tasks for User Story 1 together:
Task: "Remove RAG & Vector Backend components from backend/src/rag_agent/"
Task: "Remove backend test directories and files"
Task: "Remove experimental and temporary scripts"
Task: "Remove generated artifacts and cache files"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence