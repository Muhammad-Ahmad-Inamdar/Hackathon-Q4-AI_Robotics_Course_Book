# Backend–Frontend Integration (FastAPI + React/Docusaurus)

## Skill Name
Backend–Frontend Integration (FastAPI + React/Docusaurus)

## Problem it Solves
Establishes stable integration patterns between FastAPI backend and React/Docusaurus frontend applications. Addresses CORS issues, request/response contract stability, port configuration conflicts, and proper error handling across the backend-frontend boundary.

## Common Mistakes (based on past failures)
- Improper CORS configuration leading to cross-origin request failures
- Inconsistent request/response data formats between backend and frontend
- Port conflicts and binding issues during development and deployment
- Not implementing proper error propagation from backend to frontend
- Missing proper authentication and authorization flows
- Improper handling of file uploads/downloads between systems
- Not accounting for network latency and timeout scenarios

## Proven Working Pattern (Golden Path)
- Configure FastAPI with proper CORS middleware for frontend origins
- Establish consistent JSON request/response contract with proper error handling
- Use environment-specific port configurations with fallbacks
- Implement proper request validation and response serialization
- Apply consistent authentication patterns across API endpoints
- Use proper timeout and retry mechanisms for API calls
- Implement proper loading and error states in frontend

## Guardrails (what must NEVER be changed)
- CORS configuration must explicitly define allowed origins (never use wildcard in production)
- Request/response schemas must remain backward compatible
- Port configuration must use environment variables with defaults
- Authentication headers must be consistently applied across all protected endpoints
- Error response format must follow the same structure across all endpoints
- API endpoint paths must maintain consistent naming conventions

## Step-by-Step Execution
1. Configure FastAPI application with CORS middleware for frontend origin
2. Define consistent request/response models using Pydantic
3. Set up environment-specific port configuration with fallback values
4. Implement proper request validation and error handling in endpoints
5. Create consistent authentication/authorization middleware
6. Set up proper API response serialization
7. Configure frontend to use consistent API client with proper error handling
8. Implement proper state management for loading and error states
9. Test cross-origin requests and response handling
10. Verify timeout and retry mechanisms work properly

## Verification Checklist
- [ ] CORS configuration allows frontend origin without wildcard in production
- [ ] Request/response models consistent and properly validated
- [ ] Port configuration uses environment variables with proper defaults
- [ ] Error responses follow consistent format across all endpoints
- [ ] Authentication properly implemented and tested
- [ ] Frontend API calls handle loading and error states properly
- [ ] Timeout and retry mechanisms function correctly
- [ ] Cross-origin requests work without errors

## Reusability Notes
This integration pattern can be adapted to other frontend frameworks (Vue, Angular) by changing the client-side implementation while keeping the backend patterns consistent. The CORS configuration and request/response contract principles remain the same across different frontend technologies. The authentication and error handling patterns are universally applicable to backend-frontend integrations.