# To Run:
## To run locally:
1. Build the venv & install the python requirements:
> ./Scripts/RebuildVenvLx.sh
2. Run the app:
> uvicorn Embed-Host:app --host 0.0.0.0 --port 42443 --reload

## Project Structure (ToDo: Write proper documentation)

### File Responsibilities:

- **Embed-Host.py**: API & Frontend
  - FastAPI routes and middleware
  - Request/response handling
  - API endpoint definitions
  - No direct model handling

- **model_service.py**: Service Layer
  - Model initialization and management
  - Parameter validation and comparison
  - Service-level operations (status, metrics)
  - Delegates to appropriate model handlers

- **embeddings/_base_model_handler.py**: Base Handler
  - Generic model loading/initialization
  - Common embedding generation logic
  - Shared utilities and interfaces
  - No model-specific implementations

- **embeddings/model_srv_[model_name].py**: Specific Handlers
  - Model-specific implementations
  - Custom preprocessing and configurations
  - Override only what's needed from base handler
  - No generic handler code

### Adding New Models:
1. Create new handler in embeddings/model_srv_[model_name].py
2. Extend BaseModelHandler
3. Override only the methods needed for your specific model
4. Place model files in ./Models/[model_name]

ToDo:
- [ ] Add detailed setup instructions
- [ ] Document API endpoints
- [ ] Add configuration guide
- [ ] Explain model handler implementation
- [ ] Add troubleshooting section