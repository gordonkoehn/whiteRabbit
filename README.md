# Reinforcement Learning Project with Streamlit and Venice AI

This project is a Python-based reinforcement learning (RL) application featuring a FastAPI backend, a Streamlit frontend, and integration with a hypothetical Venice AI API. It uses a CartPole environment from Gymnasium, trained with Stable-Baselines3's PPO algorithm, to demonstrate RL predictions. The Streamlit frontend allows users to input states for RL predictions and query Venice AI for text responses. The project is containerized with Docker Compose and deployed to Render via a GitHub Actions CI/CD pipeline.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Development](#development)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
- [License](#license)

## Project Overview

The project consists of:
- **Backend**: A FastAPI server hosting:
  - An RL model (PPO trained on CartPole-v1) for predicting actions from states.
  - A Venice AI API client for querying text responses (hypothetical API).
- **Frontend**: A Streamlit app to:
  - Input CartPole states and view RL predictions.
  - Send prompts to Venice AI and display responses.
- **Infrastructure**:
  - Docker Compose for running backend and frontend containers.
  - A persistent Docker volume to store the trained RL model.
  - GitHub Actions for CI/CD, deploying to Render.
- **Tech Stack**: Python 3.10, FastAPI, Streamlit, Stable-Baselines3, Gymnasium, Docker, Render, GitHub Actions.

The RL model is trained on startup if no saved model exists, ensuring robustness. The project is designed for collaboration, with GitHub for version control and VS Code as the recommended IDE.

## Features

- Predict actions for CartPole states using a trained PPO model.
- Query a hypothetical Venice AI API for text responses.
- Interactive Streamlit dashboard for RL and AI interactions.
- Automated testing and deployment to Render.
- Persistent model storage via Docker volume.

## Prerequisites

Before setting up the project, ensure you have:
- **Python 3.10**: Installed locally (use `pyenv` to manage versions).
- **Docker Desktop**: Installed and running for containerization.
- **Git**: Installed for cloning the repository.
- **GitHub Account**: For collaboration and CI/CD.
- **Docker Hub Account**: For pushing Docker images.
- **Render Account**: For deployment.
- **VS Code**: Recommended IDE with extensions:
  - Python (Microsoft)
  - Docker (Microsoft)
  - GitLens (GitKraken)
  - Pylance (Microsoft)
  - YAML (Red Hat)
- **Venice AI API Key**: Hypothetical key for API access (set as an environment variable).

## Setup Instructions

Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/rl-project.git
   cd rl-project
   ```
   Replace `yourusername` with your GitHub username.

2. **Set Up Environment Variables**:
   Create a `.env` file in the project root:
   ```bash
   touch .env
   ```
   Add the Venice AI API key:
   ```env
   VENICE_API_KEY=your-venice-api-key
   ```
   Replace `your-venice-api-key` with the actual key (or a placeholder for testing).

3. **Build and Run with Docker Compose**:
   Ensure Docker Desktop is running.
   In the project root, build and start the containers:
   ```bash
   docker-compose up --build
   ```
   This creates:
   - backend container (FastAPI, port 8000).
   - frontend container (Streamlit, port 8501).
   - A model-data volume for the RL model.
   
   The backend trains the PPO model if `model/ppo_cartpole.zip` is missing (takes ~1-2 minutes on first run).

4. **Access the Application**:
   - Streamlit Frontend: Open http://localhost:8501 in a browser.
   - FastAPI Backend: Open http://localhost:8000/docs for the API documentation.
   
   To stop, press Ctrl+C and run:
   ```bash
   docker-compose down
   ```

5. **Optional: Local Development without Docker**:
   Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
   Run the backend:
   ```bash
   uvicorn app.main:app --reload
   ```
   Install frontend dependencies:
   ```bash
   cd ../frontend
   pip install -r requirements.txt
   ```
   Run the frontend:
   ```bash
   streamlit run app/main.py
   ```

## Usage

### Streamlit Frontend
1. **RL Prediction**:
   - Navigate to http://localhost:8501.
   - In the "Reinforcement Learning" section, enter a CartPole state as a list (e.g., [0.1, 0.2, 0.3, 0.4]).
   - Click "Predict Action" to see the PPO model's action (e.g., 0 or 1).

2. **Venice AI Query**:
   - In the "Venice AI Query" section, enter a prompt (e.g., "Explain RL in simple terms").
   - Click "Query Venice AI" to view the response (requires a valid VENICE_API_KEY).

### FastAPI Backend
Access the API at http://localhost:8000/docs.
Endpoints:
- GET `/predict/?state=[float,float,float,float]`: Returns an RL action (e.g., {"action": 0}).
  Example: http://localhost:8000/predict/?state=[0,0,0,0].
- POST `/venice/`: Sends a prompt to Venice AI and returns the response.
  Example request body: {"prompt": "What is RL?"}.

## Development

To contribute to the project, follow these steps in VS Code:

### Set Up VS Code:
- Open the project folder: File > Open Folder > rl-project.
- Select Python 3.10 interpreter: Ctrl+Shift+P, type "Python: Select Interpreter," choose Python 3.10.
- Install recommended extensions (listed in Prerequisites).

### Project Structure:
```
rl-project/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app
│   │   ├── rl_model.py      # RL model logic
│   │   ├── venice_ai.py     # Venice AI client
│   ├── tests/
│   │   ├── test_rl_model.py # Unit tests
│   ├── Dockerfile
│   ├── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── main.py          # Streamlit app
│   ├── Dockerfile
│   ├── requirements.txt
├── .github/
│   ├── workflows/
│   │   ├── ci-cd.yml        # GitHub Actions pipeline
├── docker-compose.yml
├── .env
├── .gitignore
├── README.md
```

### Create a Feature Branch:
- In VS Code's Source Control view (Ctrl+Shift+G), click the branch name (e.g., main).
- Create a new branch (e.g., feature/add-visualization).
- Make changes (e.g., edit frontend/app/main.py to add plots).
- Stage (+ icon), commit (Ctrl+Enter), and push (… > Push).

### Collaborate via GitHub:
- Push your branch:
  ```bash
  git push origin feature/add-visualization
  ```
- Open a pull request (PR) on GitHub:
  - Go to your repo > Pull requests > New pull request.
  - Select feature/add-visualization to main.
  - Assign your collaborator for review.
  - Merge after approval.

## Running Tests

The project includes unit tests for the backend RL model.

### Install Test Dependencies:
In the backend/ folder:
```bash
pip install -r requirements.txt
```
Ensures pytest is installed.

### Run Tests:
In the project root:
```bash
cd backend
pytest tests/
```
Tests in backend/tests/test_rl_model.py verify model predictions.

### Add New Tests:
Create new test files in backend/tests/ (e.g., test_venice_ai.py).
Example test:
```python
from app.rl_model import load_rl_model, predict

def test_predict_valid_action():
    model = load_rl_model()
    state = [0.0, 0.0, 0.0, 0.0]
    action = predict(model, state)
    assert action in [0, 1]
```
Run `pytest tests/` to verify.

### CI Testing:
- Tests run automatically on every push/PR via GitHub Actions (see .github/workflows/ci-cd.yml).
- Check results in GitHub > Actions tab.

## Deployment

The project is deployed to Render using a GitHub Actions CI/CD pipeline.

### Render Services:
- Backend: rl-project-backend (FastAPI, Docker, port 8000).
- Frontend: rl-project-frontend (Streamlit, Docker, port 8501).
- URLs: Provided in Render dashboard (e.g., https://rl-project-backend.onrender.com).

### CI/CD Pipeline:
- Defined in .github/workflows/ci-cd.yml.
- On push to main:
  - Runs tests (pytest backend/tests/).
  - Builds and pushes Docker images to Docker Hub.
  - Deploys to Render using API calls.
- Secrets required (set in GitHub > Settings > Secrets and variables > Actions):
  - DOCKER_USERNAME, DOCKER_PASSWORD
  - RENDER_API_KEY, RENDER_BACKEND_SERVICE_ID, RENDER_FRONTEND_SERVICE_ID

### Manual Deployment Check:
- After pushing to main, monitor GitHub Actions.
- Verify deployment in Render dashboard > Logs.
- Access deployed apps at Render URLs.

### Notes:
- Render's free tier has limited uptime; consider a paid tier for persistence.
- The model-data volume persists the RL model locally but not on Render's free tier.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

Happy coding, and enjoy building your RL project!

## Development Tips

### Ensuring Docker Compose Build Success

When running `docker-compose up --build`, look for:
- "Model not found, training new model…" (first run, ~1-2 minutes)
- PPO training output (e.g., mean_reward, ep_len)
- "Application started" from FastAPI

If you see a FileNotFoundError, stop (Ctrl+C), run docker-compose down, and rebuild.

### Check Containers:
- Open the Docker extension in VS Code (Ctrl+Shift+P, "Docker: Show Explorer")
- Confirm rl-project_backend_1 and rl-project_frontend_1 are running
- Inspect logs: Right-click rl-project_backend_1 > "View Logs"

### Test Access:
- Open http://localhost:8501 (Streamlit)
  - Try a state like [0, 0, 0, 0]; expect an action (0 or 1)
  - Venice AI prompt may fail without a real key, but the UI should load
- Open http://localhost:8000/docs (FastAPI)
  - Test /predict/ with state=[0,0,0,0]


### Clean Up:
Stop containers:
```bash
docker-compose down