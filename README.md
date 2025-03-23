# Chatbox-project-fastapi
A new project detached from <https://github.com/duy-jack-1995/chatbox-project-fastapi>


![image](/document_files/images/services_link_graph.png)

# Introduction
This project helps employees enhance their proactiveness in searching for work-related documents and guidelines on various types of company requests and tickets...

# Pre-requirements
- Docker, Nodejs, Python
- Coding IDE
- API key and exposed APIs from LLMs model

# Setup and Running
1 - Clone codes to your PC
- `git clone https://github.com/duy-jack-1995/chatbox-project-fastapi.git`

2 - Run commands - locally:
- Copy file `.env.template` -> `.env` and replace chatGPT API Key into it
- create `.npmrc` file inside chatbox-ui folder and add the following line:
- `>>> cd /chatbox-project-fastapi`
- `>>> docker-compose up --build`
- Backend API: `http://localhost:8000`
- Frontend UI: `http://localhost:3000`
- Postgres Database: `http://localhost:5432`
- PgAdmin: `http://localhost:5050`
- Login to PgAdmin with:
    - Email: `admin@example.com`
    - password: `admin`
    - To create a new server in PgAdmin:
          - Name: `app_db`
          - Host name/address: `db`
          - Port: `5432`
          - Username: `app_user`
          - Password: `app_password`
    - To crete a new table in PgAdmin:
          - Name: `User`
          - Schema: `public`
          - Tablespace: `pg_default`
          - Columns: `id`, `username`, `password`, `role`, `created_at`, `updated_at`
- Create a new user in PgAdmin:
  - User: `chatbox`
  - Password: `chatbox`

3 - Components:
- chatbox-ui: is a React app component to interacts with users.
  - src: is a source code of React app.
  - components: is defined components of React app.
- chatbox-api: is a FastAPI component to handle requests from chatbox-ui and interact with GPT-4o-mini model via Azure Open AI.
   - data_warehouse: is a FastAPI component to handle PDF files stored.
   - routers: is a FastAPI component to handle requests from chatbox-ui and interact with GPT-4o-mini model and Postgres Database.
   - db: is a Postgres Database component to store data.
   - models: is defined Question and Users models.
   - routers: is defined routers to handle requests from chatbox-ui.
## API lists

| URL               | METHOD | DECRIPTION                             |
|-------------------| ------ |----------------------------------------|
| /api/auth/        | POST   | authentication endpoint                |
| /api/ask          | POST   | Post a question and retrive the anwser |
| /api/upload       | POST   | upload a document                      |

# ChatBox API Logic flow
- Lucide chart: `https://lucid.app/lucidchart/78f63b85-8671-46bc-ae83-4d29685a3b20/edit?viewport_loc=781%2C-36%2C2219%2C1087%2Cm-5o7ONTd-nK&invitationId=inv_57f7a7a3-3f10-4bda-8852-224dfd4a37da`

![image](/document_files/images/process.png)

# AWS proposed architecture
- AI model secret key need to retrieve from Azure Open AI and store in AWS Secret Manager then use in FastAPI component as an environment variable.

![image](/document_files/images/aws_art.png)
